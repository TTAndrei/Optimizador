"""Auditoria de Cd_p para NACA0012 con la proyeccion corregida.

El objetivo es separar exceso de drag de presion por fuente: normales IBM,
normales geometricas, de-bias de presion, extrapolacion a pared y trailing edge.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-cache"))

from Simulador2D import main as sim_main  # noqa: E402
from bl_correction import compute_corrected_forces, compute_geometric_pressure_forces  # noqa: E402
from sim_defaults import PROJECTION_DEFAULTS, finite_or_nan  # noqa: E402


OUT_BASE = ROOT_DIR / "results" / "diagnosticos" / "diagnostico_drag_presion_naca0012"
PROFILE = "profiles/NACA_0012"
DEBIAS_MODES = ("none", "mean_only", "affine_only", "affine+mean")
EXTRAP_LAYERS = (1, 2, 3, 5, 7, 9)
TE_CUTS = (None, 0.98, 0.95)


@dataclass(frozen=True)
class Case:
    case_id: str
    alpha: float
    reynolds: float


CASES = (
    Case("naca0012_re100k_a0", 0.0, 100_000.0),
    Case("naca0012_re100k_a4", 4.0, 100_000.0),
    Case("naca0012_re100k_a8", 8.0, 100_000.0),
    Case("naca0012_re100k_a10", 10.0, 100_000.0),
    Case("naca0012_re1m_a0", 0.0, 1_000_000.0),
    Case("naca0012_re1m_a4", 4.0, 1_000_000.0),
    Case("naca0012_re1m_a8", 8.0, 1_000_000.0),
    Case("naca0012_re1m_a10", 10.0, 1_000_000.0),
)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    extra = sorted({k for row in rows for k in row if k not in keys})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys + extra, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_cases(spec: str | None) -> list[Case]:
    if not spec:
        return list(CASES)
    wanted = {s.strip() for s in spec.split(",") if s.strip()}
    by_id = {case.case_id: case for case in CASES}
    missing = sorted(wanted - set(by_id))
    if missing:
        raise SystemExit(f"Casos no reconocidos: {', '.join(missing)}")
    return [by_id[k] for k in by_id if k in wanted]


def case_config(case: Case, args: argparse.Namespace) -> dict[str, Any]:
    u_inf = float(args.u_inf)
    return {
        "filepath": PROFILE,
        "chord": 1.0,
        "Lx": args.lx,
        "Ly": args.ly,
        "cx": args.cx,
        "cy": args.cy,
        "dx_min": args.dx_min,
        "factor_expansion": args.factor_expansion,
        "ancho_zona_fina_x": args.fine_width_x,
        "ancho_zona_fina_y": args.fine_width_y,
        "ratio_max_malla": args.ratio_max_malla,
        "v0x": u_inf,
        "v0y": 0.0,
        "rho": args.rho,
        "p0": 0.0,
        "nu": u_inf / case.reynolds,
        "CFL": args.cfl,
        "iteraciones": args.iteraciones,
        "guardado": args.guardado,
        "divergencia": args.divergencia,
        "alpha_deg": case.alpha,
        "usar_wale": not args.no_wale,
        "graficos": False,
        "save_frames": False,
        "live_view": False,
        "mostrar_malla": False,
        "stop_on_convergence": False,
        "corregir_deriva_vertical": False,
        "ibm_wall_mode": "ghost_noslip",
        "mg_modo_turbo_hd": args.mg_turbo_hd,
        "mg_modo_turbo_ultra": args.mg_turbo_ultra,
        **PROJECTION_DEFAULTS,
    }


def q_force(mesh: Any, rho: float, chord: float) -> float:
    u_ref = max(float(getattr(mesh, "_U_ref", getattr(mesh, "_vel_ref", 1.0))), 1e-30)
    return 0.5 * float(rho) * u_ref * u_ref * max(float(chord), 1e-30)


def summarize_force_rows(rows: list[dict[str, Any]], q: float, te_cut: float | None) -> dict[str, float]:
    active = []
    for row in rows:
        xoc = finite_or_nan(row.get("x_over_c"))
        if te_cut is not None and math.isfinite(xoc) and xoc > te_cut:
            continue
        active.append(row)
    drag = sum(finite_or_nan(r.get("dDrag_p")) for r in active)
    lift = sum(finite_or_nan(r.get("dLift_p")) for r in active)
    return {
        "n_faces_or_panels": len(active),
        "Drag_p": drag,
        "Lift_p": lift,
        "Cd_p": drag / q if q > 0.0 else float("nan"),
        "Cl_p": lift / q if q > 0.0 else float("nan"),
    }


def region_rows(case_id: str, variant: dict[str, Any], rows: list[dict[str, Any]], q: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for region_name, predicate in (
        ("LE", lambda x: x < 0.1),
        ("MID", lambda x: 0.1 <= x < 0.8),
        ("TE", lambda x: x >= 0.8),
    ):
        for side in ("upper", "lower", "all"):
            subset = []
            for row in rows:
                xoc = finite_or_nan(row.get("x_over_c"))
                if not math.isfinite(xoc) or not predicate(xoc):
                    continue
                if side != "all" and str(row.get("side")) != side:
                    continue
                subset.append(row)
            s = summarize_force_rows(subset, q, None)
            out.append({
                "case_id": case_id,
                **variant,
                "region": region_name,
                "side": side,
                **s,
            })
    return out


def audit_ibm(mesh: Any, cfg: dict[str, Any], layer: int, debias: str) -> dict[str, Any]:
    rho = float(cfg["rho"])
    chord = float(cfg["chord"])
    mu = rho * float(cfg["nu"])
    return mesh.extract_surface_force_audit(
        mu=mu,
        rho=rho,
        chord=chord,
        n_extrap_layers=layer,
        pressure_debias_mode=debias,
    )


def run_case(case: Case, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = case_config(case, args)
    t0 = time.time()
    mesh = sim_main(**cfg)
    elapsed = time.time() - t0
    q = q_force(mesh, float(cfg["rho"]), float(cfg["chord"]))
    summary_rows: list[dict[str, Any]] = []
    region_out: list[dict[str, Any]] = []
    panel_out: list[dict[str, Any]] = []

    try:
        bl = compute_corrected_forces(
            mesh,
            filepath=str(ROOT_DIR / PROFILE),
            rho=float(cfg["rho"]),
            chord=float(cfg["chord"]),
            v_inf=float(cfg["v0x"]),
            Re=case.reynolds,
            alpha_deg=case.alpha,
            Cd_p_source="both",
        )
    except Exception as exc:
        bl = {"bl_warn": str(exc)}

    for layer in EXTRAP_LAYERS:
        for debias in DEBIAS_MODES:
            for normal_mode in ("ibm", "geometric"):
                if normal_mode == "ibm":
                    audit = audit_ibm(mesh, cfg, layer, debias)
                else:
                    audit = compute_geometric_pressure_forces(
                        mesh,
                        filepath=str(ROOT_DIR / PROFILE),
                        rho=float(cfg["rho"]),
                        chord=float(cfg["chord"]),
                        n_extrap_layers=layer,
                        pressure_debias_mode=debias,
                    )
                rows = audit.get("rows", [])
                for te_cut in TE_CUTS:
                    s = summarize_force_rows(rows, q, te_cut)
                    variant = {
                        "surface_normal_mode": normal_mode,
                        "n_extrap_layers": layer,
                        "pressure_debias_mode": debias,
                        "te_exclude_after": finite_or_nan(te_cut),
                    }
                    summary_rows.append({
                        "case_id": case.case_id,
                        "alpha_deg": case.alpha,
                        "Re": case.reynolds,
                        "elapsed_s": elapsed,
                        "Cl_bl": finite_or_nan(bl.get("Cl")),
                        "Cd_bl": finite_or_nan(bl.get("Cd")),
                        "Cd_bl_geom": finite_or_nan(bl.get("Cd_bl_geom")),
                        "Cd_visc_bl": finite_or_nan(bl.get("Cd_visc")),
                        **variant,
                        **s,
                    })
                    if te_cut is None:
                        region_out.extend(region_rows(case.case_id, variant, rows, q))
                if layer == 5 and debias == "affine+mean":
                    for row in rows:
                        panel_out.append({
                            "case_id": case.case_id,
                            "surface_normal_mode": normal_mode,
                            "n_extrap_layers": layer,
                            "pressure_debias_mode": debias,
                            **row,
                        })

    del mesh
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return summary_rows, region_out, panel_out


def write_existing_summary(path: Path, out_dir: Path) -> None:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    out_rows = []
    for row in rows:
        out_rows.append({
            "case_id": row.get("case_id", row.get("alpha_deg", "")),
            "alpha_deg": finite_or_nan(row.get("alpha_deg")),
            "surface_normal_mode": "existing_summary",
            "n_extrap_layers": 5,
            "pressure_debias_mode": row.get("pressure_debias_model", ""),
            "te_exclude_after": float("nan"),
            "Cd_p": finite_or_nan(row.get("Cd_p", row.get("Cd_from_Cp_force_consistent"))),
            "Cl_p": finite_or_nan(row.get("Cl_p", row.get("Cl_from_Cp_force_consistent"))),
            "Cd_final": finite_or_nan(row.get("Cd_final")),
            "Cd_bl": finite_or_nan(row.get("Cd_bl")),
            "Cd_bl_geom": finite_or_nan(row.get("Cd_bl_geom")),
        })
    write_rows(out_dir / "surface_pressure_drag_audit.csv", out_rows)
    (out_dir / "drag_pressure_recommendation.md").write_text(
        "# Auditoria Cd_p\n\n"
        "Modo `--from-existing`: se copiaron metricas disponibles del summary existente. "
        "Para barrer normales, de-bias, capas y TE hace falta ejecutar con `--force`.\n",
        encoding="utf-8",
    )


def write_report(out_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    lines = ["# Auditoria Cd_p", ""]
    for case_id, rows in sorted(by_case.items()):
        full = [
            r for r in rows
            if str(r.get("pressure_debias_mode")) == "affine+mean"
            and int(r.get("n_extrap_layers", -1)) == 5
            and not math.isfinite(finite_or_nan(r.get("te_exclude_after")))
        ]
        if not full:
            continue
        best = min(full, key=lambda r: abs(finite_or_nan(r.get("Cd_p"))))
        ibm = next((r for r in full if r.get("surface_normal_mode") == "ibm"), None)
        geom = next((r for r in full if r.get("surface_normal_mode") == "geometric"), None)
        lines.append(f"## {case_id}")
        if ibm and geom:
            lines.append(
                f"- Cd_p IBM={finite_or_nan(ibm.get('Cd_p')):.6f}; "
                f"geometrico={finite_or_nan(geom.get('Cd_p')):.6f}; "
                f"diferencia={finite_or_nan(ibm.get('Cd_p')) - finite_or_nan(geom.get('Cd_p')):+.6f}."
            )
        lines.append(
            f"- Variante de menor |Cd_p| con capa=5/debias actual: "
            f"{best.get('surface_normal_mode')} Cd_p={finite_or_nan(best.get('Cd_p')):.6f}, "
            f"Cl_p={finite_or_nan(best.get('Cl_p')):.6f}."
        )
        lines.append("")
    lines.append("Criterio: si la ruta geometrica reduce Cd_p manteniendo Cl_p, priorizar normales/paneles reales para postproceso.")
    (out_dir / "drag_pressure_recommendation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--cases", default=None, help="Lista separada por comas de case_id")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--from-existing", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-fail-on-validation", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--iteraciones", type=int, default=900)
    parser.add_argument("--guardado", type=int, default=100)
    parser.add_argument("--dx-min", type=float, default=0.001)
    parser.add_argument("--u-inf", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--divergencia", type=float, default=0.05)
    parser.add_argument("--lx", type=float, default=7.0)
    parser.add_argument("--ly", type=float, default=6.0)
    parser.add_argument("--cx", type=float, default=2.0)
    parser.add_argument("--cy", type=float, default=3.0)
    parser.add_argument("--factor-expansion", type=float, default=1.03)
    parser.add_argument("--fine-width-x", type=float, default=1.5)
    parser.add_argument("--fine-width-y", type=float, default=1.0)
    parser.add_argument("--ratio-max-malla", type=float, default=35.0)
    parser.add_argument("--no-wale", action="store_true")
    parser.add_argument("--mg-turbo-hd", action="store_true", default=True)
    parser.add_argument("--mg-turbo-ultra", action="store_true")
    args = parser.parse_args()

    cases = parse_cases(args.cases)
    out_dir = OUT_BASE if not args.output_suffix else OUT_BASE.with_name(f"{OUT_BASE.name}_{args.output_suffix}")

    if args.list_cases:
        print(f"Salida: {out_dir.relative_to(ROOT_DIR)}")
        for case in cases:
            print(f"{case.case_id}: alpha={case.alpha:g} Re={case.reynolds:.0f}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_existing is not None and not args.force:
        write_existing_summary(args.from_existing, out_dir)
        print(f"Resumen: {out_dir / 'surface_pressure_drag_audit.csv'}")
        print(f"Reporte: {out_dir / 'drag_pressure_recommendation.md'}")
        return 0

    all_summary: list[dict[str, Any]] = []
    all_regions: list[dict[str, Any]] = []
    all_panels: list[dict[str, Any]] = []
    errors: list[str] = []
    for case in cases:
        print(f"\n[case] {case.case_id} alpha={case.alpha:g} Re={case.reynolds:.0f}")
        try:
            summary_rows, region_out, panel_out = run_case(case, args)
            all_summary.extend(summary_rows)
            all_regions.extend(region_out)
            all_panels.extend(panel_out)
        except Exception as exc:
            errors.append(f"{case.case_id}: {exc}")
            if not args.no_fail_on_validation:
                raise
            print(f"[WARN] {case.case_id}: {exc}")

    write_rows(out_dir / "surface_pressure_drag_audit.csv", all_summary)
    write_rows(out_dir / "surface_pressure_drag_regions.csv", all_regions)
    write_rows(out_dir / "surface_pressure_drag_panels.csv", all_panels)
    write_report(out_dir, all_summary)
    (out_dir / "run_meta.json").write_text(
        json.dumps({"errors": errors, "n_summary_rows": len(all_summary)}, indent=2),
        encoding="utf-8",
    )

    print(f"\nResumen: {out_dir / 'surface_pressure_drag_audit.csv'}")
    print(f"Regiones: {out_dir / 'surface_pressure_drag_regions.csv'}")
    print(f"Paneles: {out_dir / 'surface_pressure_drag_panels.csv'}")
    print(f"Reporte: {out_dir / 'drag_pressure_recommendation.md'}")
    if errors:
        print(f"Validacion: WARN ({len(errors)} errores)")
        return 0 if args.no_fail_on_validation else 1
    print("Validacion: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

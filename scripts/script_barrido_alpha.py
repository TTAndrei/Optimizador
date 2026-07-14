"""
Barrido de angulo de ataque con la proyeccion corregida.

Por defecto ejecuta NACA0012 con MG `outer_sum + masked`, guarda resultados en
`results/barridos/...` y reporta `Cl_from_Cp_force_consistent` como metrica de
presion principal.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-cache"))

import matplotlib.pyplot as plt

from Simulador2D import main as sim_main  # noqa: E402
from bl_correction import compute_corrected_forces  # noqa: E402
from sim_defaults import PROJECTION_DEFAULTS  # noqa: E402


DEFAULT_ALPHAS = "0:10:1"  # rango de 0 a 10 grados con paso de 1 grado


def finite_or_nan(value: Any) -> float:
    try:
        val = float(value)
    except Exception:
        return float("nan")
    return val if math.isfinite(val) else float("nan")


def parse_alpha_spec(spec: str) -> list[float]:
    spec = str(spec).strip()
    if ":" in spec:
        parts = [float(p) for p in spec.split(":")]
        if len(parts) not in {2, 3}:
            raise ValueError("--alphas con rango debe ser inicio:fin[:paso]")
        start, stop = parts[0], parts[1]
        step = parts[2] if len(parts) == 3 else 1.0
        if abs(step) < 1e-12:
            raise ValueError("paso de --alphas no puede ser cero")
        vals = []
        x = start
        cmp = (lambda a, b: a <= b + 1e-12) if step > 0 else (lambda a, b: a >= b - 1e-12)
        while cmp(x, stop):
            vals.append(float(round(x, 10)))
            x += step
        return vals
    return [float(p.strip()) for p in spec.split(",") if p.strip()]


def summarize_vector(vec_gpu: Any, discard_frac: float) -> tuple[float, float, float, int]:
    values = cp.asnumpy(vec_gpu).astype(float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    i0 = min(max(int(values.size * discard_frac), 0), values.size - 1)
    tail = values[i0:]
    return float(np.mean(tail)), float(values[-1]), float(np.std(tail)), int(tail.size)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    extra = sorted({k for r in rows for k in r if k not in keys})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys + extra, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def base_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = {
        "filepath": args.profile,
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
        "v0x": args.u_inf,
        "v0y": 0.0,
        "rho": args.rho,
        "p0": 0.0,
        "nu": args.nu,
        "CFL": args.cfl,
        "iteraciones": args.iteraciones,
        "guardado": args.guardado,
        "divergencia": args.divergencia,
        "turb_model": args.turb_model,
        "wall_treatment": "consistent",
        "advection_scheme": args.advection_scheme,
        "graficos": False,
        "save_frames": False,
        "live_view": False,
        "mostrar_malla": False,
        "stop_on_convergence": False,
        "stop_on_clcd_convergence": not args.no_clcd_stop,
        "corregir_deriva_vertical": False,
        **PROJECTION_DEFAULTS,
        "wake_refinement_mode": "long_fine_x",
        "min_te_height_factor": 1.0,
        "ibm_wall_mode": "ghost_noslip",
        "mg_modo_turbo": False,
        "mg_modo_turbo_hd": args.mg_turbo_hd,
        "mg_modo_turbo_ultra": args.mg_turbo_ultra,
    }
    if args.flow_inclined:
        cfg.update({
            "usar_flujo_inclinado": True,
            "flujo_inclinado_bc": "auto_farfield",
            "flujo_inclinado_signo": -1.0,
        })
    return cfg


def run_alpha(alpha: float, cfg_base: dict[str, Any], discard_frac: float) -> dict[str, Any]:
    cfg = dict(cfg_base)
    cfg["alpha_deg"] = float(alpha)
    if cfg.get("usar_flujo_inclinado"):
        cfg["flujo_inclinado_angulo_deg"] = -float(alpha)

    t0 = time.time()
    mesh = sim_main(**cfg)
    elapsed = time.time() - t0

    rho = float(cfg["rho"])
    nu = float(cfg["nu"])
    mu = rho * nu
    chord = float(cfg["chord"])
    u_ref = float(math.hypot(float(cfg["v0x"]), float(cfg.get("v0y", 0.0))))
    q_dyn = 0.5 * rho * u_ref * u_ref * chord

    cd_mean, cd_final, cd_std, n_cd = summarize_vector(mesh.cdvector, discard_frac)
    cl_mean, cl_final, cl_std, n_cl = summarize_vector(mesh.clvector, discard_frac)
    div_mean, div_final, _, _ = summarize_vector(mesh.divvector, discard_frac)
    div_flux_mean, div_flux_final, _, _ = summarize_vector(getattr(mesh, "divvector_flux", []), discard_frac)
    wall_mean, wall_final, _, _ = summarize_vector(getattr(mesh, "wall_leak_max_vector", []), discard_frac)

    forces = mesh.compute_drag_lift(mu, rho=rho, n_extrap_layers=5)
    audit_summary: dict[str, Any] = {}
    try:
        audit_summary = mesh.extract_surface_force_audit(mu=mu, rho=rho, chord=chord, n_extrap_layers=5)["summary"]
    except Exception as exc:
        audit_summary = {"surface_force_audit_warn": str(exc)}

    row = {
        "alpha_deg": float(alpha),
        "status": "ok",
        "nx": int(mesh.nx),
        "ny": int(mesh.ny),
        "n_cells": int(mesh.nx * mesh.ny),
        "dx_min": float(cfg["dx_min"]),
        "CFL": float(cfg["CFL"]),
        "ancho_zona_fina_x": float(cfg["ancho_zona_fina_x"]),
        "wake_refinement_mode": str(cfg["wake_refinement_mode"]),
        "iteraciones": int(cfg["iteraciones"]),
        "guardado": int(cfg["guardado"]),
        "projection_variant": str(cfg["projection_variant"]),
        "mg_pressure_accumulation": str(cfg["mg_pressure_accumulation"]),
        "wall_pressure_gradient_mode": str(cfg["wall_pressure_gradient_mode"]),
        "Cd_mean": cd_mean,
        "Cl_mean": cl_mean,
        "Ef_mean": cl_mean / cd_mean if abs(cd_mean) > 1e-12 else float("nan"),
        "Cd_final": cd_final,
        "Cl_final": cl_final,
        "Ef_final": cl_final / cd_final if abs(cd_final) > 1e-12 else float("nan"),
        "Cd_std": cd_std,
        "Cl_std": cl_std,
        "Cd_p": finite_or_nan(forces["Drag_p"] / q_dyn),
        "Cd_v": finite_or_nan(forces["Drag_v"] / q_dyn),
        "Cl_p": finite_or_nan(forces["Lift_p"] / q_dyn),
        "Cl_v": finite_or_nan(forces["Lift_v"] / q_dyn),
        "Cl_from_Cp_force_consistent": finite_or_nan(audit_summary.get("Cl_from_Cp_force_consistent")),
        "Cd_from_Cp_force_consistent": finite_or_nan(audit_summary.get("Cd_from_Cp_force_consistent")),
        "Cl_final_minus_Cp_force_consistent": finite_or_nan(
            cl_final - finite_or_nan(audit_summary.get("Cl_from_Cp_force_consistent"))
        ),
        "Cd_final_minus_Cp_force_consistent": finite_or_nan(
            cd_final - finite_or_nan(audit_summary.get("Cd_from_Cp_force_consistent"))
        ),
        "Cd_p_debiased": finite_or_nan(audit_summary.get("Cd_from_Cp_force_consistent")),
        "Cp_force_consistent_range": finite_or_nan(audit_summary.get("Cp_force_consistent_range")),
        "pressure_debias_model": audit_summary.get("pressure_debias_model", ""),
        "div_mean": div_mean,
        "div_final": div_final,
        "div_flux_mean": div_flux_mean,
        "div_flux_final": div_flux_final,
        "wall_leak_max_mean": wall_mean,
        "wall_leak_max_final": wall_final,
        "n_muestras_cd": n_cd,
        "n_muestras_cl": n_cl,
        "elapsed_s": float(elapsed),
        "converged_clcd": bool(getattr(mesh, "converged_clcd", False)),
        "t_conv_clcd": finite_or_nan(getattr(mesh, "t_conv_clcd", float("nan"))),
        "cl_cp_at_convergence": finite_or_nan(getattr(mesh, "cl_cp_at_convergence", float("nan"))),
        "cl_cp_discrepancy": finite_or_nan(getattr(mesh, "cl_cp_discrepancy", float("nan"))),
        "cl_cp_discrepancy_flag": bool(getattr(mesh, "cl_cp_discrepancy_flag", False)),
    }
    try:
        raw_summary = mesh.extract_surface_force_audit(
            mu=mu,
            rho=rho,
            chord=chord,
            n_extrap_layers=5,
            pressure_debias_mode="none",
        )["summary"]
        row["Cd_p_raw"] = finite_or_nan(raw_summary.get("Cd_from_Cp_force_consistent"))
    except Exception as exc:
        row["Cd_p_raw"] = float("nan")
        row["surface_force_raw_audit_warn"] = str(exc)

    try:
        bl = compute_corrected_forces(
            mesh,
            filepath=str(ROOT_DIR / str(cfg["filepath"])),
            alpha_deg=float(alpha),
            rho=rho,
            chord=chord,
            v_inf=u_ref,
            Re=u_ref * chord / max(nu, 1e-30),
            Cd_p_source="both",
        )
        row.update({
            "Cl_bl": finite_or_nan(bl.get("Cl")),
            "Cd_bl": finite_or_nan(bl.get("Cd")),
            "Cd_bl_geom": finite_or_nan(bl.get("Cd_bl_geom")),
            "Cd_p_bl": finite_or_nan(bl.get("Cd_p")),
            "Cd_p_ibm_bl": finite_or_nan(bl.get("Cd_p_ibm")),
            "Cd_p_geom": finite_or_nan(bl.get("Cd_p_geom")),
            "Cd_visc_bl": finite_or_nan(bl.get("Cd_visc")),
            "Ef_bl": finite_or_nan(bl.get("Ef")),
            "Cl_inviscid": finite_or_nan(bl.get("Cl_inviscid")),
            "trans_x_upper": finite_or_nan(bl.get("trans_x_upper")),
            "trans_x_lower": finite_or_nan(bl.get("trans_x_lower")),
            "x_sep_upper": finite_or_nan(bl.get("x_sep_upper")),
            "x_sep_lower": finite_or_nan(bl.get("x_sep_lower")),
        })
    except Exception as exc:
        row["bl_warn"] = str(exc)

    del mesh
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return row


def plot_curves(rows: list[dict[str, Any]], out_dir: Path) -> None:
    rows = sorted(rows, key=lambda r: float(r["alpha_deg"]))
    a = np.asarray([float(r["alpha_deg"]) for r in rows], dtype=float)

    def arr(key: str) -> np.ndarray:
        return np.asarray([finite_or_nan(r.get(key)) for r in rows], dtype=float)

    plots = [
        ("Cl", "Cl vs alpha", [
            ("Cl_final", "Cl simulacion"),
            ("Cl_from_Cp_force_consistent", "Cl Cp force"),
            ("cl_cp_at_convergence", "Cl_cp (∮ΔCp)"),
        ]),
        ("Cd", "Cd vs alpha", [
            ("Cd_final", "Cd final"),
            ("Cd_p", "Cd presion IBM"),
            ("Cd_p_geom", "Cd presion geom"),
            ("Cd_bl", "Cd BL"),
            ("Cd_bl_geom", "Cd BL geom"),
            ("Cd_v", "Cd viscoso CFD"),
        ]),
        ("Eficiencia", "Cl/Cd vs alpha", [("Ef_final", "Ef final")]),
        ("Divergencia", "div_flux vs alpha", [("div_flux_final", "div flux final"), ("wall_leak_max_final", "wall leak max")]),
    ]
    for suffix, title, series in plots:
        fig, ax = plt.subplots(figsize=(8, 5))
        for key, label in series:
            y = arr(key)
            if np.any(np.isfinite(y)):
                ax.plot(a, y, marker="o", linewidth=1.8, label=label)
        ax.axhline(0.0, color="k", linewidth=0.6, linestyle=":")
        ax.set_xlabel("alpha [deg]")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{suffix}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(arr("Cd_final"), arr("Cl_final"), marker="o", linewidth=1.8, label="final")
    for alpha, cd, cl in zip(a, arr("Cd_final"), arr("Cl_final")):
        if math.isfinite(cd) and math.isfinite(cl):
            ax.annotate(f"{alpha:.0f}", (cd, cl), textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.set_xlabel("Cd")
    ax.set_ylabel("Cl")
    ax.set_title("Polar Cl-Cd")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "Polar.png", dpi=180)
    plt.close(fig)


def print_table(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 112)
    print(f"{'alpha':>7} {'Cl':>10} {'Cl_CpF':>10} {'dCl':>10} {'Cd':>10} {'Ef':>10} {'divF':>10} {'wall':>10} {'t[s]':>8}")
    print("-" * 112)
    for r in sorted(rows, key=lambda x: float(x["alpha_deg"])):
        print(
            f"{float(r['alpha_deg']):>7.1f} "
            f"{finite_or_nan(r.get('Cl_final')):>10.5f} "
            f"{finite_or_nan(r.get('Cl_from_Cp_force_consistent')):>10.5f} "
            f"{finite_or_nan(r.get('Cl_final_minus_Cp_force_consistent')):>10.5f} "
            f"{finite_or_nan(r.get('Cd_final')):>10.5f} "
            f"{finite_or_nan(r.get('Ef_final')):>10.3f} "
            f"{finite_or_nan(r.get('div_flux_final')):>10.4f} "
            f"{finite_or_nan(r.get('wall_leak_max_final')):>10.4g} "
            f"{finite_or_nan(r.get('elapsed_s')):>8.1f}"
        )
    print("=" * 112)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Barrido de alpha con proyeccion MG corregida.")
    parser.add_argument("--alphas", default=DEFAULT_ALPHAS,
                        help="lista '0,2,4' o rango 'inicio:fin:paso' inclusive")
    parser.add_argument("--profile", default="profiles/NACA_0012_sharp")
    parser.add_argument("--output-suffix", default="naca0012_sa_maccormack_consistent",
                        help="nombre del subdirectorio en results/barridos")
    parser.add_argument("--force", action="store_true",
                        help="recalcula aunque exista summary.csv")
    parser.add_argument("--iteraciones", type=int, default=2000)
    parser.add_argument("--guardado", type=int, default=50)
    parser.add_argument("--discard-frac", type=float, default=0.30)
    parser.add_argument("--dx-min", type=float, default=0.002)
    parser.add_argument("--u-inf", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=1e-5)
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--divergencia", type=float, default=0.02)
    parser.add_argument("--lx", type=float, default=12.0)
    parser.add_argument("--ly", type=float, default=8.0)
    parser.add_argument("--cx", type=float, default=2.0)
    parser.add_argument("--cy", type=float, default=4.0)
    parser.add_argument("--factor-expansion", type=float, default=1.10)
    parser.add_argument("--fine-width-x", type=float, default=1.5)
    parser.add_argument("--fine-width-y", type=float, default=1.0)
    parser.add_argument("--ratio-max-malla", type=float, default=50.0)
    parser.add_argument("--turb-model", default="sa", choices=["none", "wale", "sa"])
    parser.add_argument("--advection-scheme", default="maccormack", choices=["sl", "maccormack"])
    parser.add_argument("--no-clcd-stop", action="store_true",
                        help="desactiva el early-stop por convergencia de Cl/Cd")
    parser.add_argument("--mg-turbo-hd", action=argparse.BooleanOptionalAction, default=False,
                         help="turbo satura el MG (Q residual +0.026 -> Cl sesgado), off por defecto")
    parser.add_argument("--mg-turbo-ultra", action="store_true")
    parser.add_argument("--flow-inclined", action="store_true",
                        help="mantiene geometria a 0 deg e inclina el flujo libre")
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    alphas = parse_alpha_spec(args.alphas)
    out_dir = ROOT_DIR / "results" / "barridos" / f"barrido_alpha_{args.output_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "summary.csv"
    out_json = out_dir / "summary.json"

    cfg = base_config(args)
    if args.list_cases:
        print(f"Salida: {out_dir.relative_to(ROOT_DIR)}")
        for alpha in alphas:
            print(f"alpha={alpha:+.2f} profile={args.profile} dx={args.dx_min:g} mgp=outer_sum grad=masked")
        return 0

    existing: list[dict[str, Any]] = []
    if out_csv.exists() and not args.force:
        with out_csv.open(encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    done = {float(r["alpha_deg"]): r for r in existing if r.get("status") == "ok"}

    rows: list[dict[str, Any]] = []
    for idx, alpha in enumerate(alphas, 1):
        if alpha in done:
            print(f"[{idx}/{len(alphas)}] alpha={alpha:+.1f}: ya existe, salto")
            rows.append(done[alpha])
            continue
        print(f"[{idx}/{len(alphas)}] alpha={alpha:+.1f}: simulando", flush=True)
        try:
            row = run_alpha(alpha, cfg, args.discard_frac)
        except Exception as exc:
            row = {"alpha_deg": alpha, "status": "error", "error": str(exc)}
            print(f"  ERROR: {exc}")
        rows.append(row)
        write_rows(out_csv, sorted(rows, key=lambda r: float(r["alpha_deg"])))
        with out_json.open("w", encoding="utf-8") as f:
            json.dump({"config": cfg, "alphas": alphas, "results": rows}, f, indent=2, ensure_ascii=True)
        if row.get("status") == "ok":
            print(
                f"  Cl={finite_or_nan(row.get('Cl_final')):+.5f} "
                f"Cl_CpF={finite_or_nan(row.get('Cl_from_Cp_force_consistent')):+.5f} "
                f"Cd={finite_or_nan(row.get('Cd_final')):.5f} "
                f"divF={finite_or_nan(row.get('div_flux_final')):.4f} "
                f"({finite_or_nan(row.get('elapsed_s')):.1f}s)"
            )

    rows = sorted(rows, key=lambda r: float(r["alpha_deg"]))
    write_rows(out_csv, rows)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"config": cfg, "alphas": alphas, "results": rows}, f, indent=2, ensure_ascii=True)
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if ok_rows:
        plot_curves(ok_rows, out_dir)
        print_table(ok_rows)

    print("\nArchivos generados:")
    print(f"  - {out_csv.relative_to(ROOT_DIR)}")
    print(f"  - {out_json.relative_to(ROOT_DIR)}")
    print(f"  - {out_dir.relative_to(ROOT_DIR)}/Cl.png")
    print(f"  - {out_dir.relative_to(ROOT_DIR)}/Cd.png")
    print(f"  - {out_dir.relative_to(ROOT_DIR)}/Polar.png")
    return 0 if all(r.get("status") == "ok" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""
Diagnostico de codigo para lift excesivo en NACA0012.

Este script no solo compara contra teoria: ejecuta casos controlados y decide
que subsistema del solver queda implicado:

- configuracion de la llamada principal
- aplicacion del angulo de ataque
- simetria +alpha/-alpha
- equivalencia geometria rotada vs flujo inclinado
- consistencia fuerzas integradas vs Cp
- exceso de lift por presion

Uso rapido:
    .venv/bin/python scripts/diagnostico_codigo_lift_naca0012.py --iteraciones 5000 --force

Uso comparable a tu corrida:
    .venv/bin/python scripts/diagnostico_codigo_lift_naca0012.py --iteraciones 50000 --force
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cupy as cp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-cache"))

from Simulador2D import main as sim_main  # noqa: E402
from sim_defaults import PROJECTION_DEFAULTS  # noqa: E402
from validar_lift_naca0012 import (  # noqa: E402
    equivalent_alpha_deg,
    fetch_xfoil_polar,
    interpolate_polar,
    thin_airfoil_cl,
)


@dataclass(frozen=True)
class Case:
    case_id: str
    alpha: float
    mode: str
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-suffix", default="diagnostico_codigo_lift_naca0012")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--reynolds", type=float, default=100_000.0)
    parser.add_argument("--iteraciones", type=int, default=5_000)
    parser.add_argument("--guardado", type=int, default=100)
    parser.add_argument("--discard-frac", type=float, default=0.30)
    parser.add_argument("--dx-min", type=float, default=0.001)
    parser.add_argument("--cfl", type=float, default=0.25)
    parser.add_argument("--fine-width-x", type=float, default=2.4)
    parser.add_argument("--fine-width-y", type=float, default=1.0)
    parser.add_argument("--u-inf", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--lx", type=float, default=12.0)
    parser.add_argument("--ly", type=float, default=8.0)
    parser.add_argument("--cx", type=float, default=2.0)
    parser.add_argument("--cy", type=float, default=4.0)
    parser.add_argument("--factor-expansion", type=float, default=1.10)
    parser.add_argument("--ratio-max-malla", type=float, default=100.0)
    parser.add_argument("--divergencia", type=float, default=0.10)
    parser.add_argument("--wale-cw", type=float, default=0.10)
    parser.add_argument("--cases", choices=("core", "full"), default="core")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())


def finite_or_nan(value: Any) -> float:
    try:
        val = float(value)
    except Exception:
        return float("nan")
    return val if math.isfinite(val) else float("nan")


def tail_stats(values: Any, discard_frac: float) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "final": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    i0 = min(max(int(arr.size * discard_frac), 0), arr.size - 1)
    tail = arr[i0:]
    return {
        "mean": float(np.mean(tail)),
        "final": float(arr[-1]),
        "std": float(np.std(tail)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(arr.size),
    }


def static_code_audit() -> dict[str, Any]:
    path = ROOT_DIR / "Simulador2D.py"
    text = path.read_text(encoding="utf-8", errors="replace")
    audit: dict[str, Any] = {
        "simulador_path": str(path.relative_to(ROOT_DIR)),
        "compute_drag_lift_uses_freestream_angle": "_get_freestream_angle_rad()" in text,
        "load_solids_rotates_geometry": "alpha = -np.deg2rad(alpha_deg)" in text,
        "flow_inclined_supported": "usar_flujo_inclinado" in text and "flujo_inclinado_angulo_deg" in text,
    }
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test = ast.unparse(node.test) if hasattr(ast, "unparse") else ""
                if "__name__" in test and "__main__" in test:
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.Call) and getattr(sub.func, "id", "") == "main":
                            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in sub.keywords if kw.arg}
                            audit["main_call_CFL"] = kwargs.get("CFL")
                            audit["main_call_alpha_deg"] = kwargs.get("alpha_deg")
                            audit["main_call_ancho_zona_fina_x"] = kwargs.get("ancho_zona_fina_x")
                            audit["main_call_wake_refinement_mode"] = kwargs.get("wake_refinement_mode")
                            audit["main_call_save_frames"] = kwargs.get("save_frames")
                            break
    except Exception as exc:
        audit["static_parse_warn"] = str(exc)

    audit["main_call_config_ok"] = (
        finite_or_nan(audit.get("main_call_CFL")) == 0.25
        and finite_or_nan(audit.get("main_call_ancho_zona_fina_x")) == 2.4
        and audit.get("main_call_wake_refinement_mode") == "long_fine_x"
    )
    return audit


def build_cases(alpha: float, mode: str) -> list[Case]:
    base = [
        Case("geom_a0", 0.0, "geometry", "alpha 0 con geometria rotada"),
        Case("geom_p", +float(alpha), "geometry", "+alpha con geometria rotada"),
        Case("geom_m", -float(alpha), "geometry", "-alpha con geometria rotada"),
        Case("flow_p", +float(alpha), "flow", "+alpha con geometria fija y flujo inclinado"),
    ]
    if mode == "full":
        base.append(Case("flow_m", -float(alpha), "flow", "-alpha con geometria fija y flujo inclinado"))
    return base


def build_config(args: argparse.Namespace, case: Case) -> dict[str, Any]:
    nu = float(args.u_inf) / max(float(args.reynolds), 1e-30)
    cfg: dict[str, Any] = {
        **PROJECTION_DEFAULTS,
        "filepath": "profiles/NACA_0012",
        "chord": 1.0,
        "alpha_deg": float(case.alpha),
        "Lx": float(args.lx),
        "Ly": float(args.ly),
        "cx": float(args.cx),
        "cy": float(args.cy),
        "dx_min": float(args.dx_min),
        "factor_expansion": float(args.factor_expansion),
        "ancho_zona_fina_x": float(args.fine_width_x),
        "ancho_zona_fina_y": float(args.fine_width_y),
        "ratio_max_malla": float(args.ratio_max_malla),
        "v0x": float(args.u_inf),
        "v0y": 0.0,
        "rho": float(args.rho),
        "p0": 0.0,
        "nu": nu,
        "CFL": float(args.cfl),
        "iteraciones": int(args.iteraciones),
        "guardado": int(args.guardado),
        "divergencia": float(args.divergencia),
        "usar_wale": True,
        "wale_Cw": float(args.wale_cw),
        "graficos": False,
        "save_frames": False,
        "live_view": False,
        "mostrar_malla": False,
        "stop_on_convergence": False,
        "corregir_deriva_vertical": False,
        "ibm_wall_mode": "ghost_noslip",
        "wake_refinement_mode": "long_fine_x",
        "min_te_height_factor": 2.0,
        "mg_modo_turbo": False,
        "mg_modo_turbo_hd": True,
        "mg_modo_turbo_ultra": False,
    }
    if case.mode == "flow":
        cfg.update({
            "usar_flujo_inclinado": True,
            "flujo_inclinado_bc": "auto_farfield",
            "flujo_inclinado_signo": -1.0,
            "flujo_inclinado_angulo_deg": -float(case.alpha),
        })
    return cfg


def history_rows(case_dir: Path, mesh: Any, guardado: int) -> None:
    cl = cp.asnumpy(mesh.clvector).astype(float).ravel()
    cd = cp.asnumpy(mesh.cdvector).astype(float).ravel()
    div_flux = cp.asnumpy(getattr(mesh, "divvector_flux", cp.asarray([]))).astype(float).ravel()
    wall = cp.asnumpy(getattr(mesh, "wall_leak_max_vector", cp.asarray([]))).astype(float).ravel()
    n = max(len(cl), len(cd), len(div_flux), len(wall))
    with (case_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteracion", "Cl", "Cd", "div_flux", "wall_leak_max"])
        for i in range(n):
            writer.writerow([
                i * int(guardado),
                cl[i] if i < len(cl) else "",
                cd[i] if i < len(cd) else "",
                div_flux[i] if i < len(div_flux) else "",
                wall[i] if i < len(wall) else "",
            ])


def run_case(args: argparse.Namespace, out_dir: Path, case: Case) -> dict[str, Any]:
    case_dir = out_dir / "cases" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_config(args, case)
    (case_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")

    print(f"[run] {case.case_id}: {case.description}", flush=True)
    t0 = time.time()
    mesh = sim_main(**cfg)
    elapsed = time.time() - t0

    history_rows(case_dir, mesh, int(args.guardado))
    cl = cp.asnumpy(mesh.clvector).astype(float).ravel()
    cd = cp.asnumpy(mesh.cdvector).astype(float).ravel()
    cl_stats = tail_stats(cl, float(args.discard_frac))
    cd_stats = tail_stats(cd, float(args.discard_frac))

    rho = float(cfg["rho"])
    chord = float(cfg["chord"])
    u_ref = float(math.hypot(float(cfg["v0x"]), float(cfg.get("v0y", 0.0))))
    # El flujo inclinado cambia cfg antes de construir la malla, pero U_ref
    # sigue siendo la magnitud de referencia.
    q_dyn = 0.5 * rho * float(args.u_inf) ** 2 * chord
    mu = rho * float(cfg["nu"])
    forces = mesh.compute_drag_lift(mu, rho=rho, n_extrap_layers=5)
    alpha_flow_rad = float(mesh._get_freestream_angle_rad())
    alpha_flow_deg = math.degrees(alpha_flow_rad)
    alpha_geom = finite_or_nan(getattr(mesh, "alpha_geometry", float("nan")))
    cx_coeff = finite_or_nan(forces.get("Fx")) / q_dyn
    cy_coeff = finite_or_nan(forces.get("Fy")) / q_dyn

    audit_summary: dict[str, Any] = {}
    try:
        audit_summary = mesh.extract_surface_force_audit(mu=mu, rho=rho, chord=chord, n_extrap_layers=5)["summary"]
    except Exception as exc:
        audit_summary = {"surface_force_audit_warn": str(exc)}

    cp_profile_cl = float("nan")
    try:
        x, cp_u, cp_l = mesh.get_cp_profile_mean()
        cp_profile_cl = float(np.trapezoid(np.asarray(cp_l) - np.asarray(cp_u), np.asarray(x)))
        with (case_dir / "cp_profile_mean.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["x_over_c", "Cp_upper_mean", "Cp_lower_mean", "delta_Cp_lower_minus_upper"])
            for row in zip(x, cp_u, cp_l):
                writer.writerow([row[0], row[1], row[2], row[2] - row[1]])
    except Exception as exc:
        audit_summary["cp_profile_warn"] = str(exc)

    row = {
        "case_id": case.case_id,
        "status": "ok",
        "description": case.description,
        "mode": case.mode,
        "alpha_requested": float(case.alpha),
        "alpha_geometry": alpha_geom,
        "alpha_flow_deg": alpha_flow_deg,
        "alpha_effective_geom_minus_flow": alpha_geom - alpha_flow_deg if math.isfinite(alpha_geom) else float("nan"),
        "CFL": float(cfg["CFL"]),
        "dx_min": float(cfg["dx_min"]),
        "wake_refinement_mode": str(cfg["wake_refinement_mode"]),
        "iteraciones": int(cfg["iteraciones"]),
        "Cl_mean_tail": cl_stats["mean"],
        "Cl_final": cl_stats["final"],
        "Cl_std_tail": cl_stats["std"],
        "Cl_min": cl_stats["min"],
        "Cl_max": cl_stats["max"],
        "Cd_mean_tail": cd_stats["mean"],
        "Cd_final": cd_stats["final"],
        "Cd_std_tail": cd_stats["std"],
        "Cx_global": cx_coeff,
        "Cy_global": cy_coeff,
        "Cl_pressure": finite_or_nan(forces.get("Lift_p")) / q_dyn,
        "Cl_viscous": finite_or_nan(forces.get("Lift_v")) / q_dyn,
        "Cd_pressure": finite_or_nan(forces.get("Drag_p")) / q_dyn,
        "Cd_viscous": finite_or_nan(forces.get("Drag_v")) / q_dyn,
        "Cl_from_Cp_force_consistent": finite_or_nan(audit_summary.get("Cl_from_Cp_force_consistent")),
        "Cd_from_Cp_force_consistent": finite_or_nan(audit_summary.get("Cd_from_Cp_force_consistent")),
        "Cl_from_Cp_mean_profile": cp_profile_cl,
        "div_flux_final": finite_or_nan(cp.asnumpy(getattr(mesh, "divvector_flux", cp.asarray([float("nan")]))).ravel()[-1]),
        "wall_leak_max_final": finite_or_nan(cp.asnumpy(getattr(mesh, "wall_leak_max_vector", cp.asarray([float("nan")]))).ravel()[-1]),
        "elapsed_s": elapsed,
        "case_dir": str(case_dir.relative_to(ROOT_DIR)),
    }
    (case_dir / "case_summary.json").write_text(json.dumps(row, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    del mesh
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row})
    preferred = [
        "case_id", "status", "mode", "alpha_requested", "alpha_geometry", "alpha_flow_deg",
        "alpha_effective_geom_minus_flow", "Cl_final", "Cl_mean_tail", "Cl_std_tail",
        "Cd_final", "Cl_pressure", "Cl_viscous", "Cl_from_Cp_force_consistent",
        "Cl_from_Cp_mean_profile", "div_flux_final", "wall_leak_max_final", "elapsed_s",
    ]
    fieldnames = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_row(rows: list[dict[str, Any]], case_id: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("case_id") == case_id:
            return row
    return None


def diagnose(rows: list[dict[str, Any]], static_audit: dict[str, Any], ref_cl: float, ref_cd: float, alpha: float) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    def add(severity: str, subsystem: str, evidence: str, recommendation: str) -> None:
        findings.append({
            "severity": severity,
            "subsystem": subsystem,
            "evidence": evidence,
            "recommendation": recommendation,
        })

    if not static_audit.get("main_call_config_ok"):
        add(
            "high",
            "configuracion",
            f"main_call_CFL={static_audit.get('main_call_CFL')}, ancho_zona_fina_x={static_audit.get('main_call_ancho_zona_fina_x')}, wake={static_audit.get('main_call_wake_refinement_mode')}",
            "Actualizar la llamada __main__ de Simulador2D.py a CFL=0.25, ancho_zona_fina_x=2.4, wake_refinement_mode='long_fine_x'.",
        )

    geom0 = find_row(rows, "geom_a0")
    gpos = find_row(rows, "geom_p")
    gneg = find_row(rows, "geom_m")
    fpos = find_row(rows, "flow_p")

    if geom0 and abs(finite_or_nan(geom0.get("Cl_final"))) > 0.05:
        add(
            "high",
            "simetria IBM/presion",
            f"alpha=0 da Cl_final={finite_or_nan(geom0.get('Cl_final')):.4g}",
            "Depurar rasterizacion IBM, normales, presion de pared y simetria de malla antes de estudiar alpha>0.",
        )

    if gpos:
        cl = finite_or_nan(gpos.get("Cl_final"))
        ratio = cl / ref_cl if abs(ref_cl) > 1e-12 else float("nan")
        if math.isfinite(ratio) and ratio > 1.25:
            add(
                "high",
                "lift excesivo",
                f"geom +{alpha:g}deg da Cl_final={cl:.4g}, referencia={ref_cl:.4g}, ratio={ratio:.2f}",
                "El fallo se reproduce en el solver; mirar primero presion/IBM/angulo efectivo, no WALE.",
            )
        if math.isfinite(cl):
            alpha_equiv = equivalent_alpha_deg(cl)
            if abs(alpha_equiv - alpha) > 2.0:
                add(
                    "high",
                    "angulo efectivo",
                    f"Cl_final equivale a alpha={alpha_equiv:.2f}deg para teoria fina, pero se pidio {alpha:g}deg",
                    "Comprobar signo/rotacion del perfil, flujo inclinado y proyeccion de fuerzas.",
                )
        clp = finite_or_nan(gpos.get("Cl_pressure"))
        clv = finite_or_nan(gpos.get("Cl_viscous"))
        if math.isfinite(clp) and math.isfinite(cl) and abs(cl) > 1e-12 and abs(clp / cl) > 0.90:
            add(
                "medium",
                "presion",
                f"Cl_pressure={clp:.4g}, Cl_viscous={clv:.4g}, casi todo el exceso viene de presion",
                "Auditar compute_surface_forces_definitive, extrapolacion de p_wall y normales IBM.",
            )
        cl_cp_force = finite_or_nan(gpos.get("Cl_from_Cp_force_consistent"))
        if math.isfinite(cl_cp_force) and math.isfinite(cl) and abs(cl - cl_cp_force) > 0.15:
            add(
                "medium",
                "integracion Cp/fuerzas",
                f"Cl_final={cl:.4g}, Cl_from_Cp_force_consistent={cl_cp_force:.4g}",
                "Hay inconsistencia entre fuerza integrada y Cp; revisar auditoria de superficie.",
            )

    if gpos and gneg:
        antisym = finite_or_nan(gpos.get("Cl_final")) + finite_or_nan(gneg.get("Cl_final"))
        if math.isfinite(antisym) and abs(antisym) > 0.10:
            add(
                "high",
                "simetria +alpha/-alpha",
                f"Cl(+alpha)+Cl(-alpha)={antisym:.4g}",
                "El solver no es antisimetrico; buscar sesgo de malla, signo de alpha o frontera vertical.",
            )

    if gpos and fpos:
        cg = finite_or_nan(gpos.get("Cl_final"))
        cf = finite_or_nan(fpos.get("Cl_final"))
        if math.isfinite(cg) and math.isfinite(cf) and abs(cg - cf) > max(0.15, 0.25 * abs(ref_cl)):
            add(
                "high",
                "aplicacion de alpha",
                f"geometria rotada Cl={cg:.4g}, flujo inclinado Cl={cf:.4g}",
                "La equivalencia geometria/flujo falla; revisar load_solids_from_file, _get_freestream_angle_rad y BC auto_farfield.",
            )

    if gpos:
        cd = finite_or_nan(gpos.get("Cd_final"))
        if math.isfinite(ref_cd) and abs(ref_cd) > 1e-12 and math.isfinite(cd) and cd / ref_cd > 3.0:
            add(
                "medium",
                "drag/presion",
                f"Cd_final={cd:.4g}, Cd_ref={ref_cd:.4g}, ratio={cd/ref_cd:.2f}",
                "Drag excesivo confirma campo de presion/estela no fisico.",
            )

    if not findings:
        add("info", "sin fallo claro", "las pruebas principales no superaron umbrales", "Aumentar iteraciones o activar --cases full.")
    return findings


def plot_summary(out_dir: Path, rows: list[dict[str, Any]], ref_cl: float) -> None:
    labels = [str(r["case_id"]) for r in rows]
    cl = np.asarray([finite_or_nan(r.get("Cl_final")) for r in rows], dtype=float)
    cd = np.asarray([finite_or_nan(r.get("Cd_final")) for r in rows], dtype=float)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].bar(x, cl)
    axes[0].axhline(ref_cl, color="k", linestyle="--", linewidth=1.0, label="Cl ref +alpha")
    axes[0].axhline(-ref_cl, color="k", linestyle=":", linewidth=1.0, label="Cl ref -alpha")
    axes[0].set_ylabel("Cl final")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].bar(x, cd)
    axes[1].set_ylabel("Cd final")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostic_bars.png", dpi=170)
    plt.close(fig)


def write_report(out_dir: Path, args: argparse.Namespace, static_audit: dict[str, Any], rows: list[dict[str, Any]], xfoil: dict[str, float], thin_cl: float, findings: list[dict[str, Any]]) -> None:
    lines = [
        "# Diagnostico codigo lift NACA0012",
        "",
        f"- alpha objetivo: {args.alpha:g} deg",
        f"- Reynolds: {args.reynolds:g}",
        f"- iteraciones por caso: {args.iteraciones}",
        f"- referencia teoria fina Cl: {thin_cl:.6g}",
    ]
    if xfoil:
        lines.append(f"- referencia XFOIL Cl/Cd: {xfoil.get('Cl', float('nan')):.6g} / {xfoil.get('Cd', float('nan')):.6g}")
    lines.extend([
        "",
        "## Auditoria Estatica",
        "",
        "| chequeo | valor |",
        "|---|---|",
    ])
    for key in sorted(static_audit):
        lines.append(f"| {key} | {static_audit[key]} |")

    lines.extend([
        "",
        "## Casos Ejecutados",
        "",
        "| case | mode | alpha_req | alpha_geom | alpha_flow | Cl_final | Cd_final | Cl_p | Cl_v | Cl_Cp_force | Cl_Cp_profile |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            "| {case} | {mode} | {areq:.3g} | {ageom:.3g} | {aflow:.3g} | {cl:.5g} | {cd:.5g} | {clp:.5g} | {clv:.5g} | {clcpf:.5g} | {clcpp:.5g} |".format(
                case=row.get("case_id", ""),
                mode=row.get("mode", ""),
                areq=finite_or_nan(row.get("alpha_requested")),
                ageom=finite_or_nan(row.get("alpha_geometry")),
                aflow=finite_or_nan(row.get("alpha_flow_deg")),
                cl=finite_or_nan(row.get("Cl_final")),
                cd=finite_or_nan(row.get("Cd_final")),
                clp=finite_or_nan(row.get("Cl_pressure")),
                clv=finite_or_nan(row.get("Cl_viscous")),
                clcpf=finite_or_nan(row.get("Cl_from_Cp_force_consistent")),
                clcpp=finite_or_nan(row.get("Cl_from_Cp_mean_profile")),
            )
        )

    lines.extend([
        "",
        "## Diagnostico",
        "",
        "| severidad | subsistema | evidencia | recomendacion |",
        "|---|---|---|---|",
    ])
    for item in findings:
        lines.append(f"| {item['severity']} | {item['subsystem']} | {item['evidence']} | {item['recommendation']} |")
    lines.append("")
    (out_dir / "diagnostico_codigo_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = ROOT_DIR / "results" / "diagnosticos" / safe_name(args.output_suffix)
    if args.force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    static_audit = static_code_audit()
    polar_rows, xfoil_url, xfoil_warn = fetch_xfoil_polar(args.reynolds, out_dir, args.offline)
    xfoil = interpolate_polar(polar_rows, args.alpha)
    thin_cl = thin_airfoil_cl(args.alpha)
    ref_cl = finite_or_nan(xfoil.get("Cl")) if xfoil else thin_cl
    ref_cd = finite_or_nan(xfoil.get("Cd")) if xfoil else float("nan")

    rows: list[dict[str, Any]] = []
    for case in build_cases(args.alpha, args.cases):
        try:
            rows.append(run_case(args, out_dir, case))
        except Exception as exc:
            rows.append({
                "case_id": case.case_id,
                "status": "failed",
                "mode": case.mode,
                "alpha_requested": case.alpha,
                "description": case.description,
                "error": str(exc),
            })
            print(f"[failed] {case.case_id}: {exc}", flush=True)
        write_csv(out_dir / "summary.csv", rows)

    findings = diagnose([r for r in rows if r.get("status") == "ok"], static_audit, ref_cl, ref_cd, args.alpha)
    payload = {
        "args": vars(args),
        "xfoil_url": xfoil_url,
        "xfoil_warning": xfoil_warn,
        "xfoil_reference": xfoil,
        "thin_airfoil_cl": thin_cl,
        "static_audit": static_audit,
        "rows": rows,
        "findings": findings,
    }
    (out_dir / "diagnostico_codigo_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    write_csv(out_dir / "summary.csv", rows)
    plot_summary(out_dir, [r for r in rows if r.get("status") == "ok"], ref_cl)
    write_report(out_dir, args, static_audit, [r for r in rows if r.get("status") == "ok"], xfoil, thin_cl, findings)

    print(f"Salida: {out_dir.relative_to(ROOT_DIR)}")
    print(f"Resumen: {(out_dir / 'summary.csv').relative_to(ROOT_DIR)}")
    print(f"Informe: {(out_dir / 'diagnostico_codigo_report.md').relative_to(ROOT_DIR)}")
    print("Findings:")
    for item in findings:
        print(f"  [{item['severity']}] {item['subsystem']}: {item['evidence']}")
    return 1 if any(f["severity"] == "high" for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())

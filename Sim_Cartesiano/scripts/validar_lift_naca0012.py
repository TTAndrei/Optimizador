"""
Valida si el lift de NACA0012 es fisicamente razonable.

Compara los resultados del solver contra:
- teoria de perfil delgado: Cl = 2*pi*alpha_rad
- polar XFOIL de AirfoilTools para NACA0012 al Reynolds indicado

Uso rapido con los numeros impresos por el solver:
    .venv/bin/python scripts/validar_lift_naca0012.py \
        --alpha 5 --reynolds 100000 \
        --cl-mean 1.434126 --cl-final 1.624471 \
        --cd-mean 0.139471 --cd-final 0.172972

Analizar un historial ya guardado:
    .venv/bin/python scripts/validar_lift_naca0012.py --history path/to/history.csv

Ejecutar una simulacion de validacion:
    .venv/bin/python scripts/validar_lift_naca0012.py --run-solver --iteraciones 50000
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import time
import urllib.request
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


AIRFOILTOOLS_POLARS = {
    50_000: "xf-n0012-il-50000",
    100_000: "xf-n0012-il-100000",
    200_000: "xf-n0012-il-200000",
    500_000: "xf-n0012-il-500000",
    1_000_000: "xf-n0012-il-1000000",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-suffix", default="validacion_lift_naca0012_alpha5")
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--reynolds", type=float, default=100_000.0)
    parser.add_argument("--history", type=Path, help="history.csv existente con columnas iteracion, Cl, Cd")
    parser.add_argument("--run-solver", action="store_true")
    parser.add_argument("--offline", action="store_true", help="no descarga polar XFOIL; usa solo teoria fina")
    parser.add_argument("--cl-mean", type=float)
    parser.add_argument("--cl-final", type=float)
    parser.add_argument("--cd-mean", type=float)
    parser.add_argument("--cd-final", type=float)
    parser.add_argument("--iteraciones", type=int, default=50_000)
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
    parser.add_argument("--flow-inclined", action="store_true", help="mantiene geometria a 0 deg e inclina el flujo")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())


def finite_or_nan(value: Any) -> float:
    try:
        val = float(value)
    except Exception:
        return float("nan")
    return val if math.isfinite(val) else float("nan")


def thin_airfoil_cl(alpha_deg: float) -> float:
    return 2.0 * math.pi * math.radians(float(alpha_deg))


def equivalent_alpha_deg(cl: float) -> float:
    return math.degrees(float(cl) / (2.0 * math.pi))


def nearest_supported_re(reynolds: float) -> int:
    return min(AIRFOILTOOLS_POLARS, key=lambda re: abs(float(reynolds) - float(re)))


def fetch_xfoil_polar(reynolds: float, out_dir: Path, offline: bool) -> tuple[list[dict[str, float]], str, str]:
    re_key = nearest_supported_re(reynolds)
    polar_key = AIRFOILTOOLS_POLARS[re_key]
    csv_path = out_dir / f"{polar_key}.csv"
    url = f"http://airfoiltools.com/polar/csv?polar={polar_key}"
    if not offline:
        try:
            text = urllib.request.urlopen(url, timeout=20).read().decode("utf-8", "replace")
            csv_path.write_text(text, encoding="utf-8")
        except Exception as exc:
            if not csv_path.exists():
                return [], url, f"no se pudo descargar AirfoilTools: {exc}"
    elif not csv_path.exists():
        return [], url, "modo offline sin cache local de polar"

    text = csv_path.read_text(encoding="utf-8", errors="replace")
    if "Alpha,Cl,Cd,Cdp,Cm,Top_Xtr,Bot_Xtr" not in text:
        return [], url, "polar descargada no tiene cabecera esperada"
    body = text.split("Alpha,Cl,Cd,Cdp,Cm,Top_Xtr,Bot_Xtr", 1)[1].strip()
    rows: list[dict[str, float]] = []
    reader = csv.DictReader(io.StringIO("Alpha,Cl,Cd,Cdp,Cm,Top_Xtr,Bot_Xtr\n" + body))
    for raw in reader:
        try:
            rows.append({k: float(v) for k, v in raw.items()})
        except Exception:
            continue
    return rows, url, ""


def interpolate_polar(rows: list[dict[str, float]], alpha_deg: float) -> dict[str, float]:
    if not rows:
        return {}
    rows = sorted(rows, key=lambda r: r["Alpha"])
    alpha = float(alpha_deg)
    if alpha <= rows[0]["Alpha"]:
        return dict(rows[0])
    if alpha >= rows[-1]["Alpha"]:
        return dict(rows[-1])
    for lo, hi in zip(rows[:-1], rows[1:]):
        if lo["Alpha"] <= alpha <= hi["Alpha"]:
            span = hi["Alpha"] - lo["Alpha"]
            t = 0.0 if abs(span) < 1e-12 else (alpha - lo["Alpha"]) / span
            return {k: lo[k] + t * (hi[k] - lo[k]) for k in lo}
    return {}


def read_history(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            item: dict[str, float] = {}
            for key, value in raw.items():
                item[key] = finite_or_nan(value)
            rows.append(item)
    return rows


def write_history(path: Path, mesh: Any, guardado: int) -> None:
    cl = cp.asnumpy(mesh.clvector).astype(float).ravel()
    cd = cp.asnumpy(mesh.cdvector).astype(float).ravel()
    div = cp.asnumpy(getattr(mesh, "divvector_flux", cp.asarray([]))).astype(float).ravel()
    wall = cp.asnumpy(getattr(mesh, "wall_leak_max_vector", cp.asarray([]))).astype(float).ravel()
    n = max(len(cl), len(cd), len(div), len(wall))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteracion", "Cl", "Cd", "div_flux", "wall_leak_max"])
        for i in range(n):
            writer.writerow([
                i * int(guardado),
                cl[i] if i < len(cl) else "",
                cd[i] if i < len(cd) else "",
                div[i] if i < len(div) else "",
                wall[i] if i < len(wall) else "",
            ])


def summarize(values: list[float], discard_frac: float) -> dict[str, float]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
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


def analyze_history(rows: list[dict[str, float]], reference_cl: float, discard_frac: float) -> dict[str, Any]:
    if not rows:
        return {}
    it = np.asarray([r.get("iteracion", i) for i, r in enumerate(rows)], dtype=float)
    cl = np.asarray([r.get("Cl", float("nan")) for r in rows], dtype=float)
    cd = np.asarray([r.get("Cd", float("nan")) for r in rows], dtype=float)
    valid = np.isfinite(it) & np.isfinite(cl)
    result: dict[str, Any] = {}
    if np.count_nonzero(valid) >= 3:
        itv = it[valid]
        clv = cl[valid]
        split = max(3, int(0.40 * clv.size))
        early_fit = np.polyfit(itv[:split], clv[:split], 1)
        tail = clv[max(0, int((1.0 - discard_frac) * clv.size)):]
        high = np.where(np.abs(clv) > max(1.25 * abs(reference_cl), abs(reference_cl) + 0.15))[0]
        result.update({
            "history_cl_slope_early_per_iter": float(early_fit[0]),
            "history_cl_slope_early_per_1000": float(1000.0 * early_fit[0]),
            "history_cl_tail_std": float(np.std(tail)) if tail.size else float("nan"),
            "history_cl_peak_to_peak": float(np.nanmax(clv) - np.nanmin(clv)),
            "history_first_iter_excessive_cl": int(itv[high[0]]) if high.size else None,
        })
    result.update({
        "history_cl": summarize([r.get("Cl", float("nan")) for r in rows], discard_frac),
        "history_cd": summarize([r.get("Cd", float("nan")) for r in rows], discard_frac),
    })
    if np.any(np.isfinite(cd)):
        result["history_cd_final"] = float(cd[np.isfinite(cd)][-1])
    return result


def build_solver_config(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    nu = float(args.u_inf) / max(float(args.reynolds), 1e-30)
    cfg: dict[str, Any] = {
        **PROJECTION_DEFAULTS,
        "filepath": "profiles/NACA_0012",
        "chord": 1.0,
        "alpha_deg": float(args.alpha),
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
    if args.flow_inclined:
        cfg.update({
            "usar_flujo_inclinado": True,
            "flujo_inclinado_bc": "auto_farfield",
            "flujo_inclinado_signo": -1.0,
            "flujo_inclinado_angulo_deg": -float(args.alpha),
        })
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    return cfg


def run_solver(args: argparse.Namespace, out_dir: Path) -> tuple[dict[str, Any], Path]:
    cfg = build_solver_config(args, out_dir)
    t0 = time.time()
    mesh = sim_main(**cfg)
    elapsed = time.time() - t0
    history_path = out_dir / "history.csv"
    write_history(history_path, mesh, int(args.guardado))

    rho = float(cfg["rho"])
    nu = float(cfg["nu"])
    mu = rho * nu
    chord = float(cfg["chord"])
    q_dyn = 0.5 * rho * float(args.u_inf) ** 2 * chord
    forces = mesh.compute_drag_lift(mu, rho=rho, n_extrap_layers=5)
    audit_summary: dict[str, Any] = {}
    try:
        audit_summary = mesh.extract_surface_force_audit(mu=mu, rho=rho, chord=chord, n_extrap_layers=5)["summary"]
    except Exception as exc:
        audit_summary = {"surface_force_audit_warn": str(exc)}

    cp_profile_cl = float("nan")
    try:
        x, cp_u, cp_l = mesh.get_cp_profile_mean()
        cp_profile_cl = float(np.trapezoid(np.asarray(cp_l) - np.asarray(cp_u), np.asarray(x)))
    except Exception:
        pass

    summary = {
        "elapsed_s": elapsed,
        "nx": int(mesh.nx),
        "ny": int(mesh.ny),
        "n_cells": int(mesh.nx * mesh.ny),
        "Cl_force_pressure": finite_or_nan(forces.get("Lift_p", float("nan")) / q_dyn),
        "Cl_force_viscous": finite_or_nan(forces.get("Lift_v", float("nan")) / q_dyn),
        "Cd_force_pressure": finite_or_nan(forces.get("Drag_p", float("nan")) / q_dyn),
        "Cd_force_viscous": finite_or_nan(forces.get("Drag_v", float("nan")) / q_dyn),
        "Cl_from_Cp_force_consistent": finite_or_nan(audit_summary.get("Cl_from_Cp_force_consistent")),
        "Cd_from_Cp_force_consistent": finite_or_nan(audit_summary.get("Cd_from_Cp_force_consistent")),
        "Cl_from_Cp_mean_profile": cp_profile_cl,
    }
    del mesh
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return summary, history_path


def compare_results(observed: dict[str, float], xfoil: dict[str, float], thin_cl: float, alpha_deg: float) -> dict[str, Any]:
    ref_cl = finite_or_nan(xfoil.get("Cl")) if xfoil else thin_cl
    ref_cd = finite_or_nan(xfoil.get("Cd")) if xfoil else float("nan")
    cl_final = finite_or_nan(observed.get("Cl_final"))
    cl_mean = finite_or_nan(observed.get("Cl_mean"))
    cd_final = finite_or_nan(observed.get("Cd_final"))
    cd_mean = finite_or_nan(observed.get("Cd_mean"))
    return {
        "reference_cl": ref_cl,
        "reference_cd": ref_cd,
        "cl_final_ratio_vs_reference": cl_final / ref_cl if math.isfinite(cl_final) and abs(ref_cl) > 1e-12 else float("nan"),
        "cl_mean_ratio_vs_reference": cl_mean / ref_cl if math.isfinite(cl_mean) and abs(ref_cl) > 1e-12 else float("nan"),
        "cd_final_ratio_vs_xfoil": cd_final / ref_cd if math.isfinite(cd_final) and math.isfinite(ref_cd) and abs(ref_cd) > 1e-12 else float("nan"),
        "cd_mean_ratio_vs_xfoil": cd_mean / ref_cd if math.isfinite(cd_mean) and math.isfinite(ref_cd) and abs(ref_cd) > 1e-12 else float("nan"),
        "alpha_equiv_cl_final_deg": equivalent_alpha_deg(cl_final) if math.isfinite(cl_final) else float("nan"),
        "alpha_equiv_cl_mean_deg": equivalent_alpha_deg(cl_mean) if math.isfinite(cl_mean) else float("nan"),
        "alpha_equiv_error_final_deg": equivalent_alpha_deg(cl_final) - float(alpha_deg) if math.isfinite(cl_final) else float("nan"),
        "alpha_equiv_error_mean_deg": equivalent_alpha_deg(cl_mean) - float(alpha_deg) if math.isfinite(cl_mean) else float("nan"),
    }


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    xfoil: dict[str, float],
    xfoil_url: str,
    xfoil_warn: str,
    thin_cl: float,
    observed: dict[str, float],
    comparison: dict[str, Any],
    history_analysis: dict[str, Any],
    solver_summary: dict[str, Any],
) -> None:
    cl_ratio = finite_or_nan(comparison.get("cl_final_ratio_vs_reference"))
    cd_ratio = finite_or_nan(comparison.get("cd_final_ratio_vs_xfoil"))
    alpha_err = finite_or_nan(comparison.get("alpha_equiv_error_final_deg"))
    verdict = "OK"
    if math.isfinite(cl_ratio) and cl_ratio > 1.25:
        verdict = "FALLO: Cl demasiado alto"
    if math.isfinite(alpha_err) and abs(alpha_err) > 2.0:
        verdict = "FALLO: angulo efectivo incompatible"

    lines = [
        "# Validacion lift NACA0012",
        "",
        f"- alpha solicitado: {args.alpha:g} deg",
        f"- Reynolds: {args.reynolds:g}",
        f"- veredicto: {verdict}",
        "",
        "## Referencias",
        "",
        f"- Teoria fina: `Cl = 2*pi*alpha_rad = {thin_cl:.6g}`",
        f"- AirfoilTools/XFOIL: {xfoil_url}",
    ]
    if xfoil_warn:
        lines.append(f"- aviso XFOIL: {xfoil_warn}")
    if xfoil:
        lines.extend([
            f"- XFOIL interpolado: `Cl={xfoil.get('Cl', float('nan')):.6g}`, `Cd={xfoil.get('Cd', float('nan')):.6g}`",
            f"- XFOIL alpha usado/interpolado: `{xfoil.get('Alpha', float('nan')):.6g}`",
        ])

    lines.extend([
        "",
        "## Resultado Observado",
        "",
        "| metrica | valor |",
        "|---|---:|",
    ])
    for key in ("Cl_mean", "Cl_final", "Cd_mean", "Cd_final"):
        if key in observed and math.isfinite(finite_or_nan(observed.get(key))):
            lines.append(f"| {key} | {finite_or_nan(observed.get(key)):.6g} |")

    lines.extend([
        "",
        "## Comparacion",
        "",
        "| metrica | valor |",
        "|---|---:|",
    ])
    for key in (
        "cl_mean_ratio_vs_reference",
        "cl_final_ratio_vs_reference",
        "cd_mean_ratio_vs_xfoil",
        "cd_final_ratio_vs_xfoil",
        "alpha_equiv_cl_mean_deg",
        "alpha_equiv_cl_final_deg",
        "alpha_equiv_error_mean_deg",
        "alpha_equiv_error_final_deg",
    ):
        val = comparison.get(key)
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            lines.append(f"| {key} | {float(val):.6g} |")

    if history_analysis:
        lines.extend([
            "",
            "## Historial",
            "",
            "| metrica | valor |",
            "|---|---:|",
        ])
        for key, val in history_analysis.items():
            if isinstance(val, dict):
                for subkey, subval in val.items():
                    if isinstance(subval, (int, float)) and math.isfinite(float(subval)):
                        lines.append(f"| {key}.{subkey} | {float(subval):.6g} |")
            elif isinstance(val, (int, float)) and math.isfinite(float(val)):
                lines.append(f"| {key} | {float(val):.6g} |")
            elif val is not None:
                lines.append(f"| {key} | {val} |")

    if solver_summary:
        lines.extend([
            "",
            "## Auditoria Solver",
            "",
            "| metrica | valor |",
            "|---|---:|",
        ])
        for key, val in solver_summary.items():
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                lines.append(f"| {key} | {float(val):.6g} |")

    lines.extend([
        "",
        "## Lectura",
        "",
        "- Si `alpha_equiv_cl_final_deg` se acerca a 15 deg cuando se pidieron 5 deg, el fallo no es una diferencia aerodinamica normal.",
        "- Si `Cl_force_pressure` o `Cl_from_Cp_force_consistent` explican casi todo el exceso, revisar presion/IBM/proyeccion antes que friccion.",
        "- Si `history_cl_slope_early_per_1000` es positivo durante muchas iteraciones, el estado inicial esta cargando circulacion artificialmente o el transitorio aun no es estacionario.",
        "- Si el `Cl` oscila despues de estabilizarse, analizar solo la cola y guardar frames alrededor del inicio de la oscilacion.",
        "",
    ])
    (out_dir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_history(out_dir: Path, rows: list[dict[str, float]], reference_cl: float) -> None:
    if not rows:
        return
    it = np.asarray([r.get("iteracion", i) for i, r in enumerate(rows)], dtype=float)
    cl = np.asarray([r.get("Cl", float("nan")) for r in rows], dtype=float)
    cd = np.asarray([r.get("Cd", float("nan")) for r in rows], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(it, cl, label="Cl")
    axes[0].axhline(reference_cl, color="k", linestyle="--", linewidth=0.9, label="Cl ref")
    axes[0].axhline(-reference_cl, color="k", linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("Cl")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(it, cd, label="Cd")
    axes[1].set_xlabel("iteracion")
    axes[1].set_ylabel("Cd")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "history_validation.png", dpi=170)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    out_dir = ROOT_DIR / "results" / "diagnosticos" / safe_name(args.output_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)

    polar_rows, xfoil_url, xfoil_warn = fetch_xfoil_polar(args.reynolds, out_dir, args.offline)
    xfoil = interpolate_polar(polar_rows, args.alpha)
    thin_cl = thin_airfoil_cl(args.alpha)

    solver_summary: dict[str, Any] = {}
    history_path = args.history
    if args.run_solver:
        solver_summary, history_path = run_solver(args, out_dir)

    history_rows: list[dict[str, float]] = []
    if history_path:
        history_rows = read_history(history_path)

    history_analysis = analyze_history(
        history_rows,
        finite_or_nan(xfoil.get("Cl")) if xfoil else thin_cl,
        float(args.discard_frac),
    )

    observed = {
        "Cl_mean": finite_or_nan(args.cl_mean),
        "Cl_final": finite_or_nan(args.cl_final),
        "Cd_mean": finite_or_nan(args.cd_mean),
        "Cd_final": finite_or_nan(args.cd_final),
    }
    if history_analysis:
        observed.setdefault("Cl_mean", history_analysis.get("history_cl", {}).get("mean", float("nan")))
        observed.setdefault("Cl_final", history_analysis.get("history_cl", {}).get("final", float("nan")))
        observed.setdefault("Cd_mean", history_analysis.get("history_cd", {}).get("mean", float("nan")))
        observed.setdefault("Cd_final", history_analysis.get("history_cd", {}).get("final", float("nan")))
        for key, src in (
            ("Cl_mean", ("history_cl", "mean")),
            ("Cl_final", ("history_cl", "final")),
            ("Cd_mean", ("history_cd", "mean")),
            ("Cd_final", ("history_cd", "final")),
        ):
            if not math.isfinite(observed.get(key, float("nan"))):
                observed[key] = history_analysis.get(src[0], {}).get(src[1], float("nan"))

    comparison = compare_results(observed, xfoil, thin_cl, args.alpha)
    if history_rows:
        plot_history(out_dir, history_rows, finite_or_nan(comparison.get("reference_cl")))

    payload = {
        "args": vars(args),
        "xfoil_url": xfoil_url,
        "xfoil_warning": xfoil_warn,
        "xfoil_reference": xfoil,
        "thin_airfoil_cl": thin_cl,
        "observed": observed,
        "comparison": comparison,
        "history_analysis": history_analysis,
        "solver_summary": solver_summary,
    }
    (out_dir / "validation_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    write_report(out_dir, args, xfoil, xfoil_url, xfoil_warn, thin_cl, observed, comparison, history_analysis, solver_summary)

    print(f"Salida: {(out_dir).relative_to(ROOT_DIR)}")
    print(f"Informe: {(out_dir / 'validation_report.md').relative_to(ROOT_DIR)}")
    print(f"Resumen: {(out_dir / 'validation_summary.json').relative_to(ROOT_DIR)}")
    if history_rows:
        print(f"Grafico: {(out_dir / 'history_validation.png').relative_to(ROOT_DIR)}")
    ratio = finite_or_nan(comparison.get("cl_final_ratio_vs_reference"))
    if math.isfinite(ratio):
        print(f"Cl_final/ref = {ratio:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

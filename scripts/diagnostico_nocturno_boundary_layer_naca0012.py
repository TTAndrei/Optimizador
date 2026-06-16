"""
Suite nocturna para diagnosticar separacion espuria en NACA0012.

Ejecuta una matriz pequena de simulaciones comparables y guarda todo lo
necesario para decidir si el problema viene de resolucion de capa limite,
WALE, proyeccion, estela refinada o CFL.

Uso recomendado:
    .venv/bin/python scripts/diagnostico_nocturno_boundary_layer_naca0012.py --force

Reanudar sin repetir casos ya terminados:
    .venv/bin/python scripts/diagnostico_nocturno_boundary_layer_naca0012.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
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


@dataclass(frozen=True)
class Case:
    case_id: str
    description: str
    overrides: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-suffix", default="naca0012_bl_nocturno")
    parser.add_argument("--preset", choices=("smoke", "overnight", "full"), default="overnight")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--reynolds", type=float, default=100_000.0)
    parser.add_argument("--iteraciones", type=int, default=30_000)
    parser.add_argument("--guardado", type=int, default=100)
    parser.add_argument("--frame-every", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frame-dpi", type=int, default=160)
    parser.add_argument("--dx-min", type=float, default=0.001)
    parser.add_argument("--u-inf", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--cfl", type=float, default=0.50)
    parser.add_argument("--lx", type=float, default=12.0)
    parser.add_argument("--ly", type=float, default=8.0)
    parser.add_argument("--cx", type=float, default=2.0)
    parser.add_argument("--cy", type=float, default=4.0)
    parser.add_argument("--factor-expansion", type=float, default=1.10)
    parser.add_argument("--fine-width-x", type=float, default=1.2)
    parser.add_argument("--fine-width-y", type=float, default=1.0)
    parser.add_argument("--ratio-max-malla", type=float, default=100.0)
    parser.add_argument("--divergencia", type=float, default=0.10)
    parser.add_argument("--wale-cw", type=float, default=0.15)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())


def finite_or_nan(value: Any) -> float:
    try:
        val = float(value)
    except Exception:
        return float("nan")
    return val if math.isfinite(val) else float("nan")


def to_numpy(values: Any) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=np.float64)
    try:
        if hasattr(values, "get") or hasattr(values, "device"):
            return cp.asnumpy(values).astype(np.float64).ravel()
    except Exception:
        pass
    return np.asarray(values, dtype=np.float64).ravel()


def clean(values: Any) -> np.ndarray:
    arr = to_numpy(values)
    return arr[np.isfinite(arr)]


def stats(values: Any) -> dict[str, float]:
    arr = clean(values)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "p05": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "n": int(arr.size),
    }


def tail_stats(values: Any, guardado: int, late_start: float = 20_000.0) -> dict[str, float]:
    arr = clean(values)
    if arr.size == 0:
        return stats(arr)
    it = np.arange(arr.size, dtype=np.float64) * float(guardado)
    tail = arr[it >= float(late_start)]
    if tail.size == 0:
        tail = arr[max(0, int(0.70 * arr.size)):]
    return stats(tail)


def build_cases(args: argparse.Namespace) -> list[Case]:
    if args.preset == "smoke":
        return [
            Case("base", "Caso base corto para validar escritura de artefactos", {"iteraciones": 100, "save_frames_cada": 50}),
            Case("no_wale", "Mismo caso sin WALE", {"iteraciones": 100, "save_frames_cada": 50, "usar_wale": False}),
        ]

    cases = [
        Case("base", "Baseline actual: WALE, CFL nominal, proyeccion legacy", {}),
        Case("no_wale", "Sin WALE: aisla si el modelo submalla induce la separacion", {"usar_wale": False}),
        Case("wale_cw_005", "WALE mas debil: comprueba sensibilidad al coeficiente", {"usar_wale": True, "wale_Cw": 0.05}),
        Case("cfl_025", "CFL reducido: comprueba inestabilidad temporal", {"CFL": 0.25}),
        Case(
            "compatible_flux",
            "Proyeccion compatible-flux: comprueba presion/divergencia",
            {"projection_variant": "compatible_flux"},
        ),
        Case(
            "long_wake",
            "Zona fina mas larga aguas abajo: comprueba contaminacion de estela",
            {"wake_refinement_mode": "long_fine_x", "ancho_zona_fina_x": 2.4},
        ),
        Case(
            "fine_wall_y",
            "Mayor resolucion normal local: comprueba y+ y puntos en delta99",
            {"dx_min": max(float(args.dx_min) * 0.75, 1e-6), "ancho_zona_fina_y": max(float(args.fine_width_y), 1.2)},
        ),
    ]
    if args.preset == "full":
        cases.extend([
            Case(
                "fine_wall_y_050",
                "Resolucion muy fina de pared: caso caro para confirmar convergencia de BL",
                {"dx_min": max(float(args.dx_min) * 0.5, 1e-6), "ancho_zona_fina_y": max(float(args.fine_width_y), 1.4)},
            ),
            Case(
                "long_wake_cfl025",
                "Estela larga + CFL bajo: separa error temporal de error de malla",
                {"wake_refinement_mode": "long_fine_x", "ancho_zona_fina_x": 2.4, "CFL": 0.25},
            ),
        ])
    return cases


def build_base_config(args: argparse.Namespace, case_dir: Path) -> dict[str, Any]:
    nu = float(args.u_inf) / max(float(args.reynolds), 1e-30)
    chord = 1.0
    xlim = (max(0.0, float(args.cx) - 0.2 * chord), min(float(args.lx), float(args.cx) + 1.35 * chord))
    ylim = (max(0.0, float(args.cy) - 0.6 * chord), min(float(args.ly), float(args.cy) + 0.6 * chord))
    return {
        **PROJECTION_DEFAULTS,
        "filepath": "profiles/NACA_0012",
        "chord": chord,
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
        "save_frames": True,
        "frames_dir_grueso": str(case_dir / "frames_velocity_full"),
        "frames_dir_refinado": str(case_dir / "frames_velocity_zoom"),
        "frames_dir_presion": str(case_dir / "frames_pressure_full"),
        "save_frames_cada": int(args.frame_every),
        "save_frame_refined_xlim": xlim,
        "save_frame_refined_ylim": ylim,
        "save_frame_dpi": int(args.frame_dpi),
        "live_view": False,
        "mostrar_malla": False,
        "stop_on_convergence": False,
        "corregir_deriva_vertical": False,
        "ibm_wall_mode": "ghost_noslip",
        "mg_modo_turbo_hd": True,
        "mg_modo_turbo_ultra": False,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_history(case_dir: Path, mesh: Any, guardado: int) -> dict[str, Any]:
    vectors = {
        "Cl": clean(getattr(mesh, "clvector", None)),
        "Cd": clean(getattr(mesh, "cdvector", None)),
        "div_flux": clean(getattr(mesh, "divvector_flux", None)),
        "div_centered": clean(getattr(mesh, "divvector", None)),
        "wall_leak_mean": clean(getattr(mesh, "wall_leak_mean_vector", None)),
        "wall_leak_max": clean(getattr(mesh, "wall_leak_max_vector", None)),
    }
    n = max((len(v) for v in vectors.values()), default=0)
    path = case_dir / "history.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteracion", *vectors.keys()])
        for i in range(n):
            writer.writerow([
                i * int(guardado),
                *[finite_or_nan(v[i]) if i < len(v) else "" for v in vectors.values()],
            ])

    row: dict[str, Any] = {"history_csv": str(path.relative_to(ROOT_DIR))}
    for name, values in vectors.items():
        for key, value in stats(values).items():
            row[f"{name}_{key}"] = value
        for key, value in tail_stats(values, guardado).items():
            row[f"{name}_late_{key}"] = value
        if len(values):
            row[f"{name}_final"] = float(values[-1])
    return row


def plot_history(case_dir: Path) -> str:
    path = case_dir / "history.csv"
    if not path.exists() or path.stat().st_size == 0:
        return ""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if data.size == 0:
        return ""
    if data.ndim == 0:
        data = np.asarray([data])
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(data["iteracion"], data["Cl"], label="Cl")
    axes[0].plot(data["iteracion"], data["Cd"], label="Cd")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(data["iteracion"], data["div_flux"], label="div_flux")
    axes[1].plot(data["iteracion"], data["div_centered"], label="div_centered")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)
    axes[2].plot(data["iteracion"], data["wall_leak_max"], label="wall_leak_max")
    axes[2].set_xlabel("iteracion")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.25)
    fig.tight_layout()
    out = case_dir / "history.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out.relative_to(ROOT_DIR))


def local_spacing(values: np.ndarray, coords: np.ndarray) -> np.ndarray:
    diffs = np.diff(values)
    if diffs.size == 0:
        return np.full_like(coords, np.nan, dtype=np.float64)
    idx = np.searchsorted(values, coords, side="left")
    left = np.clip(idx - 1, 0, diffs.size - 1)
    right = np.clip(idx, 0, diffs.size - 1)
    return np.minimum(diffs[left], diffs[right])


def boundary_layer_audit(case_dir: Path, mesh: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    rho = float(cfg.get("rho", 1.0))
    nu = float(cfg.get("nu", 1e-5))
    chord = float(cfg.get("chord", 1.0))
    u_inf = max(math.hypot(float(cfg.get("v0x", 1.0)), float(cfg.get("v0y", 0.0))), 1e-30)
    mu = rho * nu
    try:
        audit = mesh.extract_surface_force_audit(
            mu=mu,
            rho=rho,
            chord=chord,
            n_extrap_layers=5,
            pressure_debias_mode="affine+mean",
            pressure_wall_reconstruction=str(cfg.get("pressure_wall_reconstruction", "linear_5")),
        )
    except Exception as exc:
        return {"surface_audit_warn": str(exc)}

    rows = list(audit.get("rows", []))
    summary = dict(audit.get("summary", {}))
    if not rows:
        return {"surface_audit_warn": "sin filas de superficie"}

    x_grid = cp.asnumpy(mesh.X_1d).astype(np.float64)
    y_grid = cp.asnumpy(mesh.Y_1d).astype(np.float64)
    x = np.asarray([finite_or_nan(r.get("x")) for r in rows], dtype=np.float64)
    y = np.asarray([finite_or_nan(r.get("y")) for r in rows], dtype=np.float64)
    nx = np.asarray([finite_or_nan(r.get("nx")) for r in rows], dtype=np.float64)
    ny = np.asarray([finite_or_nan(r.get("ny")) for r in rows], dtype=np.float64)
    ds = np.asarray([finite_or_nan(r.get("ds")) for r in rows], dtype=np.float64)
    dfx_v = np.asarray([finite_or_nan(r.get("dFx_v")) for r in rows], dtype=np.float64)
    dfy_v = np.asarray([finite_or_nan(r.get("dFy_v")) for r in rows], dtype=np.float64)
    tx_v = dfx_v / np.maximum(ds, 1e-30)
    ty_v = dfy_v / np.maximum(ds, 1e-30)
    tx = -ny
    ty = nx
    align = tx * float(cfg.get("v0x", 1.0)) + ty * float(cfg.get("v0y", 0.0))
    tx = np.where(align >= 0.0, tx, -tx)
    ty = np.where(align >= 0.0, ty, -ty)
    tau_signed = tx_v * tx + ty_v * ty
    tau_abs = np.abs(tau_signed)
    cf = tau_signed / (0.5 * rho * u_inf * u_inf)
    cf_abs = tau_abs / (0.5 * rho * u_inf * u_inf)
    u_tau = np.sqrt(np.maximum(tau_abs / max(rho, 1e-30), 0.0))

    dx_loc = local_spacing(x_grid, x)
    dy_loc = local_spacing(y_grid, y)
    y1 = 0.5 * np.minimum(dx_loc, dy_loc)
    y_plus = y1 * u_tau / max(nu, 1e-30)

    x_over_c = (x - float(cfg.get("cx", 2.0))) / max(chord, 1e-30)
    x_bl = np.clip(np.maximum(x_over_c, 1e-6), 1e-6, None) * chord
    rex = u_inf * x_bl / max(nu, 1e-30)
    delta99_lam = 5.0 * x_bl / np.sqrt(np.maximum(rex, 1e-30))
    delta_star_lam = 1.7208 * x_bl / np.sqrt(np.maximum(rex, 1e-30))
    theta_lam = 0.664 * x_bl / np.sqrt(np.maximum(rex, 1e-30))
    cells_delta99 = delta99_lam / np.maximum(np.minimum(dx_loc, dy_loc), 1e-30)
    useful_bl_mask = (x_over_c >= 0.05) & (x_over_c <= 0.98) & np.isfinite(cells_delta99)
    side = np.where(ny >= 0.0, "upper", "lower")
    reverse_proxy = tau_signed < 0.0

    out_rows: list[dict[str, Any]] = []
    for i, base in enumerate(rows):
        r = dict(base)
        r.update({
            "x_over_c": finite_or_nan(x_over_c[i]),
            "side": side[i],
            "dx_local": finite_or_nan(dx_loc[i]),
            "dy_local": finite_or_nan(dy_loc[i]),
            "y1_est": finite_or_nan(y1[i]),
            "tau_wall_signed_est": finite_or_nan(tau_signed[i]),
            "tau_wall_abs_est": finite_or_nan(tau_abs[i]),
            "Cf_signed_est": finite_or_nan(cf[i]),
            "Cf_abs_est": finite_or_nan(cf_abs[i]),
            "u_tau_est": finite_or_nan(u_tau[i]),
            "y_plus_est": finite_or_nan(y_plus[i]),
            "Re_x": finite_or_nan(rex[i]),
            "delta99_laminar_est": finite_or_nan(delta99_lam[i]),
            "delta_star_laminar_est": finite_or_nan(delta_star_lam[i]),
            "theta_laminar_est": finite_or_nan(theta_lam[i]),
            "cells_per_delta99_laminar_est": finite_or_nan(cells_delta99[i]),
            "reverse_tau_proxy": bool(reverse_proxy[i]),
        })
        out_rows.append(r)

    surface_csv = case_dir / "surface_boundary_layer.csv"
    write_rows(surface_csv, out_rows)
    surface_json = case_dir / "surface_force_summary.json"
    surface_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for side_name, marker in (("upper", "."), ("lower", ".")):
        mask = side == side_name
        axes[0].plot(x_over_c[mask], y_plus[mask], marker, label=side_name)
        axes[1].plot(x_over_c[mask], cells_delta99[mask], marker, label=side_name)
        axes[2].plot(x_over_c[mask], cf[mask], marker, label=side_name)
    axes[0].axhline(1.0, color="k", lw=0.8, ls="--")
    axes[0].axhline(5.0, color="k", lw=0.8, ls=":")
    axes[0].set_ylabel("y+ est.")
    axes[1].axhline(10.0, color="k", lw=0.8, ls="--")
    axes[1].set_ylabel("celdas/delta99 lam.")
    axes[2].axhline(0.0, color="k", lw=0.8)
    axes[2].set_ylabel("Cf est.")
    axes[2].set_xlabel("x/c")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    fig.tight_layout()
    plot_path = case_dir / "boundary_layer_audit.png"
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)

    row: dict[str, Any] = {
        "surface_boundary_layer_csv": str(surface_csv.relative_to(ROOT_DIR)),
        "surface_force_summary_json": str(surface_json.relative_to(ROOT_DIR)),
        "boundary_layer_plot": str(plot_path.relative_to(ROOT_DIR)),
    }
    for key, value in summary.items():
        if isinstance(value, (int, float, np.floating)):
            row[f"surface_{key}"] = finite_or_nan(value)
        elif key in {"pressure_debias_model", "pressure_wall_reconstruction"}:
            row[f"surface_{key}"] = str(value)
    for prefix, values in (
        ("y_plus", y_plus),
        ("cells_delta99", cells_delta99),
        ("y_plus_x005_098", y_plus[useful_bl_mask]),
        ("cells_delta99_x005_098", cells_delta99[useful_bl_mask]),
        ("Cf_abs", cf_abs),
        ("Cf_signed", cf),
        ("u_tau", u_tau),
        ("y1", y1),
    ):
        for key, value in stats(values).items():
            row[f"{prefix}_{key}"] = value
    upper = side == "upper"
    lower = side == "lower"
    row["reverse_tau_upper_frac"] = float(np.mean(reverse_proxy[upper])) if np.any(upper) else float("nan")
    row["reverse_tau_lower_frac"] = float(np.mean(reverse_proxy[lower])) if np.any(lower) else float("nan")
    row["cells_delta99_below_5_frac"] = float(np.mean(cells_delta99 < 5.0))
    row["cells_delta99_below_10_frac"] = float(np.mean(cells_delta99 < 10.0))
    row["cells_delta99_x005_098_below_5_frac"] = float(np.mean(cells_delta99[useful_bl_mask] < 5.0)) if np.any(useful_bl_mask) else float("nan")
    row["cells_delta99_x005_098_below_10_frac"] = float(np.mean(cells_delta99[useful_bl_mask] < 10.0)) if np.any(useful_bl_mask) else float("nan")
    row["y_plus_above_5_frac"] = float(np.mean(y_plus > 5.0))
    row["y_plus_above_30_frac"] = float(np.mean(y_plus > 30.0))
    return row


def cp_profile_audit(case_dir: Path, mesh: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        x, cp_u, cp_l = mesh.get_cp_profile_mean()
    except Exception as exc:
        return {"cp_profile_warn": str(exc)}
    x = np.asarray(x, dtype=np.float64)
    cp_u = np.asarray(cp_u, dtype=np.float64)
    cp_l = np.asarray(cp_l, dtype=np.float64)
    path = case_dir / "cp_profile_mean.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x_over_c", "Cp_upper_mean", "Cp_lower_mean", "delta_Cp_lower_minus_upper"])
        for row in zip(x, cp_u, cp_l):
            writer.writerow([row[0], row[1], row[2], row[2] - row[1]])
    cl_cp = float(np.trapezoid(cp_l - cp_u, x)) if len(x) > 1 else float("nan")
    return {
        "cp_profile_csv": str(path.relative_to(ROOT_DIR)),
        "Cl_from_Cp_mean_profile": cl_cp,
        "Cp_upper_min": float(np.nanmin(cp_u)) if cp_u.size else float("nan"),
        "Cp_lower_max": float(np.nanmax(cp_l)) if cp_l.size else float("nan"),
    }


def wake_metrics(mesh: Any, cfg: dict[str, Any]) -> dict[str, float]:
    dvdx = cp.zeros_like(mesh.v)
    dudy = cp.zeros_like(mesh.u)
    dvdx[:, 1:-1] = (
        mesh.d1x_W[1:-1] * mesh.v[:, :-2]
        + mesh.d1x_C[1:-1] * mesh.v[:, 1:-1]
        + mesh.d1x_E[1:-1] * mesh.v[:, 2:]
    )
    dudy[1:-1, :] = (
        mesh.d1y_S[1:-1, cp.newaxis] * mesh.u[:-2, :]
        + mesh.d1y_C[1:-1, cp.newaxis] * mesh.u[1:-1, :]
        + mesh.d1y_N[1:-1, cp.newaxis] * mesh.u[2:, :]
    )
    omega = dvdx - dudy
    chord = float(cfg.get("chord", 1.0))
    cx = float(cfg.get("cx", 2.0))
    cy = float(cfg.get("cy", 4.0))
    xoc = (mesh.XX - cp.float32(cx)) / cp.float32(max(chord, 1e-30))
    yoc = (mesh.YY - cp.float32(cy)) / cp.float32(max(chord, 1e-30))
    wake = (~mesh.solid) & (xoc >= 0.85) & (xoc <= 2.25) & (cp.abs(yoc) <= 0.75)
    if not bool(cp.any(wake)):
        return {"wake_omega_rms": float("nan"), "wake_omega_abs_max": float("nan"), "wake_speed_deficit_mean": float("nan")}
    vals = omega[wake]
    speed = cp.sqrt(mesh.u * mesh.u + mesh.v * mesh.v)
    u_inf = max(math.hypot(float(cfg.get("v0x", 1.0)), float(cfg.get("v0y", 0.0))), 1e-30)
    speed_w = speed[wake]
    return {
        "wake_omega_rms": float(cp.sqrt(cp.mean(vals * vals)).get() * chord / u_inf),
        "wake_omega_abs_max": float(cp.max(cp.abs(vals)).get() * chord / u_inf),
        "wake_speed_mean": float(cp.mean(speed_w).get() / u_inf),
        "wake_speed_deficit_mean": float(cp.mean(cp.maximum(cp.float32(u_inf) - speed_w, 0.0)).get() / u_inf),
    }


def wale_metrics(mesh: Any, cfg: dict[str, Any]) -> dict[str, float]:
    if not bool(cfg.get("usar_wale", False)):
        return {"nu_t_over_nu_mean": 0.0, "nu_t_over_nu_p95": 0.0, "nu_t_over_nu_max": 0.0}
    try:
        nu_t = mesh.compute_wale_viscosity()
        mask = ~mesh.solid
        ratio = cp.asnumpy((nu_t[mask] / cp.float32(max(float(cfg.get("nu", 1e-30)), 1e-30)))).astype(np.float64)
        s = stats(ratio)
        return {
            "nu_t_over_nu_mean": s["mean"],
            "nu_t_over_nu_p95": s["p95"],
            "nu_t_over_nu_max": s["max"],
        }
    except Exception as exc:
        return {"nu_t_warn": str(exc), "nu_t_over_nu_mean": float("nan"), "nu_t_over_nu_p95": float("nan"), "nu_t_over_nu_max": float("nan")}


def make_video(frames_dir: Path, video_path: Path, fps: int) -> dict[str, Any]:
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        return {"status": "skipped", "reason": "sin frames", "video": str(video_path.relative_to(ROOT_DIR))}
    if shutil.which("ffmpeg") is None:
        return {"status": "skipped", "reason": "ffmpeg no encontrado", "video": str(video_path.relative_to(ROOT_DIR))}
    list_path = frames_dir / "frames.ffconcat"
    duration = 1.0 / max(int(fps), 1)
    lines = ["ffconcat version 1.0"]
    for frame in frames:
        escaped = str(frame.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
        lines.append(f"duration {duration:.8f}")
    escaped = str(frames[-1].resolve()).replace("'", "'\\''")
    lines.append(f"file '{escaped}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-vf",
        f"fps={int(fps)},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        str(video_path),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT_DIR), text=True, capture_output=True)
    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "video": str(video_path.relative_to(ROOT_DIR)),
        "n_frames": len(frames),
        "elapsed_s": time.time() - t0,
        "stderr": proc.stderr.strip(),
    }


def create_videos(case_dir: Path, fps: int) -> dict[str, Any]:
    videos_dir = case_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    return {
        "velocity_full": make_video(case_dir / "frames_velocity_full", videos_dir / "velocity_full.mp4", fps),
        "velocity_zoom": make_video(case_dir / "frames_velocity_zoom", videos_dir / "velocity_zoom.mp4", fps),
        "pressure_full": make_video(case_dir / "frames_pressure_full", videos_dir / "pressure_full.mp4", fps),
    }


def save_final_fields(case_dir: Path, mesh: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    iter_done = int(getattr(mesh, "_iteraciones_realizadas", cfg.get("iteraciones", 0)))
    state_path = case_dir / "state_final.npz"
    np.savez_compressed(
        state_path,
        u=cp.asnumpy(mesh.u).astype(np.float32),
        v=cp.asnumpy(mesh.v).astype(np.float32),
        p=cp.asnumpy(mesh.p).astype(np.float32),
        solid=cp.asnumpy(mesh.solid).astype(np.uint8),
        x=cp.asnumpy(mesh.X_1d).astype(np.float32),
        y=cp.asnumpy(mesh.Y_1d).astype(np.float32),
        alpha_deg=np.float32(cfg.get("alpha_deg", 0.0)),
        Re=np.float32(float(cfg.get("v0x", 1.0)) / max(float(cfg.get("nu", 1e-30)), 1e-30)),
        iteraciones=np.int32(iter_done),
    )
    return {"state_npz": str(state_path.relative_to(ROOT_DIR))}


def run_case(case: Case, base_cfg: dict[str, Any], out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    case_dir = out_dir / "cases" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(base_cfg)
    cfg.update(case.overrides)
    cfg["nu"] = float(cfg.get("v0x", 1.0)) / max(float(args.reynolds), 1e-30)
    cfg["frames_dir_grueso"] = str(case_dir / "frames_velocity_full")
    cfg["frames_dir_refinado"] = str(case_dir / "frames_velocity_zoom")
    cfg["frames_dir_presion"] = str(case_dir / "frames_pressure_full")
    for key in ("frames_dir_grueso", "frames_dir_refinado", "frames_dir_presion"):
        Path(cfg[key]).mkdir(parents=True, exist_ok=True)
    (case_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")

    print(f"[run] {case.case_id}: {case.description}", flush=True)
    t0 = time.time()
    mesh = sim_main(**cfg)
    elapsed = time.time() - t0

    row: dict[str, Any] = {
        "case_id": case.case_id,
        "status": "ok",
        "description": case.description,
        "case_dir": str(case_dir.relative_to(ROOT_DIR)),
        "alpha_deg": float(cfg.get("alpha_deg", 0.0)),
        "Re": float(args.reynolds),
        "nu": float(cfg["nu"]),
        "dx_min": float(cfg["dx_min"]),
        "CFL": float(cfg["CFL"]),
        "iteraciones": int(cfg["iteraciones"]),
        "guardado": int(cfg["guardado"]),
        "frame_every": int(cfg["save_frames_cada"]),
        "usar_wale": bool(cfg.get("usar_wale", False)),
        "wale_Cw": finite_or_nan(cfg.get("wale_Cw")),
        "projection_variant": str(cfg.get("projection_variant", "")),
        "wake_refinement_mode": str(cfg.get("wake_refinement_mode", "")),
        "ancho_zona_fina_x": finite_or_nan(cfg.get("ancho_zona_fina_x")),
        "ancho_zona_fina_y": finite_or_nan(cfg.get("ancho_zona_fina_y")),
        "nx": int(mesh.nx),
        "ny": int(mesh.ny),
        "n_cells": int(mesh.nx * mesh.ny),
        "elapsed_s": elapsed,
        "its_per_s": int(cfg["iteraciones"]) / elapsed if elapsed > 0 else float("nan"),
    }
    row.update(write_history(case_dir, mesh, int(cfg["guardado"])))
    row["history_plot"] = plot_history(case_dir)
    row.update(boundary_layer_audit(case_dir, mesh, cfg))
    row.update(cp_profile_audit(case_dir, mesh, cfg))
    row.update(wake_metrics(mesh, cfg))
    row.update(wale_metrics(mesh, cfg))
    row.update(save_final_fields(case_dir, mesh, cfg))
    if not args.no_videos:
        video_meta = create_videos(case_dir, int(args.fps))
        (case_dir / "videos.json").write_text(json.dumps(video_meta, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        for name, info in video_meta.items():
            row[f"video_{name}"] = info.get("video", "")
            row[f"video_{name}_status"] = info.get("status", "")

    (case_dir / "case_summary.json").write_text(json.dumps(row, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    del mesh
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return row


def load_existing(summary_path: Path) -> dict[str, dict[str, Any]]:
    if not summary_path.exists():
        return {}
    try:
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
        return {str(row.get("case_id")): row for row in rows if row.get("status") == "ok"}
    except Exception:
        return {}


def save_summary(out_dir: Path, rows_by_id: dict[str, dict[str, Any]], cases: list[Case]) -> None:
    ordered = [rows_by_id[c.case_id] for c in cases if c.case_id in rows_by_id]
    extras = [rows_by_id[k] for k in sorted(rows_by_id) if k not in {c.case_id for c in cases}]
    rows = ordered + extras
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    if rows:
        preferred = [
            "case_id", "status", "alpha_deg", "Re", "dx_min", "CFL", "usar_wale", "wale_Cw",
            "projection_variant", "wake_refinement_mode", "Cl_final", "Cl_late_mean", "Cl_late_std",
            "Cd_final", "Cd_late_mean", "div_flux_final", "div_flux_late_mean",
            "wall_leak_max_final", "wall_leak_max_late_mean", "y_plus_mean", "y_plus_p95",
            "y_plus_max", "y_plus_above_5_frac", "cells_delta99_x005_098_p05",
            "cells_delta99_x005_098_mean", "cells_delta99_x005_098_below_10_frac",
            "cells_delta99_min", "cells_delta99_p05", "cells_delta99_mean",
            "cells_delta99_below_10_frac", "reverse_tau_upper_frac",
            "reverse_tau_lower_frac", "surface_Cl_from_Cp_force_consistent",
            "surface_Cd_from_Cp_force_consistent", "Cl_from_Cp_mean_profile",
            "nu_t_over_nu_mean", "nu_t_over_nu_p95", "nu_t_over_nu_max",
            "wake_omega_rms", "wake_omega_abs_max", "wake_speed_deficit_mean",
            "case_dir", "history_csv", "surface_boundary_layer_csv", "boundary_layer_plot",
            "cp_profile_csv", "state_npz", "elapsed_s", "its_per_s",
        ]
        keys = sorted({k for row in rows for k in row})
        fieldnames = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]
        with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    write_report(out_dir, rows)


def write_report(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Diagnostico nocturno BL NACA0012",
        "",
        "Lectura rapida:",
        "- `y_plus_*` estima si la primera celda esta en rango wall-resolved.",
        "- `cells_delta99_*` estima cuantas celdas caben en la capa limite laminar teorica.",
        "- `reverse_tau_*` marca cambio de signo de esfuerzo cortante como proxy de separacion.",
        "- `surface_Cl_from_Cp_force_consistent` debe compararse contra `Cl_final`; si no coinciden, el Cp medio no es verdad suficiente.",
        "",
        "| case | Cl_final | y+ p95 | cells/d99 p05 | rev upper | nu_t/nu p95 | wake omega rms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {cl:.4g} | {yp:.4g} | {cells:.4g} | {rev:.2%} | {nut:.4g} | {wake:.4g} |".format(
                case=row.get("case_id", ""),
                cl=finite_or_nan(row.get("Cl_final")),
                yp=finite_or_nan(row.get("y_plus_p95")),
                cells=finite_or_nan(row.get("cells_delta99_x005_098_p05")),
                rev=finite_or_nan(row.get("reverse_tau_upper_frac")),
                nut=finite_or_nan(row.get("nu_t_over_nu_p95")),
                wake=finite_or_nan(row.get("wake_omega_rms")),
            )
        )
    lines.extend([
        "",
        "Reglas de decision:",
        "- Si `no_wale` elimina el desprendimiento, WALE esta introduciendo o amplificando la inestabilidad.",
        "- Si `cfl_025` mejora mucho, el problema es temporal/numerico antes que fisico.",
        "- Si `compatible_flux` mejora, priorizar proyeccion/presion.",
        "- Si `long_wake` mejora, la transicion de malla/estela esta contaminando el borde de salida.",
        "- Si `fine_wall_y` sube mucho las celdas por `delta99` y mejora, faltaba resolucion de BL.",
        "",
    ])
    (out_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = ROOT_DIR / "results" / "diagnosticos" / safe_name(args.output_suffix)
    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(args)
    base_cfg = build_base_config(args, out_dir)
    (out_dir / "suite_config.json").write_text(
        json.dumps({"args": vars(args), "base_config": base_cfg, "cases": [c.__dict__ for c in cases]}, indent=2, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    rows_by_id = load_existing(out_dir / "summary.json")
    print(f"Salida: {out_dir.relative_to(ROOT_DIR)}", flush=True)
    print(f"Casos: {', '.join(c.case_id for c in cases)}", flush=True)
    for case in cases:
        if case.case_id in rows_by_id and not args.force:
            print(f"[skip] {case.case_id}: ya completado", flush=True)
            continue
        try:
            rows_by_id[case.case_id] = run_case(case, base_cfg, out_dir, args)
        except Exception as exc:
            rows_by_id[case.case_id] = {
                "case_id": case.case_id,
                "status": "failed",
                "description": case.description,
                "error": str(exc),
            }
            print(f"[failed] {case.case_id}: {exc}", flush=True)
        save_summary(out_dir, rows_by_id, cases)
    print(f"Resumen: {(out_dir / 'summary.csv').relative_to(ROOT_DIR)}", flush=True)
    print(f"Informe: {(out_dir / 'analysis_report.md').relative_to(ROOT_DIR)}", flush=True)
    failed = [row for row in rows_by_id.values() if row.get("status") != "ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

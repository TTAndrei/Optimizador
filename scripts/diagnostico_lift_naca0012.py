"""
Diagnostico del deficit de lift en NACA0012 a Re=1e6.

El script ejecuta una matriz reanudable de casos para separar causas:
difusion/WALE, proyeccion, convergencia temporal, resolucion IBM/malla y
consistencia entre lift de superficie, Cp LES y Cp inviscido.

Uso:
    python scripts/diagnostico_lift_naca0012.py --quick
    python scripts/diagnostico_lift_naca0012.py --full
    python scripts/diagnostico_lift_naca0012.py --full --cases current_a10 strict_a10
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-cache"))

from Simulador2D import main as sim_main  # noqa: E402
from bl_correction import cl_from_delta_cp, compute_corrected_forces  # noqa: E402


OUT_DIR = ROOT_DIR / "results" / "diagnosticos" / "diagnostico_lift_naca0012"
OUT_JSON = OUT_DIR / "summary.json"
OUT_CSV = OUT_DIR / "summary.csv"
OUT_PLAN = OUT_DIR / "plan.json"
OUT_VALIDATION = OUT_DIR / "validation.json"
OUT_FIELDS = OUT_DIR / "fields"

DISCARD_FRACS = (0.30, 0.50, 0.70)
DEFAULT_DISCARD_FRAC = 0.30
FIELD_DEBUG_ENABLED = True
FIELD_FRAME_EVERY = 500

BASE_CFG: dict[str, Any] = dict(
    filepath="profiles/NACA_0012",
    chord=1.0,
    alpha_deg=10.0,
    Lx=12.0,
    Ly=8.0,
    cx=2.0,
    dx_min=0.001,
    factor_expansion=1.10,
    ancho_zona_fina_x=1.2,
    ancho_zona_fina_y=1.0,
    v0x=1.0,
    v0y=0.0,
    rho=1.0,
    p0=0.0,
    nu=1e-6,
    CFL=0.5,
    iteraciones=2000,
    guardado=50,
    divergencia=0.10,
    usar_wale=True,
    wale_Cw=0.15,
    graficos=False,
    save_frames=False,
    live_view=False,
    mostrar_malla=False,
    stop_on_convergence=False,
    corregir_deriva_vertical=False,
)

CURRENT_PROJECTION = dict(
    mg_modo_turbo=False,
    mg_modo_turbo_hd=True,
    mg_modo_turbo_ultra=False,
)

STRICT_PROJECTION = dict(
    mg_modo_turbo=False,
    mg_modo_turbo_hd=False,
    mg_modo_turbo_ultra=False,
    divergencia=0.01,
    mg_max_outer=8,
    mg_cycles_per_outer=5,
    mg_pre_suavizado=3,
    mg_post_suavizado=3,
    mg_guard_residual_every_outer=True,
    mg_adaptive_outer0_cycles=False,
    mg_apply_ibm_each_outer=True,
    mg_rollback_on_nan=True,
    mg_compute_div_after=True,
    mg_niveles_max=1,
)

VERY_STRICT_PROJECTION = {
    **STRICT_PROJECTION,
    "divergencia": 0.005,
    "mg_max_outer": 12,
    "mg_cycles_per_outer": 6,
}


@dataclass(frozen=True)
class Case:
    case_id: str
    group: str
    description: str
    cfg: dict[str, Any]
    discard_frac: float = DEFAULT_DISCARD_FRAC


def _case(case_id: str, group: str, description: str,
          overrides: dict[str, Any] | None = None,
          discard_frac: float = DEFAULT_DISCARD_FRAC) -> Case:
    cfg = {**BASE_CFG, **CURRENT_PROJECTION}
    if overrides:
        cfg.update(overrides)
    return Case(case_id, group, description, cfg, discard_frac)


def build_cases(mode: str) -> list[Case]:
    quick = [
        _case("current_a0", "baseline", "Configuracion actual, alpha=0",
              {"alpha_deg": 0.0}),
        _case("current_a10", "baseline", "Configuracion actual, alpha=10",
              {"alpha_deg": 10.0}),
        _case("fixed_geom_a10", "fixed_geometry",
              "Geometria fija a 0deg con flujo inclinado, alpha=10",
              {"alpha_deg": 10.0, "usar_flujo_inclinado": True,
               "flujo_inclinado_signo": -1.0, "flujo_inclinado_bc": "auto_farfield"}),
        _case("wale_off_a10", "wale", "WALE desactivado, alpha=10",
              {"alpha_deg": 10.0, "usar_wale": False}),
        _case("strict_a10", "projection", "Proyeccion estricta, alpha=10",
              {"alpha_deg": 10.0, **STRICT_PROJECTION}),
    ]
    if mode == "quick":
        return quick

    full = [
        _case("current_a5", "baseline", "Configuracion actual, alpha=5",
              {"alpha_deg": 5.0}),
        _case("fixed_geom_a0", "fixed_geometry",
              "Geometria fija a 0deg con flujo inclinado, alpha=0",
              {"alpha_deg": 0.0, "usar_flujo_inclinado": True,
               "flujo_inclinado_signo": -1.0, "flujo_inclinado_bc": "auto_farfield"}),
        _case("fixed_geom_a5", "fixed_geometry",
              "Geometria fija a 0deg con flujo inclinado, alpha=5",
              {"alpha_deg": 5.0, "usar_flujo_inclinado": True,
               "flujo_inclinado_signo": -1.0, "flujo_inclinado_bc": "auto_farfield"}),
        _case("fixed_geom_flow_pos_a5", "fixed_geometry",
              "Geometria fija, flujo +alpha con top/bottom inflow, alpha=5",
              {"alpha_deg": 5.0, "usar_flujo_inclinado": True,
               "flujo_inclinado_signo": 1.0, "flujo_inclinado_bc": "inflow_top_bottom"}),
        _case("fixed_geom_flow_neg_a5", "fixed_geometry",
              "Geometria fija, flujo -alpha con top/bottom inflow, alpha=5",
              {"alpha_deg": 5.0, "usar_flujo_inclinado": True,
               "flujo_inclinado_signo": -1.0, "flujo_inclinado_bc": "inflow_top_bottom"}),
        _case("fixed_geom_flow_pos_a10", "fixed_geometry",
              "Geometria fija, flujo +alpha con top/bottom inflow, alpha=10",
              {"alpha_deg": 10.0, "usar_flujo_inclinado": True,
               "flujo_inclinado_signo": 1.0, "flujo_inclinado_bc": "inflow_top_bottom"}),
        _case("fixed_geom_flow_neg_a10", "fixed_geometry",
              "Geometria fija, flujo -alpha con top/bottom inflow, alpha=10",
              {"alpha_deg": 10.0, "usar_flujo_inclinado": True,
               "flujo_inclinado_signo": -1.0, "flujo_inclinado_bc": "inflow_top_bottom"}),
        _case("fixed_geom_open_farfield_a5", "fixed_geometry",
              "Geometria fija, fronteras oblicuas por U dot n, alpha=5",
              {"alpha_deg": 5.0, "usar_flujo_inclinado": True,
               "flujo_inclinado_signo": -1.0, "flujo_inclinado_bc": "auto_farfield"}),
        _case("fixed_geom_open_farfield_a10", "fixed_geometry",
              "Geometria fija, fronteras oblicuas por U dot n, alpha=10",
              {"alpha_deg": 10.0, "usar_flujo_inclinado": True,
               "flujo_inclinado_signo": -1.0, "flujo_inclinado_bc": "auto_farfield"}),
        _case("fixed_geom_flow_pos_a30", "fixed_geometry",
              "Geometria fija, flujo +alpha con top/bottom inflow, alpha=30",
              {"alpha_deg": 30.0, "usar_flujo_inclinado": True,
               "flujo_inclinado_signo": 1.0, "flujo_inclinado_bc": "inflow_top_bottom"}),
        _case("fixed_geom_flow_neg_a30", "fixed_geometry",
              "Geometria fija, flujo -alpha con top/bottom inflow, alpha=30",
              {"alpha_deg": 30.0, "usar_flujo_inclinado": True,
               "flujo_inclinado_signo": -1.0, "flujo_inclinado_bc": "inflow_top_bottom"}),
        _case("fixed_geom_open_farfield_a30", "fixed_geometry",
              "Geometria fija, fronteras oblicuas por U dot n, alpha=30",
              {"alpha_deg": 30.0, "usar_flujo_inclinado": True,
               "flujo_inclinado_signo": -1.0, "flujo_inclinado_bc": "auto_farfield"}),
        _case("fixed_geom_flow_pos_a45", "fixed_geometry",
              "Geometria fija, flujo +alpha con top/bottom inflow, alpha=45",
              {"alpha_deg": 45.0, "usar_flujo_inclinado": True,
               "flujo_inclinado_signo": 1.0, "flujo_inclinado_bc": "inflow_top_bottom"}),
        _case("fixed_geom_flow_neg_a45", "fixed_geometry",
              "Geometria fija, flujo -alpha con top/bottom inflow, alpha=45",
              {"alpha_deg": 45.0, "usar_flujo_inclinado": True,
               "flujo_inclinado_signo": -1.0, "flujo_inclinado_bc": "inflow_top_bottom"}),
        _case("fixed_geom_open_farfield_a45", "fixed_geometry",
              "Geometria fija, fronteras oblicuas por U dot n, alpha=45",
              {"alpha_deg": 45.0, "usar_flujo_inclinado": True,
               "flujo_inclinado_signo": -1.0, "flujo_inclinado_bc": "auto_farfield"}),
        _case("wale_cw_005_a10", "wale", "WALE Cw=0.05, alpha=10",
              {"alpha_deg": 10.0, "usar_wale": True, "wale_Cw": 0.05}),
        _case("wale_cw_010_a10", "wale", "WALE Cw=0.10, alpha=10",
              {"alpha_deg": 10.0, "usar_wale": True, "wale_Cw": 0.10}),
        _case("wale_cw_015_a10", "wale", "WALE Cw=0.15, alpha=10",
              {"alpha_deg": 10.0, "usar_wale": True, "wale_Cw": 0.15}),
        _case("wale_cw_0325_a10", "wale", "WALE Cw=0.325, alpha=10",
              {"alpha_deg": 10.0, "usar_wale": True, "wale_Cw": 0.325}),
        _case("very_strict_l0_a10", "projection",
              "Proyeccion muy estricta sin coarsening, alpha=10",
              {"alpha_deg": 10.0, **VERY_STRICT_PROJECTION, "mg_niveles_max": 0}),
        _case("very_strict_l1_a10", "projection",
              "Proyeccion muy estricta con 1 nivel MG, alpha=10",
              {"alpha_deg": 10.0, **VERY_STRICT_PROJECTION, "mg_niveles_max": 1}),
        _case("strict_adjoint_a10", "projection",
              "Proyeccion estricta con gradiente adjunto, alpha=10",
              {"alpha_deg": 10.0, **STRICT_PROJECTION,
               "usar_adjoint_correction": True}),
        _case("time_2000_a10", "time", "2000 iteraciones, alpha=10",
              {"alpha_deg": 10.0, "iteraciones": 2000}),
        _case("time_5000_a10", "time", "5000 iteraciones, alpha=10",
              {"alpha_deg": 10.0, "iteraciones": 5000}),
        _case("time_10000_a10", "time", "10000 iteraciones, alpha=10",
              {"alpha_deg": 10.0, "iteraciones": 10000}),
        _case("res_dx_0010_a10", "resolution", "dx_min=0.001, alpha=10",
              {"alpha_deg": 10.0, "dx_min": 0.001}),
        _case("res_dx_0007_a10", "resolution", "dx_min=0.0007, alpha=10",
              {"alpha_deg": 10.0, "dx_min": 0.0007}),
        _case("res_dx_0005_a10", "resolution", "dx_min=0.0005, alpha=10",
              {"alpha_deg": 10.0, "dx_min": 0.0005}),
    ]
    return quick + full


def load_done() -> dict[str, dict[str, Any]]:
    if not OUT_JSON.exists():
        return {}
    with OUT_JSON.open(encoding="utf-8") as f:
        rows = json.load(f)
    return {str(r["case_id"]): r for r in rows}


def save_json(done: dict[str, dict[str, Any]], cases: list[Case]) -> None:
    ordered_ids = [c.case_id for c in cases if c.case_id in done]
    extras = sorted(k for k in done if k not in set(ordered_ids))
    ordered = [done[k] for k in ordered_ids + extras]
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=True)


def save_csv(done: dict[str, dict[str, Any]], cases: list[Case]) -> None:
    ordered_ids = [c.case_id for c in cases if c.case_id in done]
    extras = sorted(k for k in done if k not in set(ordered_ids))
    rows = [done[k] for k in ordered_ids + extras]
    if not rows:
        return
    fixed = [
        "case_id", "group", "status", "description",
        "alpha_deg", "Re", "dx_min", "iteraciones", "discard_frac",
        "usar_wale", "wale_Cw", "projection_label", "usar_adjoint_correction",
        "usar_flujo_inclinado", "flujo_inclinado_signo",
        "flujo_inclinado_angulo_deg", "flujo_inclinado_bc",
        "nx", "ny", "n_cells",
        "Cl_mean", "Cl_final", "Cl_std",
        "Cd_mean", "Cd_final", "Cd_std",
        "Ef_mean", "Ef_final",
        "Cl_p", "Cl_v", "Cd_p", "Cd_v",
        "Cl_from_Cp_LES", "Cl_bl", "Cd_bl", "Ef_bl", "Cl_inviscid",
        "div_mean", "div_final", "div_max_mean", "div_max_final",
        "mg_cycles_mean", "mg_cycles_max", "nu_t_over_nu_mean",
        "nu_t_over_nu_max", "elapsed_s", "its_per_s",
        "field_velocity_final", "field_vectors_final", "field_zoom_final",
        "field_frames_dir",
        "diagnosis_hint",
    ]
    keys = sorted({k for row in rows for k in row.keys()})
    fieldnames = [k for k in fixed if k in keys] + [k for k in keys if k not in fixed]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_plan(done: dict[str, dict[str, Any]], cases: list[Case]) -> None:
    entries = []
    for i, c in enumerate(cases, 1):
        row = done.get(c.case_id, {})
        entries.append({
            "n": i,
            "case_id": c.case_id,
            "group": c.group,
            "status": row.get("status", "pending"),
            "description": c.description,
            "alpha_deg": c.cfg["alpha_deg"],
            "dx_min": c.cfg["dx_min"],
            "iteraciones": c.cfg["iteraciones"],
            "usar_wale": c.cfg["usar_wale"],
            "wale_Cw": c.cfg.get("wale_Cw"),
            "projection_label": projection_label(c.cfg),
            "usar_flujo_inclinado": bool(c.cfg.get("usar_flujo_inclinado", False)),
            "flujo_inclinado_signo": c.cfg.get("flujo_inclinado_signo"),
            "flujo_inclinado_angulo_deg": c.cfg.get("flujo_inclinado_angulo_deg"),
            "flujo_inclinado_bc": c.cfg.get("flujo_inclinado_bc"),
        })
    payload = {
        "total": len(cases),
        "done": sum(1 for e in entries if e["status"] in {"ok", "error"}),
        "ok": sum(1 for e in entries if e["status"] == "ok"),
        "error": sum(1 for e in entries if e["status"] == "error"),
        "pending": sum(1 for e in entries if e["status"] == "pending"),
        "steps": entries,
    }
    with OUT_PLAN.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def projection_label(cfg: dict[str, Any]) -> str:
    if cfg.get("usar_adjoint_correction"):
        return "strict_adjoint"
    if cfg.get("mg_modo_turbo_hd"):
        return "turbo_hd"
    if cfg.get("mg_modo_turbo_ultra"):
        return "turbo_ultra"
    div = float(cfg.get("divergencia", 0.10))
    outer = int(cfg.get("mg_max_outer", 8))
    cycles = int(cfg.get("mg_cycles_per_outer", 5))
    levels = int(cfg.get("mg_niveles_max", 1))
    if div <= 0.005 and outer >= 12:
        return f"very_strict_l{levels}"
    if div <= 0.01 and outer >= 8:
        return f"strict_l{levels}"
    return f"custom_l{levels}_div{div:g}_o{outer}_c{cycles}"


def _gpu_to_np(arr: Any) -> np.ndarray:
    if arr is None:
        return np.array([], dtype=float)
    if hasattr(arr, "get"):
        return cp.asnumpy(arr).astype(float).ravel()
    return np.asarray(arr, dtype=float).ravel()


def summarize_vector(arr: Any, discard_frac: float) -> tuple[float, float, float]:
    values = _gpu_to_np(arr)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    i0 = min(max(int(values.size * discard_frac), 0), values.size - 1)
    tail = values[i0:]
    return float(np.mean(tail)), float(values[-1]), float(np.std(tail))


def summarize_all_discards(prefix: str, arr: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for frac in DISCARD_FRACS:
        mean, final, std = summarize_vector(arr, frac)
        tag = f"{int(frac * 100):02d}"
        out[f"{prefix}_mean_d{tag}"] = mean
        out[f"{prefix}_final_d{tag}"] = final
        out[f"{prefix}_std_d{tag}"] = std
    return out


def finite_or_nan(value: Any) -> float:
    try:
        val = float(value)
    except Exception:
        return float("nan")
    return val if math.isfinite(val) else float("nan")


def save_cp_csv(case_id: str, mesh: Any, bl: dict[str, Any] | None) -> str:
    x_les, cpu_les, cpl_les = mesh.get_cp_profile_mean()
    x_inv = np.array([], dtype=float)
    cpu_inv = np.array([], dtype=float)
    cpl_inv = np.array([], dtype=float)
    if bl:
        x_inv = np.asarray(bl.get("x_norm_panel", []), dtype=float)
        cpu_inv = np.asarray(bl.get("Cp_upper_inv", []), dtype=float)
        cpl_inv = np.asarray(bl.get("Cp_lower_inv", []), dtype=float)

    path = OUT_DIR / f"cp_{case_id}.csv"
    n = max(len(x_les), len(x_inv))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "x_les", "Cp_upper_LES", "Cp_lower_LES",
            "x_inv", "Cp_upper_inv", "Cp_lower_inv",
        ])
        for i in range(n):
            writer.writerow([
                finite_or_nan(x_les[i]) if i < len(x_les) else "",
                finite_or_nan(cpu_les[i]) if i < len(cpu_les) else "",
                finite_or_nan(cpl_les[i]) if i < len(cpl_les) else "",
                finite_or_nan(x_inv[i]) if i < len(x_inv) else "",
                finite_or_nan(cpu_inv[i]) if i < len(cpu_inv) else "",
                finite_or_nan(cpl_inv[i]) if i < len(cpl_inv) else "",
            ])
    return str(path.relative_to(ROOT_DIR))


def extract_nut_stats(mesh: Any, nu: float) -> tuple[float, float]:
    if not bool(getattr(mesh, "usar_wale", False)):
        return float("nan"), float("nan")
    try:
        nu_t = mesh.compute_wale_viscosity()
        mask = ~mesh.solid
        ratio = nu_t[mask] / cp.float32(max(float(nu), 1e-30))
        if int(ratio.size) == 0:
            return float("nan"), float("nan")
        return float(cp.mean(ratio)), float(cp.max(ratio))
    except Exception:
        return float("nan"), float("nan")


def extract_surface_balance(mesh: Any, cfg: dict[str, Any]) -> dict[str, float]:
    try:
        diag = mesh.diagnose_surface_force_balance(
            mu=float(cfg["rho"]) * float(cfg["nu"]),
            rho=float(cfg["rho"]),
            n_extrap_layers=5,
            nbins=30,
            te_start=0.85,
            verbose=False,
            plot=False,
        )
    except Exception:
        return {}
    if not diag.get("ok"):
        return {}
    keys = [
        "Fy_total", "Fy_pressure", "Fy_viscous", "Fy_te",
        "te_share_abs_percent", "closure_nx_rel", "closure_ny_rel",
        "perimeter_diff_rel", "p_wall_mean", "Fy_bias_from_pmean",
    ]
    return {f"surface_{k}": finite_or_nan(diag.get(k)) for k in keys}


def save_velocity_debug_fields(case_id: str, mesh: Any, cfg: dict[str, Any]) -> dict[str, str]:
    case_dir = OUT_FIELDS / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "field_velocity_final": case_dir / "velocity_full_final.png",
        "field_vectors_final": case_dir / "velocity_vectors_final.png",
        "field_zoom_final": case_dir / "velocity_zoom_airfoil_final.png",
        "field_frames_dir": case_dir / "frames",
    }

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[field-debug] matplotlib no disponible: {exc}")
        return {}

    u_ref = math.hypot(float(cfg["v0x"]), float(cfg["v0y"]))

    try:
        mesh.visualize_velocity(
            title=f"{case_id} | velocidad final",
            show=False,
            save_path=str(paths["field_velocity_final"]),
        )
        mesh.visualize_velocity_vectors(
            title=f"{case_id} | velocidad final + vectores",
            max_arrows=2200,
            u_ref=u_ref,
            show=False,
            save_path=str(paths["field_vectors_final"]),
        )
    except Exception as exc:
        print(f"[field-debug] fallo guardando campo completo para {case_id}: {exc}")

    try:
        speed = cp.asnumpy(cp.sqrt(mesh.u * mesh.u + mesh.v * mesh.v))
        u = cp.asnumpy(mesh.u)
        v = cp.asnumpy(mesh.v)
        x = cp.asnumpy(mesh.XX)
        y = cp.asnumpy(mesh.YY)
        solid = cp.asnumpy(mesh.solid).astype(np.int32)

        cx = float(cfg.get("cx", 2.0))
        cy = float(cfg.get("cy", float(cfg.get("Ly", mesh.Ly)) / 2.0))
        chord = float(cfg.get("chord", 1.0))
        x0 = max(0.0, cx - 0.25 * chord)
        x1 = min(float(mesh.Lx), cx + 1.25 * chord)
        y0 = max(0.0, cy - 0.55 * chord)
        y1 = min(float(mesh.Ly), cy + 0.55 * chord)

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.pcolormesh(x, y, speed, cmap="rainbow", shading="auto")
        vmax = float(np.nanpercentile(speed, 99.5)) if speed.size else 1.0
        im.set_clim(0.0, max(vmax, 1e-12))
        fig.colorbar(im, ax=ax, label="|u|")

        mask_zoom = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        n_zoom = max(1, int(np.count_nonzero(mask_zoom)))
        stride = max(1, int(np.ceil(np.sqrt(n_zoom / 900.0))))
        xq = x[::stride, ::stride]
        yq = y[::stride, ::stride]
        uq = np.where(solid[::stride, ::stride] > 0, np.nan, u[::stride, ::stride])
        vq = np.where(solid[::stride, ::stride] > 0, np.nan, v[::stride, ::stride])
        ref_len = 0.06 * min(x1 - x0, y1 - y0)
        qscale = u_ref / ref_len if u_ref > 1e-12 and ref_len > 0 else 1.0
        ax.quiver(xq, yq, uq, vq, color="black", alpha=0.65,
                  angles="xy", scale_units="xy", scale=qscale, width=0.002)

        try:
            ax.contour(x, y, solid, levels=[0.5], colors="white", linewidths=1.1)
        except Exception:
            pass
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{case_id} | zoom perfil")
        ax.grid(True, alpha=0.15)
        fig.tight_layout()
        fig.savefig(paths["field_zoom_final"], dpi=220, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"[field-debug] fallo guardando zoom para {case_id}: {exc}")

    out: dict[str, str] = {}
    for key, path in paths.items():
        if key == "field_frames_dir":
            if path.exists():
                out[key] = str(path.relative_to(ROOT_DIR))
        elif path.exists():
            out[key] = str(path.relative_to(ROOT_DIR))
    return out


def run_case(case: Case) -> dict[str, Any]:
    cfg = dict(case.cfg)
    alpha = float(cfg["alpha_deg"])
    rho = float(cfg["rho"])
    nu = float(cfg["nu"])
    chord = float(cfg["chord"])
    u_inf = math.hypot(float(cfg["v0x"]), float(cfg["v0y"]))
    reynolds = u_inf * chord / max(nu, 1e-30)

    case_field_dir = OUT_FIELDS / case.case_id
    if FIELD_DEBUG_ENABLED:
        cfg["save_frames"] = True
        cfg["frames_dir_grueso"] = str(case_field_dir / "frames")
        cfg["save_frames_cada"] = FIELD_FRAME_EVERY

    t0 = time.time()
    mesh = sim_main(**cfg)
    elapsed = time.time() - t0

    cl_mean, cl_final, cl_std = summarize_vector(mesh.clvector, case.discard_frac)
    cd_mean, cd_final, cd_std = summarize_vector(mesh.cdvector, case.discard_frac)
    ef_mean = cl_mean / cd_mean if abs(cd_mean) > 1e-12 else float("nan")
    ef_final = cl_final / cd_final if abs(cd_final) > 1e-12 else float("nan")

    mu = rho * nu
    q_dyn = 0.5 * rho * u_inf * u_inf * chord
    forces = mesh.compute_drag_lift(mu, rho=rho, n_extrap_layers=5)

    row: dict[str, Any] = {
        "case_id": case.case_id,
        "group": case.group,
        "status": "ok",
        "description": case.description,
        "alpha_deg": alpha,
        "Re": reynolds,
        "dx_min": float(cfg["dx_min"]),
        "iteraciones": int(cfg["iteraciones"]),
        "guardado": int(cfg["guardado"]),
        "discard_frac": float(case.discard_frac),
        "usar_wale": bool(cfg["usar_wale"]),
        "wale_Cw": finite_or_nan(cfg.get("wale_Cw")),
        "projection_label": projection_label(cfg),
        "usar_adjoint_correction": bool(cfg.get("usar_adjoint_correction", False)),
        "usar_flujo_inclinado": bool(cfg.get("usar_flujo_inclinado", False)),
        "flujo_inclinado_signo": finite_or_nan(cfg.get("flujo_inclinado_signo")),
        "flujo_inclinado_angulo_deg": finite_or_nan(cfg.get("flujo_inclinado_angulo_deg")),
        "flujo_inclinado_bc": str(cfg.get("flujo_inclinado_bc", "")),
        "nx": int(mesh.nx),
        "ny": int(mesh.ny),
        "n_cells": int(mesh.nx * mesh.ny),
        "Cl_mean": cl_mean,
        "Cl_final": cl_final,
        "Cl_std": cl_std,
        "Cd_mean": cd_mean,
        "Cd_final": cd_final,
        "Cd_std": cd_std,
        "Ef_mean": ef_mean,
        "Ef_final": ef_final,
        "Cl_p": finite_or_nan(forces["Lift_p"] / q_dyn),
        "Cl_v": finite_or_nan(forces["Lift_v"] / q_dyn),
        "Cd_p": finite_or_nan(forces["Drag_p"] / q_dyn),
        "Cd_v": finite_or_nan(forces["Drag_v"] / q_dyn),
        "elapsed_s": elapsed,
        "its_per_s": float(cfg["iteraciones"] / elapsed) if elapsed > 0 else 0.0,
    }
    row.update(summarize_all_discards("Cl", mesh.clvector))
    row.update(summarize_all_discards("Cd", mesh.cdvector))

    div_mean, div_final, _ = summarize_vector(mesh.divvector, case.discard_frac)
    div_max_mean, div_max_final, _ = summarize_vector(
        getattr(mesh, "divvector_max", None), case.discard_frac)
    row.update({
        "div_mean": div_mean,
        "div_final": div_final,
        "div_max_mean": div_max_mean,
        "div_max_final": div_max_final,
    })

    cycles_np = _gpu_to_np(getattr(mesh, "mg_cycles_vector", None))
    cycles_np = cycles_np[np.isfinite(cycles_np)]
    row.update({
        "mg_cycles_mean": float(np.mean(cycles_np)) if cycles_np.size else float("nan"),
        "mg_cycles_max": float(np.max(cycles_np)) if cycles_np.size else float("nan"),
        "mg_cycles_sum": float(np.sum(cycles_np)) if cycles_np.size else float("nan"),
    })

    nu_t_mean, nu_t_max = extract_nut_stats(mesh, nu)
    row["nu_t_over_nu_mean"] = nu_t_mean
    row["nu_t_over_nu_max"] = nu_t_max

    bl: dict[str, Any] | None = None
    try:
        bl = compute_corrected_forces(
            mesh,
            filepath=str(ROOT_DIR / cfg["filepath"]),
            alpha_deg=alpha,
            rho=rho,
            chord=chord,
            v_inf=u_inf,
            Re=reynolds,
        )
        row.update({
            "Cl_bl": finite_or_nan(bl.get("Cl")),
            "Cd_bl": finite_or_nan(bl.get("Cd")),
            "Cd_p_bl": finite_or_nan(bl.get("Cd_p")),
            "Cd_visc_bl": finite_or_nan(bl.get("Cd_visc")),
            "Ef_bl": finite_or_nan(bl.get("Ef")),
            "Cl_inviscid": finite_or_nan(bl.get("Cl_inviscid")),
            "trans_x_upper": finite_or_nan(bl.get("trans_x_upper")),
            "trans_x_lower": finite_or_nan(bl.get("trans_x_lower")),
            "x_sep_upper": finite_or_nan(bl.get("x_sep_upper")),
            "x_sep_lower": finite_or_nan(bl.get("x_sep_lower")),
            "bl_separated": bool(bl.get("separated", False)),
            "bl_warn": "; ".join(str(w) for w in bl.get("warn", [])),
        })
    except Exception as exc:
        row.update({
            "Cl_bl": float("nan"),
            "Cd_bl": float("nan"),
            "Ef_bl": float("nan"),
            "Cl_inviscid": float("nan"),
            "bl_warn": f"BL failed: {exc}",
        })

    x_les, cpu_les, cpl_les = mesh.get_cp_profile_mean()
    if len(x_les) > 0:
        row["Cl_from_Cp_LES"] = finite_or_nan(
            cl_from_delta_cp(x_les, cpu_les, cpl_les, math.radians(alpha)))
    else:
        row["Cl_from_Cp_LES"] = float("nan")

    row.update(extract_surface_balance(mesh, cfg))
    row["cp_csv"] = save_cp_csv(case.case_id, mesh, bl)
    if FIELD_DEBUG_ENABLED:
        field_paths = save_velocity_debug_fields(case.case_id, mesh, cfg)
        row.update(field_paths)
        for label in ("field_velocity_final", "field_vectors_final", "field_zoom_final"):
            if label in field_paths:
                print(f"  [field-debug] {label}: {field_paths[label]}")

    del mesh
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return row


def error_row(case: Case, exc: BaseException) -> dict[str, Any]:
    cfg = case.cfg
    return {
        "case_id": case.case_id,
        "group": case.group,
        "status": "error",
        "description": case.description,
        "alpha_deg": cfg.get("alpha_deg"),
        "dx_min": cfg.get("dx_min"),
        "iteraciones": cfg.get("iteraciones"),
        "usar_wale": cfg.get("usar_wale"),
        "wale_Cw": cfg.get("wale_Cw"),
        "projection_label": projection_label(cfg),
        "usar_adjoint_correction": bool(cfg.get("usar_adjoint_correction", False)),
        "usar_flujo_inclinado": bool(cfg.get("usar_flujo_inclinado", False)),
        "flujo_inclinado_signo": cfg.get("flujo_inclinado_signo"),
        "flujo_inclinado_angulo_deg": cfg.get("flujo_inclinado_angulo_deg"),
        "flujo_inclinado_bc": cfg.get("flujo_inclinado_bc"),
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def case_hint(row: dict[str, Any], done: dict[str, dict[str, Any]]) -> str:
    if row.get("status") != "ok":
        return "caso fallido"
    cid = row["case_id"]
    current = done.get("current_a10")
    if cid == "current_a0":
        cl0 = abs(finite_or_nan(row.get("Cl_mean")))
        return "simetria correcta" if cl0 < 0.02 else "revisar simetria/integracion"
    if current and current.get("status") == "ok" and row.get("alpha_deg") == 10.0:
        dc = finite_or_nan(row.get("Cl_mean")) - finite_or_nan(current.get("Cl_mean"))
        if row.get("group") == "wale" and abs(dc) > 0.08:
            return "WALE/difusion cambia mucho Cl"
        if row.get("group") == "projection" and abs(dc) > 0.08:
            return "proyeccion cambia mucho Cl"
        if row.get("group") == "resolution" and abs(dc) > 0.08:
            return "resolucion/IBM cambia mucho Cl"
        if row.get("group") == "fixed_geometry" and abs(dc) > 0.08:
            return "rotar geometria cambia mucho Cl"
    cl_inv = finite_or_nan(row.get("Cl_inviscid"))
    cl_mean = finite_or_nan(row.get("Cl_mean"))
    if row.get("alpha_deg") == 10.0 and math.isfinite(cl_inv) and math.isfinite(cl_mean):
        if cl_inv > 1.0 and cl_mean < 0.8:
            return "geometria/panel ok; LES pierde succion/lift"
    return ""


def attach_hints(done: dict[str, dict[str, Any]]) -> None:
    for row in done.values():
        row["diagnosis_hint"] = case_hint(row, done)


def validate_results(done: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    ok_rows = [r for r in done.values() if r.get("status") == "ok"]
    error_rows = [r for r in done.values() if r.get("status") == "error"]
    add("no_failed_cases", len(error_rows) == 0,
        f"{len(error_rows)} casos fallidos")

    key_cols = ["Cl_mean", "Cd_mean", "Cl_p", "Cl_v", "Cl_from_Cp_LES"]
    bad = []
    for row in ok_rows:
        for key in key_cols:
            val = finite_or_nan(row.get(key))
            if not math.isfinite(val):
                bad.append(f"{row['case_id']}:{key}")
    add("no_nan_core_metrics", len(bad) == 0,
        ", ".join(bad[:12]) if bad else "metricas principales finitas")

    row0 = done.get("current_a0")
    if row0 and row0.get("status") == "ok":
        cl0 = finite_or_nan(row0.get("Cl_mean"))
        add("alpha0_symmetry", abs(cl0) < 0.02,
            f"current_a0 Cl_mean={cl0:+.6f}")

    inv_rows = [
        r for r in ok_rows
        if abs(finite_or_nan(r.get("alpha_deg")) - 10.0) < 1e-9
        and math.isfinite(finite_or_nan(r.get("Cl_inviscid")))
    ]
    if inv_rows:
        cl_inv = finite_or_nan(inv_rows[0].get("Cl_inviscid"))
        add("inviscid_alpha10_range", 1.05 <= cl_inv <= 1.25,
            f"Cl_inviscid alpha10={cl_inv:.6f}")

    missing_cp = [
        r["case_id"] for r in ok_rows
        if not (ROOT_DIR / str(r.get("cp_csv", ""))).exists()
    ]
    add("cp_files_exist", len(missing_cp) == 0,
        ", ".join(missing_cp[:12]) if missing_cp else "Cp CSV generado para cada caso ok")

    return {
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
        "n_ok_rows": len(ok_rows),
        "n_error_rows": len(error_rows),
    }


def generate_figures(done: dict[str, dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[fig] matplotlib no disponible: {exc}")
        return

    rows = [r for r in done.values() if r.get("status") == "ok"]
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for group in sorted({r["group"] for r in rows}):
        rr = [r for r in rows if r["group"] == group and r.get("alpha_deg") == 10.0]
        if not rr:
            continue
        labels = [r["case_id"].replace("_a10", "") for r in rr]
        vals = [finite_or_nan(r.get("Cl_mean")) for r in rr]
        ax.plot(labels, vals, marker="o", linewidth=1.5, label=group)
    ax.axhline(2.0 * math.pi * math.radians(10.0), color="black",
               linestyle="--", linewidth=1.0, label="2*pi*alpha")
    ax.set_ylabel("Cl_mean")
    ax.set_title("Diagnostico Cl a alpha=10")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "diagnostico_cl_alpha10.png", dpi=180)
    plt.close(fig)

    target = done.get("current_a10")
    if target and target.get("status") == "ok":
        cp_path = ROOT_DIR / target.get("cp_csv", "")
        if cp_path.exists():
            data = np.genfromtxt(cp_path, delimiter=",", names=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            for x_key, up_key, low_key, label in [
                ("x_les", "Cp_upper_LES", "Cp_lower_LES", "LES"),
                ("x_inv", "Cp_upper_inv", "Cp_lower_inv", "inviscido"),
            ]:
                x = np.atleast_1d(np.asarray(data[x_key], dtype=float))
                up = np.atleast_1d(np.asarray(data[up_key], dtype=float))
                low = np.atleast_1d(np.asarray(data[low_key], dtype=float))
                mask_up = np.isfinite(x) & np.isfinite(up)
                mask_low = np.isfinite(x) & np.isfinite(low)
                ax.plot(x[mask_up], up[mask_up], linewidth=1.4,
                        label=f"Cp upper {label}")
                ax.plot(x[mask_low], low[mask_low], linewidth=1.4,
                        linestyle="--", label=f"Cp lower {label}")
            ax.invert_yaxis()
            ax.set_xlabel("x/c")
            ax.set_ylabel("Cp")
            ax.set_title("Cp current_a10")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(OUT_DIR / "cp_current_a10.png", dpi=180)
            plt.close(fig)


def select_cases(cases: list[Case], ids: list[str] | None) -> list[Case]:
    if not ids:
        return cases
    wanted = set(ids)
    selected = [c for c in cases if c.case_id in wanted]
    missing = sorted(wanted - {c.case_id for c in selected})
    if missing:
        raise SystemExit(f"Case ids no encontrados: {', '.join(missing)}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostico reanudable del deficit de lift en NACA0012 Re=1e6."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true",
                      help="ejecuta un subconjunto pequeno de diagnostico")
    mode.add_argument("--full", action="store_true",
                      help="ejecuta toda la matriz diagnostica")
    parser.add_argument("--cases", nargs="*",
                        help="limita la ejecucion a case_id concretos")
    parser.add_argument("--force", action="store_true",
                        help="recalcula los casos aunque ya esten en summary.json")
    parser.add_argument("--list-cases", action="store_true",
                        help="muestra la matriz de casos y termina sin simular")
    parser.add_argument("--no-figures", action="store_true",
                        help="no genera PNGs al final")
    parser.add_argument("--no-field-debug", action="store_true",
                        help="no guarda campos de velocidad finales/frames por caso")
    parser.add_argument("--no-fail-on-validation", action="store_true",
                        help="no devuelve exit code 2 si falla una validacion")
    return parser.parse_args()


def cuda_available() -> tuple[bool, str]:
    try:
        n_devices = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        return False, str(exc)
    if n_devices <= 0:
        return False, "no CUDA-capable device is detected"
    return True, f"{n_devices} CUDA device(s)"


def main() -> int:
    global FIELD_DEBUG_ENABLED
    args = parse_args()
    FIELD_DEBUG_ENABLED = not bool(args.no_field_debug)
    mode = "full" if args.full else "quick"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_cases = build_cases(mode)
    cases = select_cases(all_cases, args.cases)

    if args.list_cases:
        for c in cases:
            print(
                f"{c.case_id:<22} group={c.group:<10} "
                f"alpha={float(c.cfg['alpha_deg']):>5.1f} "
                f"dx={float(c.cfg['dx_min']):.4g} "
                f"iter={int(c.cfg['iteraciones']):>5d} "
                f"wale={int(bool(c.cfg['usar_wale']))} "
                f"Cw={float(c.cfg.get('wale_Cw', float('nan'))):.4g} "
                f"fixed_geom={int(bool(c.cfg.get('usar_flujo_inclinado', False)))} "
                f"flow_sign={float(c.cfg.get('flujo_inclinado_signo', float('nan'))):>4.1f} "
                f"flow_bc={c.cfg.get('flujo_inclinado_bc', '')} "
                f"proj={projection_label(c.cfg)}"
            )
        return 0

    ok_cuda, cuda_msg = cuda_available()
    if not ok_cuda:
        print(f"[ERROR] No se pueden ejecutar simulaciones: {cuda_msg}")
        print("        No se ha modificado el checkpoint de resultados.")
        return 3

    done = {} if args.force else load_done()

    print(f"\nDiagnostico lift NACA0012 Re=1e6")
    print(f"Salida: {OUT_DIR.relative_to(ROOT_DIR)}")
    print(f"Modo: {mode} | casos seleccionados: {len(cases)}")

    for idx, case in enumerate(cases, 1):
        if not args.force and done.get(case.case_id, {}).get("status") == "ok":
            print(f"[{idx}/{len(cases)}] {case.case_id}: ya existe, salto")
            continue
        print(f"[{idx}/{len(cases)}] {case.case_id}: {case.description}", flush=True)
        try:
            row = run_case(case)
            done[case.case_id] = row
            print(
                f"  Cl={row['Cl_mean']:+.5f} Cd={row['Cd_mean']:.5f} "
                f"Cl_inv={row.get('Cl_inviscid', float('nan')):+.5f} "
                f"div={row.get('div_mean', float('nan')):.4g} "
                f"({row['elapsed_s']:.1f}s)"
            )
        except Exception as exc:
            done[case.case_id] = error_row(case, exc)
            print(f"  ERROR: {exc}")
            traceback.print_exc()

        attach_hints(done)
        save_json(done, all_cases)
        save_csv(done, all_cases)
        save_plan(done, all_cases)

    attach_hints(done)
    save_json(done, all_cases)
    save_csv(done, all_cases)
    save_plan(done, all_cases)

    if not args.no_figures:
        generate_figures(done)

    validation = validate_results({cid: done[cid] for cid in done if cid in {c.case_id for c in cases}})
    with OUT_VALIDATION.open("w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2, ensure_ascii=True)

    print(f"\nResumen: {OUT_CSV.relative_to(ROOT_DIR)}")
    print(f"Validacion: {'OK' if validation['ok'] else 'FALLO'}")
    for check in validation["checks"]:
        mark = "OK" if check["ok"] else "FAIL"
        print(f"  [{mark}] {check['name']}: {check['detail']}")

    if not validation["ok"] and not args.no_fail_on_validation:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

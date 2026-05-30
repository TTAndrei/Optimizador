"""
Diagnostico de campo de presiones para el deficit de Cl en NACA0012.

El objetivo es ejecutar una matriz reanudable centrada en rotated_geom y
generar todos los CSV/PNG necesarios para decidir si el deficit de lift viene
del borde de ataque, IBM/rasterizacion, dominio/BC, WALE o lectura de p_wall.

Uso principal:
    python scripts/diagnostico_presion_lift_naca0012.py --suite full --force --no-fail-on-validation

Uso rapido:
    python scripts/diagnostico_presion_lift_naca0012.py --suite smoke --force --no-fail-on-validation
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


OUT_DIR = ROOT_DIR / "results" / "diagnosticos" / "diagnostico_presion_lift_naca0012"
OUT_JSON = OUT_DIR / "summary.json"
OUT_CSV = OUT_DIR / "summary.csv"
OUT_PLAN = OUT_DIR / "plan.json"
OUT_VALIDATION = OUT_DIR / "validation.json"
OUT_CP_REGIONS = OUT_DIR / "cp_integrals_by_region.csv"
OUT_CP_PEAKS = OUT_DIR / "cp_peak_diagnostics.csv"
OUT_SENSITIVITY = OUT_DIR / "sensitivity_summary.csv"
OUT_PRESSURE_VARIANTS = OUT_DIR / "pressure_reading_variants.csv"
OUT_FORCE_AUDIT = OUT_DIR / "surface_force_audit.csv"
OUT_LE_LOCAL = OUT_DIR / "le_ibm_local_dump.csv"
OUT_LE_SUMMARY = OUT_DIR / "le_ibm_summary.csv"
OUT_LE_COMPARISON = OUT_DIR / "le_ibm_case_comparison.csv"
OUT_REPORT = OUT_DIR / "analysis_report.md"
OUT_FIELDS = OUT_DIR / "fields"

DISCARD_FRAC = 0.30
REGIONS = (
    (0.00, 0.02),
    (0.02, 0.05),
    (0.05, 0.10),
    (0.10, 0.20),
    (0.20, 0.50),
    (0.50, 1.00),
)
PRESSURE_VARIANTS = (
    ("p_wall_layer_0p5", "edges"),
    ("p_wall_linear_2_layers", "edges"),
    ("p_wall_linear_5_layers", "edges"),
    ("p_wall_lstsq_5_layers", "edges"),
    ("p_wall_lstsq_5_layers", "inflow"),
)


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
    projection_variant="legacy_centered",
    mg_pressure_accumulation="outer_sum",
    wall_pressure_gradient_mode="masked",
    ibm_wall_mode="ghost_noslip",
    ibm_sdf_smooth_passes=None,
    mg_modo_turbo=False,
    mg_modo_turbo_hd=True,
    mg_modo_turbo_ultra=False,
)


def configure_output_dir(output_suffix: str | None = None) -> None:
    global OUT_DIR, OUT_JSON, OUT_CSV, OUT_PLAN, OUT_VALIDATION
    global OUT_CP_REGIONS, OUT_CP_PEAKS, OUT_SENSITIVITY
    global OUT_PRESSURE_VARIANTS, OUT_FORCE_AUDIT, OUT_LE_LOCAL, OUT_LE_SUMMARY
    global OUT_LE_COMPARISON, OUT_REPORT, OUT_FIELDS

    suffix = str(output_suffix or "").strip()
    base_name = "diagnostico_presion_lift_naca0012"
    if suffix:
        safe_suffix = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in suffix)
        base_name = f"{base_name}_{safe_suffix}"

    OUT_DIR = ROOT_DIR / "results" / "diagnosticos" / base_name
    OUT_JSON = OUT_DIR / "summary.json"
    OUT_CSV = OUT_DIR / "summary.csv"
    OUT_PLAN = OUT_DIR / "plan.json"
    OUT_VALIDATION = OUT_DIR / "validation.json"
    OUT_CP_REGIONS = OUT_DIR / "cp_integrals_by_region.csv"
    OUT_CP_PEAKS = OUT_DIR / "cp_peak_diagnostics.csv"
    OUT_SENSITIVITY = OUT_DIR / "sensitivity_summary.csv"
    OUT_PRESSURE_VARIANTS = OUT_DIR / "pressure_reading_variants.csv"
    OUT_FORCE_AUDIT = OUT_DIR / "surface_force_audit.csv"
    OUT_LE_LOCAL = OUT_DIR / "le_ibm_local_dump.csv"
    OUT_LE_SUMMARY = OUT_DIR / "le_ibm_summary.csv"
    OUT_LE_COMPARISON = OUT_DIR / "le_ibm_case_comparison.csv"
    OUT_REPORT = OUT_DIR / "analysis_report.md"
    OUT_FIELDS = OUT_DIR / "fields"


@dataclass(frozen=True)
class Case:
    case_id: str
    group: str
    description: str
    cfg: dict[str, Any]
    discard_frac: float = DISCARD_FRAC


def finite_or_nan(value: Any) -> float:
    try:
        val = float(value)
    except Exception:
        return float("nan")
    return val if math.isfinite(val) else float("nan")


def _case(case_id: str, group: str, description: str,
          overrides: dict[str, Any] | None = None) -> Case:
    cfg = dict(BASE_CFG)
    if overrides:
        cfg.update(overrides)
    return Case(case_id=case_id, group=group, description=description, cfg=cfg)


def build_cases(suite: str) -> list[Case]:
    if suite == "le_ibm":
        inviscid = {"usar_wale": False, "nu": 1e-9}
        return [
            _case("rot_a10_ghost_noslip_baseline", "le_ibm",
                  "Baseline IBM ghost no-slip alpha=10",
                  {"alpha_deg": 10.0, "ibm_wall_mode": "ghost_noslip"}),
            _case("rot_a10_slip_only", "le_ibm",
                  "Slip-only IBM, WALE off, nu~0",
                  {"alpha_deg": 10.0, "ibm_wall_mode": "slip_only", **inviscid}),
            _case("rot_a10_solid_zero_only", "le_ibm",
                  "Solid-zero-only IBM, WALE off, nu~0",
                  {"alpha_deg": 10.0, "ibm_wall_mode": "solid_zero_only", **inviscid}),
            _case("rot_a10_no_viscous_ghost", "le_ibm",
                  "Ghost no-slip, WALE off, nu~0",
                  {"alpha_deg": 10.0, "ibm_wall_mode": "ghost_noslip", **inviscid}),
            _case("rot_a10_sdf_smooth_0", "le_ibm",
                  "Baseline with SDF smoothing disabled",
                  {"alpha_deg": 10.0, "ibm_wall_mode": "ghost_noslip",
                   "ibm_sdf_smooth_passes": 0}),
            _case("rot_a10_sdf_smooth_4", "le_ibm",
                  "Baseline with stronger SDF smoothing",
                  {"alpha_deg": 10.0, "ibm_wall_mode": "ghost_noslip",
                   "ibm_sdf_smooth_passes": 4}),
            _case("rot_a10_dx_0007_slip_only", "le_ibm",
                  "Slip-only IBM on finer dx=0.0007",
                  {"alpha_deg": 10.0, "dx_min": 0.0007,
                   "ibm_wall_mode": "slip_only", **inviscid}),
        ]

    if suite == "projection_fix":
        accum = {"mg_pressure_accumulation": "outer_sum"}
        onesided = {"wall_pressure_gradient_mode": "one_sided"}
        both = {**accum, **onesided}
        return [
            _case("rot_a0_projection_default", "projection_fix",
                  "Alpha=0 symmetry control with new default pressure accumulation",
                  {"alpha_deg": 0.0, **accum}),
            _case("rot_a10_legacy_last_masked", "projection_fix",
                  "Legacy alpha=10 with last pressure correction and masked wall gradient",
                  {"alpha_deg": 10.0, "mg_pressure_accumulation": "last",
                   "wall_pressure_gradient_mode": "masked"}),
            _case("rot_a10_baseline_current", "projection_fix",
                  "New default alpha=10 with accumulated MG pressure",
                  {"alpha_deg": 10.0, **accum}),
            _case("rot_a10_pressure_accum_wall", "projection_fix",
                  "Alpha=10 with accumulated MG pressure and one-sided wall gradient",
                  {"alpha_deg": 10.0, **both}),
            _case("rot_a10_projection_default_dx0007", "projection_fix",
                  "Alpha=10 new default on finer dx=0.0007",
                  {"alpha_deg": 10.0, "dx_min": 0.0007, **accum}),
        ]

    smoke = [
        _case("rot_a0_baseline", "base", "Rotated geometry alpha=0 control",
              {"alpha_deg": 0.0}),
        _case("rot_a10_baseline", "base", "Rotated geometry alpha=10 baseline",
              {"alpha_deg": 10.0}),
        _case("rot_a10_no_wale", "wale", "Rotated alpha=10 with WALE off",
              {"alpha_deg": 10.0, "usar_wale": False}),
        _case("rot_a10_domain_tall", "domain", "Rotated alpha=10 with Ly=12",
              {"alpha_deg": 10.0, "Ly": 12.0}),
    ]
    if suite == "smoke":
        return smoke

    full = [
        _case("rot_a5_baseline", "base", "Rotated geometry alpha=5 baseline",
              {"alpha_deg": 5.0}),
        _case("rot_a10_dx_0010", "resolution", "Rotated alpha=10 dx_min=0.0010",
              {"alpha_deg": 10.0, "dx_min": 0.0010}),
        _case("rot_a10_dx_0007", "resolution", "Rotated alpha=10 dx_min=0.0007",
              {"alpha_deg": 10.0, "dx_min": 0.0007}),
        _case("rot_a10_dx_0005", "resolution", "Rotated alpha=10 dx_min=0.0005",
              {"alpha_deg": 10.0, "dx_min": 0.0005}),
        _case("rot_a10_Lx12_Ly8_baseline", "domain", "Explicit Lx=12 Ly=8 baseline",
              {"alpha_deg": 10.0, "Lx": 12.0, "Ly": 8.0}),
        _case("rot_a10_Lx16_Ly8", "domain", "Rotated alpha=10 with Lx=16 Ly=8",
              {"alpha_deg": 10.0, "Lx": 16.0, "Ly": 8.0}),
        _case("rot_a10_Lx12_Ly12", "domain", "Rotated alpha=10 with Lx=12 Ly=12",
              {"alpha_deg": 10.0, "Lx": 12.0, "Ly": 12.0}),
        _case("rot_a10_Lx16_Ly12", "domain", "Rotated alpha=10 with Lx=16 Ly=12",
              {"alpha_deg": 10.0, "Lx": 16.0, "Ly": 12.0}),
        _case("rot_a10_topbottom_slip", "boundary", "Explicit top/bottom slip",
              {"alpha_deg": 10.0,
               "boundary_top": ("slip", None),
               "boundary_bottom": ("slip", None)}),
        _case("rot_a10_topbottom_open_farfield", "boundary",
              "Top/bottom open outflow pressure farfield",
              {"alpha_deg": 10.0,
               "boundary_top": ("outflow", 0.0),
               "boundary_bottom": ("outflow", 0.0)}),
        _case("rot_a10_wale_off", "wale", "Rotated alpha=10 WALE off",
              {"alpha_deg": 10.0, "usar_wale": False}),
        _case("rot_a10_wale_cw_005", "wale", "Rotated alpha=10 WALE Cw=0.05",
              {"alpha_deg": 10.0, "usar_wale": True, "wale_Cw": 0.05}),
        _case("rot_a10_wale_cw_015", "wale", "Rotated alpha=10 WALE Cw=0.15",
              {"alpha_deg": 10.0, "usar_wale": True, "wale_Cw": 0.15}),
        _case("rot_a10_wale_cw_0325", "wale", "Rotated alpha=10 WALE Cw=0.325",
              {"alpha_deg": 10.0, "usar_wale": True, "wale_Cw": 0.325}),
    ]
    return smoke[:2] + full


def select_cases(cases: list[Case], selected: list[str] | None) -> list[Case]:
    if not selected:
        return cases
    wanted = set(selected)
    picked = [c for c in cases if c.case_id in wanted]
    missing = sorted(wanted - {c.case_id for c in picked})
    if missing:
        raise SystemExit(f"cases desconocidos: {', '.join(missing)}")
    return picked


def load_done() -> dict[str, dict[str, Any]]:
    if OUT_JSON.exists():
        with OUT_JSON.open(encoding="utf-8") as f:
            rows = json.load(f)
    elif OUT_CSV.exists():
        rows = read_rows(OUT_CSV)
    else:
        return {}
    return {str(r["case_id"]): r for r in rows}


def save_json(done: dict[str, dict[str, Any]], cases: list[Case]) -> None:
    ordered = [done[c.case_id] for c in cases if c.case_id in done]
    extras = [done[k] for k in sorted(done) if k not in {c.case_id for c in cases}]
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(ordered + extras, f, indent=2, ensure_ascii=True)


def save_csv(done: dict[str, dict[str, Any]], cases: list[Case]) -> None:
    rows = [done[c.case_id] for c in cases if c.case_id in done]
    if not rows:
        return
    fixed = [
        "case_id", "group", "status", "description", "alpha_deg", "Re",
        "Lx", "Ly", "dx_min", "factor_expansion", "iteraciones",
        "usar_wale", "wale_Cw", "projection_variant",
        "mg_pressure_accumulation", "wall_pressure_gradient_mode",
        "ibm_wall_mode", "ibm_sdf_smooth_passes",
        "boundary_top", "boundary_bottom", "boundary_right",
        "nx", "ny", "n_cells",
        "Cl_mean", "Cl_final", "Cl_std", "Cl_p", "Cl_v",
        "Cl_from_Cp_LES", "Cl_from_Cp_force_consistent", "Cl_pressure_force",
        "Cl_inviscid", "Cl_final_over_inviscid",
        "Cl_from_Cp_LES_over_inviscid", "Cl_final_minus_Cp_LES",
        "Cl_final_minus_Cp_force_consistent",
        "Cp_force_consistent_range", "pressure_debias_model",
        "Cd_mean", "Cd_final", "Cd_p", "Cd_v",
        "div_mean", "div_final", "div_flux_final", "wall_leak_max_final",
        "nu_t_over_nu_mean", "nu_t_over_nu_max",
        "surface_closure_nx_rel", "surface_closure_ny_rel",
        "surface_perimeter_diff_rel", "surface_Fy_bias_from_pmean",
        "pressure_probable_cause", "elapsed_s", "its_per_s",
        "cp_csv", "field_velocity_final", "field_zoom_final", "field_pressure_zoom_final",
    ]
    keys = sorted({k for r in rows for k in r})
    fieldnames = [k for k in fixed if k in keys] + [k for k in keys if k not in fixed]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_plan(done: dict[str, dict[str, Any]], cases: list[Case]) -> None:
    payload = {
        "total": len(cases),
        "ok": sum(1 for c in cases if done.get(c.case_id, {}).get("status") == "ok"),
        "error": sum(1 for c in cases if done.get(c.case_id, {}).get("status") == "error"),
        "pending": sum(1 for c in cases if c.case_id not in done),
        "steps": [
            {
                "case_id": c.case_id,
                "group": c.group,
                "status": done.get(c.case_id, {}).get("status", "pending"),
                "description": c.description,
                "alpha_deg": c.cfg["alpha_deg"],
                "Lx": c.cfg["Lx"],
                "Ly": c.cfg["Ly"],
                "dx_min": c.cfg["dx_min"],
                "usar_wale": c.cfg["usar_wale"],
                "wale_Cw": c.cfg.get("wale_Cw"),
                "projection_variant": c.cfg.get("projection_variant", "legacy_centered"),
                "mg_pressure_accumulation": c.cfg.get("mg_pressure_accumulation", "outer_sum"),
                "wall_pressure_gradient_mode": c.cfg.get("wall_pressure_gradient_mode", "masked"),
                "ibm_wall_mode": c.cfg.get("ibm_wall_mode", "ghost_noslip"),
                "ibm_sdf_smooth_passes": c.cfg.get("ibm_sdf_smooth_passes"),
                "boundary_top": str(c.cfg.get("boundary_top", ("slip", None))),
                "boundary_bottom": str(c.cfg.get("boundary_bottom", ("slip", None))),
            }
            for c in cases
        ],
    }
    with OUT_PLAN.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


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


def save_cp_csv(case_id: str, x_les: np.ndarray, cpu_les: np.ndarray,
                cpl_les: np.ndarray, bl: dict[str, Any] | None) -> str:
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


def _ref_pressure_velocity(mesh: Any, rho: float, mode: str) -> tuple[Any, Any]:
    if mode == "inflow":
        p_ref = cp.mean(mesh.p[:, 0])
        U_ref = cp.sqrt(cp.mean(mesh.u[:, 0] ** 2 + mesh.v[:, 0] ** 2))
        return p_ref, U_ref

    band = max(1, int(min(mesh.nx, mesh.ny) * 0.05))
    mask = cp.zeros_like(mesh.p, dtype=cp.bool_)
    mask[:band, :] = True
    mask[-band:, :] = True
    mask[:, :band] = True
    mask[:, -band:] = True
    mask = mask & (~mesh.solid)
    if int(cp.sum(mask)) == 0:
        mask = ~mesh.solid
    p_ref = cp.mean(mesh.p[mask])
    U_ref = cp.sqrt(cp.mean(mesh.u[mask] ** 2 + mesh.v[mask] ** 2))
    return p_ref, U_ref


def _p_wall_from_layers(mesh: Any, variant: str, positions: Any,
                        p_layers: list[Any]) -> Any:
    if variant == "p_wall_layer_0p5":
        return p_layers[0]
    if variant == "p_wall_linear_2_layers":
        p1 = p_layers[0]
        p2 = p_layers[1]
        x1 = positions[0]
        x2 = positions[1]
        slope = (p2 - p1) / (x2 - x1)
        return p1 - slope * x1
    if variant == "p_wall_linear_5_layers":
        p1 = p_layers[0]
        pN = p_layers[-1]
        x1 = positions[0]
        xN = positions[-1]
        slope = (pN - p1) / (xN - x1)
        return p1 - slope * x1
    if variant == "p_wall_lstsq_5_layers":
        n_layers_f = cp.float32(len(p_layers))
        sum_x = cp.sum(positions)
        sum_x2 = cp.sum(positions * positions)
        sum_y = cp.zeros_like(p_layers[0], dtype=cp.float32)
        sum_xy = cp.zeros_like(p_layers[0], dtype=cp.float32)
        for k, pos in enumerate(positions):
            sum_y += p_layers[k]
            sum_xy += pos * p_layers[k]
        denom = n_layers_f * sum_x2 - sum_x * sum_x
        denom = cp.where(cp.abs(denom) < cp.float32(1e-12), cp.float32(1e-12), denom)
        slope = (n_layers_f * sum_xy - sum_x * sum_y) / denom
        return (sum_y - slope * sum_x) / n_layers_f
    raise ValueError(f"pressure variant desconocida: {variant}")


def _profile_from_faces(x_face: np.ndarray, ny_face: np.ndarray, cp_face: np.ndarray,
                        nbins: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x_face.size == 0:
        return np.array([]), np.array([]), np.array([])
    x_min = float(np.nanmin(x_face))
    x_max = float(np.nanmax(x_face))
    chord = max(x_max - x_min, 1e-12)
    x_norm = np.clip((x_face - x_min) / chord, 0.0, 1.0)
    idx = np.clip(np.floor(x_norm * (nbins - 1)).astype(int), 0, nbins - 1)
    x = np.linspace(0.0, 1.0, nbins)
    upper_sum = np.zeros(nbins, dtype=float)
    lower_sum = np.zeros(nbins, dtype=float)
    upper_count = np.zeros(nbins, dtype=float)
    lower_count = np.zeros(nbins, dtype=float)
    upper = ny_face > 0.0
    np.add.at(upper_sum, idx[upper], cp_face[upper])
    np.add.at(upper_count, idx[upper], 1.0)
    np.add.at(lower_sum, idx[~upper], cp_face[~upper])
    np.add.at(lower_count, idx[~upper], 1.0)
    cpu = np.zeros(nbins, dtype=float)
    cpl = np.zeros(nbins, dtype=float)
    np.divide(upper_sum, upper_count, out=cpu, where=upper_count > 0)
    np.divide(lower_sum, lower_count, out=cpl, where=lower_count > 0)
    return x, cpu, cpl


def pressure_reading_profiles(mesh: Any, rho: float) -> list[dict[str, Any]]:
    boundary = mesh._solid_boundary_mask(use_diagonals=True)
    _, nx_all, ny_all = mesh._signed_distance_and_normals()
    JJ, II = mesh.JJ, mesh.II
    positions = cp.array([0.5 + i for i in range(5)], dtype=cp.float32)
    p_layers = []
    for pos in positions:
        j_face_k = JJ + pos * nx_all
        i_face_k = II + pos * ny_all
        p_layers.append(mesh._bilinear_interpolate(mesh.p, j_face_k, i_face_k))

    j_face = JJ + cp.float32(0.5) * nx_all
    i_face = II + cp.float32(0.5) * ny_all
    x_face = mesh._bilinear_interpolate(mesh.XX, j_face, i_face)
    ny_face = ny_all

    out: list[dict[str, Any]] = []
    for variant, ref_mode in PRESSURE_VARIANTS:
        p_wall = _p_wall_from_layers(mesh, variant, positions, p_layers)
        p_ref, U_ref = _ref_pressure_velocity(mesh, rho, ref_mode)
        denom = cp.float32(0.5 * rho) * U_ref * U_ref + cp.float32(1e-12)
        cp_face = (p_wall - p_ref) / denom

        xf = cp.asnumpy(x_face[boundary]).astype(float)
        nf = cp.asnumpy(ny_face[boundary]).astype(float)
        cf = cp.asnumpy(cp_face[boundary]).astype(float)
        x, cpu, cpl = _profile_from_faces(xf, nf, cf)
        out.append({
            "pressure_variant": variant,
            "pressure_ref_mode": ref_mode,
            "x": x,
            "Cp_upper": cpu,
            "Cp_lower": cpl,
            "Cl_from_Cp": finite_or_nan(cl_from_delta_cp(x, cpu, cpl, 0.0)),
        })
    return out


def summarize_le_rows(case_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"case_id": case_id, "le_n_rows": len(rows)}
    if not rows:
        out.update({
            "le_abs_un_over_U_mean": float("nan"),
            "le_abs_un_over_U_max": float("nan"),
            "le_ghost_direct_zero_ratio": float("nan"),
            "le_image_in_fluid_ratio": float("nan"),
            "le_nx_mean": float("nan"),
            "le_ny_mean": float("nan"),
            "le_Cp_min": float("nan"),
            "le_x_Cp_min": float("nan"),
            "le_p_wall_std": float("nan"),
            "le_dCp_0_0p05": float("nan"),
            "le_dCp_0p05_0p10": float("nan"),
            "le_dCp_0p10_0p15": float("nan"),
        })
        return out

    def arr(name: str) -> np.ndarray:
        return np.asarray([finite_or_nan(r.get(name)) for r in rows], dtype=float)

    x = arr("x_over_c")
    cpv = arr("Cp")
    p_wall = arr("p_wall")
    un = np.abs(arr("u_n_over_U"))
    nx = arr("nx")
    ny = arr("ny")
    direct = np.asarray([bool(r.get("ghost_direct_zero")) for r in rows], dtype=bool)
    img_ok = np.asarray([bool(r.get("image_in_fluid")) for r in rows], dtype=bool)
    is_ghost = np.asarray([bool(r.get("is_ghost")) for r in rows], dtype=bool)

    if np.any(np.isfinite(cpv)):
        i_min = int(np.nanargmin(cpv))
        out["le_Cp_min"] = float(cpv[i_min])
        out["le_x_Cp_min"] = float(x[i_min])
    else:
        out["le_Cp_min"] = float("nan")
        out["le_x_Cp_min"] = float("nan")

    out.update({
        "le_abs_un_over_U_mean": float(np.nanmean(un)) if np.any(np.isfinite(un)) else float("nan"),
        "le_abs_un_over_U_max": float(np.nanmax(un)) if np.any(np.isfinite(un)) else float("nan"),
        "le_ghost_direct_zero_ratio": float(np.mean(direct)) if direct.size else float("nan"),
        "le_image_in_fluid_ratio": float(np.mean(img_ok[is_ghost])) if np.any(is_ghost) else float("nan"),
        "le_nx_mean": float(np.nanmean(nx)) if np.any(np.isfinite(nx)) else float("nan"),
        "le_ny_mean": float(np.nanmean(ny)) if np.any(np.isfinite(ny)) else float("nan"),
        "le_p_wall_std": float(np.nanstd(p_wall)) if np.any(np.isfinite(p_wall)) else float("nan"),
    })

    upper_rows = [r for r in rows if r.get("side") == "upper"]
    lower_rows = [r for r in rows if r.get("side") == "lower"]
    bins = np.linspace(0.0, 0.15, 31)
    centers = 0.5 * (bins[:-1] + bins[1:])
    upper_sum = np.zeros(centers.size, dtype=float)
    lower_sum = np.zeros(centers.size, dtype=float)
    upper_count = np.zeros(centers.size, dtype=float)
    lower_count = np.zeros(centers.size, dtype=float)
    for src, acc_sum, acc_count in [
        (upper_rows, upper_sum, upper_count),
        (lower_rows, lower_sum, lower_count),
    ]:
        for r in src:
            xv = finite_or_nan(r.get("x_over_c"))
            cv = finite_or_nan(r.get("Cp"))
            if not (math.isfinite(xv) and math.isfinite(cv) and 0.0 <= xv <= 0.15):
                continue
            idx = int(np.clip(np.searchsorted(bins, xv, side="right") - 1, 0, centers.size - 1))
            acc_sum[idx] += cv
            acc_count[idx] += 1.0
    valid_bins = (upper_count > 0.0) & (lower_count > 0.0)
    if np.count_nonzero(valid_bins) >= 2:
        cpu = np.divide(upper_sum, upper_count, out=np.full_like(upper_sum, np.nan), where=upper_count > 0.0)
        cpl = np.divide(lower_sum, lower_count, out=np.full_like(lower_sum, np.nan), where=lower_count > 0.0)
        dcp = cpl - cpu
        for lo, hi, key in [
            (0.0, 0.05, "le_dCp_0_0p05"),
            (0.05, 0.10, "le_dCp_0p05_0p10"),
            (0.10, 0.15, "le_dCp_0p10_0p15"),
        ]:
            m = (centers >= lo) & (centers <= hi) & valid_bins & np.isfinite(dcp)
            out[key] = float(np.trapezoid(dcp[m], centers[m])) if np.count_nonzero(m) >= 2 else float("nan")
    else:
        out["le_dCp_0_0p05"] = float("nan")
        out["le_dCp_0p05_0p10"] = float("nan")
        out["le_dCp_0p10_0p15"] = float("nan")
    return out


def le_summary_from_case_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "le_n_rows",
        "le_abs_un_over_U_mean",
        "le_abs_un_over_U_max",
        "le_ghost_direct_zero_ratio",
        "le_image_in_fluid_ratio",
        "le_nx_mean",
        "le_ny_mean",
        "le_Cp_min",
        "le_x_Cp_min",
        "le_p_wall_std",
        "le_dCp_0_0p05",
        "le_dCp_0p05_0p10",
        "le_dCp_0p10_0p15",
    ]
    out = {"case_id": str(row.get("case_id", ""))}
    for key in keys:
        out[key] = row.get(key, float("nan"))
    return out


def build_le_comparison(done: dict[str, dict[str, Any]],
                        le_summary_rows: list[dict[str, Any]],
                        suite: str | None = None) -> list[dict[str, Any]]:
    by_case = {str(r.get("case_id")): r for r in le_summary_rows}
    base_id = _baseline_case_id(done, suite)
    base = done.get(base_id)
    base_le = (
        by_case.get(base_id)
        or (le_summary_from_case_row(base) if base else {})
    )
    if not base:
        return []
    base_cl = finite_or_nan(base.get("Cl_from_Cp_LES"))
    base_cp_min = finite_or_nan(base_le.get("le_Cp_min"))
    rows = []
    for cid, row in sorted(done.items()):
        if row.get("status") != "ok":
            continue
        le = by_case.get(cid) or le_summary_from_case_row(row)
        cl = finite_or_nan(row.get("Cl_from_Cp_LES"))
        cp_min = finite_or_nan(le.get("le_Cp_min"))
        dcl = cl - base_cl if math.isfinite(cl) and math.isfinite(base_cl) else float("nan")
        dcp_min = cp_min - base_cp_min if math.isfinite(cp_min) and math.isfinite(base_cp_min) else float("nan")
        mode = str(row.get("ibm_wall_mode", "ghost_noslip"))
        if cid == base_id:
            verdict = "baseline"
        elif mode in {"slip_only", "solid_zero_only"} and math.isfinite(dcl) and dcl > 0.15:
            verdict = "ghost_noslip_artificial_bl_or_separation"
        elif "sdf_smooth" in cid and math.isfinite(dcl) and abs(dcl) > 0.08:
            verdict = "rotated_geometry_sdf_normal_bias"
        elif math.isfinite(dcl) and abs(dcl) <= 0.05:
            verdict = "pressure_projection_ibm_coupling"
        else:
            verdict = "mixed_or_inconclusive"
        rows.append({
            "case_id": cid,
            "group": row.get("group", ""),
            "ibm_wall_mode": mode,
            "ibm_sdf_smooth_passes": row.get("ibm_sdf_smooth_passes", ""),
            "Cl_from_Cp_LES": cl,
            "dCl_from_le_baseline": dcl,
            "le_Cp_min": cp_min,
            "d_le_Cp_min_from_baseline": dcp_min,
            "le_abs_un_over_U_max": finite_or_nan(le.get("le_abs_un_over_U_max")),
            "le_ghost_direct_zero_ratio": finite_or_nan(le.get("le_ghost_direct_zero_ratio")),
            "le_image_in_fluid_ratio": finite_or_nan(le.get("le_image_in_fluid_ratio")),
            "le_verdict": verdict,
        })
    return rows


def cp_region_rows(case_id: str, alpha_deg: float, source: str, x: np.ndarray,
                   cpu: np.ndarray, cpl: np.ndarray,
                   x_inv: np.ndarray | None = None,
                   cpu_inv: np.ndarray | None = None,
                   cpl_inv: np.ndarray | None = None) -> list[dict[str, Any]]:
    rows = []
    dcp = cpl - cpu
    inv_available = (
        x_inv is not None and cpu_inv is not None and cpl_inv is not None
        and len(x_inv) > 3 and len(cpu_inv) == len(x_inv) and len(cpl_inv) == len(x_inv)
    )
    dcp_inv = cpl_inv - cpu_inv if inv_available else np.array([], dtype=float)
    for a, b in REGIONS:
        m = (x >= a) & (x <= b)
        les = float(np.trapezoid(dcp[m], x[m])) if np.count_nonzero(m) >= 2 else float("nan")
        inv = float("nan")
        ratio = float("nan")
        if inv_available:
            mi = (x_inv >= a) & (x_inv <= b)
            if np.count_nonzero(mi) >= 2:
                inv = float(np.trapezoid(dcp_inv[mi], x_inv[mi]))
                ratio = les / inv if abs(inv) > 1e-12 and math.isfinite(les) else float("nan")
        rows.append({
            "case_id": case_id,
            "alpha_deg": alpha_deg,
            "source": source,
            "x_start": a,
            "x_end": b,
            "Cl_region": les * math.cos(math.radians(alpha_deg)) if math.isfinite(les) else float("nan"),
            "dCp_integral": les,
            "dCp_integral_inv": inv,
            "dCp_integral_over_inv": ratio,
            "n_points": int(np.count_nonzero(m)),
        })
    return rows


def cp_peak_row(case_id: str, alpha_deg: float, source: str, x: np.ndarray,
                cpu: np.ndarray, cpl: np.ndarray,
                x_inv: np.ndarray | None = None,
                cpu_inv: np.ndarray | None = None,
                cpl_inv: np.ndarray | None = None) -> dict[str, Any]:
    row = {"case_id": case_id, "alpha_deg": alpha_deg, "source": source}
    valid_u = np.isfinite(cpu)
    valid_l = np.isfinite(cpl)
    if np.any(valid_u):
        iu = int(np.nanargmin(np.where(valid_u, cpu, np.nan)))
        row["Cp_upper_min"] = finite_or_nan(cpu[iu])
        row["x_Cp_upper_min"] = finite_or_nan(x[iu])
    if np.any(valid_l):
        il = int(np.nanargmax(np.where(valid_l, cpl, np.nan)))
        row["Cp_lower_max"] = finite_or_nan(cpl[il])
        row["x_Cp_lower_max"] = finite_or_nan(x[il])
    if x_inv is not None and cpu_inv is not None and cpl_inv is not None and len(x_inv) > 3:
        iu_i = int(np.nanargmin(cpu_inv))
        il_i = int(np.nanargmax(cpl_inv))
        row["Cp_upper_min_inv"] = finite_or_nan(cpu_inv[iu_i])
        row["x_Cp_upper_min_inv"] = finite_or_nan(x_inv[iu_i])
        row["Cp_lower_max_inv"] = finite_or_nan(cpl_inv[il_i])
        row["x_Cp_lower_max_inv"] = finite_or_nan(x_inv[il_i])
        if abs(finite_or_nan(cpu_inv[iu_i])) > 1e-12:
            row["Cp_upper_min_over_inv"] = finite_or_nan(cpu[iu] / cpu_inv[iu_i]) if np.any(valid_u) else float("nan")
    return row


def save_field_figures(case_id: str, mesh: Any, cfg: dict[str, Any]) -> dict[str, str]:
    case_dir = OUT_FIELDS / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "field_velocity_final": case_dir / "velocity_full_final.png",
        "field_zoom_final": case_dir / "velocity_zoom_airfoil_final.png",
        "field_pressure_zoom_final": case_dir / "pressure_zoom_airfoil_final.png",
    }
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    try:
        mesh.visualize_velocity(
            title=f"{case_id} | velocidad final",
            show=False,
            save_path=str(paths["field_velocity_final"]),
        )
    except Exception as exc:
        print(f"[field-debug] fallo velocity para {case_id}: {exc}")

    try:
        speed = cp.asnumpy(cp.sqrt(mesh.u * mesh.u + mesh.v * mesh.v))
        p = cp.asnumpy(mesh.p)
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

        for key, field, cmap, label in [
            ("field_zoom_final", speed, "rainbow", "|u|"),
            ("field_pressure_zoom_final", p, "coolwarm", "p"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 5))
            im = ax.pcolormesh(x, y, field, cmap=cmap, shading="auto")
            if key == "field_zoom_final":
                vmax = float(np.nanpercentile(speed, 99.5)) if speed.size else 1.0
                im.set_clim(0.0, max(vmax, 1e-12))
                u_ref = math.hypot(float(cfg["v0x"]), float(cfg["v0y"]))
                mask_zoom = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
                n_zoom = max(1, int(np.count_nonzero(mask_zoom)))
                stride = max(1, int(np.ceil(np.sqrt(n_zoom / 900.0))))
                ref_len = 0.06 * min(x1 - x0, y1 - y0)
                qscale = u_ref / ref_len if u_ref > 1e-12 and ref_len > 0 else 1.0
                ax.quiver(
                    x[::stride, ::stride], y[::stride, ::stride],
                    np.where(solid[::stride, ::stride] > 0, np.nan, u[::stride, ::stride]),
                    np.where(solid[::stride, ::stride] > 0, np.nan, v[::stride, ::stride]),
                    color="black", alpha=0.65, angles="xy", scale_units="xy",
                    scale=qscale, width=0.002,
                )
            fig.colorbar(im, ax=ax, label=label)
            try:
                ax.contour(x, y, solid, levels=[0.5], colors="white", linewidths=1.1)
            except Exception:
                pass
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"{case_id} | {label} zoom perfil")
            ax.grid(True, alpha=0.15)
            fig.tight_layout()
            fig.savefig(paths[key], dpi=220, bbox_inches="tight")
            plt.close(fig)
    except Exception as exc:
        print(f"[field-debug] fallo zoom para {case_id}: {exc}")

    return {
        key: str(path.relative_to(ROOT_DIR))
        for key, path in paths.items()
        if path.exists()
    }


def run_case(case: Case, *, field_debug: bool) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    cfg = dict(case.cfg)
    alpha = float(cfg["alpha_deg"])
    rho = float(cfg["rho"])
    nu = float(cfg["nu"])
    chord = float(cfg["chord"])
    u_inf = math.hypot(float(cfg["v0x"]), float(cfg["v0y"]))
    reynolds = u_inf * chord / max(nu, 1e-30)

    t0 = time.time()
    mesh = sim_main(**cfg)
    elapsed = time.time() - t0

    mu = rho * nu
    q_dyn = 0.5 * rho * u_inf * u_inf * chord
    cl_mean, cl_final, cl_std = summarize_vector(mesh.clvector, case.discard_frac)
    cd_mean, cd_final, cd_std = summarize_vector(mesh.cdvector, case.discard_frac)
    forces = mesh.compute_drag_lift(mu, rho=rho, n_extrap_layers=5)

    div_mean, div_final, _ = summarize_vector(mesh.divvector, case.discard_frac)
    div_flux_mean, div_flux_final, _ = summarize_vector(getattr(mesh, "divvector_flux", None), case.discard_frac)
    wall_leak_max, wall_leak_max_final, _ = summarize_vector(getattr(mesh, "wall_leak_max_vector", None), case.discard_frac)
    nu_t_mean, nu_t_max = extract_nut_stats(mesh, nu)

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
    except Exception as exc:
        bl = {"warn": [f"BL failed: {exc}"]}

    x_les, cpu_les, cpl_les = mesh.get_cp_profile_mean()
    cl_cp = finite_or_nan(cl_from_delta_cp(x_les, cpu_les, cpl_les, math.radians(alpha))) if len(x_les) else float("nan")
    cl_inv = finite_or_nan(bl.get("Cl_inviscid")) if bl else float("nan")
    force_audit_rows: list[dict[str, Any]] = []
    force_audit_summary: dict[str, Any] = {}
    try:
        audit = mesh.extract_surface_force_audit(mu=mu, rho=rho, chord=chord, n_extrap_layers=5)
        force_audit_rows = list(audit.get("rows", []))
        for audit_row in force_audit_rows:
            audit_row["case_id"] = case.case_id
            audit_row["alpha_deg"] = alpha
        force_audit_summary = dict(audit.get("summary", {}))
    except Exception as exc:
        force_audit_summary = {"surface_force_audit_warn": str(exc)}

    row: dict[str, Any] = {
        "case_id": case.case_id,
        "group": case.group,
        "status": "ok",
        "description": case.description,
        "alpha_deg": alpha,
        "Re": reynolds,
        "Lx": float(cfg["Lx"]),
        "Ly": float(cfg["Ly"]),
        "dx_min": float(cfg["dx_min"]),
        "factor_expansion": float(cfg["factor_expansion"]),
        "iteraciones": int(cfg["iteraciones"]),
        "guardado": int(cfg["guardado"]),
        "usar_wale": bool(cfg["usar_wale"]),
        "wale_Cw": finite_or_nan(cfg.get("wale_Cw")),
        "projection_variant": str(cfg.get("projection_variant", "legacy_centered")),
        "mg_pressure_accumulation": str(cfg.get("mg_pressure_accumulation", "outer_sum")),
        "wall_pressure_gradient_mode": str(cfg.get("wall_pressure_gradient_mode", "masked")),
        "ibm_wall_mode": str(cfg.get("ibm_wall_mode", "ghost_noslip")),
        "ibm_sdf_smooth_passes": cfg.get("ibm_sdf_smooth_passes", ""),
        "boundary_top": str(cfg.get("boundary_top", ("slip", None))),
        "boundary_bottom": str(cfg.get("boundary_bottom", ("slip", None))),
        "boundary_right": str(cfg.get("boundary_right", ("outflow", 0.0))),
        "nx": int(mesh.nx),
        "ny": int(mesh.ny),
        "n_cells": int(mesh.nx * mesh.ny),
        "Cl_mean": cl_mean,
        "Cl_final": cl_final,
        "Cl_std": cl_std,
        "Cl_p": finite_or_nan(forces["Lift_p"] / q_dyn),
        "Cl_v": finite_or_nan(forces["Lift_v"] / q_dyn),
        "Cl_from_Cp_LES": cl_cp,
        "Cl_from_Cp_force_consistent": finite_or_nan(
            force_audit_summary.get("Cl_from_Cp_force_consistent")
        ),
        "Cl_pressure_force": finite_or_nan(forces["Lift_p"] / q_dyn),
        "Cl_inviscid": cl_inv,
        "Cl_final_over_inviscid": finite_or_nan(cl_final / cl_inv) if abs(cl_inv) > 1e-12 else float("nan"),
        "Cl_from_Cp_LES_over_inviscid": finite_or_nan(cl_cp / cl_inv) if abs(cl_inv) > 1e-12 else float("nan"),
        "Cl_final_minus_Cp_LES": finite_or_nan(cl_final - cl_cp),
        "Cl_final_minus_Cp_force_consistent": finite_or_nan(
            cl_final - finite_or_nan(force_audit_summary.get("Cl_from_Cp_force_consistent"))
        ),
        "Cp_force_consistent_range": finite_or_nan(force_audit_summary.get("Cp_force_consistent_range")),
        "pressure_debias_model": force_audit_summary.get("pressure_debias_model", ""),
        "surface_force_audit_warn": force_audit_summary.get("surface_force_audit_warn", ""),
        "Cd_mean": cd_mean,
        "Cd_final": cd_final,
        "Cd_std": cd_std,
        "Cd_p": finite_or_nan(forces["Drag_p"] / q_dyn),
        "Cd_v": finite_or_nan(forces["Drag_v"] / q_dyn),
        "div_mean": div_mean,
        "div_final": div_final,
        "div_flux_mean": div_flux_mean,
        "div_flux_final": div_flux_final,
        "wall_leak_max": wall_leak_max,
        "wall_leak_max_final": wall_leak_max_final,
        "nu_t_over_nu_mean": nu_t_mean,
        "nu_t_over_nu_max": nu_t_max,
        "trans_x_upper": finite_or_nan((bl or {}).get("trans_x_upper")),
        "trans_x_lower": finite_or_nan((bl or {}).get("trans_x_lower")),
        "x_sep_upper": finite_or_nan((bl or {}).get("x_sep_upper")),
        "x_sep_lower": finite_or_nan((bl or {}).get("x_sep_lower")),
        "bl_separated": bool((bl or {}).get("separated", False)),
        "bl_warn": "; ".join(str(w) for w in (bl or {}).get("warn", [])),
        "elapsed_s": elapsed,
        "its_per_s": float(cfg["iteraciones"] / elapsed) if elapsed > 0 else 0.0,
    }
    row.update(extract_surface_balance(mesh, cfg))
    row["cp_csv"] = save_cp_csv(case.case_id, x_les, cpu_les, cpl_les, bl)

    x_inv = np.asarray((bl or {}).get("x_norm_panel", []), dtype=float)
    cpu_inv = np.asarray((bl or {}).get("Cp_upper_inv", []), dtype=float)
    cpl_inv = np.asarray((bl or {}).get("Cp_lower_inv", []), dtype=float)
    region_rows = cp_region_rows(case.case_id, alpha, "LES_mean", x_les, cpu_les, cpl_les, x_inv, cpu_inv, cpl_inv)
    peak_rows = [cp_peak_row(case.case_id, alpha, "LES_mean", x_les, cpu_les, cpl_les, x_inv, cpu_inv, cpl_inv)]

    pressure_rows = []
    try:
        for pvar in pressure_reading_profiles(mesh, rho):
            source = f"{pvar['pressure_variant']}__ref_{pvar['pressure_ref_mode']}"
            x = pvar["x"]
            cpu = pvar["Cp_upper"]
            cpl = pvar["Cp_lower"]
            cl_v = finite_or_nan(cl_from_delta_cp(x, cpu, cpl, math.radians(alpha)))
            pressure_rows.append({
                "case_id": case.case_id,
                "alpha_deg": alpha,
                "pressure_variant": pvar["pressure_variant"],
                "pressure_ref_mode": pvar["pressure_ref_mode"],
                "Cl_from_Cp": cl_v,
                "Cl_from_Cp_over_inviscid": finite_or_nan(cl_v / cl_inv) if abs(cl_inv) > 1e-12 else float("nan"),
            })
            region_rows.extend(cp_region_rows(case.case_id, alpha, source, x, cpu, cpl, x_inv, cpu_inv, cpl_inv))
            peak_rows.append(cp_peak_row(case.case_id, alpha, source, x, cpu, cpl, x_inv, cpu_inv, cpl_inv))
    except Exception as exc:
        row["pressure_variant_warn"] = str(exc)

    le_rows: list[dict[str, Any]] = []
    try:
        le_rows = mesh.extract_le_ibm_diagnostics(mu=mu, rho=rho, x_over_c_max=0.15, n_layers=5)
        for le_row in le_rows:
            le_row["case_id"] = case.case_id
            le_row["alpha_deg"] = alpha
            le_row["ibm_wall_mode"] = str(cfg.get("ibm_wall_mode", "ghost_noslip"))
            le_row["ibm_sdf_smooth_passes"] = cfg.get("ibm_sdf_smooth_passes", "")
        le_summary = summarize_le_rows(case.case_id, le_rows)
        row.update(le_summary)
    except Exception as exc:
        le_summary = summarize_le_rows(case.case_id, [])
        row.update(le_summary)
        row["le_ibm_warn"] = str(exc)

    if field_debug:
        row.update(save_field_figures(case.case_id, mesh, cfg))

    del mesh
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return row, region_rows, peak_rows, pressure_rows, force_audit_rows, le_rows, le_summary


def error_row(case: Case, exc: BaseException) -> dict[str, Any]:
    cfg = case.cfg
    return {
        "case_id": case.case_id,
        "group": case.group,
        "status": "error",
        "description": case.description,
        "alpha_deg": cfg.get("alpha_deg"),
        "Lx": cfg.get("Lx"),
        "Ly": cfg.get("Ly"),
        "dx_min": cfg.get("dx_min"),
        "usar_wale": cfg.get("usar_wale"),
        "wale_Cw": cfg.get("wale_Cw"),
        "projection_variant": cfg.get("projection_variant", "legacy_centered"),
        "mg_pressure_accumulation": cfg.get("mg_pressure_accumulation", "outer_sum"),
        "wall_pressure_gradient_mode": cfg.get("wall_pressure_gradient_mode", "masked"),
        "ibm_wall_mode": cfg.get("ibm_wall_mode", "ghost_noslip"),
        "ibm_sdf_smooth_passes": cfg.get("ibm_sdf_smooth_passes", ""),
        "Cl_from_Cp_force_consistent": float("nan"),
        "Cl_final_minus_Cp_force_consistent": float("nan"),
        "Cp_force_consistent_range": float("nan"),
        "pressure_debias_model": "",
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_cp_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    def col(name: str) -> np.ndarray:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[name]))
            except Exception:
                vals.append(float("nan"))
        arr = np.asarray(vals, dtype=float)
        return arr[np.isfinite(arr)]
    return col("x_les"), col("Cp_upper_LES"), col("Cp_lower_LES"), col("x_inv"), col("Cp_upper_inv"), col("Cp_lower_inv")


def _case_row(done: dict[str, dict[str, Any]], case_id: str) -> dict[str, Any] | None:
    row = done.get(case_id)
    return row if row and row.get("status") == "ok" else None


def _baseline_case_id(done: dict[str, dict[str, Any]], suite: str | None = None) -> str:
    if suite in {"le_ibm", "projection_fix"}:
        for cid in ("rot_a10_ghost_noslip_baseline", "rot_a10_baseline_current", "rot_a10_baseline"):
            if _case_row(done, cid) is not None:
                return cid
        return "rot_a10_ghost_noslip_baseline"
    return "rot_a10_baseline"


def build_sensitivity_rows(done: dict[str, dict[str, Any]],
                           region_rows: list[dict[str, Any]],
                           pressure_rows: list[dict[str, Any]],
                           suite: str | None = None) -> list[dict[str, Any]]:
    base_id = _baseline_case_id(done, suite)
    base = _case_row(done, base_id)
    if base is None:
        return []
    base_cl = finite_or_nan(base.get("Cl_from_Cp_LES"))
    base_cl_force = finite_or_nan(base.get("Cl_from_Cp_force_consistent"))
    out = []
    for row in done.values():
        if row.get("status") != "ok" or finite_or_nan(row.get("alpha_deg")) != 10.0:
            continue
        cl = finite_or_nan(row.get("Cl_from_Cp_LES"))
        cl_force = finite_or_nan(row.get("Cl_from_Cp_force_consistent"))
        out.append({
            "case_id": row["case_id"],
            "group": row.get("group", ""),
            "Cl_from_Cp_LES": cl,
            "Cl_from_Cp_force_consistent": cl_force,
            "Cl_final": finite_or_nan(row.get("Cl_final")),
            "Cl_final_minus_Cp_force_consistent": finite_or_nan(
                row.get("Cl_final_minus_Cp_force_consistent")
            ),
            "Cl_inviscid": finite_or_nan(row.get("Cl_inviscid")),
            "baseline_case_id": base_id,
            "dCl_from_baseline": finite_or_nan(cl - base_cl),
            "dCl_force_from_baseline": finite_or_nan(cl_force - base_cl_force),
            "abs_dCl_from_baseline": finite_or_nan(abs(cl - base_cl)),
            "abs_dCl_force_from_baseline": finite_or_nan(abs(cl_force - base_cl_force)),
            "Lx": finite_or_nan(row.get("Lx")),
            "Ly": finite_or_nan(row.get("Ly")),
            "dx_min": finite_or_nan(row.get("dx_min")),
            "usar_wale": row.get("usar_wale"),
            "wale_Cw": finite_or_nan(row.get("wale_Cw")),
            "Cp_force_consistent_range": finite_or_nan(row.get("Cp_force_consistent_range")),
            "pressure_debias_model": row.get("pressure_debias_model", ""),
        })

    # Add compact pressure-reading range per case.
    by_case: dict[str, list[float]] = {}
    for pr in pressure_rows:
        by_case.setdefault(str(pr["case_id"]), []).append(finite_or_nan(pr.get("Cl_from_Cp")))
    for r in out:
        vals = [v for v in by_case.get(str(r["case_id"]), []) if math.isfinite(v)]
        r["pressure_reading_Cl_range"] = max(vals) - min(vals) if vals else float("nan")

    # Add leading-edge ratio for the mean LES source.
    le_ratio: dict[str, float] = {}
    for rr in region_rows:
        if rr.get("source") == "LES_mean" and float(rr.get("x_start", -1)) == 0.0 and float(rr.get("x_end", -1)) == 0.2:
            le_ratio[str(rr["case_id"])] = finite_or_nan(rr.get("dCp_integral_over_inv"))
    # The direct 0-0.2 row is not in REGIONS, synthesize below from two ranges.
    if not le_ratio:
        partial: dict[str, dict[str, float]] = {}
        for rr in region_rows:
            if rr.get("source") != "LES_mean":
                continue
            if (float(rr.get("x_start", -1)), float(rr.get("x_end", -1))) in {
                (0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20)
            }:
                d = partial.setdefault(str(rr["case_id"]), {"les": 0.0, "inv": 0.0})
                d["les"] += finite_or_nan(rr.get("dCp_integral"))
                d["inv"] += finite_or_nan(rr.get("dCp_integral_inv"))
        for cid, vals in partial.items():
            le_ratio[cid] = vals["les"] / vals["inv"] if abs(vals["inv"]) > 1e-12 else float("nan")
    for r in out:
        r["leading_edge_0_0p2_dCp_ratio"] = le_ratio.get(str(r["case_id"]), float("nan"))
    return out


def classify_case(row: dict[str, Any], sensitivity: dict[str, dict[str, Any]]) -> str:
    if row.get("status") != "ok":
        return "case_failed"
    cl_final = finite_or_nan(row.get("Cl_final"))
    cl_cp_les = finite_or_nan(row.get("Cl_from_Cp_LES"))
    cl_cp_force = finite_or_nan(row.get("Cl_from_Cp_force_consistent"))
    cl_cp = cl_cp_force if math.isfinite(cl_cp_force) else cl_cp_les
    cl_inv = finite_or_nan(row.get("Cl_inviscid"))
    sens = sensitivity.get(str(row.get("case_id")), {})
    le_ratio = finite_or_nan(sens.get("leading_edge_0_0p2_dCp_ratio"))
    p_range = finite_or_nan(sens.get("pressure_reading_Cl_range"))
    if math.isfinite(cl_final) and math.isfinite(cl_cp) and abs(cl_final - cl_cp) > 0.12:
        return "force_integration_mismatch"
    if math.isfinite(p_range) and p_range > 0.10:
        return "pressure_reading_artifact"
    if math.isfinite(le_ratio) and le_ratio < 0.50:
        return "pressure_leading_edge_deficit"
    if math.isfinite(cl_cp) and math.isfinite(cl_inv) and abs(cl_inv) > 1e-12 and cl_cp / cl_inv < 0.65:
        return "pressure_field_low_lift"
    return "pressure_lift_consistent"


def write_analysis_report(done: dict[str, dict[str, Any]],
                          sensitivity_rows: list[dict[str, Any]],
                          suite: str | None = None) -> None:
    sensitivity = {str(r["case_id"]): r for r in sensitivity_rows}
    for row in done.values():
        row["pressure_probable_cause"] = classify_case(row, sensitivity)

    base_id = _baseline_case_id(done, suite)
    base = _case_row(done, base_id)
    lines = [
        "# Diagnostico presion/lift NACA0012",
        "",
        "## Resumen ejecutivo",
    ]
    if base:
        lines.append(
            f"- Baseline `{base_id}` alpha=10: Cl_final={finite_or_nan(base.get('Cl_final')):+.5f}, "
            f"Cl_from_Cp_LES={finite_or_nan(base.get('Cl_from_Cp_LES')):+.5f}, "
            f"Cl_Cp_force={finite_or_nan(base.get('Cl_from_Cp_force_consistent')):+.5f}, "
            f"Cl_inviscid={finite_or_nan(base.get('Cl_inviscid')):+.5f}."
        )
        b_sens = sensitivity.get(base_id, {})
        lines.append(
            f"- Ratio dCp LE 0-0.2c={finite_or_nan(b_sens.get('leading_edge_0_0p2_dCp_ratio')):.3f}; "
            f"rango lectura p_wall={finite_or_nan(b_sens.get('pressure_reading_Cl_range')):.5f}."
        )
        lines.append(f"- Causa probable baseline: `{base.get('pressure_probable_cause', '')}`.")
    else:
        lines.append("- No hay baseline alpha=10 OK; revisar errores en summary.csv.")

    lines.extend(["", "## Sensibilidades alpha=10", ""])
    if sensitivity_rows:
        lines.append("| case_id | grupo | Cl_Cp_force | dCl_force | Cl_Cp_LES | dCl_LES | LE_ratio | pwall_range |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(sensitivity_rows, key=lambda x: (str(x.get("group")), str(x.get("case_id")))):
            lines.append(
                f"| {r['case_id']} | {r.get('group', '')} | "
                f"{finite_or_nan(r.get('Cl_from_Cp_force_consistent')):+.5f} | "
                f"{finite_or_nan(r.get('dCl_force_from_baseline')):+.5f} | "
                f"{finite_or_nan(r.get('Cl_from_Cp_LES')):+.5f} | "
                f"{finite_or_nan(r.get('dCl_from_baseline')):+.5f} | "
                f"{finite_or_nan(r.get('leading_edge_0_0p2_dCp_ratio')):.3f} | "
                f"{finite_or_nan(r.get('pressure_reading_Cl_range')):.5f} |"
            )

    lines.extend(["", "## Clasificacion por caso", ""])
    for row in sorted(done.values(), key=lambda r: str(r.get("case_id"))):
        lines.append(
            f"- `{row.get('case_id')}`: `{row.get('pressure_probable_cause', classify_case(row, sensitivity))}`"
        )

    le_rows = read_rows(OUT_LE_COMPARISON)
    if le_rows:
        lines.extend(["", "## Diagnostico IBM borde de ataque", ""])
        lines.append("| case_id | ibm | Cl_Cp | dCl | LE Cp min | |Un|/U max | direct_zero | verdict |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for r in le_rows:
            lines.append(
                f"| {r.get('case_id', '')} | {r.get('ibm_wall_mode', '')} | "
                f"{finite_or_nan(r.get('Cl_from_Cp_LES')):+.5f} | "
                f"{finite_or_nan(r.get('dCl_from_le_baseline')):+.5f} | "
                f"{finite_or_nan(r.get('le_Cp_min')):+.4f} | "
                f"{finite_or_nan(r.get('le_abs_un_over_U_max')):.4f} | "
                f"{finite_or_nan(r.get('le_ghost_direct_zero_ratio')):.4f} | "
                f"`{r.get('le_verdict', '')}` |"
            )

    lines.extend([
        "",
        "## Archivos generados",
        f"- `{OUT_CSV.relative_to(ROOT_DIR)}`",
        f"- `{OUT_CP_REGIONS.relative_to(ROOT_DIR)}`",
        f"- `{OUT_CP_PEAKS.relative_to(ROOT_DIR)}`",
        f"- `{OUT_PRESSURE_VARIANTS.relative_to(ROOT_DIR)}`",
        f"- `{OUT_FORCE_AUDIT.relative_to(ROOT_DIR)}`",
        f"- `{OUT_LE_LOCAL.relative_to(ROOT_DIR)}`",
        f"- `{OUT_LE_SUMMARY.relative_to(ROOT_DIR)}`",
        f"- `{OUT_LE_COMPARISON.relative_to(ROOT_DIR)}`",
        f"- `{OUT_SENSITIVITY.relative_to(ROOT_DIR)}`",
    ])
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_figures(done: dict[str, dict[str, Any]],
                     region_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[figures] matplotlib no disponible: {exc}")
        return

    for row in done.values():
        if row.get("status") != "ok" or not row.get("cp_csv"):
            continue
        cp_path = ROOT_DIR / str(row["cp_csv"])
        if not cp_path.exists():
            continue
        x, cpu, cpl, xi, cpui, cpli = load_cp_csv(cp_path)
        fig, ax = plt.subplots(figsize=(9, 5))
        if len(x):
            ax.plot(x, cpu, label="Cp upper LES", linewidth=1.6)
            ax.plot(x, cpl, label="Cp lower LES", linewidth=1.6)
        if len(xi):
            ax.plot(xi, cpui, "--", label="Cp upper inv", linewidth=1.1)
            ax.plot(xi, cpli, "--", label="Cp lower inv", linewidth=1.1)
        ax.invert_yaxis()
        ax.set_xlabel("x/c")
        ax.set_ylabel("Cp")
        ax.set_title(str(row["case_id"]))
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"cp_{row['case_id']}.png", dpi=180)
        plt.close(fig)

        rr = [
            r for r in region_rows
            if r.get("case_id") == row["case_id"] and r.get("source") == "LES_mean"
        ]
        if rr:
            labels = [f"{r['x_start']:.2f}-{r['x_end']:.2f}" for r in rr]
            vals = [finite_or_nan(r.get("dCp_integral")) for r in rr]
            inv = [finite_or_nan(r.get("dCp_integral_inv")) for r in rr]
            xx = np.arange(len(labels))
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(xx - 0.18, vals, width=0.36, label="LES")
            ax.bar(xx + 0.18, inv, width=0.36, label="inviscid")
            ax.set_xticks(xx, labels, rotation=30, ha="right")
            ax.set_ylabel("Integral dCp")
            ax.set_title(f"dCp por regiones | {row['case_id']}")
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(OUT_DIR / f"dcp_regions_{row['case_id']}.png", dpi=180)
            plt.close(fig)

    alpha10 = [r for r in done.values() if r.get("status") == "ok" and finite_or_nan(r.get("alpha_deg")) == 10.0]
    if alpha10:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [str(r["case_id"]).replace("rot_a10_", "") for r in alpha10]
        vals = [finite_or_nan(r.get("Cl_from_Cp_LES")) for r in alpha10]
        inv = finite_or_nan(alpha10[0].get("Cl_inviscid"))
        ax.bar(np.arange(len(labels)), vals)
        if math.isfinite(inv):
            ax.axhline(inv, color="black", linestyle="--", linewidth=1.0, label="Cl inviscid")
        ax.set_xticks(np.arange(len(labels)), labels, rotation=40, ha="right")
        ax.set_ylabel("Cl from Cp LES")
        ax.set_title("Sensibilidad Cl_from_Cp_LES alpha=10")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "sensitivity_cl_bar.png", dpi=180)
        plt.close(fig)


def validate_results(done_subset: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    ok_rows = [r for r in done_subset.values() if r.get("status") == "ok"]
    error_rows = [r for r in done_subset.values() if r.get("status") == "error"]
    add("no_failed_cases", len(error_rows) == 0, f"{len(error_rows)} casos fallidos")

    missing_cp = [str(r.get("case_id")) for r in ok_rows if not r.get("cp_csv") or not (ROOT_DIR / str(r.get("cp_csv"))).exists()]
    add("cp_files_exist", not missing_cp, ", ".join(missing_cp[:8]) if missing_cp else "Cp CSV generado")

    bad_cl = [str(r.get("case_id")) for r in ok_rows if not math.isfinite(finite_or_nan(r.get("Cl_from_Cp_LES")))]
    add("cl_from_cp_finite", not bad_cl, ", ".join(bad_cl[:8]) if bad_cl else "Cl_from_Cp_LES finito")

    bad_cl_force = [
        str(r.get("case_id")) for r in ok_rows
        if not math.isfinite(finite_or_nan(r.get("Cl_from_Cp_force_consistent")))
    ]
    add("cl_from_cp_force_finite", not bad_cl_force,
        ", ".join(bad_cl_force[:8]) if bad_cl_force else "Cl_from_Cp_force_consistent finito")

    alpha0 = [r for r in ok_rows if abs(finite_or_nan(r.get("alpha_deg"))) < 1e-9]
    if alpha0:
        worst = max(abs(finite_or_nan(r.get("Cl_final"))) for r in alpha0)
        add("alpha0_near_zero", worst < 0.08, f"max |Cl_final alpha0|={worst:.6f}")

    add("analysis_outputs_exist",
        OUT_CP_REGIONS.exists() and OUT_CP_PEAKS.exists() and OUT_PRESSURE_VARIANTS.exists()
        and OUT_FORCE_AUDIT.exists()
        and OUT_LE_SUMMARY.exists() and OUT_LE_COMPARISON.exists() and OUT_REPORT.exists(),
        "CSV de analisis y reporte presentes")

    return {"ok": all(c["ok"] for c in checks), "checks": checks}


def cuda_available() -> tuple[bool, str]:
    try:
        n_devices = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        return False, str(exc)
    if n_devices <= 0:
        return False, "no CUDA-capable device is detected"
    return True, f"{n_devices} CUDA device(s)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostico de presion/lift NACA0012 centrado en rotated_geom."
    )
    parser.add_argument("--suite", choices=["smoke", "full", "le_ibm", "projection_fix"], default="smoke",
                        help="matriz de casos a ejecutar")
    parser.add_argument("--output-suffix",
                        help="sufijo opcional para crear un directorio de salida independiente")
    parser.add_argument("--cases", nargs="*",
                        help="limita la ejecucion a case_id concretos")
    parser.add_argument("--force", action="store_true",
                        help="recalcula casos aunque existan en summary.json")
    parser.add_argument("--list-cases", action="store_true",
                        help="muestra casos y termina sin simular")
    parser.add_argument("--no-field-debug", action="store_true",
                        help="no guarda campos/zooms PNG por caso")
    parser.add_argument("--no-figures", action="store_true",
                        help="no genera figuras agregadas")
    parser.add_argument("--no-fail-on-validation", action="store_true",
                        help="no devuelve exit code 2 si falla una validacion")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_output_dir(args.output_suffix)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIELDS.mkdir(parents=True, exist_ok=True)

    all_cases = build_cases(args.suite)
    cases = select_cases(all_cases, args.cases)

    if args.list_cases:
        for c in cases:
            print(
                f"{c.case_id:<34} group={c.group:<10} "
                f"alpha={float(c.cfg['alpha_deg']):>5.1f} "
                f"Lx={float(c.cfg['Lx']):>4.1f} Ly={float(c.cfg['Ly']):>4.1f} "
                f"dx={float(c.cfg['dx_min']):.4g} "
                f"wale={int(bool(c.cfg['usar_wale']))} "
                f"Cw={float(c.cfg.get('wale_Cw', float('nan'))):.4g} "
                f"mgp={c.cfg.get('mg_pressure_accumulation', 'outer_sum')} "
                f"grad={c.cfg.get('wall_pressure_gradient_mode', 'masked')} "
                f"ibm={c.cfg.get('ibm_wall_mode', 'ghost_noslip')} "
                f"sdf={c.cfg.get('ibm_sdf_smooth_passes', '')} "
                f"top={c.cfg.get('boundary_top', ('slip', None))} "
                f"bottom={c.cfg.get('boundary_bottom', ('slip', None))}"
            )
        return 0

    ok_cuda, cuda_msg = cuda_available()
    if not ok_cuda:
        print(f"[ERROR] No se pueden ejecutar simulaciones: {cuda_msg}")
        print("        Usa --list-cases para inspeccionar la matriz sin GPU.")
        return 3

    done = {} if args.force else load_done()
    all_region_rows: list[dict[str, Any]] = [] if args.force else read_rows(OUT_CP_REGIONS)
    all_peak_rows: list[dict[str, Any]] = [] if args.force else read_rows(OUT_CP_PEAKS)
    all_pressure_rows: list[dict[str, Any]] = [] if args.force else read_rows(OUT_PRESSURE_VARIANTS)
    all_force_audit_rows: list[dict[str, Any]] = [] if args.force else read_rows(OUT_FORCE_AUDIT)
    all_le_rows: list[dict[str, Any]] = [] if args.force else read_rows(OUT_LE_LOCAL)
    all_le_summary_rows: list[dict[str, Any]] = [] if args.force else read_rows(OUT_LE_SUMMARY)

    print("\nDiagnostico presion/lift NACA0012")
    print(f"Salida: {OUT_DIR.relative_to(ROOT_DIR)}")
    print(f"Suite: {args.suite} | casos seleccionados: {len(cases)}")

    for idx, case in enumerate(cases, 1):
        existing = done.get(case.case_id, {})
        has_force_metric = math.isfinite(
            finite_or_nan(existing.get("Cl_from_Cp_force_consistent"))
        )
        has_force_rows = any(str(r.get("case_id")) == str(case.case_id) for r in all_force_audit_rows)
        if not args.force and existing.get("status") == "ok" and has_force_metric and has_force_rows:
            print(f"[{idx}/{len(cases)}] {case.case_id}: ya existe, salto")
            continue
        print(f"[{idx}/{len(cases)}] {case.case_id}: {case.description}", flush=True)
        try:
            row, region_rows, peak_rows, pressure_rows, force_audit_rows, le_rows, le_summary = run_case(
                case,
                field_debug=not bool(args.no_field_debug),
            )
            done[case.case_id] = row
            all_region_rows.extend(region_rows)
            all_peak_rows.extend(peak_rows)
            all_pressure_rows.extend(pressure_rows)
            all_force_audit_rows = [
                r for r in all_force_audit_rows
                if str(r.get("case_id")) != str(case.case_id)
            ]
            all_force_audit_rows.extend(force_audit_rows)
            all_le_rows.extend(le_rows)
            all_le_summary_rows = [
                r for r in all_le_summary_rows
                if str(r.get("case_id")) != str(le_summary.get("case_id"))
            ]
            all_le_summary_rows.append(le_summary)
            print(
                f"  Cl={row['Cl_mean']:+.5f} Cl_Cp={row['Cl_from_Cp_LES']:+.5f} "
                f"Cl_Cp_force={row.get('Cl_from_Cp_force_consistent', float('nan')):+.5f} "
                f"Cl_inv={row['Cl_inviscid']:+.5f} div_flux={row['div_flux_final']:.4g} "
                f"wall={row['wall_leak_max_final']:.4g} "
                f"LE_Cpmin={row.get('le_Cp_min', float('nan')):+.3f} "
                f"({row['elapsed_s']:.1f}s)"
            )
        except Exception as exc:
            done[case.case_id] = error_row(case, exc)
            print(f"  ERROR: {exc}")
            traceback.print_exc()

        save_json(done, all_cases)
        save_csv(done, all_cases)
        save_plan(done, all_cases)

    # Rebuild postprocess rows from current run outputs and existing rows.
    # Existing skipped cases are reconstructed from their Cp CSV where possible.
    existing_region_rows = list(all_region_rows)
    existing_peak_rows = list(all_peak_rows)
    for row in done.values():
        if row.get("status") != "ok" or not row.get("cp_csv"):
            continue
        if any(r.get("case_id") == row.get("case_id") and r.get("source") == "LES_mean" for r in existing_region_rows):
            continue
        cp_path = ROOT_DIR / str(row["cp_csv"])
        if not cp_path.exists():
            continue
        x, cpu, cpl, xi, cpui, cpli = load_cp_csv(cp_path)
        alpha = finite_or_nan(row.get("alpha_deg"))
        existing_region_rows.extend(cp_region_rows(str(row["case_id"]), alpha, "LES_mean", x, cpu, cpl, xi, cpui, cpli))
        existing_peak_rows.append(cp_peak_row(str(row["case_id"]), alpha, "LES_mean", x, cpu, cpl, xi, cpui, cpli))

    le_by_case = {str(r.get("case_id")): r for r in all_le_summary_rows}
    for row in done.values():
        if row.get("status") != "ok":
            continue
        cid = str(row.get("case_id"))
        fallback = le_summary_from_case_row(row)
        if cid not in le_by_case:
            le_by_case[cid] = fallback
        else:
            for key, value in fallback.items():
                if key == "case_id":
                    continue
                current = le_by_case[cid].get(key)
                if current in {"", None} or (
                    not math.isfinite(finite_or_nan(current))
                    and math.isfinite(finite_or_nan(value))
                ):
                    le_by_case[cid][key] = value
    all_le_summary_rows = [le_by_case[cid] for cid in sorted(le_by_case)]

    write_rows(OUT_CP_REGIONS, existing_region_rows)
    write_rows(OUT_CP_PEAKS, existing_peak_rows)
    write_rows(OUT_PRESSURE_VARIANTS, all_pressure_rows)
    write_rows(OUT_FORCE_AUDIT, all_force_audit_rows)
    write_rows(OUT_LE_LOCAL, all_le_rows)
    write_rows(OUT_LE_SUMMARY, all_le_summary_rows)

    sensitivity_rows = build_sensitivity_rows(done, existing_region_rows, all_pressure_rows, args.suite)
    write_rows(OUT_SENSITIVITY, sensitivity_rows)
    le_comparison_rows = build_le_comparison(done, all_le_summary_rows, args.suite)
    write_rows(OUT_LE_COMPARISON, le_comparison_rows)
    write_analysis_report(done, sensitivity_rows, args.suite)
    save_json(done, all_cases)
    save_csv(done, all_cases)

    if not args.no_figures:
        generate_figures(done, existing_region_rows)

    selected_ids = {c.case_id for c in cases}
    validation = validate_results({cid: done[cid] for cid in done if cid in selected_ids})
    with OUT_VALIDATION.open("w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2, ensure_ascii=True)

    print(f"\nResumen: {OUT_CSV.relative_to(ROOT_DIR)}")
    print(f"Regiones Cp: {OUT_CP_REGIONS.relative_to(ROOT_DIR)}")
    print(f"Picos Cp: {OUT_CP_PEAKS.relative_to(ROOT_DIR)}")
    print(f"Variantes lectura presion: {OUT_PRESSURE_VARIANTS.relative_to(ROOT_DIR)}")
    print(f"Auditoria fuerza superficie: {OUT_FORCE_AUDIT.relative_to(ROOT_DIR)}")
    print(f"Dump LE IBM: {OUT_LE_LOCAL.relative_to(ROOT_DIR)}")
    print(f"Resumen LE IBM: {OUT_LE_SUMMARY.relative_to(ROOT_DIR)}")
    print(f"Comparacion LE IBM: {OUT_LE_COMPARISON.relative_to(ROOT_DIR)}")
    print(f"Sensibilidades: {OUT_SENSITIVITY.relative_to(ROOT_DIR)}")
    print(f"Reporte: {OUT_REPORT.relative_to(ROOT_DIR)}")
    print(f"Validacion: {'OK' if validation['ok'] else 'FALLO'}")
    for check in validation["checks"]:
        mark = "OK" if check["ok"] else "FAIL"
        print(f"  [{mark}] {check['name']}: {check['detail']}")

    if not validation["ok"] and not args.no_fail_on_validation:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

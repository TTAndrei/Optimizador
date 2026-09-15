"""Defaults compartidos para scripts de simulacion y comparativa."""
from __future__ import annotations

import math
from typing import Any


PROJECTION_DEFAULTS: dict[str, Any] = {
    "projection_variant": "legacy_centered",
    "mg_pressure_accumulation": "outer_sum",
    "wall_pressure_gradient_mode": "masked",
    "pressure_wall_reconstruction": "linear_5",
    "min_te_height_factor": 2.0,
    "wake_refinement_mode": "base",
}


def finite_or_nan(value: Any) -> float:
    try:
        val = float(value)
    except Exception:
        return float("nan")
    return val if math.isfinite(val) else float("nan")


def add_force_consistent_metrics(row: dict[str, Any], mesh: Any, cfg: dict[str, Any]) -> None:
    """Anade metricas Cp/fuerza usando la misma integral de compute_drag_lift."""
    rho = float(cfg.get("rho", 1.0))
    nu = float(cfg.get("nu", 1e-6))
    chord = float(cfg.get("chord", 1.0))
    mu = rho * nu
    try:
        audit = mesh.extract_surface_force_audit(
            mu=mu,
            rho=rho,
            chord=chord,
            n_extrap_layers=5,
        )
        summary = audit.get("summary", {})
        cl_force = finite_or_nan(summary.get("Cl_from_Cp_force_consistent"))
        cd_force = finite_or_nan(summary.get("Cd_from_Cp_force_consistent"))
        row["Cl_from_Cp_force_consistent"] = cl_force
        row["Cd_from_Cp_force_consistent"] = cd_force
        row["Cl_final_minus_Cp_force_consistent"] = finite_or_nan(
            finite_or_nan(row.get("Cl_final")) - cl_force
        )
        row["Cd_final_minus_Cp_force_consistent"] = finite_or_nan(
            finite_or_nan(row.get("Cd_final")) - cd_force
        )
        row["Cp_force_consistent_range"] = finite_or_nan(
            summary.get("Cp_force_consistent_range")
        )
        row["pressure_debias_model"] = summary.get("pressure_debias_model", "")
        try:
            raw = mesh.extract_surface_force_audit(
                mu=mu,
                rho=rho,
                chord=chord,
                n_extrap_layers=5,
                pressure_debias_mode="none",
            ).get("summary", {})
            row["Cd_p_raw"] = finite_or_nan(raw.get("Cd_from_Cp_force_consistent"))
            row["Cd_p_debiased"] = cd_force
        except Exception as raw_exc:
            row["Cd_p_raw"] = float("nan")
            row["Cd_p_debiased"] = cd_force
            row["surface_force_raw_audit_warn"] = str(raw_exc)
    except Exception as exc:
        row["Cl_from_Cp_force_consistent"] = float("nan")
        row["Cd_from_Cp_force_consistent"] = float("nan")
        row["Cl_final_minus_Cp_force_consistent"] = float("nan")
        row["Cd_final_minus_Cp_force_consistent"] = float("nan")
        row["Cd_p_raw"] = float("nan")
        row["Cd_p_debiased"] = float("nan")
        row["Cp_force_consistent_range"] = float("nan")
        row["pressure_debias_model"] = ""
        row["surface_force_audit_warn"] = str(exc)


def preferred_cl_column(columns: set[str]) -> str:
    for name in ("Cl_from_Cp_force_consistent", "Cl_final", "Cl_mean"):
        if name in columns:
            return name
    return "Cl"


def preferred_cd_column(columns: set[str], prefer_bl: bool = False) -> str:
    order = (
        ("Cd_bl_geom", "Cd_bl", "Cd_final", "Cd_mean")
        if prefer_bl
        else ("Cd_final", "Cd_mean")
    )
    for name in order:
        if name in columns:
            return name
    return "Cd"

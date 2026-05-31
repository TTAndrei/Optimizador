"""
Calibracion de factores Cl/Cd vs alpha usando turbo dx=0.001 + XFoil.
Responde: factor constante o varia con alpha?
Si varia, ajusta polinomio f(alpha).

Salida:
  results/calibracion_factores_Cl.png
  results/calibracion_factores_Cd.png
  results/calibracion_factores_combinado.png
  consola: coefs polinomio + CV
"""
from __future__ import annotations
import csv
import json
import os
import sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np

from sim_defaults import preferred_cl_column, preferred_cd_column

BARRIDO = ROOT / "results" / "barridos" / "barrido_modos_outer_sum" / "summary.json"
XFOIL   = ROOT / "data" / "ComparativasReales" / "NACA0012_100k_Xfoil.csv"
OUT_DIR = ROOT / "results" / "calibracion"

MODO_EVAL = "turbo"
DX_EVAL   = 0.001


def load_xfoil() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    with open(XFOIL, encoding="utf-8") as f:
        in_data = False
        for line in f:
            line = line.strip()
            if line.startswith("Alpha,"):
                in_data = True
                continue
            if not in_data or not line:
                continue
            parts = line.split(",")
            try:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    rows.sort(key=lambda r: r[0])
    alphas = np.array([r[0] for r in rows])
    cls    = np.array([r[1] for r in rows])
    cds    = np.array([r[2] for r in rows])
    return alphas, cls, cds


def load_sim(modo: str, dx: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(BARRIDO, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("results", [])
    rows = [r for r in data
            if r.get("modo") == modo
            and r.get("dx_min") == dx
            and "error" not in r]
    rows.sort(key=lambda r: r["alpha"])
    if not rows:
        raise ValueError(f"No hay filas para modo={modo} dx={dx} en {BARRIDO}")
    cl_key = preferred_cl_column(set(rows[0].keys()))
    cd_key = preferred_cd_column(set(rows[0].keys()), prefer_bl=True)
    alphas = np.array([r["alpha"]   for r in rows])
    cls    = np.array([r[cl_key] for r in rows], dtype=float)
    cds    = np.array([r[cd_key] for r in rows], dtype=float)
    print(f"  Simulado: Cl={cl_key} Cd={cd_key}")
    return alphas, cls, cds


def interp_xfoil(xf_a, xf_v, sim_a):
    return np.interp(sim_a, xf_a, xf_v)


def poly_fit(alphas, factors, deg):
    mask = np.isfinite(factors)
    if mask.sum() < 3:
        return None, None
    coefs = np.polyfit(alphas[mask], factors[mask], deg)
    fitted = np.polyval(coefs, alphas)
    return coefs, fitted


def fig_factor(alphas_sim, factors, label, fitted, coefs, deg, fname):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Factor calibracion {label}(alpha)  "
                 f"[{MODO_EVAL} dx={DX_EVAL}  vs XFoil]", fontsize=12)

    ax = axes[0]
    mask = np.isfinite(factors)
    ax.scatter(alphas_sim[mask], factors[mask], color="#2196F3",
               s=70, zorder=3, label="factor puntual")
    if fitted is not None:
        ax.plot(alphas_sim, fitted, color="#FF5722", linewidth=2,
                label=f"poly deg={deg}")
    mean_f = float(np.nanmean(factors))
    ax.axhline(mean_f, color="k", linestyle="--", linewidth=1,
               label=f"media={mean_f:.3f}")
    ax.set_xlabel("alpha [deg]"); ax.set_ylabel(f"{label}_xfoil / {label}_sim")
    ax.set_title("Factor puntual")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    ax2 = axes[1]
    cv = float(np.nanstd(factors) / np.nanmean(factors))
    residuals = factors[mask] - mean_f
    ax2.bar(alphas_sim[mask], residuals, color="#4CAF50", width=0.6)
    ax2.axhline(0, color="k", linewidth=0.8)
    ax2.set_xlabel("alpha [deg]"); ax2.set_ylabel("factor - media")
    ax2.set_title(f"Residuos vs media  (CV={cv:.3f}  "
                  f"{'CONSTANTE' if cv < 0.10 else 'VARIABLE'})")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / fname
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[save] {out.name}")


def fig_combinado(alphas_sim, factors_cl, factors_cd,
                  fit_cl, fit_cd, cl_xf, cd_xf, cl_sim, cd_sim):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(f"Calibracion completa — {MODO_EVAL} dx={DX_EVAL}  vs XFoil",
                 fontsize=13)

    # Row 0: curvas brutas
    ax = axes[0, 0]
    ax.plot(alphas_sim, cl_xf, "k--", label="XFoil", linewidth=2)
    ax.plot(alphas_sim, cl_sim, color="#2196F3", marker="o", label="Sim", linewidth=1.5)
    if fit_cl is not None:
        ax.plot(alphas_sim, cl_sim * fit_cl, color="#FF5722",
                marker="s", linestyle=":", label="Sim calibrado", linewidth=1.5)
    ax.set_title("Cl vs alpha"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(alphas_sim, cd_xf, "k--", label="XFoil", linewidth=2)
    ax.plot(alphas_sim, cd_sim, color="#2196F3", marker="o", label="Sim", linewidth=1.5)
    if fit_cd is not None:
        ax.plot(alphas_sim, cd_sim * fit_cd, color="#FF5722",
                marker="s", linestyle=":", label="Sim calibrado", linewidth=1.5)
    ax.set_title("Cd vs alpha"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ef_xf  = cl_xf / cd_xf
    ef_sim = cl_sim / cd_sim
    ax = axes[0, 2]
    ax.plot(alphas_sim, ef_xf, "k--", label="XFoil", linewidth=2)
    ax.plot(alphas_sim, ef_sim, color="#2196F3", marker="o", label="Sim", linewidth=1.5)
    if fit_cl is not None and fit_cd is not None:
        ef_cal = (cl_sim * fit_cl) / (cd_sim * fit_cd)
        ax.plot(alphas_sim, ef_cal, color="#FF5722",
                marker="s", linestyle=":", label="Sim calibrado", linewidth=1.5)
    ax.set_title("Ef (Cl/Cd) vs alpha"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Row 1: factores
    ax = axes[1, 0]
    mask = np.isfinite(factors_cl)
    ax.scatter(alphas_sim[mask], factors_cl[mask], color="#2196F3", s=60, label="f_Cl puntual")
    if fit_cl is not None:
        ax.plot(alphas_sim, fit_cl, color="#FF5722", linewidth=2, label="fit")
    ax.axhline(float(np.nanmean(factors_cl)), color="k", linestyle="--",
               label=f"media={np.nanmean(factors_cl):.3f}")
    ax.set_title("Factor Cl"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    mask2 = np.isfinite(factors_cd)
    ax.scatter(alphas_sim[mask2], factors_cd[mask2], color="#FF5722", s=60, label="f_Cd puntual")
    if fit_cd is not None:
        ax.plot(alphas_sim, fit_cd, color="#2196F3", linewidth=2, label="fit")
    ax.axhline(float(np.nanmean(factors_cd)), color="k", linestyle="--",
               label=f"media={np.nanmean(factors_cd):.3f}")
    ax.set_title("Factor Cd"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Error calibrado vs sin calibrar
    ax = axes[1, 2]
    err_cl_raw = np.abs(cl_sim - cl_xf)
    err_cd_raw = np.abs(cd_sim - cd_xf)
    if fit_cl is not None and fit_cd is not None:
        err_cl_cal = np.abs(cl_sim * fit_cl - cl_xf)
        err_cd_cal = np.abs(cd_sim * fit_cd - cd_xf)
        x = np.arange(len(alphas_sim))
        w = 0.4
        ax.bar(x - w/2, err_cl_raw, w, color="#2196F3", alpha=0.6, label="|dCl| sin cal")
        ax.bar(x + w/2, err_cl_cal, w, color="#FF5722", alpha=0.6, label="|dCl| calibrado")
        ax.set_xticks(x[::3]); ax.set_xticklabels(alphas_sim[::3].astype(int))
        ax.legend(fontsize=8)
    else:
        ax.bar(range(len(alphas_sim)), err_cl_raw, color="#2196F3", label="|dCl|")
        ax.legend(fontsize=8)
    ax.set_title("Error Cl antes/despues calibracion")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "calibracion_factores_combinado.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[save] {out.name}")


def print_resumen(alphas_sim, factors_cl, factors_cd, coefs_cl, coefs_cd, deg):
    print(f"\n  Modo: {MODO_EVAL}  dx={DX_EVAL}")
    print(f"  {'alpha':>6}  {'f_Cl':>8}  {'f_Cd':>8}")
    print("  " + "-" * 28)
    for a, fcl, fcd in zip(alphas_sim, factors_cl, factors_cd):
        print(f"  {a:>6.1f}  {fcl:>8.3f}  {fcd:>8.3f}")

    cv_cl = np.nanstd(factors_cl) / np.nanmean(factors_cl)
    cv_cd = np.nanstd(factors_cd) / np.nanmean(factors_cd)
    print(f"\n  f_Cl: mean={np.nanmean(factors_cl):.3f}  CV={cv_cl:.3f}  "
          f"-> {'CONSTANTE' if cv_cl < 0.10 else 'VARIABLE'}")
    print(f"  f_Cd: mean={np.nanmean(factors_cd):.3f}  CV={cv_cd:.3f}  "
          f"-> {'CONSTANTE' if cv_cd < 0.10 else 'VARIABLE'}")

    if coefs_cl is not None:
        print(f"\n  Poly Cl deg={deg}: {np.array2string(coefs_cl, precision=4)}")
        print(f"  -> Cl_cal = Cl_sim * polyval(coefs_cl, alpha)")
    if coefs_cd is not None:
        print(f"  Poly Cd deg={deg}: {np.array2string(coefs_cd, precision=4)}")
        print(f"  -> Cd_cal = Cd_sim * polyval(coefs_cd, alpha)")

    print(f"\n  Factor GLOBAL recomendado: Cl x{np.nanmean(factors_cl):.3f}  "
          f"Cd x{np.nanmean(factors_cd):.3f}")


def main() -> None:
    xf_a, xf_cl, xf_cd = load_xfoil()
    sim_a, sim_cl, sim_cd = load_sim(MODO_EVAL, DX_EVAL)

    cl_xf_interp = interp_xfoil(xf_a, xf_cl, sim_a)
    cd_xf_interp = interp_xfoil(xf_a, xf_cd, sim_a)

    with np.errstate(divide="ignore", invalid="ignore"):
        factors_cl = np.where(np.abs(sim_cl) > 1e-6,
                              cl_xf_interp / sim_cl, np.nan)
        factors_cd = np.where(sim_cd > 1e-6,
                              cd_xf_interp / sim_cd, np.nan)

    deg = 2
    coefs_cl, fitted_cl = poly_fit(sim_a, factors_cl, deg)
    coefs_cd, fitted_cd = poly_fit(sim_a, factors_cd, deg)

    fig_factor(sim_a, factors_cl, "Cl", fitted_cl, coefs_cl, deg,
               "calibracion_factores_Cl.png")
    fig_factor(sim_a, factors_cd, "Cd", fitted_cd, coefs_cd, deg,
               "calibracion_factores_Cd.png")
    fig_combinado(sim_a, factors_cl, factors_cd,
                  fitted_cl, fitted_cd,
                  cl_xf_interp, cd_xf_interp, sim_cl, sim_cd)

    print_resumen(sim_a, factors_cl, factors_cd, coefs_cl, coefs_cd, deg)


if __name__ == "__main__":
    main()

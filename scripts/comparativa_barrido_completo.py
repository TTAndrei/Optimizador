"""
Orquestador: ejecuta script_barrido_alpha.py 6 veces (3 perfiles x 2 Reynolds),
compara resultados con datos Xfoil de ComparativasReales/ y genera graficas + estadisticas.

Perfiles: NACA0012, AG24, GM15
Reynolds: 100 000, 1 000 000
Alphas: -10 a 10 cada 1 grado
"""

import os
import re
import shutil
import subprocess
import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Any, cast
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_SRC = os.path.join(BASE_DIR, "script_barrido_alpha.py")
COMP_DIR = os.path.join(BASE_DIR, "ComparativasReales")
OUT_ROOT = os.path.join(BASE_DIR, "resultados_comparacion")
TMP_SCRIPT = os.path.join(BASE_DIR, "_tmp_barrido.py")

PROFILES = [
    {"name": "NACA0012", "filepath": "profiles/NACA_0012"},
    {"name": "AG24",     "filepath": "profiles/AG24"},
    {"name": "GM15",     "filepath": "profiles/GM15"},
]
REYNOLDS_LIST = [100_000, 1_000_000]

SIM_OUTPUT_FILES = [
    "barrido_alpha_resultados.csv",
    "barrido_alpha_resultados.json",
    "barrido_alpha_Cd.png",
    "barrido_alpha_Cl.png",
    "barrido_alpha_Eficiencia.png",
    "barrido_alpha_Componentes.png",
]


def re_label(reynolds: int) -> str:
    return "100k" if reynolds == 100_000 else "1M"


# ---------------------------------------------------------------------------
# Paso 1: parchear y ejecutar script_barrido_alpha.py
# ---------------------------------------------------------------------------

def patch_script(filepath: str, reynolds: int) -> str:
    with open(SCRIPT_SRC, "r", encoding="utf-8") as f:
        src = f.read()

    src = re.sub(
        r'("filepath"\s*:\s*")[^"]*(")',
        rf'\g<1>{filepath}\2',
        src,
    )
    src = re.sub(
        r'(nu\s*=\s*)[\d\s/]+,',
        rf'\g<1>1 / {reynolds},',
        src,
    )
    src = re.sub(
        r'ALPHAS\s*=\s*list\(range\([^)]*\)\)',
        'ALPHAS = list(range(-10, 11, 1))',
        src,
    )

    with open(TMP_SCRIPT, "w", encoding="utf-8") as f:
        f.write(src)

    return TMP_SCRIPT


def run_one(profile: dict, reynolds: int) -> str:
    label = re_label(reynolds)
    run_dir = os.path.join(OUT_ROOT, f"{profile['name']}_{label}")
    os.makedirs(run_dir, exist_ok=True)

    results_csv = os.path.join(run_dir, "barrido_alpha_resultados.csv")
    if os.path.exists(results_csv):
        print(f"\n  [SKIP] {profile['name']} Re={label} — resultados ya existen")
        return run_dir

    print(f"\n{'='*70}")
    print(f"  Ejecutando: {profile['name']} Re={label}")
    print(f"{'='*70}")

    patch_script(profile["filepath"], reynolds)
    try:
        subprocess.run(
            [sys.executable, TMP_SCRIPT],
            cwd=BASE_DIR,
            check=True,
        )
    finally:
        if os.path.exists(TMP_SCRIPT):
            os.remove(TMP_SCRIPT)

    for fname in SIM_OUTPUT_FILES:
        src_path = os.path.join(BASE_DIR, fname)
        if os.path.exists(src_path):
            shutil.move(src_path, os.path.join(run_dir, fname))

    return run_dir


# ---------------------------------------------------------------------------
# Paso 2: cargar datos
# ---------------------------------------------------------------------------

def load_simulated(run_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(run_dir, "barrido_alpha_resultados.csv"))


def load_xfoil(profile_name: str, reynolds: int) -> pd.DataFrame:
    label = re_label(reynolds)
    path = os.path.join(COMP_DIR, f"{profile_name}_{label}_Xfoil.csv")
    header_row = None
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.lstrip().startswith("Alpha"):
                header_row = i
                break
    if header_row is None:
        raise ValueError(f"No se encontro cabecera 'Alpha' en: {path}")

    df = pd.read_csv(path, skiprows=list(range(header_row)), header=0, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df = df[(df["Alpha"] >= -10) & (df["Alpha"] <= 10)].copy()
    df = df.dropna(subset=["Alpha", "Cl", "Cd"])
    df = df.sort_values("Alpha").reset_index(drop=True)
    df["Ef"] = df["Cl"] / df["Cd"].replace(0, np.nan)
    return df


# ---------------------------------------------------------------------------
# Paso 3: estadisticas
# ---------------------------------------------------------------------------

def compute_stats(sim_vals: np.ndarray, theo_interp: np.ndarray) -> dict:
    diff = sim_vals - theo_interp
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((theo_interp - np.mean(theo_interp)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-15 else float("nan")
    nonzero = np.abs(theo_interp) > 1e-10
    rel_err = np.abs(diff[nonzero] / theo_interp[nonzero]) * 100
    mean_rel = float(np.mean(rel_err)) if rel_err.size > 0 else float("nan")
    max_abs = float(np.max(np.abs(diff)))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "mean_rel_err_pct": mean_rel,
        "max_abs_err": max_abs,
    }


def interpolate_xfoil(xfoil_df: pd.DataFrame, sim_alphas: np.ndarray, col: str) -> np.ndarray:
    fn = interp1d(
        xfoil_df["Alpha"].values,
        xfoil_df[col].values,
        kind="linear",
        bounds_error=False,
        fill_value=cast(Any, "extrapolate"),
    )
    return fn(sim_alphas)


# ---------------------------------------------------------------------------
# Paso 4: graficas comparativas
# ---------------------------------------------------------------------------

COEFS = [
    ("Cd", "Cd_mean", "Cd",  "Coeficiente de resistencia Cd"),
    ("Cl", "Cl_mean", "Cl",  "Coeficiente de sustentacion Cl"),
    ("Ef", "Ef_mean", "Ef",  "Eficiencia Cl/Cd"),
]


def plot_comparison(
    profile_name: str,
    reynolds: int,
    sim_df: pd.DataFrame,
    xfoil_df: pd.DataFrame,
    run_dir: str,
) -> list[dict]:
    label = re_label(reynolds)
    stats_rows = []

    for coef_id, sim_col, xfoil_col, ylabel in COEFS:
        if sim_col not in sim_df.columns:
            continue
        if xfoil_col not in xfoil_df.columns:
            continue

        sim_alphas = sim_df["alpha_deg"].to_numpy(dtype=float)
        sim_vals = sim_df[sim_col].to_numpy(dtype=float)
        theo_interp = interpolate_xfoil(xfoil_df, sim_alphas, xfoil_col)

        valid = ~(np.isnan(sim_vals) | np.isnan(theo_interp))
        if valid.sum() < 2:
            continue

        stats = compute_stats(sim_vals[valid], theo_interp[valid])
        stats_rows.append({
            "perfil": profile_name,
            "reynolds": reynolds,
            "coeficiente": coef_id,
            **stats,
        })

        diff = sim_vals - theo_interp
        bar_colors = ["#d62728" if d > 0 else "#1f77b4" for d in diff]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"{profile_name}  Re={label}  —  Comparativa {coef_id}",
            fontsize=13, fontweight="bold",
        )

        ax1.plot(xfoil_df["Alpha"], xfoil_df[xfoil_col],
                 "b-o", markersize=4, linewidth=1.5, label="Xfoil (teorico)")
        ax1.plot(sim_alphas, sim_vals,
                 "r-s", markersize=4, linewidth=1.5, label="LBM (simulado)")
        ax1.set_xlabel("Angulo de ataque α [°]")
        ax1.set_ylabel(ylabel)
        ax1.set_title("Curvas Xfoil vs LBM")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.bar(sim_alphas, diff, color=bar_colors, alpha=0.85, width=0.7)
        ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax2.set_xlabel("Angulo de ataque α [°]")
        ax2.set_ylabel(f"Error  (LBM − Xfoil)")
        stats_text = (
            f"MAE = {stats['MAE']:.4f}\n"
            f"RMSE = {stats['RMSE']:.4f}\n"
            f"R² = {stats['R2']:.4f}\n"
            f"Err.rel.medio = {stats['mean_rel_err_pct']:.1f} %\n"
            f"Max |err| = {stats['max_abs_err']:.4f}"
        )
        ax2.set_title(f"Diferencias\n{stats_text}", fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(run_dir, f"comparativa_{coef_id}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    -> {out_path}")

    stats_lines = []
    for s in stats_rows:
        stats_lines.append(
            f"{s['coeficiente']:3s}  MAE={s['MAE']:.5f}  RMSE={s['RMSE']:.5f}"
            f"  R²={s['R2']:.4f}  rel={s['mean_rel_err_pct']:.1f}%  max={s['max_abs_err']:.5f}"
        )
    stats_txt = "\n".join(stats_lines)
    with open(os.path.join(run_dir, "estadisticas.txt"), "w", encoding="utf-8") as f:
        f.write(f"{profile_name}  Re={label}\n{'='*60}\n{stats_txt}\n")
    print(f"    -> estadisticas.txt")

    return stats_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    all_stats = []

    for profile in PROFILES:
        for reynolds in REYNOLDS_LIST:
            run_dir = run_one(profile, reynolds)

            print(f"  Generando comparativas para {profile['name']} Re={re_label(reynolds)} ...")
            sim_df = load_simulated(run_dir)
            xfoil_df = load_xfoil(profile["name"], reynolds)
            stats_rows = plot_comparison(
                profile["name"], reynolds, sim_df, xfoil_df, run_dir
            )
            all_stats.extend(stats_rows)

    # Resumen global
    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        summary_path = os.path.join(OUT_ROOT, "resumen_estadisticas.csv")
        summary_df.to_csv(summary_path, index=False, float_format="%.6f")
        print(f"\nResumen global: {summary_path}")

        print("\n" + "=" * 80)
        print(f"{'PERFIL':<10} {'Re':>7} {'COEF':>4}  {'MAE':>9}  {'RMSE':>9}  {'R²':>7}  {'rel%':>7}  {'max':>9}")
        print("-" * 80)
        for row in summary_df.itertuples(index=False):
            print(
                f"{row.perfil:<10} {cast(int, row.reynolds):>7}  {row.coeficiente:>3}  "
                f"{row.MAE:>9.5f}  {row.RMSE:>9.5f}  {row.R2:>7.4f}  "
                f"{row.mean_rel_err_pct:>6.1f}%  {row.max_abs_err:>9.5f}"
            )
        print("=" * 80)

    print("\nFinalizado. Resultados en:", OUT_ROOT)


if __name__ == "__main__":
    main()

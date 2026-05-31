"""
Ejecuta barridos de alpha y compara contra XFoil.

Usa `script_barrido_alpha.py` sin parchear codigo, por lo que hereda la ruta
corregida del simulador: MG `outer_sum` y gradiente de pared `masked`.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sim_defaults import preferred_cl_column, preferred_cd_column

SCRIPT_ALPHA = SCRIPT_DIR / "script_barrido_alpha.py"
COMP_DIR = ROOT_DIR / "data" / "ComparativasReales"
OUT_ROOT = ROOT_DIR / "results" / "comparativas" / "resultados_comparacion_outer_sum"
BARRIDOS_ROOT = ROOT_DIR / "results" / "barridos"

PROFILES = [
    {"name": "NACA0012", "filepath": "profiles/NACA_0012"},
    {"name": "AG24", "filepath": "profiles/AG24"},
    {"name": "GM15", "filepath": "profiles/GM15"},
]
REYNOLDS_LIST = [100_000, 1_000_000]


def re_label(reynolds: int) -> str:
    return "100k" if reynolds == 100_000 else "1M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Barridos alpha outer_sum vs XFoil.")
    parser.add_argument("--alphas", default="-10:10:1")
    parser.add_argument("--force", action="store_true", help="recalcula los barridos existentes")
    parser.add_argument("--iteraciones", type=int, default=2000)
    parser.add_argument("--guardado", type=int, default=50)
    parser.add_argument("--dx-min", type=float, default=0.001)
    return parser.parse_args()


def run_one(profile: dict[str, str], reynolds: int, args: argparse.Namespace) -> Path:
    label = re_label(reynolds)
    suffix = f"{profile['name'].lower()}_{label.lower()}_outer_sum"
    barrido_dir = BARRIDOS_ROOT / f"barrido_alpha_{suffix}"
    summary_csv = barrido_dir / "summary.csv"

    if summary_csv.exists() and not args.force:
        print(f"\n  [SKIP] {profile['name']} Re={label}: {summary_csv.relative_to(ROOT_DIR)}")
        return barrido_dir

    cmd = [
        sys.executable,
        str(SCRIPT_ALPHA),
        "--profile",
        profile["filepath"],
        "--nu",
        f"{1.0 / reynolds:.16g}",
        "--alphas",
        args.alphas,
        "--output-suffix",
        suffix,
        "--iteraciones",
        str(args.iteraciones),
        "--guardado",
        str(args.guardado),
        "--dx-min",
        str(args.dx_min),
    ]
    if args.force:
        cmd.append("--force")

    print(f"\n{'=' * 70}")
    print(f"  Ejecutando: {profile['name']} Re={label} -> {barrido_dir.relative_to(ROOT_DIR)}")
    print(f"{'=' * 70}")
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)
    return barrido_dir


def load_simulated(barrido_dir: Path) -> pd.DataFrame:
    path = barrido_dir / "summary.csv"
    df = pd.read_csv(path)
    cl_col = preferred_cl_column(set(df.columns))
    cd_col = preferred_cd_column(set(df.columns), prefer_bl=True)
    df = df.copy()
    df["Cl_eval"] = df[cl_col]
    df["Cd_eval"] = df[cd_col]
    df["Ef_eval"] = df["Cl_eval"] / df["Cd_eval"].replace(0, np.nan)
    df.attrs["cl_column"] = cl_col
    df.attrs["cd_column"] = cd_col
    return df


def load_xfoil(profile_name: str, reynolds: int) -> pd.DataFrame:
    label = re_label(reynolds)
    path = COMP_DIR / f"{profile_name}_{label}_Xfoil.csv"
    header_row = None
    with path.open("r", encoding="utf-8") as f:
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


def compute_stats(sim_vals: np.ndarray, theo_interp: np.ndarray) -> dict[str, float]:
    diff = sim_vals - theo_interp
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((theo_interp - np.mean(theo_interp)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-15 else float("nan")
    nonzero = np.abs(theo_interp) > 1e-10
    rel_err = np.abs(diff[nonzero] / theo_interp[nonzero]) * 100
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "mean_rel_err_pct": float(np.mean(rel_err)) if rel_err.size else float("nan"),
        "max_abs_err": float(np.max(np.abs(diff))),
    }


def interpolate_xfoil(xfoil_df: pd.DataFrame, sim_alphas: np.ndarray, col: str) -> np.ndarray:
    return np.interp(sim_alphas, xfoil_df["Alpha"].to_numpy(float), xfoil_df[col].to_numpy(float))


COEFS = [
    ("Cd", "Cd_eval", "Cd", "Coeficiente de resistencia Cd"),
    ("Cl", "Cl_eval", "Cl", "Coeficiente de sustentacion Cl"),
    ("Ef", "Ef_eval", "Ef", "Eficiencia Cl/Cd"),
]


def plot_comparison(
    profile_name: str,
    reynolds: int,
    sim_df: pd.DataFrame,
    xfoil_df: pd.DataFrame,
    out_dir: Path,
) -> list[dict[str, Any]]:
    label = re_label(reynolds)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_rows: list[dict[str, Any]] = []

    for coef_id, sim_col, xfoil_col, ylabel in COEFS:
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
            "sim_column": sim_col,
            **stats,
        })

        diff = sim_vals - theo_interp
        bar_colors = ["#d62728" if d > 0 else "#1f77b4" for d in diff]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"{profile_name} Re={label} - Comparativa {coef_id}",
            fontsize=13,
            fontweight="bold",
        )

        ax1.plot(xfoil_df["Alpha"], xfoil_df[xfoil_col], "b-o", markersize=4,
                 linewidth=1.5, label="XFoil")
        ax1.plot(sim_alphas, sim_vals, "r-s", markersize=4, linewidth=1.5,
                 label="LBM outer_sum")
        ax1.set_xlabel("Angulo de ataque alpha [deg]")
        ax1.set_ylabel(ylabel)
        ax1.set_title("Curvas XFoil vs LBM")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.bar(sim_alphas, diff, color=bar_colors, alpha=0.85, width=0.7)
        ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax2.set_xlabel("Angulo de ataque alpha [deg]")
        ax2.set_ylabel("Error LBM - XFoil")
        ax2.set_title(
            f"MAE={stats['MAE']:.4f} RMSE={stats['RMSE']:.4f}\n"
            f"R2={stats['R2']:.4f} rel={stats['mean_rel_err_pct']:.1f}%"
        )
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        out_path = out_dir / f"comparativa_{coef_id}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    -> {out_path.relative_to(ROOT_DIR)}")

    stats_lines = [
        f"{s['coeficiente']:3s} MAE={s['MAE']:.5f} RMSE={s['RMSE']:.5f} "
        f"R2={s['R2']:.4f} rel={s['mean_rel_err_pct']:.1f}% max={s['max_abs_err']:.5f}"
        for s in stats_rows
    ]
    with (out_dir / "estadisticas.txt").open("w", encoding="utf-8") as f:
        f.write(
            f"{profile_name} Re={label}\n"
            f"Cl simulado: {sim_df.attrs.get('cl_column')}\n"
            f"Cd simulado: {sim_df.attrs.get('cd_column')}\n"
            f"{'=' * 60}\n"
            + "\n".join(stats_lines)
            + "\n"
        )
    return stats_rows


def main() -> int:
    args = parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_stats: list[dict[str, Any]] = []

    for profile in PROFILES:
        for reynolds in REYNOLDS_LIST:
            barrido_dir = run_one(profile, reynolds, args)
            out_dir = OUT_ROOT / f"{profile['name']}_{re_label(reynolds)}"
            print(f"  Comparando {profile['name']} Re={re_label(reynolds)} ...")
            sim_df = load_simulated(barrido_dir)
            xfoil_df = load_xfoil(profile["name"], reynolds)
            all_stats.extend(plot_comparison(profile["name"], reynolds, sim_df, xfoil_df, out_dir))

    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        summary_path = OUT_ROOT / "resumen_estadisticas.csv"
        summary_df.to_csv(summary_path, index=False, float_format="%.6f")
        print(f"\nResumen global: {summary_path.relative_to(ROOT_DIR)}")

        print("\n" + "=" * 80)
        print(f"{'PERFIL':<10} {'Re':>7} {'COEF':>4} {'MAE':>9} {'RMSE':>9} {'R2':>7} {'rel%':>7}")
        print("-" * 80)
        for row in summary_df.itertuples(index=False):
            print(
                f"{row.perfil:<10} {cast(int, row.reynolds):>7} {row.coeficiente:>4} "
                f"{row.MAE:>9.5f} {row.RMSE:>9.5f} {row.R2:>7.4f} "
                f"{row.mean_rel_err_pct:>6.1f}%"
            )
        print("=" * 80)

    print(f"\nFinalizado. Resultados en: {OUT_ROOT.relative_to(ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

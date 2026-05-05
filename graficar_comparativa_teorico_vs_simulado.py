"""Grafica Cd, Cl y Ef comparando un polar teorico con resultados simulados.

Uso por defecto:
    python graficar_comparativa_teorico_vs_simulado.py

Genera un PNG combinado y tres PNG individuales en la carpeta de salida.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SIM = BASE_DIR / "barrido_naca0012_resultados_vopus.csv"
DEFAULT_THEO = BASE_DIR / "ComparativasReales" / "AG24_100k_Xfoil.csv"
DEFAULT_OUT = BASE_DIR / "comparativa_teorico_vs_simulado"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grafica comparativa teorico vs simulado.")
    parser.add_argument("--simulado", type=Path, default=DEFAULT_SIM, help="CSV simulado")
    parser.add_argument("--teorico", type=Path, default=DEFAULT_THEO, help="CSV teorico/XFOIL")
    parser.add_argument("--salida", type=Path, default=DEFAULT_OUT, help="Carpeta de salida")
    return parser.parse_args()


def read_simulated(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"alpha_deg", "Cd_mean", "Cl_mean", "Ef_mean"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en simulado {path}: {sorted(missing)}")
    return df.sort_values("alpha_deg").reset_index(drop=True)


def read_theoretical(path: Path) -> pd.DataFrame:
    header_row = None
    airfoil_name = None
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            stripped = line.strip()
            if stripped.startswith("Airfoil,"):
                airfoil_name = stripped.split(",", 1)[1].strip()
            if stripped.startswith("Alpha,"):
                header_row = idx
                break

    if header_row is None:
        raise ValueError(f"No se encontro la cabecera Alpha en {path}")

    df = pd.read_csv(path, skiprows=list(range(header_row)), header=0, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    required = {"Alpha", "Cl", "Cd"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en teorico {path}: {sorted(missing)}")

    df = df.dropna(subset=["Alpha", "Cl", "Cd"]).copy()
    df["Ef"] = df["Cl"] / df["Cd"].replace(0, np.nan)
    df = df.sort_values("Alpha").reset_index(drop=True)
    if airfoil_name:
        df.attrs["airfoil_name"] = airfoil_name
    return df


def plot_pair(ax, theo_x, theo_y, sim_x, sim_y, ylabel: str, title: str) -> None:
    ax.plot(theo_x, theo_y, "b-o", markersize=3.5, linewidth=1.6, label="Teorico")
    ax.plot(sim_x, sim_y, "r-s", markersize=3.5, linewidth=1.6, label="Simulado")
    ax.set_xlabel("Angulo de ataque alpha [deg]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()


def save_plots(theo_df: pd.DataFrame, sim_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    theo_label = theo_df.attrs.get("airfoil_name", "teorico")
    print(f"Teorico: {theo_label}")
    print(f"Simulado: {len(sim_df)} filas")

    saved_files: list[Path] = []

    combined_specs = [
        ("Cd", "Cd", "Cd_mean", "Coeficiente de resistencia Cd"),
        ("Cl", "Cl", "Cl_mean", "Coeficiente de sustentacion Cl"),
        ("Ef", "Ef", "Ef_mean", "Eficiencia Cl/Cd"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, theo_col, sim_col, ylabel) in zip(axes, combined_specs):
        plot_pair(
            ax,
            theo_df["Alpha"],
            theo_df[theo_col],
            sim_df["alpha_deg"],
            sim_df[sim_col],
            ylabel,
            f"{name} teorico vs simulado",
        )
    fig.suptitle("Comparativa teorico vs simulado", fontsize=14, fontweight="bold")
    fig.tight_layout()
    combined_path = out_dir / "comparativa_teorico_vs_simulado.png"
    fig.savefig(combined_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(combined_path)

    individual_specs = [
        ("Cd", "Cd", "Cd_mean", "Coeficiente de resistencia Cd"),
        ("Cl", "Cl", "Cl_mean", "Coeficiente de sustentacion Cl"),
        ("Ef", "Ef", "Ef_mean", "Eficiencia Cl/Cd"),
    ]
    for name, theo_col, sim_col, ylabel in individual_specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_pair(
            ax,
            theo_df["Alpha"],
            theo_df[theo_col],
            sim_df["alpha_deg"],
            sim_df[sim_col],
            ylabel,
            f"{name} teorico vs simulado",
        )
        fig.tight_layout()
        out_path = out_dir / f"comparativa_{name}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(out_path)

    return saved_files


def main() -> None:
    args = parse_args()

    if not args.simulado.exists():
        raise FileNotFoundError(f"No existe el CSV simulado: {args.simulado}")
    if not args.teorico.exists():
        raise FileNotFoundError(f"No existe el CSV teorico: {args.teorico}")

    sim_df = read_simulated(args.simulado)
    theo_df = read_theoretical(args.teorico)
    saved = save_plots(theo_df, sim_df, args.salida)

    print("Archivos generados:")
    for path in saved:
        print(f"- {path}")


if __name__ == "__main__":
    main()
"""Genera la comparacion NACA 0012 vs la segunda tabla de referencia."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "resultados_polares_curvo" / "NACA_0012_sharp"
REFERENCE_FILE = ROOT / "docs" / "referencia_naca0012_nueva_0_15.dat"
OUTPUT = RESULTS / "graficas" / "comparacion_referencia_nueva_0_15.png"


def leer_referencia():
    return np.loadtxt(REFERENCE_FILE, comments="#", usecols=(0, 1, 2))


def leer_calculado():
    filas = []
    for ruta in sorted(RESULTS.glob("*/resumen.json")):
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        if datos.get("estado") != "completada":
            continue
        metricas = datos["metricas_finales"]
        filas.append((float(datos["alfa"]), float(metricas["Cl"]),
                      float(metricas["Cd"])))
    if not filas:
        raise SystemExit(f"No hay resultados en {RESULTS}")
    return np.array(sorted(filas), dtype=float)


def main():
    referencia = leer_referencia()
    calculado = leer_calculado()
    comunes = np.intersect1d(referencia[:, 0], calculado[:, 0])
    comunes = comunes[(comunes >= 0.0) & (comunes <= 15.0)]
    referencia = referencia[np.isin(referencia[:, 0], comunes)]
    calculado = calculado[np.isin(calculado[:, 0], comunes)]
    referencia = referencia[np.argsort(referencia[:, 0])]
    calculado = calculado[np.argsort(calculado[:, 0])]
    if not len(comunes):
        raise SystemExit("No hay angulos comunes entre 0 y 15 grados")

    error_cl = 100 * (calculado[:, 1] - referencia[:, 1]) / np.maximum(
        np.abs(referencia[:, 1]), 0.01)
    error_cd = 100 * (calculado[:, 2] - referencia[:, 2]) / np.maximum(
        np.abs(referencia[:, 2]), 1e-6)
    comparacion = np.column_stack((calculado[:, 0], calculado[:, 1], calculado[:, 2],
                                   referencia[:, 1], referencia[:, 2], error_cl, error_cd))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo)
        writer.writerow(("alpha", "cl_calculado", "cd_calculado", "cl_referencia",
                         "cd_referencia", "error_cl_pct", "error_cd_pct"))
        writer.writerows(comparacion)

    alpha = calculado[:, 0]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    series = (
        (axes[0, 0], referencia[:, 0], referencia[:, 1], calculado[:, 0], calculado[:, 1],
         "$C_l$ frente a $\\alpha$", "$C_l$"),
        (axes[0, 1], referencia[:, 0], referencia[:, 2], calculado[:, 0], calculado[:, 2],
         "$C_d$ frente a $\\alpha$", "$C_d$"),
        (axes[1, 0], referencia[:, 2], referencia[:, 1], calculado[:, 2], calculado[:, 1],
         "Polar $C_l-C_d$", "$C_l$"),
    )
    for eje, xr, yr, xc, yc, titulo, ylabel in series:
        eje.plot(xr, yr, "k--", lw=1.8, label="Referencia nueva")
        eje.plot(xc, yc, "o-", color="#1769aa", label="Solver curvo")
        eje.set_title(titulo)
        eje.set_ylabel(ylabel)
        eje.grid(alpha=0.25)
        eje.legend()
    axes[0, 0].set_xlabel("$\\alpha$ [grados]")
    axes[0, 1].set_xlabel("$\\alpha$ [grados]")
    axes[1, 0].set_xlabel("$C_d$")
    axes[1, 1].plot(alpha, error_cl, "o-", label="Error $C_l$ [%]")
    axes[1, 1].plot(alpha, error_cd, "s-", label="Error $C_d$ [%]")
    axes[1, 1].axhline(0, color="0.3", lw=0.8)
    axes[1, 1].set_title("Error relativo")
    axes[1, 1].set_xlabel("$\\alpha$ [grados]")
    axes[1, 1].set_ylabel("[%]")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()
    fig.suptitle("NACA 0012 sharp: solver curvo vs nueva referencia | 0-15 grados")
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)
    print(f"Imagen: {OUTPUT}")
    print(f"Datos:  {csv_path}")
    print(f"Angulos comparados: {len(comparacion)} ({alpha[0]:g} a {alpha[-1]:g})")
    print(f"MAE Cl: {np.mean(np.abs(error_cl)):.2f}%")
    print(f"MAE Cd: {np.mean(np.abs(error_cd)):.2f}%")


if __name__ == "__main__":
    main()

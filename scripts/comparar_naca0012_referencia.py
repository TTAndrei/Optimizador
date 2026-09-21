"""Compara la polar calculada del NACA 0012 sharp con la tabla de referencia.

Uso desde la raiz:
    .venv/bin/python scripts/comparar_naca0012_referencia.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "resultados_polares_curvo" / "NACA_0012_sharp"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "graficas" / "comparacion_referencia_0_14.png"
REFERENCE = """\
alpha CL CD
-14.000 -0.8751 0.09584
-13.750 -0.9212 0.08276
-13.500 -0.9550 0.07345
-13.250 -0.9811 0.06643
-13.000 -1.0020 0.06089
-12.750 -1.0192 0.05635
-12.500 -1.0332 0.05253
-12.250 -1.0442 0.04926
-12.000 -1.0447 0.04681
-11.750 -1.0402 0.04492
-11.500 -1.0359 0.04320
-11.250 -1.0301 0.04152
-11.000 -1.0224 0.03971
-10.750 -1.0131 0.03782
-10.500 -0.9997 0.03620
-10.250 -0.9851 0.03484
-10.000 -0.9701 0.03353
-9.750 -0.9540 0.03219
-9.500 -0.9377 0.03089
-9.250 -0.9208 0.02978
-9.000 -0.9030 0.02869
-8.750 -0.8854 0.02758
-8.500 -0.8678 0.02663
-8.250 -0.8488 0.02572
-8.000 -0.8315 0.02476
-7.750 -0.8125 0.02394
-7.500 -0.7946 0.02308
-7.250 -0.7753 0.02235
-7.000 -0.7571 0.02157
-6.750 -0.7385 0.02085
-6.500 -0.7196 0.02017
-6.250 -0.7005 0.01953
-6.000 -0.6813 0.01892
-5.750 -0.6617 0.01835
-5.500 -0.6423 0.01779
-5.250 -0.6226 0.01728
-5.000 -0.6027 0.01681
-4.750 -0.5826 0.01637
-4.500 -0.5623 0.01598
-4.250 -0.5419 0.01562
-4.000 -0.5212 0.01530
-3.750 -0.5004 0.01501
-3.500 -0.4793 0.01476
-3.250 -0.4580 0.01454
-3.000 -0.4365 0.01435
-2.750 -0.4120 0.01419
-2.500 -0.3736 0.01405
-2.250 -0.3346 0.01394
-2.000 -0.2976 0.01384
-1.750 -0.2601 0.01376
-1.500 -0.2196 0.01370
-1.250 -0.1837 0.01365
-1.000 -0.1475 0.01362
-0.750 -0.1093 0.01360
-0.500 -0.0724 0.01359
-0.250 -0.0370 0.01358
0.000 0.0000 0.01358
0.250 0.0370 0.01358
0.500 0.0724 0.01359
0.750 0.1093 0.01360
1.000 0.1475 0.01362
1.250 0.1837 0.01365
1.500 0.2196 0.01370
1.750 0.2601 0.01376
2.000 0.2976 0.01383
2.250 0.3346 0.01394
2.500 0.3736 0.01405
2.750 0.4120 0.01419
3.000 0.4365 0.01435
3.250 0.4580 0.01454
3.500 0.4793 0.01476
3.750 0.5004 0.01501
4.000 0.5212 0.01530
4.250 0.5419 0.01562
4.500 0.5623 0.01598
4.750 0.5826 0.01637
5.000 0.6027 0.01681
5.250 0.6226 0.01728
5.500 0.6423 0.01779
5.750 0.6617 0.01835
6.000 0.6813 0.01892
6.250 0.7005 0.01953
6.500 0.7196 0.02017
6.750 0.7385 0.02085
7.000 0.7571 0.02157
7.250 0.7753 0.02235
7.500 0.7946 0.02308
7.750 0.8125 0.02394
8.000 0.8315 0.02476
8.250 0.8488 0.02572
8.500 0.8678 0.02663
8.750 0.8854 0.02758
9.000 0.9030 0.02869
9.250 0.9208 0.02978
9.500 0.9377 0.03089
9.750 0.9540 0.03219
10.000 0.9701 0.03353
10.250 0.9851 0.03484
10.500 0.9997 0.03620
10.750 1.0131 0.03782
11.000 1.0224 0.03971
11.250 1.0301 0.04152
11.500 1.0359 0.04320
11.750 1.0402 0.04492
12.000 1.0447 0.04681
12.250 1.0442 0.04926
12.500 1.0332 0.05253
12.750 1.0192 0.05635
13.000 1.0020 0.06089
13.250 0.9811 0.06643
13.500 0.9550 0.07345
13.750 0.9212 0.08276
14.000 0.8751 0.09584
"""


def referencia():
    filas = []
    for linea in REFERENCE.splitlines()[1:]:
        alpha, cl, cd = linea.split()
        filas.append((float(alpha), float(cl), float(cd)))
    return np.array(filas, dtype=float)


def calculado(results_dir):
    filas = []
    for ruta in sorted(results_dir.glob("*/resumen.json")):
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        if datos.get("estado") != "completada":
            continue
        metricas = datos["metricas_finales"]
        filas.append((float(datos["alfa"]), float(metricas["Cl"]), float(metricas["Cd"])))
    if not filas:
        raise SystemExit(f"No hay corridas completadas en {results_dir}")
    return np.array(sorted(filas), dtype=float)


def guardar_csv(ruta, datos):
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        writer = csv.writer(archivo)
        writer.writerow(("alpha", "cl_calculado", "cd_calculado", "cl_referencia",
                         "cd_referencia", "error_cl_pct", "error_cd_pct"))
        for fila in datos:
            writer.writerow(fila)


def generar(results_dir=DEFAULT_RESULTS, output=DEFAULT_OUTPUT):
    ref = referencia()
    calc = calculado(results_dir)
    comunes = np.intersect1d(ref[:, 0], calc[:, 0])
    comunes = comunes[(comunes >= 0.0) & (comunes <= 14.0)]
    if not len(comunes):
        raise SystemExit("No hay angulos comunes entre 0 y 14 grados")
    ref = ref[np.isin(ref[:, 0], comunes)]
    calc = calc[np.isin(calc[:, 0], comunes)]
    ref = ref[np.argsort(ref[:, 0])]
    calc = calc[np.argsort(calc[:, 0])]
    error_cl = 100.0 * (calc[:, 1] - ref[:, 1]) / np.maximum(np.abs(ref[:, 1]), 0.01)
    error_cd = 100.0 * (calc[:, 2] - ref[:, 2]) / np.maximum(np.abs(ref[:, 2]), 1e-6)
    comparacion = np.column_stack((calc[:, 0], calc[:, 1], calc[:, 2], ref[:, 1], ref[:, 2],
                                   error_cl, error_cd))
    output.parent.mkdir(parents=True, exist_ok=True)
    guardar_csv(output.with_suffix(".csv"), comparacion)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot(ref[:, 0], ref[:, 1], "k--", lw=1.8, label="Referencia")
    axes[0, 0].plot(calc[:, 0], calc[:, 1], "o-", color="#1769aa", label="Curvo")
    axes[0, 0].set_title("$C_l$ frente a $\\alpha$")
    axes[0, 0].set_ylabel("$C_l$")
    axes[0, 1].plot(ref[:, 0], ref[:, 2], "k--", lw=1.8, label="Referencia")
    axes[0, 1].plot(calc[:, 0], calc[:, 2], "o-", color="#d95f02", label="Curvo")
    axes[0, 1].set_title("$C_d$ frente a $\\alpha$")
    axes[0, 1].set_ylabel("$C_d$")
    axes[1, 0].plot(ref[:, 2], ref[:, 1], "k--", lw=1.8, label="Referencia")
    axes[1, 0].plot(calc[:, 2], calc[:, 1], "o-", color="#238b45", label="Curvo")
    axes[1, 0].set_title("Polar $C_l-C_d$")
    axes[1, 0].set_xlabel("$C_d$")
    axes[1, 0].set_ylabel("$C_l$")
    axes[1, 1].plot(calc[:, 0], error_cl, "o-", label="Error $C_l$ [%]")
    axes[1, 1].plot(calc[:, 0], error_cd, "s-", label="Error $C_d$ [%]")
    axes[1, 1].axhline(0.0, color="0.3", lw=0.8)
    axes[1, 1].set_title("Diferencia relativa respecto a la referencia")
    axes[1, 1].set_xlabel("$\\alpha$ [grados]")
    axes[1, 1].set_ylabel("Error [%]")
    for eje in axes.flat:
        eje.grid(alpha=0.25)
        eje.legend()
    fig.suptitle("NACA 0012 sharp: solver curvilineo vs referencia | 0-14 grados")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output, output.with_suffix(".csv"), comparacion


if __name__ == "__main__":
    imagen, csv_path, datos = generar()
    print(f"Imagen: {imagen}")
    print(f"Datos:  {csv_path}")
    print(f"Angulos comparados: {len(datos)} ({datos[0, 0]:g} a {datos[-1, 0]:g})")
    print(f"MAE Cl: {np.mean(np.abs(datos[:, 5])):.2f}%")
    print(f"MAE Cd: {np.mean(np.abs(datos[:, 6])):.2f}%")

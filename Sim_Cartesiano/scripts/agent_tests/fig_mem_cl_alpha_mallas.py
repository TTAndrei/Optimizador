"""
Figura 5.A — Cl(alpha) de los dos perfiles en las mallas del estudio final.

Un panel por perfil, una curva por malla, barras = CI95 de la media temporal.
Es la lamina que sostiene dos cosas a la vez: la separacion entre los dos
perfiles y cuanto de esa separacion se mueve al cambiar de malla.

Se dibujan dx = 0.008 / 0.004 / 0.002, que es la terna del GCI (r = 2 constante
en los dos saltos). La cuarta malla del estudio, dx = 0.006, se omite a
proposito: no entra en la terna —se calculo como contraste— y su curva se
solapa con las otras dos sin anadir informacion.

Datos: `resultados_finales/metricas_todas.csv` (2 perfiles x 4 mallas x 9
angulos, Re = 1e5, dominio 24x16). Este script solo dibuja.

Uso:
    .venv/bin/python scripts/agent_tests/fig_mem_cl_alpha_mallas.py
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "figuras_memoria")
CSV = os.path.join(ROOT, "resultados_finales/metricas_todas.csv")
DPI = 200

DXS = (0.008, 0.004, 0.002)          # terna del GCI; 0.006 fuera a proposito
# mismos colores y marcas que las figuras de resultados_finales
COLOR_DX = {0.008: "#e67e22", 0.004: "#8e44ad", 0.002: "#1f4e79"}
MARCA_DX = {0.008: "D", 0.004: "^", 0.002: "s"}
PERFILES = (("naca0012", "NACA 0012 (semilla)"),
            ("ganador_ag", "Ganador de la optimización"))


def main(solo=None):
    datos = defaultdict(dict)
    for r in csv.DictReader(open(CSV)):
        datos[(r["perfil"], float(r["dx"]))][float(r["alpha"])] = (
            float(r["cl"]), float(r["cl_ci95"] or 0.0))

    perfiles = [pf for pf in PERFILES if solo in (None, pf[0])]
    if solo and not perfiles:
        raise SystemExit(f"perfil desconocido: {solo}")
    ancho = 10.6 if len(perfiles) == 2 else 7.4
    fig, axes = plt.subplots(1, len(perfiles), figsize=(ancho, 4.6), sharey=True,
                             squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=0.065 if len(perfiles) == 2 else 0.10,
                        right=0.99, top=0.855, bottom=0.13, wspace=0.06)

    for ax, (perfil, nombre) in zip(axes, perfiles):
        for dx in DXS:
            d = datos[(perfil, dx)]
            if not d:
                continue
            a = sorted(d)
            ax.errorbar(a, [d[k][0] for k in a], yerr=[d[k][1] for k in a],
                        color=COLOR_DX[dx], marker=MARCA_DX[dx], ms=5, lw=1.4,
                        capsize=3, label=f"$dx = {dx:g}$")
            print(f"  {perfil:<11} dx={dx:<6g} Cl(4°)={d[4.0][0]:.4f} ± {d[4.0][1]:.4f}")
        ax.set_title(nombre if len(perfiles) == 2 else
                     f"{nombre} — barras: CI95 de la media temporal",
                     fontsize=10.5 if len(perfiles) == 2 else 11.5, pad=8)
        ax.set_xlabel(r"$\alpha$ [°]")
        ax.grid(alpha=0.25, lw=0.6)
        ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)

    axes[0].set_ylabel(r"$C_l$")
    axes[0].legend(loc="upper left", fontsize=9.5, framealpha=0.95)
    if len(perfiles) == 2:
        fig.suptitle("Sustentación frente al ángulo de ataque — barras: CI95 de "
                     "la media temporal", fontsize=11.5, y=0.985)
    else:
        fig.subplots_adjust(top=0.915)

    nombre_fich = ("fig_5A_cl_alpha_mallas.png" if len(perfiles) == 2
                   else f"fig_5A_cl_alpha_mallas_{perfiles[0][0]}.png")
    p = os.path.join(OUT, nombre_fich)
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("escrito:", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--perfil", choices=[k for k, _ in PERFILES],
                    help="dibuja un solo perfil en vez de los dos paneles")
    main(ap.parse_args().perfil)

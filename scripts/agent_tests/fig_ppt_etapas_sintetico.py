"""
Tira horizontal para diapositiva: lo que aporta cada pieza del solver de presion.

Ejemplo inventado, no una simulacion: se resuelve un Poisson de juguete
(lap(p) = 0 con p = 0 en el contorno, asi que la solucion exacta es cero y el
propio iterado ES el error) partiendo de un error que mezcla dos escalas, ruido
celda a celda y modos suaves de dominio entero. Los cuatro paneles enseñan LA
MISMA magnitud, ese error, con la MISMA escala de color:

  1  error inicial               brusco y suave a la vez
  2  3 barridos Gauss-Seidel     el suavizador borra lo brusco y no toca lo suave
  3  + ciclo en V (2 niveles)    lo suave es brusco en la malla gruesa: se va
  4  + 2a iteracion externa      el presupuesto de produccion

Todo el algoritmo es el de verdad —Gauss-Seidel rojo-negro, restriccion de peso
completo, prolongacion bilineal— pero en numpy y sobre una malla uniforme, que
es lo que hace la figura legible.

Sin titulo: la figura va sobre la diapositiva.

Uso:
    .venv/bin/python scripts/agent_tests/fig_ppt_etapas_sintetico.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "agent_tests"))
os.chdir(ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from fig_ppt_proyeccion import flecha, sin_ejes, GRIS

OUT = os.path.join(ROOT, "figuras_memoria")
DPI = 240
N = 129            # nodos por lado (2^k + 1, para que el coarsening cuadre)
BARRIDOS = 3       # pre y post suavizado, los mismos que usa el solver


def _gs_rojo_negro(p, f, h2, n_barridos):
    """Gauss-Seidel por paridad: dentro de cada color las celdas son independientes."""
    for _ in range(n_barridos):
        for paridad in (0, 1):
            i, j = np.indices(p.shape)
            m = ((i + j) % 2 == paridad)
            m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = False
            vecinos = np.zeros_like(p)
            vecinos[1:-1, 1:-1] = (p[:-2, 1:-1] + p[2:, 1:-1]
                                   + p[1:-1, :-2] + p[1:-1, 2:])
            p[m] = 0.25 * (vecinos[m] - h2 * f[m])
    return p


def _residuo(p, f, h2):
    r = np.zeros_like(p)
    r[1:-1, 1:-1] = f[1:-1, 1:-1] - (
        p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
        - 4.0 * p[1:-1, 1:-1]) / h2
    return r


def _restringe(r):
    """Peso completo: la media ponderada de los nueve vecinos finos."""
    R = r[::2, ::2].copy()
    R[1:-1, 1:-1] = (
        4 * r[2:-2:2, 2:-2:2]
        + 2 * (r[1:-3:2, 2:-2:2] + r[3:-1:2, 2:-2:2]
               + r[2:-2:2, 1:-3:2] + r[2:-2:2, 3:-1:2])
        + (r[1:-3:2, 1:-3:2] + r[1:-3:2, 3:-1:2]
           + r[3:-1:2, 1:-3:2] + r[3:-1:2, 3:-1:2])) / 16.0
    return R


def _prolonga(E, forma):
    """Bilineal: inyeccion en los nodos comunes y media en los intermedios."""
    e = np.zeros(forma)
    e[::2, ::2] = E
    e[1::2, ::2] = 0.5 * (e[:-1:2, ::2] + e[2::2, ::2])
    e[:, 1::2] = 0.5 * (e[:, :-1:2] + e[:, 2::2])
    return e


def ciclo_v(p, f, h):
    p = _gs_rojo_negro(p, f, h * h, BARRIDOS)
    r = _residuo(p, f, h * h)
    R = _restringe(r)
    E = np.zeros_like(R)
    # "resolver" el nivel grueso: los modos mas suaves necesitan del orden de n^2
    # barridos, y ahi son baratos
    E = _gs_rojo_negro(E, R, (2 * h) ** 2, 600)
    p += _prolonga(E, p.shape)
    return _gs_rojo_negro(p, f, h * h, BARRIDOS)


def campos():
    x = np.linspace(0, 1, N)
    XX, YY = np.meshgrid(x, x)
    rng = np.random.default_rng(7)

    # error de partida: dos modos suaves + ruido celda a celda
    e0 = (1.0 * np.sin(np.pi * XX) * np.sin(np.pi * YY)
          + 0.6 * np.sin(2 * np.pi * XX) * np.sin(3 * np.pi * YY)
          + 0.55 * rng.standard_normal(XX.shape))
    e0[0, :] = e0[-1, :] = e0[:, 0] = e0[:, -1] = 0.0

    f = np.zeros_like(e0)     # solucion exacta = 0, asi que el iterado es el error
    h = 1.0 / (N - 1)

    p1 = _gs_rojo_negro(e0.copy(), f, h * h, BARRIDOS)
    p2 = ciclo_v(p1.copy(), f, h)
    p3 = ciclo_v(p2.copy(), f, h)
    return [("error inicial", e0), ("3 barridos Gauss-Seidel", p1),
            ("+ ciclo en V", p2), ("+ 2ª iteración externa", p3)]


PASOS = [("Gauss-Seidel", "rojo-negro"),
         ("multimalla", "ciclo en V, 2 niveles"),
         ("corrección\ndel defecto", "2ª iteración")]


def figura():
    datos = campos()
    lim = float(np.nanpercentile(np.abs(datos[0][1]), 99.0))

    fig = plt.figure(figsize=(18.4, 4.8))
    izq, der, ws = 0.02, 0.935, 0.58
    gs = fig.add_gridspec(1, 4, left=izq, right=der, top=0.845, bottom=0.13,
                          wspace=ws)
    ancho = (der - izq) / (4 + 3 * ws)
    hueco = ws * ancho

    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    for k, (etiqueta, C) in enumerate(datos):
        ax = fig.add_subplot(gs[0, k])
        im = ax.pcolormesh(C, cmap="RdBu_r", shading="auto", norm=norm)
        ax.set_aspect("equal")
        sin_ejes(ax)
        ax.set_title(etiqueta, fontsize=15.5, color=GRIS, pad=11)
        ax.text(0.5, -0.055, f"rms = {np.sqrt((C ** 2).mean()):.3f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=13,
                color="0.35")

    for k, (etq, sub) in enumerate(PASOS):
        g0 = izq + (k + 1) * ancho + k * hueco
        flecha(fig, g0 + 0.13 * hueco, g0 + 0.87 * hueco, 0.52, etq, sub)

    cax = fig.add_axes([0.948, 0.28, 0.008, 0.44])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([-lim, 0.0, lim])
    cb.set_ticklabels(["−", "0", "+"])
    cb.ax.tick_params(labelsize=13, length=0, colors=GRIS)
    cb.outline.set_edgecolor("0.75")
    cb.ax.set_title("error", fontsize=13.5, color=GRIS, pad=10)

    ruta = os.path.join(OUT, "ppt_proyeccion_etapas_sintetico.png")
    fig.savefig(ruta, dpi=DPI)
    plt.close(fig)
    print("->", ruta)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    figura()

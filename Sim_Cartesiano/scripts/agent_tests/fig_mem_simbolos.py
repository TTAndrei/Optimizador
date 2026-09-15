"""
Figura 3.A — Simbolos discretos del laplaciano y de la composicion div-grad.

Analisis local de Fourier sobre malla colocalizada uniforme. Metiendo el modo
p_j = exp(i k x_j) con theta = k h en cada operador:

    exacto            -k^2           ->  h^2 * simbolo = -theta^2
    laplaciano 5 pt   (p_{j-1} - 2 p_j + p_{j+1})/h^2      ->  -2(1-cos theta)
                                                            = -4 sin^2(theta/2)
    div(grad) con gradiente centrado de paso 2h:
                      (p_{j-2} - 2 p_j + p_{j+2})/(4h^2)   ->  -sin^2(theta)

Todo es analitico, no toca el solver.

El punto de la figura: en theta = pi (modo de tablero, longitud de onda 2h) el
gradiente centrado se anula y con el la composicion div-grad, mientras que el
laplaciano compacto alcanza ahi su maximo. El modo par-impar es invisible para el
operador que la proyeccion resuelve de verdad, y por eso el multimalla no lo
reduce y hace falta el filtro del apartado 3.6.5.

Uso:
    .venv/bin/python scripts/agent_tests/fig_mem_simbolos.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "figuras_memoria")
DPI = 200

C_EX = "0.35"
C_LAP = "#1f5fa8"
C_DG = "#c0392b"
C_ERR = "#7a5195"


def main():
    os.makedirs(OUT, exist_ok=True)
    th = np.linspace(1e-4, np.pi, 2000)

    exacto = -th ** 2                      # h^2 * (-k^2)
    lap = -2.0 * (1.0 - np.cos(th))        # -4 sin^2(theta/2)
    dg = -np.sin(th) ** 2                  # gradiente centrado, stencil de 2h
    grad = np.sin(th)                      # h * simbolo del gradiente (modulo)

    err_lap = np.abs(lap - exacto) / np.abs(exacto)
    err_dg = np.abs(dg - exacto) / np.abs(exacto)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.4, 6.6), sharex=True,
                                  gridspec_kw={"height_ratios": [1.35, 1.0]})
    fig.subplots_adjust(left=0.125, right=0.975, top=0.935, bottom=0.095, hspace=0.14)

    # --- panel superior: simbolos -------------------------------------------
    ax.axvspan(0.75 * np.pi, np.pi, color="0.92", zorder=0)
    ax.plot(th, -exacto, color=C_EX, lw=1.6, ls=":", label=r"exacto  $\theta^2$")
    ax.plot(th, -lap, color=C_LAP, lw=2.2,
            label=r"laplaciano de 5 puntos  $4\sin^2(\theta/2)$")
    ax.plot(th, -dg, color=C_DG, lw=2.2,
            label=r"$\mathrm{div}\,\mathrm{grad}$ centrado  $\sin^2\theta$")
    ax.plot(th, grad, color=C_DG, lw=1.3, ls="--", alpha=0.8,
            label=r"gradiente centrado  $|\sin\theta|$")

    ax.plot([np.pi], [4.0], "o", color=C_LAP, ms=6, zorder=5)
    ax.plot([np.pi], [0.0], "o", color=C_DG, ms=6, zorder=5)
    ax.annotate(r"máximo, $4/h^2$", xy=(np.pi, 4.0), xytext=(2.30, 5.6),
                color=C_LAP, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=C_LAP, lw=1.0))
    ax.annotate("cero: el modo de tablero\nno lo ve el operador",
                xy=(np.pi, 0.0), xytext=(1.90, 1.75), color=C_DG, fontsize=10,
                ha="center",
                arrowprops=dict(arrowstyle="->", color=C_DG, lw=1.0))
    ax.axvline(np.pi, color="0.55", lw=0.9, ls="-.")

    ax.set_ylabel(r"$-h^2\,\hat{L}(\theta)$", fontsize=12)
    ax.set_ylim(0, 10.2)
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.25)

    # --- panel inferior: error relativo -------------------------------------
    ax2.axvspan(0.75 * np.pi, np.pi, color="0.92", zorder=0)
    ax2.plot(th, 100 * err_lap, color=C_LAP, lw=2.2,
             label="laplaciano de 5 puntos")
    ax2.plot(th, 100 * err_dg, color=C_DG, lw=2.2,
             label=r"$\mathrm{div}\,\mathrm{grad}$ centrado")
    ax2.axhline(100.0, color=C_ERR, lw=1.0, ls="--")
    ax2.text(np.pi, 175, "error del 100 %: el operador devuelve cero  ",
             color=C_ERR, fontsize=9, ha="right")
    ax2.axvline(np.pi, color="0.55", lw=0.9, ls="-.")

    ax2.set_yscale("log")
    ax2.set_ylim(2e-2, 400)
    ax2.set_ylabel("error relativo del símbolo [%]", fontsize=11)
    ax2.set_xlabel(r"número de onda adimensional  $\theta=k\,h$", fontsize=12)
    ax2.legend(fontsize=9.5, loc="lower right", framealpha=0.95)
    ax2.grid(alpha=0.25, which="both")

    ax2.set_xlim(0, np.pi)
    ax2.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax2.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])

    for a in (ax,):
        a.text(0.875 * np.pi, 0.965, "modos cortos", fontsize=9,
               color="0.4", ha="center", va="top", transform=a.get_xaxis_transform())

    p = os.path.join(OUT, "fig_3A_simbolos_discretos.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)

    for t in (np.pi / 2, 3 * np.pi / 4, np.pi):
        i = int(np.argmin(np.abs(th - t)))
        print(f"theta={t:5.3f}  lap={-lap[i]:7.4f} ({100*err_lap[i]:6.2f}%)   "
              f"divgrad={-dg[i]:7.4f} ({100*err_dg[i]:7.2f}%)")
    print("->", p)


if __name__ == "__main__":
    main()

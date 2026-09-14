"""
Figuras 1.A.1 y 1.A.2 — Curva de sustentacion y polar de resistencia.

Dos figuras independientes, una por fichero:
    fig_1A1_curva_cl_alpha.png   Cl frente a alpha
    fig_1A2_polar.png            Cd frente a Cl

Figura de definiciones para el capitulo introductorio: no sale de ninguna
corrida, es el esqueleto sobre el que se leen despues todos los resultados.

1.A.1, Cl(alpha):
  - tramo lineal con pendiente 2*pi por radian (valor de la teoria de perfil
    delgado), que fija la escala con la que se juzga cualquier pendiente medida
  - alpha de sustentacion nula, desplazado a negativo por la curvatura
  - perdida de linealidad, Cl_max y caida posterior

1.A.2, polar Cd(Cl):
  - forma parabolica Cd = Cd0 + k*Cl^2 en el tramo util
  - Cd minimo y cubeta laminar
  - la recta desde el origen tangente a la polar toca en el punto de eficiencia
    maxima: ese punto es el que optimiza el algoritmo genetico, y no coincide
    ni con Cd minimo ni con Cl maximo

Los numeros son representativos de un perfil de la familia NACA de cuatro
digitos a Re ~ 1e5-1e6; la figura es cualitativa y los ejes se leen como tales.

Uso:
    .venv/bin/python scripts/agent_tests/fig_mem_curvas_polar.py
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

C_LIN = "#1f5fa8"      # tramo lineal / teoria
C_CUR = "#c0392b"      # curva real
C_AUX = "0.45"
C_MARK = "#7a5195"


def curva_cl(alpha_deg, a0=2.0 * np.pi, alpha_L0=-2.0, cl_max=1.35, alpha_st=14.0):
    """Cl(alpha) con tramo lineal, redondeo de la perdida y caida posterior."""
    a = np.radians(alpha_deg - alpha_L0)
    lineal = a0 * a
    # saturacion: el exponente alto mantiene el tramo recto hasta cerca del
    # maximo, que es como se comporta un perfil sin separacion de borde de ataque
    n = 6.0
    cl = lineal / (1.0 + np.abs(lineal / cl_max) ** n) ** (1.0 / n)
    # caida despues de alpha_st
    exceso = np.clip(alpha_deg - alpha_st, 0.0, None)
    caida = 0.030 * exceso ** 1.55
    return cl - caida


def main():
    os.makedirs(OUT, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(6.6, 5.1))
    fig1.subplots_adjust(left=0.125, right=0.975, top=0.925, bottom=0.115)

    # ------------------------------------------------------------------ Cl(alpha)
    alpha_L0 = -2.0
    a0 = 2.0 * np.pi
    al = np.linspace(-8.0, 20.0, 900)
    cl = curva_cl(al, a0=a0, alpha_L0=alpha_L0)

    i_max = int(np.argmax(cl))
    al_max, cl_max_v = al[i_max], cl[i_max]

    # recta de pendiente 2*pi/rad por el alpha de sustentacion nula
    al_rec = np.linspace(-8.0, 13.5, 2)
    cl_rec = a0 * np.radians(al_rec - alpha_L0)

    ax1.axhline(0.0, color="0.2", lw=1.0, zorder=1)
    ax1.axvline(0.0, color="0.2", lw=1.0, zorder=1)
    ax1.axvspan(al_max, 20.0, color="0.93", zorder=0)

    ax1.plot(al_rec, cl_rec, color=C_LIN, lw=1.4, ls="--", zorder=3,
             label=r"tramo lineal, $\mathrm{d}C_l/\mathrm{d}\alpha = 2\pi\ \mathrm{rad}^{-1}$")
    ax1.plot(al, cl, color=C_CUR, lw=2.4, zorder=4, label=r"$C_l(\alpha)$ real")

    # alpha de sustentacion nula
    ax1.plot([alpha_L0], [0.0], marker="o", ms=8, mfc="white", mec=C_CUR,
             mew=1.8, zorder=6)
    ax1.annotate(r"$\alpha_{L=0}$", xy=(alpha_L0, 0.0), xytext=(-6.8, 0.36),
                 fontsize=10, color="0.15", ha="left",
                 arrowprops=dict(arrowstyle="->", color=C_AUX, lw=1.0))

    # Cl_max y perdida
    ax1.plot([al_max], [cl_max_v], marker="o", ms=8, mfc=C_MARK, mec="k",
             mew=0.8, zorder=6)
    ax1.annotate(r"$C_{l,\max}$", xy=(al_max, cl_max_v), xytext=(9.6, 0.62),
                 fontsize=10, color="0.15",
                 arrowprops=dict(arrowstyle="->", color=C_AUX, lw=1.0))
    ax1.text(17.3, 1.62, "pérdida", fontsize=9.5, color="0.35", ha="center",
             va="center")

    # separacion recta-curva
    # marca de la pendiente
    y0, y1 = (a0 * np.radians(0.0 - alpha_L0), a0 * np.radians(4.0 - alpha_L0))
    ax1.plot([0.0, 4.0], [y0, y0], color=C_LIN, lw=0.9, ls=":", zorder=3)
    ax1.plot([4.0, 4.0], [y0, y1], color=C_LIN, lw=0.9, ls=":", zorder=3)
    ax1.text(4.30, 0.5 * (y0 + y1), r"$a_0$", color=C_LIN, fontsize=10,
             va="center")

    ax1.set_xlim(-8.0, 20.0)
    ax1.set_ylim(-0.75, 1.85)
    ax1.set_xlabel(r"ángulo de ataque $\alpha$  [$^\circ$]")
    ax1.set_ylabel(r"coeficiente de sustentación  $C_l$")
    ax1.set_title("Sustentación frente al ángulo de ataque", fontsize=11.5)
    ax1.legend(loc="lower right", fontsize=9.0, framealpha=0.95)
    ax1.grid(alpha=0.18, lw=0.6)

    p1 = os.path.join(OUT, "fig_1A1_curva_cl_alpha.png")
    fig1.savefig(p1, dpi=DPI)
    plt.close(fig1)

    # ------------------------------------------------------------------ polar
    fig2, ax2 = plt.subplots(figsize=(6.6, 5.1))
    fig2.subplots_adjust(left=0.125, right=0.975, top=0.925, bottom=0.115)

    cd0, k = 0.0105, 0.0245
    cl_p = np.linspace(-0.55, 1.35, 700)
    cd_p = cd0 + k * cl_p ** 2
    # cubeta laminar: bajada extra de resistencia en un rango de Cl
    cubeta = 0.0030 * np.exp(-((cl_p - 0.45) / 0.32) ** 2)
    cd_p = cd_p - cubeta
    # engorde al acercarse a la perdida
    cd_p = cd_p + 0.055 * np.clip(cl_p - 1.05, 0.0, None) ** 1.9

    ax2.axhline(0.0, color="0.2", lw=1.0, zorder=1)

    ax2.plot(cd_p, cl_p, color=C_CUR, lw=2.4, zorder=4, label="polar")
    ax2.plot(cd0 + k * cl_p ** 2, cl_p, color=C_LIN, lw=1.3, ls="--", zorder=3,
             label=r"parábola $C_d = C_{d0} + k\,C_l^{2}$")

    # Cd minimo
    i_min = int(np.argmin(cd_p))
    ax2.plot([cd_p[i_min]], [cl_p[i_min]], marker="s", ms=7, mfc="white",
             mec=C_CUR, mew=1.8, zorder=6)
    ax2.annotate(r"$C_{d,\min}$", xy=(cd_p[i_min], cl_p[i_min]),
                 xytext=(0.0195, -0.42), fontsize=10, color="0.15",
                 arrowprops=dict(arrowstyle="->", color=C_AUX, lw=1.0))

    # eficiencia maxima: tangente desde el origen
    razon = cl_p / cd_p
    i_e = int(np.argmax(razon))
    cd_e, cl_e = cd_p[i_e], cl_p[i_e]
    ax2.plot([0.0, 1.85 * cd_e], [0.0, 1.85 * cl_e], color=C_MARK, lw=1.4,
             zorder=3)
    ax2.plot([cd_e], [cl_e], marker="o", ms=8, mfc=C_MARK, mec="k", mew=0.8,
             zorder=6)
    ax2.annotate(r"$(C_l/C_d)_{\max}$", xy=(cd_e, cl_e), xytext=(0.0265, 0.06),
                 fontsize=10, color="0.15",
                 arrowprops=dict(arrowstyle="->", color=C_AUX, lw=1.0))

    ax2.set_xlim(0.0, 0.068)
    ax2.set_ylim(-0.70, 1.60)
    ax2.set_xlabel(r"coeficiente de resistencia  $C_d$")
    ax2.set_ylabel(r"coeficiente de sustentación  $C_l$")
    ax2.set_title("Polar de resistencia", fontsize=11.5)
    ax2.legend(loc="lower right", fontsize=9.0, framealpha=0.95)
    ax2.grid(alpha=0.18, lw=0.6)

    p2 = os.path.join(OUT, "fig_1A2_polar.png")
    fig2.savefig(p2, dpi=DPI)
    plt.close(fig2)

    print("escrito:", p1)
    print("escrito:", p2)
    print(f"  alpha_L0 = {alpha_L0:.1f} deg, Cl_max = {cl_max_v:.3f} en alpha = {al_max:.1f} deg")
    print(f"  (Cl/Cd)_max = {razon[i_e]:.1f} en Cl = {cl_e:.2f}, Cd = {cd_e:.4f}")


if __name__ == "__main__":
    main()

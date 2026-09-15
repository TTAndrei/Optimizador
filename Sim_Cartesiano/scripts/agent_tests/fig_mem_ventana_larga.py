"""
Figura 4.C — Cl y Cd de la ventana larga (t* = 41.5) en la malla de dx = 0.002.

Prolongacion del punto alpha=4 del perfil ganador hasta t* = 41.5, mas del doble
del presupuesto t* ~ 20 del resto del capitulo. Sirve para acotar la
sensibilidad a la ventana temporal: si las medias de cola de las dos ventanas
coinciden, el presupuesto corto no sesga la polar.

Los datos son los de `results/asintotico_alpha4_domC/series.npz`, la corrida ya
hecha; este script solo dibuja.

Se recorta el arranque (t* < 2). El primer muestreo del impulso inicial vale
Cl = 19.9 y Cd = 5.53 —dos ordenes de magnitud por encima del regimen— y con el
dentro las dos curvas quedan aplastadas contra el cero, que es lo que le pasa a
`results/asintotico_alpha4_domC/evolucion_temporal.png`.

Uso:
    .venv/bin/python scripts/agent_tests/fig_mem_ventana_larga.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "figuras_memoria")
SERIE = os.path.join(ROOT, "results/asintotico_alpha4_domC/series.npz")
DPI = 200

T_MIN = 2.0                     # recorte del transitorio de arranque
VENT = ((15.0, 20.0), (36.5, 41.5))   # ventanas de cola comparadas
C_CL = "#1f5fa8"
C_CD = "#c0392b"


def main():
    d = np.load(SERIE)
    t, cl, cd = d["t"], d["cl"], d["cd"]

    m = t >= T_MIN
    print(f"  muestras: {len(t)} -> {m.sum()} tras recortar t* < {T_MIN}")
    print(f"  descartado: Cl_max={cl[~m].max():.2f}  Cd_max={cd[~m].max():.2f}"
          f"  (impulso inicial en t*={t[0]:.3f})")

    est = {}
    for (a, b) in VENT:
        s = (t >= a) & (t <= b)
        est[(a, b)] = (cl[s].mean(), cd[s].mean(), int(s.sum()))
        print(f"  ventana {a}-{b}: Cl={cl[s].mean():.4f}  Cd={cd[s].mean():.5f}"
              f"  L/D={cl[s].mean()/cd[s].mean():.3f}  n={s.sum()}")
    (cl1, cd1, _), (cl2, cd2, _) = est[VENT[0]], est[VENT[1]]
    dcl, dcd = (cl2 - cl1) / cl1, (cd2 - cd1) / cd1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.6, 5.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1]})
    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.105, hspace=0.09)

    for ax, y, color, lab, (m1, m2), dif in (
            (ax1, cl, C_CL, r"$C_l$", (cl1, cl2), dcl),
            (ax2, cd, C_CD, r"$C_d$", (cd1, cd2), dcd)):
        for (a, b), med in zip(VENT, (m1, m2)):
            ax.axvspan(a, b, color="0.90", zorder=0)
            ax.plot([a, b], [med, med], color="0.15", lw=2.2, zorder=5,
                    solid_capstyle="butt")
        ax.plot(t[m], y[m], color=color, lw=0.8, alpha=0.85, zorder=3)
        ax.axvline(20.0, color="0.45", ls=":", lw=1.1, zorder=2)
        ax.set_ylabel(lab, fontsize=13)
        ax.grid(alpha=0.18, lw=0.6)
        ax.set_xlim(T_MIN, t.max())
        ax.annotate(f"{m1:.4f}" if m1 > 0.1 else f"{m1:.5f}",
                    xy=(0.5 * (VENT[0][0] + VENT[0][1]), m1), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=9.5,
                    color="0.15", bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                            ec="none", alpha=0.85))
        ax.annotate((f"{m2:.4f}" if m2 > 0.1 else f"{m2:.5f}") + f"  ({dif*100:+.1f} %)",
                    xy=(min(VENT[1][1], t.max()), m2), xytext=(-4, 12),
                    annotation_clip=False,
                    textcoords="offset points", ha="right", fontsize=9.5,
                    color="0.15", bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                            ec="none", alpha=0.85))

    ax1.set_ylim(0.52, 0.80)
    ax2.set_ylim(0.011, 0.036)
    ax2.set_xlabel(r"$t^{*} = t\,U_\infty/c$")
    ax1.set_title(r"Perfil del punto de diseño, $\alpha = 4^\circ$, $dx = 0.002$ — "
                  r"ventana larga hasta $t^{*} = 41.5$", fontsize=11)

    p = os.path.join(OUT, "fig_4C_ventana_larga.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("escrito:", p)


if __name__ == "__main__":
    main()

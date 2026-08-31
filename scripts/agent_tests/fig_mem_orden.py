"""
Figura 4.B — Orden aparente con signo frente al angulo de ataque.

Sale directamente de resultados_finales/richardson/tabla_gci.csv, columna
`p_observado_con_signo`, que es la que expone el signo que la formula de Celik
devuelve en valor absoluto. Un punto por perfil, magnitud y angulo, con la banda
admisible p en [1, 4] sombreada.

Los casos sin orden con signo (columna a None) son las series no monotonas: los
dos saltos entre mallas tienen signo opuesto y el orden no esta definido. Se
dibujan en una banda propia bajo el eje para que no desaparezcan del recuento.

Resume las tablas 4.6 y 4.12: los fallos se concentran en la resistencia y en los
extremos del barrido.

Uso:
    .venv/bin/python scripts/agent_tests/fig_mem_orden.py
"""
from __future__ import annotations

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV = os.path.join(ROOT, "resultados_finales", "richardson", "tabla_gci.csv")
OUT = os.path.join(ROOT, "figuras_memoria")
DPI = 200

PERFIL = {"ganador_ag": "perfil optimizado", "naca0012": "NACA 0012"}
COLOR = {"cl": "#1f5fa8", "cd": "#c0392b"}
MARCA = {"cl": "o", "cd": "s"}
ETIQ = {"cl": r"$C_l$", "cd": r"$C_d$"}
Y_NM = -6.6      # fila de las series no monotonas
Y_LIM = (-7.6, 5.2)


def lee():
    d = {}
    for r in csv.DictReader(open(CSV)):
        if r["magnitud"] not in ("cl", "cd"):
            continue
        p = r["p_observado_con_signo"]
        d.setdefault((r["perfil"], r["magnitud"]), []).append(
            (float(r["alpha"]), None if p == "None" else float(p)))
    for v in d.values():
        v.sort()
    return d


def panel(ax, d, perfil, titulo):
    ax.axhspan(1.0, 4.0, color="#c8e6c9", alpha=0.55, zorder=0)
    ax.axhline(2.0, color="0.45", lw=1.0, ls="--", zorder=1)
    ax.axhline(0.0, color="0.2", lw=1.0, zorder=1)
    ax.axhspan(Y_LIM[0], Y_NM + 0.85, color="0.93", zorder=0)

    for mag in ("cl", "cd"):
        serie = d[(perfil, mag)]
        a = np.array([s[0] for s in serie])
        p = np.array([np.nan if s[1] is None else s[1] for s in serie])
        ok = ~np.isnan(p)
        ax.plot(a[ok], p[ok], color=COLOR[mag], lw=1.2, alpha=0.55, zorder=2)
        dentro = ok & (p >= 1.0) & (p <= 4.0)
        ax.scatter(a[dentro], p[dentro], marker=MARCA[mag], s=64,
                   facecolor=COLOR[mag], edgecolor="k", linewidth=0.8, zorder=4)
        ax.scatter(a[ok & ~dentro], p[ok & ~dentro], marker=MARCA[mag], s=52,
                   facecolor="white", edgecolor=COLOR[mag], linewidth=1.4, zorder=3)
        # no monotonas: sin orden definido
        ax.scatter(a[~ok], np.full((~ok).sum(), Y_NM), marker="x", s=52,
                   color=COLOR[mag], linewidth=1.6, zorder=4)

    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(*Y_LIM)
    ax.set_xticks(range(9))
    ax.set_yticks([-6.6, -4, -2, 0, 1, 2, 3, 4, 5])
    ax.set_yticklabels(["no mon.", "-4", "-2", "0", "1", "2", "3", "4", "5"])
    ax.set_xlabel(r"ángulo de ataque  $\alpha$  [°]", fontsize=11)
    ax.set_title(titulo, fontsize=12, pad=6)
    ax.grid(alpha=0.22)


def main():
    os.makedirs(OUT, exist_ok=True)
    d = lee()

    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.9), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.755, top=0.86, bottom=0.125, wspace=0.09)

    panel(axs[0], d, "ganador_ag", PERFIL["ganador_ag"])
    panel(axs[1], d, "naca0012", PERFIL["naca0012"])
    axs[0].set_ylabel(r"orden aparente con signo  $p$", fontsize=12)

    leg = [
        Line2D([], [], color="#1f5fa8", marker="o", ls="-", lw=1.2, ms=8,
               mec="k", label=r"$C_l$"),
        Line2D([], [], color="#c0392b", marker="s", ls="-", lw=1.2, ms=8,
               mec="k", label=r"$C_d$"),
        Line2D([], [], color="0.25", marker="o", ls="none", ms=8, mec="0.25",
               mfc="0.25", label=r"$p\in[1,4]$: utilizable"),
        Line2D([], [], color="0.25", marker="o", ls="none", ms=8, mec="0.25",
               mfc="white", mew=1.4, label="fuera de banda"),
        Line2D([], [], color="0.25", marker="x", ls="none", ms=8, mew=1.6,
               label="serie no monótona"),
        Line2D([], [], color="0.45", ls="--", lw=1.0,
               label="orden nominal del esquema"),
    ]
    fig.legend(handles=leg, loc="center left", bbox_to_anchor=(0.765, 0.52),
               fontsize=10, frameon=True, framealpha=0.95)

    n_ok = sum(1 for k, v in d.items() for _, p in v
               if p is not None and 1.0 <= p <= 4.0)
    n_tot = sum(len(v) for v in d.values())
    fig.suptitle(f"Orden aparente de convergencia: utilizable en {n_ok} de "
                 f"{n_tot} casos de $C_l$ y $C_d$", fontsize=13, y=0.965)

    p = os.path.join(OUT, "fig_4B_orden_aparente.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)

    for mag in ("cl", "cd"):
        ok = sum(1 for k, v in d.items() if k[1] == mag
                 for _, q in v if q is not None and 1.0 <= q <= 4.0)
        neg = sum(1 for k, v in d.items() if k[1] == mag
                  for _, q in v if q is not None and q < 0)
        nm = sum(1 for k, v in d.items() if k[1] == mag for _, q in v if q is None)
        print(f"{mag}: utilizables {ok}/18   p<0 {neg}   no monotonas {nm}")
    print("->", p)


if __name__ == "__main__":
    main()

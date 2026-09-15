"""
Tiras horizontales para diapositiva: la proyeccion, paso a paso. Sin titulo.

Tres opciones, misma idea (u* con divergencia -> Poisson multimalla -> p ->
correccion -> u solenoidal) y distinto grado de literalidad:

  A  campos reales del NACA 0012 a dx=0.002 (npz de fig_mem_pasos_fraccionados)
  B  diagrama del ciclo en V de dos niveles, con lo que hace cada tramo
  C  glifos sinteticos: flechas que divergen, blob de presion, flechas paralelas

Uso:
    .venv/bin/python scripts/agent_tests/fig_ppt_proyeccion.py
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from fig_mem_pasos_fraccionados import remuestrea, VENTANA

OUT = os.path.join(ROOT, "figuras_memoria")
NPZ = os.path.join(OUT, "datos_pasos_fraccionados.npz")
DPI = 240

AZUL = "#1f5fa8"
GRIS = "#3d4450"


def flecha(fig, x0, x1, y, texto, sub=None):
    """Flecha entre dos paneles, en coordenadas de figura, con el metodo encima."""
    fig.patches.append(FancyArrowPatch(
        (x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>",
        mutation_scale=26, lw=2.4, color=GRIS, shrinkA=0, shrinkB=0))
    fig.text((x0 + x1) / 2, y + 0.055, texto, ha="center", va="bottom",
             fontsize=13.5, color=GRIS, weight="bold")
    if sub:
        fig.text((x0 + x1) / 2, y - 0.075, sub, ha="center", va="top",
                 fontsize=11.5, color="0.35")


def sin_ejes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.75")
        s.set_linewidth(0.9)


# ----------------------------------------------------------------------
# A — campos reales
# ----------------------------------------------------------------------
def opcion_A():
    d = dict(np.load(NPZ))
    X, Y = d["X"], d["Y"]
    solid = d["solid"].astype(bool)
    jj = np.where(solid.any(axis=0))[0]
    ii = np.where(solid.any(axis=1))[0]
    xc, cuerda = X[jj[0]], X[jj[-1]] - X[jj[0]]
    yc = 0.5 * (Y[ii[0]] + Y[ii[-1]])
    x = xc + np.linspace(-0.35, 1.45, 520) * cuerda
    y = yc + np.linspace(-0.55, 0.55, 340) * cuerda
    xr, yr = (x - xc) / cuerda, (y - yc) / cuerda
    fondo = np.where(solid, np.nan, 1.0)

    cam = lambda k: remuestrea(X, Y, d[k] * fondo, x, y)
    D0, P, D1 = cam("div_difusion"), cam("p_proyeccion"), cam("div_proyeccion")
    S = np.nan_to_num(remuestrea(X, Y, solid.astype(float), x, y))

    lim_d = float(np.nanpercentile(np.abs(D0[np.isfinite(D0)]), 99.0))
    lim_p = float(np.nanpercentile(np.abs(P[np.isfinite(P)]), 99.0))

    fig = plt.figure(figsize=(17.0, 4.4))
    izq, der, ws = 0.02, 0.985, 0.46
    gs = fig.add_gridspec(1, 3, left=izq, right=der, top=0.80, bottom=0.06,
                          wspace=ws)
    ancho = (der - izq) / (3 + 2 * ws)
    hueco = ws * ancho
    datos = [(D0, lim_d, "RdBu_r", r"$\nabla\!\cdot\!\mathbf{u}^{*}\neq 0$"),
             (P, lim_p, "RdBu_r", r"$p$"),
             (D1, lim_d, "RdBu_r", r"$\nabla\!\cdot\!\mathbf{u}^{n+1}\approx 0$")]
    for k, (C, lim, cmap, etiqueta) in enumerate(datos):
        ax = fig.add_subplot(gs[0, k])
        ax.pcolormesh(xr, yr, C, cmap=cmap, shading="auto",
                      norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
        ax.contour(xr, yr, S, levels=[0.5, 1.5], colors="k", linewidths=1.1)
        ax.set_aspect("equal")
        sin_ejes(ax)
        ax.set_title(etiqueta, fontsize=17, pad=10, color=GRIS)

    for k, (etq, sub) in enumerate((
            ("multimalla", "ciclo en V, 2 niveles"),
            ("corrección",
             r"$\mathbf{u}^{n+1}=\mathbf{u}^{*}-\frac{\Delta t}{\rho}\nabla p$"))):
        g0 = izq + (k + 1) * ancho + k * hueco
        flecha(fig, g0 + 0.22 * hueco, g0 + 0.78 * hueco, 0.44, etq, sub)
    ruta = os.path.join(OUT, "ppt_proyeccion_A_campos.png")
    fig.savefig(ruta, dpi=DPI)
    plt.close(fig)
    print("->", ruta)


# ----------------------------------------------------------------------
# B — ciclo en V
# ----------------------------------------------------------------------
def opcion_B():
    fig, ax = plt.subplots(figsize=(17.0, 5.6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)

    # nodos: (x, y, etiqueta del nivel)
    nodos = [(1.0, 2.6), (3.1, 2.6), (5.0, 1.05), (6.9, 2.6), (9.0, 2.6)]
    etiquetas = ["suavizar\n(GS rojo-negro)", "restringir\nel residuo",
                 "resolver en\nmalla gruesa", "prolongar\ny corregir",
                 "suavizar\nde nuevo"]

    for i in range(len(nodos) - 1):
        (x0, y0), (x1, y1) = nodos[i], nodos[i + 1]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=22, lw=2.6, color=GRIS,
                                     shrinkA=48, shrinkB=48))
    for i, (((x, y), et)) in enumerate(zip(nodos, etiquetas), start=1):
        ax.add_patch(plt.Circle((x, y), 0.36, facecolor="white",
                                edgecolor=AZUL, lw=2.6, zorder=3))
        ax.text(x, y, str(i), ha="center", va="center", fontsize=17,
                color=AZUL, weight="bold", zorder=4)
        ax.text(x, y - 0.62, et, ha="center", va="top", fontsize=13,
                color=GRIS, linespacing=1.25)

    # bandas de nivel
    for yy, txt, col in ((2.6, "malla fina", "0.45"), (1.05, "malla gruesa", "0.45")):
        ax.plot([0.35, 9.65], [yy, yy], lw=0.9, ls=":", color="0.8", zorder=0)
        ax.text(0.30, yy, txt, ha="right", va="center", fontsize=12.5, color=col,
                rotation=90)

    # entrada y salida
    ax.text(1.0, 3.15, r"$\nabla^{2}p=\frac{\rho}{\Delta t}\nabla\!\cdot\!\mathbf{u}^{*}$",
            ha="center", va="bottom", fontsize=15, color=GRIS)
    ax.text(9.0, 3.15, r"$p$", ha="center", va="bottom", fontsize=17, color=GRIS)

    ruta = os.path.join(OUT, "ppt_proyeccion_B_ciclo_v.png")
    fig.savefig(ruta, dpi=DPI)
    plt.close(fig)
    print("->", ruta)


# ----------------------------------------------------------------------
# C — glifos sinteticos
# ----------------------------------------------------------------------
def opcion_C():
    n = 15
    g = np.linspace(-1, 1, n)
    XX, YY = np.meshgrid(g, g)
    gx = np.linspace(-1, 1, 260)
    PX, PY = np.meshgrid(gx, gx)

    def radial(X, Y, x0, y0, signo):
        dx, dy = X - x0, Y - y0
        r2 = dx * dx + dy * dy + 0.05
        return signo * dx / r2, signo * dy / r2

    def giro(X, Y, x0, y0, signo):
        dx, dy = X - x0, Y - y0
        r2 = dx * dx + dy * dy + 0.06
        return -signo * dy / r2, signo * dx / r2

    # parte solenoidal (dos torbellinos): sobrevive a la proyeccion
    Us, Vs = [a + b for a, b in zip(giro(XX, YY, -0.35, 0.25, 1.0),
                                    giro(XX, YY, 0.45, -0.30, -0.8))]
    # parte de gradiente (fuente + sumidero): la que la proyeccion elimina
    Ug, Vg = [a + b for a, b in zip(radial(XX, YY, -0.35, 0.25, 0.9),
                                    radial(XX, YY, 0.45, -0.30, -0.9))]
    U0, V0 = Us + Ug, Vs + Vg

    # el potencial de esa parte es, salvo constantes, el campo de presion
    P = (0.9 * np.log(np.hypot(PX + 0.35, PY - 0.25) + 1e-6)
         - 0.9 * np.log(np.hypot(PX - 0.45, PY + 0.30) + 1e-6))

    fig = plt.figure(figsize=(17.0, 4.8))
    izq, der, ws = 0.03, 0.98, 0.50
    gs = fig.add_gridspec(1, 3, left=izq, right=der, top=0.80, bottom=0.08,
                          wspace=ws)
    ancho = (der - izq) / (3 + 2 * ws)
    hueco = ws * ancho

    # fondo: la divergencia del propio glifo, misma escala en el panel 1 y en el
    # 3, que es lo que hace obvio de un vistazo lo que ha quitado la proyeccion
    def div_de(fx, fy):
        h = gx[1] - gx[0]
        return np.gradient(fx, h, axis=1) + np.gradient(fy, h, axis=0)

    Ug_f, Vg_f = [a + b for a, b in zip(radial(PX, PY, -0.35, 0.25, 0.9),
                                        radial(PX, PY, 0.45, -0.30, -0.9))]
    Dv = div_de(Ug_f, Vg_f)
    lim_dv = float(np.nanpercentile(np.abs(Dv), 97))

    ax = fig.add_subplot(gs[0, 0])
    ax.pcolormesh(PX, PY, Dv, cmap="RdBu_r", shading="auto", alpha=0.85,
                  norm=TwoSlopeNorm(vmin=-lim_dv, vcenter=0.0, vmax=lim_dv))
    ax.quiver(XX, YY, U0, V0, color=GRIS, width=0.008, scale=26)
    ax.set_title(r"$\mathbf{u}^{*}$   con fuentes y sumideros", fontsize=15,
                 color=GRIS, pad=12)

    ax = fig.add_subplot(gs[0, 1])
    lim = float(np.nanpercentile(np.abs(P), 98))
    ax.pcolormesh(PX, PY, P, cmap="RdBu_r", shading="auto",
                  norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
    ax.contour(PX, PY, P, levels=9, colors="k", linewidths=0.6, alpha=0.4)
    ax.set_title(r"$p$   campo de presión", fontsize=15, color=GRIS, pad=12)

    ax = fig.add_subplot(gs[0, 2])
    ax.pcolormesh(PX, PY, np.zeros_like(Dv), cmap="RdBu_r", shading="auto",
                  norm=TwoSlopeNorm(vmin=-lim_dv, vcenter=0.0, vmax=lim_dv))
    ax.quiver(XX, YY, Us, Vs, color=AZUL, width=0.008, scale=26)
    ax.set_title(r"$\mathbf{u}^{n+1}$   sin divergencia", fontsize=15,
                 color=GRIS, pad=12)

    for ax in fig.axes:
        ax.set_aspect("equal")
        ax.set_xlim(-1.12, 1.12)
        ax.set_ylim(-1.12, 1.12)
        sin_ejes(ax)

    for k, (etq, sub) in enumerate((
            ("multimalla", "ciclo en V, 2 niveles"),
            ("corrección", r"$\mathbf{u}^{*}-\frac{\Delta t}{\rho}\nabla p$"))):
        g0 = izq + (k + 1) * ancho + k * hueco
        flecha(fig, g0 + 0.20 * hueco, g0 + 0.80 * hueco, 0.44, etq, sub)

    ruta = os.path.join(OUT, "ppt_proyeccion_C_glifos.png")
    fig.savefig(ruta, dpi=DPI)
    plt.close(fig)
    print("->", ruta)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    opcion_A()
    opcion_B()
    opcion_C()

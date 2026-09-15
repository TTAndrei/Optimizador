"""Figuras finales del Richardson, una por perfil y magnitud.

    .venv/bin/python scripts/agent_tests/rf_richardson_figuras.py

Genera 6 PNG en resultados_finales/richardson/figuras/:
    richardson_<perfil>_<cl|cd|ld>.png

Cada una lleva las tres mallas de la terna principal mas la extrapolacion robusta.
robusta. Los puntos con marcador hueco son aquellos en los que el orden
observado no era utilizable y se forzo p=2 (ver extrapola_robusto en
verificacion_numerica.py): ahi el valor es indicativo, no una medida del error
de discretizacion.
"""
import json
import os
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verificacion_numerica as vn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "resultados_finales")
FIG = os.path.join(OUT, "richardson", "figuras")

TERNA = (0.002, 0.004, 0.008)          # fina, media, gruesa
PERFILES = {"ganador_ag": "Ganador de la optimizacion",
            "naca0012": "NACA 0012 sharp"}
ANGULOS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
MAGS = (("cl", r"$C_L$", "Coeficiente de sustentacion"),
        ("cd", r"$C_D$", "Coeficiente de resistencia"),
        ("ld", r"$L/D$", "Eficiencia aerodinamica"))
COLOR = {0.008: "#e67e22", 0.004: "#8e44ad", 0.002: "#1f4e79"}
MARCA = {0.008: "D", 0.004: "^", 0.002: "s"}


def suave(xs, ys, n=400):
    """Curva suavizada para el trazo. PCHIP y no un spline cubico normal: preserva
    la monotonia de los tramos y no inventa oscilaciones ni sobrepasos entre
    puntos, que en una polar se leerian como fisica que no esta en los datos."""
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return x[ok], y[ok]
    xf = np.linspace(x[ok].min(), x[ok].max(), n)
    return xf, PchipInterpolator(x[ok], y[ok])(xf)


def carga():
    return {(p, dx): json.load(open(os.path.join(
        OUT, p, "metricas", "polar_dx%.4f.json" % dx)))
        for p in PERFILES for dx in TERNA}


def extrapolados(d, p):
    """Por angulo: (cl, cd, ld) extrapolados y si cada uno es fiable."""
    r = {}
    for a in ANGULOS:
        k = "%.1f" % a
        cl = [d[(p, dx)][k]["cl"] for dx in TERNA]
        cd = [d[(p, dx)][k]["cd"] for dx in TERNA]
        cle, _, _, okl = vn.extrapola_robusto(*cl, *TERNA)
        cde, _, _, okd = vn.extrapola_robusto(*cd, *TERNA, log=min(cd) > 0)
        # El L/D se reconstruye desde Cl y Cd: extrapolar el cociente hereda las
        # dos patologias y revienta si el Cd extrapolado se acerca a cero.
        r[a] = {"cl": (cle, okl), "cd": (cde, okd),
                "ld": (cle / cde if cde else np.nan, okl and okd)}
    return r


def una(d, p, mag, etiqueta, titulo):
    ext = extrapolados(d, p)
    fig, ax = plt.subplots(figsize=(7.6, 5.2), layout="constrained")

    for dx in (0.008, 0.004, 0.002):
        y = [d[(p, dx)]["%.1f" % a][mag] for a in ANGULOS]
        xs, ys = suave(ANGULOS, y)
        ax.plot(xs, ys, color=COLOR[dx], lw=1.5, alpha=.9,
                label="malla dx = %g" % dx)
        ax.plot(ANGULOS, y, color=COLOR[dx], marker=MARCA[dx], ms=5.5,
                ls="none", alpha=.9)

    y = [ext[a][mag][0] for a in ANGULOS]
    xs, ys = suave(ANGULOS, y)
    ax.plot(xs, ys, color="#c0392b", lw=2.6, ls="--",
            label="Richardson (extrapolado)", zorder=5)
    ax.plot(ANGULOS, y, color="#c0392b", marker="o", ms=7, ls="none", zorder=6)

    if mag == "cd":
        ax.axhline(0, color="k", lw=.8, ls=":")

    ax.set_xlabel(r"$\alpha$ [$^\circ$]")
    ax.set_ylabel(etiqueta)
    ax.set_title("%s — %s\n%s" % (PERFILES[p], titulo,
                                 "terna dx = 0.008 / 0.004 / 0.002, $r=2$"),
                 fontsize=11)
    ax.grid(alpha=.28)
    ax.set_xticks(ANGULOS)
    ax.legend(fontsize=8.5)

    os.makedirs(FIG, exist_ok=True)
    ruta = os.path.join(FIG, "richardson_%s_%s.png" % (p, mag))
    fig.savefig(ruta, dpi=190)
    plt.close(fig)
    return ruta


def main():
    d = carga()
    for p in PERFILES:
        for mag, etq, tit in MAGS:
            print("  ->", os.path.relpath(una(d, p, mag, etq, tit), ROOT))


if __name__ == "__main__":
    main()

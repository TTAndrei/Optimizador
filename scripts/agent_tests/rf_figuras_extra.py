"""Figuras que faltaban en el paquete final.

    .venv/bin/python scripts/agent_tests/rf_figuras_extra.py

  1. cp_superficie_*.png   distribucion de Cp sobre el perfil (extrados/intrados)
  2. polar_resistencia.png Cl frente a Cd, los dos perfiles
  3. malla_*.png           la malla variable, dominio completo y zoom
  4. comparativa_extrapolada.png   los dos perfiles con Richardson robusto
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verificacion_numerica as vn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "resultados_finales")
TERNA = (0.002, 0.004, 0.008)
PERF = {"ganador_ag": ("Ganador de la optimizacion", "#c0392b"),
        "naca0012": ("NACA 0012 sharp", "#1f4e79")}
ANG = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
DPI = 190


def campo(p, dx, a):
    f = os.path.join(OUT, p, "campos", "dx%.4f_a%04.1f_campo.npz" % (dx, a))
    return np.load(f) if os.path.exists(f) else None


def cp_superficie(p, dx, a):
    """Cp en las celdas de fluido pegadas al solido.

    Se toma el contorno de la mascara en vez de la geometria del .dat: asi no
    hay que reproducir la rotacion ni el desplazamiento del perfil dentro del
    dominio, y se lee exactamente lo que ve el solver.
    """
    d = campo(p, dx, a)
    if d is None:
        return None
    x, y, pr, sol = d["x"], d["y"], d["p"], d["solid"].astype(bool)
    # celda de fluido con al menos un vecino solido
    vec = np.zeros_like(sol)
    vec[1:-1, 1:-1] = (sol[:-2, 1:-1] | sol[2:, 1:-1] |
                       sol[1:-1, :-2] | sol[1:-1, 2:])
    borde = vec & ~sol
    jj, ii = np.nonzero(borde)
    if len(ii) < 20:
        return None
    # Una celda por columna y superficie: la mascara es escalonada y deja varias
    # celdas de borde en la misma columna. Quedarse con la mas alta y la mas baja
    # separa extrados de intrados sin depender de la linea de cuerda, que con el
    # perfil rotado y con curvatura no reparte bien.
    p_inf = float(np.median(pr[:, 2:8]))          # lejos, banda de entrada
    col = {}
    for j, i in zip(jj, ii):
        lo, hi = col.get(i, (j, j))
        col[i] = (min(lo, j), max(hi, j))
    cols = np.array(sorted(col))
    c0, c1 = x[cols[0]], x[cols[-1]]
    xc = (x[cols] - c0) / (c1 - c0)
    cp_lo = 2.0 * (np.array([pr[col[i][0], i] for i in cols]) - p_inf)
    cp_hi = 2.0 * (np.array([pr[col[i][1], i] for i in cols]) - p_inf)
    return xc, cp_hi, cp_lo


def fig_cp(a=4.0, dx=0.002):
    """Una figura por perfil: comparar Cp de dos geometrias distintas en el mismo
    eje no aporta nada y encoge cada curva a la mitad de ancho."""
    rutas = []
    for p, (nom, col) in PERF.items():
        r = cp_superficie(p, dx, a)
        if r is None:
            continue
        xc, cp_hi, cp_lo = r
        fig, ax = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
        ax.plot(xc, cp_hi, color="#c0392b", lw=1.4, label="extrados")
        ax.plot(xc, cp_lo, color="#1f4e79", lw=1.4, label="intrados")
        ax.invert_yaxis()
        ax.set_xlabel("$x/c$")
        ax.set_ylabel("$C_p$")
        ax.set_title("%s — distribucion de presion (Re = $10^5$)\n"
                     "$\\alpha$ = %g$^\\circ$, dx = %g" % (nom, a, dx), fontsize=11)
        ax.grid(alpha=.25)
        ax.legend(fontsize=9)
        ax.axhline(0, color="k", lw=.7, ls=":")
        r = os.path.join(OUT, "comparativa", "cp_superficie_%s_a%04.1f.png" % (p, a))
        fig.savefig(r, dpi=DPI)
        plt.close(fig)
        rutas.append(r)
    return rutas or None


def fig_cp_par(a=4.0, dx=0.002):
    """Los dos perfiles en una sola figura, un panel cada uno."""
    fig, axs = plt.subplots(1, 2, figsize=(13.2, 5.2), layout="constrained")
    hecho = False
    for ax, (p, (nom, col)) in zip(axs, PERF.items()):
        r = cp_superficie(p, dx, a)
        if r is None:
            continue
        hecho = True
        xc, cp_hi, cp_lo = r
        ax.plot(xc, cp_hi, color="#c0392b", lw=1.4, label="extrados")
        ax.plot(xc, cp_lo, color="#1f4e79", lw=1.4, label="intrados")
        ax.invert_yaxis()
        ax.set_xlabel("$x/c$")
        ax.set_ylabel("$C_p$")
        ax.set_title(nom, fontsize=11)
        ax.grid(alpha=.25)
        ax.legend(fontsize=9)
        ax.axhline(0, color="k", lw=.7, ls=":")
    if not hecho:
        plt.close(fig)
        return None
    # mismo eje vertical en los dos paneles: si no, la comparacion visual enganya
    lo = min(ax.get_ylim()[1] for ax in axs)
    hi = max(ax.get_ylim()[0] for ax in axs)
    for ax in axs:
        ax.set_ylim(hi, lo)
    fig.suptitle("Distribucion de presion sobre el perfil — "
                 r"$\alpha$ = %g$^\circ$, dx = %g, Re = $10^5$" % (a, dx),
                 fontsize=12)
    r = os.path.join(OUT, "comparativa", "cp_superficie_a%04.1f.png" % a)
    fig.savefig(r, dpi=DPI)
    plt.close(fig)
    return r


def fig_polar_resistencia():
    d = {p: json.load(open(os.path.join(OUT, p, "metricas", "polar_dx0.0020.json")))
         for p in PERF}
    fig, ax = plt.subplots(figsize=(7.2, 5.6), layout="constrained")
    for p, (nom, col) in PERF.items():
        cd = [d[p]["%.1f" % a]["cd"] for a in ANG]
        cl = [d[p]["%.1f" % a]["cl"] for a in ANG]
        ax.plot(cd, cl, color=col, marker="o", ms=5.5, lw=1.6, label=nom)
        for a, xx, yy in zip(ANG, cd, cl):
            if a in (0.0, 4.0, 8.0):
                ax.annotate(r"%g$^\circ$" % a, (xx, yy), textcoords="offset points",
                            xytext=(6, -4), fontsize=8, color=col)
    ax.set_xlabel("$C_D$")
    ax.set_ylabel("$C_L$")
    ax.set_title("Polar de resistencia — malla dx = 0.002, Re = $10^5$\n"
                 r"etiquetas: $\alpha$ en grados", fontsize=11)
    ax.grid(alpha=.25)
    ax.legend(fontsize=9)
    r = os.path.join(OUT, "comparativa", "polar_resistencia.png")
    fig.savefig(r, dpi=DPI)
    plt.close(fig)
    return r


def fig_malla(p="ganador_ag", dx=0.002, a=4.0):
    d = campo(p, dx, a)
    if d is None:
        return None
    x, y, sol = d["x"], d["y"], d["solid"].astype(bool)
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2), layout="constrained")
    for ax, zoom in zip(axs, (False, True)):
        if zoom:
            # Ventana corta sobre el borde de ataque: con la cuerda entera a
            # dx=0.002 son ~900 lineas y el dibujo se vuelve un bloque gris.
            jc = np.nonzero(sol.any(axis=1))[0]
            ic = np.nonzero(sol.any(axis=0))[0]
            x0 = x[ic[0]] - 0.03
            yc = 0.5 * (y[jc[0]] + y[jc[-1]])
            m = (x > x0) & (x < x0 + 0.30)
            n = (y > yc - 0.085) & (y < yc + 0.085)
            xx, yy = x[m], y[n]
            paso = 1
        else:
            xx, yy = x, y
            paso = max(1, len(x) // 160)
        for v in xx[::paso]:
            ax.axvline(v, color="#7f8c8d", lw=.3, alpha=.65)
        for v in yy[::paso]:
            ax.axhline(v, color="#7f8c8d", lw=.3, alpha=.65)
        ax.contourf(x, y, sol.astype(float), levels=[.5, 1.5], colors=["#c0392b"])
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_aspect("equal")
        ax.set_xlabel("$x/c$")
        ax.set_ylabel("$y/c$")
        ax.set_title(("zoom al borde de ataque (todas las lineas)" if zoom
                      else "dominio completo (1 de cada %d lineas)" % paso), fontsize=10)
    fig.suptitle("Malla variable — dx$_{min}$ = %g, %d x %d nodos"
                 % (dx, len(x), len(y)), fontsize=12)
    r = os.path.join(OUT, "comparativa", "malla_dx%.4f.png" % dx)
    fig.savefig(r, dpi=DPI)
    plt.close(fig)
    return r


def fig_comparativa_ext():
    d = {(p, dx): json.load(open(os.path.join(
        OUT, p, "metricas", "polar_dx%.4f.json" % dx)))
        for p in PERF for dx in TERNA}
    fig, axs = plt.subplots(1, 3, figsize=(15.2, 4.8), layout="constrained")
    for ax, mag, lab in zip(axs, ("cl", "cd", "ld"), (r"$C_L$", r"$C_D$", r"$L/D$")):
        for p, (nom, col) in PERF.items():
            fina = [d[(p, 0.002)]["%.1f" % a][mag] for a in ANG]
            ax.plot(ANG, fina, color=col, marker="s", ms=4.5, lw=1.3, alpha=.55,
                    label="%s — malla fina" % nom)
            ext = []
            for a in ANG:
                k = "%.1f" % a
                cl = [d[(p, dx)][k]["cl"] for dx in TERNA]
                cd = [d[(p, dx)][k]["cd"] for dx in TERNA]
                cle = vn.extrapola_robusto(*cl, *TERNA)[0]
                cde = vn.extrapola_robusto(*cd, *TERNA, log=min(cd) > 0)[0]
                ext.append({"cl": cle, "cd": cde, "ld": cle / cde}[mag])
            ax.plot(ANG, ext, color=col, marker="o", ms=6.5, lw=2.3, ls="--",
                    label="%s — extrapolado" % nom)
        ax.set_xlabel(r"$\alpha$ [$^\circ$]")
        ax.set_ylabel(lab)
        ax.set_xticks(ANG)
        ax.grid(alpha=.25)
        if mag == "cd":
            ax.axhline(0, color="k", lw=.7, ls=":")
    axs[0].legend(fontsize=8)
    fig.suptitle("Los dos perfiles: malla fina frente a extrapolacion de Richardson "
                 "(terna 0.008/0.004/0.002)", fontsize=12)
    r = os.path.join(OUT, "comparativa", "comparativa_extrapolada.png")
    fig.savefig(r, dpi=DPI)
    plt.close(fig)
    return r


def main():
    for f in (lambda: fig_cp(4.0), lambda: fig_cp(0.0), lambda: fig_cp_par(4.0),
              fig_polar_resistencia,
              fig_malla, fig_comparativa_ext):
        r = f()
        for q in (r if isinstance(r, list) else [r]):
            print("  ->", os.path.relpath(q, ROOT) if q else "(sin datos, omitida)")


if __name__ == "__main__":
    main()

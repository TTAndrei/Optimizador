"""Variantes de la figura de malla: distintas formas de tapar el hueco del
borde de ataque, donde el perfil es mas fino que una celda y la mascara del
solver deja celdas sueltas.

    .venv/bin/python scripts/agent_tests/rf_malla_variantes.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "resultados_finales")
DPI = 190


def relleno_columnas(sol, engorde=0):
    """Une la mascara columna a columna.

    En las columnas sin solido (la punta, donde el perfil no llega a una celda)
    se interpolan los limites superior e inferior de las columnas vecinas. El
    cierre morfologico no vale: el puente es de una sola fila y la erosion se
    lo come.
    """
    ic = np.nonzero(sol.any(axis=0))[0]
    cols = np.arange(ic[0], ic[-1] + 1)
    llenas = [i for i in cols if sol[:, i].any()]
    lo = np.array([np.nonzero(sol[:, i])[0][0] for i in llenas], float)
    hi = np.array([np.nonzero(sol[:, i])[0][-1] for i in llenas], float)
    lo = np.interp(cols, llenas, lo)
    hi = np.interp(cols, llenas, hi)
    r = np.zeros_like(sol)
    for i, a, b in zip(cols, lo, hi):
        j0 = int(np.floor(a)) - engorde
        j1 = int(np.ceil(b)) + engorde
        r[j0:j1 + 1, i] = True
    return r


def dibuja(nombre, sol_dib, x, y, sol_ref, x0_extra=0.0, suave=False, nota="",
           encima=None):
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2), layout="constrained")
    for ax, zoom in zip(axs, (False, True)):
        if zoom:
            jc = np.nonzero(sol_ref.any(axis=1))[0]
            ic = np.nonzero(sol_ref.any(axis=0))[0]
            x0 = x[ic[0]] - 0.03 + x0_extra
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
        if suave:
            ax.contourf(x, y, sol_dib, levels=[.5, 9], colors=["#c0392b"])
            ax.contour(x, y, sol_dib, levels=[.5], colors=["#7b241c"], linewidths=.9)
        else:
            ax.contourf(x, y, sol_dib.astype(float), levels=[.5, 1.5],
                        colors=["#c0392b"])
        if encima is not None:
            ax.contour(x, y, encima, levels=[.5], colors=["#7b241c"], linewidths=1.1)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_aspect("equal")
        ax.set_xlabel("$x/c$")
        ax.set_ylabel("$y/c$")
        ax.set_title(("zoom al borde de ataque (todas las lineas)" if zoom
                      else "dominio completo (1 de cada %d lineas)" % paso),
                     fontsize=10)
    fig.suptitle("Malla variable — dx$_{min}$ = 0.002, %d x %d nodos%s"
                 % (len(x), len(y), nota), fontsize=12)
    r = os.path.join(OUT, "comparativa", "malla_variante_%s.png" % nombre)
    fig.savefig(r, dpi=DPI)
    plt.close(fig)
    print("  ->", os.path.relpath(r, ROOT))


def main():
    d = np.load(os.path.join(OUT, "ganador_ag", "campos",
                             "dx0.0020_a04.0_campo.npz"))
    x, y, sol = d["x"], d["y"], d["solid"].astype(bool)

    a = relleno_columnas(sol)
    dibuja("A_unido", a, x, y, sol, nota=" — punta unida, celdas del solver")

    b = relleno_columnas(sol, engorde=1)
    dibuja("B_grueso", b, x, y, sol, nota=" — perfil engordado una celda")

    # el suavizado parte del perfil engordado: sobre la mascara cruda el
    # blur baja la punta de una celda por debajo del nivel 0.5 y la vuelve a
    # separar
    c = ndimage.gaussian_filter(b.astype(float), 1.0)
    c /= c.max()
    dibuja("C_contorno", c, x, y, sol, suave=True, nota=" — contorno continuo")

    dibuja("D_recorte", sol, x, y, sol, x0_extra=0.042,
           nota=" — ventana desplazada tras el borde de ataque")

    dibuja("E_hibrido", a, x, y, sol, encima=c,
           nota=" — celdas del solver con contorno continuo")


if __name__ == "__main__":
    main()

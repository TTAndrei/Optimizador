"""AG24: borde de salida antes y despues de la cola de cierre."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

sys.path.insert(0, "/home/ttandrei/Proyectos/Optimizador/Optimizador")
import curvo.malla as m

S = "/tmp/claude-1000/-home-ttandrei-Proyectos-Optimizador-Optimizador/a0629037-80d8-4d70-9dfd-fa69ccf7c8bd/scratchpad/"
px, py = m.leer_dat("/home/ttandrei/Proyectos/Optimizador/Optimizador/profiles/AG24")

casos = []
for etiqueta, cola in (("ANTES  --  base plana (cola_te = 0)", 0.0),
                       ("DESPUES  --  cola de cierre (cola_te = 4)", 4.0)):
    X, Y, info = m.generar_c(px, py, cola_te=cola)
    q = m.calidad(X, Y, perfil=info)
    casos.append((etiqueta, X, Y, info, q))

ZOOM = [("x 40", 0.012), ("x 400", 0.0012)]

fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.4))
for fila, (etiqueta, X, Y, info, q) in enumerate(casos):
    J = m.metricas(X, Y)["J"]
    malas = np.argwhere(J <= 0.0)
    cx = 0.25 * (X[:-1, :-1] + X[:-1, 1:] + X[1:, 1:] + X[1:, :-1])
    cy = 0.25 * (Y[:-1, :-1] + Y[:-1, 1:] + Y[1:, 1:] + Y[1:, :-1])
    i0, i1 = info["perfil"]
    nb = info["n_base"]
    x_c, y_c = float(X[0, i0]), float(Y[0, i0])      # punta / punto medio
    M_, N_ = X.shape

    for col, (titulo_zoom, media) in enumerate(ZOOM):
        ax = axes[fila, col]
        segs = [np.column_stack([X[j], Y[j]]) for j in range(M_)]
        segs += [np.column_stack([X[:, i], Y[:, i]]) for i in range(N_)]
        ax.add_collection(LineCollection(segs, colors="0.62", lw=0.5))

        # superficie solida y cola, en colores distintos
        ax.plot(X[0, i0 + nb:i1 - nb], Y[0, i0 + nb:i1 - nb], "-",
                color="#111111", lw=2.0, zorder=5, solid_capstyle="round")
        if nb:
            ax.plot(X[0, i0:i0 + nb + 1], Y[0, i0:i0 + nb + 1], "-",
                    color="#d62728", lw=2.0, zorder=5)
            ax.plot(X[0, i1 - nb - 1:i1], Y[0, i1 - nb - 1:i1], "-",
                    color="#d62728", lw=2.0, zorder=5)
        # corte de estela
        ax.plot(X[0, :i0 + 1], Y[0, :i0 + 1], "-", color="#1f6feb", lw=1.6, zorder=4)

        if len(malas):
            ax.plot(cx[malas[:, 0], malas[:, 1]], cy[malas[:, 0], malas[:, 1]],
                    "o", color="red", ms=7, mew=1.2, mec="darkred", zorder=9)

        ax.set_xlim(x_c - 1.35 * media, x_c + 0.75 * media)
        ax.set_ylim(y_c - media, y_c + media)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"zoom {titulo_zoom}", fontsize=10, color="0.35")
        for lado in ax.spines.values():
            lado.set_color("0.7")

    veredicto = "BUENA" if q["buena"] else ("valida" if q["valida"] else "NO VALIDA")
    color = "#146c2e" if q["valida"] else "#b3261e"
    y_fila = 0.735 - 0.475 * fila
    fig.text(0.018, y_fila, etiqueta.split("  --  ")[0], rotation=90,
             va="center", ha="center", fontsize=13, color=color, weight="bold")
    fig.text(0.052, y_fila,
             f"{etiqueta.split('  --  ')[1]}\n"
             f"J<=0: {q['j_negativos']} celdas   ort. pared: "
             f"{q['ortogonalidad_pared_min']:.1f} deg\n"
             f"oblic. p99: {q['oblicuidad_p99']:.3f}   ->  {veredicto}",
             rotation=90, va="center", ha="center", fontsize=9.5, color=color,
             family="monospace")

fig.suptitle("AG24, borde de salida  --  negro: perfil   rojo: cola de cierre   "
             "azul: corte de estela   puntos rojos: jacobiano <= 0", fontsize=11.5)
fig.tight_layout(rect=[0.075, 0.0, 1, 0.965])
fig.savefig(S + "te_ag24_antes_despues.png", dpi=125)
print(S + "te_ag24_antes_despues.png")

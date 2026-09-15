"""
Figura 3.B — Modo de tablero antes y despues del filtro.

Dos corridas identicas del punto que revienta —perfil optimizado, dx=0.004,
alpha=0, presupuesto corto 2x3, warm_start activo— que solo se diferencian en
`warm_start_filtered`, es decir en si la semilla de proyeccion pasa o no por el
paso de Jacobi ponderado del apartado 3.6.5. Las dos se paran en la misma
iteracion, 700, muy por debajo de la 1340 en la que la version sin filtro aborta
por blowup, para que la comparacion sea entre dos campos vivos.

QUE CAMPO SE DIBUJA Y POR QUE. La divergencia residual tras la proyeccion, no la
presion. El modo par-impar de una malla colocalizada es casi invisible para el
operador de presion (figura 3.A), asi que no se acumula en p —comprobado: el
campo de presion de las dos corridas es indistinguible a simple vista— sino en la
divergencia que la proyeccion deja sin corregir, que es tambien donde estaba
medido (46 % de la divergencia residual en ese modo). Ahi el damero se ve a
simple vista, celda a celda, y se ve desaparecer.

Las dos corridas se ejecutan enteras y se guardan sus campos en .npz: retocar la
figura no obliga a repetirlas.

Uso:
    .venv/bin/python scripts/agent_tests/fig_mem_tablero.py [--solo-figura]
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "agent_tests"))
os.chdir(ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

OUT = os.path.join(ROOT, "figuras_memoria")
NPZ = os.path.join(OUT, "datos_3B_tablero.npz")
DPI = 200

DX, ALPHA, ITERS = 0.004, 0.0, 700
BASE = ("warm_start", "coarse_mask_majority", "fast_masks", "interp_float32")
CASOS = (("sin_filtro", BASE),
         ("con_filtro", BASE + ("warm_start_filtered",)))
# Recorte: extrados, intrados y arranque de la estela del perfil, que ocupa
# x de 6 a 7. Es donde el damero tiene amplitud; mas lejos la divergencia
# residual es cero a esta escala.
VENTANA = (6.02, 6.78, 7.955, 8.115)


def frac_tablero(d, solid):
    """Fraccion de la norma que NO sobrevive a promediar en bloques 2x2, o sea
    la parte par-impar. Misma definicion que scripts/agent_tests/diag_divergencia.py."""
    d = np.where(solid, 0.0, d)
    ny, nx = d.shape
    dy, dx = ny - ny % 2, nx - nx % 2
    b = d[:dy, :dx].reshape(dy // 2, 2, dx // 2, 2).mean(axis=(1, 3))
    suave = np.repeat(np.repeat(b, 2, axis=0), 2, axis=1)
    resto = d[:dy, :dx] - suave
    ns, nr = np.linalg.norm(suave), np.linalg.norm(resto)
    return float(nr / max(np.hypot(ns, nr), 1e-30))


def corre():
    import cupy as cp
    import RunGA
    import Simulador2D
    import opt_solver
    import resultados_finales as RF
    import verificacion_numerica as vn

    salida = {}
    for etiqueta, flags in CASOS:
        opt_solver.reset()
        opt_solver.enable(*flags)
        assert opt_solver.active() == sorted(flags), opt_solver.active()

        cfg = dict(RunGA.CONFIG)
        sim = {
            "filepath": os.path.join(RF.OUT, "perfiles", "ganador_ag.dat"),
            "iteraciones": ITERS,
            "v0x": 1.0, "v0y": 0.0, "CFL": vn.CFL,
            "dx_min": DX, "alpha_deg": ALPHA, "chord": 1.0,
            "rho": 1.0, "nu": 1.0 / RF.RE,
            "turb_model": cfg.get("turb_model", "sa"),
            "wall_treatment": cfg.get("wall_treatment", "consistent"),
            "advection_scheme": cfg.get("advection_scheme", "maccormack"),
            "min_te_height_factor": cfg.get("min_te_height_factor", 1.0),
            "wake_refinement_mode": cfg.get("wake_refinement_mode", "long_fine_x"),
            "mg_niveles_max": cfg.get("mg_niveles_max", 2),
            "divergencia": cfg.get("divergencia", 0.02),
            "graficos": False, "live_view": False, "mostrar_malla": False,
            **RF.DOMINIO, **RF.PRESUPUESTO, **vn.EXTRA,
            "stop_on_convergence": False, "stop_on_clcd_convergence": False,
        }
        mesh = Simulador2D.main(**sim)
        # Misma divergencia que ve la proyeccion con wall_treatment="consistent":
        # la de flujos por cara, con flujo nulo contra la pared.
        div = cp.asnumpy(mesh._compute_flux_divergence_field_uv()).astype(np.float64)
        solid = cp.asnumpy(mesh.solid).astype(bool)
        salida[f"div_{etiqueta}"] = div
        salida["solid"] = solid
        salida["x"] = cp.asnumpy(mesh.X_1d)
        salida["y"] = cp.asnumpy(mesh.Y_1d)
        print(f"{etiqueta}: |div| media = {np.abs(div[~solid]).mean():.3e}   "
              f"fraccion tablero = {frac_tablero(div, solid):.3f}")

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(NPZ, dx=DX, alpha=ALPHA, iters=ITERS, **salida)
    print("campos guardados en", NPZ)


def figura():
    z = np.load(NPZ)
    x, y, solid = z["x"], z["y"], z["solid"].astype(bool)
    x0, x1, y0, y1 = VENTANA
    i0, i1 = np.searchsorted(x, [x0, x1])
    j0, j1 = np.searchsorted(y, [y0, y1])
    sj, si = slice(j0, j1), slice(i0, i1)
    s = solid[sj, si]
    ext = [x[i0], x[i1 - 1], y[j0], y[j1 - 1]]

    campos = {k: z[f"div_{k}"][sj, si] for k, _ in CASOS}
    lim = float(np.percentile(np.abs(campos["sin_filtro"][~s]), 99.0))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.3), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.885, top=0.80, bottom=0.235, wspace=0.06)

    titulo = {"sin_filtro": "semilla reutilizada sin filtrar",
              "con_filtro": r"con un paso de Jacobi ponderado, $\omega=1/2$"}
    pie = []
    for ax, (k, _) in zip(axs, CASOS):
        c = campos[k]
        im = ax.imshow(np.ma.masked_where(s, c), origin="lower", extent=ext,
                       cmap="RdBu_r", norm=norm, interpolation="nearest",
                       aspect="auto")
        ax.contour(x[si], y[sj], s.astype(float), levels=[0.5], colors="k",
                   linewidths=1.1)
        ax.set_title(titulo[k], fontsize=11.5, pad=6)
        ax.set_xlabel("x [c]", fontsize=11)
        med = np.abs(c[~s]).mean()
        fr = frac_tablero(c, s)
        pie.append((med, fr))
        print(f"{k}: recorte  |div| media = {med:.3e}   fraccion tablero = {fr:.3f}")
    axs[0].set_ylabel("y [c]", fontsize=11)

    cax = fig.add_axes([0.898, 0.235, 0.015, 0.565])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"divergencia residual  $\nabla\!\cdot\!\mathbf{u}$", fontsize=10)

    fig.suptitle("Modo de tablero en la divergencia residual tras la proyección  "
                 f"(dx={float(z['dx']):g}, "
                 rf"$\alpha={float(z['alpha']):.0f}^\circ$, "
                 f"iteración {int(z['iters'])}, presupuesto 2×3)",
                 fontsize=12.5, y=0.955)
    g = [np.abs(z[f"div_{k}"][~solid]).mean() for k, _ in CASOS]
    fig.text(0.5, 0.095,
             r"$|\nabla\!\cdot\!\mathbf{u}|$ media:  "
             f"{pie[0][0]:.2e} → {pie[1][0]:.2e} en el recorte"
             f"       (×{pie[0][0] / max(pie[1][0], 1e-30):.1f}),        "
             f"{g[0]:.2e} → {g[1]:.2e} en todo el dominio"
             f"       (×{g[0] / max(g[1], 1e-30):.0f})",
             ha="center", fontsize=10.5, color="0.2")
    fig.text(0.5, 0.028,
             "Misma escala de color en los dos paneles. El filtro no toca el fondo "
             "suave: lo que desaparece es la alternancia celda a celda.",
             ha="center", fontsize=9, color="0.45")

    p = os.path.join(OUT, "fig_3B_modo_tablero.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("->", p)


if __name__ == "__main__":
    if "--solo-figura" not in sys.argv:
        corre()
    figura()

"""
Tira horizontal para diapositiva: lo que aporta cada pieza de la proyeccion.

Los cuatro paneles enseñan LA MISMA magnitud, la divergencia del campo, con la
misma escala de color, sobre el mismo instante y el mismo campo de partida:

  1  u* sin proyectar                     lo que dejan adveccion y difusion
  2  Gauss-Seidel rojo-negro (sin niveles) el suavizador solo: borra lo brusco y
                                           deja el error suave de gran escala
  3  + multimalla (ciclo en V, 2 niveles)  el nivel grueso se lleva esa parte
                                           suave, que en el es brusca
  4  + bucle externo de defecto (2x3)      el presupuesto de produccion

Sin titulo: la figura va sobre la diapositiva.

Uso:
    .venv/bin/python scripts/agent_tests/fig_ppt_etapas_proyeccion.py
    .venv/bin/python scripts/agent_tests/fig_ppt_etapas_proyeccion.py --solo-figura
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "agent_tests"))
os.chdir(ROOT)
os.environ.setdefault("OPT_SOLVER_OFF", "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from fig_mem_pasos_fraccionados import remuestrea, DX, ALPHA, ITERS, PERFIL
from fig_ppt_proyeccion import flecha, sin_ejes, GRIS

OUT = os.path.join(ROOT, "figuras_memoria")
NPZ = os.path.join(OUT, "datos_etapas_proyeccion.npz")
DPI = 240

# (clave, niveles de coarsening, ciclos, iteraciones externas)
ETAPAS = [
    ("gauss_seidel", 0, 3, 1),
    ("multimalla", 2, 3, 1),
    ("bucle_externo", 2, 3, 2),
]
PIES = [
    ("sin_proyectar", r"$\mathbf{u}^{*}$  sin proyectar", None, None),
    ("gauss_seidel", "solo Gauss-Seidel", "Gauss-Seidel", "rojo-negro"),
    ("multimalla", "+ multimalla", "multimalla", "ciclo en V"),
    ("bucle_externo", "+ bucle externo", "defecto", "2 externas"),
]


def corre(dx=DX, iters=ITERS):
    import cupy as cp
    import RunGA
    import Simulador2D
    import opt_solver
    import resultados_finales as RF
    import verificacion_numerica as vn

    opt_solver.reset()
    cfg = dict(RunGA.CONFIG)
    rho, nu = 1.0, 1.0 / RF.RE
    sim = {
        "filepath": PERFIL, "iteraciones": iters,
        "v0x": 1.0, "v0y": 0.0, "CFL": vn.CFL,
        "dx_min": dx, "alpha_deg": ALPHA, "chord": 1.0, "rho": rho, "nu": nu,
        "turb_model": cfg.get("turb_model", "sa"),
        "wall_treatment": cfg.get("wall_treatment", "consistent"),
        "advection_scheme": cfg.get("advection_scheme", "maccormack"),
        "min_te_height_factor": cfg.get("min_te_height_factor", 1.0),
        "wake_refinement_mode": cfg.get("wake_refinement_mode", "long_fine_x"),
        "mg_niveles_max": cfg.get("mg_niveles_max", 2),
        "divergencia": cfg.get("divergencia", 0.02),
        "graficos": False, "live_view": False, "mostrar_malla": False,
        "shm_publish": False, "save_frames": False,
        **RF.DOMINIO, **RF.PRESUPUESTO, **vn.EXTRA,
        "stop_on_convergence": False, "stop_on_clcd_convergence": False,
    }
    t0 = time.time()
    mesh = Simulador2D.main(**sim)
    print(f"estado desarrollado en {(time.time() - t0)/60:.1f} min")

    speed = cp.sqrt(mesh.u * mesh.u + mesh.v * mesh.v)
    Umax = float(cp.max(speed[~mesh.solid]))
    dt_adv = vn.CFL * min(mesh.dx, mesh.dy) / max(Umax, 1e-12)
    nu_eff = nu + float(cp.max(mesh.sa_nu_t)) if mesh.sa_nu_t is not None else nu
    dt = float(min(dt_adv, 0.25 * min(mesh.dx, mesh.dy) ** 2 / max(nu_eff, 1e-30)))
    print(f"dt = {dt:.3e}")

    # un paso de adveccion + difusion: este es el u* que las cuatro variantes
    # reciben, identico para todas
    mesh.advect_velocities(dt)
    mesh.advect_sa()
    mesh.update_sa(nu, dt)
    mesh.apply_boundaries(after_projection=False)
    mesh.diffuse_velocity(nu, dt, usar_wale=False, nu_t_field=mesh.sa_nu_t)
    mesh.apply_boundaries(after_projection=False)

    campos_estado = ("u", "v", "p", "nu_tilde", "sa_nu_t", "sa_gamma")
    estado = {k: (getattr(mesh, k).copy() if getattr(mesh, k, None) is not None
                  else None) for k in campos_estado}

    def restaura():
        for k, val in estado.items():
            if val is not None:
                setattr(mesh, k, val.copy())

    def guarda(cont, nombre):
        d = cp.asnumpy(mesh._compute_flux_divergence_field_uv())
        cont[f"div_{nombre}"] = d
        libre = ~cp.asnumpy(mesh.solid)
        print(f"  {nombre:15s} rms |div| = {np.sqrt((d[libre]**2).mean()):.3e}")
        return cont

    campos = {}
    guarda(campos, "sin_proyectar")

    for clave, niveles, ciclos, outers in ETAPAS:
        restaura()
        # la jerarquia se construye una sola vez y despues el parametro se
        # ignora: hay que forzar la reconstruccion para cambiar de niveles
        mesh._mg_initialized = False
        mesh.project_multigrid(
            rho, dt, tol_div=None, tol_div_rel=cfg.get("divergencia", 0.02),
            max_outer=outers, cycles_per_outer=ciclos, niveles_max=niveles,
            pre_suavizado=3, post_suavizado=3,
            modo_adaptativo=False,          # el presupuesto es el que se pide
            guard_residual_every_outer=True, adaptive_outer0_cycles=False,
            apply_ibm_each_outer=True, rollback_on_nan=True, compute_div_after=True,
            projection_variant="legacy_centered",
            mg_pressure_accumulation="outer_sum",
            wall_pressure_gradient_mode="one_sided",
            divergence_form="face_flux", verbose=False)
        mesh.apply_boundaries(after_projection=True)
        guarda(campos, clave)

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(NPZ, X=cp.asnumpy(mesh.X_1d), Y=cp.asnumpy(mesh.Y_1d),
                        solid=cp.asnumpy(mesh.solid), dt=dt, dx=dx, **campos)
    print("->", NPZ)


def figura():
    d = dict(np.load(NPZ))
    X, Y = d["X"], d["Y"]
    solid = d["solid"].astype(bool)
    jj = np.where(solid.any(axis=0))[0]
    ii = np.where(solid.any(axis=1))[0]
    xc, cuerda = X[jj[0]], X[jj[-1]] - X[jj[0]]
    yc = 0.5 * (Y[ii[0]] + Y[ii[-1]])
    x = xc + np.linspace(-0.25, 1.35, 520) * cuerda
    y = yc + np.linspace(-0.42, 0.42, 300) * cuerda
    xr, yr = (x - xc) / cuerda, (y - yc) / cuerda
    fondo = np.where(solid, np.nan, 1.0)

    campos = {k: remuestrea(X, Y, d[f"div_{k}"] * fondo, x, y) for k, *_ in PIES}
    S = np.nan_to_num(remuestrea(X, Y, solid.astype(float), x, y))

    # la divergencia se concentra en la pared: con el maximo la ventana entera
    # sale blanca y con un percentil bajo sale toda saturada. El 97 del campo sin
    # proyectar deja ver la estructura y mantiene la escala comun a los cuatro
    ref = campos["sin_proyectar"]
    lim = float(np.nanpercentile(np.abs(ref[np.isfinite(ref)]), 97.0))

    fig = plt.figure(figsize=(19.2, 4.6))
    izq, der, ws = 0.018, 0.945, 0.62
    gs = fig.add_gridspec(1, 4, left=izq, right=der, top=0.845, bottom=0.175,
                          wspace=ws)
    ancho = (der - izq) / (4 + 3 * ws)
    hueco = ws * ancho

    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    for k, (clave, pie, _etq, _sub) in enumerate(PIES):
        ax = fig.add_subplot(gs[0, k])
        im = ax.pcolormesh(xr, yr, campos[clave], cmap="RdBu_r", shading="auto",
                           norm=norm)
        ax.contour(xr, yr, S, levels=[0.5, 1.5], colors="k", linewidths=1.0)
        ax.set_aspect("equal")
        sin_ejes(ax)
        ax.set_title(pie, fontsize=15, color=GRIS, pad=11)
        rms = float(np.sqrt((d[f"div_{clave}"][~solid] ** 2).mean()))
        ax.text(0.5, -0.11, rf"rms $|\nabla\!\cdot\!\mathbf{{u}}|$ = {rms:.1e}",
                transform=ax.transAxes, ha="center", va="top", fontsize=12.5,
                color="0.35")

    for k, (_clave, _pie, etq, sub) in enumerate(PIES[1:]):
        g0 = izq + (k + 1) * ancho + k * hueco
        flecha(fig, g0 + 0.14 * hueco, g0 + 0.86 * hueco, 0.53, etq, sub)

    cax = fig.add_axes([0.958, 0.28, 0.008, 0.44])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([-lim, 0.0, lim])
    cb.set_ticklabels(["−", "0", "+"])
    cb.ax.tick_params(labelsize=13, length=0, colors=GRIS)
    cb.outline.set_edgecolor("0.75")
    cb.ax.set_title(r"$\nabla\!\cdot\!\mathbf{u}$", fontsize=14, color=GRIS,
                    pad=10)

    ruta = os.path.join(OUT, "ppt_proyeccion_etapas.png")
    fig.savefig(ruta, dpi=DPI)
    plt.close(fig)
    print("->", ruta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-figura", action="store_true")
    ap.add_argument("--dx", type=float, default=DX)
    ap.add_argument("--iters", type=int, default=ITERS)
    a = ap.parse_args()
    if not a.solo_figura:
        corre(a.dx, a.iters)
    figura()

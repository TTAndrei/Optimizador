"""
Figuras del paso fraccionado — el campo tras adveccion, difusion y proyeccion.

Se lleva el NACA 0012 a dx=0.002, alpha=4, Re=1e5, dominio C 24x16, con la
configuracion de produccion (SA + transicion SA-BC, MacCormack, presupuesto de
proyeccion 2x3) hasta un estado desarrollado. Ahi se congela el campo y se da UN
paso temporal capturando el estado despues de cada uno de los tres subpasos.

Cada figura tiene tres paneles sobre la misma ventana alrededor del perfil:

  a) |u| con lineas de corriente: el campo tal como queda tras el subpaso.
  b) divergencia por flujo del mismo campo, con la MISMA escala en las tres
     figuras. Es lo que hace visible el reparto de trabajo: adveccion y difusion
     dejan un campo con divergencia; la proyeccion la borra.
  c) el incremento que ese subpaso concreto ha metido, (du, dv) respecto al campo
     con el que entro, en magnitud + vectores.

Los campos se guardan en un npz para rehacer las figuras sin repetir la corrida.

Uso:
    .venv/bin/python scripts/agent_tests/fig_mem_pasos_fraccionados.py
    .venv/bin/python scripts/agent_tests/fig_mem_pasos_fraccionados.py --solo-figura
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
# el estudio de dx=0.002 se corrio con el solver optimizado apagado (blowup del
# warm_start con presupuesto 2x3); se mantiene
os.environ.setdefault("OPT_SOLVER_OFF", "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import RegularGridInterpolator

OUT = os.path.join(ROOT, "figuras_memoria")
NPZ = os.path.join(OUT, "datos_pasos_fraccionados.npz")
DPI = 200

DX, ALPHA, ITERS = 0.002, 4.0, 13000     # t ~ 10 con CFL=0.5
PERFIL = os.path.join(ROOT, "profiles", "NACA_0012")

# ventana de dibujo, en cuerdas y relativa al perfil (el dominio tiene el perfil
# en x=cx, y=Ly/2; la caja se saca del bounding box de la mascara solida)
VENTANA = (-0.45, 1.55, -0.70, 0.70)

ETIQUETAS = [
    ("adveccion", "a) tras la ADVECCIÓN",
     r"semi-lagrangiana + MacCormack: transporta $\mathbf{u}$ por sus propias "
     r"trayectorias"),
    ("difusion", "b) tras la DIFUSIÓN",
     r"$\nu_{\rm eff}\nabla^2\mathbf{u}$ (molecular + SA): alisa gradientes, "
     r"engorda la capa límite"),
    ("proyeccion", "c) tras la PROYECCIÓN",
     r"resta $\nabla p$ hasta $\nabla\!\cdot\!\mathbf{u}\approx 0$: el único "
     r"subpaso que impone incompresibilidad"),
]


def corre(dx=DX, iters=ITERS):
    import cupy as cp
    import RunGA
    import Simulador2D
    import opt_solver
    import resultados_finales as RF
    import verificacion_numerica as vn

    # dx=0.002 se corrio en el estudio de precision con el solver optimizado
    # apagado; se mantiene aqui para no arriesgar el blowup del warm_start.
    opt_solver.reset()

    cfg = dict(RunGA.CONFIG)
    rho, nu = 1.0, 1.0 / RF.RE
    sim = {
        "filepath": PERFIL,
        "iteraciones": iters,
        "v0x": 1.0, "v0y": 0.0, "CFL": vn.CFL,
        "dx_min": dx, "alpha_deg": ALPHA, "chord": 1.0,
        "rho": rho, "nu": nu,
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

    # --- dt del paso, mismo criterio que el bucle principal ----------------
    speed = cp.sqrt(mesh.u * mesh.u + mesh.v * mesh.v)
    Umax = float(cp.max(speed[~mesh.solid]))
    dt_adv = vn.CFL * min(mesh.dx, mesh.dy) / max(Umax, 1e-12)
    nu_eff = nu + float(cp.max(mesh.sa_nu_t)) if mesh.sa_nu_t is not None else nu
    dt = float(min(dt_adv, 0.25 * min(mesh.dx, mesh.dy) ** 2 / max(nu_eff, 1e-30)))
    print(f"dt del paso = {dt:.3e}")

    campos = {}

    def snapshot(nombre):
        div = mesh._compute_flux_divergence_field_uv()
        campos[f"u_{nombre}"] = cp.asnumpy(mesh.u)
        campos[f"v_{nombre}"] = cp.asnumpy(mesh.v)
        campos[f"p_{nombre}"] = cp.asnumpy(mesh.p)
        campos[f"div_{nombre}"] = cp.asnumpy(div)

    snapshot("inicial")

    mesh.advect_velocities(dt)
    mesh.advect_sa()
    mesh.update_sa(nu, dt)
    mesh.apply_boundaries(after_projection=False)
    snapshot("adveccion")

    mesh.diffuse_velocity(nu, dt, usar_wale=False, nu_t_field=mesh.sa_nu_t)
    mesh.apply_boundaries(after_projection=False)
    snapshot("difusion")

    mesh.project_multigrid(
        rho, dt,
        tol_div=None, tol_div_rel=cfg.get("divergencia", 0.02),
        max_outer=RF.PRESUPUESTO["mg_max_outer"],
        cycles_per_outer=RF.PRESUPUESTO["mg_cycles_per_outer"],
        niveles_max=cfg.get("mg_niveles_max", 2),
        pre_suavizado=3, post_suavizado=3,
        guard_residual_every_outer=True,
        adaptive_outer0_cycles=False,
        apply_ibm_each_outer=True,
        rollback_on_nan=True,
        compute_div_after=True,
        projection_variant="legacy_centered",
        mg_pressure_accumulation="outer_sum",
        wall_pressure_gradient_mode="one_sided",
        divergence_form="face_flux",
        verbose=False,
    )
    mesh.apply_boundaries(after_projection=True)
    snapshot("proyeccion")

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(
        NPZ,
        X=cp.asnumpy(mesh.X_1d), Y=cp.asnumpy(mesh.Y_1d),
        solid=cp.asnumpy(mesh.solid),
        dt=dt, dx=dx, alpha=ALPHA, iters=iters,
        **campos)
    print("->", NPZ)


# ----------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------
def remuestrea(X, Y, campo, x, y):
    """Campo de malla estirada -> rejilla uniforme de la ventana."""
    f = RegularGridInterpolator((Y, X), campo, bounds_error=False, fill_value=np.nan)
    XX, YY = np.meshgrid(x, y)
    return f(np.stack([YY, XX], axis=-1))


def figura(d, clave, titulo, subtitulo, escalas):
    X, Y = d["X"], d["Y"]
    solid = d["solid"].astype(bool)
    jj = np.where(solid.any(axis=0))[0]
    ii = np.where(solid.any(axis=1))[0]
    xc, cuerda = X[jj[0]], X[jj[-1]] - X[jj[0]]
    yc = 0.5 * (Y[ii[0]] + Y[ii[-1]])
    x0, x1, y0, y1 = VENTANA
    x = xc + np.linspace(x0, x1, 640) * cuerda
    y = yc + np.linspace(y0, y1, 460) * cuerda
    xr, yr = (x - xc) / cuerda, (y - yc) / cuerda

    prev = {"adveccion": "inicial", "difusion": "adveccion",
            "proyeccion": "difusion"}[clave]
    fondo = np.where(solid, np.nan, 1.0)

    u = d[f"u_{clave}"] * fondo
    v = d[f"v_{clave}"] * fondo
    pres = d[f"p_{clave}"] * fondo
    div = d[f"div_{clave}"] * fondo
    du = (d[f"u_{clave}"] - d[f"u_{prev}"]) * fondo
    dv = (d[f"v_{clave}"] - d[f"v_{prev}"]) * fondo

    U = remuestrea(X, Y, u, x, y)
    V = remuestrea(X, Y, v, x, y)
    P = remuestrea(X, Y, pres, x, y)
    D = remuestrea(X, Y, div, x, y)
    dU = remuestrea(X, Y, du, x, y)
    dV = remuestrea(X, Y, dv, x, y)
    S = remuestrea(X, Y, solid.astype(float), x, y)

    fig, axs = plt.subplots(1, 4, figsize=(19.6, 4.5))
    fig.subplots_adjust(left=0.036, right=0.972, top=0.80, bottom=0.115, wspace=0.30)

    perfil = dict(levels=[0.5, 1.5], colors="k", linewidths=1.2)

    # --- a) velocidad, contraste alrededor de la corriente libre ----------
    # Con una rampa desde cero (viridis) la ventana entera sale del mismo color:
    # |u| se mueve entre 0.8 y 1.2. Divergente centrada en U_inf: rojo acelera,
    # azul frena, y la capa limite y la estela se separan solas.
    ax = axs[0]
    mag = np.hypot(U, V)
    lim = escalas["umax_dev"]
    im = ax.pcolormesh(xr, yr, mag, cmap="RdBu_r", shading="auto",
                       norm=TwoSlopeNorm(vmin=1.0 - lim, vcenter=1.0,
                                         vmax=1.0 + lim))
    Uf, Vf = np.nan_to_num(U), np.nan_to_num(V)
    ax.streamplot(xr, yr, Uf, Vf, color="0.20", linewidth=0.40, density=1.15,
                  arrowsize=0.55)
    fig.colorbar(im, ax=ax, pad=0.02).set_label(r"$|\mathbf{u}|$   ($U_\infty=1$)")
    ax.set_title(r"velocidad y líneas de corriente", fontsize=10)

    # --- b) presion --------------------------------------------------------
    ax = axs[1]
    lim = escalas["pmax"]
    im = ax.pcolormesh(xr, yr, P, cmap="RdBu_r", shading="auto",
                       norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
    fig.colorbar(im, ax=ax, pad=0.02).set_label(r"$p/\rho$")
    ax.set_title("presión", fontsize=10)

    # --- c) divergencia ----------------------------------------------------
    ax = axs[2]
    lim = escalas["divmax"]
    im = ax.pcolormesh(xr, yr, D, cmap="RdBu_r", shading="auto",
                       norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
    fig.colorbar(im, ax=ax, pad=0.02).set_label(r"$\nabla\!\cdot\!\mathbf{u}$")
    rms = float(np.sqrt(np.nanmean(D[np.isfinite(D)] ** 2)))
    ax.set_title(rf"divergencia   (rms $={rms:.2e}$)", fontsize=10)

    # --- d) incremento del subpaso ----------------------------------------
    ax = axs[3]
    dmag = np.hypot(dU, dV)
    # aqui SI se normaliza panel a panel: los tres incrementos se llevan casi
    # dos ordenes de magnitud y con escala comun dos de los tres salen negros.
    # La comparacion cuantitativa va en el maximo impreso en el titulo.
    vmax = float(np.nanpercentile(dmag, 99.5))
    im = ax.pcolormesh(xr, yr, dmag, cmap="magma", shading="auto",
                       vmin=0.0, vmax=vmax)
    k = 16
    ax.quiver(xr[::k], yr[::k], np.nan_to_num(dU)[::k, ::k],
              np.nan_to_num(dV)[::k, ::k], color="w",
              scale=vmax * 26.0, scale_units="width",
              width=0.0032, alpha=0.85)
    fig.colorbar(im, ax=ax, pad=0.02).set_label(r"$|\Delta\mathbf{u}|$ del subpaso")
    dm = float(np.nanmax(np.hypot(dU, dV)))
    ax.set_title(rf"incremento aportado   (máx $={dm:.2e}$)", fontsize=10)

    for ax in axs:
        ax.contour(xr, yr, np.nan_to_num(S), **perfil)
        ax.set_aspect("equal")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xlabel("x/c", fontsize=9)
    axs[0].set_ylabel("y/c", fontsize=9)

    fig.suptitle(f"{titulo}\n{subtitulo}", fontsize=12.5, y=0.975)
    ruta = os.path.join(OUT, f"fig_paso_{clave}.png")
    fig.savefig(ruta, dpi=DPI)
    plt.close(fig)
    print("->", ruta)


def figuras():
    d = dict(np.load(NPZ))
    solid = d["solid"].astype(bool)

    # escalas comunes a las tres figuras: si cada una tiene la suya, la
    # comparacion que la figura quiere enseñar deja de existir
    mag = np.hypot(d["u_proyeccion"], d["v_proyeccion"])[~solid]
    umax_dev = float(np.nanpercentile(np.abs(mag - 1.0), 99.0))
    pmax = float(np.nanpercentile(np.abs(d["p_proyeccion"][~solid]), 99.0))
    divmax = float(np.nanpercentile(
        np.abs(d["div_adveccion"][~solid]), 99.0))
    escalas = {"umax_dev": umax_dev, "pmax": pmax, "divmax": divmax}
    print(f"escala |u|-1 = ±{umax_dev:.3f}   p = ±{pmax:.3f}"
          f"   div = ±{divmax:.3e}")

    for clave, titulo, sub in ETIQUETAS:
        figura(d, clave, titulo, sub, escalas)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-figura", action="store_true")
    ap.add_argument("--dx", type=float, default=DX)
    ap.add_argument("--iters", type=int, default=ITERS)
    a = ap.parse_args()
    if not a.solo_figura:
        corre(a.dx, a.iters)
    figuras()

"""
Figuras de presentacion — que hace cada operador del paso fraccionado.

Un solo paso temporal no sirve para enseñarlo: con dt = 7.5e-4 los tres subpasos
mueven el campo del orden de 1e-3 y las tres imagenes salen identicas. Aqui se
hace lo contrario: se congela el campo desarrollado (NACA 0012, dx=0.002,
alpha=4, Re=1e5, dominio C) y desde el MISMO estado se deja actuar cada operador
SOLO, repetido, hasta que su efecto es visible. No es una simulacion fisica: es
el experimento de quitar los otros dos.

  adveccion sola   el campo se transporta a si mismo sin presion que lo cierre:
                   la capa limite se estira y la divergencia se dispara.
  difusion sola    nu_eff * lap(u) repetido: los gradientes se borran, la capa
                   limite engorda y la estela se disuelve.
  proyeccion sola  desde el campo ya advectado y difundido, aplicada varias
                   veces: la divergencia se hunde y el campo vuelve a ser
                   solenoidal. Es la unica que impone incompresibilidad.
  sin proyeccion   adveccion + difusion repetidas sin proyectar: el contraste,
                   por si hace falta el argumento de por que esta ahi.

Uso:
    .venv/bin/python scripts/agent_tests/fig_mem_operadores_aislados.py
    .venv/bin/python scripts/agent_tests/fig_mem_operadores_aislados.py --solo-figura
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

from fig_mem_pasos_fraccionados import remuestrea, VENTANA, DX, ALPHA, ITERS, PERFIL

OUT = os.path.join(ROOT, "figuras_memoria")
NPZ = os.path.join(OUT, "datos_operadores_aislados.npz")
DPI = 200

# tiempos convectivos (c/U_inf) que se deja actuar a cada operador, no numero de
# pasos: dt depende de la malla y con N fijo las figuras de dos mallas no serian
# el mismo experimento. La difusion necesita mucho mas tiempo que la adveccion.
T_OBJ = {"adveccion": 0.35, "difusion": 2.5, "sin_proyeccion": 0.12}
N_PROYECCIONES = 12
N_PASOS = {}

CASOS = [
    ("adveccion", "ADVECCIÓN sola",
     "el campo se transporta a sí mismo; sin presión que lo cierre la capa "
     "límite se estira y aparecen fuentes y sumideros por todas partes"),
    ("difusion", "DIFUSIÓN sola",
     r"$\nu_{\rm eff}\nabla^2\mathbf{u}$ repetido: borra gradientes, engorda la "
     r"capa límite y disuelve la estela"),
    ("proyeccion", "PROYECCIÓN sola",
     r"resta $\nabla p$ sobre el campo ya advectado y difundido hasta "
     r"$\nabla\!\cdot\!\mathbf{u}\to 0$"),
    ("sin_proyeccion", "ADVECCIÓN + DIFUSIÓN, SIN proyectar",
     "el paso completo al que se le quita la proyección: la divergencia se "
     "acumula y el campo deja de tener sentido físico"),
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

    campos_estado = ("u", "v", "p", "nu_tilde", "sa_nu_t", "sa_gamma")
    estado = {k: (getattr(mesh, k).copy() if getattr(mesh, k, None) is not None
                  else None) for k in campos_estado}

    def restaura():
        for k, val in estado.items():
            if val is not None:
                setattr(mesh, k, val.copy())

    def proyecta():
        mesh.project_multigrid(
            rho, dt, tol_div=None, tol_div_rel=cfg.get("divergencia", 0.02),
            max_outer=RF.PRESUPUESTO["mg_max_outer"],
            cycles_per_outer=RF.PRESUPUESTO["mg_cycles_per_outer"],
            niveles_max=cfg.get("mg_niveles_max", 2),
            pre_suavizado=3, post_suavizado=3,
            guard_residual_every_outer=True, adaptive_outer0_cycles=False,
            apply_ibm_each_outer=True, rollback_on_nan=True, compute_div_after=True,
            projection_variant="legacy_centered",
            mg_pressure_accumulation="outer_sum",
            wall_pressure_gradient_mode="one_sided",
            divergence_form="face_flux", verbose=False)
        mesh.apply_boundaries(after_projection=True)

    def sano():
        """Los operadores sueltos no son un esquema estable: se para si revienta."""
        return (not bool(cp.isnan(mesh.u).any())
                and float(cp.max(cp.abs(mesh.u))) < 8.0)

    def guarda(cont, nombre):
        div = mesh._compute_flux_divergence_field_uv()
        cont[f"u_{nombre}"] = cp.asnumpy(mesh.u)
        cont[f"v_{nombre}"] = cp.asnumpy(mesh.v)
        cont[f"p_{nombre}"] = cp.asnumpy(mesh.p)
        cont[f"div_{nombre}"] = cp.asnumpy(div)

    for k, T in T_OBJ.items():
        N_PASOS[k] = max(int(round(T / dt)), 1)
    N_PASOS["proyeccion"] = N_PROYECCIONES
    print("pasos por caso:", N_PASOS)

    campos = {}
    guarda(campos, "inicial")

    # --- adveccion sola ----------------------------------------------------
    restaura()
    for i in range(N_PASOS["adveccion"]):
        mesh.advect_velocities(dt)
        mesh.apply_boundaries(after_projection=False)
        if i % 50 == 0 and not sano():
            print(f"[adveccion] parado en el paso {i}: campo fuera de rango")
            N_PASOS["adveccion"] = i
            break
    guarda(campos, "adveccion")

    # --- difusion sola -----------------------------------------------------
    restaura()
    for i in range(N_PASOS["difusion"]):
        mesh.diffuse_velocity(nu, dt, usar_wale=False, nu_t_field=mesh.sa_nu_t)
        mesh.apply_boundaries(after_projection=False)
        if i % 50 == 0 and not sano():
            print(f"[difusion] parado en el paso {i}: campo fuera de rango")
            N_PASOS["difusion"] = i
            break
    guarda(campos, "difusion")

    # --- adveccion + difusion sin proyectar --------------------------------
    restaura()
    for i in range(N_PASOS["sin_proyeccion"]):
        mesh.advect_velocities(dt)
        mesh.apply_boundaries(after_projection=False)
        mesh.diffuse_velocity(nu, dt, usar_wale=False, nu_t_field=mesh.sa_nu_t)
        mesh.apply_boundaries(after_projection=False)
        if i % 50 == 0 and not sano():
            print(f"[sin_proyeccion] parado en el paso {i}: campo fuera de rango")
            N_PASOS["sin_proyeccion"] = i
            break
    guarda(campos, "sin_proyeccion")

    # --- proyeccion sola: parte del campo sin proyectar de arriba ----------
    # (mismo estado de entrada que el caso anterior, para que la comparacion sea
    #  exactamente la de aplicar o no aplicar el operador)
    for k in ("u", "v", "p", "div"):
        campos[f"{k}_pre_proyeccion"] = campos[f"{k}_sin_proyeccion"]
    for _ in range(N_PROYECCIONES):
        proyecta()
    guarda(campos, "proyeccion")

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(
        NPZ, X=cp.asnumpy(mesh.X_1d), Y=cp.asnumpy(mesh.Y_1d),
        solid=cp.asnumpy(mesh.solid), dt=dt, dx=dx, alpha=ALPHA,
        **{f"n_{k}": v for k, v in N_PASOS.items()}, **campos)
    print("->", NPZ)


# ----------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------
def _rejilla(d):
    X, Y = d["X"], d["Y"]
    solid = d["solid"].astype(bool)
    jj = np.where(solid.any(axis=0))[0]
    ii = np.where(solid.any(axis=1))[0]
    xc, cuerda = X[jj[0]], X[jj[-1]] - X[jj[0]]
    yc = 0.5 * (Y[ii[0]] + Y[ii[-1]])
    x0, x1, y0, y1 = VENTANA
    x = xc + np.linspace(x0, x1, 640) * cuerda
    y = yc + np.linspace(y0, y1, 460) * cuerda
    return X, Y, solid, x, y, (x - xc) / cuerda, (y - yc) / cuerda


def panel_velocidad(fig, ax, xr, yr, U, V, lim, titulo):
    im = ax.pcolormesh(xr, yr, np.hypot(U, V), cmap="RdBu_r", shading="auto",
                       norm=TwoSlopeNorm(vmin=1.0 - lim, vcenter=1.0, vmax=1.0 + lim))
    ax.streamplot(xr, yr, np.nan_to_num(U), np.nan_to_num(V), color="0.20",
                  linewidth=0.40, density=1.15, arrowsize=0.55)
    fig.colorbar(im, ax=ax, pad=0.02).set_label(r"$|\mathbf{u}|$   ($U_\infty=1$)")
    ax.set_title(titulo, fontsize=10)


def panel_divergencia(fig, ax, xr, yr, D, lim, titulo):
    im = ax.pcolormesh(xr, yr, D, cmap="RdBu_r", shading="auto",
                       norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
    fig.colorbar(im, ax=ax, pad=0.02).set_label(r"$\nabla\!\cdot\!\mathbf{u}$")
    ax.set_title(titulo, fontsize=10)


def figura(d, clave, titulo, subtitulo, escalas):
    X, Y, solid, x, y, xr, yr = _rejilla(d)
    fondo = np.where(solid, np.nan, 1.0)
    dt, n = float(d["dt"]), int(d[f"n_{clave}"])

    antes = "pre_proyeccion" if clave == "proyeccion" else "inicial"

    def campo(k, c):
        return remuestrea(X, Y, d[f"{k}_{c}"] * fondo, x, y)

    U0, V0, D0 = campo("u", antes), campo("v", antes), campo("div", antes)
    U1, V1, D1 = campo("u", clave), campo("v", clave), campo("div", clave)
    S = remuestrea(X, Y, solid.astype(float), x, y)

    fig, axs = plt.subplots(2, 3, figsize=(15.6, 8.2))
    fig.subplots_adjust(left=0.045, right=0.965, top=0.865, bottom=0.065,
                        wspace=0.28, hspace=0.16)

    lim_u, lim_d = escalas["umax_dev"], escalas["divmax"]
    et_antes = ("tras advección + difusión sin proyectar"
                if clave == "proyeccion" else "campo desarrollado (partida)")
    et_desp = (f"tras {n} proyecciones" if clave == "proyeccion"
               else f"tras {n} pasos ({n * dt:.2f} t. convectivos)")

    panel_velocidad(fig, axs[0, 0], xr, yr, U0, V0, lim_u, f"|u| — {et_antes}")
    panel_velocidad(fig, axs[0, 1], xr, yr, U1, V1, lim_u, f"|u| — {et_desp}")

    dU, dV = U1 - U0, V1 - V0
    dmag = np.hypot(dU, dV)
    vmax = float(np.nanpercentile(dmag, 99.5))
    ax = axs[0, 2]
    im = ax.pcolormesh(xr, yr, dmag, cmap="magma", shading="auto", vmin=0.0, vmax=vmax)
    k = 16
    ax.quiver(xr[::k], yr[::k], np.nan_to_num(dU)[::k, ::k],
              np.nan_to_num(dV)[::k, ::k], color="w", scale=vmax * 26.0,
              scale_units="width", width=0.0032, alpha=0.85)
    fig.colorbar(im, ax=ax, pad=0.02).set_label(r"$|\Delta\mathbf{u}|$ acumulado")
    ax.set_title(rf"cambio del operador (máx $={float(np.nanmax(dmag)):.2f}$)",
                 fontsize=10)

    def rms(A):
        return float(np.sqrt(np.nanmean(A[np.isfinite(A)] ** 2)))

    panel_divergencia(fig, axs[1, 0], xr, yr, D0, lim_d,
                      rf"$\nabla\!\cdot\!\mathbf{{u}}$ antes (rms $={rms(D0):.2e}$)")
    panel_divergencia(fig, axs[1, 1], xr, yr, D1, lim_d,
                      rf"$\nabla\!\cdot\!\mathbf{{u}}$ después (rms $={rms(D1):.2e}$)")

    ax = axs[1, 2]
    linea = clave != "proyeccion"
    if linea:
        # corte vertical de u: es donde mejor se lee lo que ha hecho el operador
        # (la capa limite engorda con la difusion, se estira con la adveccion)
        # el corte va en la estela, no sobre el perfil: en x/c<1 la vertical
        # atraviesa el solido y la curva sale partida
        jx = int(np.argmin(np.abs(xr - 1.20)))
        ax.plot(U0[:, jx], yr, color="0.35", lw=1.8, label="antes")
        ax.plot(U1[:, jx], yr, color="#c0392b", lw=1.8, label="después")
        ax.axvline(1.0, color="0.75", lw=0.9, ls=":")
        ax.set_xlabel(r"$u$   en $x/c=1.20$  (estela)", fontsize=9)
        ax.set_ylabel("y/c", fontsize=9)
        ax.set_ylim(-0.35, 0.35)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, loc="lower right")
        ax.set_title("perfil de velocidad en la sección", fontsize=10)
        ax.set_box_aspect(0.62)
    else:
        # la presion es el producto de este operador; los otros dos no la tocan
        P1 = campo("p", clave)
        lim_p = max(float(np.nanpercentile(np.abs(P1[np.isfinite(P1)]), 99.0)),
                    escalas["pmax"])
        im = ax.pcolormesh(xr, yr, P1, cmap="RdBu_r", shading="auto",
                           norm=TwoSlopeNorm(vmin=-lim_p, vcenter=0.0, vmax=lim_p))
        fig.colorbar(im, ax=ax, pad=0.02).set_label(r"$p/\rho$")
        ax.set_title(rf"presión resuelta   ($\pm{lim_p:.2f}$)", fontsize=10)

    x0, x1, y0, y1 = VENTANA
    campos_2d = [a for a in axs.ravel() if not (a is axs[1, 2] and linea)]
    for ax in campos_2d:
        ax.contour(xr, yr, np.nan_to_num(S), levels=[0.5, 1.5], colors="k",
                   linewidths=1.2)
        ax.set_aspect("equal")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    for ax in axs[1]:
        if ax in campos_2d:
            ax.set_xlabel("x/c", fontsize=9)
    for ax in axs[:, 0]:
        ax.set_ylabel("y/c", fontsize=9)

    fig.suptitle(f"{titulo}\n{subtitulo}", fontsize=13.5, y=0.975)
    ruta = os.path.join(OUT, f"fig_operador_{clave}.png")
    fig.savefig(ruta, dpi=DPI)
    plt.close(fig)
    print("->", ruta)


def figuras():
    d = dict(np.load(NPZ))
    solid = d["solid"].astype(bool)
    mag = np.hypot(d["u_inicial"], d["v_inicial"])[~solid]
    escalas = {
        "umax_dev": float(np.nanpercentile(np.abs(mag - 1.0), 99.0)),
        "pmax": float(np.nanpercentile(np.abs(d["p_inicial"][~solid]), 99.0)),
        "divmax": float(np.nanpercentile(np.abs(d["div_inicial"][~solid]), 99.0)),
    }
    print(escalas)
    for clave, titulo, sub in CASOS:
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

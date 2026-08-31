"""
Figura 4.A — Convergencia de la proyeccion dentro de un paso temporal.

Se lleva el caso de referencia (perfil optimizado, dx=0.004, alpha=4, dominio
24x16, Re=1e5, configuracion de produccion con las cinco optimizaciones) hasta un
estado desarrollado, y ahi se congela el campo. Desde ese mismo estado se repite
UN paso temporal completo —adveccion, SA, difusion, proyeccion— variando solo el
numero de iteraciones externas de la proyeccion, de 1 a 12, con los mismos 3
ciclos en V por iteracion externa y el modo adaptativo desactivado para que el
presupuesto sea exactamente el pedido.

De cada repeticion se miden dos cosas sobre el mismo campo final: la divergencia
media y el coeficiente de sustentacion, este ultimo con la misma integral de
esfuerzos que usa el bucle principal.

Lo que la figura enseña: las dos curvas responden al presupuesto a ritmos muy
distintos. Multiplicando el presupuesto por 12 la divergencia media solo baja un
35 % —de 2.01e-3 a 1.31e-3, con la pendiente ya casi plana— mientras el Cl se
mueve un 10.9 % —de 0.6733 a 0.7514— y sigue subiendo en el ultimo punto. O sea
que la divergencia media no certifica que el presupuesto sea suficiente para las
fuerzas: es la base de la incertidumbre de presupuesto de proyeccion del
capitulo 4.

Los resultados se guardan en JSON: la figura se rehace sin repetir la corrida.

Uso:
    .venv/bin/python scripts/agent_tests/fig_mem_proyeccion.py [--solo-figura]
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "agent_tests"))
os.chdir(ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(ROOT, "figuras_memoria")
JSON = os.path.join(OUT, "datos_4A_proyeccion.json")
DPI = 200

DX, ALPHA, ITERS = 0.004, 4.0, 6000     # t ~ 9.6 con CFL=0.5
OUTERS = (1, 2, 3, 4, 5, 6, 8, 10, 12)
CICLOS = 3                               # los mismos que el presupuesto 2x3


def corre():
    import cupy as cp
    import RunGA
    import Simulador2D
    import opt_solver
    import resultados_finales as RF
    import verificacion_numerica as vn

    opt_solver.reset()
    opt_solver.enable(*RF.FLAGS)
    assert opt_solver.active() == sorted(RF.FLAGS), opt_solver.active()

    cfg = dict(RunGA.CONFIG)
    rho, nu = 1.0, 1.0 / RF.RE
    sim = {
        "filepath": os.path.join(RF.OUT, "perfiles", "ganador_ag.dat"),
        "iteraciones": ITERS,
        "v0x": 1.0, "v0y": 0.0, "CFL": vn.CFL,
        "dx_min": DX, "alpha_deg": ALPHA, "chord": 1.0,
        "rho": rho, "nu": nu,
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
    t0 = time.time()
    mesh = Simulador2D.main(**sim)
    print(f"estado desarrollado en {time.time() - t0:.0f} s")

    # --- estado congelado -------------------------------------------------
    campos = ("u", "v", "p", "nu_tilde", "sa_nu_t", "sa_gamma")
    estado = {k: (getattr(mesh, k).copy() if getattr(mesh, k, None) is not None
                  else None) for k in campos}
    semilla = getattr(mesh, "_opt_p_warm", None)
    semilla = semilla.copy() if semilla is not None else None
    semilla_dt = getattr(mesh, "_opt_p_warm_dt", None)

    def restaura():
        for k, v in estado.items():
            if v is not None:
                setattr(mesh, k, v.copy())
        if semilla is not None:
            setattr(mesh, "_opt_p_warm", semilla.copy())
            if semilla_dt is not None:
                setattr(mesh, "_opt_p_warm_dt", semilla_dt)

    # --- dt del paso, con el mismo criterio del bucle principal ------------
    speed = cp.sqrt(mesh.u * mesh.u + mesh.v * mesh.v)
    Umax = float(cp.max(speed[~mesh.solid]))
    dt_adv = vn.CFL * min(mesh.dx, mesh.dy) / max(Umax, 1e-12)
    nu_eff = nu + float(cp.max(mesh.sa_nu_t)) if mesh.sa_nu_t is not None else nu
    dt_visc = 0.25 * min(mesh.dx, mesh.dy) ** 2 / max(nu_eff, 1e-30)
    dt = float(min(dt_adv, dt_visc))
    print(f"dt del paso = {dt:.3e}  (adv {dt_adv:.3e}, visc {dt_visc:.3e})")

    mu = rho * nu
    filas = []
    for n in OUTERS:
        restaura()
        mesh.advect_velocities(dt)
        mesh.advect_sa()
        mesh.update_sa(nu, dt)
        mesh.apply_boundaries(after_projection=False)
        mesh.diffuse_velocity(nu, dt, usar_wale=False, nu_t_field=mesh.sa_nu_t)
        mesh.apply_boundaries(after_projection=False)

        info = mesh.project_multigrid(
            rho, dt,
            tol_div=None, tol_div_rel=cfg.get("divergencia", 0.02),
            max_outer=n, cycles_per_outer=CICLOS,
            niveles_max=cfg.get("mg_niveles_max", 2),
            pre_suavizado=3, post_suavizado=3,
            modo_adaptativo=False,          # el presupuesto es el que se pide
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

        f = mesh.compute_drag_lift(mu, rho=rho, n_extrap_layers=5)
        cl = 2 * f["Lift"] / (rho * 1.0 ** 2 * 1.0)
        cd = 2 * f["Drag"] / (rho * 1.0 ** 2 * 1.0)
        d = np.abs(np.asarray(
            __import__("cupy").asnumpy(mesh._compute_flux_divergence_field_uv())))
        solid = __import__("cupy").asnumpy(mesh.solid).astype(bool)
        div_media = float(d[~solid].mean())

        filas.append({"outers": n, "ciclos": n * CICLOS,
                      "div_antes": info["div_before"], "div_despues": info["div_after"],
                      "div_media_campo": div_media, "cl": float(cl), "cd": float(cd)})
        print(f"  outers={n:2d}  div {info['div_before']:.3e} -> "
              f"{info['div_after']:.3e}   Cl={cl:+.5f}  Cd={cd:+.5f}")

    os.makedirs(OUT, exist_ok=True)
    json.dump({"dx": DX, "alpha": ALPHA, "iters": ITERS, "dt": dt,
               "ciclos_por_outer": CICLOS, "Re": RF.RE,
               "dominio": RF.DOMINIO, "flags": list(RF.FLAGS),
               "puntos": filas},
              open(JSON, "w"), indent=2)
    print("->", JSON)


def figura():
    d = json.load(open(JSON))
    f = d["puntos"]
    n = np.array([r["outers"] for r in f], float)
    div = np.array([r["div_despues"] for r in f], float)
    cl = np.array([r["cl"] for r in f], float)

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    fig.subplots_adjust(left=0.155, right=0.855, top=0.855, bottom=0.185)
    ax2 = ax.twinx()

    l1, = ax.plot(n, div, "o-", color="#1f5fa8", lw=2.0, ms=7,
                  label="divergencia media del campo")
    ax.set_yscale("log")
    ax.set_xlabel("iteraciones externas de la proyección", fontsize=12)
    ax.set_ylabel(r"$\overline{|\nabla\!\cdot\!\mathbf{u}|}$  tras el paso",
                  fontsize=12, color="#1f5fa8")
    ax.tick_params(axis="y", colors="#1f5fa8")
    ax.grid(alpha=0.25)

    l2, = ax2.plot(n, cl, "s--", color="#c0392b", lw=2.0, ms=7,
                   label="coeficiente de sustentación")
    ax2.set_ylabel(r"$C_l$", fontsize=13, color="#c0392b")
    ax2.tick_params(axis="y", colors="#c0392b")

    # presupuesto de produccion
    ax.axvline(2, color="0.4", lw=1.1, ls=":")
    ax.text(2.12, div.max(), " presupuesto\n de producción", fontsize=9.5,
            color="0.35", va="top")
    ax.set_xticks(list(OUTERS))
    ax.set_xticklabels([str(v) for v in OUTERS])

    d_rel = 100 * (div.max() - div.min()) / div.max()
    cl_rel = 100 * (cl.max() - cl.min()) / abs(np.mean(cl))
    ax.legend(handles=[l1, l2], loc="center right", fontsize=10, framealpha=0.95)

    ax.set_title("Un paso temporal repetido con presupuestos crecientes\n"
                 f"dx={d['dx']:g}, "
                 rf"$\alpha={d['alpha']:.0f}^\circ$, "
                 f"{d['ciclos_por_outer']} ciclos en V por iteración externa",
                 fontsize=11.5, pad=9)
    fig.text(0.5, 0.058,
             f"multiplicar el presupuesto por {int(n.max())} baja la divergencia "
             f"un {d_rel:.0f} % (de {div[0]:.2e} a {div[-1]:.2e})",
             ha="center", fontsize=9.5, color="0.3")
    fig.text(0.5, 0.016,
             rf"y mueve el $C_l$ un {cl_rel:.1f} %, de {cl[0]:.4f} a {cl[-1]:.4f}, "
             "sin señal de haberse estabilizado",
             ha="center", fontsize=9.5, color="0.3")

    p = os.path.join(OUT, "fig_4A_convergencia_proyeccion.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print(f"divergencia {div[0]:.3e} -> {div[-1]:.3e}   Cl {cl[0]:.5f} -> {cl[-1]:.5f}")
    print("->", p)


if __name__ == "__main__":
    if "--solo-figura" not in sys.argv:
        corre()
    figura()

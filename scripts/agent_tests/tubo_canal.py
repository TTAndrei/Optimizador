"""Flujo en un canal 2D: paredes superior e inferior no-slip, dominio vacio,
inicialmente en reposo, con el flujo entrando por la izquierda (llenado de tubo).

POR QUE
  Es el unico caso del proyecto con solucion analitica. Sirve para dos cosas:
    1. Validacion: a Reynolds bajo el perfil desarrollado es Poiseuille exacto,
       u(y) = 1.5*U*(1-(2y/H-1)^2), con dp/dx = -12*nu*U/H^2 y longitud de
       entrada Le/H ~ 0.05*Re. Contrastar contra eso mide el solver sin IBM
       de por medio, o sea sin el sesgo del inmersed boundary.
    2. Transitorio: el frente de llenado avanzando por un tubo en reposo.

COMO
  - `filepath=None`: dominio sin solido. Requiere el guardado de
    `generar_graficos_y_outputs`, que sin solido no puede calcular circulacion
    ni Cp.
  - `v0x_ic=0, v0y_ic=0` ponen TODA la malla en reposo sin tocar las fronteras,
    que siguen gobernadas por v0x (Simulador2D.py:381-391). El dt inicial se
    calcula de v0x, no de la IC, asi que arrancar parado no lo rompe.
  - Arranque impulsivo: no hay rampa de entrada en el solver. Es justo el frente
    de llenado que se busca, pero el primer tramo del transitorio es un golpe de
    presion y no debe leerse como fisica.
  - Malla uniforme (zona fina = dominio entero): sin perfil no hay nada alrededor
    de lo que refinar, y el estirado solo meteria error.
  - Los frames NO se generan en el solver: un canal de 30x1 sale en una figura
    de relacion 30:1 ilegible. Se vuelcan campos periodicos y los renderiza
    render_videos.py con la escala vertical exagerada.

Uso:
    .venv/bin/python scripts/agent_tests/tubo_canal.py
    .venv/bin/python scripts/agent_tests/tubo_canal.py --caso T1
    .venv/bin/python scripts/agent_tests/tubo_canal.py --solo-analisis
Parada limpia: crear PARAR_FORMAS.trigger en la raiz.
"""
from __future__ import annotations

import argparse
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

import opt_solver

OUT = os.path.join(ROOT, "results", "tubo")
TRIGGER = os.path.join(ROOT, "PARAR_FORMAS.trigger")
FLAGS = ("warm_start", "warm_start_filtered", "coarse_mask_majority",
         "fast_masks", "interp_float32")
CFL = 0.5
GUARDADO = 50
N_FRAMES = 200
U = 1.0

# t_obj: validacion = 1.5*Lx (llenar) + 2*Le (desarrollar), con Le = 0.05*Re*H.
# Llenado = 1.5*Lx, que basta para ver el frente recorrer el tubo.
CASOS = {
    "T1": dict(Lx=30.0, Ly=1.0, Re=100.0,   dx=0.010, sa=False, t_obj=55.0,
               tipo="validacion"),
    "T2": dict(Lx=30.0, Ly=1.0, Re=500.0,   dx=0.010, sa=False, t_obj=80.0,
               tipo="validacion"),
    "T3": dict(Lx=20.0, Ly=2.0, Re=1000.0,  dx=0.010, sa=False, t_obj=30.0,
               tipo="llenado"),
    "T4": dict(Lx=20.0, Ly=2.0, Re=1e4,     dx=0.010, sa=True,  t_obj=30.0,
               tipo="llenado"),
    "T5": dict(Lx=10.0, Ly=4.0, Re=1e4,     dx=0.010, sa=False, t_obj=20.0,
               tipo="llenado"),
    "T6": dict(Lx=20.0, Ly=1.0, Re=1e5,     dx=0.005, sa=True,  t_obj=30.0,
               tipo="llenado"),
    # --- Escalera de malla en UN MISMO canal, para medir si refinar mejora ---
    # A Re=1e3 la longitud de entrada es Le = 0.05*Re*H = 50*H, o sea el tubo
    # debe medir ~80 alturas. Con eso el coste NO depende de la altura, solo de
    # las celdas a lo ancho N = H/dx: celdas = 80*N^2 e iteraciones ~ 0.75*Re*N,
    # asi que doblar N cuesta 8 veces mas. Se fija H=0.2 y se barre dx, con lo
    # que N = 40 / 100 / 200 (T1 tenia 100).
    "T7": dict(Lx=16.0, Ly=0.2, Re=1e3,     dx=0.005, sa=False, t_obj=80.0,
               tipo="validacion"),
    "T8": dict(Lx=16.0, Ly=0.2, Re=1e3,     dx=0.002, sa=False, t_obj=80.0,
               tipo="validacion"),
    # Re=1e6 no se desarrolla nunca (Le = 10000 en un tubo de 16): es transitorio
    # de llenado, y por eso le basta con t = 1.5*Lx.
    "T10": dict(Lx=16.0, Ly=0.2, Re=1e6,    dx=0.001, sa=False, t_obj=20.0,
                tipo="llenado"),
    "T9": dict(Lx=16.0, Ly=0.2, Re=1e3,     dx=0.001, sa=False, t_obj=80.0,
               tipo="validacion"),
}
PRIOR_S = {"T1": 500, "T2": 700, "T3": 500, "T4": 700, "T5": 400, "T6": 1200,
           "T7": 500, "T8": 3500, "T9": 10500, "T10": 5200}


def log(msg):
    linea = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(linea, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ejecucion.log"), "a") as f:
        f.write(linea + "\n")


def guardar(path, d):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False, default=float)
    os.replace(tmp, path)


def params(c, iters, campos_dir, frame_every):
    H = c["Ly"]
    nu = U * H / c["Re"]
    modelo = ({"turb_model": "sa", "transition_model": "sa_bc", "freestream_Tu": 0.1}
              if c["sa"] else {"turb_model": None, "transition_model": "none"})
    return dict(
        filepath=None, chord=1.0, alpha_deg=0.0,
        Lx=c["Lx"], Ly=c["Ly"], cx=c["Lx"] * 0.5, cy=c["Ly"] * 0.5,
        dx_min=c["dx"], factor_expansion=1.02,
        ancho_zona_fina_x=c["Lx"], ancho_zona_fina_y=c["Ly"],
        v0x=U, v0y=0.0, v0x_ic=0.0, v0y_ic=0.0,
        rho=1.0, nu=nu, p0=0.0, CFL=CFL,
        boundary_left=("inflow", None), boundary_right=("outflow", 0.0),
        boundary_top=("noslip", None), boundary_bottom=("noslip", None),
        advection_scheme="maccormack", wall_treatment="consistent",
        stop_on_convergence=False, stop_on_clcd_convergence=False,
        mg_niveles_max=2, mg_max_outer=2, mg_cycles_per_outer=3,
        iteraciones=iters, guardado=GUARDADO,
        dump_fields_dir=campos_dir, dump_fields_cada=frame_every,
        dump_fields_max_nx=1200,
        graficos=False, live_view=False, mostrar_malla=False,
        **modelo,
    )


def analiza(mesh, c):
    """Perfiles, longitud de entrada y gradiente de presion contra Poiseuille."""
    import cupy as cp
    u = cp.asnumpy(mesh.u).astype(float)
    p = cp.asnumpy(mesh.p).astype(float)
    x = cp.asnumpy(mesh.X_1d).astype(float)
    y = cp.asnumpy(mesh.Y_1d).astype(float)
    H, nu = c["Ly"], U * c["Ly"] / c["Re"]
    jc = int(np.argmin(np.abs(y - 0.5 * H)))

    u_centro = u[jc, :]
    u_max_teo = 1.5 * U
    # DOS longitudes de entrada, porque miden cosas distintas:
    #  - `le` (absoluta): primer x donde la linea central alcanza el 99% de 1.5U,
    #    el maximo EXACTO de Poiseuille. Es la comparacion honesta contra la
    #    teoria, pero se vuelve None en cuanto la malla no llega a 1.5 aunque el
    #    perfil este perfectamente desarrollado (medido: N=40 satura en 1.4837).
    #  - `le_rel`: mismo 99% pero del valor asintotico QUE ALCANZA esta malla.
    #    Mide donde acaba la region de entrada con independencia de cuanto se
    #    quede corta la malla en el pico, que es otro error y va aparte en
    #    `err_u_centro_pct`.
    ok = np.where(u_centro >= 0.99 * u_max_teo)[0]
    le = float(x[ok[0]]) if len(ok) else None
    u_asint = float(u_centro[-1])
    ok_r = np.where(u_centro >= 0.99 * u_asint)[0]
    le_rel = float(x[ok_r[0]]) if len(ok_r) else None

    # dp/dx en la mitad de aguas abajo, donde el flujo deberia estar desarrollado
    i0, i1 = int(0.6 * len(x)), int(0.95 * len(x))
    p_cl = p[jc, i0:i1]
    dpdx = float(np.polyfit(x[i0:i1], p_cl, 1)[0])
    dpdx_teo = -12.0 * nu * U / H**2

    # Perfil en estaciones x/H, contra Poiseuille
    eta = y / H
    pois = 1.5 * U * (1.0 - (2.0 * eta - 1.0) ** 2)
    perfiles = {}
    for xr in (1.0, 2.0, 5.0, 10.0, 20.0):
        xp = xr * H
        if xp > x[-1]:
            continue
        i = int(np.argmin(np.abs(x - xp)))
        up = u[:, i]
        perfiles[f"x/H={xr:g}"] = {
            "u_centro": round(float(up[jc]), 4),
            "err_rms_vs_poiseuille": round(float(np.sqrt(np.mean((up - pois) ** 2)) / U), 4),
            "caudal": round(float(np.trapezoid(up, y) / H), 4),
        }
    # La comparacion contra Poiseuille solo tiene sentido si el flujo llego a
    # desarrollarse dentro del dominio; si no, se reporta como n/a en vez de dar
    # un error del 5000% que no dice nada.
    # Desarrollado = la linea central alcanza el maximo de Poiseuille ANTES de
    # la salida. Atarlo a "Le < 0.8*Lx" descartaba T2, que llega a 1.4884 (99.2 %
    # de 1.5U) con Le=26.6 en un tubo de 30: desarrollado de sobra, solo que sin
    # margen de cortesia por detras.
    # Desarrollado exige DOS cosas, y hacen falta las dos:
    #  1. que el perfil haya dejado de cambiar con x (le_rel dentro del tubo), y
    #  2. que lo que quedo se parezca a Poiseuille (pico >= 95% de 1.5U).
    # Sin la segunda, un caso de llenado con perfil casi plano (u_c ~ 1.03) pasa
    # el primer filtro en x~0.3 —su asintota es plana desde el principio— y
    # reactiva la comparacion contra Poiseuille, que ahi da +130000%.
    desarrollado = bool(le_rel is not None and le_rel < 0.9 * c["Lx"]
                        and u_asint >= 0.95 * u_max_teo)
    return {
        "Le_medida": le,
        "Le_relativa": le_rel,
        "Le_teorica_005ReH": 0.05 * c["Re"] * H,
        "celdas_a_lo_ancho": int(round(H / c["dx"])),
        "err_u_centro_pct": round(100.0 * (u_asint / u_max_teo - 1.0), 3),
        "desarrollado": desarrollado,
        "u_centro_salida": round(float(u_centro[-1]), 4),
        "u_centro_teorico": u_max_teo,
        "dpdx_medido": dpdx, "dpdx_teorico": dpdx_teo,
        "err_dpdx_pct": (round(100.0 * (dpdx / dpdx_teo - 1.0), 2)
                         if desarrollado and dpdx_teo else None),
        "u_pared_max": round(float(np.abs(u[[0, -1], :]).max()), 6),
        "perfiles": perfiles,
    }


def figuras(mesh, c, tag, d):
    import cupy as cp
    u = cp.asnumpy(mesh.u).astype(float)
    x = cp.asnumpy(mesh.X_1d).astype(float)
    y = cp.asnumpy(mesh.Y_1d).astype(float)
    H = c["Ly"]
    fig, ax = plt.subplots(2, 1, figsize=(11, 7))
    eta, pois = y / H, 1.5 * U * (1.0 - (2.0 * y / H - 1.0) ** 2)
    for xr in (1.0, 2.0, 5.0, 10.0, 20.0):
        if xr * H > x[-1]:
            continue
        i = int(np.argmin(np.abs(x - xr * H)))
        ax[0].plot(u[:, i] / U, eta, label=f"x/H={xr:g}")
    ax[0].plot(pois / U, eta, "k--", lw=2, label="Poiseuille")
    ax[0].set_xlabel("u/U"); ax[0].set_ylabel("y/H"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"{tag}: perfiles  (Re_H={c['Re']:.0f}, {c['Lx']:g}x{c['Ly']:g})")
    jc = int(np.argmin(np.abs(y - 0.5 * H)))
    ax[1].plot(x / H, u[jc, :] / U)
    ax[1].axhline(1.5, ls="--", c="k", label="1.5 U (Poiseuille)")
    if d["Le_medida"]:
        ax[1].axvline(d["Le_medida"] / H, ls=":", c="r", label="Le medida")
    ax[1].axvline(d["Le_teorica_005ReH"] / H, ls=":", c="g", label="0.05 Re H")
    ax[1].set_xlabel("x/H"); ax[1].set_ylabel("u_centro/U"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, tag, "figuras")
    os.makedirs(p, exist_ok=True)
    fig.savefig(os.path.join(p, "perfiles.png"), dpi=140)
    plt.close(fig)


def corre_caso(tag):
    from Simulador2D import main as sim_main
    import cupy as cp
    c = CASOS[tag]
    d = os.path.join(OUT, tag)
    js = os.path.join(d, "tubo.json")
    if os.path.exists(js):
        log(f"{tag} YA HECHO")
        return True
    os.makedirs(d, exist_ok=True)

    dt_est = CFL * c["dx"] / (1.5 * U)
    iters = int(np.clip(round(c["t_obj"] / dt_est / 1000.0) * 1000, 2000, 120000))
    frame_every = max(GUARDADO, int(round(iters / N_FRAMES / GUARDADO)) * GUARDADO)
    log(f"{tag}: {c['Lx']:g}x{c['Ly']:g} Re={c['Re']:.0e} dx={c['dx']} "
        f"{'sa_bc' if c['sa'] else 'laminar'} -> {iters} iters (t_obj={c['t_obj']:g})")

    opt_solver.reset()
    opt_solver.enable(*FLAGS)
    t0 = time.time()
    mesh = sim_main(**params(c, iters, os.path.join(d, "campos_t"), frame_every))
    wall = round(time.time() - t0, 1)

    r = {"caso": tag, **{k: v for k, v in c.items()},
         "nu": U * c["Ly"] / c["Re"], "iters": iters, "wall_s": wall,
         "t_final": round(float(mesh._t_fisico), 3),
         "dt_medio": float(mesh._t_fisico / iters),
         "modelo": "sa+sa_bc" if c["sa"] else "laminar",
         **analiza(mesh, c)}
    np.savez_compressed(
        os.path.join(d, "campo.npz"),
        x=cp.asnumpy(mesh.X_1d), y=cp.asnumpy(mesh.Y_1d),
        u=cp.asnumpy(mesh.u), v=cp.asnumpy(mesh.v), p=cp.asnumpy(mesh.p))
    figuras(mesh, c, tag, r)
    guardar(js, r)
    log(f"  -> t={r['t_final']}  N={r['celdas_a_lo_ancho']}  Le={r['Le_medida']} / "
        f"rel {r['Le_relativa']} (teo {r['Le_teorica_005ReH']:.1f})  "
        f"u_c={r['u_centro_salida']} (err {r['err_u_centro_pct']}%)  "
        f"desarrollado={r['desarrollado']}  err dp/dx={r['err_dpdx_pct']}%  ({wall}s)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caso", default=None)
    ap.add_argument("--horas", type=float, default=None)
    args = ap.parse_args()
    casos = [args.caso] if args.caso else list(CASOS)
    t0 = time.time()
    log(f"=== tubo: {len(casos)} casos ===")
    for tag in casos:
        if os.path.exists(TRIGGER):
            os.remove(TRIGGER)
            log("PARAR_FORMAS.trigger: parada limpia")
            break
        if args.horas and (time.time() - t0) + PRIOR_S[tag] > args.horas * 3600:
            log(f"plazo agotado, queda {tag} en adelante")
            break
        try:
            corre_caso(tag)
        except Exception as e:
            log(f"{tag} FALLO: {type(e).__name__}: {e}")
    log(f"=== fin tubo en {(time.time()-t0)/3600:.2f} h ===")


if __name__ == "__main__":
    main()

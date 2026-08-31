"""
Test de geometria: cilindro (seccion de esfera) de 1 m de diametro en el sitio
que ocupaba el perfil, en la malla dx=0.002 y con el resto de la configuracion
del estudio final intacta (dominio 24x16 en cx=6, Re=1e5, SA + transicion sa_bc,
MacCormack, wall_treatment consistent, presupuesto MG 2x3).

Presupuesto: t=20 tiempos convectivos como maximo, con la parada por
convergencia ACTIVA.

DOS AVISOS SOBRE LA PARADA
  1. `stop_on_convergence` (criterio de campo u/v/p) se deja en False. Compara el
     campo contra el de hace 50 iteraciones, que a dx=0.002 son 0.02 tiempos
     convectivos: con dt tan pequeno el cambio relativo baja del 1% en pleno
     transitorio y cortaria el run en la iteracion ~200. Es la misma razon por la
     que el estudio final lo lleva apagado.
  2. `stop_on_clcd_convergence` SI va activo, con el criterio calibrado. Pero el
     criterio mide L/D, y un cilindro a alpha=0 tiene Cl medio ~0, o sea L/D ~0:
     el CI95 del criterio es sigma/|media| y se dispara. En la practica no va a
     disparar nunca. Por eso el script evalua ADEMAS el mismo criterio offline
     sobre la serie de Cd, que es la magnitud con sentido en un cuerpo romo, y
     guarda donde habria parado.

El numero de iteraciones para t=20 no se puede fijar a priori: dt es adaptativo
y en un cilindro la velocidad maxima es mayor que en un perfil (unos 2U en el
ecuador), asi que las 26000 iteraciones que dan t~20 con perfil se quedarian
cortas. Se mide dt con un run corto de sonda y se extrapola.

Uso:
    .venv/bin/python scripts/agent_tests/esfera_dx002.py
    .venv/bin/python scripts/agent_tests/esfera_dx002.py --solo-figuras
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

OUT = os.path.join(ROOT, "results", "esfera_dx002")
DAT = os.path.join(OUT, "cilindro_D1.dat")
SERIE = os.path.join(OUT, "serie.npz")
CAMPO = os.path.join(OUT, "campo.npz")
SONDA = os.path.join(OUT, "sonda.npz")
JSON = os.path.join(OUT, "esfera_dx002.json")

DX = 0.002
RE = 1e5
ALPHA = 0.0
T_OBJETIVO = 20.0
ITERS_SONDA = 600

DOMINIO = {"Lx": 24.0, "Ly": 16.0, "cx": 6.0}
PRESUPUESTO = {"mg_max_outer": 2, "mg_cycles_per_outer": 3}
PARADAS = {"stop_on_clcd_convergence": True, "stop_on_convergence": False}
FLAGS = ("warm_start", "warm_start_filtered", "coarse_mask_majority",
         "fast_masks", "interp_float32")

# Modelo de turbulencia. Vacio = el del estudio (SA + transicion sa_bc). Con
# --sin-sa se pone a laminar, que es la unica via barata de llegar a t=20: con SA
# el dt lo manda dt_visc = 0.25*dx^2/nu_eff y nu_t satura en 2000*nu en la estela
# del cilindro, asi que el paso cae x8 y el run se queda en t=5.3.
MODELO = {}


def set_out(tag):
    """Reapunta las rutas de salida a results/<tag>/."""
    global OUT, DAT, SERIE, CAMPO, SONDA, JSON
    OUT = os.path.join(ROOT, "results", tag)
    DAT = os.path.join(OUT, "cilindro_D1.dat")
    SERIE = os.path.join(OUT, "serie.npz")
    CAMPO = os.path.join(OUT, "campo.npz")
    SONDA = os.path.join(OUT, "sonda.npz")
    JSON = os.path.join(OUT, f"{tag}.json")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def escribe_cilindro(n=1440):
    """Circulo de diametro 1 en formato Selig: arranca en el borde de salida
    (x=1), sube por el extrados hasta x=0 y vuelve por el intrados."""
    th = np.linspace(0.0, 2.0 * np.pi, n + 1)
    x = 0.5 + 0.5 * np.cos(th)
    y = 0.5 * np.sin(th)
    os.makedirs(OUT, exist_ok=True)
    with open(DAT, "w") as f:
        f.write("Cilindro D=1 (seccion de esfera)\n")
        for xi, yi in zip(x, y):
            f.write(f"  {xi:.8f}  {yi:.8f}\n")
    return DAT


def corre(iters, dump, dump_field):
    import verificacion_numerica as vn
    opt_solver.reset()
    opt_solver.enable(*FLAGS)
    t0 = time.time()
    r = vn.simular(DAT, RE, DX, alpha=ALPHA, iters=iters,
                   extra={**DOMINIO, **PARADAS, **PRESUPUESTO, **MODELO},
                   dump=dump, dump_field=dump_field)
    if r is not None:
        r["wall_s"] = round(time.time() - t0, 1)
    return r


def parada_offline_cd(t, cd):
    """El mismo criterio calibrado, pero sobre Cd en vez de sobre L/D."""
    from Simulador2D import _detect_series_convergence
    import verificacion_numerica as vn
    c = json.load(open(f"{vn.OUT}/criterio_parada.json"))["criterio"]
    for i in range(10, len(t) + 1):
        if t[i - 1] < c["min_t"]:
            continue
        d, ci, ok = _detect_series_convergence(
            t[:i], cd[:i], c["tol_drift"], c["tol_ci95"], c["window"])
        if ok:
            return {"t": round(float(t[i - 1]), 3),
                    "cd": round(float(np.mean(cd[max(0, i - 20):i])), 5),
                    "drift": float(d), "ci95": float(ci),
                    "frac_del_run": round(float(t[i - 1] / t[-1]), 3)}
    return None


def strouhal(t, cl):
    """St = f*D/U con D=1 y U=1, o sea el pico del espectro de Cl."""
    m = t > 0.5 * t[-1]
    if m.sum() < 32:
        return None
    tt, s = t[m], cl[m] - np.mean(cl[m])
    dt = float(np.mean(np.diff(tt)))
    F = np.abs(np.fft.rfft(s * np.hanning(len(s))))
    fr = np.fft.rfftfreq(len(s), dt)
    k = int(np.argmax(F[1:]) + 1)
    return round(float(fr[k]), 4)


def figuras(d):
    z = np.load(SERIE)
    t, cl, cd = z["t"], z["cl"], z["cd"]
    m = np.isfinite(t) & np.isfinite(cd) & (t > 0)
    t, cl, cd = t[m], cl[m], cd[m]

    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
    ax[0].plot(t, cd, color="#1f4e79", lw=0.9)
    ax[0].axhline(d["cd"], color="#c0392b", ls="--", lw=1,
                  label=f"media ultimo 20%: {d['cd']:.4f}")
    ax[0].axhspan(1.1, 1.3, color="#27ae60", alpha=0.15,
                  label="referencia experimental Cd 1.1-1.3 (subcritico)")
    po = d.get("parada_offline_cd")
    if po:
        ax[0].axvline(po["t"], color="#8e44ad", ls=":", lw=1.2,
                      label=f"parada offline sobre Cd: t={po['t']}")
    ax[0].set_ylabel("$C_d$"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"Cilindro D=1, Re=1e5, dx={DX}, dominio 24x16")
    ax[1].plot(t, cl, color="#c0392b", lw=0.9)
    ax[1].set_ylabel("$C_l$"); ax[1].set_xlabel("$t$ (tiempos convectivos)")
    fig.savefig(f"{OUT}/historia_fuerzas.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if os.path.exists(CAMPO):
        c = np.load(CAMPO)
        x, y, u, v = c["x"], c["y"], c["u"], c["v"]
        w = np.gradient(v, x, axis=1) - np.gradient(u, y, axis=0)
        w = np.where(c["solid"], np.nan, w)
        fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
        im = ax.pcolormesh(x, y, w, cmap="RdBu_r", vmin=-20, vmax=20, shading="auto")
        ax.set_xlim(5.0, 12.0); ax.set_ylim(5.0, 11.0); ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, label=r"$\omega_z$")
        ax.set_title("Vorticidad, campo final")
        fig.savefig(f"{OUT}/vorticidad.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-figuras", action="store_true")
    ap.add_argument("--sin-sa", action="store_true",
                    help="turb_model=none: sin modelo de turbulencia ni transicion")
    a = ap.parse_args()

    global MODELO
    if a.sin_sa:
        MODELO = {"turb_model": "none", "transition_model": "none"}
        set_out("esfera_dx002_laminar")

    if a.solo_figuras:
        figuras(json.load(open(JSON)))
        return

    os.makedirs(OUT, exist_ok=True)
    escribe_cilindro()
    log(f"cilindro D=1 escrito en {DAT}")

    log(f"sonda de dt: {ITERS_SONDA} iteraciones")
    rs = corre(ITERS_SONDA, SONDA, None)
    if rs is None:
        log("la sonda fallo"); return
    ts = np.load(SONDA)["t"]
    ts = ts[np.isfinite(ts)]
    dt = float(ts[-1]) / ITERS_SONDA
    iters = int(round(T_OBJETIVO / dt / 1000.0) * 1000)
    log(f"sonda: t={ts[-1]:.3f} en {ITERS_SONDA} iters -> dt={dt:.2e} "
        f"-> {iters} iteraciones para t={T_OBJETIVO:.0f}  ({rs['wall_s']}s)")

    log(f"run principal: {iters} iteraciones, parada Cl/Cd activa")
    r = corre(iters, SERIE, CAMPO)
    if r is None:
        log("el run fallo"); return

    z = np.load(SERIE)
    t, cl, cd = z["t"], z["cl"], z["cd"]
    m = np.isfinite(t) & np.isfinite(cd) & (t > 0)
    t, cl, cd = t[m], cl[m], cd[m]

    r.update(DOMINIO)
    r["geometria"] = "cilindro D=1 (seccion de esfera)"
    r["t_final"] = round(float(t[-1]), 3)
    r["dt_medio"] = float(t[-1] / r["iters_efectivas"])
    r["dt_sonda"] = dt
    r["iters_programadas"] = iters
    r["cl_amplitud"] = round(float(np.std(cl[t > 0.5 * t[-1]])), 5)
    r["strouhal"] = strouhal(t, cl)
    r["parada_offline_cd"] = parada_offline_cd(t, cd)
    r["modelo"] = MODELO or {"turb_model": "sa", "transition_model": "sa_bc"}
    r["optimizaciones"] = list(FLAGS)
    r["paradas"] = dict(PARADAS)
    json.dump(r, open(JSON, "w"), indent=2, ensure_ascii=False)

    figuras(r)
    log(f"Cd={r['cd']:.4f}  Cl={r['cl']:+.4f}  t_final={r['t_final']}  "
        f"conv_LD={r['converged_clcd']}  St={r['strouhal']}  ({r['wall_s']}s)")
    log(f"parada offline sobre Cd: {r['parada_offline_cd']}")
    log(f"resultados en {OUT}")


if __name__ == "__main__":
    main()

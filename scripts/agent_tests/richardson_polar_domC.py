"""
Estudio de Richardson sobre la POLAR del perfil optimizado (AG), no sobre un
solo punto: tres mallas (dx = 0.004 / 0.002 / 0.001, r=2) por cada uno de cinco
angulos (alpha = 0, 2, 4, 6, 8) en el dominio C (Lx=24, Ly=16, cx=6).

Por que la polar y no un angulo suelto: el GCI de tres mallas del ganador
(results/verificacion_numerica/gci_ganador.json) dio Cl monotono con p=1.52
pero Cd y L/D oscilatorios, y con un solo angulo no hay forma de distinguir
"Richardson no aplica" de "este angulo cayo mal". Cinco angulos dan cinco
estimaciones independientes de p: si p sale consistente, la extrapolacion es
citable; si oscila o cambia de signo, lo honesto es publicar la banda entre
mallas y decir por que.

--- Criterio de parada -------------------------------------------------------
DESACTIVADO en las tres mallas, y tambien el guardian de residual de campo
(stop_on_convergence). Las dos paradas son incompatibles con un estudio de
malla: cortan a t distinto en cada malla y meten dependencia de ventana dentro
de la diferencia entre mallas, que es justo lo que se quiere medir. Ademas el
guardian de residual fue el que trunco alpha=8 (10100 de 26000 iters) y
alpha=10 (7950) en results/polar_2grados_dom24x16/.

El criterio calibrado (el ultimo, recalibrado por criterio_ci95.py: puerta de
CI95 en vez de sigma/media -> tol_drift=0.002, tol_ci95=0.02, window=1.0,
n_sostenido=3, min_t=5.0) SI se pasa al solver, para que registre su veredicto
(converged_clcd / t_conv_clcd) sin actuar; y ademas se reproduce offline sobre
la serie volcada de cada punto, con la funcion del propio solver. Asi queda
medido donde habria parado sin haber pagado el sesgo de que parara.

--- Presupuesto de iteraciones ----------------------------------------------
t ~ 20 tiempos convectivos en las tres mallas (13000 / 26000 / 52000 iters).
Suficiente, medido por dos vias independientes:

  * dx=0.002, alpha=4, dominio C hasta t=41.5 (results/asintotico_alpha4_domC):
    L/D de cola 34.038 en t=20 frente a 34.049 en t=41.5 -> 0.03%.
  * dx=0.001 hasta t=71 (ventana_larga_dx001_series.npz), media de cola:
    t=20 -> Cl 0.6865 / Cd 0.02529 / L/D 27.15
    t=71 -> Cl 0.6878 / Cd 0.02542 / L/D 27.06   -> 0.2% / 0.5% / 0.3%.

Esa sensibilidad de ventana (<0.5%) esta un orden por debajo de las
diferencias entre mallas medidas en el GCI (Cd 6.4%, L/D 7.6%), que es la
condicion para que el estudio mida malla y no ventana. Ojo: ese margen esta
comprobado en alpha=4; cada punto vuelca su serie completa para poder
verificar la meseta a posteriori angulo por angulo.

--- Salidas ------------------------------------------------------------------
Por punto: escalares en el JSON de su malla, serie temporal completa (.npz con
t/cl/cd/clcd/res), campo final (.npz con u, v, p, mascara solida y los ejes 1D
de la malla estirada) y dos figuras (mapa de |u| y lineas de corriente).
Al final: richardson.json con p, extrapolado y GCI por angulo para Cl, Cd y
L/D, mas las figuras de p(alpha) y de la polar con banda de malla.

Los 4 puntos de dx=0.002 ya simulados en results/polar_2grados_dom24x16
(alpha 0, 2, 4, 6, los que llegaron a 26000/26000) se reutilizan: misma
config, mismo dominio, mismo presupuesto.

Reanudable: cada punto se cachea y una segunda invocacion salta los hechos.
Pausable con el centinela STOP_SIMULATION.trigger en la raiz.

Uso:
    .venv/bin/python scripts/agent_tests/richardson_polar_domC.py
    .venv/bin/python scripts/agent_tests/richardson_polar_domC.py --solo-analisis
    .venv/bin/python scripts/agent_tests/richardson_polar_domC.py --plan
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
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
from matplotlib.colors import Normalize

import verificacion_numerica as vn
import criterio_ci95 as cc
import polar_2grados_dom24x16 as p2

OUT = os.path.join(ROOT, "results", "richardson_polar_domC")
CAMPOS = os.path.join(OUT, "campos")
SERIES = os.path.join(OUT, "series")
FIGS = os.path.join(OUT, "figuras")
for d in (OUT, CAMPOS, SERIES, FIGS):
    os.makedirs(d, exist_ok=True)

STOP = os.path.join(ROOT, "STOP_SIMULATION.trigger")

PERFIL = ("results/convergence_study/islands_tfg2/NACA_0012_sharp/epoch10/"
          "NACA_0012_sharp_winner_Re100000_a4.0_LD27.75.dat")
DOMINIO = {"Lx": 24.0, "Ly": 16.0, "cx": 6.0}
RE = 1e5
ANGULOS = (0.0, 2.0, 4.0, 6.0, 8.0)

# De gruesa a fina: la malla barata valida la infraestructura y da una polar
# completa en menos de una hora antes de comprometer los dos dias de la fina.
DXS = (0.004, 0.002, 0.001)
ITERS = {0.004: 13000, 0.002: 26000, 0.001: 52000}   # t ~ 20 en las tres

# Ambas paradas fuera: ver el bloque "Criterio de parada" del docstring.
SIN_PARADAS = {"stop_on_clcd_convergence": False, "stop_on_convergence": False}

# Puntos ya pagados con esta misma config que se copian en vez de repetirse.
SEMILLA = {
    0.002: ("results/polar_2grados_dom24x16", "ganador_tfg2", (0.0, 2.0, 4.0, 6.0)),
}

COLS = ["dx", "alpha", "cl", "cd", "ld", "cl_ci95", "cd_ci95", "ld_ci95",
        "cl_std", "cd_std", "n_samples", "converged_clcd", "t_conv_clcd",
        "t_parada_offline", "ld_parada_offline", "coste_parada_offline",
        "iters_efectivas", "iters_programadas", "cl_cp_discrepancy",
        "cl_cp_discrepancy_flag", "Re", "Lx", "Ly", "cx", "wall_s"]

DX_COLOR = {0.004: "#8e44ad", 0.002: "#1f4e79", 0.001: "#c0392b"}
DX_MARKER = {0.004: "^", 0.002: "s", 0.001: "o"}


def k_dx(dx):
    return f"{dx:.4f}"


def k_a(a):
    return f"{a:.1f}"


def json_path(dx):
    return os.path.join(OUT, f"polar_dx{k_dx(dx)}.json")


def campo_path(dx, a):
    return os.path.join(CAMPOS, f"dx{k_dx(dx)}_a{a:04.1f}.npz")


def serie_path(dx, a):
    return os.path.join(SERIES, f"dx{k_dx(dx)}_a{a:04.1f}_series.npz")


def cargar(dx):
    p = json_path(dx)
    return json.load(open(p)) if os.path.exists(p) else {}


def guardar(dx, d):
    with open(json_path(dx), "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------
# Replay offline del criterio de parada sobre la serie volcada
# ----------------------------------------------------------------------
def parada_offline(dx, a):
    """Donde habria parado el criterio calibrado, sin haber dejado que parase."""
    p = serie_path(dx, a)
    if not os.path.exists(p):
        return {}
    z = np.load(p)
    s = cc._serie(z["t"], z["cl"], z["cd"], f"dx{k_dx(dx)}_a{a:.0f}", "richardson")
    if s is None:
        return {}
    combo = json.load(open(f"{vn.OUT}/criterio_parada.json"))["criterio"]
    r = cc.evaluar(s, **combo)
    if r is None:
        return {"t_parada_offline": None, "ld_parada_offline": None,
                "coste_parada_offline": None}
    return {"t_parada_offline": round(r[0], 3),
            "ld_parada_offline": round(r[1], 4),
            "coste_parada_offline": round(r[2], 3)}


# ----------------------------------------------------------------------
# Figuras de campo por punto
# ----------------------------------------------------------------------
def figuras_campo(dx, a, meta):
    f = campo_path(dx, a)
    if not os.path.exists(f):
        return
    npz = np.load(f)
    xi, yi, u, v, mask = p2._remuestrear(npz)
    speed = np.where(mask > 0.5, np.nan, np.hypot(u, v))
    cab = (f"Perfil optimizado (AG) — α={a:.0f}°, Re=1e5, dx={dx:g}, dominio 24×16\n"
           f"campo final: {meta.get('iters_efectivas')} iters (t ≈ 20)")

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    im = ax.pcolormesh(xi, yi, speed, cmap="turbo", shading="auto",
                       vmin=0.0, vmax=p2.VMAX)
    p2._cuerpo(ax, xi, yi, mask)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label(
        r"$|\mathbf{u}|/U_\infty$ [-]")
    p2._ejes(ax, "Campo de velocidad — " + cab)
    fig.tight_layout()
    fig.savefig(os.path.join(CAMPOS, f"dx{k_dx(dx)}_a{a:04.1f}_velocidad.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    ax.set_facecolor("#f7f7f7")
    st = ax.streamplot(xi, yi, np.where(mask > 0.5, 0.0, u),
                       np.where(mask > 0.5, 0.0, v),
                       color=speed, cmap="turbo", norm=Normalize(0.0, p2.VMAX),
                       density=2.4, linewidth=0.9, arrowsize=0.8)
    p2._cuerpo(ax, xi, yi, mask)
    fig.colorbar(st.lines, ax=ax, fraction=0.035, pad=0.02).set_label(
        r"$|\mathbf{u}|/U_\infty$ [-]")
    p2._ejes(ax, "Líneas de corriente — " + cab)
    fig.tight_layout()
    fig.savefig(os.path.join(CAMPOS, f"dx{k_dx(dx)}_a{a:04.1f}_streamlines.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Richardson por angulo
# ----------------------------------------------------------------------
MAGNITUDES = [("cl", r"$C_l$"), ("cd", r"$C_d$"), ("ld", r"$C_l/C_d$")]


def analisis(datos):
    """gci_triplete por angulo y magnitud. f1 = malla fina (dx=0.001)."""
    out = {}
    for a in ANGULOS:
        ka = k_a(a)
        if not all(ka in datos[dx] for dx in DXS):
            continue
        out[ka] = {}
        for col, _ in MAGNITUDES:
            f1, f2, f3 = (datos[dx][ka][col] for dx in (0.001, 0.002, 0.004))
            r = vn.gci_triplete(f1, f2, f3, 0.001, 0.002, 0.004)
            # Sin monotonia el p y el extrapolado de Richardson no significan
            # nada: lo unico defendible es la banda entre las tres mallas.
            vals = [f1, f2, f3]
            r["banda_pct"] = round(100 * (max(vals) - min(vals)) / abs(np.mean(vals)), 4)
            out[ka][col] = r
    return out


def resumen_p(res):
    """p por angulo, solo donde la convergencia es monotona."""
    tabla = {}
    for col, _ in MAGNITUDES:
        ps = [(float(ka), res[ka][col]["p_observado"])
              for ka in sorted(res, key=float)
              if res[ka][col]["convergencia_monotona"]]
        n_mon = len(ps)
        vals = [p for _, p in ps]
        tabla[col] = {
            "n_monotonos": n_mon, "n_total": len(res),
            "p_por_angulo": {f"{a:.1f}": round(p, 4) for a, p in ps},
            "p_medio": round(float(np.mean(vals)), 4) if vals else None,
            "p_std": round(float(np.std(vals)), 4) if vals else None,
            "banda_media_pct": round(float(np.mean(
                [res[ka][col]["banda_pct"] for ka in res])), 4),
        }
    return tabla


def escribir_csv(datos):
    with open(os.path.join(OUT, "polares.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for dx in DXS:
            for a in ANGULOS:
                if k_a(a) in datos[dx]:
                    w.writerow(datos[dx][k_a(a)])


def figuras_analisis(datos, res, tabla):
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 14,
                         "axes.titlesize": 15, "legend.fontsize": 11})

    # Polar por malla
    for col, ylabel in MAGNITUDES:
        fig, ax = plt.subplots(figsize=(9.5, 6.2))
        for dx in DXS:
            aa = [a for a in ANGULOS if k_a(a) in datos[dx]]
            if not aa:
                continue
            y = [datos[dx][k_a(a)][col] for a in aa]
            e = [datos[dx][k_a(a)].get(f"{col}_ci95") or 0.0 for a in aa]
            ax.errorbar(aa, y, yerr=e, color=DX_COLOR[dx], marker=DX_MARKER[dx],
                        ms=7, lw=1.8, capsize=3, label=f"dx = {dx:g}")
        ax.set_xlabel("α [°]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} — convergencia de malla, dominio 24×16, Re=1e5")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, f"polar_{col}.png"), dpi=170)
        plt.close(fig)

    if not res:
        return

    # p(alpha): el resultado central del estudio
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for (col, ylabel), c in zip(MAGNITUDES, ("#1f4e79", "#c0392b", "#27ae60")):
        pts = [(float(ka), res[ka][col]["p_observado"], res[ka][col]["convergencia_monotona"])
               for ka in sorted(res, key=float)]
        mon = [(a, p) for a, p, m in pts if m]
        osc = [(a, p) for a, p, m in pts if not m]
        if mon:
            ax.plot(*zip(*mon), "o-", color=c, ms=8, label=f"{ylabel} (monótono)")
        if osc:
            ax.plot(*zip(*osc), "x", color=c, ms=10, mew=2.5,
                    label=f"{ylabel} (oscilatorio — p sin sentido)")
    ax.axhline(2.0, color="k", ls="--", lw=1.2, label="orden formal del esquema")
    ax.set_xlabel("α [°]")
    ax.set_ylabel("orden observado p")
    ax.set_title("Orden de convergencia observado a lo largo de la polar")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "orden_p_vs_alpha.png"), dpi=170)
    plt.close(fig)

    # Banda entre mallas: lo citable cuando p no lo es
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    aa = sorted(float(k) for k in res)
    for (col, ylabel), c in zip(MAGNITUDES, ("#1f4e79", "#c0392b", "#27ae60")):
        ax.plot(aa, [res[k_a(a)][col]["banda_pct"] for a in aa], "o-",
                color=c, ms=7, label=ylabel)
    ax.set_xlabel("α [°]")
    ax.set_ylabel("banda entre las tres mallas [%]")
    ax.set_title("Dispersión entre mallas — incertidumbre de discretización")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "banda_mallas.png"), dpi=170)
    plt.close(fig)


def rehacer_salidas(datos):
    res = analisis(datos)
    tabla = resumen_p(res) if res else {}
    with open(os.path.join(OUT, "richardson.json"), "w") as f:
        json.dump({"config": {"perfil": PERFIL, "Re": RE, "dominio": DOMINIO,
                              "dxs": list(DXS), "angulos": list(ANGULOS),
                              "iters": {k_dx(d): ITERS[d] for d in DXS},
                              "paradas": "desactivadas (presupuesto fijo t~20)",
                              "criterio_replay": json.load(
                                  open(f"{vn.OUT}/criterio_parada.json"))["criterio"]},
                   "por_angulo": res, "resumen_p": tabla}, f, indent=2,
                  ensure_ascii=False)
    escribir_csv(datos)
    figuras_analisis(datos, res, tabla)
    return res, tabla


# ----------------------------------------------------------------------
def sembrar(datos):
    """Copia los puntos ya pagados con config identica."""
    for dx, (src, tag, angulos) in SEMILLA.items():
        origen = os.path.join(ROOT, src, f"polar_{tag}.json")
        if not os.path.exists(origen):
            continue
        prev = json.load(open(origen))
        n = 0
        for a in angulos:
            ka = k_a(a)
            r = prev.get(ka)
            if ka in datos[dx] or r is None:
                continue
            if r.get("iters_efectivas") != ITERS[dx]:
                vn._log(f"semilla dx={dx} α={a}: {r.get('iters_efectivas')} iters "
                        f"!= {ITERS[dx]}, se re-simula")
                continue
            for orig, dest in ((os.path.join(ROOT, src, "series", f"{tag}_a{a:04.1f}_series.npz"),
                                serie_path(dx, a)),
                               (os.path.join(ROOT, src, "campos", f"{tag}_a{a:04.1f}.npz"),
                                campo_path(dx, a))):
                if os.path.exists(orig) and not os.path.exists(dest):
                    shutil.copy2(orig, dest)
            r = dict(r)
            r["origen"] = src
            r.update(parada_offline(dx, a))
            datos[dx][ka] = r
            figuras_campo(dx, a, r)
            n += 1
        if n:
            guardar(dx, datos[dx])
            vn._log(f"dx={dx}: {n} puntos reutilizados de {src}")


def plan(datos):
    # s/iter medidos en este dominio sobre 300 iteraciones (0.125 / 0.190 /
    # 0.405), por 1.25: en regimen los ciclos MG salen mas caros que en el
    # arranque. Calibrado contra la corrida real de dx=0.002 alpha=4, que dio
    # 6172 s para 26000 iters -> 0.237 s/it.
    s_it = {0.004: 0.156, 0.002: 0.237, 0.001: 0.506}
    print("\n" + "=" * 74)
    print(f"{'dx':>8s} {'iters':>7s} {'pendientes':>11s} {'h/punto':>8s} {'h total':>9s}")
    print("-" * 74)
    tot = 0.0
    for dx in DXS:
        pend = [a for a in ANGULOS if k_a(a) not in datos[dx]]
        h = ITERS[dx] * s_it[dx] / 3600.0
        tot += h * len(pend)
        print(f"{dx:>8.4f} {ITERS[dx]:>7d} {len(pend):>11d} {h:>8.2f} {h*len(pend):>9.2f}")
    print("-" * 74)
    print(f"{'TOTAL':>8s} {'':>7s} {'':>11s} {'':>8s} {tot:>9.2f}")
    print("=" * 74 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-analisis", action="store_true",
                    help="rehace Richardson y figuras sobre lo ya simulado")
    ap.add_argument("--plan", action="store_true",
                    help="lista lo pendiente y su coste estimado, sin simular")
    args = ap.parse_args()

    datos = {dx: cargar(dx) for dx in DXS}
    sembrar(datos)

    if args.plan:
        plan(datos)
        return

    if args.solo_analisis:
        res, tabla = rehacer_salidas(datos)
        vn._log(f"análisis regenerado en {OUT}")
        informe(datos, res, tabla)
        return

    crit = json.load(open(f"{vn.OUT}/criterio_parada.json"))
    vn._log(f"criterio de parada (replay offline, NO actúa): {crit['criterio']}")
    vn._log(f"procedencia: {crit.get('procedencia', 'criterio_parada.py')}")
    vn._log(f"presupuesto fijo t≈20: {ITERS}  (early-stop y guardián de residual OFF)")
    vn._log(f"dominio {DOMINIO}, Re={RE:.0e}, α={list(ANGULOS)}  ->  {OUT}")
    plan(datos)

    for dx in DXS:
        for a in ANGULOS:
            ka = k_a(a)
            if ka in datos[dx]:
                continue
            if os.path.exists(STOP):
                vn._log(f"centinela {STOP} presente -> parada")
                rehacer_salidas(datos)
                return
            vn._log(f"--- dx={dx:g}  α={a:.1f}°  ({ITERS[dx]} iters) ---")
            r = vn.simular(PERFIL, RE, dx, alpha=a, iters=ITERS[dx],
                           extra={**DOMINIO, **SIN_PARADAS},
                           dump=serie_path(dx, a), dump_field=campo_path(dx, a))
            if r is None:
                vn._log(f"dx={dx} α={a}: simulación fallida")
                continue
            r.update(DOMINIO)
            r.update(parada_offline(dx, a))
            datos[dx][ka] = r
            guardar(dx, datos[dx])
            vn._log(f"dx={dx:g} α={a}: Cl={r['cl']:.4f} Cd={r['cd']:.5f} "
                    f"L/D={r['ld']:.2f} n={r['n_samples']} "
                    f"t_parada_offline={r.get('t_parada_offline')} ({r['wall_s']}s)")
            figuras_campo(dx, a, r)
            rehacer_salidas(datos)

    res, tabla = rehacer_salidas(datos)
    informe(datos, res, tabla)


def informe(datos, res, tabla):
    print("\n" + "=" * 92)
    print(f"{'dx':>8s} {'alpha':>6s} {'Cl':>9s} {'Cd':>10s} {'L/D':>8s} "
          f"{'n':>5s} {'t_off':>7s} {'coste':>6s} {'wall_s':>8s}")
    print("-" * 92)
    for dx in DXS:
        for a in ANGULOS:
            r = datos[dx].get(k_a(a))
            if r is None:
                continue
            to = r.get("t_parada_offline")
            co = r.get("coste_parada_offline")
            print(f"{dx:>8.4f} {a:>6.1f} {r['cl']:>9.4f} {r['cd']:>10.5f} "
                  f"{r['ld']:>8.2f} {r['n_samples']:>5d} "
                  f"{(f'{to:.1f}' if to else '-'):>7s} "
                  f"{(f'{co:.2f}' if co else '-'):>6s} {r['wall_s']:>8.0f}")
    print("=" * 92)

    if not res:
        print("\nsin tripletes completos todavía: no hay Richardson que hacer\n")
        return

    for col, ylabel in MAGNITUDES:
        t = tabla[col]
        print(f"\n{col.upper()}  —  monótonos {t['n_monotonos']}/{t['n_total']}, "
              f"banda media entre mallas {t['banda_media_pct']:.2f} %")
        print(f"{'alpha':>6s} {'p':>8s} {'monot':>7s} {'f_ext':>10s} "
              f"{'GCI_fina%':>10s} {'banda%':>8s}")
        for ka in sorted(res, key=float):
            g = res[ka][col]
            print(f"{float(ka):>6.1f} {g['p_observado']:>8.3f} "
                  f"{str(g['convergencia_monotona']):>7s} "
                  f"{g['f_extrapolado_richardson']:>10.5f} "
                  f"{g['GCI_fina_pct']:>10.3f} {g['banda_pct']:>8.3f}")
        if t["p_medio"] is not None:
            print(f"  p medio (solo monótonos) = {t['p_medio']:.3f} ± {t['p_std']:.3f}")


if __name__ == "__main__":
    main()

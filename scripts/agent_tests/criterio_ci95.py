"""
Recalibracion del criterio de parada: la puerta de ruido pasa a ser el CI95 de
la media.

Por que. El criterio en produccion pedia drift<0.005 y sigma/|media|<0.05 sobre
una ventana movil de L/D. La segunda condicion mide la amplitud de la oscilacion
de la estela, no la incertidumbre del promedio, asi que en cuanto hay
desprendimiento no abre nunca: sobre el ganador a dx=0.001, sigma/|media| vale
8.9 % con ventana 1 y sube a 11.7 % con ventana 16, mientras el drift ya pasaba
holgado. Resultado medido: 192000 iteraciones, t=71 fisico, converged=false.

La magnitud correcta para una fuerza promediada es cuanto se mueve la *media*:
CI95 = 1.96*sigma/sqrt(N_ef), con N_ef corregido por autocorrelacion (ver
_n_efectivo en Simulador2D.py). Si la ventana contiene muchos ciclos la media
esta determinada aunque la senal oscile; si contiene medio ciclo, no.

Este script no reimplementa nada: importa _detect_series_convergence del solver
y lo evalua sobre las series guardadas, de modo que lo validado es el codigo que
corre.

Dos grupos, y los dos entran en la calibracion. Ajustar solo sobre las
estacionarias es precisamente lo que produjo el criterio que no abre nunca con
desprendimiento, asi que la eleccion se hace por el peor error de los pliegues
retenidos *de ambos grupos*:

  estacionario   20 series a dx=0.004 del historial del GA (results/verificacion
                 _numerica/series_dx004), las mismas del ajuste anterior.
  desprendimiento  series de la polar del dominio C a dx=0.002, la corrida
                 asintotica de alpha=4 hasta t=41.5 y la ventana larga a dx=0.001.

Solo se admiten series con t_final >= T_MIN_REF: en una corrida cortada en t=5 la
cola no es un valor asintotico y no sirve de referencia contra la que medir error.
Eso deja fuera los puntos que el guardian de estado estacionario trunco (ganador
alpha=8 y 10, NACA alpha=10).

El conjunto de test se aparta antes de barrer y no interviene en ninguna eleccion.

Uso:  .venv/bin/python scripts/agent_tests/criterio_ci95.py [--aplicar]

--aplicar reescribe el bloque "criterio" de criterio_parada.json con el elegido.
Sin GPU.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from Simulador2D import _detect_series_convergence, _n_efectivo  # noqa: E402


def _ventana_usada(t, s, tol_ci95, window):
    """La ventana que acabo usando el detector, para promediar sobre la misma."""
    span = float(t[-1] - t[0])
    v = float(window)
    while v < span:
        w = t >= (t[-1] - v)
        if w.sum() >= 8:
            sw = s[w]
            ci = 1.96 * float(np.std(sw, ddof=1)) / np.sqrt(_n_efectivo(sw)) / abs(np.mean(sw))
            if ci < tol_ci95:
                break
        v = min(2.0 * v, span)
    return v

OUT = "results/verificacion_numerica"
JSON_CRIT = f"{OUT}/criterio_parada.json"
SALIDA = f"{OUT}/criterio_ci95.json"

COLA = 0.15      # fraccion final que define el valor de referencia
CHECK = 2        # se chequea cada CHECK muestras, como en el solver
T_MIN_REF = 15.0  # tiempos convectivos minimos para que la cola valga de referencia
SEED = 0

REJILLA = {
    "tol_drift": [0.01, 0.005, 0.002],
    "tol_ci95": [0.03, 0.02, 0.01],
    "window": [1.0, 2.0, 4.0],     # minimo; la ventana se ensancha sola
    "n_sostenido": [1, 3],
    "min_t": [5.0],
}
# Como se elige entre combinaciones, sin umbrales inventados:
#
#   1. se descartan las que no paran. Un criterio que casi nunca dispara no es un
#      criterio, es un tope de iteraciones con pasos extra: seguro, y sin ahorro.
#      Se exige que pare en COBERTURA_MIN de las series del pliegue retenido.
#   2. de las que quedan, la de menor peor-error en los pliegues retenidos.
#   3. a igualdad de error, la mas barata.
#
# El orden importa y esta puesto a proposito: la version anterior del criterio se
# eligio por coste dentro de un tope de error, y asi entro n_sostenido=1, que cumplia
# en los pliegues (1.83 %) y daba 3.9 % en el test apartado.
COBERTURA_MIN = 0.8


def _serie(t, cl, cd, ident, grupo):
    ok = np.isfinite(t) & np.isfinite(cl) & np.isfinite(cd) & (t > 0)
    if ok.sum() < 40:
        return None
    t, cl, cd = t[ok].astype(float), cl[ok].astype(float), cd[ok].astype(float)
    ld = np.where(np.abs(cd) > 1e-9, cl / np.where(np.abs(cd) > 1e-9, cd, 1), np.nan)
    if not np.all(np.isfinite(ld)):
        return None
    return {"id": ident, "grupo": grupo, "t": t, "ld": ld}


def cargar():
    series = []
    for p in sorted(glob.glob(f"{OUT}/series_dx004/serie_*.npz")):
        z = np.load(p)
        s = _serie(z["t"], z["cl"], z["cd"],
                   int(os.path.basename(p).split("_")[1].split(".")[0]), "estacionario")
        if s:
            series.append(s)
    fuentes = [(p, os.path.basename(p).replace("_series.npz", ""))
               for p in sorted(glob.glob("results/polar_2grados_dom24x16/series/*.npz"))]
    fuentes += [("results/asintotico_alpha4_domC/series.npz", "asintotico_a4_t41"),
                (f"{OUT}/ventana_larga_dx001_series.npz", "ventana_larga_dx001")]
    for p, nombre in fuentes:
        if not os.path.exists(p):
            continue
        z = np.load(p)
        s = _serie(z["t"], z["cl"], z["cd"], nombre, "desprendimiento")
        if s:
            series.append(s)
    return series


def asintotico(s):
    n = max(3, int(len(s["ld"]) * COLA))
    return float(np.mean(s["ld"][-n:]))


def evaluar(s, tol_drift, tol_ci95, window, n_sostenido, min_t):
    """Donde habria parado, con la funcion del solver. (t, ld_reportado, coste) o None."""
    t, ld = s["t"], s["ld"]
    seguidos = 0
    for i in range(10, len(t), CHECK):
        if t[i] < min_t:
            continue
        _, _, conv = _detect_series_convergence(t[:i + 1], ld[:i + 1],
                                                tol_drift, tol_ci95, window)
        seguidos = seguidos + 1 if conv else 0
        if seguidos >= n_sostenido:
            w = (t >= t[i] - _ventana_usada(t[:i + 1], ld[:i + 1], tol_ci95, window))
            w &= np.arange(len(t)) <= i
            return float(t[i]), float(np.mean(ld[w])), float(t[i] / t[-1])
    return None


def barrer(series, combo):
    errs, costes, paradas = [], [], 0
    for s in series:
        ref = asintotico(s)
        r = evaluar(s, **combo)
        if r is None:                 # agotar presupuesto es seguro y caro, no un fallo
            errs.append(0.0)
            costes.append(1.0)
            continue
        paradas += 1
        errs.append(100 * abs(r[1] - ref) / abs(ref))
        costes.append(r[2])
    return {"err_max": max(errs), "err_medio": float(np.mean(errs)),
            "coste_medio": float(np.mean(costes)), "n_para": paradas,
            "n_series": len(series)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    args = ap.parse_args()

    series = [s for s in cargar() if s["t"][-1] >= T_MIN_REF]
    est = [s for s in series if s["grupo"] == "estacionario"]
    des = [s for s in series if s["grupo"] == "desprendimiento"]
    print(f"{len(est)} series estacionarias, {len(des)} con desprendimiento "
          f"(t_final >= {T_MIN_REF})")

    # Test apartado: los mismos 5 estacionarios del ajuste anterior, para que las
    # cifras sean comparables, mas 2 con desprendimiento sacados por semilla.
    prev = json.load(open(JSON_CRIT))
    test_ids = set(map(str, prev["ids_test"]))
    ids_des = sorted(str(s["id"]) for s in des)
    rng = np.random.default_rng(SEED)
    test_ids |= {ids_des[i] for i in rng.permutation(len(ids_des))[:2]}
    test = [s for s in series if str(s["id"]) in test_ids]
    cv = [s for s in series if str(s["id"]) not in test_ids]
    print(f"CV   ({len(cv)}): {sorted(str(s['id']) for s in cv)}")
    print(f"test ({len(test)}): {sorted(test_ids & {str(s['id']) for s in series})}")

    keys = list(REJILLA)
    combos = [dict(zip(keys, v)) for v in itertools.product(*(REJILLA[k] for k in keys))]
    K = 3
    # Pliegues estratificados: cada uno lleva estacionarias y con desprendimiento.
    pliegues = []
    for k in range(K):
        pl = set()
        for grupo in ("estacionario", "desprendimiento"):
            ids = sorted(str(s["id"]) for s in cv if s["grupo"] == grupo)
            pl |= {ids[i] for i in range(k, len(ids), K)}
        pliegues.append(pl)
    print(f"barriendo {len(combos)} combinaciones sobre {K} pliegues estratificados ...")

    punt = []
    for c in combos:
        peor, coste, peor_des, cob = 0.0, [], 0.0, []
        for pl in pliegues:
            held = [s for s in cv if str(s["id"]) in pl]
            r = barrer(held, c)
            peor = max(peor, r["err_max"])
            hd = [s for s in held if s["grupo"] == "desprendimiento"]
            if hd:
                peor_des = max(peor_des, barrer(hd, c)["err_max"])
            coste.append(r["coste_medio"])
            cob.append(r["n_para"] / r["n_series"])
        punt.append({"combo": c, "peor_err_cv": peor, "peor_err_desprendimiento": peor_des,
                     "coste_cv": float(np.mean(coste)), "cobertura_cv": float(np.mean(cob))})

    admisibles = [p for p in punt if p["cobertura_cv"] >= COBERTURA_MIN]
    if not admisibles:
        print(f"[!] ninguna combinacion para en el {COBERTURA_MIN:.0%} de las series")
        admisibles = punt
    elegido = min(admisibles, key=lambda p: (round(p["peor_err_cv"], 2), p["coste_cv"]))
    print(f"{len(admisibles)} de {len(punt)} combinaciones paran en >= {COBERTURA_MIN:.0%}")
    c = elegido["combo"]
    print(f"\nelegido: {c}\n  peor error en pliegues retenidos {elegido['peor_err_cv']:.2f} %"
          f"  (solo desprendimiento {elegido['peor_err_desprendimiento']:.2f} %)"
          f"   coste {elegido['coste_cv']:.2f}")

    res = {"criterio": {"tol_drift": c["tol_drift"], "tol_ci95": c["tol_ci95"],
                        "window": c["window"], "n_sostenido": c["n_sostenido"],
                        "min_t": c["min_t"]},
           "cv": elegido, "test": barrer(test, c),
           "estacionario": barrer(est, c), "desprendimiento": barrer(des, c),
           "t_min_ref": T_MIN_REF, "cobertura_min": COBERTURA_MIN,
           "frontera": sorted([p for p in punt if p["cobertura_cv"] >= COBERTURA_MIN],
                              key=lambda p: (p["peor_err_cv"], p["coste_cv"]))[:15],
           "detalle": []}
    print(f"\ntest (no interviene en la eleccion): {res['test']}")
    print(f"todas las estacionarias:   {res['estacionario']}")
    print(f"todas las de desprendimiento: {res['desprendimiento']}")

    print(f"\n{'serie':>26} {'grupo':>15} {'ld_asin':>9} {'t_par':>7} {'ld_rep':>8} "
          f"{'err%':>7} {'coste':>6} {'t_fin':>7}")
    for s in sorted(series, key=lambda z: (z["grupo"], str(z["id"]))):
        ref = asintotico(s)
        r = evaluar(s, **c)
        fila = {"id": str(s["id"]), "grupo": s["grupo"], "ld_asintotico": round(ref, 4),
                "t_parada": None if r is None else round(r[0], 3),
                "ld_reportado": None if r is None else round(r[1], 4),
                "error_pct": None if r is None else round(100 * abs(r[1] - ref) / abs(ref), 2),
                "coste": None if r is None else round(r[2], 3),
                "t_final": round(float(s["t"][-1]), 2)}
        res["detalle"].append(fila)
        err = "-" if r is None else f"{fila['error_pct']:7.2f}"
        print(f"{str(s['id']):>26} {s['grupo']:>15} {ref:9.3f} "
              f"{'-' if r is None else f'{r[0]:7.3f}'} "
              f"{'-' if r is None else f'{r[1]:8.3f}'} {err:>7} "
              f"{'-' if r is None else f'{r[2]:6.3f}'} {s['t'][-1]:7.2f}")

    with open(SALIDA, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n-> {SALIDA}")

    if args.aplicar:
        prev["criterio"] = res["criterio"]
        prev["procedencia"] = ("recalibrado por criterio_ci95.py: la puerta de ruido "
                               "pasa de sigma/|media| a CI95 de la media")
        with open(JSON_CRIT, "w") as f:
            json.dump(prev, f, indent=2, ensure_ascii=False)
        print(f"-> {JSON_CRIT} actualizado")


if __name__ == "__main__":
    sys.exit(main())

"""
Compara las dos polares del dominio C (t~10 de polar_fina_1grado_dom24x16 contra
t~20 de polar_2grados_dom24x16) y extrapola cada punto a t -> infinito.

Sobre el metodo de extrapolacion. Richardson (el del GCI en malla) supone error
en ley de potencias, y en el tiempo no lo es: la cola de L/D es un transitorio
que decae y luego mesetea. Se ve en el unico punto con verdad medida, alpha=4
del ganador llevado a t=41.5 (results/asintotico_alpha4_domC/), donde el valor
real de la cola es 34.084:

    Richardson sobre T=5/10/20  ->  39.41   (+15.6 %)
    exponencial sobre T=5/10/20 ->  34.66   ( +1.7 %)

Asi que se ajusta f(T) = Finf - A*exp(-T/tau) sobre las colas a T, 2T y 4T, que
con tres puntos queda determinado: con x = exp(-T/tau), la razon de incrementos
vale x(1+x), de donde x, luego A y Finf.

El ajuste solo aplica si la razon de incrementos cae en (0, 1) — decae y no
oscila. Cuando no (desprendimiento sin meseta, o serie cortada por el guardian
de estado estacionario antes de la meseta) se marca n/a: ahi no hay valor
asintotico que extraer, no es que el ajuste falle por poco.

Cada cola usa <Cl>/<Cd> sobre la misma ventana [T/2, T] (misma correccion que
08af2c4).

Uso:  .venv/bin/python scripts/agent_tests/extrapolacion_polar.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)

T10 = "results/polar_fina_1grado_dom24x16"
T20 = "results/polar_2grados_dom24x16"
SERIES = f"{T20}/series"
VERDAD = "results/asintotico_alpha4_domC/series.npz"
LD_VERDAD = 34.084
OUT = f"{T20}/extrapolacion_t_inf.json"


def leer_csv(path):
    datos = {}
    for r in csv.DictReader(open(path)):
        datos[float(r["alpha"])] = {
            "cl": float(r["cl"]), "cd": float(r["cd"]), "ld": float(r["ld"]),
            "iters": int(float(r["iters_efectivas"])),
        }
    return datos


def cola(t, cl, cd, T, frac=0.5):
    m = (t >= frac * T) & (t <= T)
    if m.sum() < 10:
        return None
    return float(cl[m].mean()), float(cd[m].mean()), float(cl[m].mean() / cd[m].mean())


def exp_plateau(f1, f2, f3):
    """f(T) = Finf - A*exp(-T/tau) sobre T, 2T, 4T."""
    e1, e2 = f2 - f1, f3 - f2
    if e1 == 0:
        return None
    ratio = e2 / e1
    if not 0 < ratio < 1:
        return {"ok": False, "ratio": ratio}
    x = (-1 + math.sqrt(1 + 4 * ratio)) / 2
    A = e1 / (x - x * x)
    finf = f3 + A * x ** 4
    return {"ok": True, "ratio": ratio, "tau_sobre_T": -1 / math.log(x),
            "finf": finf, "resto_pct": (finf - f3) / finf * 100}


def serie(path):
    d = np.load(path)
    t, cl, cd = d["t"].astype(float), d["cl"].astype(float), d["cd"].astype(float)
    ok = np.isfinite(t) & np.isfinite(cl) & np.isfinite(cd)
    return t[ok], cl[ok], cd[ok]


def comparar():
    tabla = {}
    for tag in ("ganador_tfg2", "naca0012"):
        a, b = leer_csv(f"{T10}/polar_{tag}.csv"), leer_csv(f"{T20}/polar_{tag}.csv")
        filas = []
        for al in sorted(set(a) & set(b)):
            x, y = a[al], b[al]
            filas.append({"alpha": al,
                          **{f"{k}_10": x[k] for k in ("cl", "cd", "ld")},
                          **{f"{k}_20": y[k] for k in ("cl", "cd", "ld")},
                          **{f"d_{k}_pct": (y[k] - x[k]) / x[k] * 100 if x[k] else None
                             for k in ("cl", "cd", "ld")},
                          "iters_10": x["iters"], "iters_20": y["iters"]})
        tabla[tag] = filas
    return tabla


def validar():
    t, cl, cd = serie(VERDAD)
    out = []
    for Ts in ([5, 10, 20], [10, 20, 40], [4, 8, 16], [6, 12, 24]):
        v = [cola(t, cl, cd, T) for T in Ts]
        if any(z is None for z in v):
            continue
        ld = [z[2] for z in v]
        r = exp_plateau(*ld)
        out.append({"T": Ts, "ld": ld, "finf": r["finf"] if r and r["ok"] else None,
                    "err_pct": (r["finf"] - LD_VERDAD) / LD_VERDAD * 100 if r and r["ok"] else None,
                    "tau": r["tau_sobre_T"] * Ts[0] if r and r["ok"] else None})
    return {"ld_verdad": LD_VERDAD, "fuente": VERDAD, "triples": out}


def extrapolar():
    out = {}
    for tag in ("ganador_tfg2", "naca0012"):
        filas = []
        for a in (0, 2, 4, 6, 8, 10):
            p = f"{SERIES}/{tag}_a{a:02d}.0_series.npz"
            if not os.path.exists(p):
                continue
            t, cl, cd = serie(p)
            tmax = float(t[-1])
            Ts = [5, 10, 20] if tmax >= 19 else [tmax / 4, tmax / 2, tmax]
            v = [cola(t, cl, cd, T) for T in Ts]
            fila = {"alpha": float(a), "t_max": tmax, "T": Ts}
            if any(z is None for z in v):
                fila["nota"] = "serie demasiado corta"
                filas.append(fila)
                continue
            for i, nom in enumerate(("cl", "cd", "ld")):
                vals = [z[i] for z in v]
                r = exp_plateau(*vals)
                fila[nom] = {"colas": vals, "final": vals[-1],
                             **({"inf": r["finf"], "resto_pct": r["resto_pct"],
                                 "tau": r["tau_sobre_T"] * Ts[0]}
                                if r and r["ok"] else
                                {"inf": None, "razon": r["ratio"] if r else None})}
            filas.append(fila)
        out[tag] = filas
    return out


def main():
    res = {"comparacion_t10_t20": comparar(), "validacion": validar(),
           "extrapolacion": extrapolar()}
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print("=== t=10 -> t=20 (angulos comunes) ===")
    for tag, filas in res["comparacion_t10_t20"].items():
        print(f"\n--- {tag} ---")
        print(f"{'a':>4} {'Cl_10':>8} {'Cl_20':>8} {'d%':>7} {'Cd_10':>8} {'Cd_20':>8} "
              f"{'d%':>7} {'LD_10':>7} {'LD_20':>7} {'d%':>7} {'it10':>6} {'it20':>6}")
        for r in filas:
            print(f"{r['alpha']:4.0f} {r['cl_10']:8.4f} {r['cl_20']:8.4f} {r['d_cl_pct']:7.2f} "
                  f"{r['cd_10']:8.5f} {r['cd_20']:8.5f} {r['d_cd_pct']:7.2f} "
                  f"{r['ld_10']:7.2f} {r['ld_20']:7.2f} {r['d_ld_pct']:7.2f} "
                  f"{r['iters_10']:6d} {r['iters_20']:6d}")

    print(f"\n=== validacion del ajuste (verdad L/D={LD_VERDAD} a t=41.5) ===")
    for r in res["validacion"]["triples"]:
        s = f"  T={str(r['T']):>12}  colas={['%.2f' % q for q in r['ld']]}"
        if r["finf"] is not None:
            s += f"  ->  Finf={r['finf']:7.2f} ({r['err_pct']:+6.2f} %)  tau={r['tau']:5.2f}"
        print(s)

    print("\n=== extrapolacion t -> infinito ===")
    for tag, filas in res["extrapolacion"].items():
        print(f"\n--- {tag} ---")
        print(f"{'a':>4} {'tmax':>6} | {'Cl_fin':>7} {'Cl_inf':>7} | {'Cd_fin':>8} {'Cd_inf':>8} "
              f"| {'LD_fin':>7} {'LD_inf':>7} {'resto%':>7} {'tau':>5}")
        for r in filas:
            if "nota" in r:
                print(f"{r['alpha']:4.0f} {r['t_max']:6.2f} | {r['nota']}")
                continue
            g = lambda k, w: (f"{r[k]['inf']:{w}}" if r[k]["inf"] is not None else "n/a".rjust(int(w.split('.')[0])))
            ld = r["ld"]
            print(f"{r['alpha']:4.0f} {r['t_max']:6.2f} | {r['cl']['final']:7.4f} {g('cl','7.4f')} "
                  f"| {r['cd']['final']:8.5f} {g('cd','8.5f')} | {ld['final']:7.2f} {g('ld','7.2f')} "
                  f"{(('%6.2f' % ld['resto_pct']) if ld['inf'] is not None else 'n/a'):>7} "
                  f"{(('%5.2f' % ld['tau']) if ld['inf'] is not None else 'n/a'):>5}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    sys.exit(main())

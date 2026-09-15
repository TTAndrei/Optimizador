"""¿El adelgazamiento que premia el GA es física o falta de resolución?

El GA lleva el espesor de 12% (semilla) a 1.9% en 6 épocas y el L/D sube monótonamente
con ello. A dx=0.004 un perfil del 2% tiene 5 celdas de espesor máximo, así que la capa
límite no está resuelta y el Cd puede estar subestimado justo en las geometrías que el
optimizador prefiere.

Se evalúan a dx=0.002 (10 celdas en vez de 5) tres ganadores del mismo linaje con
espesores decrecientes. Si el adelgazamiento es real, la ventaja del fino se mantiene;
si es artefacto de malla, se estrecha o se invierte al refinar.

Reanudable: cachea cada resultado en el JSON de salida.
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

import RunGA

ISLAS = "results/convergence_study/islands_tfg2"
OUT = "results/verificacion_numerica/espesor_vs_malla.json"

ALPHA, CFL, U_FAC, T_TARGET = 4.0, 0.5, 1.6, 12.0
RE = 1e5

# Mismo linaje (isla GM15 para 0 y 3, s1014 para el 6: es el mejor actual) con
# espesores 6.1% / 3.5% / 2.1%, para que la comparación no mezcle familias.
CASOS = [
    ("e0_GM15", f"{ISLAS}/GM15/epoch0/estado_ga_final.json"),
    ("e3_GM15", f"{ISLAS}/GM15/epoch3/estado_ga_final.json"),
    ("e6_GM15", f"{ISLAS}/GM15/epoch6/estado_ga_final.json"),
    ("e6_s1014", f"{ISLAS}/s1014/epoch6/estado_ga_final.json"),
]
DXS = [0.002]   # dx=0.004 ya está medido: es el fitness con el que corrió el GA


def iters_for(dx, t=T_TARGET):
    return int(-(-(t * U_FAC / (CFL * dx)) // 1000) * 1000)


def criterio():
    p = "results/verificacion_numerica/criterio_parada.json"
    c = json.load(open(p))["criterio"]
    return {"clcd_tol_drift": c["tol_drift"], "clcd_tol_ci95": c["tol_ci95"],
            "clcd_window_conv_time": c["window"], "clcd_n_sostenido": c["n_sostenido"],
            "clcd_min_t_fisico_before_check": c["min_t"]}


def geometria(genes):
    le = int(np.argmin(genes[:, 0]))
    x, c, t = RunGA.descomponer_camber_espesor(genes, le, n_muestras=200)
    return {"t_max_pct": float(np.max(t) * 100),
            "x_t_max": float(x[int(np.argmax(t))]),
            "f_max_pct": float(np.max(np.abs(c)) * 100)}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    hecho = json.load(open(OUT)) if os.path.exists(OUT) else {}

    RunGA.CONFIG["nu"] = 1.0 / RE
    RunGA.CONFIG["sim_extra_params"] = {
        **RunGA.CONFIG.get("sim_extra_params", {}),
        "transition_model": "sa_bc", "freestream_Tu": 0.1, **criterio()}

    for nombre, path in CASOS:
        mg = json.load(open(path))["mejor_global"]
        genes = np.asarray(mg["genes"], float)
        geo = geometria(genes)
        hecho.setdefault(nombre, {"ld_dx004_ga": mg["fitness"], **geo})

        tmp = f"_test_esp_{nombre}.dat"
        RunGA.guardar_perfil(tmp, genes, f"{nombre}\n")
        try:
            for dx in DXS:
                clave = f"ld_dx{int(dx*1000):03d}"
                if clave in hecho[nombre]:
                    print(f"[skip] {nombre} @ dx={dx}", flush=True)
                    continue
                cfg = dict(RunGA.CONFIG)
                cfg.update(dx_min=dx, CFL=CFL,
                           simulacion_iteraciones=iters_for(dx))
                print(f"\n>>> {nombre} @ dx={dx}  (t_max={geo['t_max_pct']:.2f}%, "
                      f"{geo['t_max_pct']/100/dx:.0f} celdas de espesor)", flush=True)
                t0 = time.time()
                res = RunGA.simular_perfil(tmp, ALPHA, cfg)
                dt = time.time() - t0
                if res is None:
                    hecho[nombre][clave] = None
                else:
                    hecho[nombre][clave] = res["ld"]
                    hecho[nombre][f"cl_dx{int(dx*1000):03d}"] = res["cl"]
                    hecho[nombre][f"cd_dx{int(dx*1000):03d}"] = res["cd"]
                hecho[nombre][f"wall_s_dx{int(dx*1000):03d}"] = round(dt, 1)
                json.dump(hecho, open(OUT, "w"), indent=2, ensure_ascii=False)
                print(f"    L/D={hecho[nombre][clave]}  ({dt/60:.1f} min)", flush=True)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    print("\n" + "=" * 78)
    print(f"{'caso':12} {'t_max%':>7} {'celdas@004':>11} {'L/D 0.004':>10} "
          f"{'L/D 0.002':>10} {'cambio':>9}")
    for n, _ in CASOS:
        d = hecho[n]
        a, b = d.get("ld_dx004_ga"), d.get("ld_dx002")
        camb = f"{(b-a)/a*100:+.1f}%" if (a and b) else "—"
        print(f"{n:12} {d['t_max_pct']:7.2f} {d['t_max_pct']/100/0.004:11.1f} "
              f"{a:10.2f} {(b if b else float('nan')):10.2f} {camb:>9}")
    print("=" * 78)
    print(f"Escrito en {OUT}")


if __name__ == "__main__":
    main()

"""GCI de tres mallas sobre el ganador de la campaña.

El refinado del top-3 acabó de demostrar que dos niveles no bastan: s1014/epoch9
lideraba a dx=0.004 con L/D=28.66 y se cayó a 24.81 a dx=0.002, cambiando el
ganador. Con dos puntos no hay forma de saber si el 28.88 del ganador está
convergido o sigue moviéndose, así que se añade el tercer nivel.

Mallas dx = 0.004 / 0.002 / 0.001 (razón de refinado r=2 uniforme) y GCI de
Roache con factor de seguridad 1.25, sobre Cl, Cd y L/D por separado: el L/D es
un cociente y puede converger mejor o peor que sus dos componentes.

Reanudable: cachea cada malla en el JSON de salida.
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

GANADOR = "results/convergence_study/islands_tfg2/NACA_0012_sharp/epoch10/estado_ga_final.json"
OUT = "results/verificacion_numerica/gci_ganador.json"

ALPHA, CFL, U_FAC, T_TARGET = 4.0, 0.5, 1.6, 12.0
RE = 1e5
DXS = [0.004, 0.002, 0.001]
FS = 1.25


def iters_for(dx, t=T_TARGET):
    return int(-(-(t * U_FAC / (CFL * dx)) // 1000) * 1000)


def criterio():
    c = json.load(open("results/verificacion_numerica/criterio_parada.json"))["criterio"]
    return {"clcd_tol_drift": c["tol_drift"], "clcd_tol_ci95": c["tol_ci95"],
            "clcd_window_conv_time": c["window"], "clcd_n_sostenido": c["n_sostenido"],
            "clcd_min_t_fisico_before_check": c["min_t"]}


def gci(f3, f2, f1, r=2.0):
    """f1 la malla fina. Devuelve orden observado, extrapolación y GCI de cada par."""
    e32, e21 = f3 - f2, f2 - f1
    if e21 == 0 or e32 == 0:
        return {"nota": "diferencia nula entre mallas"}
    ratio = e32 / e21
    if ratio <= 0:
        # Convergencia oscilatoria: la solución no se acerca de forma monótona y
        # la extrapolación de Richardson no aplica. Se reporta la banda observada.
        return {"convergencia": "oscilatoria", "ratio": ratio,
                "banda": abs(e21) / abs(f1) * 100}
    p = np.log(ratio) / np.log(r)
    f_ext = f1 + e21 / (r ** p - 1)
    return {"convergencia": "monótona", "p": p, "f_extrapolado": f_ext,
            "gci_21_pct": FS * abs(e21 / f1) / (r ** p - 1) * 100,
            "gci_32_pct": FS * abs(e32 / f2) / (r ** p - 1) * 100,
            "error_extrap_pct": abs((f_ext - f1) / f_ext) * 100}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    d = json.load(open(OUT)) if os.path.exists(OUT) else {"mallas": {}}

    RunGA.CONFIG["nu"] = 1.0 / RE
    RunGA.CONFIG["sim_extra_params"] = {
        **RunGA.CONFIG.get("sim_extra_params", {}),
        "transition_model": "sa_bc", "freestream_Tu": 0.1, **criterio()}

    genes = np.asarray(json.load(open(GANADOR))["mejor_global"]["genes"], float)
    le = int(np.argmin(genes[:, 0]))
    x, c, t = RunGA.descomponer_camber_espesor(genes, le, n_muestras=200)
    d["geometria"] = {"t_max_pct": float(np.max(t) * 100),
                      "x_t_max": float(x[int(np.argmax(t))]),
                      "f_max_pct": float(np.max(np.abs(c)) * 100)}
    d["ganador"] = GANADOR

    tmp = "_gci_ganador.dat"
    RunGA.guardar_perfil(tmp, genes, "ganador_tfg2\n")
    try:
        for dx in DXS:
            k = f"{dx:.4f}"
            if k in d["mallas"]:
                print(f"[skip] dx={dx}", flush=True)
                continue
            cfg = dict(RunGA.CONFIG)
            cfg.update(dx_min=dx, CFL=CFL, simulacion_iteraciones=iters_for(dx))
            print(f"\n>>> dx={dx}  ({d['geometria']['t_max_pct']/100/dx:.1f} celdas "
                  f"de espesor, {iters_for(dx)} iteraciones)", flush=True)
            t0 = time.time()
            res = RunGA.simular_perfil(tmp, ALPHA, cfg)
            dt = time.time() - t0
            d["mallas"][k] = ({"fallo": True} if res is None else
                              {"cl": res["cl"], "cd": res["cd"], "ld": res["ld"]})
            d["mallas"][k]["wall_s"] = round(dt, 1)
            json.dump(d, open(OUT, "w"), indent=2, ensure_ascii=False)
            print(f"    {d['mallas'][k]}  ({dt/60:.1f} min)", flush=True)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    ks = [f"{dx:.4f}" for dx in DXS]
    if all(k in d["mallas"] and "ld" in d["mallas"][k] for k in ks):
        d["gci"] = {mag: gci(*[d["mallas"][k][mag] for k in ks])
                    for mag in ("cl", "cd", "ld")}
        json.dump(d, open(OUT, "w"), indent=2, ensure_ascii=False)
        print("\n" + "=" * 70)
        for mag in ("cl", "cd", "ld"):
            v = [d["mallas"][k][mag] for k in ks]
            print(f"{mag:>3}  0.004={v[0]:9.5f}  0.002={v[1]:9.5f}  0.001={v[2]:9.5f}")
            print(f"     {d['gci'][mag]}")
        print("=" * 70)
    print(f"Escrito en {OUT}")


if __name__ == "__main__":
    main()

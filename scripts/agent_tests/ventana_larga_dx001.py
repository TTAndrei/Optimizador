"""¿La oscilación del Cd en el GCI es espacial o de promediado temporal?

El GCI de tres mallas dejó el Cd oscilatorio (0.02528 -> 0.02388 -> 0.02551) y el
L/D con una banda del 7.6%. Pero la malla fina nunca llegó a estacionario: el
residuo de v se quedó en 4e-01 y la simulación agotó las 39000 iteraciones sin que
saltara el criterio de parada, así que su L/D es una media sobre una ventana que
puede no ser representativa. Las mallas gruesas sí convergen porque la difusión
numérica amortigua el desprendimiento: estaríamos comparando cosas distintas.

Aquí se repite dx=0.001 dejando que pare el criterio y no el contador. El tope de
iteraciones es una red de seguridad (t=60, cinco veces el de la campaña), no un
objetivo: lo que interesa es a qué tiempo físico converge y qué Cd deja.

Si el Cd se asienta cerca de 0.0239 la oscilación era de promediado y el GCI se
recalcula. Si se queda en 0.0255 el sospechoso es la condición de Kutta, que
empeora de -0.127 a -0.321 al refinar.

Pausar sin perder la corrida: crear STOP_SIMULATION.trigger en la raíz.
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
OUT = "results/verificacion_numerica/ventana_larga_dx001.json"
SERIES = "results/verificacion_numerica/ventana_larga_dx001_series.npz"

ALPHA, CFL, U_FAC, DX, RE = 4.0, 0.5, 1.6, 0.001, 1e5
T_GUARDA = 60.0   # tope de seguridad, no objetivo


def criterio():
    c = json.load(open("results/verificacion_numerica/criterio_parada.json"))["criterio"]
    return {"clcd_tol_drift": c["tol_drift"], "clcd_tol_noise": c["tol_noise"],
            "clcd_window_conv_time": c["window"], "clcd_n_sostenido": c["n_sostenido"],
            "clcd_min_t_fisico_before_check": c["min_t"]}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    iters = int(-(-(T_GUARDA * U_FAC / (CFL * DX)) // 1000) * 1000)

    RunGA.CONFIG["nu"] = 1.0 / RE
    RunGA.CONFIG["sim_extra_params"] = {
        **RunGA.CONFIG.get("sim_extra_params", {}),
        "transition_model": "sa_bc", "freestream_Tu": 0.1, **criterio()}

    genes = np.asarray(json.load(open(GANADOR))["mejor_global"]["genes"], float)
    tmp = "_ventana_larga.dat"
    RunGA.guardar_perfil(tmp, genes, "ganador_tfg2\n")

    cfg = dict(RunGA.CONFIG)
    cfg.update(dx_min=DX, CFL=CFL, simulacion_iteraciones=iters,
               dump_series_path=SERIES)

    crit = criterio()
    print(f">>> dx={DX}  tope de seguridad {iters} iteraciones (t={T_GUARDA})", flush=True)
    print(f"    criterio: drift<{crit['clcd_tol_drift']}  ruido<{crit['clcd_tol_noise']}  "
          f"ventana={crit['clcd_window_conv_time']}  t_min={crit['clcd_min_t_fisico_before_check']}", flush=True)
    print("    referencia dx=0.001 con t=12: Cd=0.025507  L/D=26.71  (sin converger)", flush=True)

    t0 = time.time()
    try:
        res = RunGA.simular_perfil(tmp, ALPHA, cfg)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    dt = time.time() - t0

    d = {"dx": DX, "alpha": ALPHA, "iters_tope": iters, "t_guarda": T_GUARDA,
         "criterio": crit, "wall_s": round(dt, 1), "resultado": res,
         "referencia_t12": {"cl": 0.681299, "cd": 0.025507, "ld": 26.71},
         "gci_previo": {"cd_dx004": 0.025281, "cd_dx002": 0.023883, "cd_dx001": 0.025507}}
    json.dump(d, open(OUT, "w"), indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    if res is None:
        print("La simulación falló.")
    else:
        print(f"  converged_clcd : {res['converged_clcd']}")
        print(f"  t_conv_clcd    : {res['t_conv_clcd']}   (t=12 no convergía)")
        print(f"  iters efectivas: {res['iters_efectivas']} de {iters}")
        print(f"  Cl = {res['cl']:.6f} +/- {res['cl_ci95']:.6f}")
        print(f"  Cd = {res['cd']:.6f} +/- {res['cd_ci95']:.6f}   (t=12 daba 0.025507)")
        print(f"  L/D= {res['ld']:.4f}                        (t=12 daba 26.71)")
        print(f"  muestras promediadas: {res['n_samples']}")
        dif = (res["cd"] - 0.023883) / 0.023883 * 100
        print(f"\n  Cd frente al de dx=0.002 (0.023883): {dif:+.2f}%")
        print("  -> si esto es ~0, la oscilacion del GCI era de promediado temporal")
    print(f"  wall: {dt/3600:.2f} h")
    print("=" * 70)
    print(f"Escrito en {OUT}  |  series en {SERIES}")


if __name__ == "__main__":
    main()

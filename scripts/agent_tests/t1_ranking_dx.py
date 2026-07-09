"""T1 — Consistencia de ranking dx=0.004 (estudio) vs dx=0.002 (refinado).

Re-evalúa los 5 ganadores del convergence study a dx=0.002 y compara con el
fitness dx=0.004 con que fueron seleccionados. Responde: ¿es fiable el ranking
producido a dx=0.004? ¿AG24 sigue #1?

Salida: results/convergence_study/t1_ranking_dx.json
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import RunGA

ALPHA = 4.0
CS = os.path.join(ROOT, "results", "convergence_study")

# (nombre, ruta_dat, ld_dx004_estudio)
GANADORES = [
    ("AG24", "armA/AG24/AG24_Re100000_a4.0_LD9.95.dat", 9.9469),
    ("GM15", "armA/GM15/GM15_Re100000_a4.0_LD9.31.dat", 9.3051),
    ("NACA_0012_sharp", "armA/NACA_0012_sharp/NACA_0012_sharp_Re100000_a4.0_LD6.95.dat", 6.9534),
    ("s1014", "armA/s1014/s1014_Re100000_a4.0_LD6.71.dat", 6.7127),
    ("mixed", "armB/mixed/NACA_0012_sharp_Re100000_a4.0_LD9.03.dat", 9.0306),
]


def rank(pares):
    orden = sorted(pares, key=lambda kv: kv[1], reverse=True)
    return {n: i for i, (n, _) in enumerate(orden)}


def main():
    cfg = dict(RunGA.CONFIG)
    cfg["dx_min"] = 0.002
    cfg["simulacion_iteraciones"] = 13000  # ~13000 en calibración dx=0.002
    cfg["CFL"] = 0.5

    filas = []
    for nombre, rel, ld004 in GANADORES:
        ruta = os.path.join(CS, rel)
        t0 = time.time()
        res = RunGA.simular_perfil(ruta, ALPHA, cfg)
        dt = time.time() - t0
        if res is None:
            filas.append({"nombre": nombre, "error": True, "ld_dx004": ld004})
            print(f"[T1] {nombre}: FALLO ({dt:.0f}s)")
            continue
        fila = {
            "nombre": nombre,
            "ld_dx004": ld004,
            "ld_dx002": res["ld"],
            "cl_dx002": res["cl"],
            "cd_dx002": res["cd"],
            "delta_ld": round(res["ld"] - ld004, 4),
            "wall_s": round(dt, 1),
        }
        filas.append(fila)
        print(f"[T1] {nombre}: dx004 L/D={ld004:.2f} -> dx002 L/D={res['ld']:.2f} "
              f"(Cl={res['cl']:.3f} Cd={res['cd']:.4f}) [{dt:.0f}s]")

    ok = [f for f in filas if not f.get("error")]
    r004 = rank([(f["nombre"], f["ld_dx004"]) for f in ok])
    r002 = rank([(f["nombre"], f["ld_dx002"]) for f in ok])
    inversiones = [n for n in r004 if r004[n] != r002[n]]
    ganador_004 = min(r004, key=r004.get) if r004 else None
    ganador_002 = min(r002, key=r002.get) if r002 else None

    out = {
        "alpha": ALPHA,
        "filas": filas,
        "rank_dx004": r004,
        "rank_dx002": r002,
        "posiciones_cambiadas": inversiones,
        "ganador_dx004": ganador_004,
        "ganador_dx002": ganador_002,
        "ranking_estable": len(inversiones) == 0,
        "veredicto": (
            f"Ranking {'ESTABLE' if not inversiones else 'CAMBIA'} entre dx=0.004 y dx=0.002. "
            f"Ganador dx004={ganador_004}, dx002={ganador_002}."
        ),
    }
    dst = os.path.join(CS, "t1_ranking_dx.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n[T1] " + out["veredicto"])
    print(f"[T1] Escrito {dst}")


if __name__ == "__main__":
    main()

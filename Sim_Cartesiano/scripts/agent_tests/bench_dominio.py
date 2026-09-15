"""
Coste por iteracion en funcion del tamano de dominio, a dx=0.002 fijo.

El estirado satura en dx_max, asi que ampliar el dominio apenas anade celdas;
lo que puede encarecerse es el Poisson de presion. Este script lo mide.

Un caso por invocacion (subproceso limpio -> sin fragmentacion de memoria GPU
entre casos). Escribe una linea JSON en el fichero de resultados.

Uso:
    .venv/bin/python scripts/agent_tests/bench_dominio.py --lx 12 --ly 8 --cx 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "agent_tests"))
os.chdir(ROOT)

import verificacion_numerica as vn

PERFIL = ("results/convergence_study/islands_tfg2/NACA_0012_sharp/epoch10/"
          "NACA_0012_sharp_winner_Re100000_a4.0_LD27.75.dat")
OUT = f"{vn.OUT}/bench_dominio.jsonl"
ITERS_PLENO = 13000
N_ALPHAS = 6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lx", type=float, required=True)
    p.add_argument("--ly", type=float, required=True)
    p.add_argument("--cx", type=float, required=True)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=4.0)
    a = p.parse_args()

    extra = {"Lx": a.lx, "Ly": a.ly, "cx": a.cx}
    t0 = time.time()
    r = vn.simular(PERFIL, 1e5, 0.002, alpha=a.alpha, iters=a.iters, extra=extra)
    wall = time.time() - t0
    if r is None:
        print(f"FALLO Lx={a.lx} Ly={a.ly}")
        return 1

    fila = {
        "Lx": a.lx, "Ly": a.ly, "cx": a.cx, "cy": a.ly / 2.0,
        "iters": a.iters,
        "wall_s": round(wall, 1),
        "it_s": round(a.iters / wall, 2),
        "cl": r["cl"], "cd": r["cd"], "ld": r["ld"],
        # extrapolacion a la polar completa (6 alphas x presupuesto pleno)
        "polar_h": round(ITERS_PLENO / (a.iters / wall) * N_ALPHAS / 3600.0, 2),
    }
    with open(OUT, "a") as f:
        f.write(json.dumps(fila) + "\n")
    print(json.dumps(fila))
    return 0


if __name__ == "__main__":
    sys.exit(main())

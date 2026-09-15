"""
Informe de ghost cells en el borde de salida para las tres mallas del estudio
de Richardson de la polar (dx = 0.004 / 0.002 / 0.001, dominio C 24x16).

Es el chequeo previo del punto 1 de RESUMEN.md: si n_irrep_te salta entre
mallas, el error de discretizacion no sera suave en dx y el orden observado p
no significara nada, por muchos angulos que se barran.

Solo construye la malla y corre una iteracion (no hay simulacion): ~1 min.

Uso:
    .venv/bin/python scripts/agent_tests/te_report_mallas.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "agent_tests"))
os.chdir(ROOT)

import Simulador2D
import RunGA
import verificacion_numerica as vn

DAT = ("results/convergence_study/islands_tfg2/NACA_0012_sharp/epoch10/"
       "NACA_0012_sharp_winner_Re100000_a4.0_LD27.75.dat")
DOMINIO = {"Lx": 24.0, "Ly": 16.0, "cx": 6.0}
DXS = (0.004, 0.002, 0.001)
ALPHA = 4.0
OUT = os.path.join(ROOT, "results", "verificacion_numerica", "te_report_mallas.json")


def informe(dx):
    mesh = Simulador2D.main(
        filepath=DAT, alpha_deg=ALPHA, chord=1.0,
        iteraciones=1, guardado=1,
        v0x=1.0, v0y=0.0, rho=1.0, nu=1e-5, CFL=0.5,
        dx_min=dx, min_te_height_factor=1.0,
        turb_model="sa", transition_model="sa_bc", freestream_Tu=0.1,
        wall_treatment="consistent", advection_scheme="maccormack",
        wake_refinement_mode="long_fine_x",
        mg_niveles_max=2, mg_max_outer=8, divergencia=0.02,
        stop_on_convergence=False, stop_on_clcd_convergence=False,
        graficos=False, live_view=False, mostrar_malla=False,
        **DOMINIO,
    )
    r = mesh.debug_te_report()
    r["dx"] = dx
    r["nodos"] = [int(len(mesh.X_1d_f64)), int(len(mesh.Y_1d_f64))]
    r["te"] = [float(mesh._airfoil_te_x), float(mesh._airfoil_te_y)]
    return r


def main():
    salida = {}
    for dx in DXS:
        vn._log(f"--- dx={dx} ---")
        salida[f"{dx:.4f}"] = informe(dx)
        with open(OUT, "w") as f:
            json.dump(salida, f, indent=2)

    print("\n" + "=" * 72)
    print(f"{'dx':>8s} {'nodos':>14s} {'ghosts TE':>10s} {'irrep TE':>9s} {'irrep tot':>10s}")
    print("-" * 72)
    for dx in DXS:
        r = salida[f"{dx:.4f}"]
        nod = f"{r['nodos'][0]}x{r['nodos'][1]}"
        print(f"{dx:>8.4f} {nod:>14s} {r['n_ghost_te']:>10d} "
              f"{r['n_irrep_te']:>9d} {r['n_irrep_total']:>10d}")
    print("=" * 72)


if __name__ == "__main__":
    main()

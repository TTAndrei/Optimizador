"""Compara bit a bit dos versiones de Simulador2D sobre el mismo caso.

El repo tiene un estudio cerrado y publicado, asi que cualquier refactor tiene
que demostrar que no ha movido ni un bit por el camino antiguo. `bench_opt.py
--compare` mide Cl y Cd con una puerta del 0.5%, que sirve para aceptar una
optimizacion pero NO para demostrar equivalencia: un cambio puede colarse dentro
de esa banda. Esto compara los campos enteros con np.array_equal.

Cada version corre en su propio proceso: main() deja estado global (pools de
CuPy, figuras, memoria compartida) y compartirlo entre dos modulos distintos
que se llaman igual es pedir un falso negativo.

Uso:
    .venv/bin/python scripts/agent_tests/equivalencia_bit.py \
        --ref /ruta/Simulador2D_antiguo.py [--iters 200] [--dx 0.008]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

import numpy as np

PERFIL = os.path.join(
    ROOT, "results/convergence_study/islands_tfg2/NACA_0012_sharp/epoch10/"
          "NACA_0012_sharp_winner_Re100000_a4.0_LD27.75.dat")

CASO = {
    "v0x": 1.0, "v0y": 0.0, "chord": 1.0, "rho": 1.0, "nu": 1e-5,
    "CFL": 0.5, "alpha_deg": 4.0,
    "Lx": 24.0, "Ly": 16.0, "cx": 6.0,
    "ancho_zona_fina_x": 1.5, "ancho_zona_fina_y": 1.0, "factor_expansion": 1.1,
    "turb_model": "sa", "wall_treatment": "consistent",
    "advection_scheme": "maccormack", "min_te_height_factor": 1.0,
    "wake_refinement_mode": "long_fine_x",
    "transition_model": "sa_bc", "freestream_Tu": 0.1,
    "mg_niveles_max": 2, "mg_max_outer": 2, "mg_cycles_per_outer": 3,
    "divergencia": 0.02,
    "stop_on_clcd_convergence": False, "stop_on_convergence": False,
    "guardado": 25, "graficos": False, "live_view": False,
    "mostrar_malla": False, "save_frames": False,
    "filepath": PERFIL,
}

CAMPOS = ("u", "v", "p", "solid")


def corre(modulo_path, salida, iters, dx):
    """Carga el modulo indicado, simula y vuelca los campos. Un proceso."""
    import cupy as cp
    # spec_from_file_location decide el loader por la extension, asi que un
    # fichero de referencia guardado como .py.algo no se puede cargar: se copia
    # a un .py temporal.
    if not modulo_path.endswith(".py"):
        tmp_py = os.path.join(tempfile.mkdtemp(prefix="equiv_mod_"), "Simulador2D_ref.py")
        shutil.copyfile(modulo_path, tmp_py)
        modulo_path = tmp_py
    spec = importlib.util.spec_from_file_location("_sim_bajo_prueba", modulo_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_sim_bajo_prueba"] = mod
    spec.loader.exec_module(mod)

    box = tempfile.mkdtemp(prefix="equiv_")
    prev = os.getcwd()
    os.chdir(box)
    try:
        mesh = mod.main(**{**CASO, "dx_min": dx, "iteraciones": iters})
    finally:
        os.chdir(prev)

    datos = {c: cp.asnumpy(getattr(mesh, c)) for c in CAMPOS}
    datos["cl"] = cp.asnumpy(mesh.clvector)
    datos["cd"] = cp.asnumpy(mesh.cdvector)
    np.savez(salida, **datos)
    print(f"[volcado] {salida}")


def compara(a, b):
    A, B = np.load(a), np.load(b)
    filas, todo_igual = [], True
    for k in list(CAMPOS) + ["cl", "cd"]:
        x, y = A[k], B[k]
        if x.shape != y.shape:
            filas.append((k, "FORMA DISTINTA", f"{x.shape} vs {y.shape}"))
            todo_igual = False
            continue
        igual = np.array_equal(x, y)
        todo_igual &= igual
        if igual:
            filas.append((k, "identico", f"{x.size} valores"))
        else:
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            n = int((d > 0).sum())
            filas.append((k, "DISTINTO",
                          f"{n}/{x.size} valores, max |d|={d.max():.3e}"))
    ancho = max(len(f[0]) for f in filas)
    for k, estado, detalle in filas:
        print(f"  {k:<{ancho}}  {estado:<14} {detalle}")
    print("\n" + ("EQUIVALENCIA BIT A BIT: SI" if todo_igual
                  else "EQUIVALENCIA BIT A BIT: NO"))
    return todo_igual


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="ruta al Simulador2D de referencia")
    ap.add_argument("--nueva", default=os.path.join(ROOT, "Simulador2D.py"))
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--dx", type=float, default=0.004)
    ap.add_argument("--corre", nargs=2, metavar=("MODULO", "SALIDA"),
                    help="modo interno: una sola version")
    a = ap.parse_args()

    if a.corre:
        corre(a.corre[0], a.corre[1], a.iters, a.dx)
        sys.exit(0)

    tmp = tempfile.mkdtemp(prefix="equiv_out_")
    npz_ref = os.path.join(tmp, "ref.npz")
    npz_new = os.path.join(tmp, "new.npz")
    for mod, out in ((a.ref, npz_ref), (a.nueva, npz_new)):
        r = subprocess.run([sys.executable, __file__, "--corre", mod, out,
                            "--iters", str(a.iters), "--dx", str(a.dx)],
                           cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"fallo corriendo {mod}")
    sys.exit(0 if compara(npz_ref, npz_new) else 1)

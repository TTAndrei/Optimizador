"""
Perfilado del BUCLE de tiempo solo, sin los diagnosticos finales.

El primer perfilado mezclaba las dos cosas. Al terminar, main() ejecuta
diagnosticos —auditoria de fuerzas de superficie, perfiles de Cp, circulacion,
ajustes polinomicos— que son un coste FIJO, se ejecutan una vez, y usan float64.
Con 1000 iteraciones eran ~20 s de una traza de 84 s, y encima concentrados en
justo el tipo de operacion que se estaba investigando: no habia forma de saber si
el 24% de float64 era del bucle o de la cola.

Aqui se anula esa funcion antes de correr, asi que la traza contiene el bucle y
nada mas. Los numeros que salgan si son atribuibles al paso de tiempo.

Uso:
    nsys profile --trace=cuda -o salida .venv/bin/python \\
        scripts/agent_tests/perfilar_bucle.py [ITERS]
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "agent_tests"))
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")

import Simulador2D
import opt_solver
import bench_opt as B

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
# Las opciones se pasan por argumento para poder re-perfilar configuraciones
# distintas: el reparto porcentual de un perfil deja de valer en cuanto se quita
# una de las partidas grandes, porque las demas suben de peso sin haber cambiado
# de coste absoluto.
FLAGS = sys.argv[2].split(",") if len(sys.argv) > 2 else \
    ["warm_start", "coarse_mask_majority"]

# Anular los diagnosticos finales: son coste fijo y contaminan el reparto.
Simulador2D.generar_graficos_y_outputs = lambda *a, **k: None

# Las opciones NO pueden ir por OPT_SOLVER aqui si se pasa por bench_opt, porque
# run_variant hace reset(). Aqui se llama a main() directamente, asi que se
# activan a mano y se comprueba que quedaron puestas.
opt_solver.reset()
opt_solver.enable(*FLAGS)
assert opt_solver.active() == sorted(FLAGS), (opt_solver.active(), FLAGS)
print(f"[perfil] opciones activas: {opt_solver.active()}")

params = {
    **B.BASE,
    "filepath": B.PERFIL,
    "iteraciones": ITERS,
    "mg_max_outer": 2,
    "mg_cycles_per_outer": 3,
    "guardado": 50,
}

prev = os.getcwd()
os.chdir(B.SANDBOX)
try:
    Simulador2D.main(**params)
finally:
    os.chdir(prev)

print(f"[perfil] bucle de {ITERS} iteraciones terminado, sin diagnosticos")

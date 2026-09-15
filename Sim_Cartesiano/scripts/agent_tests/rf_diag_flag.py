"""
Cual de las cuatro optimizaciones no soporta el presupuesto recortado.

El cruce anterior (rf_diag_blowup.py) dejo claro que el blowup necesita las dos
cosas a la vez:

    optimizaciones ON  + presupuesto 2x3  ->  revienta en la iteracion 1340
    optimizaciones ON  + presupuesto 8x5  ->  OK
    optimizaciones OFF + presupuesto 2x3  ->  OK

O sea que una de las cuatro opciones deja de ser inocua cuando la proyeccion no
tiene ciclos de sobra. Aqui se quita una cada vez, manteniendo 2x3: la que al
quitarla haga sobrevivir la simulacion es la responsable.

Se hace leave-one-out y no one-at-a-time porque lo que se busca es la opcion
necesaria para el fallo, no la suficiente: si dos interactuan, one-at-a-time no
lo veria.

Uso:
    .venv/bin/python scripts/agent_tests/rf_diag_flag.py [ITERS]
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "agent_tests"))
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")

import opt_solver
import resultados_finales as RF
import verificacion_numerica as vn

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
DX, ALPHA = 0.004, 0.0
DAT = os.path.join(RF.OUT, "perfiles", "ag24.dat")
PRESU = {"mg_max_outer": 2, "mg_cycles_per_outer": 3}

CASOS = [(f"sin_{q}", tuple(x for x in RF.FLAGS if x != q)) for q in RF.FLAGS]

OUT = os.path.join(ROOT, "results", "bench_opt", "diag_flag.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

for tag, flags in CASOS:
    if tag in res:
        print(f"[{tag}] ya hecho: {res[tag]['estado']}")
        continue
    opt_solver.reset()
    if flags:
        opt_solver.enable(*flags)
    assert opt_solver.active() == sorted(flags), opt_solver.active()
    print(f"\n=== {tag}  activas={opt_solver.active()} ===", flush=True)
    t0 = time.time()
    try:
        r = vn.simular(DAT, RF.RE, DX, alpha=ALPHA, iters=ITERS,
                       extra={**RF.DOMINIO, **RF.SIN_PARADAS, **PRESU})
        d = {"estado": "OK" if r is not None else "ABORTADA",
             "wall_s": round(time.time() - t0, 1),
             "cl": (r or {}).get("cl"), "cd": (r or {}).get("cd")}
    except Exception as e:
        d = {"estado": "EXCEPCION", "wall_s": round(time.time() - t0, 1),
             "error": str(e)[:200]}
    res[tag] = d
    json.dump(res, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"[{tag}] {d}", flush=True)

print("\n" + "=" * 70)
print(f"{'quitando':<28}{'estado':>12}{'wall':>8}   {'Cl':>9}")
print("-" * 70)
for tag, flags in CASOS:
    d = res.get(tag, {})
    cl = d.get("cl")
    print(f"{tag[4:]:<28}{d.get('estado','?'):>12}{d.get('wall_s',0):>7.0f}s"
          f"{(f'{cl:>10.5f}' if cl is not None else f'{'-':>10}')}")
print("=" * 70)
print("La opcion que al quitarla evita el blowup es la responsable.")

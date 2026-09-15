"""
Que hace warm_start en la malla fina, que es donde esta el 85% del coste.

El bucle de proyeccion NO tiene salida por convergencia: el unico break es por
NaN, asi que siempre ejecuta mg_max_outer x mg_cycles_per_outer ciclos completos.
Luego warm_start no ahorra trabajo — hace los mismos ciclos partiendo de mejor
semilla. Lo que aporta es calidad por ciclo, y eso es lo que hay que medir.

Dos cosas por configuracion, con dx=0.001 y pocas iteraciones:

    it/s                velocidad real a esta escala. Si la diferencia es ~0,
                        quitarlo no cuesta tiempo y la "trampa" no compra nada.
    divergencia media   calidad de la proyeccion, sacada de la serie volcada.
                        Es la medida directa de cuanto se degrada al quitarlo.
    estabilidad         si tambien revienta aqui, como en dx=0.004.

Uso:
    .venv/bin/python scripts/agent_tests/rf_diag_fina.py [ITERS]
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

import numpy as np
import matplotlib
matplotlib.use("Agg")

import opt_solver
import resultados_finales as RF
import verificacion_numerica as vn

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 800
DX, ALPHA = 0.001, 4.0
PRESU = {"mg_max_outer": 2, "mg_cycles_per_outer": 3}
DAT = os.path.join(RF.OUT, "perfiles", "ag24.dat")
SER = os.path.join(ROOT, "results", "bench_opt", "fina_%s.npz")

CASOS = [("con_warm", RF.FLAGS),
         ("sin_warm", tuple(f for f in RF.FLAGS if f != "warm_start"))]

OUT = os.path.join(ROOT, "results", "bench_opt", "diag_fina.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

for tag, flags in CASOS:
    if tag in res:
        print(f"[{tag}] ya hecho")
        continue
    opt_solver.reset()
    opt_solver.enable(*flags)
    assert opt_solver.active() == sorted(flags), opt_solver.active()
    print(f"\n=== {tag}  dx={DX} alpha={ALPHA}  {ITERS} iters  "
          f"activas={opt_solver.active()} ===", flush=True)
    t0 = time.time()
    r = vn.simular(DAT, RF.RE, DX, alpha=ALPHA, iters=ITERS,
                   extra={**RF.DOMINIO, **RF.SIN_PARADAS, **PRESU},
                   dump=SER % tag)
    wall = time.time() - t0
    d = {"estado": "OK" if r is not None else "ABORTADA", "wall_s": round(wall, 1),
         "cl": (r or {}).get("cl"), "cd": (r or {}).get("cd")}
    p = SER % tag
    if os.path.exists(p):
        z = np.load(p)
        if "res" in z.files and z["res"].size:
            # Ultima columna de res = divergencia; se promedia la segunda mitad
            # para no mezclar el arranque, donde todavia no hay semilla que reusar.
            dv = np.abs(np.asarray(z["res"])[:, -1])
            d["div_media_2a_mitad"] = float(np.mean(dv[len(dv) // 2:]))
            d["div_final"] = float(dv[-1])
        d["n_muestras"] = int(len(z["t"]))
    res[tag] = d
    json.dump(res, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"[{tag}] {d}", flush=True)

print("\n" + "=" * 78)
print(f"{'caso':<12}{'estado':>10}{'wall':>9}{'it/s':>8}{'div media':>14}{'div final':>14}")
print("-" * 78)
for tag, _ in CASOS:
    d = res.get(tag, {})
    w = d.get("wall_s") or 0
    print(f"{tag:<12}{d.get('estado','?'):>10}{w:>8.0f}s{ITERS/w if w else 0:>8.2f}"
          f"{d.get('div_media_2a_mitad', float('nan')):>14.3e}"
          f"{d.get('div_final', float('nan')):>14.3e}")
print("-" * 78)
c, s = res.get("con_warm", {}), res.get("sin_warm", {})
if c.get("wall_s") and s.get("wall_s"):
    print(f"coste de quitar warm_start: {100*(s['wall_s']-c['wall_s'])/c['wall_s']:+.1f}% de wall")
if c.get("div_media_2a_mitad") and s.get("div_media_2a_mitad"):
    print(f"divergencia al quitarlo:    x{s['div_media_2a_mitad']/c['div_media_2a_mitad']:.2f}")
print("El wall incluye el mallado (4.75M celdas), igual en los dos casos,")
print("asi que la diferencia porcentual real del bucle es mayor que la de la tabla.")

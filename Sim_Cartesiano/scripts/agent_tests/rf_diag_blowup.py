"""
Por que revienta ag24 dx=0.004 alpha=0 con la configuracion nueva.

El punto homologo del estudio publicado completo sus 13000 iteraciones con el
solver antiguo y presupuesto de proyeccion 8x5. Con las optimizaciones activas y
presupuesto 2x3 revienta en la iteracion 1340 por blowup de velocidad, despues de
que la viscosidad efectiva del modelo SA suba a 76 veces la molecular.

Hay dos sospechosos y no se pueden separar razonando, asi que se cruzan:

    presupuesto 2x3 / 8x5   x   optimizaciones ON / OFF

Basta con 2000 iteraciones: si va a reventar, revienta antes de la 1400.

Uso:
    .venv/bin/python scripts/agent_tests/rf_diag_blowup.py [ITERS]
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

CASOS = [
    ("opt_2x3", RF.FLAGS, {"mg_max_outer": 2, "mg_cycles_per_outer": 3}),
    ("opt_8x5", RF.FLAGS, {"mg_max_outer": 8, "mg_cycles_per_outer": 5}),
    ("sin_2x3", (),       {"mg_max_outer": 2, "mg_cycles_per_outer": 3}),
    ("sin_8x5", (),       {"mg_max_outer": 8, "mg_cycles_per_outer": 5}),
]

OUT = os.path.join(ROOT, "results", "bench_opt", "diag_blowup.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

for tag, flags, presu in CASOS:
    if tag in res:
        print(f"[{tag}] ya hecho: {res[tag]['estado']}")
        continue
    opt_solver.reset()
    if flags:
        opt_solver.enable(*flags)
    assert opt_solver.active() == sorted(flags), opt_solver.active()
    print(f"\n=== {tag}  opts={opt_solver.active()}  {presu} ===", flush=True)

    t0 = time.time()
    try:
        r = vn.simular(DAT, RF.RE, DX, alpha=ALPHA, iters=ITERS,
                       extra={**RF.DOMINIO, **RF.SIN_PARADAS, **presu})
        estado = "OK" if r is not None else "ABORTADA"
        d = {"estado": estado, "wall_s": round(time.time() - t0, 1),
             "iters_efectivas": (r or {}).get("iters_efectivas"),
             "cl": (r or {}).get("cl"), "cd": (r or {}).get("cd")}
    except Exception as e:
        d = {"estado": "EXCEPCION", "wall_s": round(time.time() - t0, 1),
             "error": str(e)[:200]}
    res[tag] = d
    json.dump(res, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"[{tag}] {d}", flush=True)

print("\n" + "=" * 66)
print(f"{'caso':<10}{'optimiz.':>10}{'presup.':>10}{'estado':>12}{'wall':>8}")
print("-" * 66)
for tag, flags, presu in CASOS:
    d = res.get(tag, {})
    print(f"{tag:<10}{'ON' if flags else 'OFF':>10}"
          f"{presu['mg_max_outer']}x{presu['mg_cycles_per_outer']:<8}"
          f"{d.get('estado','?'):>12}{d.get('wall_s',0):>7.0f}s")
print("=" * 66)

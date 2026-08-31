"""
El L/D de 41.26 viene de warm_start o de que el presupuesto se quedo corto?

Mismo punto —ganador del AG, dx=0.002, alpha=4, 26000 iteraciones, sin paradas—
con las cuatro optimizaciones y presupuesto creciente. Los tres valores conocidos
de este punto salen de configuraciones distintas y difieren en un factor 2.1:

    publicada, solver viejo, 8x5      Cl=0.68039  Cd=0.01999  L/D=34.04
    optimizada 2x3 CON warm_start     Cl=0.69770  Cd=0.01691  L/D=41.26
    optimizada 2x3 SIN warm_start     Cl=0.59070  Cd=0.02979  L/D=19.83

Si al subir el presupuesto manteniendo warm_start el L/D se queda cerca de 41, la
exactitud la aporta warm_start y el presupuesto solo cuesta tiempo. Si baja hacia
34, el 41.26 era el artefacto de una proyeccion sin converger y el presupuesto
corto es lo que lo producia.

Se corren dos presupuestos, no uno: con 2x3 (ya conocido), 4x3 y 8x5 salen tres
puntos de una curva. Un valor suelto no distingue "converge a 41" de "va de
camino a 34".

Uso:
    .venv/bin/python scripts/agent_tests/rf_diag_presupuesto.py
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

DX, ALPHA, ITERS = 0.002, 4.0, 26000
DAT = os.path.join(RF.OUT, "perfiles", "ag24.dat")
# Sin paradas, como la corrida que dio 41.26: con parada la ventana de promediado
# seria otra y la comparacion dejaria de ser del presupuesto.
SIN_PARADAS = {"stop_on_clcd_convergence": False, "stop_on_convergence": False}

# De mas barato a mas caro: si 4x3 ya se mueve hacia 34, no hace falta esperar
# la hora larga de 8x5 para saber la respuesta.
CASOS = [("o4c3", {"mg_max_outer": 4, "mg_cycles_per_outer": 3}),
         ("o8c5", {"mg_max_outer": 8, "mg_cycles_per_outer": 5})]

REF = {"2x3 con warm (conocido)": (0.697701, 0.016912, 41.2558),
       "8x5 solver viejo (publicado)": (0.680389, 0.019986, 34.0440),
       "2x3 sin warm": (0.590700, 0.029790, 19.8300)}

OUT = os.path.join(ROOT, "results", "bench_opt", "diag_presupuesto.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

for tag, presu in CASOS:
    if tag in res:
        print(f"[{tag}] ya hecho: L/D={res[tag].get('ld')}")
        continue
    opt_solver.reset()
    opt_solver.enable(*RF.FLAGS)          # las cuatro, warm_start incluido
    assert opt_solver.active() == sorted(RF.FLAGS), opt_solver.active()
    ciclos = presu["mg_max_outer"] * presu["mg_cycles_per_outer"]
    print(f"\n=== {tag}  {presu}  = {ciclos} ciclos totales  "
          f"activas={opt_solver.active()} ===", flush=True)
    t0 = time.time()
    try:
        r = vn.simular(DAT, RF.RE, DX, alpha=ALPHA, iters=ITERS,
                       extra={**RF.DOMINIO, **SIN_PARADAS, **presu},
                       dump=os.path.join(ROOT, "results", "bench_opt",
                                         f"presu_{tag}.npz"))
        d = ({"estado": "OK", "cl": r["cl"], "cd": r["cd"], "ld": r["ld"],
              "cl_ci95": r.get("cl_ci95"), "cd_ci95": r.get("cd_ci95"),
              "iters_efectivas": r.get("iters_efectivas"), "ciclos": ciclos}
             if r is not None else {"estado": "ABORTADA", "ciclos": ciclos})
        d["wall_s"] = round(time.time() - t0, 1)
    except Exception as e:
        d = {"estado": "EXCEPCION", "error": str(e)[:200], "ciclos": ciclos,
             "wall_s": round(time.time() - t0, 1)}
    res[tag] = d
    json.dump(res, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"[{tag}] {d}", flush=True)

print("\n" + "=" * 84)
print("GANADOR AG, dx=0.002, alpha=4 — L/D frente al presupuesto de proyeccion")
print("=" * 84)
print(f"{'configuracion':<32}{'ciclos':>8}{'Cl':>10}{'Cd':>10}{'L/D':>9}{'wall':>9}")
print("-" * 84)
print(f"{'2x3 CON warm (conocido)':<32}{6:>8}{REF['2x3 con warm (conocido)'][0]:>10.5f}"
      f"{REF['2x3 con warm (conocido)'][1]:>10.5f}{REF['2x3 con warm (conocido)'][2]:>9.3f}"
      f"{1053:>8}s")
for tag, _ in CASOS:
    d = res.get(tag, {})
    if d.get("estado") != "OK":
        print(f"{tag:<32}{d.get('ciclos',0):>8}{d.get('estado','?'):>29}")
        continue
    print(f"{tag+' CON warm':<32}{d['ciclos']:>8}{d['cl']:>10.5f}{d['cd']:>10.5f}"
          f"{d['ld']:>9.3f}{d['wall_s']:>8.0f}s")
print("-" * 84)
for n, (cl, cd, ld) in REF.items():
    if "conocido" not in n:
        print(f"  referencia {n:<32} Cl={cl:.5f} Cd={cd:.5f} L/D={ld:.3f}")
print("\n  Si el L/D se queda cerca de 41 al subir ciclos -> lo aporta warm_start.")
print("  Si baja hacia 34 -> el 41.26 era artefacto del presupuesto corto.")

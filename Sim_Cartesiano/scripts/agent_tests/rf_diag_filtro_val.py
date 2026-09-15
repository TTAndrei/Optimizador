"""
Validacion del filtro de tablero: no rompe lo que funcionaba, y arregla lo que no.

El filtro ya demostro que estabiliza el punto que reventaba (dx=0.004, alpha=0:
de abortar en la iteracion 1340 a completar las 13000). Faltan las dos preguntas
que deciden si sirve para el estudio:

  exactitud   En dx=0.002 alpha=4 el solver con warm_start da L/D 41.26 a 2x3 y
              42.05 a 8x5, o sea una curva de presupuesto plana. Si el filtro se
              lleva por delante contenido util de la semilla, ese numero se
              movera. Debe seguir cerca de 42.

  malla fina  dx=0.001 es donde reventaba en la iteracion ~1740 y donde estan las
              11 h del estudio. Con 3000 iteraciones hay margen de sobra sobre
              ese umbral para dar veredicto de estabilidad.

Referencias del punto de exactitud (ganador AG, dx=0.002, alpha=4):
    2x3 CON warm, sin filtro     Cl=0.697701  Cd=0.016912  L/D=41.256
    4x3 CON warm, sin filtro     Cl=0.697249  Cd=0.016604  L/D=41.994
    8x5 CON warm, sin filtro     Cl=0.696940  Cd=0.016574  L/D=42.051

Uso:
    .venv/bin/python scripts/agent_tests/rf_diag_filtro_val.py
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

FLAGS = ("coarse_mask_majority", "fast_masks", "interp_float32",
         "warm_start", "warm_start_filtered")
PRESU = {"mg_max_outer": 2, "mg_cycles_per_outer": 3}
DAT = os.path.join(RF.OUT, "perfiles", "ag24.dat")
SIN_PARADAS = {"stop_on_clcd_convergence": False, "stop_on_convergence": False}

# La fina primero: es la mas arriesgada y la mas barata de las dos, asi que si
# falla no se han gastado 18 min en la otra.
CASOS = [
    ("fina_estabilidad", 0.001, 4.0,  3000),
    ("media_exactitud",  0.002, 4.0, 26000),
]

REF = {"2x3 sin filtro": 41.2558, "4x3 sin filtro": 41.9939, "8x5 sin filtro": 42.0508}

OUT = os.path.join(ROOT, "results", "bench_opt", "diag_filtro_val.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

for tag, dx, alpha, iters in CASOS:
    if tag in res:
        print(f"[{tag}] ya hecho: {res[tag]['estado']}")
        continue
    opt_solver.reset()
    opt_solver.enable(*FLAGS)
    assert opt_solver.active() == sorted(FLAGS), opt_solver.active()
    print(f"\n=== {tag}  dx={dx} alpha={alpha} {iters} iters {PRESU}\n"
          f"    activas={opt_solver.active()} ===", flush=True)
    t0 = time.time()
    try:
        r = vn.simular(DAT, RF.RE, dx, alpha=alpha, iters=iters,
                       extra={**RF.DOMINIO, **SIN_PARADAS, **PRESU})
        d = ({"estado": "OK", "cl": r["cl"], "cd": r["cd"], "ld": r["ld"],
              "cl_ci95": r.get("cl_ci95"), "cd_ci95": r.get("cd_ci95"),
              "iters_efectivas": r.get("iters_efectivas")}
             if r is not None else {"estado": "ABORTADA"})
        d.update(dx=dx, alpha=alpha, iters=iters,
                 wall_s=round(time.time() - t0, 1))
    except Exception as e:
        d = {"estado": "EXCEPCION", "error": str(e)[:200], "dx": dx,
             "wall_s": round(time.time() - t0, 1)}
    res[tag] = d
    json.dump(res, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"[{tag}] {d}", flush=True)

print("\n" + "=" * 88)
print("FILTRO DE TABLERO — validacion")
print("=" * 88)
for tag, dx, alpha, iters in CASOS:
    d = res.get(tag, {})
    print(f"\n  {tag}   dx={dx:g}  alpha={alpha:g}  {iters} iters")
    if d.get("estado") != "OK":
        print(f"    -> {d.get('estado','?')}   {d.get('error','')}")
        continue
    print(f"    -> OK   Cl={d['cl']:.5f}  Cd={d['cd']:.5f}  L/D={d['ld']:.3f}"
          f"   ({d['wall_s']:.0f}s)")
    if tag == "media_exactitud":
        print("       comparado con el mismo punto SIN filtro:")
        for n, v in REF.items():
            print(f"         {n:<18} L/D={v:7.3f}   dif = {100*(d['ld']-v)/v:+6.2f}%")
    if tag == "fina_estabilidad":
        print("       el blowup sin filtro llegaba en la iteracion ~1740")
print()

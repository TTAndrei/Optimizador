"""
Prueba del reescalado de la semilla de warm_start por el cambio de dt.

Punto elegido: ganador del AG, dx=0.004, alpha=0. Es EL punto que revienta —con
warm_start y presupuesto 2x3 aborta por blowup en la iteracion 1340, de forma
reproducible— asi que es el unico juez util.

Se corren las 13000 iteraciones enteras, no un trozo. La leccion de la prueba de
dx=0.001 fue justamente esa: 800 iteraciones dieron "OK" y el blowup llegaba en
la 1740, o sea que una prueba corta valida algo falso. Aqui el veredicto de
estabilidad se da a longitud completa.

Tres casos:
    warm            reproduce el fallo, para no dar por hecho que sigue ahi
    warm_escalado   el arreglo
    sin_warm        el que ya sabemos que sobrevive, como referencia de valor

Referencias de este punto:
    publicada, solver viejo 8x5     Cl=0.07441  Cd=0.04833
    2x3 sin warm_start (estudio)    Cl=0.07102  Cd=0.04833

Uso:
    .venv/bin/python scripts/agent_tests/rf_diag_escalado.py
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

DX, ALPHA, ITERS = 0.004, 0.0, 13000
PRESU = {"mg_max_outer": 2, "mg_cycles_per_outer": 3}
DAT = os.path.join(RF.OUT, "perfiles", "ag24.dat")
SIN_PARADAS = {"stop_on_clcd_convergence": False, "stop_on_convergence": False}
SER = os.path.join(ROOT, "results", "bench_opt", "esc_%s.npz")

BASE = ("coarse_mask_majority", "fast_masks", "interp_float32")
CASOS = [
    ("warm",          BASE + ("warm_start",)),
    ("warm_escalado", BASE + ("warm_start", "warm_start_scaled")),
    # El reescalado por dt solo movio el fallo de la iteracion 1340 a la 1800:
    # la magnitud de la semilla no era la causa. Lo que queda es su estructura,
    # o sea el modo de tablero que arrastra y que el suavizador no reduce.
    ("warm_filtrado", BASE + ("warm_start", "warm_start_filtered")),
    ("warm_filt_esc", BASE + ("warm_start", "warm_start_filtered",
                              "warm_start_scaled")),
]

OUT = os.path.join(ROOT, "results", "bench_opt", "diag_escalado.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

for tag, flags in CASOS:
    if tag in res:
        print(f"[{tag}] ya hecho: {res[tag]['estado']}")
        continue
    opt_solver.reset()
    opt_solver.enable(*flags)
    assert opt_solver.active() == sorted(flags), opt_solver.active()
    print(f"\n=== {tag}  dx={DX} alpha={ALPHA} {ITERS} iters {PRESU}\n"
          f"    activas={opt_solver.active()} ===", flush=True)
    t0 = time.time()
    try:
        r = vn.simular(DAT, RF.RE, DX, alpha=ALPHA, iters=ITERS,
                       extra={**RF.DOMINIO, **SIN_PARADAS, **PRESU},
                       dump=SER % tag)
        d = ({"estado": "OK", "cl": r["cl"], "cd": r["cd"], "ld": r["ld"],
              "cl_ci95": r.get("cl_ci95"), "cd_ci95": r.get("cd_ci95"),
              "iters_efectivas": r.get("iters_efectivas")}
             if r is not None else {"estado": "ABORTADA"})
        d["wall_s"] = round(time.time() - t0, 1)
    except Exception as e:
        d = {"estado": "EXCEPCION", "error": str(e)[:200],
             "wall_s": round(time.time() - t0, 1)}
    p = SER % tag
    if os.path.exists(p):
        z = np.load(p)
        if "res" in z.files and z["res"].size:
            dv = np.abs(np.asarray(z["res"])[:, -1])
            d["div_media_2a_mitad"] = float(np.mean(dv[len(dv) // 2:]))
    res[tag] = d
    json.dump(res, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"[{tag}] {d}", flush=True)

print("\n" + "=" * 86)
print(f"GANADOR AG, dx={DX}, alpha={ALPHA}, {ITERS} iters, presupuesto 2x3")
print("=" * 86)
print(f"{'caso':<16}{'estado':>10}{'Cl':>10}{'Cd':>10}{'L/D':>9}{'div media':>13}{'wall':>9}")
print("-" * 86)
for tag, _ in CASOS:
    d = res.get(tag, {})
    if d.get("estado") != "OK":
        print(f"{tag:<16}{d.get('estado','?'):>10}{'':>28}{d.get('wall_s',0):>31.0f}s")
        continue
    print(f"{tag:<16}{'OK':>10}{d['cl']:>10.5f}{d['cd']:>10.5f}{d['ld']:>9.3f}"
          f"{d.get('div_media_2a_mitad', float('nan')):>13.3e}{d['wall_s']:>8.0f}s")
print("-" * 86)
print("  referencia publicada (solver viejo, 8x5):  Cl=0.07441  Cd=0.04833")
print("\n  Si warm_escalado sobrevive las 13000, el arreglo funciona y warm_start")
print("  vuelve a ser utilizable en las tres mallas.")

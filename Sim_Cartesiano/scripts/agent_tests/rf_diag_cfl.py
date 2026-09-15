"""
Malla gruesa a CFL=0.25: sobrevive warm_start, y converge al mismo sitio?

Con CFL=0.5 la malla gruesa CON warm_start revienta en la iteracion 1340 por
blowup de velocidad. La hipotesis es que con dt a la mitad el paso de tiempo deja
de amplificar la divergencia residual que la proyeccion no llega a limpiar con
presupuesto 2x3, y deja de reventar.

Dos casillas, no una. La primera contesta si sobrevive; la segunda contesta algo
mas util: si a CFL bajo las dos configuraciones dan el MISMO resultado. Si
coinciden, la solucion ha dejado de depender de la semilla, la proyeccion si esta
convergida a ese CFL, y no hace falta mezclar configuraciones entre mallas.

CFL=0.25 dobla las iteraciones necesarias para el mismo tiempo fisico: 26000 en
vez de 13000 para t~20. Se activa la parada por convergencia de Cl/Cd (criterio
recalibrado) para recuperar parte de ese coste; donde para queda registrado en
iters_efectivas y t_conv_clcd, asi que es auditable.

Referencia publicada (dx=0.004, alpha=0, CFL=0.5, solver antiguo, 8x5):
    Cl = 0.07441   Cd = 0.04833

Uso:
    .venv/bin/python scripts/agent_tests/rf_diag_cfl.py [ALPHA]
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

ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
DX, CFL = 0.004, 0.25
ITERS = 26000              # t ~ 20 con dt a la mitad
PRESU = {"mg_max_outer": 2, "mg_cycles_per_outer": 3}
DAT = os.path.join(RF.OUT, "perfiles", "ag24.dat")

# Parada por convergencia de Cl/Cd activada. El criterio calibrado (tolerancias,
# ventana y sostenido) lo inyecta vn.simular desde criterio_parada.json.
CON_PARADA = {"stop_on_clcd_convergence": True, "stop_on_convergence": False}

CASOS = [("con_warm", RF.FLAGS),
         ("sin_warm", tuple(f for f in RF.FLAGS if f != "warm_start"))]

OUT = os.path.join(ROOT, "results", "bench_opt", "diag_cfl.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

for tag, flags in CASOS:
    if tag in res:
        print(f"[{tag}] ya hecho: {res[tag]['estado']}")
        continue
    opt_solver.reset()
    opt_solver.enable(*flags)
    assert opt_solver.active() == sorted(flags), opt_solver.active()
    print(f"\n=== {tag}  dx={DX} alpha={ALPHA} CFL={CFL} {ITERS} iters "
          f"parada=ON  activas={opt_solver.active()} ===", flush=True)
    t0 = time.time()
    try:
        r = vn.simular(DAT, RF.RE, DX, alpha=ALPHA, iters=ITERS,
                       extra={**RF.DOMINIO, **CON_PARADA, **PRESU, "CFL": CFL},
                       dump=os.path.join(ROOT, "results", "bench_opt",
                                         f"cfl025_{tag}.npz"))
        d = ({"estado": "OK", "cl": r["cl"], "cd": r["cd"], "ld": r["ld"],
              "cl_ci95": r.get("cl_ci95"), "cd_ci95": r.get("cd_ci95"),
              "cl_std": r.get("cl_std"), "cd_std": r.get("cd_std"),
              "n_samples": r.get("n_samples"),
              "converged_clcd": r.get("converged_clcd"),
              "t_conv_clcd": r.get("t_conv_clcd"),
              "iters_efectivas": r.get("iters_efectivas"),
              "iters_programadas": ITERS}
             if r is not None else {"estado": "ABORTADA"})
        d["wall_s"] = round(time.time() - t0, 1)
    except Exception as e:
        d = {"estado": "EXCEPCION", "wall_s": round(time.time() - t0, 1),
             "error": str(e)[:200]}
    res[tag] = d
    json.dump(res, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"[{tag}] {d}", flush=True)

print("\n" + "=" * 96)
print(f"MALLA GRUESA a CFL={CFL}, alpha={ALPHA}  (referencia publicada CFL=0.5: "
      f"Cl=0.07441 Cd=0.04833)")
print("=" * 96)
print(f"{'caso':<10}{'estado':>10}{'Cl':>11}{'Cd':>11}{'L/D':>9}"
      f"{'iters':>14}{'parado en t':>13}{'wall':>8}")
print("-" * 96)
for tag, _ in CASOS:
    d = res.get(tag, {})
    if d.get("estado") != "OK":
        print(f"{tag:<10}{d.get('estado','?'):>10}")
        continue
    tc = d.get("t_conv_clcd")
    print(f"{tag:<10}{'OK':>10}{d['cl']:>11.5f}{d['cd']:>11.5f}{d['ld']:>9.3f}"
          f"{d['iters_efectivas']:>7}/{d['iters_programadas']:<6}"
          f"{(f'{tc:.2f}' if tc else 'no paro'):>13}{d['wall_s']:>7.0f}s")
print("-" * 96)
c, s = res.get("con_warm", {}), res.get("sin_warm", {})
if c.get("estado") == "OK" and s.get("estado") == "OK":
    for m in ("cl", "cd", "ld"):
        dif = 100.0 * (s[m] - c[m]) / abs(c[m])
        ci = 100.0 * (c.get(f"{m}_ci95") or 0.0) / abs(c[m])
        veredicto = ("indistinguible del ruido -> la proyeccion ya no depende "
                     "de la semilla" if abs(dif) <= ci else
                     "sigue dependiendo de la semilla")
        print(f"  {m.upper():<4} sin-con = {dif:+7.2f}%   CI95 = {ci:5.2f}%   {veredicto}")
elif c.get("estado") == "OK":
    print("  con warm_start SOBREVIVE a CFL=0.25 (con CFL=0.5 reventaba en la iter 1340)")

"""
Cuanto sesga el Richardson desactivar warm_start solo en la malla gruesa.

El plan es correr dx=0.004 sin warm_start (con el, revienta) y dx=0.002 y 0.001
con las cuatro optimizaciones. Eso mezcla configuraciones dentro de la terna, y
la pregunta es si eso contamina el orden aparente y el GCI.

No se puede medir de frente: el caso "dx=0.004 CON warm_start" no existe, es el
que diverge. El sustituto es medirlo donde las dos configuraciones si corren, o
sea en dx=0.002, y comparar el desplazamiento que mete warm_start contra el que
mete el cambio de malla:

    si  |Cl(con) - Cl(sin)| << |Cl(dx=0.004) - Cl(dx=0.002)|   la mezcla se sostiene
    si son del mismo orden                                      no se sostiene

La referencia CON warm_start ya esta pagada (results/polar_optimizada). Aqui solo
se calcula la de SIN warm_start, misma malla, mismo presupuesto, mismas 26000
iteraciones.

Se usa alpha=0 porque es el peor caso por los dos lados: es donde la malla mas
mueve el resultado (+50.7% de 0.004 a 0.002 en el estudio previo) y donde la
proyeccion peor converge. Si el sesgo es pequeno ahi, lo es en todas partes.

Uso:
    .venv/bin/python scripts/agent_tests/rf_diag_mezcla.py [ALPHA ...]
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

ANGULOS = [float(a) for a in sys.argv[1:]] or [0.0]
DX = 0.002
SIN_WARM = tuple(f for f in RF.FLAGS if f != "warm_start")
PRESU = {"mg_max_outer": 2, "mg_cycles_per_outer": 3}
DAT = os.path.join(RF.OUT, "perfiles", "ag24.dat")

REF = json.load(open("results/polar_optimizada/polar_dx0.0020.json"))
GRUESA = json.load(open("results/richardson_polar_domC/polar_dx0.0040.json"))

OUT = os.path.join(ROOT, "results", "bench_opt", "diag_mezcla.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

opt_solver.reset()
opt_solver.enable(*SIN_WARM)
assert opt_solver.active() == sorted(SIN_WARM), opt_solver.active()
print(f"activas: {opt_solver.active()}   presupuesto: {PRESU}", flush=True)

for a in ANGULOS:
    ka = f"{a:.1f}"
    if ka in res:
        print(f"[a={ka}] ya hecho")
        continue
    print(f"\n=== dx={DX} alpha={a} SIN warm_start, {RF.ITERS[DX]} iters ===", flush=True)
    t0 = time.time()
    r = vn.simular(DAT, RF.RE, DX, alpha=a, iters=RF.ITERS[DX],
                   extra={**RF.DOMINIO, **RF.SIN_PARADAS, **PRESU})
    if r is None:
        print(f"[a={ka}] ABORTADA")
        continue
    r["wall_s"] = round(time.time() - t0, 1)
    res[ka] = r
    json.dump(res, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"[a={ka}] Cl={r['cl']:.6f} Cd={r['cd']:.6f} L/D={r['ld']:.4f} "
          f"({r['wall_s']:.0f}s)", flush=True)

print("\n" + "=" * 92)
print("SESGO DE LA MEZCLA frente al EFECTO DE MALLA  (perfil ag24)")
print("=" * 92)
print(f"{'a':>5} {'mag':>4} {'con warm':>11}{'sin warm':>11}{'sesgo':>9}{'CI95':>8}"
      f"   {'dx=0.004':>10}{'efecto malla':>14}{'ratio':>8}")
print("-" * 92)
for a in ANGULOS:
    ka = f"{a:.1f}"
    if ka not in res or ka not in REF:
        continue
    for mag in ("cl", "cd", "ld"):
        con, sin = REF[ka][mag], res[ka][mag]
        sesgo = 100.0 * (sin - con) / abs(con)
        ci = 100.0 * (REF[ka].get(f"{mag}_ci95") or 0.0) / abs(con)
        g = GRUESA.get(ka, {}).get(mag)
        if g is None:
            print(f"{a:>5.1f} {mag:>4} {con:>11.5f}{sin:>11.5f}{sesgo:>8.2f}%{ci:>7.2f}%")
            continue
        malla = 100.0 * (con - g) / abs(con)
        print(f"{a:>5.1f} {mag:>4} {con:>11.5f}{sin:>11.5f}{sesgo:>8.2f}%{ci:>7.2f}%"
              f"   {g:>10.5f}{malla:>13.1f}%{abs(sesgo/malla) if malla else 0:>8.3f}")
print("-" * 92)
print("sesgo        = cuanto mueve quitar warm_start, a malla igual")
print("efecto malla = cuanto mueve pasar de dx=0.004 a dx=0.002 (la senal del Richardson)")
print("ratio        = sesgo / efecto malla. Cuanto mas pequeno, mas inocua es la mezcla.")
print("CI95         = ruido estadistico del punto de referencia: por debajo de el,")
print("               el sesgo no es distinguible del ruido de promediado.")

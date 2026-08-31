"""
Polar del perfil ganador del AG con el solver optimizado, para comparar contra
la que ya esta publicada.

Config identica a la del estudio de Richardson en el dominio C: mismo perfil,
Lx=24, Ly=16, cx=6, dx=0.002, Re=1e5, CFL=0.5, 26000 iteraciones (t~20) y las
dos paradas desactivadas. Lo unico que cambia son las optimizaciones del solver
de presion.

Se corre por la MISMA ruta de codigo que el estudio —vn.simular, que arma la
config desde RunGA.CONFIG y llama a RunGA.simular_perfil— en vez de por el banco
de pruebas. El banco llama a Simulador2D.main directamente y calcula sus propias
medias, asi que produciria un subconjunto distinto de metricas; por aqui salen
exactamente las mismas 28 claves que guardo el estudio, y la comparacion es
columna a columna sin conversiones por medio.

Angulos 0, 2, 4, 6, 8, 10. Los cinco primeros tienen homologo directo en
results/richardson_polar_domC/polar_dx0.0020.json. El de 10 solo lo tiene en
results/polar_2grados_dom24x16, donde quedo truncado en 7950 de 26000
iteraciones por el guardian de residual, asi que ese punto no es comparable a
t=20 y hay que decirlo al leer la tabla.

Reanudable: cada punto se guarda segun sale y una segunda invocacion salta los
que ya estan.

Uso:
    .venv/bin/python scripts/agent_tests/polar_optimizada.py
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
import verificacion_numerica as vn
import richardson_polar_domC as rp

OUT = os.path.join(ROOT, "results", "polar_optimizada")
SERIES = os.path.join(OUT, "series")
CAMPOS = os.path.join(OUT, "campos")
for d in (OUT, SERIES, CAMPOS):
    os.makedirs(d, exist_ok=True)

DX = 0.002
ITERS = rp.ITERS[DX]          # 26000, t~20, el mismo presupuesto del estudio
ANGULOS = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)

# Optimizaciones y presupuesto de proyeccion. Las tres primeras se validaron a
# t=20 contra el control: fast_masks es bit a bit identico, interp_float32 mueve
# Cl un +0.012%, y warm_start + coarse_mask_majority dejan la divergencia por
# debajo de la de produccion.
FLAGS = ("warm_start", "coarse_mask_majority", "fast_masks", "interp_float32")
PRESUPUESTO = {"mg_max_outer": 2, "mg_cycles_per_outer": 3}

JSON = os.path.join(OUT, f"polar_dx{DX:.4f}.json")


def carga():
    if os.path.exists(JSON):
        with open(JSON) as f:
            return json.load(f)
    return {}


def guarda(d):
    with open(JSON, "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def main():
    opt_solver.reset()
    opt_solver.enable(*FLAGS)
    # Comprobar en vez de confiar: si una opcion no quedara activa, la polar
    # saldria con el solver antiguo y el resultado pareceria valido.
    assert opt_solver.active() == sorted(FLAGS), opt_solver.active()
    vn._log(f"optimizaciones activas: {opt_solver.active()}")
    vn._log(f"presupuesto de proyeccion: {PRESUPUESTO}")
    vn._log(f"perfil: {rp.PERFIL}")
    vn._log(f"dominio: {rp.DOMINIO}  dx={DX}  iters={ITERS}  Re={rp.RE:.0e}")

    datos = carga()
    for a in ANGULOS:
        ka = f"{a:.1f}"
        if ka in datos:
            vn._log(f"alpha={a}: ya hecho, se salta")
            continue
        vn._log(f"--- alpha={a:.1f}º ({ITERS} iters) ---")
        t0 = time.time()
        r = vn.simular(
            rp.PERFIL, rp.RE, DX, alpha=a, iters=ITERS,
            extra={**rp.DOMINIO, **rp.SIN_PARADAS, **PRESUPUESTO},
            dump=os.path.join(SERIES, f"dx{DX:.4f}_a{a:04.1f}_series.npz"),
            dump_field=os.path.join(CAMPOS, f"dx{DX:.4f}_a{a:04.1f}_campo.npz"),
        )
        if r is None:
            vn._log(f"alpha={a}: simulacion fallida")
            continue
        r.update(rp.DOMINIO)
        r["optimizaciones"] = list(FLAGS)
        r["presupuesto"] = dict(PRESUPUESTO)
        datos[ka] = r
        guarda(datos)
        vn._log(f"alpha={a}: Cl={r['cl']:.5f} Cd={r['cd']:.5f} L/D={r['ld']:.3f} "
                f"({time.time()-t0:.0f}s)")

    vn._log(f"polar completa: {len(datos)} puntos en {JSON}")


if __name__ == "__main__":
    main()

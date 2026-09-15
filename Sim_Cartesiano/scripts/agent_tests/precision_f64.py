"""Corre el mismo caso en float32 y en float64 y compara Cl, Cd y divergencia.

Todo el solver esta escrito en float32 (418 apariciones y trece kernels CUDA con
`float`), asi que la version de doble precision se genera reescribiendo el
codigo fuente a una copia temporal en vez de tocar `Simulador2D.py`: el estudio
publicado depende de que ese fichero no se mueva. La reescritura es textual y
acotada:

  - dentro de los bloques r''' de CUDA: `float` -> `double`, los intrinsecos de
    simple precision (sqrtf, fabsf, floorf...) a su version doble y el sufijo
    `f` de los literales fuera;
  - en los `in_params`/`out_params` de los ElementwiseKernel: float32 -> float64;
  - en el resto del modulo, solo `cp.float32` -> `cp.float64`. Los `np.float32`
    se dejan como estan: son volcados a CPU (memoria compartida con la GUI,
    frames) con aritmetica de tamanos en bytes que si depende del tipo.

Se reescriben `Simulador2D.py` y `opt_solver.py`, que tiene su propio kernel de
relajacion por lineas. Cada precision corre en su proceso, como en
`equivalencia_bit.py`: main() deja estado global (pools de CuPy, memoria
compartida) y compartirlo entre dos modulos que se llaman igual da falsos
resultados.

Uso:
    .venv/bin/python scripts/agent_tests/precision_f64.py [--dx 0.004]
        [--alpha 4] [--iters 4000] [--salida results/precision_f64]
    .venv/bin/python scripts/agent_tests/precision_f64.py --corre f64 out.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

MODULOS = ("Simulador2D.py", "opt_solver.py")

# intrinsecos de simple precision -> doble (el nombre doble es el mismo sin la f)
INTRINSECOS = ("sqrtf", "rsqrtf", "fabsf", "floorf", "ceilf", "fmaxf", "fminf",
               "expf", "logf", "log10f", "powf", "sinf", "cosf", "tanf",
               "tanhf", "atanf", "atan2f", "asinf", "acosf", "hypotf",
               "copysignf", "fmodf", "roundf", "truncf", "erff", "fdimf")

CASO = {
    "v0x": 1.0, "v0y": 0.0, "chord": 1.0, "rho": 1.0, "nu": 1e-5,
    "CFL": 0.5,
    "filepath": os.path.join(ROOT, "profiles/NACA_0012"),
    "min_te_height_factor": 2.0,
    "Lx": 24.0, "Ly": 16.0, "cx": 6.0,
    "ancho_zona_fina_x": 1.5, "ancho_zona_fina_y": 1.0, "factor_expansion": 1.1,
    "turb_model": "sa", "wall_treatment": "consistent",
    "advection_scheme": "maccormack", "wake_refinement_mode": "long_fine_x",
    "transition_model": "sa_bc", "freestream_Tu": 0.1,
    "mg_niveles_max": 2, "mg_max_outer": 2, "mg_cycles_per_outer": 3,
    "divergencia": 0.02,
    # presupuesto fijo en las dos corridas: si una parara antes que la otra los
    # numeros no serian comparables
    "stop_on_convergence": False, "stop_on_clcd_convergence": False,
    "guardado": 25, "graficos": False, "live_view": False, "mostrar_malla": False,
    "save_frames": False, "shm_publish": False,
}


def reescribir_cuda(src: str) -> str:
    for f in INTRINSECOS:
        src = re.sub(rf"\b{f}\b", f[:-1], src)
    src = re.sub(r"\bfloat\b", "double", src)
    # sufijo de literal: 1.0f, 0.125f, 1e-30f -> sin la f
    src = re.sub(r"(?<![\w.])((?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)f(?![\w])",
                 r"\1", src)
    return src


def reescribir_modulo(texto: str) -> tuple[str, int]:
    n_cuda = 0

    def _bloque(m):
        nonlocal n_cuda
        b = m.group(0)
        if "__global__" not in b and not re.search(r"\bfloat\b", b):
            return b
        n_cuda += 1
        return reescribir_cuda(b)

    texto = re.sub(r"r'''(?:.|\n)*?'''", _bloque, texto)
    texto = re.sub(r"(in_params=|out_params=)('[^']*')",
                   lambda m: m.group(1) + m.group(2).replace("float32", "float64"),
                   texto)
    texto = texto.replace("cp.float32", "cp.float64")
    return texto, n_cuda


def preparar_f64(destino: str) -> str:
    os.makedirs(destino, exist_ok=True)
    for nombre in MODULOS:
        texto, n = reescribir_modulo(open(os.path.join(ROOT, nombre)).read())
        with open(os.path.join(destino, nombre), "w") as fh:
            fh.write(texto)
        assert "cp.float32" not in texto, nombre
        print(f"[f64] {nombre}: {n} bloques CUDA reescritos")
    return destino


def corre(precision: str, salida: str, dx: float, alpha: float, iters: int,
          extra: dict | None = None):
    if precision == "f64":
        caja = preparar_f64(os.path.join(tempfile.gettempdir(), "sim2d_f64"))
        sys.path.insert(0, caja)
    import cupy as cp
    import Simulador2D as sim

    assert (sim.__file__.startswith(caja) if precision == "f64"
            else sim.__file__.startswith(ROOT)), sim.__file__

    box = tempfile.mkdtemp(prefix=f"prec_{precision}_")
    prev = os.getcwd()
    os.chdir(box)
    t0 = time.time()
    try:
        mesh = sim.main(**{**CASO, "dx_min": dx, "alpha_deg": alpha,
                           "iteraciones": iters, **(extra or {})})
    finally:
        os.chdir(prev)
    wall = time.time() - t0

    def arr(nombre):
        return cp.asnumpy(getattr(mesh, nombre)).astype(float).tolist()

    datos = {
        "precision": precision,
        "dtype_u": str(mesh.u.dtype),
        "dx_min": dx, "alpha_deg": alpha, "iteraciones": iters,
        "nx": int(mesh.nx), "ny": int(mesh.ny),
        "opt_solver_off": os.environ.get("OPT_SOLVER_OFF", ""),
        "wall_s": wall,
        "t": arr("tvector"), "cl": arr("clvector"), "cd": arr("cdvector"),
        "div": arr("divvector"), "div_max": arr("divvector_max"),
    }
    with open(salida, "w") as fh:
        json.dump(datos, fh)
    print(f"[{precision}] u.dtype={datos['dtype_u']}  malla {datos['nx']}x{datos['ny']}"
          f"  {wall/60:.1f} min")


def ventana(d, frac=0.25):
    """Media sobre el ultimo `frac` de muestras con t > 0 (descarta el relleno)."""
    t = np.array(d["t"])
    n = int(np.sum(t > 0))
    i0 = n - max(int(n * frac), 1)
    sl = slice(i0, n)
    return {k: float(np.mean(np.array(d[k])[sl])) for k in ("cl", "cd", "div", "div_max")} | {
        "t_ini": float(t[i0]), "t_fin": float(t[n - 1]), "n": n - i0}


def compara(a, b):
    A, B = ventana(a), ventana(b)
    print(f"\nmalla {a['nx']}x{a['ny']}   dx={a['dx_min']}   alpha={a['alpha_deg']}"
          f"   {a['iteraciones']} iteraciones")
    print(f"media sobre t = {A['t_ini']:.2f}-{A['t_fin']:.2f} ({A['n']} muestras)\n")
    print(f"{'magnitud':<12}{'float32':>14}{'float64':>14}{'dif':>14}{'dif rel':>11}")
    filas = []
    for k, etiqueta in (("cl", "Cl"), ("cd", "Cd"), ("div", "div media"),
                        ("div_max", "div max")):
        x, y = A[k], B[k]
        rel = (y - x) / abs(x) if x else float("nan")
        print(f"{etiqueta:<12}{x:>14.6f}{y:>14.6f}{y-x:>14.3e}{rel*100:>10.2f}%")
        filas.append({"magnitud": k, "f32": x, "f64": y, "dif": y - x, "dif_rel": rel})
    ld32, ld64 = A["cl"] / A["cd"], B["cl"] / B["cd"]
    print(f"{'L/D':<12}{ld32:>14.6f}{ld64:>14.6f}{ld64-ld32:>14.3e}"
          f"{(ld64-ld32)/ld32*100:>10.2f}%")
    print(f"\ncoste: f32 {a['wall_s']/60:.1f} min   f64 {b['wall_s']/60:.1f} min"
          f"   (x{b['wall_s']/a['wall_s']:.2f})")
    return {"ventana_f32": A, "ventana_f64": B, "filas": filas,
            "ld_f32": ld32, "ld_f64": ld64,
            "wall_f32_s": a["wall_s"], "wall_f64_s": b["wall_s"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.004)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--salida", default=os.path.join(ROOT, "results/precision_f64"))
    ap.add_argument("--corre", nargs=2, metavar=("PRECISION", "SALIDA"))
    a = ap.parse_args()

    if a.corre:
        corre(a.corre[0], a.corre[1], a.dx, a.alpha, a.iters)
        sys.exit(0)

    os.makedirs(a.salida, exist_ok=True)
    jsons = {}
    for prec in ("f32", "f64"):
        jsons[prec] = os.path.join(a.salida, f"{prec}.json")
        r = subprocess.run([sys.executable, __file__, "--corre", prec, jsons[prec],
                            "--dx", str(a.dx), "--alpha", str(a.alpha),
                            "--iters", str(a.iters)], cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"fallo la corrida {prec}")

    d32 = json.load(open(jsons["f32"]))
    d64 = json.load(open(jsons["f64"]))
    resumen = compara(d32, d64)
    with open(os.path.join(a.salida, "comparacion.json"), "w") as fh:
        json.dump({"caso": {**CASO, "dx_min": a.dx, "alpha_deg": a.alpha,
                            "iteraciones": a.iters}, **resumen}, fh, indent=2)

"""
Harness A/B para optimizar el solver de presion sin perder fiabilidad.

Config identica a la de produccion (richardson_polar_domC): perfil ganador de
islands_tfg2, dominio C (Lx=24, Ly=16, cx=6), alpha=4, Re=1e5, CFL=0.5, con las
dos paradas DESACTIVADAS para que todas las variantes recorran exactamente las
mismas iteraciones. Comparar variantes que paran en iteraciones distintas no
mide el solver, mide la parada.

Se llama a Simulador2D.main() directamente, con los mismos sim_params que arma
RunGA.simular_perfil, porque simular_perfil destruye la malla al salir y con
ella los vectores de ciclos multigrid y divergencia, que son justo el
diagnostico que se quiere. La media de Cl/Cd reproduce su ventana (ultimo 20%,
minimo 10 muestras) para que los numeros sean comparables con los de campana.

Se mide en dos ejes a la vez, y una variante solo se acepta si gana en el
primero sin perder en el segundo:

  VELOCIDAD    it/s, s/paso, ciclos MG por paso, ocupacion de GPU.
  FIABILIDAD   Cl y Cd de la ventana final frente al baseline (puerta: 0.5%),
               trayectoria Cl(t)/Cd(t) completa, y divergencia media/maxima
               tras proyeccion (no puede empeorar).

La 0.5% es la sensibilidad de ventana temporal ya medida en este proyecto
(results/asintotico_alpha4_domC: L/D de cola 34.038 en t=20 vs 34.049 en t=41.5),
y esta un orden por debajo de la banda entre mallas del GCI (Cd 6.4%, L/D 7.6%).

Todo corre con cwd en un sandbox propio: main() escribe sim_last.npz y
polar_results.json con rutas relativas, y no deben pisar los del repo.

Uso:
    .venv/bin/python scripts/agent_tests/bench_opt.py --run baseline
    .venv/bin/python scripts/agent_tests/bench_opt.py --run o4 --max-outer 4
    .venv/bin/python scripts/agent_tests/bench_opt.py --run warm --opt warm_start
    .venv/bin/python scripts/agent_tests/bench_opt.py --sweep        # fase 1 completa
    .venv/bin/python scripts/agent_tests/bench_opt.py --compare
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import os
import re
import subprocess
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")

import cupy as cp

import Simulador2D
import opt_solver

OUT = os.path.join(ROOT, "results", "bench_opt")
SERIES = os.path.join(OUT, "series")
SANDBOX = os.path.join(OUT, "_sandbox")
for d in (OUT, SERIES, SANDBOX):
    os.makedirs(d, exist_ok=True)

# Absoluta: main() se ejecuta con cwd en el sandbox.
PERFIL = os.path.join(
    ROOT, "results/convergence_study/islands_tfg2/NACA_0012_sharp/epoch10/"
          "NACA_0012_sharp_winner_Re100000_a4.0_LD27.75.dat")

DX = 0.002
ALPHA = 4.0
RE = 1e5
ITERS = 4000
GUARDADO = 25          # denso: 160 muestras de Cl/Cd para comparar trayectorias
TOL_REL = 0.005        # puerta de aceptacion sobre Cl y Cd

# Config de referencia validada, la misma de RunGA.CONFIG + dominio C + SA-BC.
BASE = {
    "v0x": 1.0, "v0y": 0.0, "chord": 1.0, "rho": 1.0, "nu": 1.0 / RE,
    "CFL": 0.5, "alpha_deg": ALPHA, "dx_min": DX,
    "Lx": 24.0, "Ly": 16.0, "cx": 6.0,
    "ancho_zona_fina_x": 1.5, "ancho_zona_fina_y": 1.0, "factor_expansion": 1.1,
    "turb_model": "sa", "wall_treatment": "consistent",
    "advection_scheme": "maccormack", "min_te_height_factor": 1.0,
    "wake_refinement_mode": "long_fine_x",
    "transition_model": "sa_bc", "freestream_Tu": 0.1,
    "mg_niveles_max": 2, "mg_max_outer": 8, "divergencia": 0.02,
    "stop_on_clcd_convergence": False, "stop_on_convergence": False,
    "guardado": GUARDADO,
    "graficos": False, "live_view": False, "mostrar_malla": False,
    "save_frames": False,
}


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------
# Muestreo de ocupacion de GPU en paralelo a la simulacion
# ----------------------------------------------------------------------
class GpuSampler(threading.Thread):
    """nvidia-smi en un hilo aparte. La ocupacion distingue un solver limitado
    por computo (alta) de uno limitado por lanzamiento de kernels (baja), que es
    lo que decide si el margen esta en el algoritmo o en la ingenieria."""

    def __init__(self, period=1.0):
        super().__init__(daemon=True)
        self.period = period
        self.samples = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.wait(self.period):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                self.samples.append(float(out.stdout.strip().splitlines()[0]))
            except Exception:
                pass

    def stop(self):
        self._stop.set()
        self.join(timeout=5)
        return self.samples


# ----------------------------------------------------------------------
# Ejecucion de una variante
# ----------------------------------------------------------------------
def _np(x):
    return cp.asnumpy(x) if isinstance(x, cp.ndarray) else np.asarray(x)


class FactorMG:
    """Mide el factor de convergencia del multigrid: ||r|| despues de un V-cycle
    dividido por ||r|| antes, sobre el residuo del Poisson en el nivel fino.

    Es distinto de la reduccion de divergencia por outer, y la diferencia es la
    que decide donde esta el problema. La divergencia se mide despues del
    post-procesado IBM (ghost-cell + impermeabilidad), que reintroduce
    divergencia en la interfaz solido-fluido: si el multigrid resolviera bien y
    el IBM le deshiciera el trabajo, la divergencia apenas bajaria aunque el
    solver fuese perfecto. El factor por V-cycle no ve el IBM.

    Referencia: un multigrid sano da 0.1-0.2 por V-cycle (un orden de magnitud
    cada uno o dos ciclos). Por encima de ~0.7 el multigrid no esta actuando
    como multigrid y los niveles gruesos son coste sin retorno.

    Se obtiene activando verbose en pasos sueltos y leyendo las lineas [DBG] que
    el propio solver ya imprime, sin tocar el simulador.
    """

    PAT = re.compile(r"\|\|r\|\| (inicial|tras V-cycle COMPLETO)=([0-9.eE+-]+)")

    def __init__(self, cada=200):
        self.cada = cada
        self.factores = []
        self.n = 0
        self._orig = None

    def __enter__(self):
        self._orig = Simulador2D.Mesh.project_multigrid
        orig = self._orig

        def wrapped(mesh, *a, **k):
            self.n += 1
            if self.n % self.cada:
                return orig(mesh, *a, **k)
            buf = io.StringIO()
            k = {**k, "verbose": True}
            with contextlib.redirect_stdout(buf):
                info = orig(mesh, *a, **k)
            self._parse(buf.getvalue())
            return info

        Simulador2D.Mesh.project_multigrid = wrapped
        return self

    def __exit__(self, *exc):
        Simulador2D.Mesh.project_multigrid = self._orig
        return False

    def _parse(self, txt):
        # El solver imprime, para los dos primeros V-cycles del outer 0:
        #   ||r|| inicial=...      (antes del ciclo)
        #   ||r|| tras V-cycle COMPLETO=...   (despues)
        vals = self.PAT.findall(txt)
        ini = None
        for kind, v in vals:
            v = float(v)
            if kind == "inicial":
                ini = v
            elif ini is not None and ini > 0:
                self.factores.append(v / ini)
                ini = v      # el fin de un ciclo es el inicio del siguiente

    def resumen(self):
        if not self.factores:
            return {}
        f = np.array(self.factores, float)
        return {
            "mg_factor_medio": round(float(f.mean()), 4),
            "mg_factor_mediana": round(float(np.median(f)), 4),
            "mg_factor_p95": round(float(np.percentile(f, 95)), 4),
            "mg_factor_n": int(f.size),
            # Ciclos que harian falta para bajar el residuo un orden de magnitud.
            "mg_ciclos_por_decada": round(float(np.log(0.1) / np.log(np.median(f))), 1)
            if 0 < np.median(f) < 1 else None,
        }


class Instrumento:
    """Envuelve project_multigrid para registrar, paso a paso, cuantos outers
    gasta y si alcanza la tolerancia.

    Es la medida que decide la fase 1: si el solver converge y le sobra
    presupuesto, recortar max_outer es gratis; si lo agota sin converger,
    recortarlo es perder precision y hay que atacar la convergencia del
    multigrid en vez de la configuracion.

    Monkeypatch del harness, no una modificacion del simulador: se instala al
    empezar la variante y se retira al acabarla.
    """

    def __init__(self):
        self.outers = []
        self.converged = []
        self.div_before = []
        self.div_after = []
        self._orig = None

    def __enter__(self):
        self._orig = Simulador2D.Mesh.project_multigrid
        orig = self._orig

        def wrapped(mesh, *a, **k):
            info = orig(mesh, *a, **k)
            self.outers.append(info.get("outers"))
            self.converged.append(bool(info.get("converged")))
            self.div_before.append(info.get("div_before"))
            self.div_after.append(info.get("div_after"))
            return info

        Simulador2D.Mesh.project_multigrid = wrapped
        return self

    def __exit__(self, *exc):
        Simulador2D.Mesh.project_multigrid = self._orig
        return False

    def resumen(self):
        if not self.outers:
            return {}
        o = np.array([x for x in self.outers if x is not None], float)
        c = np.array(self.converged, float)
        da = np.array([x for x in self.div_after if x is not None], float)
        db = np.array([x for x in self.div_before if x is not None], float)
        return {
            "outers_media": round(float(o.mean()), 2),
            "outers_p95": round(float(np.percentile(o, 95)), 2),
            "outers_max": int(o.max()),
            "frac_converge": round(float(c.mean()), 3),
            "div_before_media": float(f"{db.mean():.4g}") if db.size else None,
            "div_after_media": float(f"{da.mean():.4g}") if da.size else None,
            "reduccion_div": round(float(db.mean() / da.mean()), 2)
            if da.size and da.mean() > 0 else None,
        }


def run_variant(name, opts=(), iters=ITERS, factor=False, **overrides):
    """Corre una variante y vuelca metricas + series a results/bench_opt/.

    factor=True anade la medida del factor de convergencia del multigrid. Cuesta
    tiempo (activa verbose en pasos sueltos, que recalcula la divergencia por
    outer), asi que falsea el it/s: usarlo para diagnosticar, no para cronometrar.
    """
    opt_solver.reset()
    if opts:
        opt_solver.enable(*opts)

    params = {**BASE, "filepath": PERFIL, "iteraciones": iters, **overrides}

    _log(f"--- {name}  opts={list(opts) or '-'}  overrides={overrides or '-'}")

    sampler = GpuSampler()
    prev_cwd = os.getcwd()
    mesh = None
    os.chdir(SANDBOX)
    sampler.start()
    t0 = time.time()
    try:
        with contextlib.ExitStack() as stack:
            # FactorMG envuelve por fuera para que Instrumento siga viendo la
            # llamada real y su dict de retorno.
            fmg = stack.enter_context(FactorMG()) if factor else None
            instr = stack.enter_context(Instrumento())
            mesh = Simulador2D.main(**params)
        wall = time.time() - t0
        util = sampler.stop()
        os.chdir(prev_cwd)
        diag = {**instr.resumen(), **(fmg.resumen() if fmg else {})}
        if factor:
            diag["cronometraje_valido"] = False
        rec = _collect(name, mesh, wall, util, opts, overrides, params, diag)
    finally:
        sampler.stop()
        os.chdir(prev_cwd)
        opt_solver.reset()
        del mesh
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    return rec


def _collect(name, mesh, wall, util, opts, overrides, params, diag=None):
    cl = _np(mesh.clvector).astype(float)
    cd = _np(mesh.cdvector).astype(float)
    tv = _np(mesh.tvector).astype(float)
    div = _np(mesh.divvector).astype(float)
    div_max = _np(mesh.divvector_max).astype(float)
    cyc = _np(mesh.mg_cycles_vector).astype(float)

    # Misma ventana que RunGA.simular_perfil: ultimo 20%, minimo 10 muestras.
    n = len(cd)
    w = min(n, max(n // 5, 10))
    cl_val, cd_val = float(cl[-w:].mean()), float(cd[-w:].mean())
    ld = cl_val / cd_val if abs(cd_val) > 1e-6 else 0.0
    # El intervalo de muestreo real, no la constante del modulo: una variante
    # puede pasar `guardado` como override y entonces el numero de iteraciones
    # sale mal por el factor entre ambos.
    g = int(params.get("guardado", GUARDADO))
    n_it = int(n * g)

    rec = {
        "name": name,
        "opts": list(opts),
        "overrides": dict(overrides),
        "dx": params["dx_min"],
        "iters": n_it,
        "wall_s": round(wall, 1),
        "it_s": round(n_it / wall, 3) if wall > 0 else None,
        "s_paso": round(wall / max(n_it, 1), 5),
        "gpu_util_media": round(float(np.mean(util)), 1) if util else None,
        "gpu_util_p95": round(float(np.percentile(util, 95)), 1) if util else None,
        "gpu_util_n": len(util),
        "cl": round(cl_val, 6),
        "cd": round(cd_val, 6),
        "ld": round(ld, 4),
        "cl_std": round(float(cl[-w:].std()), 6),
        "cd_std": round(float(cd[-w:].std()), 6),
        "n_samples": int(w),
        "mg_cycles_media": round(float(cyc.mean()), 2) if cyc.size else None,
        "mg_cycles_p95": round(float(np.percentile(cyc, 95)), 2) if cyc.size else None,
        "mg_cycles_total": int(cyc.sum()) if cyc.size else None,
        "div_mean": float(f"{div[-w:].mean():.6g}"),
        "div_max": float(f"{div_max[-w:].max():.6g}"),
    }
    rec.update(diag or {})

    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.join(SERIES, f"{name}.npz"),
                        t=tv, cl=cl, cd=cd, div=div, div_max=div_max, cycles=cyc)

    _log(f"{name}: {rec['it_s']} it/s  gpu={rec['gpu_util_media']}%  "
         f"cyc={rec['mg_cycles_media']}  outers={rec.get('outers_media')}  "
         f"conv={rec.get('frac_converge')}  Cl={rec['cl']}  Cd={rec['cd']}  "
         f"div={rec['div_mean']:.3g}  ({rec['wall_s']}s)")
    return rec


# ----------------------------------------------------------------------
# Comparacion
# ----------------------------------------------------------------------
def load(name):
    p = os.path.join(OUT, f"{name}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _traj_dev(name, base="baseline"):
    """Maxima desviacion relativa de la trayectoria Cl(t) frente al baseline.
    Mas sensible que la media de la ventana: detecta variantes que acaban en el
    mismo sitio recorriendo un camino distinto."""
    pa = os.path.join(SERIES, f"{name}.npz")
    pb = os.path.join(SERIES, f"{base}.npz")
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return None
    a, b = np.load(pa), np.load(pb)
    n = min(len(a["cl"]), len(b["cl"]))
    if n < 4:
        return None
    ref = np.abs(b["cl"][:n]).max()
    if ref < 1e-12:
        return None
    return float(np.abs(a["cl"][:n] - b["cl"][:n]).max() / ref)


def compare(names, base="baseline"):
    b = load(base)
    if b is None:
        _log(f"falta el baseline ({base}.json); correr primero --run {base}")
        return

    def rel(v, r):
        if v is None or not r:
            return None
        return (v - r) / abs(r)

    avisos = []
    hdr = (f"{'variante':<12}{'it/s':>7}{'x':>6}{'gpu%':>6}{'cyc':>7}{'out':>6}"
           f"{'conv':>7}{'Cl':>9}{'dCl':>8}{'Cd':>9}{'dCd':>8}{'L/D':>8}"
           f"{'div':>10}{'traj':>8}{'':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for n in names:
        r = load(n)
        if r is None:
            print(f"{n:<12}  (sin datos)")
            continue
        dcl, dcd = rel(r["cl"], b["cl"]), rel(r["cd"], b["cd"])
        speed = r["it_s"] / b["it_s"] if b["it_s"] else 0.0
        traj = _traj_dev(n, base)
        peor_div = r["div_mean"] > b["div_mean"] * 1.05
        ok = (abs(dcl) <= TOL_REL and abs(dcd) <= TOL_REL and not peor_div)
        # La puerta del 0.5% supone que el baseline es la referencia correcta, y
        # eso solo vale para variantes que degradan. Una variante que deja MENOS
        # divergencia que el baseline esta mejor convergida que el, no peor: su
        # diferencia en Cl mide el error del baseline, no el suyo. En la curva de
        # presupuesto menos convergencia da Cl mas bajo de forma monotona, asi
        # que un Cl mas alto con menos divergencia va en el sentido correcto.
        # Marcarla FUERA seria descartar precisamente lo que se busca.
        mejor_conv = r["div_mean"] < b["div_mean"] * 0.98
        # Comparar variantes que han recorrido tiempos fisicos distintos no mide
        # el solver, mide el transitorio: Cl y Cd todavia estan cambiando. Se
        # marca aparte para que la fila no se lea como un veredicto.
        if r["iters"] != b["iters"]:
            marca = "n/c"
            avisos.append(f"{n}: {r['iters']} iters frente a {b['iters']} del baseline")
        elif r.get("cronometraje_valido") is False:
            marca = "s/t"
            avisos.append(f"{n}: lleva medida del factor MG, el it/s no vale")
        elif n == base:
            marca = "-"
        elif mejor_conv:
            marca = "+conv"
        else:
            marca = "OK" if ok else "FUERA"
        print(f"{n:<12}{r['it_s']:>7.2f}{speed:>6.2f}"
              f"{(r['gpu_util_media'] or 0):>6.1f}{(r['mg_cycles_media'] or 0):>7.1f}"
              f"{(r.get('outers_media') or 0):>6.1f}{(r.get('frac_converge') or 0):>7.2f}"
              f"{r['cl']:>9.5f}{100*dcl:>7.2f}%{r['cd']:>9.5f}{100*dcd:>7.2f}%"
              f"{r['ld']:>8.3f}{r['div_mean']:>10.3g}"
              f"{(100*traj if traj is not None else float('nan')):>7.2f}%{marca:>7}")
    print(f"\npuerta: |dCl| y |dCd| <= {100*TOL_REL:.1f}% y div no peor de 5%   "
          f"(baseline: {base}, {b['iters']} iters, dx={b['dx']})")
    print("out  = outers gastados por paso;  conv = fraccion de pasos que alcanzan tol_div")
    print("traj = max|Cl(t)-Cl_base(t)| / max|Cl_base|, sobre la serie completa")
    print("+conv = deja MENOS divergencia que el baseline: esta mejor convergida que")
    print("        el, asi que su dCl/dCd mide el error del baseline, no el suyo")
    if avisos:
        print("\nno comparables (n/c) o sin cronometraje fiable (s/t):")
        for a in avisos:
            print(f"  {a}")

    # Veredicto: solo lo que pasa la puerta, ordenado por aceleracion. Una
    # variante mas rapida que se sale de la puerta no es un resultado parcial,
    # es un resultado descartado, y conviene que se lea asi.
    aptos = []
    for n in names:
        r = load(n)
        if r is None or n == base or r["iters"] != b["iters"]:
            continue
        if r.get("cronometraje_valido") is False:
            continue
        dcl, dcd = rel(r["cl"], b["cl"]), rel(r["cd"], b["cd"])
        if dcl is None or dcd is None:
            continue
        if abs(dcl) <= TOL_REL and abs(dcd) <= TOL_REL \
           and r["div_mean"] <= b["div_mean"] * 1.05:
            aptos.append((r["it_s"] / b["it_s"], n, r, dcl, dcd))
    aptos.sort(reverse=True)
    print("\n" + "=" * 60)
    print("PASAN LA PUERTA, ordenados por aceleracion")
    print("=" * 60)
    if not aptos:
        print("  ninguna variante pasa: |dCl| o |dCd| > "
              f"{100*TOL_REL:.1f}% o la divergencia empeora")
    else:
        for sp, n, r, dcl, dcd in aptos:
            print(f"  {sp:5.2f}x  {n:<18} dCl={100*dcl:+.2f}%  dCd={100*dcd:+.2f}%  "
                  f"div={r['div_mean']:.3g}  opts={','.join(r['opts']) or '-'}")
            if r["overrides"]:
                print(f"{'':11}{'':<18} {r['overrides']}")
    print()


# ----------------------------------------------------------------------
# Fase 1: barrido de configuracion, sin tocar una linea del simulador
# ----------------------------------------------------------------------
# Ordenado por valor informativo, no por comodidad: el barrido guarda cada
# variante segun acaba y --sweep salta las ya hechas, asi que se puede leer el
# resultado parcial y cortar en cuanto la tendencia sea clara.
#
# La instrumentacion ya dice que el solver agota los 8 outers sin converger
# NUNCA (frac_converge=0.0, reduccion de divergencia x1.75 en 40 V-cycles), asi
# que recortar presupuesto no es gratis: cada outer que se quita es precision
# que se pierde. Lo que mide este barrido es cuanto de esa precision se traslada
# de verdad a Cl y Cd, que es lo que decide si la proyeccion esta
# sobre-especificada para el resultado que se publica.
SWEEP = [
    ("baseline",  {}),
    ("o4",        {"mg_max_outer": 4}),
    ("o2",        {"mg_max_outer": 2}),
    ("c3",        {"mg_cycles_per_outer": 3}),
    ("div10",     {"divergencia": 0.10}),
    ("pre2post2", {"mg_pre_suavizado": 2, "mg_post_suavizado": 2}),
    ("o6",        {"mg_max_outer": 6}),
    ("div05",     {"divergencia": 0.05}),
    ("c7",        {"mg_cycles_per_outer": 7}),
    ("o4div05",   {"mg_max_outer": 4, "divergencia": 0.05}),
]


# ----------------------------------------------------------------------
# Fases 2-4: las optimizaciones de opt_solver
# ----------------------------------------------------------------------
# Cada optimizacion se prueba dos veces, y las dos medidas responden preguntas
# distintas:
#
#   A PRESUPUESTO IGUAL (mismos max_outer/cycles que el baseline) se mide cuanta
#   precision gana. No se espera que vaya mas rapido; se espera que la
#   divergencia baje. Es la prueba de que la optimizacion hace lo que dice.
#
#   A PRESUPUESTO RECORTADO se mide la velocidad. Si con la mitad de ciclos
#   iguala la divergencia del baseline, esa es la aceleracion real. Convertir
#   convergencia en tiempo exige recortar el presupuesto: si no, un solver mejor
#   solo produce numeros mas exactos al mismo precio.
QUEUE = [
    # 1. Control. Con todas las opciones apagadas debe reproducir el baseline
    # exactamente. Si esto no sale igual, el cableado no es neutro y el resto de
    # la tabla no significa nada.
    ("equiv",        [], {}),

    # 2. Curva de presupuesto con el suavizador actual. Es el candidato de mayor
    # valor y riesgo cero: la instrumentacion dice que el solver nunca alcanza la
    # tolerancia (conv=0.0) y por tanto gasta SIEMPRE los 8 outers. Si el suelo de
    # divergencia no lo fija el numero de iteraciones sino otra cosa, la mayor
    # parte de esos outers no compra nada y se pueden quitar sin tocar codigo.
    # Esta curva es ademas el control honesto de todo lo que viene despues: sin
    # ella no se distingue "el suavizador nuevo ayuda" de "bastaba con gastar
    # menos".
    ("punto_o4c3",   [], {"mg_max_outer": 4, "mg_cycles_per_outer": 3}),
    ("punto_o2c3",   [], {"mg_max_outer": 2, "mg_cycles_per_outer": 3}),
    ("punto_o2c2",   [], {"mg_max_outer": 2, "mg_cycles_per_outer": 2,
                          "mg_pre_suavizado": 1, "mg_post_suavizado": 1}),
    ("punto_o1c2",   [], {"mg_max_outer": 1, "mg_cycles_per_outer": 2,
                          "mg_pre_suavizado": 1, "mg_post_suavizado": 1}),

    # 3. Semilla de presion entre pasos. No cambia la ecuacion ni la tolerancia,
    # solo el punto de partida del iterativo.
    ("warm",         ["warm_start"], {}),
    ("warm_o4c3",    ["warm_start"], {"mg_max_outer": 4, "mg_cycles_per_outer": 3}),

    # 4. Suavizador por lineas, a presupuesto recortado, que es donde su mejor
    # convergencia se puede convertir en tiempo. Cuesta 3.2x por barrido con
    # pre=post=1 y deja la divergencia 2.1x mas baja: para ganar hay que gastar
    # ese margen en ciclos.
    ("lines_o2c2",   ["line_smoother"],
                     {"mg_max_outer": 2, "mg_cycles_per_outer": 2,
                      "mg_pre_suavizado": 1, "mg_post_suavizado": 1}),
    ("lines_o1c2",   ["line_smoother"],
                     {"mg_max_outer": 1, "mg_cycles_per_outer": 2,
                      "mg_pre_suavizado": 1, "mg_post_suavizado": 1}),

    # Solo lineas Y: un tercio del coste del barrido completo. Cubre la estela,
    # que es la zona anisotropa mas extensa, pero no la banda sobre el perfil.
    ("linesy_o2c2",  ["line_smoother_y"],
                     {"mg_max_outer": 2, "mg_cycles_per_outer": 2,
                      "mg_pre_suavizado": 1, "mg_post_suavizado": 1}),
    ("linesy_o4c3",  ["line_smoother_y"],
                     {"mg_max_outer": 4, "mg_cycles_per_outer": 3,
                      "mg_pre_suavizado": 1, "mg_post_suavizado": 1}),

    # 5. Mascara del solido en los niveles gruesos, a presupuesto completo para
    # aislar su efecto sobre la precision.
    ("maskmaj",      ["coarse_mask_majority"], {}),
    ("masknone",     ["coarse_mask_none"], {}),

    # 6. Combinaciones de lo que sobreviva por separado.
    ("todo_o2c2",    ["line_smoother", "coarse_mask_majority", "warm_start"],
                     {"mg_max_outer": 2, "mg_cycles_per_outer": 2,
                      "mg_pre_suavizado": 1, "mg_post_suavizado": 1}),
    ("todoy_o4c3",   ["line_smoother_y", "coarse_mask_majority", "warm_start"],
                     {"mg_max_outer": 4, "mg_cycles_per_outer": 3,
                      "mg_pre_suavizado": 1, "mg_post_suavizado": 1}),
]



def pareto(base="baseline"):
    """Frontera precision-velocidad.

    Recortar el presupuesto acelera y degrada a la vez, asi que comparar dos
    variantes por su it/s a secas no dice nada: siempre gana la que menos
    trabajo hace. La pregunta util es si una variante es mas rapida A IGUAL
    PRECISION que el suavizador actual, y para eso hace falta la curva de
    referencia completa, no un solo punto.

    Las variantes `punto_*` son esa curva: mismo solver, distinto presupuesto.
    Cualquier otra variante se situa contra ella interpolando en el error que
    consigue. Quedar por encima de la curva es aportar algo; quedar por debajo
    significa que el mismo resultado se lograba antes gastando menos.
    """
    b = load(base)
    if b is None:
        _log(f"falta el baseline ({base}.json)")
        return

    def err_de(r):
        """Error frente al baseline: el peor de Cl y Cd, en tanto por uno."""
        e = []
        for k in ("cl", "cd"):
            if r.get(k) is not None and b.get(k):
                e.append(abs((r[k] - b[k]) / b[k]))
        return max(e) if e else None

    todos = []
    for f in sorted(os.listdir(OUT)):
        if not f.endswith(".json") or f.startswith("diag"):
            continue
        r = load(f[:-5])
        if r is None or r.get("it_s") is None:
            continue
        if r["iters"] != b["iters"] or r.get("cronometraje_valido") is False:
            continue
        e = err_de(r)
        if e is None:
            continue
        todos.append((e, r["it_s"] / b["it_s"], r))

    ref = sorted((e, v) for e, v, r in todos if r["name"].startswith("punto_")
                 or r["name"] in (base, "equiv"))
    if len(ref) < 2:
        _log("faltan puntos de la curva de referencia (variantes punto_*)")

    def ref_en(e):
        """Aceleracion que da el suavizador actual para ese nivel de error,
        interpolando entre los dos puntos de la curva que lo rodean."""
        if len(ref) < 2:
            return None
        if e <= ref[0][0]:
            return ref[0][1]
        if e >= ref[-1][0]:
            return ref[-1][1]
        for (e0, v0), (e1, v1) in zip(ref, ref[1:]):
            if e0 <= e <= e1:
                t = (e - e0) / (e1 - e0) if e1 > e0 else 0.0
                return v0 + t * (v1 - v0)
        return None

    # Modelo de coste, ajustado sobre las variantes punto_* (mismo suavizador,
    # distinto presupuesto): el tiempo por paso es afin en el numero de V-cycles.
    # La ordenada es el trabajo que no depende de la proyeccion (adveccion,
    # difusion, IBM, fuerzas) y por tanto marca el techo de acelerar el solver.
    # La pendiente es lo que cuesta un V-cycle con el suavizador de referencia,
    # y sirve para ver de un vistazo cuanto mas caro es cada suavizador nuevo.
    pts = [(r["mg_cycles_media"], 1000 * r["s_paso"]) for e, v, r in todos
           if r["name"].startswith("punto_") or r["name"] in (base, "equiv")]
    fijo = por_ciclo = None
    if len(pts) >= 3:
        import numpy as _np
        c = _np.array([q[0] for q in pts]); m = _np.array([q[1] for q in pts])
        fijo, por_ciclo = _np.linalg.lstsq(
            _np.vstack([_np.ones_like(c), c]).T, m, rcond=None)[0]
        print("\n" + "=" * 78)
        print("MODELO DE COSTE")
        print("=" * 78)
        print(f"  ms/paso = {fijo:.1f} fijos + {por_ciclo:.2f} por V-cycle")
        print(f"  parte fija (adveccion, difusion, IBM, fuerzas): {fijo:.0f} ms")
        print(f"  techo con la proyeccion gratis: {1000*b['s_paso']/fijo:.2f}x")
        print("  ciclos que harian falta, con la precision del baseline:")
        for obj in (1.5, 2.0, 3.0, 4.0):
            print(f"     {obj:.1f}x -> {(1000*b['s_paso']/obj - fijo)/por_ciclo:5.1f} ciclos"
                  f"   (hoy {b['mg_cycles_media']:.0f})")

    print("\n" + "=" * 78)
    print("FRONTERA PRECISION - VELOCIDAD   (referencia: suavizador actual)")
    print("=" * 78)
    print(f"{'variante':<20}{'error':>8}{'cic':>6}{'ms/cic':>8}{'x vel':>7}"
          f"{'x ref':>7}{'ventaja':>9}  qué es")
    print("-" * 78)
    for e, v, r in sorted(todos):
        vr = ref_en(e)
        ventaja = v / vr if vr else None
        etiqueta = ",".join(r["opts"]) or "config"
        cic = r.get("mg_cycles_media") or 0
        # Coste marginal de un ciclo con ESTE suavizador, descontando la parte
        # fija medida. Comparado con la pendiente de referencia dice cuanto mas
        # caro sale cada ciclo del precondicionador nuevo.
        ms_cic = ((1000 * r["s_paso"] - fijo) / cic
                  if (fijo is not None and cic > 0) else 0.0)
        marca = ""
        if ventaja is not None and r["name"] not in (base, "equiv") \
           and not r["name"].startswith("punto_"):
            marca = "MEJOR" if ventaja > 1.05 else ("igual" if ventaja > 0.95 else "peor")
        print(f"{r['name']:<20}{100*e:>7.2f}%{cic:>6.1f}{ms_cic:>8.2f}{v:>7.2f}"
              f"{(vr if vr else 0):>7.2f}{(ventaja if ventaja else 0):>8.2f}x  "
              f"{etiqueta:<24}{marca}")
    print("-" * 78)
    print("error   = max(|dCl|, |dCd|) frente al baseline")
    print("ms/cic  = coste marginal de un V-cycle con ese suavizador"
          + (f" (referencia: {por_ciclo:.2f})" if por_ciclo else ""))
    print("x vel   = aceleracion frente al baseline")
    print("x ref   = aceleracion que da el suavizador ACTUAL con ese mismo error")
    print("ventaja = x vel / x ref. Por encima de 1 la variante aporta; por debajo,")
    print("          el mismo resultado se conseguia ya recortando presupuesto.")
    print(f"\npuerta de aceptacion: error <= {100*TOL_REL:.1f}%\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", metavar="NOMBRE")
    ap.add_argument("--opt", action="append", default=[],
                    help="opcion de opt_solver a activar (repetible)")
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--dx", type=float, default=DX)
    ap.add_argument("--max-outer", type=int)
    ap.add_argument("--cycles", type=int)
    ap.add_argument("--divergencia", type=float)
    ap.add_argument("--niveles", type=int)
    ap.add_argument("--pre", type=int)
    ap.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR",
                    help="parametro suelto de main() (repetible); el valor se "
                         "interpreta como JSON y si falla se deja como texto")
    ap.add_argument("--post", type=int)
    ap.add_argument("--factor", action="store_true",
                    help="mide el factor de convergencia del multigrid "
                         "(invalida el cronometraje)")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--queue", action="store_true",
                    help="fases 2-4: las optimizaciones de opt_solver")
    ap.add_argument("--compare", nargs="*", metavar="NOMBRE")
    ap.add_argument("--pareto", action="store_true",
                    help="frontera precision-velocidad frente al suavizador actual")
    a = ap.parse_args()

    if a.pareto:
        pareto()
        return

    if a.compare is not None:
        todos = [n for n, _ in SWEEP] + [n for n, _, _ in QUEUE]
        compare(a.compare or todos)
        return

    if a.sweep:
        for name, ov in SWEEP:
            if load(name) is not None:
                _log(f"{name}: ya hecho, se salta")
                continue
            run_variant(name, iters=a.iters, dx_min=a.dx, **ov)
        compare([n for n, _ in SWEEP])
        return

    if a.queue:
        for name, opts, ov in QUEUE:
            if load(name) is not None:
                _log(f"{name}: ya hecho, se salta")
                continue
            try:
                run_variant(name, opts=opts, iters=a.iters, dx_min=a.dx, **ov)
            except Exception as e:
                # Una variante que revienta no puede llevarse por delante el
                # resto de la cola: se anota y se sigue.
                _log(f"{name}: FALLO -> {type(e).__name__}: {e}")
                with open(os.path.join(OUT, f"{name}.FALLO.txt"), "w") as f:
                    f.write(f"{type(e).__name__}: {e}\n")
        compare(["baseline"] + [n for n, _, _ in QUEUE])
        return

    if a.run:
        ov = {"dx_min": a.dx}
        if a.max_outer is not None:
            ov["mg_max_outer"] = a.max_outer
        if a.cycles is not None:
            ov["mg_cycles_per_outer"] = a.cycles
        if a.divergencia is not None:
            ov["divergencia"] = a.divergencia
        if a.niveles is not None:
            ov["mg_niveles_max"] = a.niveles
        if a.pre is not None:
            ov["mg_pre_suavizado"] = a.pre
        if a.post is not None:
            ov["mg_post_suavizado"] = a.post
        for kv in a.set:
            k, _, v = kv.partition("=")
            try:
                ov[k] = json.loads(v)
            except ValueError:
                ov[k] = v
        run_variant(a.run, opts=a.opt, iters=a.iters, factor=a.factor, **ov)
        return

    ap.print_help()


if __name__ == "__main__":
    main()

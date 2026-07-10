"""
Estudio de convergencia de perfiles (desatendido, deadline-driven, resumible).
================================================================================

Responde a la pregunta: bajo idénticas condiciones de vuelo (α=4°, Re=1e5), ¿el
optimizador hace que perfiles de partida distintos converjan a la MISMA solución?

Ahora el GA usa la config CFD de referencia validada (consistent + SA + MacCormack),
así que el fitness es fiable. El estudio tiene dos brazos:

  - Arm A (diagnóstico): un GA independiente por cada semilla base. Mide cuánto
    divergen los ganadores CON fitness fiable.
  - Arm B (arreglo): un único GA con población MIXTA sembrada de TODAS las semillas
    + cruce. Mide si colapsan a una forma común.

Uso (SIEMPRE con el venv, desde cualquier cwd):
    .venv/bin/python scripts/agent_tests/run_convergence_study.py --deadline-hours 8
    .venv/bin/python scripts/agent_tests/run_convergence_study.py --analyze-only
    .venv/bin/python scripts/agent_tests/run_convergence_study.py --calib-only

REANUDABLE: se salta la calibración si existe calibration.json y cada corrida GA si
su estado_ga_final.json ya existe. El análisis (Fase 3) se re-ejecuta siempre.
Salida en results/convergence_study/.
"""
import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import RunGA

OUT_DIR = os.path.join(ROOT, "results", "convergence_study")
BASE_DIR = os.path.join(OUT_DIR, "base_study")   # estudio base (Arm A/B + análisis)
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
TESTS_DIR = os.path.join(OUT_DIR, "tests")
ARMA_DIR = os.path.join(BASE_DIR, "armA")
ARMB_DIR = os.path.join(BASE_DIR, "armB")
for d in (OUT_DIR, BASE_DIR, PLOTS_DIR, TESTS_DIR, ARMA_DIR, ARMB_DIR):
    os.makedirs(d, exist_ok=True)

CALIB_JSON = os.path.join(OUT_DIR, "calibration.json")

# Condiciones de vuelo del estudio (fijas)
ALPHA = 4.0
CFL = 0.5
T_TARGET = 8.0     # tiempos convectivos por evaluación (estacionario en ~4-5)
U_FAC = 1.6        # dt ≈ CFL·dx/U_FAC (medido en la config de referencia)

SEEDS_ALL = [
    "profiles/NACA_0012_sharp",
    "profiles/AG24",
    "profiles/GM15",
    "profiles/s1014.dat",
]


def iters_for(cfl, dx, t_target=T_TARGET):
    n = t_target * U_FAC / (cfl * dx)
    return int(-(-n // 1000) * 1000)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# FASE 0 — Calibración de dx
# ============================================================
def calibrar():
    if os.path.exists(CALIB_JSON):
        with open(CALIB_JSON) as f:
            cal = json.load(f)
        _log(f"Calibración ya existente -> USE_DX={cal.get('use_dx')} "
             f"(abort={cal.get('abort')})")
        return cal

    _log("FASE 0: calibración dx (0.004 y 0.002 a α=4 y α=5)...")
    resultados = {}
    for dx in (0.004, 0.002):
        for a in (4.0, 5.0):
            cfg = dict(RunGA.CONFIG)
            cfg["dx_min"] = dx
            cfg["simulacion_iteraciones"] = iters_for(CFL, dx)
            cfg["CFL"] = CFL
            key = f"dx{int(dx*1000)}_a{a:.0f}"
            _log(f"  calib {key}: iters={cfg['simulacion_iteraciones']} ...")
            t0 = time.time()
            res = RunGA.simular_perfil("profiles/NACA_0012_sharp", a, cfg)
            dt = time.time() - t0
            entry = {"dx": dx, "alpha": a, "wall_s": round(dt, 1),
                     "iters": cfg["simulacion_iteraciones"]}
            if res:
                entry.update({"cl": res["cl"], "cd": res["cd"], "ld": res["ld"]})
            else:
                entry.update({"cl": None, "cd": None, "ld": None})
            resultados[key] = entry
            _log(f"    -> {entry}")

    def _cl(dx, a):
        return resultados.get(f"dx{int(dx*1000)}_a{a:.0f}", {}).get("cl")
    def _cd(dx, a):
        return resultados.get(f"dx{int(dx*1000)}_a{a:.0f}", {}).get("cd")

    # Gate del solver: NACA α=5 dx=0.002 debe dar Cl físico (~0.45).
    cl_ref = _cl(0.002, 5)
    abort = not (cl_ref is not None and 0.30 <= cl_ref <= 0.60)

    # ¿Aceptar dx=0.004? (no colapsa, Cd físico, offset consistente con dx=0.002)
    accept004 = True
    reasons = []
    for a in (4.0, 5.0):
        c4, c2, d4 = _cl(0.004, a), _cl(0.002, a), _cd(0.004, a)
        if c4 is None or c2 is None or d4 is None:
            accept004 = False; reasons.append(f"a{a:.0f}: NaN"); continue
        if c4 < 0.20:
            accept004 = False; reasons.append(f"a{a:.0f}: cl004={c4:.3f} colapsa")
        if not (0.015 <= d4 <= 0.08):
            accept004 = False; reasons.append(f"a{a:.0f}: cd004={d4:.4f} no físico")
        if abs(c4 - c2) > 0.15:
            accept004 = False; reasons.append(f"a{a:.0f}: |Δcl|={abs(c4-c2):.3f}>0.15")

    use_dx = 0.004 if (accept004 and not abort) else 0.002
    # tiempo por evaluación al dx elegido (usar α=4, el design point)
    t_eval = resultados.get(f"dx{int(use_dx*1000)}_a4", {}).get("wall_s")

    cal = {
        "resultados": resultados,
        "cl_ref_naca_a5_dx002": cl_ref,
        "abort": abort,
        "accept_dx004": accept004,
        "accept_dx004_reasons": reasons,
        "use_dx": use_dx,
        "iters": iters_for(CFL, use_dx),
        "t_eval_s": t_eval,
        "cfl": CFL,
    }
    with open(CALIB_JSON, "w") as f:
        json.dump(cal, f, indent=2, ensure_ascii=False)
    _log(f"FASE 0 done: use_dx={use_dx} accept004={accept004} abort={abort} "
         f"reasons={reasons}")
    return cal


# ============================================================
# FASE 1/2 — corridas del GA
# ============================================================
def _run_ga(nombre, dir_out, cfg_over, deadline_s):
    final = os.path.join(dir_out, "estado_ga_final.json")
    if os.path.exists(final):
        _log(f"  [skip] {nombre}: ya completado")
        return
    os.makedirs(dir_out, exist_ok=True)
    cfg = {
        "directorio_resultados": dir_out,
        "alpha_deg": ALPHA,
        "CFL": CFL,
        "multi_angulo": False,
        "usar_ia": False,
        "tiempo_limite_s": max(60.0, deadline_s),
    }
    cfg.update(cfg_over)
    _log(f"  RUN {nombre}: pop={cfg.get('poblacion_tamano')} "
         f"gen={cfg.get('generaciones')} dx={cfg.get('dx_min')} "
         f"iters={cfg.get('simulacion_iteraciones')} deadline={deadline_s/60:.0f}min")
    try:
        RunGA.main(cfg)
    except Exception as e:
        import traceback
        _log(f"  [!] {nombre} crasheó: {e}")
        with open(os.path.join(dir_out, "FAILED.txt"), "w") as f:
            f.write(traceback.format_exc())


def ejecutar_estudio(cal, deadline_total_s, t_start, reforzado=False):
    use_dx = cal["use_dx"]
    iters = cal["iters"]

    if reforzado:
        # S1: run larga con población grande + más generaciones + sigma adaptativa.
        seeds = SEEDS_ALL
        pop_a, gen_a, pop_b, gen_b = 20, 15, 28, 20
    elif use_dx == 0.004:
        seeds = SEEDS_ALL
        pop_a, gen_a, pop_b, gen_b = 8, 4, 12, 6
    else:
        seeds = SEEDS_ALL[:3]
        pop_a, gen_a, pop_b, gen_b = 6, 2, 8, 2

    base_sim = dict(dx_min=use_dx, simulacion_iteraciones=iters, CFL=CFL)
    if reforzado:
        base_sim.update(sigma_adaptativa=True,
                        sigma_factor_inicial=3.0, sigma_factor_final=1.0)

    # Directorios separados para no colisionar con la run base ya completada.
    arma_dir = os.path.join(OUT_DIR, "reforzado", "armA") if reforzado else ARMA_DIR
    armb_dir = os.path.join(OUT_DIR, "reforzado", "armB") if reforzado else ARMB_DIR

    # Lista de jobs con pesos (Arm B pesa más: es más grande y más valioso)
    jobs = []
    for sp in seeds:
        nm = os.path.splitext(os.path.basename(sp))[0]
        jobs.append(dict(
            nombre=f"armA/{nm}", weight=1.0,
            dir_out=os.path.join(arma_dir, nm),
            cfg={**base_sim, "archivo_base": sp, "archivos_base": None,
                 "poblacion_tamano": pop_a, "generaciones": gen_a,
                 "elites": max(1, pop_a // 4)},
        ))
    jobs.append(dict(
        nombre="armB/mixed", weight=1.6,
        dir_out=os.path.join(armb_dir, "mixed"),
        cfg={**base_sim, "archivo_base": seeds[0], "archivos_base": seeds,
             "poblacion_tamano": pop_b, "generaciones": gen_b,
             "elites": max(2, pop_b // 4)},
    ))

    # Reparto adaptativo: antes de cada job, recalcular con el tiempo que queda.
    for i, job in enumerate(jobs):
        pend = jobs[i:]
        # saltar los ya completados sin gastar presupuesto
        if os.path.exists(os.path.join(job["dir_out"], "estado_ga_final.json")):
            _log(f"  [skip] {job['nombre']}: ya completado")
            continue
        pend_no_done = [j for j in pend
                        if not os.path.exists(
                            os.path.join(j["dir_out"], "estado_ga_final.json"))]
        w_sum = sum(j["weight"] for j in pend_no_done) or 1.0
        remaining = deadline_total_s - (time.time() - t_start)
        if remaining <= 120:
            _log("  Sin presupuesto restante; se detiene la ejecución de GAs.")
            break
        share = remaining * 0.92 * job["weight"] / w_sum
        _run_ga(job["nombre"], job["dir_out"], job["cfg"], share)


ISLAS_DIR = os.path.join(OUT_DIR, "islands")


def _shape_dist(g1, g2, n=120):
    _, c1, t1 = _camber_espesor(g1, n)
    _, c2, t2 = _camber_espesor(g2, n)
    return float(np.sqrt(np.mean((c1 - c2) ** 2) + np.mean((t1 - t2) ** 2)))


def ejecutar_islas(cal, deadline_total_s, t_start,
                   n_epocas=4, migrate_every=2, pop=8):
    """S2 — Modelo de islas con migración.

    Cada semilla es una isla. Cada época corre `migrate_every` generaciones
    partiendo de su mejor actual + los migrantes (mejores de las otras islas).
    Responde la pregunta central: con flujo de genes inter-cuenca, ¿las islas
    convergen a la misma forma? Escribe islands/convergencia_islas.json.
    """
    os.makedirs(ISLAS_DIR, exist_ok=True)
    use_dx, iters = cal["use_dx"], cal["iters"]
    base_sim = dict(dx_min=use_dx, simulacion_iteraciones=iters, CFL=CFL,
                    poblacion_tamano=pop, generaciones=migrate_every,
                    elites=max(1, pop // 4), sigma_adaptativa=True,
                    sigma_factor_inicial=3.0, sigma_factor_final=1.0)

    islas = [(os.path.splitext(os.path.basename(sp))[0], sp) for sp in SEEDS_ALL]
    # dat actual de cada isla (arranca en la semilla original)
    actuales = {nm: os.path.join(ROOT, sp) if not os.path.isabs(sp) else sp
                for nm, sp in islas}
    originales = dict(actuales)  # semilla genuina de cada isla (para graficar)

    hist = []  # por época: fitness y dispersión de forma inter-isla
    n_islas = len(islas)
    # Presupuesto: n_epocas * n_islas evaluaciones-GA
    for ep in range(n_epocas):
        remaining = deadline_total_s - (time.time() - t_start)
        if remaining <= 180:
            _log("  Islas: sin presupuesto; se detiene.")
            break
        jobs_rest = (n_epocas - ep) * n_islas
        share = max(120.0, remaining * 0.9 / jobs_rest)

        ganadores_ep = {}
        for nm, _ in islas:
            dir_ep = os.path.join(ISLAS_DIR, nm, f"epoch{ep}")
            migrantes = [actuales[o] for o, _ in islas if o != nm]
            semillas = [actuales[nm]] + migrantes
            cfg = {**base_sim,
                   "archivo_base": actuales[nm], "archivos_base": semillas,
                   "archivo_original": originales[nm]}
            _run_ga(f"islands/{nm}/e{ep}", dir_ep, cfg, share)
            g = _cargar_ganador(dir_ep)
            if g is None:
                continue
            ganadores_ep[nm] = g
            # persistir ganador como .dat para migración en la época siguiente
            dat = os.path.join(dir_ep, f"{nm}_winner.dat")
            RunGA.guardar_perfil(dat, g["genes"], f"{nm}_islas_e{ep}\n")
            actuales[nm] = dat

        if len(ganadores_ep) >= 2:
            nombres = list(ganadores_ep.keys())
            lds = [ganadores_ep[n]["fitness"] for n in nombres]
            genes = [ganadores_ep[n]["genes"] for n in nombres]
            dists = [_shape_dist(genes[i], genes[j])
                     for i in range(len(genes)) for j in range(i + 1, len(genes))]
            hist.append({
                "epoca": ep,
                "islas": nombres,
                "ld": {n: ganadores_ep[n]["fitness"] for n in nombres},
                "ld_spread": float(max(lds) - min(lds)),
                "shape_dist_media": float(np.mean(dists)) if dists else 0.0,
                "shape_dist_max": float(max(dists)) if dists else 0.0,
            })
            _log(f"  Época {ep}: L/D spread={hist[-1]['ld_spread']:.2f} "
                 f"shape_dist_media={hist[-1]['shape_dist_media']:.4f}")

    out = {
        "condiciones": {"alpha_deg": ALPHA, "Re": 1e5, "CFL": CFL},
        "config": {"n_epocas": n_epocas, "migrate_every": migrate_every,
                   "pop": pop, "dx": use_dx},
        "historia": hist,
        "veredicto": _veredicto_islas(hist),
    }
    with open(os.path.join(ISLAS_DIR, "convergencia_islas.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    _log("  Islas: convergencia_islas.json escrito")
    return out


def _veredicto_islas(hist):
    if len(hist) < 2:
        return ["Insuficientes épocas para veredicto de convergencia."]
    d0, d1 = hist[0]["shape_dist_media"], hist[-1]["shape_dist_media"]
    s0, s1 = hist[0]["ld_spread"], hist[-1]["ld_spread"]
    v = []
    v.append(f"Distancia de forma media: {d0:.4f} (época 0) -> {d1:.4f} "
             f"(época {hist[-1]['epoca']}). "
             + ("CONVERGE (migración acerca las islas)." if d1 < d0 * 0.7
                else "NO converge claramente."))
    v.append(f"Spread L/D: {s0:.2f} -> {s1:.2f}.")
    return v


REFINE_JSON = os.path.join(ISLAS_DIR, "refine_dx002.json")


def refinar_top_k(cal, k=3, dx_ref=0.002, deadline_s=None, t_start=None):
    """S3 — Evaluación en dos niveles.

    Recoge los ganadores de todos los brazos (exploración a dx grueso),
    toma el top-k por fitness de exploración y los re-evalúa a dx_ref (fino).
    El ranking de exploración es inestable (ver T1): el ganador real solo se
    conoce tras refinar. Escribe refine_dx002.json.
    """
    # Recoger candidatos: cualquier estado_ga_final.json bajo el estudio.
    dirs = sorted(set(os.path.dirname(p) for p in glob.glob(
        os.path.join(OUT_DIR, "**", "estado_ga_final.json"), recursive=True)))
    cands = []
    for d in dirs:
        g = _cargar_ganador(d)
        if g is None or g.get("fitness") is None:
            continue
        rel = os.path.relpath(d, OUT_DIR)
        cands.append({"id": rel, "genes": g["genes"],
                      "ld_expl": float(g["fitness"]), "header": g["nombre"]})
    if not cands:
        _log("  S3: no hay ganadores que refinar.")
        return None

    cands.sort(key=lambda c: c["ld_expl"], reverse=True)
    top = cands[:k]
    _log(f"  S3: {len(cands)} candidatos, refinando top-{len(top)} a dx={dx_ref}")

    cfg = dict(RunGA.CONFIG)
    cfg["dx_min"] = dx_ref
    cfg["simulacion_iteraciones"] = iters_for(CFL, dx_ref)
    cfg["CFL"] = CFL

    tmp_dir = os.path.join(ISLAS_DIR, "refine_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    filas = []
    for c in top:
        if deadline_s is not None and t_start is not None:
            if (time.time() - t_start) > deadline_s - 120:
                _log("  S3: sin presupuesto; refinado parcial.")
                break
        dat = os.path.join(tmp_dir, c["id"].replace("/", "__") + ".dat")
        RunGA.guardar_perfil(dat, c["genes"], c["header"] + "\n")
        res = RunGA.simular_perfil(dat, ALPHA, cfg)
        if res is None:
            filas.append({"id": c["id"], "ld_expl": c["ld_expl"], "error": True})
            _log(f"    {c['id']}: FALLO")
            continue
        filas.append({"id": c["id"], "ld_expl": round(c["ld_expl"], 4),
                      "ld_ref": res["ld"], "cl_ref": res["cl"], "cd_ref": res["cd"]})
        _log(f"    {c['id']}: expl {c['ld_expl']:.2f} -> dx{dx_ref} {res['ld']:.2f}")

    ok = [f for f in filas if not f.get("error")]
    if not ok:
        # Sin refinados (p.ej. deadline agotado): no sobrescribir un refine_dx002
        # previo válido con un resultado vacío.
        _log("  S3: 0 refinados; se conserva refine_dx002.json previo si existe.")
        return None
    ok.sort(key=lambda f: f["ld_ref"], reverse=True)
    ganador_expl = top[0]["id"] if top else None
    ganador_ref = ok[0]["id"] if ok else None
    out = {
        "alpha": ALPHA, "dx_exploracion": cal["use_dx"], "dx_refinado": dx_ref,
        "k": k, "filas_ordenadas_por_ref": ok,
        "ganador_exploracion": ganador_expl, "ganador_refinado": ganador_ref,
        "cambio_ganador": ganador_expl != ganador_ref,
        "veredicto": (
            f"Ganador exploración={ganador_expl} -> refinado={ganador_ref}. "
            + ("El refinado CAMBIA al ganador (dos niveles necesario)."
               if ganador_expl != ganador_ref else
               "El ganador se mantiene tras refinar.")),
    }
    with open(REFINE_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    _log("  S3: " + out["veredicto"])
    _log(f"  S3: escrito {REFINE_JSON}")
    return out


# ============================================================
# FASE 3 — Análisis de convergencia
# ============================================================
def _cargar_ganador(dir_out):
    f = os.path.join(dir_out, "estado_ga_final.json")
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        est = json.load(fh)
    mg = est.get("mejor_global") or {}
    genes = mg.get("genes")
    if not genes:
        return None
    g = np.asarray(genes, dtype=float)
    res = mg.get("resultados") or {}
    r = res.get(f"{ALPHA:.1f}") or (next(iter(res.values())) if res else {})
    return {
        "dir": dir_out,
        "nombre": os.path.basename(dir_out),
        "genes": g,
        "fitness": mg.get("fitness"),
        "cl": r.get("cl"), "cd": r.get("cd"), "ld": r.get("ld"),
        "historial": est.get("historial", []),
        "poblacion_resumen": est.get("poblacion_resumen", []),
    }


def _camber_espesor(genes, n=120):
    le = int(np.argmin(genes[:, 0]))
    x, c, t = RunGA.descomponer_camber_espesor(genes, le, n_muestras=n)
    return x, c, t


def analizar_convergencia():
    _log("FASE 3: análisis de convergencia...")
    arma = [w for w in (_cargar_ganador(d)
            for d in sorted(glob.glob(os.path.join(ARMA_DIR, "*")))) if w]
    armb = _cargar_ganador(os.path.join(ARMB_DIR, "mixed"))

    out = {"alpha": ALPHA, "condiciones": {"alpha_deg": ALPHA, "Re": 1e5, "CFL": CFL},
           "armA": [], "armB": None, "metricas": {}}

    for w in arma:
        out["armA"].append({k: w[k] for k in ("nombre", "fitness", "cl", "cd", "ld")})
    if armb:
        out["armB"] = {k: armb[k] for k in ("nombre", "fitness", "cl", "cd", "ld")}

    # Dispersión inter-semilla (Arm A) con fitness fiable
    lds = [w["ld"] for w in arma if w["ld"]]
    cls = [w["cl"] for w in arma if w["cl"] is not None]
    cds = [w["cd"] for w in arma if w["cd"] is not None]
    def _disp(v):
        if len(v) < 2:
            return None
        v = np.asarray(v, float)
        return {"min": float(v.min()), "max": float(v.max()),
                "spread": float(v.max() - v.min()),
                "mean": float(v.mean()), "std": float(v.std()),
                "cv": float(v.std() / v.mean()) if v.mean() else None}
    out["metricas"]["dispersion_armA"] = {
        "LD": _disp(lds), "Cl": _disp(cls), "Cd": _disp(cds)}

    # Matriz de distancia de forma (camber+espesor L2) entre todos los ganadores
    todos = arma + ([armb] if armb else [])
    formas = {}
    for w in todos:
        try:
            x, c, t = _camber_espesor(w["genes"])
            formas[w["nombre"]] = (c, t)
        except Exception:
            pass
    nombres = list(formas.keys())
    dist = np.zeros((len(nombres), len(nombres)))
    for i, ni in enumerate(nombres):
        for j, nj in enumerate(nombres):
            ci, ti = formas[ni]; cj, tj = formas[nj]
            dist[i, j] = float(np.sqrt(np.mean((ci - cj) ** 2) + np.mean((ti - tj) ** 2)))
    out["metricas"]["shape_distance"] = {"nombres": nombres, "matriz": dist.tolist()}
    if len(nombres) > 1:
        iu = np.triu_indices(len(nombres), 1)
        out["metricas"]["shape_distance_media"] = float(dist[iu].mean())
        out["metricas"]["shape_distance_max"] = float(dist[iu].max())

    # Arm B: convergencia de la población (spread de fitness por generación)
    if armb:
        hist = armb["historial"]
        out["metricas"]["armB_convergencia"] = [
            {"gen": h["gen"], "mejor": h.get("mejor"), "media": h.get("media"),
             "peor": h.get("peor")} for h in hist]
        fits_fin = [p["fitness"] for p in armb["poblacion_resumen"]
                    if p.get("fitness")]
        if len(fits_fin) > 1:
            fa = np.asarray(fits_fin, float)
            out["metricas"]["armB_poblacion_final"] = {
                "n": len(fits_fin), "mean": float(fa.mean()),
                "std": float(fa.std()), "spread": float(fa.max() - fa.min())}

    # Veredicto heurístico
    verdict = []
    d_ld = out["metricas"]["dispersion_armA"].get("LD")
    if d_ld:
        if d_ld["cv"] is not None and d_ld["cv"] < 0.08:
            verdict.append(f"Arm A: L/D converge (CV={d_ld['cv']:.3f}, spread={d_ld['spread']:.2f}).")
        else:
            verdict.append(f"Arm A: L/D NO converge (CV={d_ld['cv']:.3f}, spread={d_ld['spread']:.2f}) "
                           f"-> las semillas siguen en óptimos locales distintos.")
    sdm = out["metricas"].get("shape_distance_media")
    if sdm is not None:
        verdict.append(f"Distancia de forma media entre ganadores = {sdm:.4f} "
                       f"(0 = misma forma).")
    if armb and lds:
        best_a = max(lds)
        if armb["ld"] and armb["ld"] >= best_a - 1e-6:
            verdict.append(f"Arm B (población mixta+cruce) iguala/supera al mejor Arm A "
                           f"(L/D {armb['ld']:.2f} vs {best_a:.2f}) -> la mezcla ayuda a converger.")
        elif armb["ld"]:
            verdict.append(f"Arm B L/D={armb['ld']:.2f} < mejor Arm A {best_a:.2f} "
                           f"(quizá pocas generaciones dentro del deadline).")
    out["veredicto"] = verdict

    with open(os.path.join(BASE_DIR, "convergencia.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    _generar_plots(arma, armb, formas, nombres, dist)
    _escribir_informe(out, arma, armb)
    _log("FASE 3 done: convergencia.json + RESUMEN_convergencia.md + PNGs")
    return out


def _generar_plots(arma, armb, formas, nombres, dist):
    todos = arma + ([armb] if armb else [])
    try:  # 1) formas ganadoras superpuestas
        plt.figure(figsize=(9, 4))
        for w in todos:
            g = w["genes"]
            plt.plot(g[:, 0], g[:, 1], lw=1.2, label=w["nombre"])
        plt.axis("equal"); plt.legend(fontsize=8); plt.title("Ganadores (α=4°, Re=1e5)")
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "formas_ganadoras.png"), dpi=130); plt.close()
    except Exception as e:
        _log(f"  [!] plot formas: {e}")

    try:  # 2) barras L/D
        plt.figure(figsize=(7, 4))
        nm = [w["nombre"] for w in todos]; ld = [w["ld"] or 0 for w in todos]
        cols = ["#4c78a8"] * len(arma) + (["#e45756"] if armb else [])
        plt.bar(nm, ld, color=cols)
        plt.ylabel("L/D"); plt.title("L/D por semilla (azul=Arm A, rojo=Arm B mixto)")
        plt.xticks(rotation=20, ha="right"); plt.grid(alpha=0.3, axis="y")
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "ld_por_semilla.png"), dpi=130)
        plt.close()
    except Exception as e:
        _log(f"  [!] plot LD: {e}")

    try:  # 3) camber + espesor superpuestos
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        for w in todos:
            if w["nombre"] in formas:
                x, c, t = _camber_espesor(w["genes"])
                a1.plot(x, c, lw=1.3, label=w["nombre"]); a2.plot(x, t, lw=1.3)
        a1.set_title("Camber c(x)"); a2.set_title("Espesor t(x)")
        a1.legend(fontsize=8); a1.grid(alpha=0.3); a2.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "camber_espesor.png"), dpi=130)
        plt.close()
    except Exception as e:
        _log(f"  [!] plot camber: {e}")

    if armb and armb["historial"]:  # 4) convergencia Arm B
        try:
            h = armb["historial"]
            g = [x["gen"] for x in h]
            plt.figure(figsize=(7, 4))
            plt.plot(g, [x.get("mejor") for x in h], "-o", label="mejor")
            plt.plot(g, [x.get("media") for x in h], "-s", label="media")
            plt.plot(g, [x.get("peor") for x in h], "-^", label="peor")
            plt.xlabel("generación"); plt.ylabel("L/D")
            plt.title("Arm B: convergencia de la población mixta")
            plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "armB_convergencia.png"), dpi=130); plt.close()
        except Exception as e:
            _log(f"  [!] plot armB conv: {e}")

    if len(nombres) > 1:  # 5) heatmap distancia de forma
        try:
            plt.figure(figsize=(6, 5))
            im = plt.imshow(dist, cmap="viridis")
            plt.colorbar(im, label="distancia de forma (L2 camber+espesor)")
            plt.xticks(range(len(nombres)), nombres, rotation=45, ha="right", fontsize=8)
            plt.yticks(range(len(nombres)), nombres, fontsize=8)
            plt.title("Distancia entre ganadores"); plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "distancia_forma.png"), dpi=130); plt.close()
        except Exception as e:
            _log(f"  [!] plot heatmap: {e}")


def _escribir_informe(out, arma, armb):
    L = ["# Estudio de convergencia de perfiles\n",
         f"Condiciones: α={ALPHA}°, Re=1e5, CFL={CFL}. Config CFD de referencia "
         "(consistent + SA + MacCormack).\n",
         "\n## Veredicto\n"]
    for v in out.get("veredicto", []):
        L.append(f"- {v}\n")
    L.append("\n## Arm A — GA independiente por semilla (diagnóstico)\n")
    L.append("| semilla | L/D | Cl | Cd |\n|---|---|---|---|\n")
    for w in arma:
        L.append(f"| {w['nombre']} | {w['ld']} | {w['cl']} | {w['cd']} |\n")
    d = out["metricas"].get("dispersion_armA", {}).get("LD")
    if d:
        L.append(f"\nDispersión L/D: spread={d['spread']:.3f}, CV={d['cv']:.3f} "
                 f"(mean={d['mean']:.3f}).\n")
    if armb:
        L.append("\n## Arm B — población mixta + cruce (arreglo)\n")
        L.append(f"Ganador mixto: L/D={armb['ld']}, Cl={armb['cl']}, Cd={armb['cd']}.\n")
        pf = out["metricas"].get("armB_poblacion_final")
        if pf:
            L.append(f"Población final: std L/D={pf['std']:.3f}, spread={pf['spread']:.3f} "
                     f"(n={pf['n']}).\n")
    sdm = out["metricas"].get("shape_distance_media")
    if sdm is not None:
        L.append(f"\nDistancia de forma media entre ganadores: {sdm:.4f} "
                 f"(máx {out['metricas'].get('shape_distance_max'):.4f}).\n")
    L.append("\n## Figuras\n- formas_ganadoras.png\n- ld_por_semilla.png\n"
             "- camber_espesor.png\n- armB_convergencia.png\n- distancia_forma.png\n")
    with open(os.path.join(BASE_DIR, "RESUMEN_convergencia.md"), "w") as f:
        f.writelines(L)


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline-hours", type=float, default=8.0)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--calib-only", action="store_true")
    ap.add_argument("--islands", action="store_true",
                    help="S2: modelo de islas con migración (experimento de convergencia definitivo)")
    ap.add_argument("--island-epochs", type=int, default=4)
    ap.add_argument("--island-migrate-every", type=int, default=2)
    ap.add_argument("--island-pop", type=int, default=8)
    ap.add_argument("--reforzado", action="store_true",
                    help="S1: run larga (pop grande, más gen, sigma adaptativa) en results/convergence_study/reforzado/")
    ap.add_argument("--refine", action="store_true",
                    help="S3: re-rankea los ganadores existentes a dx 0.002 (dos niveles)")
    ap.add_argument("--refine-k", type=int, default=3)
    args = ap.parse_args()

    t_start = time.time()
    deadline_total_s = args.deadline_hours * 3600.0
    _log(f"INICIO estudio. deadline={args.deadline_hours}h")

    if args.analyze_only:
        analizar_convergencia(); return

    cal = calibrar()
    if args.calib_only:
        return
    if args.refine:
        refinar_top_k(cal, k=args.refine_k,
                      deadline_s=deadline_total_s, t_start=t_start)
        return
    if cal.get("abort"):
        _log("ABORTADO: el solver no reproduce Cl físico (NACA α=5 dx=0.002 fuera de "
             f"[0.30,0.60]: cl={cal.get('cl_ref_naca_a5_dx002')}). Revisar config antes "
             "de gastar presupuesto.")
        return

    if args.islands:
        ejecutar_islas(cal, deadline_total_s, t_start,
                       n_epocas=args.island_epochs,
                       migrate_every=args.island_migrate_every,
                       pop=args.island_pop)
        refinar_top_k(cal, k=args.refine_k, deadline_s=deadline_total_s, t_start=t_start)
        _log(f"FIN islas. Tiempo total: {(time.time()-t_start)/60:.1f} min")
        return

    ejecutar_estudio(cal, deadline_total_s, t_start, reforzado=args.reforzado)
    refinar_top_k(cal, k=args.refine_k, deadline_s=deadline_total_s, t_start=t_start)
    if not args.reforzado:
        analizar_convergencia()
    _log(f"FIN. Tiempo total: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()

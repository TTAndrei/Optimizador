"""
Base de datos de series temporales a dx=0.004 para dos cosas a la vez.

1. ¿Sirve dx=0.004 para *rankear*? El GCI dice que esa malla está fuera del rango
   asintótico y sus valores absolutos no valen, pero el GA no necesita valores
   absolutos, necesita orden. Se mide comparando estos L/D con los de dx=0.002 de
   la revalidación (mismos 20 individuos).

2. ¿Se puede parar por convergencia en vez de por presupuesto fijo? El criterio
   actual se ajustó sobre 2 series y no generaliza (errores de hasta -24%). Con 20
   series completas se puede ajustar sobre 10 y validar sobre las otras 10, que es
   lo que faltaba.

Las series se corren sin parada anticipada y hasta t=12 tiempos convectivos, muy
por encima del t≈5.5 donde el L/D entra en el 2%: el asintótico de referencia tiene
que estar fuera de duda, porque todo lo demás se mide contra él.

    .venv/bin/python scripts/agent_tests/series_dx004.py
    .venv/bin/python scripts/agent_tests/series_dx004.py --analizar-solo

Salida: results/verificacion_numerica/series_dx004/  (npz por individuo + JSON)
Cachea cada individuo, así que se puede interrumpir y reanudar.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

import numpy as np

import RunGA

# La campaña anterior está archivada; de ahí salen las referencias a dx=0.002.
ARCHIVO = os.path.join(ROOT, "results", "1eraGranOptimizacion", "verificacion_numerica")
OUT = os.path.join(ROOT, "results", "verificacion_numerica", "series_dx004")
TMP = os.path.join(OUT, "perfiles")

RE_OBJETIVO = 1e5
ALPHA = 4.0
DX = 0.004
T_TARGET = 12.0
CFL = 0.5
U_FAC = 1.6
EXTRA = {"transition_model": "sa_bc", "freestream_Tu": 0.1}


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def iters_for(dx, t_target=T_TARGET):
    return int(-(-(t_target * U_FAC / (CFL * dx)) // 1000) * 1000)


def referencias():
    """Los 20 individuos de la revalidación, con su L/D a dx=0.002."""
    d = json.load(open(os.path.join(ARCHIVO, "revalidacion_ranking.json")))
    return {r["id"]: r for r in d["individuos"]}


def geometrias(ids):
    """Perfiles del historial, indexados por id."""
    out = {}
    with open(os.path.join(ROOT, "aprendizaje_ML.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("id") not in ids:
                continue
            p = np.asarray(d.get("perfil") or [], dtype=float)
            if p.ndim == 2 and len(p) >= 20:
                out[d["id"]] = p
    return out


def escribir_dat(perfil, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("series_dx004\n")
        for x, y in perfil:
            f.write(f"{x:.6f} {y:.6f}\n")
    return path


def correr():
    os.makedirs(TMP, exist_ok=True)
    path = os.path.join(OUT, "series_dx004.json")
    hecho = json.load(open(path))["individuos"] if os.path.exists(path) else []
    ya = {r["id"] for r in hecho}

    ref = referencias()
    geo = geometrias(set(ref))
    faltan = set(ref) - set(geo)
    if faltan:
        _log(f"[!] sin geometría en el historial: {sorted(faltan)}")

    cfg = dict(RunGA.CONFIG)
    cfg["sim_extra_params"] = {**cfg.get("sim_extra_params", {}), **EXTRA,
                               "stop_on_clcd_convergence": False,
                               "stop_on_convergence": False}
    cfg.update(nu=1.0 / RE_OBJETIVO, v0x=1.0, chord=1.0, rho=1.0, CFL=CFL, dx_min=DX)
    cfg["simulacion_iteraciones"] = iters_for(DX)
    _log(f"dx={DX}, {cfg['simulacion_iteraciones']} iters (t={T_TARGET}), sin early-stop")

    pend = [i for i in sorted(ref, key=lambda k: ref[k]["ld_original"]) if i in geo]
    for n, ind_id in enumerate(pend, 1):
        if ind_id in ya:
            continue
        npz = os.path.join(OUT, f"serie_{ind_id}.npz")
        cfg["dump_series_path"] = npz
        dat = escribir_dat(geo[ind_id], os.path.join(TMP, f"ind_{ind_id}.dat"))
        t0 = time.time()
        r = ref[ind_id]
        _log(f"[{n}/{len(pend)}] id={ind_id} (dx0.002 L/D={r['ld_revalidado']:.3f}) ...")
        res = RunGA.simular_perfil(dat, ALPHA, cfg)
        if res is None:
            _log("   fallida, se omite")
            continue
        fila = {
            "id": ind_id,
            "ld_dx004": res["ld"], "cl_dx004": res["cl"], "cd_dx004": res["cd"],
            "ld_ci95": res.get("ld_ci95"),
            "ld_dx002": r["ld_revalidado"], "ld_ga": r["ld_original"],
            "serie": os.path.relpath(npz, ROOT), "wall_s": round(time.time() - t0, 1),
        }
        hecho.append(fila)
        _log(f"   -> dx0.004 L/D={res['ld']:.3f}  vs dx0.002 {r['ld_revalidado']:.3f} "
             f"({100*(res['ld']-r['ld_revalidado'])/r['ld_revalidado']:+.1f}%)  ({fila['wall_s']}s)")
        with open(path, "w") as f:
            json.dump({"individuos": hecho}, f, indent=2, ensure_ascii=False)

    return analizar(hecho)


def analizar(filas):
    if len(filas) < 5:
        _log("muestra insuficiente")
        return None
    from scipy import stats

    a = np.array([f["ld_dx004"] for f in filas], float)
    b = np.array([f["ld_dx002"] for f in filas], float)
    g = np.array([f["ld_ga"] for f in filas], float)

    def rangos(x, y, etiqueta):
        rho = stats.spearmanr(x, y)
        tau = stats.kendalltau(x, y)
        alto = np.argsort(x)[len(x) // 2:]
        k = max(3, len(x) // 5)
        solape = len(set(np.argsort(-x)[:k]) & set(np.argsort(-y)[:k])) / k
        d = {"comparacion": etiqueta,
             "spearman_rho": round(float(rho[0]), 4), "spearman_p": float(f"{rho[1]:.3g}"),
             "kendall_tau": round(float(tau[0]), 4),
             "rho_mitad_alta": round(float(stats.spearmanr(x[alto], y[alto])[0]), 4),
             "top_k": k, "solape_top_k": round(solape, 3)}
        _log(f"{etiqueta}: rho={d['spearman_rho']} (mitad alta {d['rho_mitad_alta']}) "
             f"tau={d['kendall_tau']} solape top-{k}={d['solape_top_k']}")
        return d

    out = {
        "n": len(filas), "Re": RE_OBJETIVO, "alpha": ALPHA, "dx": DX,
        "t_target": T_TARGET,
        # La pregunta que decide si el GA puede explorar barato.
        "dx004_vs_dx002": rangos(a, b, "dx=0.004 vs dx=0.002"),
        # Control: el GA original contra la verdad, ya conocido (rho 0.80).
        "ga_vs_dx002": rangos(g, b, "fitness GA original vs dx=0.002"),
        "sesgo_medio_pct": round(float(np.mean(100 * (a - b) / b)), 2),
        "sesgo_std_pct": round(float(np.std(100 * (a - b) / b)), 2),
        "individuos": filas,
    }
    with open(os.path.join(OUT, "series_dx004.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    _log(f"sesgo dx0.004 vs dx0.002: {out['sesgo_medio_pct']:+.1f}% ± {out['sesgo_std_pct']}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analizar-solo", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.analizar_solo:
        p = os.path.join(OUT, "series_dx004.json")
        analizar(json.load(open(p))["individuos"])
    else:
        correr()

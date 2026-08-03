"""
¿Sobrevive el ranking del GA al recalibrado del criterio de parada?

El early-stop por defecto medía flujo no desarrollado y sesgaba el L/D hasta un
47%, con signo que cambia según el régimen (ver docs/verificacion_numerica.md).
Re-evaluar los 4 ganadores no basta para saber si el GA *seleccionó* bien: si el
sesgo reordena a los individuos, el GA optimizó una función distinta de la que
creíamos. Esto lo comprueba re-simulando una muestra estratificada del historial
con el criterio calibrado y midiendo la correlación de rangos.

    .venv/bin/python scripts/agent_tests/revalidar_ranking.py --n 30

Salida: results/verificacion_numerica/revalidacion_ranking.{json,png}
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import RunGA
from verificacion_numerica import OUT, EXTRA, criterio_calibrado, iters_for, _log

RE_OBJETIVO = 1e5
ALPHA = 4.0
DX = 0.002
TMP = os.path.join(ROOT, "results", "verificacion_numerica", "perfiles_revalidacion")


DX_GA = 0.004  # malla con la que el GA exploró y seleccionó


def cargar_historial(re=RE_OBJETIVO, dx_original=DX_GA):
    """Registros del historial en la condición dada, con geometría y fitness.

    Se filtra por dx_original porque el historial mezcla dos poblaciones: 2425
    evaluaciones a dx=0.004 (la exploración del GA) y 368 a dx=0.001, con
    distribuciones de L/D muy distintas (mediana 10.04 vs 6.05). Mezclarlas
    metería la diferencia de malla dentro de la correlación de rangos.
    """
    regs = []
    with open(os.path.join(ROOT, "aprendizaje_ML.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            cond = d.get("condiciones") or {}
            if abs(float(cond.get("Re", 0)) - re) > 1:
                continue
            res = (d.get("resultados") or {}).get(f"{ALPHA:.1f}")
            if not res or not np.isfinite(res.get("ld", np.nan)) or res["ld"] <= 0:
                continue
            if dx_original is not None and abs(float(cond.get("dx_min", 0)) - dx_original) > 1e-9:
                continue
            perfil = np.asarray(d.get("perfil") or [], dtype=float)
            if perfil.ndim != 2 or len(perfil) < 20:
                continue
            regs.append({
                "id": d.get("id"), "ld_original": float(res["ld"]),
                "cl_original": float(res.get("cl", np.nan)),
                "cd_original": float(res.get("cd", np.nan)),
                "perfil": perfil,
                "dx_original": float(cond.get("dx_min", np.nan)),
                "generacion": (d.get("metadata") or {}).get("generacion"),
            })
    return regs


def muestra_estratificada(regs, n):
    """n individuos repartidos por deciles de fitness original.

    Un muestreo aleatorio simple concentraría la muestra en el grueso de la
    distribución y no diría nada sobre los extremos, que es justo donde el
    ranking importa para la selección.
    """
    regs = sorted(regs, key=lambda r: r["ld_original"])
    idx = np.unique(np.linspace(0, len(regs) - 1, n).astype(int))
    return [regs[i] for i in idx]


def escribir_dat(perfil, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("revalidacion\n")
        for x, y in perfil:
            f.write(f"{x:.6f} {y:.6f}\n")
    return path


def revalidar(n=20):
    os.makedirs(TMP, exist_ok=True)
    path = f"{OUT}/revalidacion_ranking.json"
    hecho = json.load(open(path))["individuos"] if os.path.exists(path) else []
    ya = {r["id"] for r in hecho}

    regs = cargar_historial()
    _log(f"historial Re={RE_OBJETIVO:.0e}: {len(regs)} registros con geometría y L/D")
    sel = muestra_estratificada(regs, n)
    _log(f"muestra estratificada: {len(sel)} individuos, "
         f"L/D original {sel[0]['ld_original']:.2f} … {sel[-1]['ld_original']:.2f}")

    # Sin parada anticipada, presupuesto fijo. El criterio recalibrado se ajustó
    # sobre dos casos y no generaliza: en la polar del NACA0012 cortó α=6° en
    # t=1.75 con 10 muestras. Ese ruido entraría justo en la correlación de
    # rangos que se quiere medir, y haría confundir un fallo de medición con una
    # pérdida real de ranking.
    cfg = dict(RunGA.CONFIG)
    cfg["sim_extra_params"] = {**cfg.get("sim_extra_params", {}), **EXTRA,
                               "stop_on_clcd_convergence": False,
                               "stop_on_convergence": False}
    cfg.update(nu=1.0 / RE_OBJETIVO, v0x=1.0, chord=1.0, rho=1.0, CFL=0.5, dx_min=DX)
    cfg["simulacion_iteraciones"] = iters_for(DX)
    _log(f"presupuesto fijo: {cfg['simulacion_iteraciones']} iters, sin early-stop")

    for i, r in enumerate(sel, 1):
        if r["id"] in ya:
            continue
        dat = escribir_dat(r["perfil"], os.path.join(TMP, f"ind_{r['id']}.dat"))
        t0 = time.time()
        _log(f"[{i}/{len(sel)}] id={r['id']} L/D_original={r['ld_original']:.3f} ...")
        res = RunGA.simular_perfil(dat, ALPHA, cfg)
        if res is None:
            _log("   fallida, se omite")
            continue
        fila = {
            "id": r["id"], "generacion": r["generacion"],
            "ld_original": r["ld_original"], "cl_original": r["cl_original"],
            "cd_original": r["cd_original"], "dx_original": r["dx_original"],
            "ld_revalidado": res["ld"], "cl_revalidado": res["cl"], "cd_revalidado": res["cd"],
            "ld_ci95": res.get("ld_ci95"), "n_samples": res.get("n_samples"),
            "converged_clcd": res.get("converged_clcd"), "wall_s": round(time.time() - t0, 1),
        }
        hecho.append(fila)
        _log(f"   -> L/D {r['ld_original']:.3f} → {res['ld']:.3f} "
             f"({100*(res['ld']-r['ld_original'])/r['ld_original']:+.1f}%)  ({fila['wall_s']}s)")
        with open(path, "w") as f:
            json.dump({"individuos": hecho}, f, indent=2, ensure_ascii=False)

    return analizar(hecho)


def analizar(filas):
    if len(filas) < 5:
        _log("revalidación: muestra insuficiente para estadística de rangos")
        return None
    from scipy import stats

    a = np.array([f["ld_original"] for f in filas], float)
    b = np.array([f["ld_revalidado"] for f in filas], float)
    rho, p_rho = stats.spearmanr(a, b)
    tau, p_tau = stats.kendalltau(a, b)
    r_pearson = float(np.corrcoef(a, b)[0, 1])

    # Lo que de verdad importa para un GA no es el orden completo, sino si el
    # mejor del historial sigue estando arriba tras recalibrar.
    k = max(3, len(a) // 5)
    top_orig = set(np.argsort(-a)[:k])
    top_reval = set(np.argsort(-b)[:k])
    solape = len(top_orig & top_reval) / k

    out = {
        "n": len(filas), "Re": RE_OBJETIVO, "alpha": ALPHA, "dx": DX,
        "spearman_rho": round(float(rho), 4), "spearman_p": float(f"{p_rho:.3g}"),
        "kendall_tau": round(float(tau), 4), "kendall_p": float(f"{p_tau:.3g}"),
        "pearson_r": round(r_pearson, 4),
        "top_k": k, "solape_top_k": round(solape, 3),
        "cambio_medio_pct": round(float(np.mean(100 * (b - a) / a)), 2),
        "cambio_std_pct": round(float(np.std(100 * (b - a) / a)), 2),
        "individuos": filas,
    }
    with open(f"{OUT}/revalidacion_ranking.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    _log(f"Spearman ρ={out['spearman_rho']} (p={out['spearman_p']})  "
         f"Kendall τ={out['kendall_tau']}  solape top-{k}={out['solape_top_k']}  "
         f"cambio medio {out['cambio_medio_pct']:+.1f}% ± {out['cambio_std_pct']}")
    figura(out)
    return out


def figura(out):
    f = out["individuos"]
    a = np.array([x["ld_original"] for x in f]); b = np.array([x["ld_revalidado"] for x in f])
    err = np.array([x["ld_ci95"] or 0.0 for x in f])
    ra = len(a) - 1 - np.argsort(np.argsort(a)); rb = len(b) - 1 - np.argsort(np.argsort(b))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), layout="constrained")
    ax = axes[0]
    ax.errorbar(a, b, yerr=err, fmt="o", ms=6, capsize=3, color="#c0392b", alpha=0.85)
    lo, hi = min(a.min(), b.min()) * 0.9, max(a.max(), b.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "--", color="#7f8c8d", lw=1.2, label="identidad")
    ax.set_xlabel("$L/D$ original (criterio sesgado)"); ax.set_ylabel("$L/D$ revalidado")
    ax.set_title(f"Pearson r = {out['pearson_r']}", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(ra, rb, "o", ms=6, color="#2980b9")
    ax.plot([0, len(a) - 1], [0, len(a) - 1], "--", color="#7f8c8d", lw=1.2)
    ax.set_xlabel("rango original"); ax.set_ylabel("rango revalidado")
    ax.set_title(f"Spearman ρ = {out['spearman_rho']} · Kendall τ = {out['kendall_tau']}", fontsize=10)
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.hist(100 * (b - a) / a, bins=12, color="#8e44ad", alpha=0.8)
    ax.axvline(out["cambio_medio_pct"], color="#c0392b", lw=1.5,
               label=f"media {out['cambio_medio_pct']:+.1f}%")
    ax.axvline(0, color="#7f8c8d", ls="--", lw=1.2)
    ax.set_xlabel("cambio en $L/D$ [%]"); ax.set_ylabel("individuos")
    ax.set_title("Distribución del sesgo", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle(f"¿Sobrevive el ranking del GA al recalibrado? · n={out['n']}, "
                 f"Re={out['Re']:.0e}, α={out['alpha']}°", fontsize=11)
    fig.savefig(f"{OUT}/revalidacion_ranking.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="individuos a re-evaluar")
    ap.add_argument("--analizar-solo", action="store_true",
                    help="recalcula estadística y figura sin simular")
    args = ap.parse_args()
    if args.analizar_solo:
        p = f"{OUT}/revalidacion_ranking.json"
        analizar(json.load(open(p))["individuos"])
    else:
        revalidar(args.n)

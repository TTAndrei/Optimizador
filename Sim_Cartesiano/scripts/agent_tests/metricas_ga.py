"""
Métricas y figuras del optimizador genético — post-proceso puro, sin GPU.

Fuentes:
  - results/convergence_study/mixto_*/estado_ga*.json  -> historial por generación
  - aprendizaje_ML.jsonl                               -> 1 registro por evaluación CFD
  - results/convergence_study/refine_todos.json        -> ganadores por Re

Salida: results/metricas_ga/  (figuras PNG + metricas.json + INFORME.md)

Uso:
    .venv/bin/python scripts/agent_tests/metricas_ga.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent.parent.parent
STUDY = ROOT / "results" / "convergence_study"
JSONL = ROOT / "aprendizaje_ML.jsonl"
OUT = ROOT / "results" / "metricas_ga"

PARAMS = ["espesor_max", "espesor_pos", "camber_max", "camber_pos", "le_radius", "te_gap"]
PARAM_LABEL = {
    "espesor_max": "espesor máx. $t/c$",
    "espesor_pos": "posición espesor máx. $x_t/c$",
    "camber_max": "curvatura máx. $f/c$",
    "camber_pos": "posición curvatura máx. $x_f/c$",
    "le_radius": "radio borde ataque $r_{LE}/c$",
    "te_gap": "espesor borde salida $t_{TE}/c$",
}

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
    "legend.frameon": False,
})


# ----------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------
def cargar_estudios() -> dict:
    """Cada mixto_*/ con su historial por generación."""
    estudios = {}
    for d in sorted(STUDY.glob("mixto_*")):
        if not d.is_dir():
            continue
        f = d / "estado_ga_final.json"
        if not f.exists():
            f = d / "estado_ga.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        if not data.get("historial"):
            continue
        estudios[d.name.replace("mixto_", "")] = {
            "hist": data["historial"],
            "config": data.get("config", {}),
            "mejor": data.get("mejor_global", {}),
            "poblacion": data.get("poblacion_resumen", []),
            "completado": f.name == "estado_ga_final.json",
        }
    return estudios


def cargar_evaluaciones() -> list:
    """aprendizaje_ML.jsonl aplanado. Segmenta corridas cuando `generacion` reinicia."""
    regs = []
    prev_gen = None
    corrida = -1
    for i, line in enumerate(JSONL.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        md, cond = d["metadata"], d["condiciones"]
        gen = md["generacion"]
        if prev_gen is None or gen < prev_gen:
            corrida += 1
        prev_gen = gen
        res = d["resultados"]
        ang = sorted(res, key=lambda a: abs(float(a) - 4.0))[0]
        r = res[ang]
        par = md.get("parametrizacion") or {}
        conv = md.get("convergencia") or {}
        regs.append({
            "idx": i, "corrida": corrida, "gen": gen, "ind": md["individuo"],
            "ts": d["timestamp"], "Re": cond.get("Re"), "dx": cond.get("dx_min"),
            "iters": cond.get("iteraciones_cfd"), "alpha": float(ang),
            "fitness": md.get("fitness"), "base": md.get("perfil_base"),
            "cl": r.get("cl"), "cd": r.get("cd"), "ld": r.get("ld"),
            "perfil": np.asarray(d["perfil"], dtype=float),
            "cl_std": conv.get("cl_std_mean"), "cd_std": conv.get("cd_std_mean"),
            "n_samples": conv.get("n_samples"),
            **{p: par.get(p) for p in PARAMS},
        })
    return regs


N_RESAMPLE = 128


def remuestrear(perfil: np.ndarray, n: int = N_RESAMPLE) -> np.ndarray:
    """Contorno cerrado -> vector [x(s), y(s)] de longitud fija, parametrizado
    por longitud de arco normalizada. Los perfiles del proyecto tienen entre 33
    y 161 puntos según la semilla, así que sin esto no son comparables."""
    P = perfil if perfil.ndim == 2 else np.c_[np.linspace(0, 1, len(perfil)), perfil]
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1])))]
    if d[-1] <= 0:
        return np.zeros(2 * n)
    s = d / d[-1]
    t = np.linspace(0, 1, n)
    return np.r_[np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])]


def col(regs, k):
    return np.array([np.nan if r[k] is None else r[k] for r in regs], dtype=float)


def fmt_re(re):
    return f"Re={re:.0e}".replace("e+0", "e")


# ----------------------------------------------------------------------
# A. Dinámica del GA (historial por generación)
# ----------------------------------------------------------------------
def fig_convergencia(estudios, metricas):
    n = len(estudios)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.1 * nrows), squeeze=False)
    for ax, (tag, e) in zip(axes.ravel(), estudios.items()):
        h = e["hist"]
        g = [x["gen"] for x in h]
        ax.plot(g, [x["mejor"] for x in h], "-o", ms=3, lw=1.6, color="#c0392b", label="mejor")
        ax.plot(g, [x["media"] for x in h], "-s", ms=2.5, lw=1.2, color="#2980b9", label="media")
        ax.plot(g, [x["peor"] for x in h], "--", lw=1.0, color="#7f8c8d", label="peor")
        ax.fill_between(g, [x["peor"] for x in h], [x["mejor"] for x in h], color="#2980b9", alpha=0.08)
        ax.set_title(f"{tag}  ({'completado' if e['completado'] else 'parcial'})", fontsize=9)
        ax.set_xlabel("generación"); ax.set_ylabel("$L/D$")
        ax.legend(fontsize=7, loc="lower right")
        metricas.setdefault("estudios", {})[tag] = {
            "generaciones": len(h),
            "ld_inicial": h[0]["mejor"], "ld_final": h[-1]["mejor"],
            "mejora_pct": 100.0 * (h[-1]["mejor"] / h[0]["mejor"] - 1.0) if h[0]["mejor"] else None,
            "completado": e["completado"],
        }
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Convergencia del algoritmo genético por estudio", fontsize=12)
    fig.savefig(OUT / "01_convergencia.png"); plt.close(fig)


def fig_convergencia_normalizada(estudios):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for tag, e in estudios.items():
        h = e["hist"]
        g = np.array([x["gen"] for x in h], float)
        best = np.array([x["mejor"] for x in h], float)
        a1.plot(g, best / best[0], "-o", ms=3, lw=1.4, label=tag)
        # ganancia marginal por generación
        a2.plot(g[1:], np.diff(best) / best[0] * 100, "-o", ms=3, lw=1.2, label=tag)
    a1.set_xlabel("generación"); a1.set_ylabel("$L/D$ / $L/D_{gen\\,0}$")
    a1.set_title("Convergencia normalizada")
    a1.axhline(1.0, color="k", lw=0.6, ls=":")
    a2.set_xlabel("generación"); a2.set_ylabel("ganancia marginal [% de $L/D_{gen\\,0}$]")
    a2.set_title("Rendimientos decrecientes")
    a2.axhline(0.0, color="k", lw=0.6, ls=":")
    a1.legend(fontsize=7, ncol=2)
    fig.savefig(OUT / "02_convergencia_normalizada.png"); plt.close(fig)


def fig_presion_selectiva(estudios, metricas):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for tag, e in estudios.items():
        h = e["hist"]
        g = [x["gen"] for x in h]
        best = np.array([x["mejor"] for x in h], float)
        mean = np.array([x["media"] for x in h], float)
        worst = np.array([x["peor"] for x in h], float)
        rango = np.maximum(best - worst, 1e-9)
        a1.plot(g, (best - mean) / rango, "-o", ms=3, lw=1.3, label=tag)
        a2.plot(g, (best - worst) / np.maximum(mean, 1e-9), "-o", ms=3, lw=1.3, label=tag)
        metricas.setdefault("estudios", {}).setdefault(tag, {})["presion_selectiva_media"] = \
            round(float(np.nanmean((best - mean) / rango)), 4)
    a1.set_xlabel("generación"); a1.set_ylabel(r"$(f_{max}-\bar{f})\,/\,(f_{max}-f_{min})$")
    a1.set_title("Presión selectiva\n(alto = élite despegada de la población)")
    a2.set_xlabel("generación"); a2.set_ylabel(r"$(f_{max}-f_{min})\,/\,\bar{f}$")
    a2.set_title("Dispersión relativa de fitness\n(baja = convergencia prematura)")
    a1.legend(fontsize=7, ncol=2)
    fig.savefig(OUT / "03_presion_selectiva.png"); plt.close(fig)


def fig_coste(estudios, metricas):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for tag, e in estudios.items():
        h = e["hist"]
        t = np.cumsum([x.get("tiempo_seg", 0.0) for x in h]) / 3600.0
        best = np.array([x["mejor"] for x in h], float)
        ev = np.cumsum([x.get("evaluados", 0) for x in h])
        a1.plot(t, best, "-o", ms=3, lw=1.4, label=tag)
        a2.plot([x["gen"] for x in h], np.array([x.get("tiempo_seg", 0.0) for x in h]) / 3600.0,
                "-o", ms=3, lw=1.2, label=tag)
        metricas.setdefault("estudios", {}).setdefault(tag, {}).update({
            "gpu_horas": round(float(t[-1]), 2),
            "evaluaciones_cfd": int(ev[-1]),
            "gpu_horas_por_eval": round(float(t[-1] / max(ev[-1], 1)), 4),
            "ld_ganado_por_gpu_hora": round(float((best[-1] - best[0]) / max(t[-1], 1e-9)), 4),
        })
    a1.set_xlabel("GPU-horas acumuladas"); a1.set_ylabel("mejor $L/D$")
    a1.set_title("Coste computacional vs calidad")
    a2.set_xlabel("generación"); a2.set_ylabel("GPU-horas por generación")
    a2.set_title("Coste por generación")
    a1.legend(fontsize=7, ncol=2)
    fig.savefig(OUT / "04_coste_computacional.png"); plt.close(fig)


def fig_surrogate(estudios, metricas):
    """Cribado del surrogate (RandomForest) y del filtro geométrico."""
    tags, ia, geom_ev, geom_rep, rep, fb, ev = [], [], [], [], [], [], []
    for tag, e in estudios.items():
        h = e["hist"]
        s = lambda k: sum(x.get(k, 0) or 0 for x in h)
        tags.append(tag)
        ev.append(s("evaluados")); ia.append(s("descartados_ia"))
        geom_ev.append(s("descartados_geom_eval")); geom_rep.append(s("descartados_geom_repro"))
        rep.append(s("reparados_param_repro")); fb.append(s("fallback_legacy_repro"))

    x = np.arange(len(tags))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    props = np.array(ia, float) / np.maximum(np.array(ev, float) + np.array(ia, float), 1)
    a1.bar(x, 100 * props, color="#16a085")
    a1.set_xticks(x); a1.set_xticklabels(tags, rotation=25, ha="right")
    a1.set_ylabel("% candidatos cribados por el surrogate")
    a1.set_title("Ahorro de CFD por el modelo sustituto")
    for xi, p, n in zip(x, props, ia):
        a1.text(xi, 100 * p, f"{100*p:.0f}%\n({n})", ha="center", va="bottom", fontsize=7)

    w = 0.27
    a2.bar(x - w, geom_rep, w, label="descartes geom. (reproducción)", color="#c0392b")
    a2.bar(x, rep, w, label="reparados (paramétrico)", color="#f39c12")
    a2.bar(x + w, fb, w, label="fallback legacy", color="#7f8c8d")
    a2.set_xticks(x); a2.set_xticklabels(tags, rotation=25, ha="right")
    a2.set_ylabel("nº de individuos"); a2.set_title("Salud del operador geométrico")
    a2.legend(fontsize=7)
    fig.savefig(OUT / "05_surrogate_y_operadores.png"); plt.close(fig)

    for t, e_, i_, gr, r_, f_ in zip(tags, ev, ia, geom_rep, rep, fb):
        metricas.setdefault("estudios", {}).setdefault(t, {}).update({
            "cribados_surrogate": int(i_),
            "frac_cribado_surrogate": round(float(i_ / max(e_ + i_, 1)), 4),
            "descartes_geom_reproduccion": int(gr),
            "reparados_parametrico": int(r_),
            "fallback_legacy": int(f_),
        })


# ----------------------------------------------------------------------
# B. Espacio de diseño (aprendizaje_ML.jsonl)
# ----------------------------------------------------------------------
def fig_diversidad(regs, metricas):
    """Diversidad genética por generación, dentro de cada corrida larga."""
    corridas = {}
    for r in regs:
        corridas.setdefault(r["corrida"], []).append(r)
    largas = {c: v for c, v in corridas.items()
              if len({r["gen"] for r in v}) >= 6 and len(v) >= 60}
    if not largas:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for c, v in sorted(largas.items()):
        re = v[0]["Re"]
        gens = sorted({r["gen"] for r in v})
        div_geom, div_par = [], []
        for g in gens:
            sub = [r for r in v if r["gen"] == g]
            P = np.stack([remuestrear(r["perfil"]) for r in sub])
            div_geom.append(float(np.mean(np.std(P, axis=0))))
            M = np.stack([[r[p] if r[p] is not None else np.nan for p in PARAMS] for r in sub])
            sd = np.nanstd(M, axis=0) / np.maximum(np.abs(np.nanmean(M, axis=0)), 1e-9)
            div_par.append(float(np.nanmean(sd)))
        lbl = f"corrida {c} · {fmt_re(re)}"
        a1.plot(gens, np.array(div_geom) / max(div_geom[0], 1e-12), "-o", ms=3, lw=1.3, label=lbl)
        a2.plot(gens, div_par, "-o", ms=3, lw=1.3, label=lbl)
        metricas.setdefault("diversidad", {})[f"corrida_{c}"] = {
            "Re": re, "generaciones": len(gens),
            "diversidad_geom_final_rel": round(float(div_geom[-1] / max(div_geom[0], 1e-12)), 4),
            "cv_parametros_final": round(float(div_par[-1]), 4),
        }
    a1.set_xlabel("generación"); a1.set_ylabel("dispersión geométrica normalizada")
    a1.set_title("Diversidad en el espacio de coordenadas\n(caída fuerte = convergencia prematura)")
    a2.set_xlabel("generación"); a2.set_ylabel("CV medio de los 6 parámetros")
    a2.set_title("Diversidad en el espacio paramétrico")
    a1.legend(fontsize=6, ncol=2)
    fig.savefig(OUT / "06_diversidad.png"); plt.close(fig)


def fig_pca(regs, metricas):
    """PCA sobre las ordenadas del perfil; color = L/D. Un panel por Re."""
    from sklearn.decomposition import PCA

    Y = np.stack([remuestrear(r["perfil"]) for r in regs])
    ld = col(regs, "ld")
    res = np.array([r["Re"] for r in regs], float)

    pca = PCA(n_components=4).fit(Y)
    Z = pca.transform(Y)
    metricas["pca"] = {
        "n_muestras": int(Y.shape[0]), "n_dim": int(Y.shape[1]),
        "varianza_explicada": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "varianza_acumulada_pc2": round(float(pca.explained_variance_ratio_[:2].sum()), 4),
    }

    res_u = sorted(set(res[np.isfinite(res)]))
    fig, axes = plt.subplots(1, len(res_u) + 1, figsize=(4.6 * (len(res_u) + 1), 3.8),
                             squeeze=False, layout="constrained")
    axes = axes.ravel()
    for ax, rv in zip(axes, res_u):
        m = (res == rv) & np.isfinite(ld) & (ld > 0)
        sc = ax.scatter(Z[m, 0], Z[m, 1], c=ld[m], s=9, cmap="viridis", alpha=0.85, lw=0)
        j = np.argmax(np.where(m, ld, -np.inf))
        ax.plot(Z[j, 0], Z[j, 1], "*", ms=16, mfc="#e74c3c", mec="k", mew=0.5)
        ax.set_title(f"{fmt_re(rv)}  (n={m.sum()})", fontsize=9)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        plt.colorbar(sc, ax=ax, label="$L/D$")
    ax = axes[-1]
    ax.plot(np.arange(1, 5), np.cumsum(pca.explained_variance_ratio_) * 100, "-o", color="#2c3e50")
    ax.set_xlabel("componente principal"); ax.set_ylabel("varianza acumulada [%]")
    ax.set_title("Dimensionalidad efectiva del espacio")
    ax.set_xticks(np.arange(1, 5)); ax.set_ylim(0, 101)
    fig.suptitle("Espacio de diseño proyectado (PCA sobre las ordenadas del perfil) · ★ = óptimo", fontsize=11)
    fig.savefig(OUT / "07_pca_espacio_diseno.png"); plt.close(fig)


def _heatmap(ax, x, y, z, nbins, xlabel, ylabel, vmin=None, vmax=None):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    if x.size < 20:
        ax.axis("off"); return None
    xe = np.linspace(np.percentile(x, 1), np.percentile(x, 99), nbins + 1)
    ye = np.linspace(np.percentile(y, 1), np.percentile(y, 99), nbins + 1)
    s, _, _ = np.histogram2d(x, y, bins=[xe, ye], weights=z)
    c, _, _ = np.histogram2d(x, y, bins=[xe, ye])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(c > 0, s / c, np.nan)
    im = ax.pcolormesh(xe, ye, mean.T, cmap="magma", shading="auto", vmin=vmin, vmax=vmax)
    j = np.nanargmax(z)
    ax.plot(x[j], y[j], "*", ms=15, mfc="#2ecc71", mec="k", mew=0.6)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(False)
    return im


def fig_heatmaps(regs, metricas):
    """Mapas de calor L/D sobre pares de parámetros geométricos interpretables."""
    res = np.array([r["Re"] for r in regs], float)
    ld = col(regs, "ld")
    pares = [("espesor_max", "espesor_pos"), ("camber_max", "camber_pos"),
             ("espesor_max", "camber_max"), ("le_radius", "te_gap")]
    res_u = [rv for rv in sorted(set(res[np.isfinite(res)])) if (res == rv).sum() >= 150]

    fig, axes = plt.subplots(len(res_u), len(pares),
                             figsize=(4.0 * len(pares), 3.2 * len(res_u)),
                             squeeze=False, layout="constrained")
    for i, rv in enumerate(res_u):
        m = (res == rv) & np.isfinite(ld) & (ld > 0)
        vmax = np.nanpercentile(ld[m], 99)
        vmin = np.nanpercentile(ld[m], 10)
        for j, (px, py) in enumerate(pares):
            ax = axes[i][j]
            im = _heatmap(ax, col(regs, px)[m], col(regs, py)[m], ld[m],
                          nbins=14, xlabel=PARAM_LABEL[px], ylabel=PARAM_LABEL[py],
                          vmin=vmin, vmax=vmax)
            if im is not None:
                plt.colorbar(im, ax=ax, label="$L/D$ medio")
            if j == 0:
                ax.set_ylabel(f"{fmt_re(rv)}\n{PARAM_LABEL[py]}")
    fig.suptitle("Mapas de calor del espacio de diseño · ★ = mejor individuo", fontsize=12)
    fig.savefig(OUT / "08_heatmaps_geometria.png"); plt.close(fig)

    # óptimos por Re en el espacio paramétrico
    opt = {}
    for rv in res_u:
        m = (res == rv) & np.isfinite(ld)
        j = np.argmax(np.where(m, ld, -np.inf))
        opt[fmt_re(rv)] = {"ld": round(float(ld[j]), 3),
                           **{p: (round(float(regs[j][p]), 5) if regs[j][p] is not None else None)
                              for p in PARAMS}}
    metricas["optimos_por_re"] = opt


def fig_correlaciones(regs):
    res = np.array([r["Re"] for r in regs], float)
    res_u = [rv for rv in sorted(set(res[np.isfinite(res)])) if (res == rv).sum() >= 150]
    campos = PARAMS + ["cl", "cd", "ld"]
    fig, axes = plt.subplots(1, len(res_u), figsize=(5.0 * len(res_u), 4.4),
                             squeeze=False, layout="constrained")
    for ax, rv in zip(axes.ravel(), res_u):
        m = res == rv
        M = np.stack([col(regs, c)[m] for c in campos])
        good = np.all(np.isfinite(M), axis=0)
        C = np.corrcoef(M[:, good])
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        lbl = [PARAM_LABEL.get(c, c.upper()) for c in campos]
        ax.set_xticks(range(len(campos))); ax.set_xticklabels(lbl, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(campos))); ax.set_yticklabels(lbl, fontsize=7)
        for a in range(len(campos)):
            for b in range(len(campos)):
                ax.text(b, a, f"{C[a,b]:.2f}", ha="center", va="center", fontsize=6,
                        color="w" if abs(C[a, b]) > 0.5 else "k")
        ax.set_title(f"{fmt_re(rv)}  (n={good.sum()})", fontsize=9)
        ax.grid(False)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Correlación de Pearson: geometría vs aerodinámica", fontsize=12)
    fig.savefig(OUT / "09_correlaciones.png"); plt.close(fig)


def _pareto(cd, cl):
    """Índices no dominados minimizando Cd y maximizando Cl."""
    order = np.argsort(cd)
    best_cl, keep = -np.inf, []
    for i in order:
        if cl[i] > best_cl:
            keep.append(i); best_cl = cl[i]
    return np.array(keep, dtype=int)


def fig_pareto(regs, metricas):
    res = np.array([r["Re"] for r in regs], float)
    cl, cd, ld = col(regs, "cl"), col(regs, "cd"), col(regs, "ld")
    res_u = [rv for rv in sorted(set(res[np.isfinite(res)])) if (res == rv).sum() >= 150]
    fig, axes = plt.subplots(1, len(res_u), figsize=(4.8 * len(res_u), 3.9),
                             squeeze=False, layout="constrained")
    pareto_out = {}
    for ax, rv in zip(axes.ravel(), res_u):
        m = (res == rv) & np.isfinite(cl) & np.isfinite(cd) & (cd > 0)
        sc = ax.scatter(cd[m], cl[m], c=ld[m], s=8, cmap="viridis", alpha=0.7, lw=0)
        idx = np.where(m)[0]
        p = idx[_pareto(cd[m], cl[m])]
        o = np.argsort(cd[p])
        ax.plot(cd[p][o], cl[p][o], "-", color="#e74c3c", lw=1.8, label=f"frente de Pareto (n={len(p)})")
        ax.set_xlabel("$C_d$"); ax.set_ylabel("$C_l$")
        ax.set_title(f"{fmt_re(rv)}  (n={m.sum()})", fontsize=9)
        ax.set_xscale("log")
        ax.legend(fontsize=7)
        plt.colorbar(sc, ax=ax, label="$L/D$")
        pareto_out[fmt_re(rv)] = {
            "n_evaluaciones": int(m.sum()), "n_pareto": int(len(p)),
            "cd_min": round(float(np.nanmin(cd[m])), 6),
            "cl_max": round(float(np.nanmax(cl[m])), 6),
            "ld_max": round(float(np.nanmax(ld[m])), 4),
        }
    metricas["pareto"] = pareto_out
    fig.suptitle("Nube de diseños evaluados y frente de Pareto $C_l$–$C_d$", fontsize=12)
    fig.savefig(OUT / "10_pareto_cl_cd.png"); plt.close(fig)


def fig_historial_global(regs):
    res = np.array([r["Re"] for r in regs], float)
    ld = col(regs, "ld")
    idx = np.arange(len(regs))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4))
    for rv in sorted(set(res[np.isfinite(res)])):
        m = (res == rv) & np.isfinite(ld)
        a1.scatter(idx[m], ld[m], s=5, alpha=0.5, lw=0, label=fmt_re(rv))
    a1.set_xlabel("evaluación CFD (orden cronológico)"); a1.set_ylabel("$L/D$")
    a1.set_title(f"Historial completo del proyecto ({len(regs)} evaluaciones CFD)")
    a1.legend(fontsize=7, markerscale=2)

    a2.hist([ld[np.isfinite(ld) & (res == rv)] for rv in sorted(set(res[np.isfinite(res)]))],
            bins=40, stacked=True, label=[fmt_re(rv) for rv in sorted(set(res[np.isfinite(res)]))])
    a2.set_xlabel("$L/D$"); a2.set_ylabel("nº de evaluaciones")
    a2.set_title("Distribución de la calidad evaluada")
    a2.legend(fontsize=7)
    fig.savefig(OUT / "11_historial_global.png"); plt.close(fig)


def fig_ld_vs_re(metricas):
    f = STUDY / "refine_todos.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    filas = sorted(d.values(), key=lambda v: v["re"])
    re = np.array([v["re"] for v in filas], float)
    ld_ex = np.array([v["exploracion"]["ld"] for v in filas], float)
    ld_rf = np.array([v["refinado"]["ld"] for v in filas], float)
    cl = np.array([v["refinado"]["cl"] for v in filas], float)
    cd = np.array([v["refinado"]["cd"] for v in filas], float)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.semilogx(re, ld_ex, "--s", ms=6, lw=1.3, color="#95a5a6", label="exploración (malla gruesa)")
    a1.semilogx(re, ld_rf, "-o", ms=7, lw=2.0, color="#c0392b", label="refinado (dx=0.002)")
    a1.set_xlabel("Reynolds"); a1.set_ylabel("$L/D$ del óptimo")
    a1.set_title("Rendimiento del óptimo frente a Reynolds")
    a1.legend(fontsize=8)
    a2.loglog(re, cl, "-o", ms=6, label="$C_l$", color="#2980b9")
    a2.loglog(re, cd, "-s", ms=6, label="$C_d$", color="#e67e22")
    a2.set_xlabel("Reynolds"); a2.set_ylabel("coeficiente")
    a2.set_title("Descomposición $C_l$ / $C_d$ del óptimo")
    a2.legend(fontsize=8)
    fig.savefig(OUT / "12_ld_vs_reynolds.png"); plt.close(fig)
    metricas["ld_vs_re"] = {
        fmt_re(r): {"ld_exploracion": float(a), "ld_refinado": float(b),
                    "delta_refinado_pct": round(float(100 * (b / a - 1)), 2)}
        for r, a, b in zip(re, ld_ex, ld_rf)
    }


def fig_perfiles(estudios, regs):
    """Geometría de los ganadores por estudio, comparada con la semilla NACA0012."""
    base = None
    for r in regs:
        if r["base"] and "0012" in str(r["base"]):
            base = r["perfil"]; break
    fig, ax = plt.subplots(figsize=(9, 3.2))
    if base is not None:
        b = base if base.ndim == 2 else np.c_[np.linspace(0, 1, len(base)), base]
        ax.plot(b[:, 0], b[:, 1], "k--", lw=1.0, label="semilla NACA 0012")
    for tag, e in estudios.items():
        g = e["mejor"].get("genes")
        if not g:
            continue
        G = np.asarray(g, float)
        if G.ndim == 1:
            G = G.reshape(-1, 2)
        ax.plot(G[:, 0], G[:, 1], lw=1.5, label=f"{tag}  (L/D={e['mejor'].get('fitness', float('nan')):.2f})")
    ax.set_aspect("equal"); ax.set_xlabel("$x/c$"); ax.set_ylabel("$y/c$")
    ax.set_title("Geometrías óptimas por estudio")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig(OUT / "13_perfiles_ganadores.png"); plt.close(fig)


def fig_convergencia_cfd(regs, metricas):
    """Calidad estadística de las evaluaciones CFD (cl_std/cd_std registrados)."""
    cls, cds = col(regs, "cl_std"), col(regs, "cd_std")
    cl, cd = col(regs, "cl"), col(regs, "cd")
    idx = np.arange(len(regs))
    with np.errstate(invalid="ignore", divide="ignore"):
        rcl = np.abs(cls / cl) * 100
        rcd = np.abs(cds / cd) * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4))
    a1.scatter(idx, rcl, s=4, alpha=0.4, lw=0, label="$\\sigma_{C_l}/|C_l|$")
    a1.scatter(idx, rcd, s=4, alpha=0.4, lw=0, color="#e67e22", label="$\\sigma_{C_d}/|C_d|$")
    a1.set_yscale("symlog", linthresh=1e-3)
    a1.set_xlabel("evaluación CFD (cronológico)"); a1.set_ylabel("dispersión relativa en la ventana [%]")
    a1.set_title("Estabilidad temporal de cada evaluación")
    a1.legend(fontsize=7, markerscale=2)

    cero = float(np.mean(cls[np.isfinite(cls)] == 0.0)) if np.isfinite(cls).any() else np.nan
    finite = np.isfinite(rcl) & np.isfinite(rcd)
    a2.hist(rcl[finite & (rcl > 0)], bins=np.logspace(-4, 2, 50), alpha=0.6, label="$C_l$")
    a2.hist(rcd[finite & (rcd > 0)], bins=np.logspace(-4, 2, 50), alpha=0.6, label="$C_d$")
    a2.set_xscale("log")
    a2.set_xlabel("dispersión relativa [%]"); a2.set_ylabel("nº de evaluaciones")
    a2.set_title(f"Distribución (σ=0 exacto en {100*cero:.1f}% de los casos:\nartefacto del relleno, corregido)")
    a2.legend(fontsize=7)
    fig.savefig(OUT / "14_estabilidad_cfd.png"); plt.close(fig)
    metricas["estabilidad_cfd"] = {
        "frac_std_cero_exacto": round(cero, 4),
        "cl_disp_rel_mediana_pct": round(float(np.nanmedian(rcl[rcl > 0])), 4),
        "cd_disp_rel_mediana_pct": round(float(np.nanmedian(rcd[rcd > 0])), 4),
    }


def fig_surrogate_offline(regs, metricas):
    """`usar_ia=False` en todos los estudios: el surrogate nunca cribó nada.
    Se cuantifica aquí, offline, cuánto CFD habría ahorrado y con qué error."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold

    res = np.array([r["Re"] for r in regs], float)
    ld = col(regs, "ld")
    rv = 1e5  # el Re con muestra suficiente para validar
    m = (res == rv) & np.isfinite(ld) & (ld > 0)
    if m.sum() < 200:
        return
    idx = np.where(m)[0]
    X = np.stack([np.r_[remuestrear(regs[i]["perfil"]),
                        [regs[i][p] if regs[i][p] is not None else 0.0 for p in PARAMS]]
                  for i in idx])
    y = ld[idx]

    pred = np.full(y.shape, np.nan)
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=0).split(X):
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0)
        rf.fit(X[tr], y[tr])
        pred[te] = rf.predict(X[te])

    err = pred - y
    r2 = 1.0 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)
    mae = float(np.mean(np.abs(err)))

    # Política de cribado: descartar si L/D predicho < percentil q de lo ya visto.
    qs = np.arange(10, 91, 5)
    ahorro, falsos_neg = [], []
    for q in qs:
        umbral = np.percentile(y, q)
        descarta = pred < umbral
        ahorro.append(100.0 * descarta.mean())
        # falso negativo = descartado pero realmente estaba en el 10% superior
        elite = y >= np.percentile(y, 90)
        falsos_neg.append(100.0 * (descarta & elite).sum() / max(elite.sum(), 1))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.scatter(y, pred, s=7, alpha=0.4, lw=0, color="#2980b9")
    lim = [np.nanmin(y), np.nanmax(y)]
    a1.plot(lim, lim, "k--", lw=1.0)
    a1.set_xlabel("$L/D$ real (CFD)"); a1.set_ylabel("$L/D$ predicho (RF, 5-fold CV)")
    a1.set_title(f"Precisión del modelo sustituto @ {fmt_re(rv)}\n$R^2$={r2:.3f}   MAE={mae:.2f}")

    a2.plot(ahorro, falsos_neg, "-o", ms=4, color="#c0392b")
    for q, a, f in zip(qs, ahorro, falsos_neg):
        if q % 20 == 0:
            a2.annotate(f"q{q}", (a, f), fontsize=7, xytext=(3, 3), textcoords="offset points")
    a2.set_xlabel("CFD evitado [%]"); a2.set_ylabel("élite (top 10%) perdida [%]")
    a2.set_title("Compromiso del cribado por surrogate\n(nunca activado: usar_ia=False)")
    fig.savefig(OUT / "15_surrogate_offline.png"); plt.close(fig)

    # punto de operación razonable: máximo ahorro con <5% de élite perdida
    ok = [(a, f, q) for a, f, q in zip(ahorro, falsos_neg, qs) if f < 5.0]
    mejor = max(ok) if ok else None
    metricas["surrogate_offline"] = {
        "Re": rv, "n_muestras": int(len(y)),
        "r2_cv": round(float(r2), 4), "mae_cv": round(mae, 4),
        "usar_ia_en_estudios": False,
        "punto_operacion": ({"cfd_evitado_pct": round(mejor[0], 1),
                             "elite_perdida_pct": round(mejor[1], 2),
                             "percentil_umbral": int(mejor[2])} if mejor else None),
    }


# ----------------------------------------------------------------------
def informe(metricas, regs, estudios):
    L = ["# Métricas del optimizador genético", ""]
    L.append(f"Evaluaciones CFD registradas: **{len(regs)}**  |  estudios con historial: **{len(estudios)}**")
    tot_h = sum(v.get("gpu_horas", 0) for v in metricas.get("estudios", {}).values())
    L.append(f"Coste acumulado en los estudios con historial: **{tot_h:.1f} GPU-horas**")
    L += ["", "## Estudios", "",
          "| estudio | gens | L/D inicial | L/D final | mejora | GPU-h | evals | cribado surrogate | presión selectiva |",
          "|---|---|---|---|---|---|---|---|---|"]
    for t, v in metricas.get("estudios", {}).items():
        L.append("| {} | {} | {:.3f} | {:.3f} | {:+.1f}% | {:.1f} | {} | {:.0f}% | {:.2f} |".format(
            t, v.get("generaciones", 0), v.get("ld_inicial", float("nan")),
            v.get("ld_final", float("nan")), v.get("mejora_pct") or 0.0,
            v.get("gpu_horas", 0.0), v.get("evaluaciones_cfd", 0),
            100 * (v.get("frac_cribado_surrogate") or 0.0),
            v.get("presion_selectiva_media") or float("nan")))

    if "optimos_por_re" in metricas:
        L += ["", "## Óptimo geométrico por Reynolds", "",
              "| Re | L/D | $t/c$ | $x_t/c$ | $f/c$ | $x_f/c$ |", "|---|---|---|---|---|---|"]
        for k, v in metricas["optimos_por_re"].items():
            L.append("| {} | {:.2f} | {} | {} | {} | {} |".format(
                k, v["ld"], v.get("espesor_max"), v.get("espesor_pos"),
                v.get("camber_max"), v.get("camber_pos")))

    if "pca" in metricas:
        p = metricas["pca"]
        L += ["", "## Dimensionalidad del espacio de diseño", "",
              f"- {p['n_dim']} coordenadas por perfil, {p['n_muestras']} muestras.",
              f"- PC1+PC2 explican **{100*p['varianza_acumulada_pc2']:.1f}%** de la varianza geométrica."]

    if "estabilidad_cfd" in metricas:
        e = metricas["estabilidad_cfd"]
        L += ["", "## Estabilidad de las evaluaciones CFD", "",
              f"- σ = 0 exacto en **{100*e['frac_std_cero_exacto']:.1f}%** de las evaluaciones históricas.",
              "  Artefacto del relleno de la cola del vector al disparar el early-stop "
              "(`Simulador2D.py`); corregido: ahora se trunca a las muestras reales.",
              f"- Dispersión relativa mediana: $C_l$ {e['cl_disp_rel_mediana_pct']:.3f}%, "
              f"$C_d$ {e['cd_disp_rel_mediana_pct']:.3f}%."]

    if "surrogate_offline" in metricas:
        s = metricas["surrogate_offline"]
        L += ["", "## Modelo sustituto (RandomForest)", "",
              "- **`usar_ia=False` en los 6 estudios**: el surrogate nunca cribó ningún candidato.",
              f"- Reentrenado offline @ {fmt_re(s['Re'])} con {s['n_muestras']} muestras: "
              f"$R^2$ (5-fold CV) = **{s['r2_cv']:.3f}**, MAE = {s['mae_cv']:.2f}."]
        if s.get("punto_operacion"):
            po = s["punto_operacion"]
            L.append(f"- Punto de operación viable: evitar **{po['cfd_evitado_pct']:.0f}% del CFD** "
                     f"perdiendo solo {po['elite_perdida_pct']:.1f}% de la élite "
                     f"(umbral = percentil {po['percentil_umbral']}).")

    if "ld_vs_re" in metricas:
        L += ["", "## Efecto del refinado de malla sobre el óptimo", "",
              "| Re | L/D exploración | L/D refinado | Δ |", "|---|---|---|---|"]
        for k, v in metricas["ld_vs_re"].items():
            L.append(f"| {k} | {v['ld_exploracion']:.3f} | {v['ld_refinado']:.3f} | {v['delta_refinado_pct']:+.1f}% |")

    L += ["", "## Figuras", ""]
    for f in sorted(OUT.glob("*.png")):
        L.append(f"- `{f.name}`")
    (OUT / "INFORME.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    estudios = cargar_estudios()
    regs = cargar_evaluaciones()
    print(f"estudios={len(estudios)}  evaluaciones={len(regs)}")
    metricas = {}

    fig_convergencia(estudios, metricas)
    fig_convergencia_normalizada(estudios)
    fig_presion_selectiva(estudios, metricas)
    fig_coste(estudios, metricas)
    fig_surrogate(estudios, metricas)
    fig_diversidad(regs, metricas)
    fig_pca(regs, metricas)
    fig_heatmaps(regs, metricas)
    fig_correlaciones(regs)
    fig_pareto(regs, metricas)
    fig_historial_global(regs)
    fig_ld_vs_re(metricas)
    fig_perfiles(estudios, regs)
    fig_convergencia_cfd(regs, metricas)
    fig_surrogate_offline(regs, metricas)

    (OUT / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    informe(metricas, regs, estudios)
    print(f"OK -> {OUT}")
    for f in sorted(OUT.iterdir()):
        print("  ", f.name)


if __name__ == "__main__":
    sys.exit(main())

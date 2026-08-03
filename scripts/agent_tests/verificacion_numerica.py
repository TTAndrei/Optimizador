"""
Verificación numérica del solver — dos bloques, ambos con GPU:

  --reeval   Re-evalúa los ganadores (Re=1e3/1e5/1e6/1e7) con la estadística ya
             corregida (sin relleno de la cola del vector), reportando CI95 real
             y el estado del criterio de convergencia Cl/Cd.

  --gci      Convergencia de malla en dx = 0.008/0.004/0.002/0.001: orden
             observado p, extrapolación de Richardson y GCI de Roache para Cl/Cd.

  --polar    Polar multi-ángulo (α = 0..10) del ganador Re=1e5, base para la
             comparación con XFOIL/experimental y para justificar el multipunto.

Uso:
    .venv/bin/python scripts/agent_tests/verificacion_numerica.py --reeval --gci

Salida: results/verificacion_numerica/  (JSON + figuras + INFORME.md)
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

OUT = os.path.join(ROOT, "results", "verificacion_numerica")
STUDY = os.path.join(ROOT, "results", "convergence_study")
os.makedirs(OUT, exist_ok=True)

ALPHA = 4.0
CFL = 0.5
T_TARGET = 8.0
U_FAC = 1.6

# Mismos sim_extra_params que el estudio (transición SA-BC validada)
EXTRA = {"transition_model": "sa_bc", "freestream_Tu": 0.1}

GANADORES = {
    "re1e3": (1e3, f"{STUDY}/mixto_re1e3/NACA_0012_sharp_Re1000_a4.0_LD5.01.dat"),
    "re1e5": (1e5, f"{STUDY}/mixto_masgen/NACA_0012_sharp_Re100000_a4.0_LD18.18.dat"),
    "re1e6": (1e6, f"{STUDY}/mixto_re1e6/"),
    "re1e7": (1e7, f"{STUDY}/mixto_re1e7/"),
}

DX_GCI = [0.008, 0.004, 0.002, 0.001]

# A Re bajo el flujo desprende y el L/D oscila: la serie de calib_re1e3 sigue a un
# 7-9% del asintótico en t=8-9 y no baja del 1% hasta t≈15. Cada caso necesita su
# propio horizonte temporal; T_TARGET=8 solo vale de Re=1e5 hacia arriba.
T_TARGET_CASO = {"re1e3": 16.0}


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def iters_for(dx, t_target=T_TARGET):
    n = t_target * U_FAC / (CFL * dx)
    return int(-(-n // 1000) * 1000)


def resolver_dat(spec):
    """Acepta un .dat directo o un directorio del que se toma el mejor ganador."""
    if spec.endswith(".dat") and os.path.exists(spec):
        return spec
    if os.path.isdir(spec):
        cands = [f for f in os.listdir(spec) if f.endswith(".dat") and "_LD" in f]
        if not cands:
            return None
        cands.sort(key=lambda f: float(f.rsplit("_LD", 1)[1][:-4]), reverse=True)
        return os.path.join(spec, cands[0])
    return None


def simular(dat, re, dx, alpha=ALPHA, iters=None, extra=None, dump=None):
    cfg = dict(RunGA.CONFIG)
    cfg["sim_extra_params"] = {**cfg.get("sim_extra_params", {}), **EXTRA,
                               **criterio_calibrado(), **(extra or {})}
    if dump:
        cfg["dump_series_path"] = dump
    cfg["nu"] = 1.0 / re
    cfg["v0x"] = 1.0
    cfg["chord"] = 1.0
    cfg["rho"] = 1.0
    cfg["CFL"] = CFL
    cfg["dx_min"] = dx
    cfg["simulacion_iteraciones"] = iters or iters_for(dx)
    t0 = time.time()
    res = RunGA.simular_perfil(dat, alpha, cfg)
    if res is not None:
        res["wall_s"] = round(time.time() - t0, 1)
        res["dx"] = dx
        res["iters_programadas"] = cfg["simulacion_iteraciones"]
        res["alpha"] = alpha
        res["Re"] = re
    return res


# ----------------------------------------------------------------------
# Bloque 1 — re-evaluación de ganadores
# ----------------------------------------------------------------------
def reeval():
    salida = {}
    for tag, (re, spec) in GANADORES.items():
        dat = resolver_dat(spec)
        if dat is None:
            _log(f"{tag}: sin .dat, se omite ({spec})")
            continue
        dx = 0.002
        tt = T_TARGET_CASO.get(tag, T_TARGET)
        n_it = iters_for(dx, tt)
        _log(f"{tag}: Re={re:.0e} dx={dx} t_target={tt} iters={n_it}  {os.path.basename(dat)}")
        r = simular(dat, re, dx, iters=n_it)
        if r is None:
            _log(f"{tag}: simulación fallida")
            continue
        r["dat"] = os.path.relpath(dat, ROOT)
        salida[tag] = r
        _log(f"  -> L/D={r['ld']:.3f} ± {r.get('ld_ci95')}  "
             f"conv={r.get('converged_clcd')}  n={r.get('n_samples')}  "
             f"σCl={r.get('cl_std')}  σCd={r.get('cd_std')}  ({r['wall_s']}s)")
        with open(f"{OUT}/reeval_ganadores.json", "w") as f:
            json.dump(salida, f, indent=2, ensure_ascii=False)
    return salida


def fig_reeval(datos):
    if not datos:
        return
    ref = {}
    p = f"{STUDY}/refine_todos.json"
    if os.path.exists(p):
        d = json.load(open(p))
        clave = {"exprimir_re1e5": "re1e5", "re1e3": "re1e3", "re1e6": "re1e6", "re1e7": "re1e7"}
        for k, v in d.items():
            ref[clave.get(k, k)] = v["refinado"]["ld"]

    tags = list(datos)
    x = np.arange(len(tags))
    ld = np.array([datos[t]["ld"] for t in tags], float)
    err = np.array([datos[t].get("ld_ci95") or 0.0 for t in tags], float)
    old = np.array([ref.get(t, np.nan) for t in tags], float)

    fig, ax = plt.subplots(figsize=(7.5, 4), layout="constrained")
    ax.bar(x - 0.2, old, 0.38, color="#95a5a6", label="estudio previo (σ artefactual)")
    ax.bar(x + 0.2, ld, 0.38, yerr=err, capsize=4, color="#c0392b",
           label="re-evaluación (CI95 real)")
    ax.set_xticks(x); ax.set_xticklabels(tags)
    ax.set_ylabel("$L/D$"); ax.set_title("Ganadores: estadística corregida vs previa")
    ax.legend(fontsize=8)
    fig.savefig(f"{OUT}/reeval_ganadores.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Bloque 2 — GCI de Roache
# ----------------------------------------------------------------------
def orden_observado(f1, f2, f3, r21, r32):
    """Orden aparente p resolviendo la ecuación implícita de Celik et al. (2008).
    f1 = malla más fina. Devuelve (p, eps21, eps32)."""
    e21, e32 = f2 - f1, f3 - f2
    s = np.sign(e32 / e21) if abs(e21) > 1e-30 else 1.0
    p = 2.0
    for _ in range(200):
        q = np.log((r21 ** p - s) / (r32 ** p - s))
        p_new = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
        if not np.isfinite(p_new):
            break
        if abs(p_new - p) < 1e-10:
            p = p_new
            break
        p = 0.5 * p + 0.5 * p_new
    return float(p), float(e21), float(e32)


def gci_triplete(f1, f2, f3, h1, h2, h3, Fs=1.25):
    """f1/h1 = malla fina. Devuelve dict con p, extrapolado y GCI fino/grueso."""
    r21, r32 = h2 / h1, h3 / h2
    p, e21, e32 = orden_observado(f1, f2, f3, r21, r32)
    f_ext = (r21 ** p * f1 - f2) / (r21 ** p - 1.0) if abs(r21 ** p - 1.0) > 1e-30 else np.nan
    ea21 = abs((f1 - f2) / f1) if abs(f1) > 1e-30 else np.nan
    eext21 = abs((f_ext - f1) / f_ext) if abs(f_ext) > 1e-30 else np.nan
    gci21 = Fs * ea21 / (r21 ** p - 1.0) if abs(r21 ** p - 1.0) > 1e-30 else np.nan
    gci32 = Fs * abs((f2 - f3) / f2) / (r32 ** p - 1.0) if abs(r32 ** p - 1.0) > 1e-30 else np.nan
    return {
        "p_observado": round(p, 4),
        "f_fina": round(float(f1), 6), "f_media": round(float(f2), 6),
        "f_gruesa": round(float(f3), 6),
        "f_extrapolado_richardson": round(float(f_ext), 6),
        "error_aprox_ea21_pct": round(100 * float(ea21), 4),
        "error_extrapolado_pct": round(100 * float(eext21), 4),
        "GCI_fina_pct": round(100 * float(gci21), 4),
        "GCI_gruesa_pct": round(100 * float(gci32), 4),
        "convergencia_monotona": bool(np.sign(e21) == np.sign(e32)),
    }


def gci(perfil=None, re=1e5):
    if perfil is None:
        perfil = resolver_dat(GANADORES["re1e5"][1]) or "profiles/NACA_0012_sharp"
    crudo = {}
    path = f"{OUT}/gci_crudo.json"
    if os.path.exists(path):
        crudo = json.load(open(path))
    for dx in DX_GCI:
        k = f"dx{dx}"
        if k in crudo:
            _log(f"GCI {k}: ya calculado, se salta")
            continue
        # Sin parada anticipada: si unas mallas paran antes que otras el tiempo
        # físico deja de ser el mismo y el orden observado p mide esa diferencia,
        # no la discretización espacial.
        _log(f"GCI {k}: iters={iters_for(dx)} (sin early-stop) ...")
        r = simular(perfil, re, dx, extra={"stop_on_clcd_convergence": False})
        if r is None:
            _log(f"GCI {k}: fallida")
            continue
        crudo[k] = r
        _log(f"  -> Cl={r['cl']:.5f} Cd={r['cd']:.6f} L/D={r['ld']:.3f} ({r['wall_s']}s)")
        with open(path, "w") as f:
            json.dump(crudo, f, indent=2, ensure_ascii=False)

    dxs = sorted([float(k[2:]) for k in crudo])
    if len(dxs) < 3:
        _log("GCI: hacen falta al menos 3 mallas")
        return None

    # Triplete más fino disponible (fino, medio, grueso)
    h1, h2, h3 = dxs[0], dxs[1], dxs[2]
    out = {"perfil": os.path.relpath(perfil, ROOT), "Re": re, "alpha": ALPHA,
           "mallas_dx": dxs, "factor_refinamiento": [round(h2 / h1, 3), round(h3 / h2, 3)]}
    for mag in ("cl", "cd", "ld"):
        f = [crudo[f"dx{h}"][mag] for h in (h1, h2, h3)]
        out[mag] = gci_triplete(f[0], f[1], f[2], h1, h2, h3)
    out["crudo"] = {f"dx{h}": {m: crudo[f"dx{h}"][m] for m in
                               ("cl", "cd", "ld", "cl_std", "cd_std", "n_samples",
                                "converged_clcd", "wall_s", "iters_programadas")}
                    for h in dxs}
    with open(f"{OUT}/gci.json", "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    _log(f"GCI: p(Cl)={out['cl']['p_observado']:.2f}  p(Cd)={out['cd']['p_observado']:.2f}  "
         f"GCI_fina(Cl)={out['cl']['GCI_fina_pct']:.2f}%  GCI_fina(Cd)={out['cd']['GCI_fina_pct']:.2f}%")
    fig_gci(out, crudo)
    return out


def fig_gci(out, crudo):
    dxs = np.array(out["mallas_dx"], float)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), layout="constrained")
    for ax, mag, lbl in zip(axes, ("cl", "cd", "ld"), ("$C_l$", "$C_d$", "$L/D$")):
        y = np.array([crudo[f"dx{d}"][mag] for d in dxs], float)
        ax.plot(dxs, y, "-o", ms=6, color="#2980b9", label="simulado")
        fx = out[mag]["f_extrapolado_richardson"]
        ax.axhline(fx, color="#c0392b", ls="--", lw=1.3,
                   label=f"Richardson $h\\to0$ = {fx:.4g}")
        g = out[mag]["GCI_fina_pct"] / 100.0
        ax.fill_between([dxs.min(), dxs.max()],
                        y[0] * (1 - g), y[0] * (1 + g),
                        color="#c0392b", alpha=0.12,
                        label=f"banda GCI = ±{100*g:.2f}%")
        ax.set_xscale("log")
        ax.set_xlabel("$\\Delta x$ mínimo"); ax.set_ylabel(lbl)
        ax.set_title(f"{lbl} — orden observado $p$={out[mag]['p_observado']:.2f}", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"Convergencia de malla (GCI de Roache) · Re={out['Re']:.0e}, α={out['alpha']}°",
                 fontsize=11)
    fig.savefig(f"{OUT}/gci.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Bloque 2b — recalibrado del criterio de parada anticipada
# ----------------------------------------------------------------------
CALIB_CASOS = ["re1e5", "re1e3"]
CALIB_ITERS = 30000
CALIB_DX = 0.002

# El solver comprueba la banda cada clcd_check_every iteraciones; con guardado=50
# eso son 2 muestras. Se reproduce el mismo paso en el barrido offline.
CHECK_EVERY_SAMPLES = 2

REJILLA = {
    "tol_abs": [0.005, 0.002, 0.001],
    "tol_rel": [0.02, 0.01, 0.005, 0.002],
    "window_conv_time": [0.5, 1.0, 2.0],
    "min_t": [1.0, 3.0, 5.0],
}


def calibrar_correr():
    """Corre los casos de referencia con early-stop desactivado y vuelca las series."""
    meta_path = f"{OUT}/calib_series.json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    for tag in CALIB_CASOS:
        if tag in meta and os.path.exists(os.path.join(ROOT, meta[tag]["npz"])):
            _log(f"calib {tag}: serie ya volcada, se salta")
            continue
        re, spec = GANADORES[tag]
        dat = resolver_dat(spec)
        if dat is None:
            _log(f"calib {tag}: sin .dat, se omite")
            continue
        npz = f"{OUT}/calib_{tag}.npz"
        _log(f"calib {tag}: Re={re:.0e} dx={CALIB_DX} iters={CALIB_ITERS} sin early-stop ...")
        r = simular(dat, re, CALIB_DX, iters=CALIB_ITERS, dump=npz,
                    extra={"stop_on_clcd_convergence": False, "stop_on_convergence": False})
        if r is None:
            _log(f"calib {tag}: fallida")
            continue
        r["npz"] = os.path.relpath(npz, ROOT)
        r["dat"] = os.path.relpath(dat, ROOT)
        meta[tag] = r
        _log(f"  -> L/D={r['ld']:.3f} n={r['n_samples']} ({r['wall_s']}s)")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def _media_ventana(v):
    """Mismo promediado que RunGA.simular_perfil: último 20%, mínimo 10 muestras."""
    n = len(v)
    w = min(n, max(n // 5, 10))
    return float(np.mean(v[-w:]))


def _replay(t, cl, cd, tol_abs, tol_rel, window, min_t):
    """Reproduce el criterio del solver sobre la serie completa.
    Devuelve el índice de muestra en el que habría parado, o None."""
    from Simulador2D import _detect_series_convergence
    n = len(t)
    for k in range(10, n + 1, CHECK_EVERY_SAMPLES):
        if t[k - 1] < min_t:
            continue
        _, c1 = _detect_series_convergence(cl[:k], t[k - 1], tol_abs, tol_rel, window)
        if not c1:
            continue
        _, c2 = _detect_series_convergence(cd[:k], t[k - 1], tol_abs, tol_rel, window)
        if c2:
            return k
    return None


def calibrar_barrido(meta):
    series = {}
    for tag, m in meta.items():
        d = np.load(os.path.join(ROOT, m["npz"]))
        series[tag] = (d["t"].astype(float), d["cl"].astype(float), d["cd"].astype(float))
    if not series:
        _log("calib: sin series, nada que barrer")
        return None

    # Valor asintótico = promedio de la cola de la serie completa
    ref = {tag: {"cl": _media_ventana(cl), "cd": _media_ventana(cd)}
           for tag, (t, cl, cd) in series.items()}
    for tag in ref:
        ref[tag]["ld"] = ref[tag]["cl"] / ref[tag]["cd"]

    filas = []
    for ta in REJILLA["tol_abs"]:
        for tr in REJILLA["tol_rel"]:
            for w in REJILLA["window_conv_time"]:
                for mt in REJILLA["min_t"]:
                    fila = {"tol_abs": ta, "tol_rel": tr, "window_conv_time": w, "min_t": mt,
                            "casos": {}}
                    errs, costes, todos_paran = [], [], True
                    for tag, (t, cl, cd) in series.items():
                        k = _replay(t, cl, cd, ta, tr, w, mt)
                        if k is None:
                            todos_paran = False
                            fila["casos"][tag] = {"paro": False}
                            errs.append(0.0)
                            costes.append(1.0)
                            continue
                        ld_k = _media_ventana(cl[:k]) / _media_ventana(cd[:k])
                        e = abs(ld_k - ref[tag]["ld"]) / abs(ref[tag]["ld"])
                        fila["casos"][tag] = {
                            "paro": True, "muestra": int(k), "t": round(float(t[k - 1]), 3),
                            "ld": round(ld_k, 4), "err_pct": round(100 * e, 3),
                            "coste_rel": round(k / len(t), 3),
                        }
                        errs.append(e)
                        costes.append(k / len(t))
                    fila["err_max_pct"] = round(100 * max(errs), 3)
                    fila["coste_medio"] = round(float(np.mean(costes)), 3)
                    fila["todos_paran"] = todos_paran
                    filas.append(fila)

    # Elegir: error máximo < 2% y, entre esas, la más barata.
    validas = [f for f in filas if f["todos_paran"] and f["err_max_pct"] < 2.0]
    elegida = min(validas, key=lambda f: f["coste_medio"]) if validas else \
        min(filas, key=lambda f: (f["err_max_pct"], f["coste_medio"]))

    out = {
        "referencia": {tag: {k: round(v, 6) for k, v in r.items()} for tag, r in ref.items()},
        "n_muestras_serie": {tag: int(len(s[0])) for tag, s in series.items()},
        "criterio_actual": next(
            (f for f in filas if (f["tol_abs"], f["tol_rel"], f["window_conv_time"], f["min_t"])
             == (0.005, 0.02, 0.5, 1.0)), None),
        "elegido": elegida,
        "cumplen_2pct": len(validas),
        "barrido": filas,
    }
    with open(f"{OUT}/calibracion_earlystop.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    act = out["criterio_actual"]
    if act:
        _log(f"criterio ACTUAL: err_max={act['err_max_pct']}%  coste={act['coste_medio']}")
    _log(f"criterio ELEGIDO: tol_abs={elegida['tol_abs']} tol_rel={elegida['tol_rel']} "
         f"window={elegida['window_conv_time']} min_t={elegida['min_t']} "
         f"-> err_max={elegida['err_max_pct']}%  coste={elegida['coste_medio']}")
    fig_calib(series, ref, out)
    return out


def fig_calib(series, ref, out):
    n = len(series)
    fig, axes = plt.subplots(2, n, figsize=(6.2 * n, 6.6), squeeze=False, layout="constrained")
    for j, (tag, (t, cl, cd)) in enumerate(series.items()):
        ld = cl / np.where(np.abs(cd) > 1e-12, cd, np.nan)
        for i, (y, lbl, r) in enumerate(((cl, "$C_l$", ref[tag]["cl"]),
                                         (ld, "$L/D$", ref[tag]["ld"]))):
            ax = axes[i][j]
            ax.plot(t, y, lw=0.9, color="#2c3e50")
            ax.axhline(r, color="#27ae60", ls="--", lw=1.2, label=f"asintótico = {r:.4g}")
            ax.axhspan(r * 0.98, r * 1.02, color="#27ae60", alpha=0.12, label="±2%")
            for crit, col, nm in ((out["criterio_actual"], "#c0392b", "actual"),
                                  (out["elegido"], "#e67e22", "elegido")):
                c = (crit or {}).get("casos", {}).get(tag, {})
                if c.get("paro"):
                    ax.axvline(c["t"], color=col, lw=1.4,
                               label=f"parada {nm}: t={c['t']} ({c['err_pct']}%)")
            ax.set_xlabel("tiempo convectivo"); ax.set_ylabel(lbl)
            ax.legend(fontsize=7); ax.grid(alpha=0.2)
            if i == 0:
                ax.set_title(tag, fontsize=11)
    fig.suptitle("Calibrado del criterio de parada anticipada", fontsize=12)
    fig.savefig(f"{OUT}/calibracion_earlystop.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def calibrar():
    return calibrar_barrido(calibrar_correr())


def criterio_calibrado():
    """sim_extra_params con el criterio de parada validado, si existe.

    Lo produce criterio_parada.py, que lo elige por validación cruzada sobre 20
    series completas y lo verifica sobre 5 que no intervienen en la elección.
    Sustituye al barrido de calibracion_earlystop.json, que se ajustaba sobre 2
    series y no generalizaba.
    """
    p = f"{OUT}/criterio_parada.json"
    if not os.path.exists(p):
        return {}
    c = json.load(open(p))["criterio"]
    return {
        "clcd_tol_drift": c["tol_drift"],
        "clcd_tol_noise": c["tol_noise"],
        "clcd_window_conv_time": c["window"],
        "clcd_n_sostenido": c["n_sostenido"],
        "clcd_min_t_fisico_before_check": c["min_t"],
    }


# ----------------------------------------------------------------------
# Bloque 3 — polar del ganador (justifica el multipunto)
# ----------------------------------------------------------------------
def polar(angulos=(0.0, 2.0, 4.0, 6.0, 8.0, 10.0), re=1e5, dx=0.002):
    dat = resolver_dat(GANADORES["re1e5"][1])
    base = "profiles/NACA_0012_sharp"
    path = f"{OUT}/polar.json"
    out = json.load(open(path)) if os.path.exists(path) else {}
    for nombre, p in (("ganador_re1e5", dat), ("NACA_0012", base)):
        out.setdefault(nombre, {})
        for a in angulos:
            k = f"{a:.1f}"
            if k in out[nombre]:
                continue
            _log(f"polar {nombre} α={a}° ...")
            r = simular(p, re, dx, alpha=a)
            if r is None:
                continue
            out[nombre][k] = r
            _log(f"  -> Cl={r['cl']:.4f} Cd={r['cd']:.5f} L/D={r['ld']:.2f}")
            with open(path, "w") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
    fig_polar(out)
    return out


def repolar(t_conv_min=5.0, re=1e5, dx=0.002):
    """Repite sin parada anticipada los puntos de la polar que cortaron pronto.

    El criterio recalibrado se ajustó sobre dos casos y no generaliza: en la
    polar del NACA0012 cortó α=6° en t=1.75 con 10 muestras, que es el mismo
    fallo que este trabajo corrigió. Un punto medido en transitorio no es
    comparable con XFOIL ni con los demás ángulos de su propia curva.
    """
    path = f"{OUT}/polar.json"
    if not os.path.exists(path):
        _log("repolar: falta polar.json")
        return None
    out = json.load(open(path))
    sospechosos = []
    for nombre, d in out.items():
        for k, v in d.items():
            if v.get("_sin_earlystop"):
                continue
            # Solo es sospechoso si la parada llegó a dispararse y lo hizo pronto.
            # converged_clcd=False significa que corrió el presupuesto entero, que
            # es justamente lo que se quiere: ese punto ya es válido.
            if not v.get("converged_clcd"):
                continue
            tc = v.get("t_conv_clcd")
            if tc is None or tc < t_conv_min:
                sospechosos.append((nombre, k, tc, v.get("n_samples")))
    if not sospechosos:
        _log("repolar: todos los puntos con t_conv suficiente")
        return out

    _log(f"repolar: {len(sospechosos)} puntos cortaron en t<{t_conv_min} -> "
         + ", ".join(f"{n} α={k} (t={tc}, n={ns})" for n, k, tc, ns in sospechosos))
    dat_map = {"ganador_re1e5": resolver_dat(GANADORES["re1e5"][1]),
               "NACA_0012": "profiles/NACA_0012_sharp"}
    for nombre, k, _tc, _ns in sospechosos:
        p = dat_map.get(nombre)
        if p is None:
            continue
        a = float(k)
        # α=0 en perfil simétrico da Cl=0 por simetría: converge de inmediato y
        # su t_conv corto no indica transitorio.
        if nombre == "NACA_0012" and abs(a) < 1e-9:
            _log(f"repolar: {nombre} α=0 se mantiene (simetría, Cl=0 exacto)")
            out[nombre][k]["_sin_earlystop"] = True
            continue
        _log(f"repolar {nombre} α={a}° sin early-stop ...")
        r = simular(p, re, dx, alpha=a,
                    extra={"stop_on_clcd_convergence": False, "stop_on_convergence": False})
        if r is None:
            _log("   fallida")
            continue
        r["_sin_earlystop"] = True
        r["_previo_earlystop"] = {m: out[nombre][k].get(m)
                                  for m in ("cl", "cd", "ld", "t_conv_clcd", "n_samples")}
        prev = out[nombre][k]["ld"]
        out[nombre][k] = r
        _log(f"   -> L/D {prev:.2f} → {r['ld']:.2f} "
             f"({100*(r['ld']-prev)/abs(prev):+.1f}%)  n={r['n_samples']}  ({r['wall_s']}s)")
        with open(path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    fig_polar(out)
    return out


def fig_polar(out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), layout="constrained")
    for nombre, d in out.items():
        a = np.array(sorted(float(k) for k in d))
        cl = np.array([d[f"{x:.1f}"]["cl"] for x in a])
        cd = np.array([d[f"{x:.1f}"]["cd"] for x in a])
        ld = np.array([d[f"{x:.1f}"]["ld"] for x in a])
        ecl = np.array([d[f"{x:.1f}"].get("cl_ci95") or 0.0 for x in a])
        axes[0].errorbar(a, cl, yerr=ecl, fmt="-o", ms=5, capsize=3, label=nombre)
        axes[1].plot(a, cd, "-o", ms=5, label=nombre)
        axes[2].plot(a, ld, "-o", ms=5, label=nombre)
    for ax, lbl in zip(axes, ("$C_l$", "$C_d$", "$L/D$")):
        ax.set_xlabel(r"$\alpha$ [°]"); ax.set_ylabel(lbl); ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    axes[2].axvspan(2, 6, color="#f39c12", alpha=0.15)
    axes[2].set_title("banda del multipunto α∈{2,4,6}", fontsize=9)
    fig.suptitle("Polar del óptimo frente a la semilla — Re=1e5", fontsize=11)
    fig.savefig(f"{OUT}/polar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Bloque 4 — validación externa contra XFOIL
# ----------------------------------------------------------------------
XFOIL_DIR = os.path.join(ROOT, "data", "ComparativasReales")


def cargar_xfoil(nombre="NACA0012", re=1e5):
    """Polar de airfoiltools (XFOIL, Ncrit=9). La cabecera real empieza en la
    línea que arranca por 'Alpha'; encima hay metadatos de la corrida."""
    import pandas as pd
    lbl = "100k" if abs(re - 1e5) < 1 else f"{int(re/1e6)}M"
    path = os.path.join(XFOIL_DIR, f"{nombre}_{lbl}_Xfoil.csv")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        h = next((i for i, l in enumerate(f) if l.lstrip().startswith("Alpha")), None)
    if h is None:
        return None
    d = pd.read_csv(path, skiprows=list(range(h)), header=0, skipinitialspace=True)
    d.columns = d.columns.str.strip()
    d = d.dropna(subset=["Alpha", "Cl", "Cd"]).sort_values("Alpha").reset_index(drop=True)
    d["LD"] = d["Cl"] / d["Cd"].replace(0, np.nan)
    return d


def validacion_externa(re=1e5):
    """Compara la polar simulada del NACA0012 con la de XFOIL al mismo Re.

    Solo el NACA0012 es comparable: el ganador del GA es una geometría propia
    sin referencia externa. XFOIL corre con Ncrit=9 (transición libre), que es
    el escenario que el modelo SA-BC del solver intenta reproducir.
    """
    p = f"{OUT}/polar.json"
    if not os.path.exists(p):
        _log("validación: falta polar.json")
        return None
    pol = json.load(open(p))
    if "NACA_0012" not in pol:
        _log("validación: la polar del NACA0012 aún no está calculada")
        return None
    xf = cargar_xfoil("NACA0012", re)
    if xf is None:
        _log("validación: sin CSV de XFOIL")
        return None

    sim = pol["NACA_0012"]
    a = np.array(sorted(float(k) for k in sim))
    filas = []
    for x in a:
        k = f"{x:.1f}"
        cl_x = float(np.interp(x, xf["Alpha"], xf["Cl"]))
        cd_x = float(np.interp(x, xf["Alpha"], xf["Cd"]))
        ld_x = cl_x / cd_x if abs(cd_x) > 1e-12 else np.nan
        filas.append({
            "alpha": float(x),
            "cl_sim": sim[k]["cl"], "cl_xfoil": round(cl_x, 4),
            "cd_sim": sim[k]["cd"], "cd_xfoil": round(cd_x, 5),
            "ld_sim": sim[k]["ld"], "ld_xfoil": round(ld_x, 3),
            "cl_err_abs": round(sim[k]["cl"] - cl_x, 4),
            "cd_err_pct": round(100 * (sim[k]["cd"] - cd_x) / cd_x, 2) if cd_x else None,
        })

    cl_s = np.array([f["cl_sim"] for f in filas]); cl_x = np.array([f["cl_xfoil"] for f in filas])
    cd_s = np.array([f["cd_sim"] for f in filas]); cd_x = np.array([f["cd_xfoil"] for f in filas])
    # Pendiente de sustentación en la zona lineal (α ≤ 6°), el descriptor que
    # menos depende del modelo de transición.
    lin = np.array([f["alpha"] for f in filas]) <= 6.0
    def pend(y, x):
        return float(np.polyfit(x[lin], y[lin], 1)[0]) if lin.sum() >= 2 else float("nan")
    aa = np.array([f["alpha"] for f in filas])

    out = {
        "perfil": "NACA0012", "Re": re, "fuente_xfoil": "airfoiltools, Ncrit=9",
        "puntos": filas,
        "cl_mae": round(float(np.mean(np.abs(cl_s - cl_x))), 4),
        "cd_mae": round(float(np.mean(np.abs(cd_s - cd_x))), 5),
        "cd_err_medio_pct": round(float(np.mean(100 * (cd_s - cd_x) / cd_x)), 2),
        "cl_pendiente_sim": round(pend(cl_s, aa), 5),
        "cl_pendiente_xfoil": round(pend(cl_x, aa), 5),
        "cl_pendiente_teorica_2pi": round(2 * np.pi * np.pi / 180, 5),
    }
    with open(f"{OUT}/validacion_xfoil.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    _log(f"validación XFOIL: MAE(Cl)={out['cl_mae']}  MAE(Cd)={out['cd_mae']}  "
         f"Cd medio {out['cd_err_medio_pct']:+.1f}%  "
         f"dCl/dα sim={out['cl_pendiente_sim']} vs xfoil={out['cl_pendiente_xfoil']}")
    fig_validacion(out, xf)
    return out


def fig_validacion(out, xf):
    a = np.array([f["alpha"] for f in out["puntos"]])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), layout="constrained")
    for ax, ks, kx, lbl in ((axes[0], "cl_sim", "cl_xfoil", "$C_l$"),
                            (axes[1], "cd_sim", "cd_xfoil", "$C_d$"),
                            (axes[2], "ld_sim", "ld_xfoil", "$L/D$")):
        ys = [f[ks] for f in out["puntos"]]
        m = {"cl_xfoil": "Cl", "cd_xfoil": "Cd", "ld_xfoil": "LD"}[kx]
        ax.plot(xf["Alpha"], xf[m], "-", color="#7f8c8d", lw=1.2, alpha=0.7, label="XFOIL (curva)")
        ax.plot(a, [f[kx] for f in out["puntos"]], "s", ms=6, color="#2980b9", label="XFOIL")
        ax.plot(a, ys, "o-", ms=6, color="#c0392b", label="solver")
        ax.set_xlim(-0.5, 10.5)
        ax.set_xlabel(r"$\alpha$ [°]"); ax.set_ylabel(lbl)
        ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.suptitle(f"Validación externa — NACA0012, Re={out['Re']:.0e} (XFOIL Ncrit=9)", fontsize=11)
    fig.savefig(f"{OUT}/validacion_xfoil.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
def informe():
    L = ["# Verificación numérica", ""]
    p = f"{OUT}/gci.json"
    if os.path.exists(p):
        g = json.load(open(p))
        L += [f"## Convergencia de malla (GCI) — Re={g['Re']:.0e}, α={g['alpha']}°", "",
              f"Mallas: dx = {g['mallas_dx']}  |  factores de refinamiento r = {g['factor_refinamiento']}", "",
              "| magnitud | p observado | valor malla fina | Richardson h→0 | GCI fina | monótona |",
              "|---|---|---|---|---|---|"]
        for m, lbl in (("cl", "Cl"), ("cd", "Cd"), ("ld", "L/D")):
            v = g[m]
            L.append("| {} | {:.2f} | {:.5g} | {:.5g} | ±{:.2f}% | {} |".format(
                lbl, v["p_observado"], v["f_fina"], v["f_extrapolado_richardson"],
                v["GCI_fina_pct"], "sí" if v["convergencia_monotona"] else "**no**"))
        L += ["", "El GCI acota la incertidumbre de discretización de la malla más fina.", ""]

    p = f"{OUT}/reeval_ganadores.json"
    if os.path.exists(p):
        d = json.load(open(p))
        L += ["## Ganadores re-evaluados (estadística corregida)", "",
              "| caso | Re | L/D | CI95 | σCl | σCd | n | early-stop Cl/Cd | iters |",
              "|---|---|---|---|---|---|---|---|---|"]
        for t, v in d.items():
            L.append("| {} | {:.0e} | {:.3f} | ±{} | {} | {} | {} | {} | {} |".format(
                t, v["Re"], v["ld"], v.get("ld_ci95"), v.get("cl_std"), v.get("cd_std"),
                v.get("n_samples"), v.get("converged_clcd"), v.get("iters_efectivas")))
        L.append("")

    p = f"{OUT}/polar.json"
    if os.path.exists(p):
        d = json.load(open(p))
        L += ["## Polar", "", "| α | " + " | ".join(f"L/D {n}" for n in d) + " |",
              "|---|" + "---|" * len(d)]
        angs = sorted({float(k) for v in d.values() for k in v})
        for a in angs:
            fila = [f"{d[n][f'{a:.1f}']['ld']:.2f}" if f"{a:.1f}" in d[n] else "—" for n in d]
            L.append(f"| {a:.1f} | " + " | ".join(fila) + " |")
        L.append("")

    p = f"{OUT}/validacion_xfoil.json"
    if os.path.exists(p):
        v = json.load(open(p))
        L += [f"## Validación externa — {v['perfil']} vs XFOIL ({v['fuente_xfoil']})", "",
              f"MAE(Cl) = {v['cl_mae']}  |  MAE(Cd) = {v['cd_mae']}  |  "
              f"error medio de Cd = {v['cd_err_medio_pct']:+.1f}%", "",
              f"Pendiente dCl/dα (α≤6°): solver {v['cl_pendiente_sim']} · "
              f"XFOIL {v['cl_pendiente_xfoil']} · teoría 2π {v['cl_pendiente_teorica_2pi']} por grado", "",
              "| α | Cl sim | Cl XFOIL | Cd sim | Cd XFOIL | L/D sim | L/D XFOIL | ΔCd |",
              "|---|---|---|---|---|---|---|---|"]
        for f in v["puntos"]:
            L.append("| {alpha:.1f} | {cl_sim:.4f} | {cl_xfoil:.4f} | {cd_sim:.5f} | "
                     "{cd_xfoil:.5f} | {ld_sim:.2f} | {ld_xfoil:.2f} | {cd_err_pct:+.1f}% |".format(**f))
        L.append("")

    L += ["## Figuras", ""]
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".png"):
            L.append(f"- `{f}`")
    with open(f"{OUT}/INFORME.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reeval", action="store_true")
    ap.add_argument("--calibrar", action="store_true",
                    help="Series largas sin early-stop + barrido offline del criterio")
    ap.add_argument("--gci", action="store_true")
    ap.add_argument("--polar", action="store_true")
    ap.add_argument("--validacion", action="store_true",
                    help="Compara la polar del NACA0012 con XFOIL (sin GPU)")
    ap.add_argument("--repolar", action="store_true",
                    help="Repite sin early-stop los puntos de la polar cortados en transitorio")
    ap.add_argument("--re", type=float, default=1e5, help="Reynolds del estudio GCI/polar")
    args = ap.parse_args()
    if not (args.calibrar or args.reeval or args.gci or args.polar or args.validacion
            or args.repolar):
        ap.error("elige al menos --calibrar, --reeval, --gci, --polar, --repolar o --validacion")

    if args.calibrar:
        calibrar()
    if args.reeval:
        fig_reeval(reeval())
    if args.gci:
        gci(re=args.re)
    if args.polar:
        polar(re=args.re)
    if args.repolar:
        repolar(re=args.re)
    if args.validacion:
        validacion_externa(re=args.re)
    informe()
    _log(f"OK -> {OUT}")


if __name__ == "__main__":
    main()

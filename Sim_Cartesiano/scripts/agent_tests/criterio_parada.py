"""
SUPERADO por scripts/agent_tests/criterio_ci95.py -- se conserva porque documenta
el paso anterior, pero ya no produce el criterio en produccion y su rejilla usa la
puerta de ruido crudo (sigma/|media|), que es la que hubo que quitar.

El motivo: sigma/|media| mide la amplitud de la oscilacion de la estela, no la
incertidumbre del promedio, asi que con desprendimiento no abre nunca. La puerta
ahora es el CI95 de la media sobre N efectivo, y la ventana se ensancha sola.

Criterio de parada nuevo, ajustado y validado sobre series separadas.

El criterio actual (`_detect_series_convergence`) compara la señal con la media de
su propio último 15%. Pregunta "¿me parezco a mí mismo hace poco?", no "¿he dejado
de cambiar". Una deriva lenta y monótona lo pasa siempre, y por eso paraba en t≈1-2
midiendo flujo no desarrollado. Recalibrar sus tolerancias no arregla eso: se ajustó
sobre 2 series (error 0.82%) y al aplicarlo a casos nuevos dio hasta -23.7%.

El de aquí copia lo que hacen los solvers comerciales, que son tres condiciones a la
vez y no una:

  1. residual de campo por debajo de umbral (u y p; v no sirve, ver abajo),
  2. pendiente del monitor Cl/Cd plana -- medida como *tendencia*, no como
     desviación respecto de la media reciente,
  3. sostenido N chequeos consecutivos, con tope duro de iteraciones.

`change_v` se descarta: se normaliza por la norma de v, que es pequeña en un flujo
casi horizontal, así que vale ~0.5 permanentemente y domina cualquier maximo sin
significar nada.

    .venv/bin/python scripts/agent_tests/criterio_parada.py

Ajusta sobre la mitad de las series y valida sobre la otra mitad, que es lo que
faltó la vez anterior. Sin GPU.

Salida: results/verificacion_numerica/criterio_parada.{json,png}
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = os.path.join(ROOT, "results", "verificacion_numerica", "series_dx004")
OUT = os.path.join(ROOT, "results", "verificacion_numerica")

# Fracción final de la serie que define el valor asintótico de referencia.
COLA = 0.15
# El chequeo se hace cada CHECK muestras, como en el solver.
CHECK = 2

REJILLA = {
    # deriva relativa del monitor por tiempo convectivo
    "tol_drift": [0.02, 0.01, 0.005, 0.002],
    # ruido relativo dentro de la ventana
    "tol_noise": [0.05, 0.02, 0.01],
    # longitud de la ventana en tiempos convectivos
    "window": [1.0, 2.0, 3.0],
    # residual de campo (max de u y p)
    "tol_res": [np.inf, 0.02, 0.01, 0.005],
    # chequeos consecutivos que deben cumplirse
    "n_sostenido": [1, 3],
    # tiempo convectivo mínimo antes de admitir una parada. No es un parámetro
    # libre más: por debajo de t≈4-5 la burbuja laminar no ha reatacado y el L/D
    # todavía puede dar un salto, así que cualquier meseta anterior es falsa.
    "min_t": [0.0, 3.0, 5.0],
}


def cargar_series():
    """Cada .npz es (t, cl, cd, clcd, res[:, 3]) de una simulación completa."""
    out = []
    for p in sorted(glob.glob(os.path.join(SERIES, "serie_*.npz"))):
        z = np.load(p)
        t, cl, cd = z["t"], z["cl"], z["cd"]
        res = z["res"] if "res" in z else np.full((len(t), 3), np.nan)
        ok = np.isfinite(t) & np.isfinite(cl) & np.isfinite(cd) & (t > 0)
        if ok.sum() < 40:
            continue
        ld = np.where(np.abs(cd) > 1e-9, cl / np.where(np.abs(cd) > 1e-9, cd, 1), np.nan)
        out.append({
            "id": int(os.path.basename(p).split("_")[1].split(".")[0]),
            "t": t[ok], "cl": cl[ok], "cd": cd[ok], "ld": ld[ok], "res": res[ok],
        })
    return out


def asintotico(s):
    """Valor de referencia: media de la cola de la serie completa."""
    n = max(3, int(len(s["ld"]) * COLA))
    return float(np.mean(s["ld"][-n:]))


def evaluar(s, tol_drift, tol_noise, window, tol_res, n_sostenido, min_t):
    """Dónde habría parado este criterio y qué L/D habría reportado.

    Devuelve (t_parada, ld_reportado, fraccion_serie) o None si no para.
    """
    t, ld, res = s["t"], s["ld"], s["res"]
    # Residual de campo utilizable: u y p. v queda fuera (ver cabecera).
    r_up = np.nanmax(res[:, [0, 2]], axis=1) if res.ndim == 2 else np.full(len(t), np.nan)
    seguidos = 0
    for i in range(10, len(t), CHECK):
        if t[i] < min_t:
            continue
        w = t >= (t[i] - window)
        w &= np.arange(len(t)) <= i
        if w.sum() < 8:
            continue
        tw, lw = t[w], ld[w]
        media = np.mean(lw)
        if abs(media) < 1e-9:
            continue
        pend = np.polyfit(tw, lw, 1)[0]
        drift = abs(pend) / abs(media)          # cambio relativo por tiempo convectivo
        noise = np.std(lw) / abs(media)
        r_ok = True if not np.isfinite(tol_res) else (
            np.isfinite(r_up[i]) and r_up[i] < tol_res)
        if drift < tol_drift and noise < tol_noise and r_ok:
            seguidos += 1
            if seguidos >= n_sostenido:
                # Se reporta la media de la ventana, como hace el solver.
                return float(t[i]), float(media), float(t[i] / t[-1])
        else:
            seguidos = 0
    return None


def barrer(series, combos):
    """Error y coste de cada combinación.

    No parar no se penaliza como fallo: con tope duro de iteraciones -- que es lo
    que hacen los solvers comerciales y lo que ya hace este -- agotar el
    presupuesto devuelve el asintótico, con error ~0 y coste 1.0. Es seguro y
    caro, no incorrecto. Rechazarlo dejaba fuera criterios conservadores buenos.
    """
    ref = {s["id"]: asintotico(s) for s in series}
    filas = []
    for c in combos:
        errs, costes, paradas = [], [], 0
        for s in series:
            r = evaluar(s, **c)
            if r is None:
                errs.append(0.0)
                costes.append(1.0)
                continue
            paradas += 1
            _, ld, frac = r
            errs.append(100 * abs(ld - ref[s["id"]]) / abs(ref[s["id"]]))
            costes.append(frac)
        filas.append({**c, "err_max": max(errs), "err_medio": float(np.mean(errs)),
                      "coste_medio": float(np.mean(costes)),
                      "n_para": paradas, "n_series": len(series)})
    return filas


def main(seed=0):
    series = cargar_series()
    if len(series) < 8:
        print(f"[!] solo {len(series)} series en {SERIES}; hacen falta >=8")
        return None
    print(f"{len(series)} series cargadas")

    # Tres bloques: CV para elegir, test para la cifra final. El test no
    # interviene en ninguna decisión.
    #
    # Una sola partición ajuste/validación no basta para elegir: con 288
    # combinaciones, la más barata que cumple en un conjunto concreto está
    # explotando las particularidades de ese conjunto. Se selecciona por el peor
    # error en los pliegues retenidos, no por el error de ajuste.
    ids = sorted(s["id"] for s in series)
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(len(ids)))
    n_test = max(4, len(ids) // 4)
    test_ids = {ids[i] for i in perm[:n_test]}
    cv_ids = [ids[i] for i in perm[n_test:]]
    test = [s for s in series if s["id"] in test_ids]
    cv = [s for s in series if s["id"] in set(cv_ids)]
    print(f"CV   ({len(cv)}): {sorted(s['id'] for s in cv)}")
    print(f"test ({len(test)}): {sorted(test_ids)}")

    keys = list(REJILLA)
    combos = [dict(zip(keys, v)) for v in itertools.product(*(REJILLA[k] for k in keys))]

    K = 3
    pliegues = [set(cv_ids[i::K]) for i in range(K)]
    print(f"barriendo {len(combos)} combinaciones, {K} pliegues ...")

    # Para cada combinación: el peor error sobre los pliegues retenidos.
    puntuacion = []
    for c in combos:
        peor, coste = 0.0, []
        for pl in pliegues:
            held = [s for s in cv if s["id"] in pl]
            r = barrer(held, [c])[0]
            peor = max(peor, r["err_max"])
            coste.append(r["coste_medio"])
        puntuacion.append({**c, "err_max_cv": peor, "coste_cv": float(np.mean(coste))})

    validas = [f for f in puntuacion if f["err_max_cv"] < 2.0]
    pool = validas or puntuacion
    mejor = min(pool, key=lambda f: (f["coste_cv"] if validas else f["err_max_cv"]))
    print(f"\nCV: {len(validas)}/{len(puntuacion)} combinaciones con err_max<2% en retenidos")

    cfg = {k: mejor[k] for k in keys}
    print(f"elegido: {cfg}")
    print(f"  CV         err_max={mejor['err_max_cv']:.2f}%  coste={mejor['coste_cv']:.2f}")

    ajuste_ids = set(cv_ids)
    aj = barrer(cv, [cfg])[0]
    print(f"  todo CV    err_max={aj['err_max']:.2f}%  medio={aj['err_medio']:.2f}%  "
          f"coste={aj['coste_medio']:.2f}  para en {aj['n_para']}/{aj['n_series']}")

    # LA cifra: series que no han intervenido en ninguna decisión.
    gen = barrer(test, [cfg])[0]
    print(f"  TEST       err_max={gen['err_max']:.2f}%  medio={gen['err_medio']:.2f}%  "
          f"coste={gen['coste_medio']:.2f}  para en {gen['n_para']}/{gen['n_series']}")
    mejor = aj

    detalle = []
    for s in series:
        r = evaluar(s, **cfg)
        ref = asintotico(s)
        detalle.append({
            "id": s["id"], "conjunto": "cv" if s["id"] in ajuste_ids else "test",
            "ld_asintotico": round(ref, 4),
            "t_parada": round(r[0], 3) if r else None,
            "ld_reportado": round(r[1], 4) if r else None,
            "error_pct": round(100 * (r[1] - ref) / abs(ref), 2) if r else None,
            "coste": round(r[2], 3) if r else None,
            "t_final": round(float(s["t"][-1]), 2),
        })

    out = {
        "criterio": cfg,
        "cv": {k: mejor[k] for k in ("err_max", "err_medio", "coste_medio", "n_para")},
        "test": {k: gen[k] for k in ("err_max", "err_medio", "coste_medio", "n_para")},
        "ids_cv": sorted(ajuste_ids), "ids_test": sorted(test_ids),
        "n_series": len(series), "seed": seed,
        "detalle": detalle,
    }
    with open(os.path.join(OUT, "criterio_parada.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    figura(series, cfg, detalle)
    return out


def figura(series, cfg, detalle):
    d = {x["id"]: x for x in detalle}
    n = min(9, len(series))
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), layout="constrained")
    for ax, s in zip(axes.ravel(), series[:n]):
        x = d[s["id"]]
        ax.plot(s["t"], s["ld"], lw=0.9, color="#2980b9")
        ax.axhline(x["ld_asintotico"], ls="--", lw=1.0, color="#7f8c8d")
        if x["t_parada"]:
            ax.axvline(x["t_parada"], color="#c0392b", lw=1.4)
            ax.set_title(f"id {s['id']} ({x['conjunto']}) · err {x['error_pct']:+.1f}%",
                         fontsize=9)
        else:
            ax.set_title(f"id {s['id']} ({x['conjunto']}) · no para", fontsize=9)
        ax.set_xlabel("t convectivo", fontsize=8); ax.set_ylabel("L/D", fontsize=8)
        ax.grid(alpha=0.25); ax.tick_params(labelsize=7)
    fig.suptitle("Criterio de parada: " + ", ".join(f"{k}={v}" for k, v in cfg.items()),
                 fontsize=10)
    fig.savefig(os.path.join(OUT, "criterio_parada.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="partición ajuste/validación")
    main(ap.parse_args().seed)

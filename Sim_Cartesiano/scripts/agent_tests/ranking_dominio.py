"""
Robustez del ranking del GA frente al dominio.

Pregunta: el GA ordena individuos por L/D en el dominio estrecho 8x5. Si el sesgo
de dominio fuese dependiente del perfil, el orden cambiaria al ensanchar y el GA
estaria optimizando para el tunel. Si el sesgo es uniforme, se cancela al comparar
y el GA es valido tal cual, corrigiendo solo el valor absoluto del ganador.

Se reevalua el top-3 de la poblacion final en dos dominios, con el MISMO presupuesto
(13000 iters) para los seis: la campana original uso presupuestos distintos por
individuo (10000/7750/5950 iters, por el early-stop) y con n distinto las barras no
son comparables entre si.

  A  Lx=8  Ly=5  cx=2   el de la campana del GA
  B  Lx=16 Ly=10 cx=4   en la meseta medida del estudio de dominio

Dos preguntas separadas, y conviene no confundirlas:
  1. se conserva el ORDEN entre dominios
  2. las diferencias entre individuos superan su propio RUIDO
La 2 puede fallar aunque la 1 se cumpla: un orden estable pero no resuelto es
casualidad, no senal.

NO SE LANZA SOLO.
    .venv/bin/python scripts/agent_tests/ranking_dominio.py
    .venv/bin/python scripts/agent_tests/ranking_dominio.py --analisis
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "agent_tests"))
os.chdir(ROOT)

import numpy as np
import verificacion_numerica as vn

EPOCH = ("results/convergence_study/islands_tfg2/NACA_0012_sharp/epoch10")
ESTADO = f"{EPOCH}/estado_ga_final.json"
N_TOP = 3
ALPHA, RE, DX, ITERS = 4.0, 1e5, 0.002, 13000

# Solo se simula B. La referencia en A es la de la propia campana del GA, que es
# el ranking que el GA realmente uso para seleccionar. Ojo: esos puntos corrieron
# con presupuesto distinto por individuo (10000/7750/5950 iters), asi que el
# desplazamiento A->B mezcla dominio con presupuesto. El ORDEN si es comparable;
# el sesgo por individuo hay que leerlo con esa reserva.
DOMINIOS = {
    "B": {"Lx": 16.0, "Ly": 10.0, "cx": 4.0},
}
# top1 en A con presupuesto igualado (13000) ya existe: es el caso A de
# estudio_dominio, L/D=27.70 frente a 27.75 de la campana a 10000 iters.
REF_A = {"top1": 27.7455, "top2": 27.2286, "top3": 26.0306}
REF_A_CI = {"top1": 1.182, "top2": 0.8079, "top3": 0.1254}

OUT = os.path.join(ROOT, "results", "verificacion_numerica", "ranking_dominio")
PERF_DIR = f"{OUT}/perfiles"
JSON_PATH = f"{OUT}/ranking_dominio.json"
CSV_PATH = f"{OUT}/ranking_dominio.csv"
STOP = os.path.join(ROOT, "STOP_SIMULATION.trigger")


def extraer_top():
    """Escribe el top-N como .dat y devuelve [(id, ruta, fitness_ga, ld_ci95_ga)]."""
    os.makedirs(PERF_DIR, exist_ok=True)
    pob = [x for x in json.load(open(ESTADO))["poblacion_resumen"] if x.get("evaluado")]
    pob.sort(key=lambda x: x["fitness"], reverse=True)
    top = []
    for i, x in enumerate(pob[:N_TOP], 1):
        ind = f"top{i}"
        p = f"{PERF_DIR}/{ind}_LD{x['fitness']:.2f}.dat"
        if not os.path.exists(p):
            with open(p, "w") as f:
                f.write(f"{ind}_epoch10_LD{x['fitness']:.4f}\n")
                for gx, gy in x["genes"]:
                    f.write(f"  {gx:.6f}  {gy:.6f}\n")
        r = x["resultados"][f"{ALPHA}"]
        top.append((ind, p, x["fitness"], r.get("ld_ci95"), r.get("iters_efectivas")))
    return top


def escribir_csv(out):
    filas = []
    for k, r in sorted(out.items()):
        filas.append({"caso": k, **{c: r.get(c) for c in
                      ("individuo", "dominio", "Lx", "Ly", "cx", "fitness_ga",
                       "cl", "cl_ci95", "cd", "cd_ci95", "ld", "ld_ci95",
                       "n_samples", "converged_clcd", "t_conv_clcd",
                       "iters_efectivas", "cl_cp_discrepancy", "wall_s")}})
    if not filas:
        return
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader(); w.writerows(filas)
    vn._log(f"csv -> {CSV_PATH} ({len(filas)} filas)")


def analizar(out):
    inds = sorted({r["individuo"] for r in out.values()})
    doms = [d for d in DOMINIOS if any(r["dominio"] == d for r in out.values())]
    res = {"orden": {}, "pares": {}, "resueltos": {}}

    for d in doms:
        f = {r["individuo"]: r for r in out.values() if r["dominio"] == d}
        if len(f) < 2:
            continue
        orden = sorted(f, key=lambda i: f[i]["ld"], reverse=True)
        res["orden"][d] = orden
        print(f"\n--- dominio {d} ({DOMINIOS[d]['Lx']:g}x{DOMINIOS[d]['Ly']:g}) ---")
        for pos, i in enumerate(orden, 1):
            r = f[i]
            print(f"  {pos}. {i:6} L/D={r['ld']:7.2f} +-{r['ld_ci95'] or 0:5.2f}   "
                  f"Cl={r['cl']:.4f}  Cd={r['cd']:.5f}  n={r['n_samples']}  "
                  f"conv={r['converged_clcd']}")
        # Una diferencia esta resuelta si supera la suma en cuadratura de ambos CI95.
        pares = {}
        for a, b in itertools.combinations(orden, 2):
            dif = f[a]["ld"] - f[b]["ld"]
            s = float(np.hypot(f[a]["ld_ci95"] or 0, f[b]["ld_ci95"] or 0))
            pares[f"{a}-{b}"] = {"delta": round(dif, 4), "sigma": round(s, 4),
                                 "n_sigma": round(dif / s, 2) if s else None,
                                 "resuelto": bool(s and abs(dif) > s)}
        res["pares"][d] = pares
        for k, v in pares.items():
            print(f"     {k}: delta={v['delta']:+.3f}  sigma={v['sigma']:.3f}  "
                  f"{v['n_sigma']:+.2f}sigma  {'RESUELTO' if v['resuelto'] else 'dentro del ruido'}")

    if "B" in res["orden"]:
        oa = sorted(REF_A, key=lambda i: REF_A[i], reverse=True)
        ob = res["orden"]["B"]
        res["orden"]["A_campana"] = oa
        doms = ["A_campana", "B"]
        res["orden_conservado"] = oa == ob
        ra = {i: k for k, i in enumerate(oa)}; rb = {i: k for k, i in enumerate(ob)}
        n = len(oa)
        conc = sum(np.sign(ra[a] - ra[b]) == np.sign(rb[a] - rb[b])
                   for a, b in itertools.combinations(oa, 2))
        tot = n * (n - 1) // 2
        res["kendall_tau"] = round(2.0 * conc / tot - 1.0, 4) if tot else None
        print(f"\norden A (campana GA, 8x5): {' > '.join(oa)}")
        print(f"orden B (16x10, 13000 iters): {' > '.join(ob)}")
        print(f"conservado: {res['orden_conservado']}   "
              f"tau de Kendall={res['kendall_tau']}  ({conc}/{tot} pares concordantes)")
        # Sesgo de dominio por individuo: si es igual para todos, se cancela en el GA.
        print("\nsesgo por individuo (L/D en B menos L/D de la campana en A).")
        print("OJO: mezcla dominio con presupuesto, la campana no igualo iteraciones.")
        sesgos = {}
        for i in inds:
            b = next((r for r in out.values() if r["individuo"] == i and r["dominio"] == "B"), None)
            if b and i in REF_A:
                sesgos[i] = round(b["ld"] - REF_A[i], 4)
                print(f"  {i:6} {REF_A[i]:7.2f} -> {b['ld']:7.2f}   delta={sesgos[i]:+7.3f}")
        res["sesgo_por_individuo"] = sesgos
        if len(sesgos) >= 2:
            v = list(sesgos.values())
            res["sesgo_medio"] = round(float(np.mean(v)), 4)
            res["sesgo_dispersion"] = round(float(np.max(v) - np.min(v)), 4)
            print(f"  medio={res['sesgo_medio']:+.3f}   dispersion={res['sesgo_dispersion']:.3f}")
            print("  (dispersion pequena frente al medio -> sesgo uniforme -> el GA se salva)")

    json.dump(res, open(f"{OUT}/analisis.json", "w"), indent=2, ensure_ascii=False)
    escribir_csv(out)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analisis", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out = json.load(open(JSON_PATH)) if os.path.exists(JSON_PATH) else {}

    if a.analisis:
        analizar(out)
        return

    top = extraer_top()
    vn._log(f"top-{N_TOP} de {ESTADO}")
    for ind, p, fit, ci, it in top:
        vn._log(f"  {ind}: fitness_ga={fit:.4f} +-{ci} ({it} iters en la campana) -> {p}")
    vn._log(f"presupuesto igualado: {ITERS} iters para los {len(top)*len(DOMINIOS)} casos")

    for (ind, path, fit, _, _), (dom, geo) in itertools.product(top, DOMINIOS.items()):
        k = f"{ind}_{dom}"
        if k in out:
            vn._log(f"{k}: ya en cache, se salta")
            continue
        if os.path.exists(STOP):
            vn._log(f"centinela {STOP} presente -> parada")
            break
        vn._log(f"--- {k}: {ind} en {geo['Lx']:g}x{geo['Ly']:g}")
        r = vn.simular(path, RE, DX, alpha=ALPHA, iters=ITERS, extra=dict(geo))
        if r is None:
            vn._log(f"{k}: fallida")
            continue
        r.update({"individuo": ind, "dominio": dom, "fitness_ga": fit, **geo})
        out[k] = r
        json.dump(out, open(JSON_PATH, "w"), indent=2, ensure_ascii=False)
        vn._log(f"{k}: Cl={r['cl']:.4f} Cd={r['cd']:.5f} L/D={r['ld']:.2f}"
                f"+-{r['ld_ci95'] or 0:.2f} conv={r['converged_clcd']} ({r['wall_s']:.0f}s)")
        escribir_csv(out)

    analizar(out)


if __name__ == "__main__":
    main()

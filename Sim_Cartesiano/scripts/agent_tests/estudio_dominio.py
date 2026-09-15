"""
Estudio de independencia de dominio del ganador de islands_tfg2, a alpha=4 y
dx=0.002 fijos. Solo cambia el dominio.

Motivacion: top/bottom son 'slip' (v=0), o sea paredes de tunel cerrado, no campo
lejano (Simulador2D.py:7234-7235). Con Ly=5 el perfil ve h=2.5c y la interferencia
de sustentacion vale sigma=(pi^2/48)(c/h)^2 = 0.033. El inflow fuerza v=0 a solo
2c del perfil, donde el vortice ligado induciria un 2.7% de U_inf. Toda la campana
del GA se corrio ahi, asi que el L/D 27.75 es el de ese tunel, no el del perfil.

Los dos errores decaen con leyes distintas -- lateral como (c/h)^2, longitudinal
como c/d -- asi que el diseno separa ambos ejes:

  escalera A-B-C-D   crece todo a la vez, da la extrapolacion a dominio infinito
  factorial A-E-F-C  2x2 en (Lx, Ly), separa lateral de longitudinal y mide si
                     los dos efectos son aditivos o interaccionan
  G                  Ly muy por encima de Lx, explota que ny crece mas despacio
                     que nx (el error lateral es el que decae rapido)

Metricas por caso: los coeficientes con su CI95, el estado del criterio de parada,
la dispersion entre estimadores de Cl, y cuatro diagnosticos leidos del campo final
que miden directamente cuanto molesta cada frontera.

NO SE LANZA SOLO. Ejecutar explicitamente:
    .venv/bin/python scripts/agent_tests/estudio_dominio.py            # todos
    .venv/bin/python scripts/agent_tests/estudio_dominio.py --solo A,E # subconjunto
    .venv/bin/python scripts/agent_tests/estudio_dominio.py --analisis # sin simular
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "agent_tests"))
os.chdir(ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import verificacion_numerica as vn

PERFIL = ("results/convergence_study/islands_tfg2/NACA_0012_sharp/epoch10/"
          "NACA_0012_sharp_winner_Re100000_a4.0_LD27.75.dat")
ALPHA = 4.0
RE = 1e5
DX = 0.002
ITERS = 13000          # t=8 convectivos, el mismo presupuesto que las polares
U_INF = 1.0
CHORD = 1.0

OUT = os.path.join(ROOT, "results", "verificacion_numerica", "estudio_dominio")
CAMPOS_DIR = os.path.join(OUT, "campos")
JSON_PATH = f"{OUT}/estudio_dominio.json"
CSV_PATH = f"{OUT}/estudio_dominio.csv"
STOP = os.path.join(ROOT, "STOP_SIMULATION.trigger")

# tag, Lx, Ly, cx, rol
CASOS = [
    ("A",  8.0,  5.0,  2.0, "base: el dominio con el que se corrio todo el GA"),
    ("B", 16.0, 10.0,  4.0, "escalera"),
    ("C", 24.0, 16.0,  6.0, "escalera + esquina (Lx grande, Ly grande) del factorial"),
    ("D", 40.0, 24.0, 10.0, "escalera: el mas grande, ambos errores por debajo del 0.5%"),
    ("E",  8.0, 16.0,  2.0, "factorial: solo Ly (Lx del GA) -> aisla el error lateral"),
    ("F", 24.0,  5.0,  6.0, "factorial: solo Lx (Ly del GA) -> aisla el longitudinal"),
    ("G", 12.0, 24.0,  3.0, "Ly >> Lx: maximo lateral por el minimo coste"),
]
ESCALERA = ["A", "B", "C", "D"]
FACTORIAL = {"00": "A", "01": "E", "10": "F", "11": "C"}   # (Lx grande, Ly grande)

CAMPOS = ["tag", "Lx", "Ly", "cx", "h", "c_h", "sigma", "d_up", "d_down",
          "v_ind_teor", "nx", "ny", "celdas",
          "cl", "cl_ci95", "cd", "cd_ci95", "ld", "ld_ci95",
          "cl_std", "cd_std", "n_samples",
          "converged_clcd", "t_conv_clcd", "iters_efectivas",
          "cl_cp_discrepancy", "cl_cp_discrepancy_flag", "cl_audit_inst",
          "eps_top", "eps_bot", "v_in_max", "u_out_min", "cp_wall_range",
          "wall_s"]


def geometria(Lx, Ly, cx):
    """Parametros que fijan el error de cada frontera, antes de simular."""
    h = Ly / 2.0
    return {
        "Lx": Lx, "Ly": Ly, "cx": cx, "h": h,
        "c_h": round(CHORD / h, 4),
        # interferencia de sustentacion en tunel cerrado 2D (Barlow-Rae-Pope).
        # El coeficiente pi^2/48 esta citado de memoria: verificar antes de la
        # memoria del TFG. El escalado (c/h)^2 si es solido.
        "sigma": round((np.pi ** 2 / 48.0) * (CHORD / h) ** 2, 5),
        "d_up": cx,
        "d_down": Lx - cx - CHORD,
        # v inducida por el vortice ligado en el plano de entrada, Cl*c/(4*pi*d).
        # Es lo que el BC de inflow esta forzando a cero.
        "v_ind_teor": round(0.67 * CHORD / (4 * np.pi * cx), 5),
    }


def diagnostico_contorno(npz_path):
    """Cuanto molesta cada frontera, medido sobre el campo final.

    u tiene forma (ny, nx): fila 0 = pared inferior, fila -1 = superior,
    columna 0 = inflow, columna -1 = outflow.
    """
    if not os.path.exists(npz_path):
        return {}
    d = np.load(npz_path)
    u, v, p = d["u"], d["v"], d["p"]
    q = 0.5 * U_INF ** 2      # rho=1
    # Las paredes slip aceleran el flujo: es el bloqueo, medido y no estimado.
    eps_top = float(np.mean(u[-1, :]) / U_INF - 1.0)
    eps_bot = float(np.mean(u[0, :]) / U_INF - 1.0)
    # El inflow impone v=0; la primera columna interior dice cuanto v queria haber.
    v_in_max = float(np.max(np.abs(v[:, 1])) / U_INF)
    # Deficit de estela en el plano de salida: si es grande, la estela sale viva.
    u_out_min = float(np.min(u[:, -1]) / U_INF)
    # Si la pared lejana aun "siente" el perfil, su Cp no es plano.
    cp_wall_range = float((np.max(p[-1, :]) - np.min(p[-1, :])) / q)
    return {"eps_top": round(eps_top, 5), "eps_bot": round(eps_bot, 5),
            "v_in_max": round(v_in_max, 5), "u_out_min": round(u_out_min, 5),
            "cp_wall_range": round(cp_wall_range, 5)}


def escribir_csv(out):
    filas = [out[t] for t, *_ in CASOS if t in out]
    if not filas:
        return
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    vn._log(f"csv -> {CSV_PATH} ({len(filas)} casos)")


# ----------------------------------------------------------------------
# Analisis
# ----------------------------------------------------------------------
def extrapolar(out):
    """Cl y Cd a dominio infinito.

    Dos ajustes, porque los dos errores no decaen igual:
      simple  y = y_inf + k*sigma        sobre la escalera (conflaciona ambos)
      doble   y = y_inf + k1*sigma + k2*(c/d_up)   sobre los 7 casos
    El doble es el que separa lateral de longitudinal; necesita >=4 casos.
    """
    r = {}
    esc = [out[t] for t in ESCALERA if t in out]
    if len(esc) >= 3:
        s = np.array([c["sigma"] for c in esc])
        for m in ("cl", "cd"):
            k, y0 = np.polyfit(s, np.array([c[m] for c in esc]), 1)
            r[f"{m}_inf_simple"] = round(float(y0), 6)
            r[f"{m}_pend_sigma"] = round(float(k), 6)

    todos = [out[t] for t, *_ in CASOS if t in out]
    if len(todos) >= 4:
        A = np.column_stack([np.ones(len(todos)),
                             [c["sigma"] for c in todos],
                             [CHORD / c["d_up"] for c in todos]])
        for m in ("cl", "cd"):
            y = np.array([c[m] for c in todos])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            resid = y - A @ coef
            r[f"{m}_inf"] = round(float(coef[0]), 6)
            r[f"{m}_k_lateral"] = round(float(coef[1]), 6)
            r[f"{m}_k_longitud"] = round(float(coef[2]), 6)
            r[f"{m}_resid_max"] = round(float(np.max(np.abs(resid))), 6)
    return r


def factorial(out):
    """2x2 en (Lx, Ly). Separa los dos efectos y mide si son aditivos."""
    if not all(t in out for t in FACTORIAL.values()):
        return {}
    a, e, f, c = (out[FACTORIAL[k]] for k in ("00", "01", "10", "11"))
    r = {}
    for m in ("cl", "cd", "ld"):
        lat_lo, lat_hi = e[m] - a[m], c[m] - f[m]      # Ly 5->16, a Lx=8 y a Lx=24
        lon_lo, lon_hi = f[m] - a[m], c[m] - e[m]      # Lx 8->24, a Ly=5 y a Ly=16
        r[m] = {
            "lateral_Lx8": round(lat_lo, 6),
            "lateral_Lx24": round(lat_hi, 6),
            "longitud_Ly5": round(lon_lo, 6),
            "longitud_Ly16": round(lon_hi, 6),
            # Si no es ~0, los dos errores no se suman y no vale corregir por separado.
            "interaccion": round(lat_hi - lat_lo, 6),
            "total_A_a_C": round(c[m] - a[m], 6),
        }
    return r


def figuras(out, ext):
    casos = [out[t] for t, *_ in CASOS if t in out]
    if len(casos) < 3:
        return
    tags = [c["tag"] for c in casos]
    sig = np.array([c["sigma"] for c in casos])
    esc = [out[t] for t in ESCALERA if t in out]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    for ax, m, lbl in zip(axes[:2], ("cl", "cd"), ("$C_l$", "$C_d$")):
        y = np.array([c[m] for c in casos])
        err = np.array([c.get(f"{m}_ci95") or 0.0 for c in casos])
        ax.errorbar(sig, y, yerr=err, fmt="o", ms=7, capsize=3, color="#1f77b4",
                    ls="none", label="casos")
        if esc:
            se = np.array([c["sigma"] for c in esc])
            ax.plot(se, [c[m] for c in esc], "-", lw=1, color="#1f77b4",
                    alpha=0.5, label="escalera A-B-C-D")
        yinf = ext.get(f"{m}_inf")
        if yinf is not None:
            ax.axhline(yinf, ls="--", color="#2ca02c",
                       label=f"dominio infinito = {yinf:.4f}")
        for t, x, v in zip(tags, sig, y):
            ax.annotate(t, (x, v), textcoords="offset points", xytext=(6, 5), fontsize=9)
        ax.set_xlabel(r"$\sigma = (\pi^2/48)(c/h)^2$   [interferencia lateral]")
        ax.set_ylabel(lbl)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    ax = axes[2]
    if yinf is not None and ext.get("cl_inf"):
        err_pct = [abs(c["cl"] - ext["cl_inf"]) / abs(ext["cl_inf"]) * 100 for c in casos]
        ci_pct = [(c.get("cl_ci95") or 0.0) / abs(c["cl"]) * 100 for c in casos]
        x = np.arange(len(casos))
        ax.bar(x - 0.2, err_pct, 0.4, color="#d62728", label="error vs dominio infinito")
        ax.bar(x + 0.2, ci_pct, 0.4, color="#7f7f7f", label="CI95 del propio punto")
        ax.set_xticks(x); ax.set_xticklabels(tags)
        ax.set_ylabel("% de $C_l$")
        ax.set_title("Un dominio esta convergido cuando la barra roja\nbaja de la gris",
                     fontsize=9)
        ax.grid(alpha=0.25, axis="y"); ax.legend(fontsize=8)

    fig.suptitle(f"Independencia de dominio — ganador islands_tfg2, "
                 f"$\\alpha$={ALPHA}°, Re=1e5, dx={DX} (fijos)", fontsize=11)
    p = f"{OUT}/estudio_dominio.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    vn._log(f"figura -> {p}")


def informe(out, ext, fac):
    casos = [out[t] for t, *_ in CASOS if t in out]
    L = ["# Independencia de dominio — ganador islands_tfg2", "",
         f"alpha={ALPHA}deg, Re={RE:.0e}, dx={DX}, {ITERS} iters. Solo cambia el dominio.",
         "",
         "Las fronteras top/bottom son `slip` (v=0): paredes de tunel cerrado, no",
         "campo lejano. El error lateral decae como (c/h)^2 y el de entrada como c/d.",
         "", "## Casos", "",
         "| | Lx x Ly | cx | h | sigma | d_up | Cl | CI95 | Cd | CI95 | L/D | conv | s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in casos:
        L.append(f"| {c['tag']} | {c['Lx']:g}x{c['Ly']:g} | {c['cx']:g} | {c['h']:g} | "
                 f"{c['sigma']:.4f} | {c['d_up']:g} | {c['cl']:.4f} | "
                 f"{c.get('cl_ci95') or 0:.4f} | {c['cd']:.5f} | "
                 f"{c.get('cd_ci95') or 0:.5f} | {c['ld']:.2f} | "
                 f"{c['converged_clcd']} | {c['wall_s']:.0f} |")

    L += ["", "## Diagnostico de fronteras (campo final)", "",
          "`eps_top/bot` = aceleracion media en la pared slip (bloqueo medido).",
          "`v_in_max` = |v| maxima justo dentro del inflow, que el BC fuerza a 0.",
          "`u_out_min` = velocidad minima en el plano de salida (estela viva si << 1).",
          "`cp_wall_range` = excursion de Cp en la pared: si no es ~0, aun ve el perfil.",
          "",
          "| | eps_top | eps_bot | v_in_max | u_out_min | cp_wall_range | discrep. Cl |",
          "|---|---|---|---|---|---|---|"]
    for c in casos:
        L.append(f"| {c['tag']} | {c.get('eps_top', float('nan')):.5f} | "
                 f"{c.get('eps_bot', float('nan')):.5f} | {c.get('v_in_max', float('nan')):.5f} | "
                 f"{c.get('u_out_min', float('nan')):.4f} | "
                 f"{c.get('cp_wall_range', float('nan')):.5f} | "
                 f"{c.get('cl_cp_discrepancy') or 0:.5f} |")

    if ext:
        L += ["", "## Extrapolacion a dominio infinito", "",
              "Modelo `y = y_inf + k1*sigma + k2*(c/d_up)`, un termino por frontera.", ""]
        for m in ("cl", "cd"):
            if f"{m}_inf" in ext:
                L.append(f"- **{m}_inf = {ext[f'{m}_inf']:.6f}**  "
                         f"(k_lateral={ext[f'{m}_k_lateral']:+.4f}, "
                         f"k_longitud={ext[f'{m}_k_longitud']:+.4f}, "
                         f"residuo max={ext[f'{m}_resid_max']:.6f})")
        L += ["", "| | Cl | err vs inf | CI95 | convergido en dominio |",
              "|---|---|---|---|---|"]
        if "cl_inf" in ext:
            for c in casos:
                e = abs(c["cl"] - ext["cl_inf"])
                ci = c.get("cl_ci95") or 0.0
                L.append(f"| {c['tag']} | {c['cl']:.4f} | {e:.4f} "
                         f"({e/abs(ext['cl_inf'])*100:.2f}%) | {ci:.4f} | "
                         f"{'SI' if e <= ci else 'no'} |")

    if fac:
        L += ["", "## Factorial 2x2 (A=8x5, E=8x16, F=24x5, C=24x16)", "",
              "Si `interaccion` no es ~0, los dos errores no son aditivos y no vale",
              "corregirlos por separado.", "",
              "| | lateral@Lx8 | lateral@Lx24 | longitud@Ly5 | longitud@Ly16 | interaccion | total A->C |",
              "|---|---|---|---|---|---|---|"]
        for m, d in fac.items():
            L.append(f"| {m} | {d['lateral_Lx8']:+.5f} | {d['lateral_Lx24']:+.5f} | "
                     f"{d['longitud_Ly5']:+.5f} | {d['longitud_Ly16']:+.5f} | "
                     f"{d['interaccion']:+.5f} | {d['total_A_a_C']:+.5f} |")

    p = f"{OUT}/INFORME.md"
    open(p, "w").write("\n".join(L) + "\n")
    vn._log(f"informe -> {p}")


def analizar(out):
    ext, fac = extrapolar(out), factorial(out)
    escribir_csv(out)
    figuras(out, ext)
    informe(out, ext, fac)
    json.dump({"casos": out, "extrapolacion": ext, "factorial": fac},
              open(f"{OUT}/analisis.json", "w"), indent=2, ensure_ascii=False)
    return ext, fac


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", default=None, help="subconjunto de tags, ej: A,E,F,C")
    ap.add_argument("--analisis", action="store_true",
                    help="rehace tablas y figuras sobre lo ya simulado, sin GPU")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CAMPOS_DIR, exist_ok=True)
    out = json.load(open(JSON_PATH)) if os.path.exists(JSON_PATH) else {}

    if a.analisis:
        analizar(out)
        return

    pedidos = set(a.solo.split(",")) if a.solo else {t for t, *_ in CASOS}
    vn._log(f"perfil={PERFIL}")
    vn._log(f"alpha={ALPHA}  Re={RE:.0e}  dx={DX}  iters={ITERS}  (fijos)")
    vn._log(f"casos: {sorted(pedidos)}  ->  {OUT}")

    for tag, Lx, Ly, cx, rol in CASOS:
        if tag not in pedidos:
            continue
        if tag in out:
            vn._log(f"{tag}: ya en cache, se salta")
            continue
        if os.path.exists(STOP):
            vn._log(f"centinela {STOP} presente -> parada")
            break
        g = geometria(Lx, Ly, cx)
        vn._log(f"--- {tag}: Lx={Lx:g} Ly={Ly:g} cx={cx:g}  "
                f"sigma={g['sigma']:.4f}  {rol}")
        campo = f"{CAMPOS_DIR}/{tag}_campo.npz"
        r = vn.simular(PERFIL, RE, DX, alpha=ALPHA, iters=ITERS,
                       extra={"Lx": Lx, "Ly": Ly, "cx": cx},
                       dump=f"{CAMPOS_DIR}/{tag}_series.npz",
                       dump_field=campo)
        if r is None:
            vn._log(f"{tag}: simulacion fallida")
            continue
        r["tag"] = tag
        r.update(g)
        r.update(diagnostico_contorno(campo))
        out[tag] = r
        json.dump(out, open(JSON_PATH, "w"), indent=2, ensure_ascii=False)
        vn._log(f"{tag}: Cl={r['cl']:.4f}+-{r.get('cl_ci95') or 0:.4f}  "
                f"Cd={r['cd']:.5f}  L/D={r['ld']:.2f}  "
                f"eps_top={r.get('eps_top', float('nan')):.5f}  "
                f"conv={r['converged_clcd']}  ({r['wall_s']:.0f}s)")
        analizar(out)

    ext, fac = analizar(out)

    print("\n" + "=" * 100)
    print(f"{'':3}{'LxxLy':>9}{'sigma':>9}{'Cl':>10}{'CI95':>9}{'Cd':>10}"
          f"{'L/D':>9}{'eps_top':>10}{'v_in':>9}{'s':>8}")
    print("-" * 100)
    for tag, *_ in CASOS:
        if tag not in out:
            continue
        c = out[tag]
        dom = f"{c['Lx']:g}x{c['Ly']:g}"
        print(f"{tag:<3}{dom:>9}{c['sigma']:>9.4f}"
              f"{c['cl']:>10.4f}{c.get('cl_ci95') or 0:>9.4f}{c['cd']:>10.5f}"
              f"{c['ld']:>9.2f}{c.get('eps_top', float('nan')):>10.5f}"
              f"{c.get('v_in_max', float('nan')):>9.5f}{c['wall_s']:>8.0f}")
    print("=" * 100)
    if "cl_inf" in ext:
        print(f"Cl dominio infinito = {ext['cl_inf']:.6f}   "
              f"Cd = {ext.get('cd_inf', float('nan')):.6f}")
    print(f"Informe: {OUT}/INFORME.md")


if __name__ == "__main__":
    main()

"""
Polar del NACA0012 a dx=0.002, alpha = 0..10 en saltos de 2 grados, Re=1e5.

Reutiliza verificacion_numerica.simular(), que es la ruta que produjo la polar
validada contra XFOIL: config de referencia (consistent + SA + SA-BC +
MacCormack) y criterio de parada calibrado desde criterio_parada.json.

Una simulacion independiente por angulo (geometria rotada, sin arrastre de
transitorio entre alphas). Reanudable: cada punto se cachea en el JSON y una
segunda invocacion salta los ya hechos. Pausable con el centinela STOP.

Salidas en results/verificacion_numerica/:
    polar_naca0012.json  serie completa por angulo (cl, cd, ld, CI95, t_conv...)
    polar_naca0012.csv   tabla plana
    polar_naca0012.png   Cl, Cd y L/D vs alpha, con XFOIL superpuesto
"""
from __future__ import annotations

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

PERFIL = os.environ.get("POLAR_PERFIL", "profiles/NACA_0012_sharp")
TAG = os.environ.get("POLAR_TAG", "naca0012")
ANGULOS = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)
RE = 1e5
DX = 0.002

# Dominio: por defecto el de RunGA.CONFIG (8x5). Se puede pisar para medir el
# efecto del bloqueo sin tocar dx_min.
DOMINIO = {k: float(os.environ[e]) for k, e in
           (("Lx", "POLAR_LX"), ("Ly", "POLAR_LY"), ("cx", "POLAR_CX"))
           if e in os.environ}

OUT = os.environ.get("POLAR_OUT", vn.OUT)
os.makedirs(OUT, exist_ok=True)
JSON_PATH = f"{OUT}/polar_{TAG}.json"
CSV_PATH = f"{OUT}/polar_{TAG}.csv"
PNG_PATH = f"{OUT}/polar_{TAG}.png"
STOP = os.path.join(ROOT, "STOP_SIMULATION.trigger")

CAMPOS = ["alpha", "cl", "cd", "ld", "cl_ci95", "cd_ci95", "ld_ci95",
          "cl_std", "cd_std", "n_samples", "converged_clcd", "t_conv_clcd",
          "iters_efectivas", "iters_programadas", "cl_cp_discrepancy",
          "cl_cp_discrepancy_flag", "dx", "Re", "Lx", "Ly", "wall_s"]


def escribir_csv(out):
    filas = [out[f"{a:.1f}"] for a in ANGULOS if f"{a:.1f}" in out]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS, extrasaction="ignore")
        w.writeheader()
        for r in filas:
            w.writerow(r)
    vn._log(f"csv -> {CSV_PATH} ({len(filas)} puntos)")


def figura(out):
    a = np.array([x for x in ANGULOS if f"{x:.1f}" in out])
    if len(a) < 2:
        return
    d = [out[f"{x:.1f}"] for x in a]
    cl = np.array([r["cl"] for r in d])
    cd = np.array([r["cd"] for r in d])
    ld = np.array([r["ld"] for r in d])
    ecl = np.array([r.get("cl_ci95") or 0.0 for r in d])
    ecd = np.array([r.get("cd_ci95") or 0.0 for r in d])

    xf = vn.cargar_xfoil("NACA0012", RE)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    sim_kw = dict(fmt="-o", ms=6, capsize=3, color="#1f77b4", label="Simulador2D (dx=0.002)")

    axes[0].errorbar(a, cl, yerr=ecl, **sim_kw)
    axes[1].errorbar(a, cd, yerr=ecd, **sim_kw)
    axes[2].plot(a, ld, "-o", ms=6, color="#1f77b4", label="Simulador2D (dx=0.002)")

    if xf is not None:
        m = (xf["Alpha"] >= a.min() - 0.5) & (xf["Alpha"] <= a.max() + 0.5)
        x = xf[m]
        for ax, col in zip(axes, ("Cl", "Cd", "LD")):
            ax.plot(x["Alpha"], x[col], "--s", ms=4, color="#d62728",
                    alpha=0.8, label="XFOIL NACA0012 Ncrit=9")

    # Valores de Cl sobre los puntos simulados: es la magnitud validada.
    for x, y in zip(a, cl):
        axes[0].annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                         xytext=(0, 9), ha="center", fontsize=8, color="#1f77b4")
    for x, y in zip(a, ld):
        axes[2].annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                         xytext=(0, 9), ha="center", fontsize=8, color="#1f77b4")

    for ax, lbl in zip(axes, ("$C_l$", "$C_d$", "$L/D = C_l/C_d$")):
        ax.set_xlabel(r"$\alpha$ [$^\circ$]")
        ax.set_ylabel(lbl)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[1].set_yscale("log")

    dom = (f"Lx={DOMINIO['Lx']:g} Ly={DOMINIO['Ly']:g}" if DOMINIO else "Lx=8 Ly=5")
    fig.suptitle(f"Polar {TAG} — Re=1e5, dx=0.002, {dom}, consistent+SA+SA-BC+MacCormack",
                 fontsize=11)
    fig.savefig(PNG_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    vn._log(f"figura -> {PNG_PATH}")


def main():
    out = json.load(open(JSON_PATH)) if os.path.exists(JSON_PATH) else {}
    crit = vn.criterio_calibrado()
    vn._log(f"criterio de parada: {crit or 'FICHERO AUSENTE -> defaults de main()'}")
    vn._log(f"perfil={PERFIL}  Re={RE:.0e}  dx={DX}  alphas={list(ANGULOS)}")
    vn._log(f"dominio: {DOMINIO or 'RunGA.CONFIG (Lx=8, Ly=5, cx=2)'}  ->  {OUT}")
    vn._log(f"presupuesto por punto: {vn.iters_for(DX)} iters (t={vn.T_TARGET} conv)")

    for a in ANGULOS:
        k = f"{a:.1f}"
        if k in out:
            vn._log(f"alpha={a}: ya en cache, se salta")
            continue
        if os.path.exists(STOP):
            vn._log(f"centinela {STOP} presente -> parada")
            break
        vn._log(f"--- alpha={a}deg ---")
        r = vn.simular(PERFIL, RE, DX, alpha=a, extra=DOMINIO or None)
        if r is None:
            vn._log(f"alpha={a}: simulacion fallida")
            continue
        r["Lx"] = DOMINIO.get("Lx", 8.0)
        r["Ly"] = DOMINIO.get("Ly", 5.0)
        out[k] = r
        with open(JSON_PATH, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        vn._log(f"alpha={a}: Cl={r['cl']:.4f}  Cd={r['cd']:.5f}  L/D={r['ld']:.2f}  "
                f"conv={r['converged_clcd']}  t_conv={r['t_conv_clcd']}  "
                f"n={r['n_samples']}  ({r['wall_s']}s)")
        escribir_csv(out)
        figura(out)

    escribir_csv(out)
    figura(out)

    print("\n" + "=" * 78)
    print(f"{'alpha':>7s} {'Cl':>9s} {'Cd':>10s} {'L/D':>8s} {'conv':>6s} "
          f"{'t_conv':>8s} {'n':>5s} {'s':>8s}")
    print("-" * 78)
    for a in ANGULOS:
        k = f"{a:.1f}"
        if k not in out:
            continue
        r = out[k]
        tc = r.get("t_conv_clcd")
        print(f"{a:>7.1f} {r['cl']:>9.4f} {r['cd']:>10.5f} {r['ld']:>8.2f} "
              f"{str(r['converged_clcd']):>6s} "
              f"{(f'{tc:.2f}' if tc is not None else '-'):>8s} "
              f"{r['n_samples']:>5d} {r['wall_s']:>8.0f}")
    print("=" * 78)


if __name__ == "__main__":
    main()

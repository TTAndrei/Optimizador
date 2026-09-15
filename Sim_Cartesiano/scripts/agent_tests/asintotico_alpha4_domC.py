"""
Asintótico temporal del perfil ganador del AG a alpha=4, Re=1e5, dx=0.002, en el
dominio C (Lx=24, Ly=16, cx=6).

La polar de results/polar_fina_1grado_dom24x16/ corta en 13000 iteraciones, que
en este dominio son t≈10 tiempos convectivos. La serie de ventana larga a dx=0.001
mostró que la media acumulada todavía sube un 2.3% entre t=10 y t=20 y un 1.5%
más hasta t=71, así que los valores de la tabla son cota inferior, no asintótico.

Este run alarga hasta t≈40 manteniendo dx y dominio, para medir cuánto vale esa
cola en el punto que importa. Vuelca la serie completa (t, cl, cd) y el campo
final.

El early-stop sigue activo. Con tol_noise=0.05 frente a un ruido medido de 0.11-0.20
no puede dispararse en este flujo; si lo hiciera sería un dato en sí mismo.

Uso:
    .venv/bin/python scripts/agent_tests/asintotico_alpha4_domC.py
    .venv/bin/python scripts/agent_tests/asintotico_alpha4_domC.py --analisis
"""
from __future__ import annotations

import argparse
import json
import os
import sys

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

OUT = os.path.join(ROOT, "results", "asintotico_alpha4_domC")
os.makedirs(OUT, exist_ok=True)

PERFIL = ("results/convergence_study/islands_tfg2/NACA_0012_sharp/epoch10/"
          "NACA_0012_sharp_winner_Re100000_a4.0_LD27.75.dat")
ALPHA, RE, DX = 4.0, 1e5, 0.002
DOMINIO = {"Lx": 24.0, "Ly": 16.0, "cx": 6.0}

# 13000 iters -> t=10.1 en este dominio. Para t~40 hacen falta ~51500.
ITERS = 52000

JSON_PATH = f"{OUT}/asintotico_alpha4_domC.json"
SERIES = f"{OUT}/series.npz"
CAMPO = f"{OUT}/campo_final.npz"
PNG = f"{OUT}/evolucion_temporal.png"

# La polar reporta media(Cl)/media(Cd); la media del cociente instantáneo es otra
# cosa (Jensen: en la serie larga difieren un 3%). Se dan las dos.
REF_POLAR = 31.70   # L/D de alpha=4 en results/polar_fina_1grado_dom24x16/


def cargar_serie():
    z = np.load(SERIES)
    t, cl, cd = z["t"], z["cl"], z["cd"]
    ok = np.isfinite(t) & np.isfinite(cl) & np.isfinite(cd) & (t > 0)
    return t[ok], cl[ok], cd[ok]


def tabla(t, cl, cd, t0=3.0):
    """Qué habría reportado el run si hubiera acabado en t."""
    filas = []
    for tt in (5, 8, 10, 12, 15, 20, 25, 30, 35, float(t[-1])):
        if tt > t[-1]:
            continue
        m = (t >= t0) & (t <= tt)
        if m.sum() < 10:
            continue
        # Como RunGA: media de la cola (ultimo 20%) de cl y de cd por separado.
        idx = np.where(m)[0]
        cola = idx[int(0.8 * len(idx)):]
        filas.append({
            "t": round(float(tt), 2),
            "ld_cola": round(float(cl[cola].mean() / cd[cola].mean()), 3),
            "ld_acum": round(float(cl[m].mean() / cd[m].mean()), 3),
            "ld_ratio_medio": round(float((cl[m] / cd[m]).mean()), 3),
            "cl_cola": round(float(cl[cola].mean()), 5),
            "cd_cola": round(float(cd[cola].mean()), 6),
            "n": int(m.sum()),
        })
    return filas


def figura(t, cl, cd, t0=3.0):
    ld = cl / cd
    m = t >= t0
    n = np.arange(1, m.sum() + 1)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, layout="constrained")
    for ax, y, lb in zip(axes, (cl, cd, ld), (r"$C_l$", r"$C_d$", r"$C_l/C_d$")):
        ax.plot(t, y, lw=0.6, color="#c0392b", alpha=0.4, label="instantáneo")
        ax.plot(t[m], np.cumsum(y[m]) / n, lw=2.2, color="#7b241c",
                label=f"media acumulada desde t={t0:g}")
        ax.set_ylabel(lb)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
    axes[2].axhline(REF_POLAR, ls="--", lw=1.4, color="#1f4e79",
                    label=f"polar a 13000 iters ({REF_POLAR})")
    axes[2].axvline(10.1, ls=":", lw=1.4, color="#1f4e79")
    axes[2].legend(fontsize=9, loc="lower right")
    axes[0].set_title("Perfil ganador AG — α=4°, Re=1e5, dx=0.002, dominio 24×16\n"
                      "asintótico temporal: hasta dónde sigue subiendo la media",
                      fontsize=12)
    axes[2].set_xlabel("t [tiempos convectivos]")
    fig.savefig(PNG, dpi=170, bbox_inches="tight")
    plt.close(fig)


def analisis():
    t, cl, cd = cargar_serie()
    filas = tabla(t, cl, cd)
    figura(t, cl, cd)
    d = json.load(open(JSON_PATH)) if os.path.exists(JSON_PATH) else {}
    d["evolucion"] = filas
    with open(JSON_PATH, "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)

    print(f"\n{'t':>7} {'L/D cola':>9} {'L/D acum':>9} {'<L/D> inst':>11} "
          f"{'Cl cola':>9} {'Cd cola':>9} {'vs polar':>9}")
    print("-" * 72)
    for r in filas:
        print(f"{r['t']:>7.1f} {r['ld_cola']:>9.2f} {r['ld_acum']:>9.2f} "
              f"{r['ld_ratio_medio']:>11.2f} {r['cl_cola']:>9.4f} {r['cd_cola']:>9.5f} "
              f"{100*(r['ld_cola']-REF_POLAR)/REF_POLAR:>+8.1f}%")
    print(f"\nfigura -> {PNG}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analisis", action="store_true")
    args = ap.parse_args()

    if args.analisis:
        analisis()
        return

    if os.path.exists(SERIES):
        vn._log(f"{SERIES} ya existe -> solo analisis")
        analisis()
        return

    vn._log(f"perfil={os.path.basename(PERFIL)}  alpha={ALPHA}  Re={RE:.0e}  dx={DX}")
    vn._log(f"dominio={DOMINIO}  iters={ITERS} (~t=40, ~3h20 a 4.29 it/s)")
    vn._log(f"criterio de parada activo: {vn.criterio_calibrado()}")

    r = vn.simular(PERFIL, RE, DX, alpha=ALPHA, iters=ITERS, extra=DOMINIO,
                   dump=SERIES, dump_field=CAMPO)
    if r is None:
        vn._log("simulacion fallida")
        return
    r.update(DOMINIO)
    r["perfil"] = PERFIL
    r["ref_polar_13000it"] = REF_POLAR
    with open(JSON_PATH, "w") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    vn._log(f"Cl={r['cl']:.4f} Cd={r['cd']:.5f} L/D={r['ld']:.2f} "
            f"conv={r['converged_clcd']} n={r['n_samples']} ({r['wall_s']}s)")
    analisis()


if __name__ == "__main__":
    main()

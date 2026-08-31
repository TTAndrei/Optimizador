"""NACA 0012: polar extrapolada del estudio final frente a XFOIL.

    .venv/bin/python scripts/agent_tests/rf_xfoil.py

Referencia: airfoiltools, Re=1e5, Ncrit=9, en
results/1eraGranOptimizacion/verificacion_numerica/validacion_xfoil.json

OJO: el campo `cl_sim`/`cd_sim` de ese JSON es de la configuracion ANTIGUA del
solver (otro presupuesto de proyeccion, sin las optimizaciones) y de la campaña
vieja. Aqui se ignora: la comparacion se hace contra los datos del estudio final.
"""
import json
import os
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verificacion_numerica as vn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "resultados_finales")
XF = os.path.join(ROOT, "results", "1eraGranOptimizacion", "verificacion_numerica",
                  "validacion_xfoil.json")
TERNA = (0.002, 0.004, 0.008)
ANG = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
DPI = 190

C_XF, C_FINA, C_EXT = "#2e7d32", "#1f4e79", "#c0392b"


def suave(x, y, n=400):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3:
        return x, y
    xf = np.linspace(x.min(), x.max(), n)
    return xf, PchipInterpolator(x, y)(xf)


def carga():
    d = {dx: json.load(open(os.path.join(
        OUT, "naca0012", "metricas", "polar_dx%.4f.json" % dx))) for dx in TERNA}
    fina, ext = {}, {}
    for a in ANG:
        k = "%.1f" % a
        cl = [d[dx][k]["cl"] for dx in TERNA]
        cd = [d[dx][k]["cd"] for dx in TERNA]
        cle = vn.extrapola_robusto(*cl, *TERNA)[0]
        cde = vn.extrapola_robusto(*cd, *TERNA, log=min(cd) > 0)[0]
        fina[a] = (cl[0], cd[0], cl[0] / cd[0] if cd[0] else np.nan)
        ext[a] = (cle, cde, cle / cde if cde else np.nan)
    xf = {p["alpha"]: (p["cl_xfoil"], p["cd_xfoil"], p["ld_xfoil"])
          for p in json.load(open(XF))["puntos"]}
    return fina, ext, xf


def pendiente(aa, cl):
    """dCl/dalpha por minimos cuadrados en el tramo lineal (alpha <= 5)."""
    m = [i for i, a in enumerate(aa) if a <= 5.0]
    return float(np.polyfit([aa[i] for i in m], [cl[i] for i in m], 1)[0])


def main():
    fina, ext, xf = carga()
    comunes = sorted(set(ANG) & set(xf))

    print("NACA 0012, Re=1e5 — extrapolado del estudio final frente a XFOIL\n")
    print("%6s | %8s %8s %8s | %8s %8s %8s | %8s %8s" % (
        "alpha", "Cl xfoil", "Cl fina", "Cl ext", "Cd xfoil", "Cd fina", "Cd ext",
        "L/D xf", "L/D ext"))
    for a in comunes:
        print("%6.0f | %8.4f %8.4f %8.4f | %8.5f %8.5f %8.5f | %8.2f %8.2f" % (
            a, xf[a][0], fina[a][0], ext[a][0],
            xf[a][1], fina[a][1], ext[a][1], xf[a][2], ext[a][2]))

    def err(v, ref):
        return 100.0 * (v - ref) / ref if abs(ref) > 1e-9 else np.nan

    print("\n%6s | %10s %10s | %10s %10s" % (
        "alpha", "err Cl fina", "err Cl ext", "err Cd fina", "err Cd ext"))
    for a in comunes:
        print("%6.0f | %9.1f%% %9.1f%% | %9.1f%% %9.1f%%" % (
            a, err(fina[a][0], xf[a][0]), err(ext[a][0], xf[a][0]),
            err(fina[a][1], xf[a][1]), err(ext[a][1], xf[a][1])))

    sin0 = [a for a in comunes if a > 0]
    res = {}
    for nom, src in (("fina", fina), ("extrapolado", ext)):
        res[nom] = {
            "cl_mae": float(np.mean([abs(src[a][0] - xf[a][0]) for a in comunes])),
            "cd_mae": float(np.mean([abs(src[a][1] - xf[a][1]) for a in comunes])),
            "cl_err_medio_pct": float(np.mean([err(src[a][0], xf[a][0]) for a in sin0])),
            "cd_err_medio_pct": float(np.mean([err(src[a][1], xf[a][1]) for a in comunes])),
            "pendiente_cl": pendiente(ANG, [src[a][0] for a in ANG]),
        }
    res["xfoil"] = {"pendiente_cl": pendiente(comunes, [xf[a][0] for a in comunes])}
    res["teorica_2pi"] = {"pendiente_cl": 2 * np.pi * np.pi / 180}

    print("\n%-14s %10s %10s %14s %14s %12s" % (
        "", "MAE Cl", "MAE Cd", "err medio Cl", "err medio Cd", "dCl/dalpha"))
    for nom in ("fina", "extrapolado"):
        r = res[nom]
        print("%-14s %10.4f %10.5f %13.1f%% %13.1f%% %12.5f" % (
            nom, r["cl_mae"], r["cd_mae"], r["cl_err_medio_pct"],
            r["cd_err_medio_pct"], r["pendiente_cl"]))
    print("%-14s %10s %10s %14s %14s %12.5f" % (
        "XFOIL", "-", "-", "-", "-", res["xfoil"]["pendiente_cl"]))
    print("%-14s %10s %10s %14s %14s %12.5f" % (
        "teorica 2pi", "-", "-", "-", "-", res["teorica_2pi"]["pendiente_cl"]))

    # ---------------- figura ----------------
    fig, axs = plt.subplots(1, 3, figsize=(15.4, 4.9), layout="constrained")
    series = ((xf, comunes, C_XF, "o", "XFOIL (Ncrit=9)"),
              (fina, ANG, C_FINA, "s", "CFD, malla fina dx=0.002"),
              (ext, ANG, C_EXT, "D", "CFD, extrapolado (Richardson)"))
    for ax, i, lab in zip(axs, (0, 1, 2), (r"$C_L$", r"$C_D$", r"$L/D$")):
        for src, aa, col, mk, nom in series:
            y = [src[a][i] for a in aa]
            ls = "--" if col == C_EXT else "-"
            xs, ys = suave(aa, y)
            ax.plot(xs, ys, color=col, lw=2.2 if col == C_EXT else 1.5, ls=ls)
            ax.plot(aa, y, color=col, marker=mk, ms=6, ls="none", label=nom)
        ax.set_xlabel(r"$\alpha$ [$^\circ$]")
        ax.set_ylabel(lab)
        ax.set_xticks(ANG)
        ax.grid(alpha=.25)
        if i == 1:
            ax.axhline(0, color="k", lw=.7, ls=":")
    axs[0].legend(fontsize=8.5)
    fig.suptitle("NACA 0012, Re = $10^5$ — estudio final frente a XFOIL "
                 "(airfoiltools, Ncrit = 9)", fontsize=12)
    r1 = os.path.join(OUT, "comparativa", "xfoil_naca0012.png")
    fig.savefig(r1, dpi=DPI)
    plt.close(fig)

    # ---------------- figura de errores ----------------
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.6), layout="constrained")
    an = [a for a in comunes if a > 0]
    for ax, i, lab in zip(axs, (0, 1), (r"error en $C_L$ [%]", r"error en $C_D$ [%]")):
        for src, col, mk, nom in ((fina, C_FINA, "s", "malla fina"),
                                  (ext, C_EXT, "D", "extrapolado")):
            y = [err(src[a][i], xf[a][i]) for a in an]
            xs, ys = suave(an, y)
            ax.plot(xs, ys, color=col, lw=2.0,
                    ls="--" if col == C_EXT else "-")
            ax.plot(an, y, color=col, marker=mk, ms=6, ls="none", label=nom)
        ax.axhline(0, color="k", lw=1.0)
        ax.set_xlabel(r"$\alpha$ [$^\circ$]")
        ax.set_ylabel(lab)
        ax.set_xticks(an)
        ax.grid(alpha=.25)
        ax.legend(fontsize=9)
    axs[0].set_title(r"Sustentacion — se acerca a XFOIL al extrapolar", fontsize=10)
    axs[1].set_title(r"Resistencia — el sesgo persiste, no es de malla", fontsize=10)
    fig.suptitle(r"Error frente a XFOIL ($\alpha$ = 0 excluido: $C_L \approx 0$ "
                 "hace estallar el error relativo)", fontsize=11)
    r2 = os.path.join(OUT, "comparativa", "xfoil_naca0012_errores.png")
    fig.savefig(r2, dpi=DPI)
    plt.close(fig)

    salida = {"perfil": "NACA0012", "Re": 1e5, "fuente_xfoil": "airfoiltools, Ncrit=9",
              "terna": list(TERNA), "angulos_comunes": comunes,
              "resumen": res,
              "puntos": [{"alpha": a, "cl_xfoil": xf[a][0], "cd_xfoil": xf[a][1],
                          "ld_xfoil": xf[a][2],
                          "cl_fina": fina[a][0], "cd_fina": fina[a][1],
                          "cl_ext": ext[a][0], "cd_ext": ext[a][1],
                          "ld_ext": ext[a][2]} for a in comunes]}
    r3 = os.path.join(OUT, "comparativa", "xfoil_naca0012.json")
    with open(r3, "w") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    for r in (r1, r2, r3):
        print("  ->", os.path.relpath(r, ROOT))


if __name__ == "__main__":
    main()

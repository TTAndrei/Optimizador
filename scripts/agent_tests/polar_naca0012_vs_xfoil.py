"""NACA 0012 a Re=1e5: polar del solver (dominio C, dx=0.002) frente a XFOIL.

Referencia: airfoiltools, Re=1e5, Ncrit=5 (no Ncrit=9 como la comparativa vieja
de resultados_finales/comparativa; con Ncrit=5 la transición es más temprana, que
es lo que corresponde a un caso con SA-BC y Tu=0.1 %).

    .venv/bin/python scripts/agent_tests/polar_naca0012_vs_xfoil.py
"""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)

SIM = "results/polar_fina_1grado_dom24x16/polar_naca0012.csv"
XFOIL = "data/xfoil/xf-n0012-il-100000-n5.csv"
FIG = "plots/naca0012_vs_xfoil_ncrit5.png"
ALPHA_FIABLE = 6.0  # por encima, el solver no cumple el criterio de parada
OUT_JSON = "results/polar_fina_1grado_dom24x16/comparativa_xfoil_ncrit5.json"

C_SIM, C_XF = "#1f4e79", "#2e7d32"


def leer_xfoil(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        filas = list(csv.reader(f))
    i = next(i for i, r in enumerate(filas) if r and r[0].strip() == "Alpha")
    d = np.array([[float(v) for v in r[:3]] for r in filas[i + 1:] if len(r) >= 3])
    return {"alpha": d[:, 0], "cl": d[:, 1], "cd": d[:, 2], "ld": d[:, 1] / d[:, 2]}


def leer_sim(path):
    with open(path, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    g = lambda k: np.array([float(r[k]) for r in filas])
    return {"alpha": g("alpha"), "cl": g("cl"), "cd": g("cd"), "ld": g("ld"),
            "cl_ci95": g("cl_ci95"), "cd_ci95": g("cd_ci95"), "ld_ci95": g("ld_ci95"),
            "convergido": [r["converged_clcd"] == "True" for r in filas]}


def pendiente(alpha, cl, amax=6.0):
    """dCl/dalpha por grado, ajustada solo en el tramo lineal 0-amax."""
    m = (alpha >= 0.0) & (alpha <= amax)
    return float(np.polyfit(alpha[m], cl[m], 1)[0])


def main():
    sim, xf = leer_sim(SIM), leer_xfoil(XFOIL)

    # XFOIL interpolado a los ángulos del solver (malla de 0.25° frente a 1°).
    xf_i = {k: np.interp(sim["alpha"], xf["alpha"], xf[k]) for k in ("cl", "cd", "ld")}
    err = {k: (sim[k] - xf_i[k]) / xf_i[k] * 100.0 for k in ("cl", "cd", "ld")}
    fiable = sim["alpha"] <= ALPHA_FIABLE

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6))
    ax = axes.ravel()

    def panel(a, mag, titulo, ylabel):
        a.errorbar(sim["alpha"], sim[mag], yerr=sim[mag + "_ci95"], color=C_SIM,
                   marker="o", ms=4.5, lw=1.6, capsize=3,
                   label="Solver 2D (dx=0.002, dom. C)")
        no_conv = ~np.array(sim["convergido"])
        a.plot(sim["alpha"][no_conv], sim[mag][no_conv], "o", ms=8, mfc="none",
               mec=C_SIM, mew=1.2, label="sin converger")
        # Solo el tramo visible: si no, el autoescalado se lo lleva el α<0 de XFOIL.
        v = (xf["alpha"] >= -1.0) & (xf["alpha"] <= 11.0)
        a.plot(xf["alpha"][v], xf[mag][v], color=C_XF, ls="--", lw=1.6,
               label="XFOIL Re=1e5, Ncrit=5")
        a.set_xlim(-0.5, 10.5); a.set_xlabel(r"$\alpha$ [°]"); a.set_ylabel(ylabel)
        a.set_title(titulo, fontsize=10); a.grid(alpha=.3); a.legend(fontsize=7.5)

    panel(ax[0], "cl", "Sustentación", r"$C_l$")
    panel(ax[1], "cd", "Resistencia", r"$C_d$")
    panel(ax[2], "ld", "Eficiencia", r"$C_l/C_d$")

    a = ax[3]
    m = (xf["alpha"] >= 0) & (xf["alpha"] <= 11)
    a.plot(xf["cd"][m], xf["cl"][m], color=C_XF, ls="--", lw=1.8, label="XFOIL (0-11°)")
    a.errorbar(sim["cd"], sim["cl"], xerr=sim["cd_ci95"], yerr=sim["cl_ci95"],
               color=C_SIM, marker="o", ms=4.5, lw=1.6, capsize=3, label="Solver 2D")
    for i, al in enumerate(sim["alpha"]):
        a.annotate(f"{al:.0f}°", (sim["cd"][i], sim["cl"][i]), fontsize=7,
                   xytext=(4, -9), textcoords="offset points", color=C_SIM)
    a.set_xlim(0, 0.175); a.set_ylim(-0.05, 1.05)
    a.set_xlabel(r"$C_d$"); a.set_ylabel(r"$C_l$")
    a.set_title("Polar", fontsize=10); a.grid(alpha=.3); a.legend(fontsize=7.5)

    a = ax[4]
    for mag, col, mk in (("cl", "#1f4e79", "o"), ("cd", "#c0392b", "s"),
                         ("ld", "#7d3c98", "^")):
        a.plot(sim["alpha"], err[mag], color=col, marker=mk, ms=4.5, lw=1.5,
               label={"cl": r"$C_l$", "cd": r"$C_d$", "ld": r"$C_l/C_d$"}[mag])
    a.axhline(0, color="k", lw=.8)
    a.axvspan(ALPHA_FIABLE, 10.5, color="0.85", zorder=0)
    a.text(8.2, a.get_ylim()[1] * 0.55, "sin converger", fontsize=7.5, ha="center")
    a.set_xlim(-0.5, 10.5); a.set_xlabel(r"$\alpha$ [°]")
    a.set_ylabel("desviación frente a XFOIL [%]")
    a.set_title("Error relativo", fontsize=10); a.grid(alpha=.3); a.legend(fontsize=8)

    a = ax[5]
    pend = {"sim": pendiente(sim["alpha"], sim["cl"]),
            "xfoil": pendiente(xf["alpha"], xf["cl"]),
            "2pi": 2.0 * np.pi * np.pi / 180.0}
    aa = np.linspace(0, 6, 50)
    a.plot(sim["alpha"][fiable], sim["cl"][fiable], "o", color=C_SIM, ms=5,
           label=f"Solver 2D: {pend['sim']:.4f} /°")
    a.plot(aa, pend["sim"] * aa + np.mean(sim["cl"][fiable] - pend["sim"] * sim["alpha"][fiable]),
           color=C_SIM, lw=1.5)
    mx = (xf["alpha"] >= 0) & (xf["alpha"] <= 6)
    a.plot(xf["alpha"][mx], xf["cl"][mx], "s", color=C_XF, ms=4,
           label=f"XFOIL: {pend['xfoil']:.4f} /°")
    a.plot(aa, pend["xfoil"] * aa, color=C_XF, ls="--", lw=1.5)
    a.plot(aa, pend["2pi"] * aa, color="0.45", ls=":", lw=1.5,
           label=f"Perfil delgado $2\\pi$: {pend['2pi']:.4f} /°")
    a.set_xlabel(r"$\alpha$ [°]"); a.set_ylabel(r"$C_l$")
    a.set_title(r"Pendiente $dC_l/d\alpha$ (ajuste 0-6°)", fontsize=10)
    a.grid(alpha=.3); a.legend(fontsize=7.5)

    fig.suptitle("NACA 0012, Re = 1e5 — solver 2D (dominio C, dx=0.002) frente a XFOIL Ncrit=5",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=170)

    print(f"{'α':>4} {'Cl sim':>8} {'Cl xf':>8} {'Δ%':>7} "
          f"{'Cd sim':>8} {'Cd xf':>8} {'Δ%':>7} "
          f"{'L/D sim':>8} {'L/D xf':>8} {'Δ%':>7}  conv")
    for i, al in enumerate(sim["alpha"]):
        print(f"{al:4.0f} {sim['cl'][i]:8.4f} {xf_i['cl'][i]:8.4f} {err['cl'][i]:7.1f} "
              f"{sim['cd'][i]:8.5f} {xf_i['cd'][i]:8.5f} {err['cd'][i]:7.1f} "
              f"{sim['ld'][i]:8.2f} {xf_i['ld'][i]:8.2f} {err['ld'][i]:7.1f}"
              f"   {'sí' if sim['convergido'][i] else 'no'}")

    res = {
        "sim": SIM, "xfoil": XFOIL, "ncrit": 5, "Re": 1e5, "dx": 0.002,
        "dominio": {"Lx": 24.0, "Ly": 16.0, "cx": 6.0},
        "alpha": sim["alpha"].tolist(),
        "error_pct": {k: np.round(v, 2).tolist() for k, v in err.items()},
        "error_medio_abs_pct": {k: float(np.mean(np.abs(v))) for k, v in err.items()},
        "error_medio_abs_pct_hasta_6deg": {
            k: float(np.mean(np.abs(v[fiable]))) for k, v in err.items()},
        "pendiente_cl_por_grado": pend,
        "ld_max": {"sim": float(np.max(sim["ld"])),
                   "sim_alpha": float(sim["alpha"][int(np.argmax(sim["ld"]))]),
                   "xfoil": float(np.max(xf["ld"])),
                   "xfoil_alpha": float(xf["alpha"][int(np.argmax(xf["ld"]))])},
    }
    json.dump(res, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)

    print("\nerror medio |Δ| (1-10°): " + "  ".join(
        f"{k}={res['error_medio_abs_pct'][k]:.1f}%" for k in ("cl", "cd", "ld")))
    print("error medio |Δ| (1-6°):  " + "  ".join(
        f"{k}={res['error_medio_abs_pct_hasta_6deg'][k]:.1f}%" for k in ("cl", "cd", "ld")))
    print(f"pendiente dCl/dα: solver {pend['sim']:.4f}  XFOIL {pend['xfoil']:.4f}  "
          f"2π {pend['2pi']:.4f} /°")
    print(f"L/D máx: solver {res['ld_max']['sim']:.2f} @ {res['ld_max']['sim_alpha']:.0f}°"
          f"   XFOIL {res['ld_max']['xfoil']:.2f} @ {res['ld_max']['xfoil_alpha']:.2f}°")
    print(f"figura -> {FIG}\njson   -> {OUT_JSON}")


if __name__ == "__main__":
    main()

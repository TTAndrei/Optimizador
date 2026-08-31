"""Convergencia del modelo de islas: L/D del mejor de cada isla y distancia de
forma media entre islas, epoca a epoca.

    .venv/bin/python scripts/agent_tests/rf_fig_islas.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "results", "convergence_study", "islands_tfg2",
                   "convergencia_islas.json")
DST = os.path.join(ROOT, "resultados_finales", "comparativa",
                   "convergencia_islas.png")
COL = {"NACA_0012_sharp": ("#1f4e79", "o", "NACA 0012 sharp"),
       "AG24": ("#c0392b", "s", "AG24"),
       "GM15": ("#1e8449", "^", "GM15"),
       "s1014": ("#8e44ad", "D", "s1014")}
DPI = 190


def main():
    d = json.load(open(SRC))
    h = d["historia"]
    cfg = d["config"]
    ep = [e["epoca"] for e in h]
    dist = [e["shape_dist_media"] for e in h]

    fig, (ax, axd) = plt.subplots(2, 1, figsize=(8.6, 7.2), sharex=True,
                                  height_ratios=[2.0, 1.0], layout="constrained")

    # las migraciones caen cada migrate_every epocas
    mig = [k for k in ep if k > 0 and k % cfg["migrate_every"] == 0]
    for a in (ax, axd):
        for k in mig:
            a.axvline(k, color="#7f8c8d", lw=.8, ls=":", zorder=0)
        a.grid(alpha=.25)
    ax.plot([], [], color="#7f8c8d", lw=.8, ls=":", label="migracion")

    for isla in h[0]["islas"]:
        c, m, nom = COL[isla]
        ax.plot(ep, [e["ld"][isla] for e in h], color=c, marker=m, ms=5,
                lw=1.6, label="isla %s" % nom)
    ax.set_ylabel("$L/D$ del mejor individuo de cada isla")
    ax.legend(fontsize=8.5, loc="lower right", ncol=2)
    ax.set_title("Convergencia del modelo de islas — %d epocas, poblacion %d por "
                 "isla,\nmigracion cada %d epocas (dx = %g, "
                 r"$\alpha$ = %g$^\circ$, Re = $10^5$)"
                 % (cfg["n_epocas"], cfg["pop"], cfg["migrate_every"],
                    cfg["dx"], d["condiciones"]["alpha_deg"]), fontsize=10.5)

    axd.plot(ep, dist, color="#d35400", marker="v", ms=5.5, lw=2.0,
             label="media entre islas")
    axd.plot(ep, [e["shape_dist_max"] for e in h], color="#d35400", marker=".",
             ms=5, lw=1.1, ls="--", alpha=.75, label="maxima entre islas")
    axd.set_yscale("log")
    axd.set_ylabel("distancia de forma")
    axd.set_xlabel("epoca")
    axd.set_xticks(ep)
    axd.legend(fontsize=8.5, loc="upper right")

    fig.savefig(DST, dpi=DPI)
    plt.close(fig)
    print("  ->", os.path.relpath(DST, ROOT))


if __name__ == "__main__":
    main()

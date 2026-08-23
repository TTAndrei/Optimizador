"""
Figuras del analisis de las dos polares del dominio C.

Fuentes:
  results/polar_fina_1grado_dom24x16/   alpha = 1..10 de grado en grado, t ~ 10
  results/polar_2grados_dom24x16/       alpha = 0..10 de dos en dos,     t ~ 20
  results/asintotico_alpha4_domC/       alpha = 4 del ganador hasta t = 41.5
  results/verificacion_numerica/gci_recalculado.json   3 mallas, alpha = 4, dominio del GA

La extrapolacion a t -> infinito es la de scripts/agent_tests/extrapolacion_polar.py
(ajuste f(T) = Finf - A*exp(-T/tau) sobre las colas a T, 2T y 4T).

La polar corregida a 1 grado NO es simulacion nueva: es la polar de 1 grado a
t ~ 10 multiplicada por el factor Finf/f(t=10) medido en los angulos pares e
interpolado entre ellos. Solo se dibuja donde ese factor esta acotado por dos
angulos con meseta; fuera de ahi van los puntos crudos con marcador hueco.

Uso:  .venv/bin/python scripts/agent_tests/figuras_analisis_polar.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, NullFormatter, NullLocator

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "agent_tests"))
os.chdir(ROOT)

from extrapolacion_polar import cola, exp_plateau, serie, leer_csv  # noqa: E402

T10 = "results/polar_fina_1grado_dom24x16"
T20 = "results/polar_2grados_dom24x16"
OUT = "results/analisis_polar"
os.makedirs(OUT, exist_ok=True)

PERFILES = {
    "ganador_tfg2": {"label": "Ganador TFG (AG)", "color": "#c1272d", "marker": "o"},
    "naca0012":     {"label": "NACA 0012",        "color": "#0b4f9e", "marker": "s"},
}
ANG = (0, 2, 4, 6, 8, 10)

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 14.5,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
    "figure.facecolor": "white",
})


def ejes(ax, xlabel, ylabel, titulo, xpaso=2.0, ypaso=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if titulo:
        ax.set_title(titulo)
    if xpaso and ax.get_xscale() == "linear":
        ax.xaxis.set_major_locator(MultipleLocator(xpaso))
    if ax.get_xscale() == "linear":
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    if ypaso and ax.get_yscale() == "linear":
        ax.yaxis.set_major_locator(MultipleLocator(ypaso))
    if ax.get_yscale() == "linear":
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which="major", linestyle="-", linewidth=0.6, alpha=0.4)
    ax.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.28)
    ax.tick_params(which="both", direction="in", top=True, right=True, length=6)
    ax.tick_params(which="minor", length=3)


def segmentos(ax, x, y, mask, extender=True, **kw):
    """dibuja los tramos contiguos de mask sin unir los que estan separados."""
    x, y, mask = np.asarray(x), np.asarray(y), np.asarray(mask)
    i = 0
    primero = True
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(mask) and mask[j + 1]:
            j += 1
        lo, hi = (max(0, i - 1), min(len(mask) - 1, j + 1)) if extender else (i, j)
        ax.plot(x[lo:hi + 1], y[lo:hi + 1], **({**kw, "label": None} if not primero else kw))
        primero = False
        i = j + 1


def guarda(fig, nombre):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, nombre), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {OUT}/{nombre}")


# ----------------------------------------------------------------------
# datos
# ----------------------------------------------------------------------
def cargar():
    ex = json.load(open(f"{T20}/extrapolacion_t_inf.json"))
    d = {}
    for tag in PERFILES:
        d[tag] = {
            "t10": leer_csv(f"{T10}/polar_{tag}.csv"),
            "t20": leer_csv(f"{T20}/polar_{tag}.csv"),
            "inf": {f["alpha"]: f for f in ex["extrapolacion"][tag] if "nota" not in f},
        }
    return d, ex


def factores(datos, tag, col):
    """Dos etapas, separadas a proposito:

    medido[a]  = f(t=20)/f(t=10)   efecto de doblar la ventana, MEDIDO en los pares.
    resto[a]   = Finf/f(t=20)      lo que aun falta hasta la meseta, EXTRAPOLADO;
                                   1.0 donde no hay meseta que extraer.
    """
    medido, resto, hay_meseta = {}, {}, {}
    for a in sorted(set(datos[tag]["t10"]) & set(datos[tag]["t20"])):
        medido[a] = datos[tag]["t20"][a][col] / datos[tag]["t10"][a][col]
    for a, f in datos[tag]["inf"].items():
        if a not in medido:
            continue
        inf = f[col]["inf"]
        hay_meseta[a] = inf is not None
        resto[a] = inf / datos[tag]["t20"][a][col] if inf is not None else 1.0
    return medido, resto, hay_meseta


def corregida(datos, tag, col):
    """polar de 1 grado llevada a t->infinito. Devuelve (a, val, fiable).

    fiable = el angulo cae entre dos pares y ambos tienen meseta; si no, el valor
    lleva solo la correccion de ventana medida y es una cota inferior del asintotico.
    """
    medido, resto, hay = factores(datos, tag, col)
    if len(medido) < 2:
        return [], [], []
    xm = np.array(sorted(medido))
    xr = np.array(sorted(resto))
    a1 = np.array(sorted(datos[tag]["t10"]))
    base = np.array([datos[tag]["t10"][a][col] for a in a1])
    val = base * np.interp(a1, xm, [medido[k] for k in xm]) \
               * np.interp(a1, xr, [resto[k] for k in xr])
    con = np.array(sorted(k for k in hay if hay[k]))
    fiable = (a1 >= con[0]) & (a1 <= con[-1]) if len(con) else np.zeros(len(a1), bool)
    for a_bad in (k for k in hay if not hay[k]):
        fiable &= ~((a1 > a_bad - 2) & (a1 < a_bad + 2)) | (a1 == a_bad) & False
    return a1, val, fiable


# ----------------------------------------------------------------------
# 1-3. polares Cl, Cd, L/D
# ----------------------------------------------------------------------
CURVAS = [
    ("ld", r"$C_l/C_d$ [-]", "Eficiencia aerodinámica", "polar_eficiencia.png", 2.0),
    ("cl", r"$C_l$ [-]", "Coeficiente de sustentación", "polar_cl.png", 0.1),
    ("cd", r"$C_d$ [-]", "Coeficiente de resistencia", "polar_cd.png", 0.02),
]


def factor_malla():
    """Correccion de malla por magnitud, del estudio de 3 mallas de alpha=4.

    Richardson necesita convergencia monotona y aqui las tres magnitudes salen
    oscilatorias (razon de residuos negativa), asi que no hay valor extrapolado:
    se toma como estimacion de dx->0 la malla mas fina y como incertidumbre la
    banda entre las tres. El factor se aplica a TODOS los angulos, que es la
    suposicion fuerte de estas figuras: solo esta medido en alpha=4.
    """
    g = json.load(open("results/verificacion_numerica/gci_recalculado.json"))["gci_despues"]
    out = {}
    for k in ("cl", "cd", "ld"):
        v = g[k]["valores"]                      # [dx=0.004, 0.002, 0.001]
        out[k] = {"factor": v[2] / v[1], "banda_pct": g[k]["banda_pct"],
                  "convergencia": g[k]["convergencia"], "valores": v}
    return out


CASOS = [
    ("original", "0.45", (0, (6, 3)), 1.9, "medido: 1°, t≈10, dx=0.002"),
    ("malla",    "#2b6cb0", "-", 2.2, "extrapolado dx→0 (factor de α=4)"),
    ("tiempo",   "#1a7f37", "-", 2.2, "extrapolado t→∞"),
    ("fusion",   "#c1272d", "-", 3.0, "fusión: dx→0 y t→∞"),
]


def fig_polares(datos):
    fm = factor_malla()
    for col, ylabel, titulo, salida, ypaso in CURVAS:
        for tag, meta in PERFILES.items():
            a1, val_t, fiable = corregida(datos, tag, col)
            if not len(a1):
                continue
            a1 = np.asarray(a1)
            base = np.array([datos[tag]["t10"][a][col] for a in a1])
            k, banda = fm[col]["factor"], fm[col]["banda_pct"] / 100.0
            curvas = {"original": base, "malla": base * k,
                      "tiempo": val_t, "fusion": val_t * k}

            fig, ax = plt.subplots(figsize=(10.2, 7.0))
            for nombre, color, ls, lw, etq in CASOS:
                y = curvas[nombre]
                if nombre in ("tiempo", "fusion"):
                    segmentos(ax, a1, y, fiable, extender=False, color=color, lw=lw,
                              ls=ls, zorder=5, label=etq)
                    segmentos(ax, a1, y, ~fiable, color=color, lw=lw * 0.7,
                              ls=(0, (4, 2.5)), alpha=0.65, zorder=4,
                              label=f"{etq} — sin meseta, cota inferior")
                else:
                    ax.plot(a1, y, color=color, lw=lw, ls=ls, zorder=3, label=etq)
                if nombre in ("malla", "fusion"):
                    ax.fill_between(a1, y * (1 - banda), y * (1 + banda), color=color,
                                    alpha=0.13, lw=0, zorder=1)
            a20 = sorted(datos[tag]["t20"])
            ax.plot(a20, [datos[tag]["t20"][a][col] for a in a20], lw=0, marker=meta["marker"],
                    ms=7, color="0.25", mfc="white", zorder=6, label="medido: 2°, t≈20")

            ejes(ax, r"Ángulo de ataque $\alpha$ [°]", ylabel,
                 f"{titulo} — {meta['label']}\nRe=1e5, dominio C 24×16   "
                 f"(banda = incertidumbre de malla ±{fm[col]['banda_pct']:.1f} %)",
                 ypaso=ypaso)
            ax.set_xlim(-0.4, 10.4)
            ax.legend(frameon=True, framealpha=0.95, edgecolor="0.6", fontsize=10,
                      ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13))
            guarda(fig, salida.replace(".png", f"_{tag}.png"))


def fig_comparativa(datos):
    """una sola curva por perfil: la fusion. Para comparar sin ruido visual."""
    fm = factor_malla()
    fig, ax = plt.subplots(figsize=(10.2, 6.6))
    for tag, meta in PERFILES.items():
        a1, val_t, fiable = corregida(datos, tag, "ld")
        if not len(a1):
            continue
        a1, y = np.asarray(a1), np.asarray(val_t) * fm["ld"]["factor"]
        banda = fm["ld"]["banda_pct"] / 100.0
        segmentos(ax, a1, y, fiable, extender=False, color=meta["color"], lw=3.0,
                  zorder=5, label=f"{meta['label']} — dx→0 y t→∞")
        segmentos(ax, a1, y, ~fiable, color=meta["color"], lw=2.0, ls=(0, (4, 2.5)),
                  alpha=0.65, zorder=4, label=f"{meta['label']} — sin meseta")
        ax.fill_between(a1, y * (1 - banda), y * (1 + banda), color=meta["color"],
                        alpha=0.13, lw=0, zorder=1)
    ejes(ax, r"Ángulo de ataque $\alpha$ [°]", r"$C_l/C_d$ [-]",
         "Polares extrapoladas (dx→0 y t→∞) — Re=1e5, dominio C 24×16", ypaso=2.0)
    ax.set_xlim(-0.4, 10.4)
    ax.legend(frameon=True, framealpha=0.95, edgecolor="0.6", fontsize=10, ncol=2,
              loc="best")
    guarda(fig, "comparativa_extrapolada.png")


# ----------------------------------------------------------------------
# 4. polar de resistencia Cl vs Cd
# ----------------------------------------------------------------------
def fig_drag(datos):
    fig, ax = plt.subplots(figsize=(8.6, 7.0))
    for tag, meta in PERFILES.items():
        c, m = meta["color"], meta["marker"]
        a20 = sorted(datos[tag]["t20"])
        cd = [datos[tag]["t20"][a]["cd"] for a in a20]
        cl = [datos[tag]["t20"][a]["cl"] for a in a20]
        ax.plot(cd, cl, color=c, lw=2.0, marker=m, ms=8, mfc="white",
                label=f"{meta['label']} — t≈20")
        for a, x, y in zip(a20, cd, cl):
            ax.annotate(f"{a:.0f}°", (x, y), textcoords="offset points",
                        xytext=(7, -4), fontsize=10, color=c)
    ymax = 1.3
    for ld in (10, 20, 30):
        x = np.linspace(0.0, 0.19, 2)
        ax.plot(x, ld * x, color="0.65", lw=0.9, ls="--", zorder=0)
        ax.annotate(f"L/D={ld}", (ymax * 0.96 / ld, ymax * 0.96), color="0.45",
                    fontsize=10, ha="left", va="top", rotation=0)
    ejes(ax, r"$C_d$ [-]", r"$C_l$ [-]", "Polar de resistencia — Re=1e5, dx=0.002",
         xpaso=0.04, ypaso=0.2)
    ax.set_xlim(0, 0.19)
    ax.set_ylim(0, 1.3)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="0.6")
    guarda(fig, "polar_resistencia.png")


# ----------------------------------------------------------------------
# 5. convergencia temporal: cola de L/D frente a la longitud de la corrida
# ----------------------------------------------------------------------
def barrido_cola(path, n=70):
    t, cl, cd = serie(path)
    tmax = float(t[-1])
    Ts = np.linspace(2.0, tmax, n)
    out = [(T, cola(t, cl, cd, T)) for T in Ts]
    out = [(T, v[2]) for T, v in out if v is not None]
    return np.array([x[0] for x in out]), np.array([x[1] for x in out]), tmax


def fig_convergencia(datos):
    fig, axs = plt.subplots(1, 2, figsize=(14.5, 6.2), sharex=True)
    for ax, (tag, meta) in zip(axs, PERFILES.items()):
        cmap = plt.get_cmap("viridis")
        for i, a in enumerate(ANG):
            p = f"{T20}/series/{tag}_a{a:02d}.0_series.npz"
            if not os.path.exists(p):
                continue
            T, ld, tmax = barrido_cola(p)
            if len(T) < 3:
                continue
            c = cmap(i / (len(ANG) - 1))
            ax.plot(T, ld, color=c, lw=2.0, label=fr"$\alpha$={a}°")
            f = datos[tag]["inf"].get(float(a))
            if f and f["ld"]["inf"] is not None:
                ax.plot([T[-1] * 1.02], [f["ld"]["inf"]], color=c, marker="*", ms=14,
                        mec="black", mew=0.5, clip_on=False)
                ax.axhline(f["ld"]["inf"], color=c, lw=0.8, ls=":", alpha=0.55)
            if tmax < 19:
                ax.plot([tmax], [ld[-1]], color=c, marker="X", ms=9, mec="black", mew=0.6)
        ejes(ax, "longitud de la corrida $T$ [tiempos convectivos]",
             r"$\langle C_l\rangle/\langle C_d\rangle$ en $[T/2,\,T]$",
             meta["label"], xpaso=5.0)
        ax.axvline(10, color="0.5", lw=1.0, ls="--")
        ax.axvline(20, color="0.5", lw=1.0, ls="--")
        ax.annotate("t≈10", (10, ax.get_ylim()[0]), fontsize=10, color="0.4",
                    xytext=(3, 6), textcoords="offset points")
        ax.annotate("t≈20", (20, ax.get_ylim()[0]), fontsize=10, color="0.4",
                    xytext=(3, 6), textcoords="offset points")
        ax.legend(ncol=2, fontsize=10, frameon=True, framealpha=0.95, edgecolor="0.6")
    fig.suptitle("Convergencia temporal de la cola.  ★ = extrapolación t→∞   "
                 "✕ = corrida cortada por el guardián de estado estacionario", y=1.005)
    guarda(fig, "convergencia_temporal.png")


# ----------------------------------------------------------------------
# 6. validacion del metodo de extrapolacion (alpha=4, serie hasta t=41.5)
# ----------------------------------------------------------------------
def fig_validacion(ex):
    t, cl, cd = serie("results/asintotico_alpha4_domC/series.npz")
    Ts = np.linspace(2.0, float(t[-1]), 120)
    med = [(T, cola(t, cl, cd, T)) for T in Ts]
    med = [(T, v[2]) for T, v in med if v is not None]
    Tm, ldm = np.array([x[0] for x in med]), np.array([x[1] for x in med])
    verdad = ex["validacion"]["ld_verdad"]

    tri = [x for x in ex["validacion"]["triples"] if x["T"] == [5, 10, 20]][0]
    f1, f2, f3 = tri["ld"]
    r = exp_plateau(f1, f2, f3)
    tau = r["tau_sobre_T"] * 5.0
    A = (r["finf"] - f3) / np.exp(-20.0 / tau)
    Tg = np.linspace(4, 45, 200)

    e21, e32 = f2 - f1, f3 - f2
    p = np.log(e21 / e32) / np.log(2.0)
    rich = f3 + e32 / (2.0 ** p - 1)

    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    ax.plot(Tm, ldm, color="0.25", lw=2.4, label="medido (corrida hasta t=41.5)")
    ax.plot([5, 10, 20], [f1, f2, f3], "o", ms=11, color="#c1272d", mec="black",
            mew=0.6, zorder=6, label="colas usadas para ajustar (T=5, 10, 20)")
    ax.plot(Tg, r["finf"] - A * np.exp(-Tg / tau), color="#1a7f37", lw=2.0, ls="--",
            label=fr"ajuste exponencial ($\tau$={tau:.1f})")
    ax.axhline(verdad, color="black", lw=1.4, ls="-", alpha=0.8)
    ax.annotate(f"verdad medida = {verdad:.2f}", (3, verdad), ha="left", va="top",
                fontsize=11)
    ax.axhline(r["finf"], color="#1a7f37", lw=1.2, ls=":")
    ax.annotate(f"exponencial → {r['finf']:.2f}  ({(r['finf']-verdad)/verdad*100:+.1f} %)",
                (44, r["finf"]), ha="right", va="bottom", fontsize=11, color="#1a7f37")
    ax.axhline(rich, color="#b26b00", lw=1.2, ls=":")
    ax.annotate(f"Richardson → {rich:.2f}  ({(rich-verdad)/verdad*100:+.1f} %)",
                (44, rich), ha="right", va="bottom", fontsize=11, color="#b26b00")
    ejes(ax, "longitud de la corrida $T$ [tiempos convectivos]", r"$C_l/C_d$ [-]",
         "Validación de la extrapolación temporal — ganador, α=4°, dx=0.002",
         xpaso=5.0, ypaso=2.0)
    ax.set_xlim(2, 45)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="0.6")
    guarda(fig, "validacion_extrapolacion.png")


# ----------------------------------------------------------------------
# 7. cambio t=10 -> t=20
# ----------------------------------------------------------------------
def fig_deltas(ex):
    comp = ex["comparacion_t10_t20"]
    alphas = [r["alpha"] for r in comp["ganador_tfg2"]]
    x = np.arange(len(alphas))
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True)
    for ax, (col, nom) in zip(axs, (("cl", r"$C_l$"), ("cd", r"$C_d$"), ("ld", r"$C_l/C_d$"))):
        for k, (tag, meta) in enumerate(PERFILES.items()):
            v = [r[f"d_{col}_pct"] for r in comp[tag]]
            ax.bar(x + (k - 0.5) * 0.38, v, width=0.36, color=meta["color"],
                   alpha=0.85, edgecolor="black", linewidth=0.5, label=meta["label"])
        ax.axhline(0, color="black", lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a:.0f}°" for a in alphas])
        ejes(ax, r"$\alpha$", "cambio [%]", nom, xpaso=None, ypaso=5.0)
    axs[0].legend(frameon=True, framealpha=0.95, edgecolor="0.6", loc="best")
    fig.suptitle("Efecto de doblar la ventana temporal: t≈10 → t≈20   "
                 "(α=8 y 10 del ganador cortan en el mismo iterado: cambio nulo)", y=1.02)
    guarda(fig, "cambio_t10_t20.png")


# ----------------------------------------------------------------------
# 8. inestabilidad de la estela
# ----------------------------------------------------------------------
def fig_inestabilidad(datos):
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.set_yscale("log")
    for tag, meta in PERFILES.items():
        aa, sc, sd = [], [], []
        for a in ANG:
            p = f"{T20}/series/{tag}_a{a:02d}.0_series.npz"
            if not os.path.exists(p):
                continue
            t, cl, cd = serie(p)
            m = t >= t[-1] / 2
            if m.sum() < 10:
                continue
            aa.append(a)
            sc.append(cl[m].std() / abs(cl[m].mean()) * 100 if cl[m].mean() else np.nan)
            sd.append(cd[m].std() / abs(cd[m].mean()) * 100)
            if float(t[-1]) < 19:   # cortada por el guardian: campo congelado
                ax.plot([a, a], [sc[-1], sd[-1]], lw=0, marker="o", ms=17,
                        mfc="none", mec="0.35", mew=1.6, zorder=1)
        ax.plot(aa, sc, color=meta["color"], lw=2.0, marker=meta["marker"], ms=8,
                mfc="white", label=f"{meta['label']} — $C_l$")
        ax.plot(aa, sd, color=meta["color"], lw=2.0, ls="--", marker=meta["marker"],
                ms=8, label=f"{meta['label']} — $C_d$")
    ejes(ax, r"Ángulo de ataque $\alpha$ [°]",
         "fluctuación en la cola  $\\sigma/|\\mu|$ [%]",
         "Desprendimiento: dispersión de la señal en la segunda mitad de la corrida\n"
         "(círculo gris = corrida cortada por el guardián de estado estacionario)",
         ypaso=None)
    ax.legend(frameon=True, framealpha=0.95, edgecolor="0.6", ncol=2)
    guarda(fig, "inestabilidad.png")


# ----------------------------------------------------------------------
# 9. GCI de malla (alpha=4, dominio del GA)
# ----------------------------------------------------------------------
def fig_gci():
    g = json.load(open("results/verificacion_numerica/gci_recalculado.json"))["gci_despues"]
    dx = [0.004, 0.002, 0.001]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.0))
    for ax, (k, nom, c) in zip(axs, (("cl", r"$C_l$", "#0b4f9e"),
                                     ("cd", r"$C_d$", "#b26b00"),
                                     ("ld", r"$C_l/C_d$", "#c1272d"))):
        v = g[k]["valores"]
        ax.plot(dx, v, color=c, lw=2.2, marker="o", ms=10, mfc="white", mec=c)
        for x, y in zip(dx, v):
            ax.annotate(f"{y:.4g}", (x, y), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=10)
        ax.set_xscale("log")
        ax.set_xticks(dx)
        ax.set_xticklabels(["0.004", "0.002", "0.001"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.invert_xaxis()
        ejes(ax, "$dx$ [-]", f"{nom} [-]",
             f"{nom} — banda {g[k]['banda_pct']:.2f} %  ({g[k]['convergencia']})",
             xpaso=None, ypaso=None)
        ax.margins(y=0.22)
    fig.suptitle("Convergencia de malla del ganador (α=4°, t=12–71, dominio del GA 8×5): "
                 "Cd y L/D oscilan → no hay extrapolación a dx→0, solo banda", y=1.03)
    guarda(fig, "convergencia_malla_alpha4.png")


# ----------------------------------------------------------------------
# 10. ventaja del ganador
# ----------------------------------------------------------------------
def fig_ventaja(datos):
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for etq, clave, ls, alpha in (("t≈10 (1°)", "t10", ":", 0.5), ("t≈20 (2°)", "t20", "-", 1.0)):
        comunes = sorted(set(datos["ganador_tfg2"][clave]) & set(datos["naca0012"][clave]))
        r = [datos["ganador_tfg2"][clave][a]["ld"] / datos["naca0012"][clave][a]["ld"]
             for a in comunes if datos["naca0012"][clave][a]["ld"] > 0.1]
        aa = [a for a in comunes if datos["naca0012"][clave][a]["ld"] > 0.1]
        ax.plot(aa, r, lw=2.4, ls=ls, alpha=alpha, marker="o", ms=7, mfc="white",
                color="#c1272d", label=f"ventaja L/D — {etq}")
    ax.axhline(1.0, color="black", lw=1.2)
    ejes(ax, r"Ángulo de ataque $\alpha$ [°]", "L/D ganador / L/D NACA 0012 [-]",
         "Ventaja del perfil optimizado sobre la semilla", ypaso=0.2)
    ax.legend(frameon=True, framealpha=0.95, edgecolor="0.6", loc="best")
    guarda(fig, "ventaja_sobre_naca.png")


# ----------------------------------------------------------------------
def csv_corregida(datos):
    path = os.path.join(OUT, "polar_1grado_corregida.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["perfil", "alpha", "cl_t10", "cd_t10", "ld_t10",
                    "cl_tinf", "cd_tinf", "ld_tinf",
                    "cl_malla", "cd_malla", "ld_malla",
                    "cl_fusion", "cd_fusion", "ld_fusion", "meseta"])
        fm = factor_malla()
        for tag in PERFILES:
            cols = {c: corregida(datos, tag, c) for c in ("cl", "cd", "ld")}
            a1 = cols["ld"][0]
            for i, a in enumerate(a1):
                base = {c: datos[tag]["t10"][a][c] for c in ("cl", "cd", "ld")}
                tinf = {c: (cols[c][1][i] if len(cols[c][0]) else None) for c in ("cl", "cd", "ld")}
                fmt = lambda v: f"{v:.6f}" if v is not None else ""
                w.writerow([tag, a,
                            *[fmt(base[c]) for c in ("cl", "cd", "ld")],
                            *[fmt(tinf[c]) for c in ("cl", "cd", "ld")],
                            *[fmt(base[c] * fm[c]["factor"]) for c in ("cl", "cd", "ld")],
                            *[fmt(tinf[c] * fm[c]["factor"] if tinf[c] is not None else None)
                              for c in ("cl", "cd", "ld")],
                            bool(cols["ld"][2][i])])
    print(f"  {path}")


def main():
    datos, ex = cargar()
    print("figuras:")
    fig_polares(datos)
    fig_comparativa(datos)
    fig_drag(datos)
    fig_convergencia(datos)
    fig_validacion(ex)
    fig_deltas(ex)
    fig_inestabilidad(datos)
    fig_gci()
    fig_ventaja(datos)
    csv_corregida(datos)


if __name__ == "__main__":
    sys.exit(main())

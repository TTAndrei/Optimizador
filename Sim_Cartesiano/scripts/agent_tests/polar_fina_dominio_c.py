"""
Polar fina (alpha = 1..10 en pasos de 1 grado) de los dos perfiles del TFG a
dx=0.002, Re=1e5, en el **dominio C**: Lx=24, Ly=16, cx=6 (cy=Ly/2=8).

Repite el estudio de results/polar_fina_1grado, que se corrió en el dominio del
GA (Lx=8, Ly=5, cx=2). Ese dominio sesga: las fronteras top/bottom son `slip`
—paredes de túnel cerrado, no campo lejano— y la estela se trunca contra el p=0
de la salida. El estudio de independencia de dominio
(results/verificacion_numerica/estudio_dominio/) fijó C como el punto en el que
ambos errores caen por debajo del CI95 sin el sobrecoste de 40x24.

Los resultados de esta carpeta NO son comparables punto a punto con los de
polar_fina_1grado: distinto dominio, distinta magnitud del bloqueo.

Misma ruta de cálculo por lo demás (verificacion_numerica.simular): config de
referencia consistent + SA + SA-BC + MacCormack y criterio de parada calibrado.
Una simulación independiente por (perfil, ángulo).

Además del escalar, cada punto vuelca el campo final (u, v, p, máscara sólida y
los ejes de la malla estirada) en el instante de parada — por convergencia o por
agotar el contador — y de ahí salen dos imágenes: mapa de |u| y streamlines.

Reanudable: cada punto se cachea en su JSON y una segunda invocación salta los
ya hechos. Pausable con el centinela STOP_SIMULATION.trigger.

Uso:
    .venv/bin/python scripts/agent_tests/polar_fina_dominio_c.py
    .venv/bin/python scripts/agent_tests/polar_fina_dominio_c.py --solo-figuras
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
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from scipy.interpolate import RegularGridInterpolator

import verificacion_numerica as vn

OUT = os.path.join(ROOT, "results", "polar_fina_1grado_dom24x16")
CAMPOS = os.path.join(OUT, "campos")
SERIES = os.path.join(OUT, "series")
FIGS = os.path.join(OUT, "figuras")
for d in (OUT, CAMPOS, SERIES, FIGS):
    os.makedirs(d, exist_ok=True)

STOP = os.path.join(ROOT, "STOP_SIMULATION.trigger")

ANGULOS = tuple(float(a) for a in range(1, 11))
RE = 1e5
DX = 0.002

DOMINIO = {"Lx": 24.0, "Ly": 16.0, "cx": 6.0}

PERFILES = {
    "ganador_tfg2": {
        "dat": "results/convergence_study/islands_tfg2/NACA_0012_sharp/epoch10/"
               "NACA_0012_sharp_winner_Re100000_a4.0_LD27.75.dat",
        "label": "Perfil optimizado (AG)",
        "color": "#c0392b",
        "marker": "o",
        "ls": "-",
    },
    "naca0012": {
        "dat": "profiles/NACA_0012_sharp",
        "label": "NACA 0012",
        "color": "#1f4e79",
        "marker": "s",
        "ls": "--",
    },
}

COLS = ["alpha", "cl", "cd", "ld", "cl_ci95", "cd_ci95", "ld_ci95",
        "cl_std", "cd_std", "n_samples", "converged_clcd", "t_conv_clcd",
        "iters_efectivas", "iters_programadas", "cl_cp_discrepancy",
        "cl_cp_discrepancy_flag", "dx", "Re", "Lx", "Ly", "cx", "wall_s"]

# Recorte alrededor del perfil: el cuerpo va de x=cx a x=cx+chord, con cy=Ly/2.
CX, CY = DOMINIO["cx"], DOMINIO["Ly"] / 2.0
ZOOM_X = (CX - 0.45, CX + 1.85)
ZOOM_Y = (CY - 0.70, CY + 0.70)

# Escala de color común a todos los ángulos y a los dos perfiles: sin ella cada
# imagen se autoescala y dejan de ser comparables entre sí. Mismo VMAX que el
# estudio a 8x5 para poder comparar las imágenes de uno y otro dominio.
VMAX = 1.8


def json_path(tag):
    return os.path.join(OUT, f"polar_{tag}.json")


def csv_path(tag):
    return os.path.join(OUT, f"polar_{tag}.csv")


def campo_path(tag, a):
    return os.path.join(CAMPOS, f"{tag}_a{a:04.1f}.npz")


def serie_path(tag, a):
    """Sin esto solo queda el campo final y la evolucion temporal se pierde."""
    return os.path.join(SERIES, f"{tag}_a{a:04.1f}_series.npz")


def cargar(tag):
    p = json_path(tag)
    return json.load(open(p)) if os.path.exists(p) else {}


def escribir_csv(tag, out):
    filas = [out[f"{a:.1f}"] for a in ANGULOS if f"{a:.1f}" in out]
    with open(csv_path(tag), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in filas:
            w.writerow(r)


# ----------------------------------------------------------------------
# Campos: mapa de |u| y streamlines sobre malla uniforme interpolada
# ----------------------------------------------------------------------
def _remuestrear(npz, n=900):
    """La malla es cartesiana estirada; streamplot exige paso constante."""
    x, y = npz["x"], npz["y"]
    u, v, solid = npz["u"], npz["v"], npz["solid"].astype(np.float64)
    xi = np.linspace(*ZOOM_X, n)
    yi = np.linspace(*ZOOM_Y, int(n * (ZOOM_Y[1] - ZOOM_Y[0]) / (ZOOM_X[1] - ZOOM_X[0])))
    XI, YI = np.meshgrid(xi, yi)
    pts = np.column_stack([YI.ravel(), XI.ravel()])
    campos = []
    for f in (u, v, solid):
        g = RegularGridInterpolator((y, x), f, bounds_error=False, fill_value=None)
        campos.append(g(pts).reshape(XI.shape))
    return xi, yi, campos[0], campos[1], campos[2]


def _cuerpo(ax, xi, yi, mask):
    ax.contourf(xi, yi, mask, levels=[0.5, 2.0], colors=["#111111"], zorder=5)
    ax.contour(xi, yi, mask, levels=[0.5], colors=["#111111"], linewidths=1.0, zorder=6)


def _ejes(ax, titulo):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*ZOOM_X)
    ax.set_ylim(*ZOOM_Y)
    ax.set_xlabel("x [c]")
    ax.set_ylabel("y [c]")
    ax.set_title(titulo, fontsize=12)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which="both", direction="out", top=False, right=False)


def figuras_campo(tag, a, meta):
    f = campo_path(tag, a)
    if not os.path.exists(f):
        return
    npz = np.load(f)
    xi, yi, u, v, mask = _remuestrear(npz)
    speed = np.hypot(u, v)
    speed_plot = np.where(mask > 0.5, np.nan, speed)

    lab = PERFILES[tag]["label"]
    cab = (f"{lab} — α={a:.0f}°, Re=1e5, dx=0.002, dominio 24×16\n"
           f"campo final: {meta.get('iters_efectivas')} iters"
           f"{', convergido' if meta.get('converged_clcd') else ', fin de presupuesto'}")

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    im = ax.pcolormesh(xi, yi, speed_plot, cmap="turbo", shading="auto",
                       vmin=0.0, vmax=VMAX)
    _cuerpo(ax, xi, yi, mask)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(r"$|\mathbf{u}|/U_\infty$ [-]")
    _ejes(ax, "Campo de velocidad — " + cab)
    fig.tight_layout()
    fig.savefig(os.path.join(CAMPOS, f"{tag}_a{a:04.1f}_velocidad.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    ax.set_facecolor("#f7f7f7")
    st = ax.streamplot(xi, yi, np.where(mask > 0.5, 0.0, u),
                       np.where(mask > 0.5, 0.0, v),
                       color=speed_plot, cmap="turbo", norm=Normalize(0.0, VMAX),
                       density=2.4, linewidth=0.9, arrowsize=0.8)
    _cuerpo(ax, xi, yi, mask)
    cb = fig.colorbar(st.lines, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(r"$|\mathbf{u}|/U_\infty$ [-]")
    _ejes(ax, "Líneas de corriente — " + cab)
    fig.tight_layout()
    fig.savefig(os.path.join(CAMPOS, f"{tag}_a{a:04.1f}_streamlines.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Comparativas Cl, Cd y L/D
# ----------------------------------------------------------------------
GRAFICOS = [
    ("cl", "cl_ci95", r"$C_l$", "Coeficiente de sustentación",
     "comparativa_cl.png", 0.1),
    ("cd", "cd_ci95", r"$C_d$", "Coeficiente de resistencia",
     "comparativa_cd.png", 0.01),
    ("ld", "ld_ci95", r"$C_l/C_d$", "Eficiencia aerodinámica",
     "comparativa_eficiencia.png", 2.0),
]


def figuras_polar(datos):
    plt.rcParams.update({
        "font.size": 13, "axes.labelsize": 15, "axes.titlesize": 16,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    })
    for col, err, ylabel, titulo, salida, paso in GRAFICOS:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        hay = False
        for tag, meta in PERFILES.items():
            d = datos.get(tag, {})
            a = [x for x in ANGULOS if f"{x:.1f}" in d]
            if len(a) < 2:
                continue
            hay = True
            r = [d[f"{x:.1f}"] for x in a]
            ax.errorbar(a, [q[col] for q in r],
                        yerr=[q.get(err) or 0.0 for q in r],
                        label=meta["label"], color=meta["color"],
                        marker=meta["marker"], markersize=6.5,
                        linestyle=meta["ls"], linewidth=2.0, capsize=4,
                        elinewidth=1.2, markeredgecolor="white",
                        markeredgewidth=0.7)
        if not hay:
            plt.close(fig)
            continue
        ax.set_xlabel(r"Ángulo de ataque $\alpha$ [°]")
        ax.set_ylabel(f"{ylabel} [-]")
        ax.set_title(f"{titulo} — Re = 100 000, dx = 0.002, dominio 24×16")
        ax.set_xlim(0.65, 10.35)
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(paso))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(which="major", linestyle="-", linewidth=0.6, alpha=0.45)
        ax.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.3)
        ax.tick_params(which="both", direction="in", top=True, right=True, length=6)
        ax.tick_params(which="minor", length=3)
        ax.legend(frameon=True, framealpha=0.95, edgecolor="0.6", loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, salida), dpi=200, bbox_inches="tight")
        plt.close(fig)


def rehacer_salidas(datos):
    for tag, d in datos.items():
        escribir_csv(tag, d)
    figuras_polar(datos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-figuras", action="store_true")
    args = ap.parse_args()

    datos = {tag: cargar(tag) for tag in PERFILES}

    if args.solo_figuras:
        for tag, d in datos.items():
            for a in ANGULOS:
                k = f"{a:.1f}"
                if k in d:
                    figuras_campo(tag, a, d[k])
        rehacer_salidas(datos)
        vn._log(f"figuras regeneradas en {OUT}")
        return

    crit = vn.criterio_calibrado()
    vn._log(f"criterio de parada: {crit or 'FICHERO AUSENTE -> defaults de main()'}")
    vn._log(f"dominio: {DOMINIO} (cy={CY:g})  ->  {OUT}")
    vn._log(f"alphas={list(ANGULOS)}  Re={RE:.0e}  dx={DX}  "
            f"presupuesto/punto={vn.iters_for(DX)} iters")
    pendientes = sum(1 for a in ANGULOS for t in PERFILES
                     if f"{a:.1f}" not in datos[t])
    vn._log(f"puntos pendientes: {pendientes} de {len(ANGULOS) * len(PERFILES)}")

    for a in ANGULOS:
        for tag, meta in PERFILES.items():
            k = f"{a:.1f}"
            if k in datos[tag]:
                continue
            if os.path.exists(STOP):
                vn._log(f"centinela {STOP} presente -> parada")
                rehacer_salidas(datos)
                return
            vn._log(f"--- {tag}  alpha={a:.1f}deg ---")
            r = vn.simular(meta["dat"], RE, DX, alpha=a, extra=DOMINIO,
                           dump=serie_path(tag, a), dump_field=campo_path(tag, a))
            if r is None:
                vn._log(f"{tag} alpha={a}: simulación fallida")
                continue
            r.update(DOMINIO)
            datos[tag][k] = r
            with open(json_path(tag), "w") as f:
                json.dump(datos[tag], f, indent=2, ensure_ascii=False)
            vn._log(f"{tag} alpha={a}: Cl={r['cl']:.4f} Cd={r['cd']:.5f} "
                    f"L/D={r['ld']:.2f} conv={r['converged_clcd']} "
                    f"t_conv={r['t_conv_clcd']} n={r['n_samples']} ({r['wall_s']}s)")
            figuras_campo(tag, a, r)
            escribir_csv(tag, datos[tag])
            figuras_polar(datos)

    rehacer_salidas(datos)

    print("\n" + "=" * 86)
    print(f"{'perfil':>14s} {'alpha':>6s} {'Cl':>9s} {'Cd':>10s} {'L/D':>8s} "
          f"{'conv':>6s} {'t_conv':>8s} {'n':>5s} {'s':>8s}")
    print("-" * 86)
    for tag in PERFILES:
        for a in ANGULOS:
            k = f"{a:.1f}"
            if k not in datos[tag]:
                continue
            r = datos[tag][k]
            tc = r.get("t_conv_clcd")
            print(f"{tag:>14s} {a:>6.1f} {r['cl']:>9.4f} {r['cd']:>10.5f} "
                  f"{r['ld']:>8.2f} {str(r['converged_clcd']):>6s} "
                  f"{(f'{tc:.2f}' if tc is not None else '-'):>8s} "
                  f"{r['n_samples']:>5d} {r['wall_s']:>8.0f}")
    print("=" * 86)


if __name__ == "__main__":
    main()

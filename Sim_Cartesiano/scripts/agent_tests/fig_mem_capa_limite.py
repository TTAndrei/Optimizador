"""
Figura 1.C — Crecimiento de la capa limite, gradiente adverso y separacion.

Tres cosas en un solo dibujo, sobre el extrados de un perfil estirado a linea
recta:

  1. la capa limite engorda aguas abajo
  2. tras el pico de succion el gradiente de presion se vuelve adverso y el
     perfil de velocidad se vacia por abajo
  3. cuando el esfuerzo en la pared se anula, el flujo se invierte y la capa se
     separa

Los perfiles de velocidad no estan dibujados a mano: son soluciones de
Falkner-Skan,

    f''' + f f'' + beta (1 - f'^2) = 0,   f(0) = f'(0) = 0,  f'(inf) = 1,

integradas por disparo sobre f''(0). El parametro beta es el del flujo exterior
U_e ~ x^m con beta = 2m/(m+1): beta > 0 es gradiente favorable, beta = 0 es
Blasius y beta < 0 adverso. La separacion cae en beta = -0.19884, donde
f''(0) = 0, y ese es el ultimo perfil que el modelo admite: pasado ese punto no
hay solucion de capa limite, y el perfil invertido que se dibuja detras es ya un
esquema.

La curvatura en la pared no es un adorno: de la ecuacion de cantidad de
movimiento en y = 0, mu d2u/dy2 = dp/dx. Con gradiente adverso la curvatura en
la pared es positiva, aparece un punto de inflexion dentro de la capa y con el
la inestabilidad que acaba en separacion.

Uso:
    .venv/bin/python scripts/agent_tests/fig_mem_capa_limite.py
"""
from __future__ import annotations

import os

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "figuras_memoria")
DPI = 200

C_FAV = "#1f5fa8"
C_ADV = "#c0392b"
C_PAR = "0.25"
C_AUX = "0.45"
C_MARK = "#7a5195"

ETA_INF = 9.0
BETA_SEP = -0.198838


def falkner_skan(beta, eta_inf=ETA_INF, n=600):
    """Resuelve el problema por disparo sobre f''(0). Devuelve eta, f', f''."""
    def rhs(_, y):
        f, fp, fpp = y
        return [fp, fpp, -f * fpp - beta * (1.0 - fp ** 2)]

    def falta(s):
        sol = solve_ivp(rhs, (0.0, eta_inf), [0.0, 0.0, s], rtol=1e-10,
                        atol=1e-12, dense_output=True)
        return sol.y[1, -1] - 1.0

    # f''(0) crece con beta. En la separacion la raiz es exactamente f''(0) = 0,
    # que es un extremo del intervalo y el disparo no puede encerrar: se impone.
    if beta <= BETA_SEP + 1e-6:
        s = 0.0
        assert abs(falta(0.0)) < 1e-5, "beta de separacion mal fijado"
    else:
        s = brentq(falta, 1e-3, 2.0, xtol=1e-12)
    eta = np.linspace(0.0, eta_inf, n)
    sol = solve_ivp(rhs, (0.0, eta_inf), [0.0, 0.0, s], t_eval=eta,
                    rtol=1e-10, atol=1e-12)
    return eta, sol.y[1], sol.y[2], s


def perfil_invertido(eta, eta99):
    """Perfil esquematico tras la separacion: flujo hacia atras pegado a la
    pared y recuperacion por encima. No sale de Falkner-Skan, que ahi ya no
    tiene solucion."""
    z = eta / eta99
    return np.tanh(2.4 * np.clip(z - 0.42, 0.0, None)) - 0.30 * np.exp(-((z - 0.12) / 0.16) ** 2)


def main():
    os.makedirs(OUT, exist_ok=True)

    # --- perfiles -----------------------------------------------------------
    betas = [0.5, 0.0, -0.12, BETA_SEP]
    perfiles = []
    for b in betas:
        eta, fp, fpp, s0 = falkner_skan(b)
        eta99 = float(np.interp(0.99, fp, eta))
        perfiles.append(dict(beta=b, eta=eta, u=fp, upp=fpp, f2_0=s0, eta99=eta99))
        print(f"  beta = {b:+.5f}   f''(0) = {s0:.5f}   eta_99 = {eta99:.3f}")
    assert abs(perfiles[-1]["f2_0"]) < 1e-6, "el perfil de separacion no da tau_w = 0"

    # estaciones en x y espesor de la capa: crecimiento tipo sqrt(x) en el tramo
    # favorable y engorde rapido en cuanto el gradiente se vuelve adverso
    x_pico = 0.30
    x_sep = 0.72
    xs = np.linspace(0.02, 1.0, 400)
    delta = 0.030 * np.sqrt(xs / x_pico)
    delta = delta * (1.0 + 1.35 * np.clip(xs - x_pico, 0.0, None) ** 1.2
                     / (x_sep - x_pico) ** 1.2)
    d_de = lambda xq: float(np.interp(xq, xs, delta))

    x_est = [0.10, 0.30, 0.52, x_sep]
    x_inv = 0.90

    # --- figura -------------------------------------------------------------
    fig, (axu, ax) = plt.subplots(2, 1, figsize=(9.6, 7.2), sharex=True,
                                  gridspec_kw={"height_ratios": [1.0, 2.5]})
    fig.subplots_adjust(left=0.085, right=0.985, top=0.935, bottom=0.085,
                        hspace=0.08)

    # ---- panel superior: velocidad exterior y gradiente de presion
    ue = np.where(xs < x_pico,
                  1.0 + 0.55 * np.sqrt(np.clip(xs, 0.0, None) / x_pico),
                  1.55 - 0.62 * ((xs - x_pico) / (1.0 - x_pico)) ** 0.85)
    axu.axvspan(0.0, x_pico, color="#eaf2fa", zorder=0)
    axu.axvspan(x_pico, 1.02, color="#fdeeee", zorder=0)
    axu.plot(xs, ue, color="0.15", lw=2.0, zorder=3)
    axu.axvline(x_pico, color=C_AUX, lw=1.0, ls="--", zorder=2)
    axu.axvline(x_sep, color=C_ADV, lw=1.0, ls=":", zorder=2)
    axu.text(x_sep - 0.012, 0.70, "separación", fontsize=8.8, color=C_ADV,
             ha="right", va="center")
    axu.text(0.5 * x_pico, 0.75, "favorable\n" r"$\mathrm{d}p/\mathrm{d}x < 0$",
             fontsize=9.2, color=C_FAV, ha="center", va="center")
    axu.text(0.68, 1.45, "adverso\n" r"$\mathrm{d}p/\mathrm{d}x > 0$",
             fontsize=9.2, color=C_ADV, ha="center", va="center")
    axu.annotate("pico de succión", xy=(x_pico, np.interp(x_pico, xs, ue)),
                 xytext=(0.365, 1.82), fontsize=9.0, color="0.2",
                 arrowprops=dict(arrowstyle="->", color=C_AUX, lw=1.0))
    axu.set_ylim(0.55, 2.05)
    axu.set_ylabel(r"$U_e / U_\infty$")
    axu.set_title("Capa límite sobre el extradós: crecimiento, gradiente adverso y separación",
                  fontsize=11.5)
    axu.grid(alpha=0.15, lw=0.6)

    # ---- panel inferior: capa limite
    ax.axvspan(0.0, x_pico, color="#eaf2fa", zorder=0)
    ax.axvspan(x_pico, 1.02, color="#fdeeee", zorder=0)

    # pared
    ax.axhline(0.0, color=C_PAR, lw=2.6, zorder=6)
    ax.fill_between([-0.02, 1.02], -0.028, 0.0, color="0.82", zorder=5)
    ax.text(0.015, -0.022, "pared", fontsize=9.0, color="0.3", zorder=7)

    # espesor
    m_d = xs <= x_sep                      # pasado el desprendimiento delta pierde sentido
    ax.plot(xs[m_d], delta[m_d], color="0.35", lw=1.5, ls="--", zorder=4)
    ax.annotate(r"$\delta(x)$", xy=(0.235, d_de(0.235)), xytext=(0.195, 0.105),
                fontsize=11, color="0.3",
                arrowprops=dict(arrowstyle="->", color=C_AUX, lw=1.0))

    ESC = 0.105          # cuanto mide en x un u/U_e = 1
    etiquetas = ["perfil lleno", "Blasius", "punto de inflexión",
                 r"$\tau_w = 0$"]

    for (xe, pf, txt) in zip(x_est, perfiles, etiquetas):
        d = d_de(xe)
        y = pf["eta"] / pf["eta99"] * d
        u = pf["u"]
        m = y <= 1.12 * d
        col = C_FAV if pf["beta"] > 0 else (C_ADV if pf["beta"] < 0 else "0.25")
        ax.plot(xe + ESC * u[m], y[m], color=col, lw=1.9, zorder=8)
        ax.plot([xe, xe], [0.0, y[m][-1]], color=col, lw=0.8, ls=":", zorder=7)
        # flechas del perfil
        for yy in np.linspace(0.08, 1.0, 5) * y[m][-1]:
            uu = float(np.interp(yy, y, u))
            ax.annotate("", xy=(xe + ESC * uu, yy), xytext=(xe, yy),
                        arrowprops=dict(arrowstyle="->", color=col, lw=0.9,
                                        shrinkA=0, shrinkB=0), zorder=8)
        # punto de inflexion de u(y): d2u/dy2 = 0, o sea f''' = 0, que es el
        # maximo de f''. Solo existe dentro de la capa con gradiente adverso.
        if pf["beta"] < 0:
            i_inf = int(np.argmax(pf["upp"]))
            if 0 < i_inf < len(y) - 1:
                ax.plot([xe + ESC * u[i_inf]], [y[i_inf]], marker="o", ms=6.0,
                        mfc="white", mec=C_MARK, mew=1.7, zorder=10)
        ax.text(xe, 1.16 * d + 0.010, txt, fontsize=8.8, color=col, ha="center",
                va="bottom", zorder=9)

    # perfil invertido tras la separacion
    d = d_de(x_inv)
    eta = perfiles[-1]["eta"]
    y = eta / perfiles[-1]["eta99"] * d
    u_inv = perfil_invertido(eta, perfiles[-1]["eta99"])
    m = y <= 1.15 * d
    ax.plot(x_inv + ESC * u_inv[m], y[m], color=C_ADV, lw=1.9, zorder=8)
    ax.plot([x_inv, x_inv], [0.0, y[m][-1]], color=C_ADV, lw=0.8, ls=":", zorder=7)
    for yy in np.linspace(0.05, 1.0, 5) * y[m][-1]:
        uu = float(np.interp(yy, y, u_inv))
        ax.annotate("", xy=(x_inv + ESC * uu, yy), xytext=(x_inv, yy),
                    arrowprops=dict(arrowstyle="->", color=C_ADV, lw=0.9,
                                    shrinkA=0, shrinkB=0), zorder=8)
    ax.text(x_inv, 1.20 * d + 0.010, "flujo invertido", fontsize=8.8,
            color=C_ADV, ha="center", va="bottom", zorder=9)

    # linea divisoria y punto de separacion
    xd = np.linspace(x_sep, 1.02, 120)
    yd = 0.95 * d_de(x_sep) * (1.0 - np.exp(-((xd - x_sep) / 0.11) ** 1.4))
    ax.fill_between(xd, 0.0, yd, color="#e9a7a4", alpha=0.30, lw=0.0, zorder=1)
    ax.plot([x_sep], [0.0], marker="o", ms=9, mfc=C_MARK, mec="k", mew=0.9,
            zorder=10)
    ax.annotate("separación", xy=(x_sep, 0.0), xytext=(0.615, -0.075),
                fontsize=9.5, color="0.15", ha="center",
                arrowprops=dict(arrowstyle="->", color=C_AUX, lw=1.0))
    # corriente exterior
    for yy in (0.178, 0.212):
        ax.annotate("", xy=(0.140, yy), xytext=(0.025, yy),
                    arrowprops=dict(arrowstyle="->", color="0.55", lw=1.2))
    ax.text(0.082, 0.224, r"$U_\infty$", fontsize=10, color="0.4", ha="center")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.105, 0.265)
    ax.set_xlabel("distancia recorrida sobre la superficie")
    ax.set_ylabel("distancia a la pared")
    ax.set_yticks([])
    ax.set_xticks([])
    for lado in ("left", "right", "top"):
        ax.spines[lado].set_visible(False)

    p = os.path.join(OUT, "fig_1C_capa_limite.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("escrito:", p)


if __name__ == "__main__":
    main()

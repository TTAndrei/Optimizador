"""
Figura 1.B — Distribucion tipica de Cp sobre un perfil que sustenta.

Figura de definiciones: sirve para nombrar el punto de remanso, el pico de
succion, la recuperacion de presion del extrados y el hecho de que la
sustentacion es el area encerrada entre las dos ramas.

El Cp no esta dibujado a mano: sale de un metodo de paneles de Hess-Smith
(fuentes de intensidad constante por panel + un torbellino comun, condicion de
Kutta en el borde de salida) sobre un NACA 2412 con 200 paneles en reparto
coseno. Es flujo potencial, sin capa limite: por eso vale como referencia de la
forma de la curva —y solo de eso—, ya que la separacion que discute el
capitulo siguiente es justo lo que este modelo no ve.

Comprobaciones que hace el propio script al generar la figura:
  - Cp en el punto de remanso ~ 1
  - alpha de sustentacion nula frente al -2.1 deg tabulado del NACA 2412
  - Cl del metodo de paneles frente a 2*pi*(alpha - alpha_L0)

Uso:
    .venv/bin/python scripts/agent_tests/fig_mem_cp_perfil.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "figuras_memoria")
DPI = 200

C_EXT = "#1f5fa8"   # extrados
C_INT = "#c0392b"   # intrados
C_AUX = "0.45"
C_MARK = "#7a5195"


def naca4(m_p_t="2412", n=100):
    """Coordenadas NACA de 4 digitos, reparto coseno, de TE a LE por el intrados
    y de vuelta al TE por el extrados (orden que pide Hess-Smith)."""
    m = int(m_p_t[0]) / 100.0
    p = int(m_p_t[1]) / 10.0
    t = int(m_p_t[2:]) / 100.0

    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))                     # 0 -> 1, denso en LE y TE

    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                    + 0.2843 * x ** 3 - 0.1036 * x ** 4)   # cierre exacto en TE

    yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2),
                  m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
    dyc = np.where(x < p, 2 * m / p ** 2 * (p - x),
                   2 * m / (1 - p) ** 2 * (p - x))
    th = np.arctan(dyc)

    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)

    # TE -> intrados -> LE -> extrados -> TE
    X = np.concatenate([xl[::-1], xu[1:]])
    Y = np.concatenate([yl[::-1], yu[1:]])
    return X, Y


def hess_smith(X, Y, alpha_deg):
    """Fuentes constantes por panel + torbellino comun. Devuelve Cp, vt, Cl."""
    a = np.radians(alpha_deg)
    N = len(X) - 1

    xc = 0.5 * (X[:-1] + X[1:])
    yc = 0.5 * (Y[:-1] + Y[1:])
    dx, dy = X[1:] - X[:-1], Y[1:] - Y[:-1]
    s = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)

    # geometria panel j visto desde el punto de control i
    xi = xc[:, None]
    yi = yc[:, None]
    r1 = np.hypot(xi - X[None, :-1], yi - Y[None, :-1])
    r2 = np.hypot(xi - X[None, 1:], yi - Y[None, 1:])
    # angulo subtendido, con el signo del lado desde el que se ve el panel
    dth = th[None, :]
    e1x, e1y = (xi - X[None, :-1]), (yi - Y[None, :-1])
    e2x, e2y = (xi - X[None, 1:]), (yi - Y[None, 1:])
    cross = e1x * e2y - e1y * e2x
    dot = e1x * e2x + e1y * e2y
    beta = np.arctan2(cross, dot)
    np.fill_diagonal(beta, np.pi)

    dti = th[:, None] - dth
    log = np.log(r1 / r2)
    np.fill_diagonal(log, 0.0)

    # Influencias en ejes del panel j (xi a lo largo, eta normal), por unidad de
    # intensidad:  fuente -> (log, beta)/2pi ;  torbellino -> (-beta, log)/2pi.
    # Proyectadas sobre la normal y la tangente del panel i:
    An = (-np.sin(dti) * log + np.cos(dti) * beta) / (2.0 * np.pi)   # fuente, normal
    At = (np.cos(dti) * log + np.sin(dti) * beta) / (2.0 * np.pi)    # fuente, tangencial
    Vn, Vt = At, -An                                                 # torbellino

    M = np.zeros((N + 1, N + 1))
    rhs = np.zeros(N + 1)
    M[:N, :N] = An
    M[:N, N] = Vn.sum(axis=1)
    rhs[:N] = np.sin(th - a)                       # -U_inf . n

    # Kutta: la velocidad tangencial de los dos paneles del borde de salida es
    # opuesta, porque sus tangentes apuntan en sentidos contrarios
    M[N, :N] = At[0, :] + At[N - 1, :]
    M[N, N] = (Vt[0, :] + Vt[N - 1, :]).sum()
    rhs[N] = -(np.cos(th[0] - a) + np.cos(th[N - 1] - a))

    sol = np.linalg.solve(M, rhs)
    q, gamma = sol[:N], sol[N]

    vt = np.cos(th - a) + At @ q + gamma * Vt.sum(axis=1)
    cp = 1.0 - vt ** 2

    # Kutta-Joukowski con la circulacion de la propia solucion, integrada en el
    # sentido en que estan ordenados los paneles (horario, del TE al TE)
    cl_kj = 2.0 * float(np.sum(vt * s))

    # integral de presion, que es la que se dibuja: n = (-sin th, cos th) sale
    # del perfil, y el Cl es la componente perpendicular a la corriente
    nx, ny = -np.sin(th), np.cos(th)
    cl_p = -np.sum(cp * s * (ny * np.cos(a) - nx * np.sin(a)))
    return xc, yc, cp, vt, cl_kj, cl_p, q, gamma


def cl_de_alpha(X, Y, alpha_deg):
    return hess_smith(X, Y, alpha_deg)[4]


def main():
    os.makedirs(OUT, exist_ok=True)
    alpha = 6.0
    X, Y = naca4("2412", n=101)
    xc, yc, cp, vt, cl, cl_pres = hess_smith(X, Y, alpha)[:6]

    # --- comprobaciones -----------------------------------------------------
    cl0 = cl_de_alpha(X, Y, 0.0)
    a0_num = (cl_de_alpha(X, Y, 1.0) - cl_de_alpha(X, Y, -1.0)) / np.radians(2.0)
    alpha_L0 = -np.degrees(cl0 / a0_num)
    cl_delgado = 2.0 * np.pi * np.radians(alpha - alpha_L0)
    print(f"  paneles: {len(xc)}   alpha = {alpha:.1f} deg")
    print(f"  Cp maximo (remanso) = {cp.max():.4f}   (exacto 1)")
    print(f"  a0 = {a0_num:.3f} /rad   (2*pi = {2*np.pi:.3f})")
    print(f"  alpha_L0 = {alpha_L0:.2f} deg   (tabulado NACA 2412: -2.1)")
    print(f"  Cl Kutta-Joukowski = {cl:.4f}   integral de presion = {cl_pres:.4f}"
          f"   perfil delgado = {cl_delgado:.4f}")
    assert abs(cl - cl_pres) < 0.01, "las dos integrales de Cl no coinciden"
    assert abs(cp.max() - 1.0) < 0.02, "el punto de remanso no da Cp = 1"

    n = len(xc)
    i_le = int(np.argmin(xc))
    x_int, cp_int = xc[:i_le + 1], cp[:i_le + 1]        # intrados (TE -> LE)
    x_ext, cp_ext = xc[i_le:], cp[i_le:]                # extrados (LE -> TE)

    i_pico = i_le + int(np.argmin(cp[i_le:]))
    x_pico, cp_pico = xc[i_pico], cp[i_pico]
    i_rem = int(np.argmax(cp))

    # --- figura -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.6, 5.3))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.915, bottom=0.115)

    # gradiente adverso sobre el extrados: desde el pico hasta el borde de salida
    ax.axvspan(x_pico, 1.0, color="#fdf4f4", zorder=0)

    ax.axhline(0.0, color="0.35", lw=1.0, zorder=1)
    ax.fill_between(x_ext, cp_ext, np.interp(x_ext, x_int[::-1], cp_int[::-1]),
                    color="#dfe8f2", alpha=0.75, zorder=1)

    ax.plot(x_ext, cp_ext, color=C_EXT, lw=2.2, zorder=4, label="extradós")
    ax.plot(x_int, cp_int, color=C_INT, lw=2.2, zorder=4, label="intradós")

    ax.plot([x_pico], [cp_pico], marker="o", ms=8, mfc=C_MARK, mec="k",
            mew=0.8, zorder=6)
    ax.annotate("pico de succión\n" rf"($C_p = {cp_pico:.2f}$, $x/c = {x_pico:.3f}$)",
                xy=(x_pico, cp_pico), xytext=(0.135, cp_pico - 0.55),
                fontsize=9.5, color="0.15",
                arrowprops=dict(arrowstyle="->", color=C_AUX, lw=1.0))

    ax.plot([xc[i_rem]], [cp[i_rem]], marker="o", ms=7, mfc="white",
            mec=C_INT, mew=1.8, zorder=6)
    ax.annotate("punto de remanso\n" r"($C_p = 1$)",
                xy=(xc[i_rem], cp[i_rem]), xytext=(0.135, 1.30),
                fontsize=9.5, color="0.15",
                arrowprops=dict(arrowstyle="->", color=C_AUX, lw=1.0))

    ax.annotate("recuperación de presión:\n" r"$\mathrm{d}p/\mathrm{d}x > 0$ (gradiente adverso)",
                xy=(0.55, np.interp(0.55, x_ext, cp_ext)), xytext=(0.46, -1.85),
                fontsize=9.5, color="#8c2b2b", ha="left",
                arrowprops=dict(arrowstyle="->", color="#c0797a", lw=1.1))

    ax.annotate("borde de salida:\nlas dos ramas se juntan\n(condición de Kutta)",
                xy=(1.0, 0.5 * (cp_ext[-1] + cp_int[0])), xytext=(0.66, 1.22),
                fontsize=9.0, color="0.35", ha="center",
                arrowprops=dict(arrowstyle="->", color=C_AUX, lw=1.0))

    ax.invert_yaxis()
    # se baja el limite inferior para hacerle sitio al perfil dentro del mismo eje
    ax.set_ylim(2.45, cp_pico - 0.95)
    ax.set_ylabel(r"$C_p = (p - p_\infty)\,/\,\frac{1}{2}\rho U_\infty^{2}$")
    ax.set_title(rf"NACA 2412, $\alpha = {alpha:.0f}^\circ$ — método de paneles "
                 "(flujo potencial)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    ax.grid(alpha=0.18, lw=0.6)
    # sin marca en Cp=2: cae sobre la silueta y se lee como si fuera un dato
    ax.set_yticks([-3, -2, -1, 0, 1])
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel(r"$x/c$")

    # --- perfil dentro del propio eje, alineado en x -------------------------
    # eje superpuesto en vez de dibujar el perfil en unidades de Cp: asi la
    # silueta conserva su proporcion y sigue cuadrando con el eje x de arriba
    axg = ax.inset_axes([0.0, 0.0, 1.0, 0.175])
    axg.set_facecolor("none")
    axg.plot(X, Y, color="0.2", lw=1.3)
    axg.fill(X, Y, color="0.88", zorder=0)
    axg.plot([x_pico], [np.interp(x_pico, xc[i_le:], yc[i_le:])], marker="o",
             ms=6, mfc=C_MARK, mec="k", mew=0.7, zorder=5)
    axg.set_xlim(-0.03, 1.03)
    axg.set_ylim(-0.13, 0.13)
    axg.set_xticks([])
    axg.set_yticks([])
    for lado in ("left", "right", "top", "bottom"):
        axg.spines[lado].set_visible(False)

    p = os.path.join(OUT, "fig_1B_cp_perfil.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("escrito:", p)


if __name__ == "__main__":
    main()

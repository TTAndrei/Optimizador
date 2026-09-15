"""
Figura 1.D — Campo de presion alrededor de un perfil que sustenta.

Version "de portada" de la figura de definiciones: corriente uniforme por la
izquierda y, sobre el perfil, el campo de Cp completo, para que se vean de un
vistazo la zona de sobrepresion del intrados/borde de ataque y la de succion
del extrados. Casi sin texto: U_inf, alpha, L, D y la barra de Cp.

El campo no esta dibujado a mano. Sale del mismo metodo de paneles de
Hess-Smith de la figura 1.B (fuentes constantes por panel + torbellino comun,
Kutta en el borde de salida): con las intensidades ya resueltas se evalua la
velocidad inducida en cada punto de una rejilla y se aplica Bernoulli,
Cp = 1 - |V|^2 / U_inf^2. Flujo potencial, sin capa limite: vale para la forma
del campo, no para la estela ni para la resistencia.

El perfil se gira -alpha y se resuelve con incidencia nula, asi la corriente
libre queda horizontal en el dibujo.

Uso:
    .venv/bin/python scripts/agent_tests/fig_mem_campo_presion.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Arc, Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_mem_cp_perfil import naca4, hess_smith  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "figuras_memoria")
DPI = 220

C_FLUJO = "#2b6cb0"
C_L = "#1a7f37"
C_D = "#c0392b"
C_FRIC = "#e07b39"
C_PRES = "0.30"


def campo_velocidad(X, Y, q, gamma, xp, yp):
    """Velocidad inducida por los paneles + corriente libre (1, 0) en (xp, yp).

    Mismas influencias que en el ensamblaje de Hess-Smith, pero evaluadas en
    puntos cualesquiera: en ejes del panel j, la fuente induce (log, beta)/2pi
    y el torbellino (-beta, log)/2pi, con log = ln(r1/r2) y beta el angulo
    subtendido por el panel visto desde el punto.
    """
    dx, dy = X[1:] - X[:-1], Y[1:] - Y[:-1]
    th = np.arctan2(dy, dx)
    ct, st = np.cos(th), np.sin(th)

    u = np.ones_like(xp)
    v = np.zeros_like(xp)
    # por bloques de paneles: la matriz completa (puntos x paneles) no cabe
    for j0 in range(0, len(th), 32):
        sl = slice(j0, j0 + 32)
        e1x = xp[..., None] - X[:-1][sl]
        e1y = yp[..., None] - Y[:-1][sl]
        e2x = xp[..., None] - X[1:][sl]
        e2y = yp[..., None] - Y[1:][sl]
        r1 = np.hypot(e1x, e1y)
        r2 = np.hypot(e2x, e2y)
        log = np.log(np.maximum(r1, 1e-12) / np.maximum(r2, 1e-12))
        beta = np.arctan2(e1x * e2y - e1y * e2x, e1x * e2x + e1y * e2y)

        uxi = (q[sl] * log - gamma * beta) / (2.0 * np.pi)
        ueta = (q[sl] * beta + gamma * log) / (2.0 * np.pi)
        u += np.sum(uxi * ct[sl] - ueta * st[sl], axis=-1)
        v += np.sum(uxi * st[sl] + ueta * ct[sl], axis=-1)
    return u, v


def main():
    os.makedirs(OUT, exist_ok=True)
    alpha = 8.0

    X0, Y0 = naca4("2412", n=161)
    a = np.radians(alpha)
    X = X0 * np.cos(a) + Y0 * np.sin(a)          # giro -alpha: corriente horizontal
    Y = -X0 * np.sin(a) + Y0 * np.cos(a)

    xc, yc, cp_s, vt, cl, cl_pres, q, gamma = hess_smith(X, Y, 0.0)
    print(f"  alpha = {alpha:.1f} deg   Cl = {cl:.4f}   (integral de p: {cl_pres:.4f})")
    print(f"  Cp maximo en la pared = {cp_s.max():.4f} (remanso, exacto 1)"
          f"   minimo = {cp_s.min():.4f}")
    assert abs(cl - cl_pres) < 0.01
    assert abs(cp_s.max() - 1.0) < 0.02

    # --- campo --------------------------------------------------------------
    xlim, ylim = (-1.15, 1.95), (-0.78, 0.78)
    gx = np.linspace(*xlim, 900)
    gy = np.linspace(*ylim, 460)
    GX, GY = np.meshgrid(gx, gy)
    U, V = campo_velocidad(X, Y, q, gamma, GX, GY)
    CP = 1.0 - (U ** 2 + V ** 2)

    dentro = matplotlib.path.Path(np.column_stack([X, Y])).contains_points(
        np.column_stack([GX.ravel(), GY.ravel()]), radius=0.004
    ).reshape(GX.shape)
    CP = np.ma.masked_array(CP, dentro)

    # el pico de succion se sale de la escala (Cp ~ -4 en la pared): se satura
    # en -1 para que el resto del campo no salga plano
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    niveles = np.linspace(-1.0, 1.0, 41)

    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    fig.subplots_adjust(left=0.005, right=0.885, top=0.995, bottom=0.005)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")

    cf = ax.contourf(GX, GY, CP, levels=niveles, cmap="RdBu_r", norm=norm,
                     extend="min", zorder=1, antialiased=False)
    ax.contour(GX, GY, CP, levels=[-0.7, -0.4, -0.15, 0.25, 0.6], colors="white",
               linewidths=0.5, alpha=0.4, zorder=2)

    # unas pocas lineas de corriente: el campo dentro del perfil es basura, se
    # enmascara antes de integrar
    Us = np.where(dentro, np.nan, U)
    Vs = np.where(dentro, np.nan, V)
    # arrancan a la derecha de las flechas de U_inf y solo hacia aguas abajo,
    # para no repetir la corriente libre ni tapar el pico de succion
    # ninguna semilla en y = 0: esa es la linea de remanso y su flecha se queda
    # clavada en la nariz apuntando aguas arriba
    semillas = np.column_stack([np.full(5, -0.70),
                                np.array([-0.50, -0.24, 0.16, 0.34, 0.58])])
    ax.streamplot(gx, gy, Us, Vs, start_points=semillas, color=(0.25, 0.25, 0.25, 0.65),
                  linewidth=0.7, arrowsize=0.8, density=8, zorder=3,
                  integration_direction="forward")

    ax.add_patch(Polygon(np.column_stack([X, Y]), closed=True, facecolor="white",
                         edgecolor="black", lw=1.6, zorder=5))

    # --- presion y friccion en la pared -------------------------------------
    # La presion es la del propio metodo de paneles (-Cp n, hacia dentro donde
    # comprime y hacia fuera donde succiona). La friccion no: el flujo potencial
    # no tiene capa limite, asi que sus vectores son cualitativos —modulo fijo,
    # direccion la del flujo en la pared— y solo estan para nombrar la segunda
    # mitad de la resistencia.
    thc = np.arctan2(Y[1:] - Y[:-1], X[1:] - X[:-1])
    nx_p, ny_p = -np.sin(thc), np.cos(thc)                 # normal exterior
    sg = np.sign(vt)                                       # sentido del flujo
    tx_p, ty_p = np.cos(thc) * sg, np.sin(thc) * sg
    paso = max(len(xc) // 20, 1)
    sel = np.arange(len(xc))[::paso]
    for i in sel:
        lon = 0.040 + 0.070 * min(abs(cp_s[i]), 1.5) / 1.5
        px, py = xc[i], yc[i]
        if cp_s[i] > 0:                                    # comprime: entra
            ax.annotate("", xy=(px, py), xytext=(px + lon * nx_p[i], py + lon * ny_p[i]),
                        zorder=6, arrowprops=dict(arrowstyle="-|>", color=C_PRES,
                                                  lw=0.9, mutation_scale=7))
        else:                                              # succiona: sale
            ax.annotate("", xy=(px + lon * nx_p[i], py + lon * ny_p[i]), xytext=(px, py),
                        zorder=6, arrowprops=dict(arrowstyle="-|>", color=C_PRES,
                                                  lw=0.9, mutation_scale=7))
        ax.annotate("", xy=(px + 0.055 * tx_p[i], py + 0.055 * ty_p[i]),
                    xytext=(px, py), zorder=7,
                    arrowprops=dict(arrowstyle="-|>", color=C_FRIC, lw=1.1,
                                    mutation_scale=7))

    caja = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8)
    ax.text(-1.10, 0.72, "presión", color=C_PRES, fontsize=10, zorder=8, bbox=caja)
    ax.text(-1.10, 0.63, "fricción", color=C_FRIC, fontsize=10, zorder=8, bbox=caja)

    # --- corriente libre ----------------------------------------------------
    for k, yv in enumerate(np.linspace(-0.55, 0.55, 5)):
        principal = k == 2
        ax.annotate("", xy=(-0.74, yv), xytext=(-1.10, yv), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=C_FLUJO,
                                    lw=2.0 if principal else 1.2,
                                    alpha=1.0 if principal else 0.55))
    ax.text(-1.10, 0.07, r"$U_\infty$", color=C_FLUJO, fontsize=15, zorder=8,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))

    # --- cuerda, alpha y fuerzas -------------------------------------------
    te = np.array([X[0], Y[0]])
    le = np.array([X[np.argmin(X)], Y[np.argmin(X)]])
    fin = te + 0.40 * (te - le)                   # cuerda prolongada tras el TE
    ax.plot([le[0], fin[0]], [le[1], fin[1]], ls=(0, (5, 4)), color="0.25", lw=1.0,
            zorder=6)
    ax.plot([te[0], te[0] + 0.50], [te[1], te[1]], ls=":", color="0.45", lw=1.0,
            zorder=6)
    ax.add_patch(Arc(te, 0.84, 0.84, angle=0.0, theta1=-alpha, theta2=0.0,
                     color="0.25", lw=1.0, zorder=6))
    ax.text(te[0] + 0.46, te[1] - 0.085, r"$\alpha$", fontsize=13, color="0.15",
            zorder=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.8))

    cuarto = le + 0.25 * (te - le)
    ax.annotate("", xy=(cuarto[0], cuarto[1] + 0.60), xytext=tuple(cuarto), zorder=7,
                arrowprops=dict(arrowstyle="-|>", color=C_L, lw=2.4))
    ax.text(cuarto[0] - 0.055, cuarto[1] + 0.66, r"$L$", color=C_L, fontsize=15,
            zorder=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.85))
    ax.annotate("", xy=(cuarto[0] + 0.46, cuarto[1]), xytext=tuple(cuarto), zorder=7,
                arrowprops=dict(arrowstyle="-|>", color=C_D, lw=2.4))
    ax.text(cuarto[0] + 0.52, cuarto[1] + 0.075, r"$D$", color=C_D, fontsize=15,
            zorder=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.85))
    ax.plot([cuarto[0]], [cuarto[1]], "o", ms=5, color="black", zorder=8)

    # --- barra de color -----------------------------------------------------
    cax = fig.add_axes([0.845, 0.13, 0.015, 0.72])
    cb = fig.colorbar(cf, cax=cax, ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=9)
    cb.ax.set_title(r"$C_p$", fontsize=13, pad=8)
    # las dos palabras van a la derecha de los numeros, no encima de la barra:
    # centradas se salian de la figura
    cb.ax.text(4.0, 0.98, "sobrepresión", transform=cb.ax.transAxes, ha="left",
               va="top", fontsize=9, color="#b2182b", rotation=90)
    cb.ax.text(4.0, 0.02, "succión", transform=cb.ax.transAxes, ha="left",
               va="bottom", fontsize=9, color="#2166ac", rotation=90)

    ruta = os.path.join(OUT, "fig_1D_campo_presion.png")
    fig.savefig(ruta, dpi=DPI)
    plt.close(fig)
    print(f"  -> {ruta}")


if __name__ == "__main__":
    main()

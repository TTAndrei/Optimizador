"""
Figura 2.A — Descomposicion de Helmholtz-Hodge.

Tres paneles sobre el mismo rectangulo: un campo sintetico w, su parte solenoidal
tangente a la frontera y el gradiente del potencial, con w = u + grad(phi).

El campo de partida NO se construye sumando las dos piezas: se define entero y se
descompone resolviendo el mismo problema de Poisson con Neumann que resuelve el
solver (ec. 2.7). Asi los paneles 2 y 3 son el resultado de una proyeccion de
verdad y no los ingredientes de una receta.

El fondo de color es la divergencia de cada panel, con la misma escala en los
tres: toda la divergencia de w esta en el gradiente y el panel central sale
blanco, que es lo que hace la etapa de proyeccion.

Uso:
    .venv/bin/python scripts/agent_tests/fig_mem_helmholtz.py
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "figuras_memoria")
DPI = 200

LX, LY = 2.0, 1.2
NX, NY = 96, 58


def campo(X, Y):
    """Campo arbitrario: dos torbellinos, una fuente y un sumidero, y una cizalla
    de fondo. No es solenoidal ni es un gradiente."""
    def g(x0, y0, s):
        return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / s)

    # parte con rotacional
    psi = 0.9 * g(0.55, 0.62, 0.045) - 0.7 * g(1.35, 0.50, 0.055)
    # parte con divergencia: fuente y sumidero
    q = 1.0 * g(0.95, 0.85, 0.030) - 1.0 * g(1.15, 0.35, 0.030)

    dy = Y[1, 0] - Y[0, 0]
    dx = X[0, 1] - X[0, 0]
    u = np.gradient(psi, dy, axis=0) + np.gradient(q, dx, axis=1)
    v = -np.gradient(psi, dx, axis=1) + np.gradient(q, dy, axis=0)
    # cizalla de fondo (ni una cosa ni la otra sobre un dominio acotado)
    u += 0.35 * (Y - LY / 2.0)
    v += 0.10 * (X - LX / 2.0)
    return u, v


def divergencia(u, v, dx, dy):
    return np.gradient(u, dx, axis=1) + np.gradient(v, dy, axis=0)


def poisson_neumann(rhs, dx, dy, bn):
    """Resuelve lap(phi) = rhs con dphi/dn dado en el contorno (bn), volumenes
    finitos sobre celdas. La media de phi se fija a cero para quitar la
    indeterminacion de la constante.

    bn: dict lado -> array del flujo saliente por unidad de longitud.
    """
    ny, nx = rhs.shape
    n = nx * ny
    idx = lambda j, i: j * nx + i
    A = lil_matrix((n, n))
    b = np.zeros(n)
    ix2, iy2 = 1.0 / dx ** 2, 1.0 / dy ** 2

    for j in range(ny):
        for i in range(nx):
            k = idx(j, i)
            diag = 0.0
            if i > 0:
                A[k, idx(j, i - 1)] = ix2
                diag -= ix2
            else:
                b[k] += -bn["left"][j] / dx
            if i < nx - 1:
                A[k, idx(j, i + 1)] = ix2
                diag -= ix2
            else:
                b[k] += -bn["right"][j] / dx
            if j > 0:
                A[k, idx(j - 1, i)] = iy2
                diag -= iy2
            else:
                b[k] += -bn["bottom"][i] / dy
            if j < ny - 1:
                A[k, idx(j + 1, i)] = iy2
                diag -= iy2
            else:
                b[k] += -bn["top"][i] / dy
            A[k, k] = diag
            b[k] += rhs[j, i]

    # Compatibilidad: el problema de Neumann puro es singular, se ancla una celda
    # y despues se le resta la media a la solucion.
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = 0.0
    phi = spsolve(csr_matrix(A), b).reshape(ny, nx)
    return phi - phi.mean()


def panel(ax, X, Y, u, v, div, norm, titulo):
    im = ax.pcolormesh(X, Y, div, cmap="RdBu_r", norm=norm, shading="gouraud")
    m = np.hypot(u, v)
    ax.streamplot(X[0], Y[:, 0], u, v, color="0.15", density=0.85,
                  linewidth=0.7 + 1.5 * m / max(m.max(), 1e-12),
                  arrowsize=0.7)
    ax.set_xlim(0, LX)
    ax.set_ylim(0, LY)
    ax.set_aspect("equal")
    ax.set_title(titulo, fontsize=11, pad=7)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.0)
    return im


def main():
    os.makedirs(OUT, exist_ok=True)
    x = (np.arange(NX) + 0.5) * LX / NX
    y = (np.arange(NY) + 0.5) * LY / NY
    dx, dy = LX / NX, LY / NY
    X, Y = np.meshgrid(x, y)

    u, v = campo(X, Y)

    # Neumann: dphi/dn = w.n, que es lo que hace tangente a la frontera la parte
    # solenoidal (ec. 2.5 y 2.7).
    bn = {"left": -u[:, 0], "right": u[:, -1],
          "bottom": -v[0, :], "top": v[-1, :]}
    rhs = divergencia(u, v, dx, dy)
    phi = poisson_neumann(rhs, dx, dy, bn)

    gx = np.gradient(phi, dx, axis=1)
    gy = np.gradient(phi, dy, axis=0)
    us, vs = u - gx, v - gy

    d_w = divergencia(u, v, dx, dy)
    d_s = divergencia(us, vs, dx, dy)
    d_g = divergencia(gx, gy, dx, dy)

    lim = float(np.percentile(np.abs(d_w), 99.5))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    res = np.hypot(u - (us + gx), v - (vs + gy)).max() / np.hypot(u, v).max()
    orto = float(np.sum(us * gx + vs * gy) * dx * dy)
    esc = float(np.sqrt(np.sum(us ** 2 + vs ** 2) * dx * dy)
                * np.sqrt(np.sum(gx ** 2 + gy ** 2) * dx * dy))
    print(f"residuo de la suma: {res:.2e}   ortogonalidad <u,grad phi>/norma: "
          f"{orto / max(esc, 1e-30):.2e}")
    print(f"div media |w|={np.abs(d_w).mean():.3e}  solenoidal={np.abs(d_s).mean():.3e}")

    fig, axs = plt.subplots(1, 3, figsize=(13.2, 3.5))
    fig.subplots_adjust(left=0.02, right=0.90, top=0.86, bottom=0.06, wspace=0.30)

    panel(axs[0], X, Y, u, v, d_w, norm, r"campo de partida  $\mathbf{w}$")
    panel(axs[1], X, Y, us, vs, d_s, norm,
          r"parte solenoidal  $\mathbf{u}$,   $\nabla\!\cdot\!\mathbf{u}=0$")
    im = panel(axs[2], X, Y, gx, gy, d_g, norm, r"gradiente  $\nabla\phi$")

    for ax, s in zip(axs[:2], ["=", "+"]):
        fig.text(ax.get_position().x1 + 0.028, 0.46, s, fontsize=22,
                 ha="center", va="center")

    cax = fig.add_axes([0.915, 0.10, 0.013, 0.72])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\nabla\!\cdot\!(\,\cdot\,)$", fontsize=10)

    fig.suptitle(r"$\mathbf{w}=\mathbf{u}+\nabla\phi$   con   "
                 r"$\nabla\!\cdot\!\mathbf{u}=0$   y   "
                 r"$\mathbf{u}\cdot\mathbf{n}|_{\partial\Omega}=0$",
                 fontsize=12.5, y=0.975)

    p = os.path.join(OUT, "fig_2A_helmholtz_hodge.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("->", p)


if __name__ == "__main__":
    main()

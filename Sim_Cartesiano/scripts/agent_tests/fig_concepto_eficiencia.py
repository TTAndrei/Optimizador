"""
Figura conceptual: flujo adherido sobre un cuerpo fuselado (arriba) frente a
flujo desprendido sobre un cuerpo romo (abajo). NO es una simulacion, es flujo
potencial analitico: sirve para la portada o la introduccion, no para citar
ningun numero.

    Fuselado   ovalo de Rankine (fuente + sumidero en corriente uniforme). Las
               lineas se cierran detras del cuerpo: sin estela.
    Romo       cilindro (dipolo) mas un par de torbellinos contrarrotantes
               detras. El par es lo que dibuja la burbuja de recirculacion; un
               cilindro en flujo potencial puro sale simetrico y no se
               distingue del caso eficiente.

Cuatro variantes de estilo, todas con las mismas dos geometrias y el mismo
campo, para elegir.

Uso:
    .venv/bin/python scripts/agent_tests/fig_concepto_eficiencia.py --out DIR
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, FancyArrowPatch

U = 1.0
NX, NY = 900, 420
XLIM, YLIM = (-3.0, 5.6), (-1.9, 1.9)
# El campo se calcula mas alto de lo que se enseña: en el borde de arriba
# streamplot amontona lineas y deja una mancha negra que no significa nada.
VISTA_Y = (-1.5, 1.5)

# Semillas dentro de la burbuja de recirculacion. Sin ellas ninguna linea cae
# ahi —a densidad baja el sembrado es regular y se salta los torbellinos— y el
# caso ineficiente sale como un simple abultamiento de las lineas.
# Semillas dentro de la burbuja de recirculacion. Sin ellas ninguna linea cae
# ahi —a densidad baja el sembrado es regular y se salta los torbellinos— y el
# caso ineficiente sale como un simple abultamiento de las lineas. Van en anillo
# ALREDEDOR del nucleo, no en el: en el centro la velocidad es cero y la linea
# muere en el primer paso.
def _anillo(xv, yv, radios=(0.42,)):
    ang = np.linspace(0, 2 * np.pi, 5)[:-1]
    pts = [[xv + r * np.cos(t), yv + r * np.sin(t)] for r in radios for t in ang]
    return np.array(pts)


SEMILLAS_ESTELA = np.vstack([_anillo(1.18, 0.42), _anillo(1.18, -0.42)])


def _grid():
    x = np.linspace(*XLIM, NX)
    y = np.linspace(*YLIM, NY)
    return np.meshgrid(x, y)


def campo_fuselado(a=1.5, m=0.42):
    """Ovalo de Rankine: fuente en -a, sumidero en +a."""
    X, Y = _grid()
    Z = X + 1j * Y
    W = U + m / (2 * np.pi * (Z + a)) - m / (2 * np.pi * (Z - a))
    psi = (U * Y
           + m / (2 * np.pi) * (np.arctan2(Y, X + a) - np.arctan2(Y, X - a)))
    return X, Y, W.real, -W.imag, psi


def campo_romo(R=0.62, gamma=5.0, xv=1.18, yv=0.42, rc=0.26):
    """Cilindro mas par contrarrotante: la burbuja de recirculacion.

    Los torbellinos llevan nucleo de radio rc. Con el vortice puntual la
    velocidad diverge en el centro y streamplot dibuja una espiral apretada que
    se come la figura; con nucleo la linea cierra en un ovalo limpio.
    """
    X, Y = _grid()
    Z = X + 1j * Y
    W = U * (1 - R ** 2 / Z ** 2)
    psi = U * Y * (1 - R ** 2 / (X ** 2 + Y ** 2))
    # Signo: en una estela el torbellino de ARRIBA gira en sentido horario
    # (circulacion negativa) y el de abajo al reves. Con los signos cambiados el
    # par empuja el fluido aguas abajo entre los dos y no hay recirculacion.
    for signo, y0 in ((-1.0, yv), (+1.0, -yv)):
        dz = Z - (xv + 1j * y0)
        r2 = np.abs(dz) ** 2
        W = W - 1j * signo * gamma * np.conj(dz) / (2 * np.pi * (r2 + rc ** 2))
        psi = psi - signo * gamma / (4 * np.pi) * np.log(r2 + rc ** 2)
    return X, Y, W.real, -W.imag, psi


def contorno_cuerpo(X, Y, psi):
    """Poligono del ovalo de Rankine a partir de la linea psi=0.

    psi=0 es tambien el eje y=0 aguas arriba y aguas abajo, asi que el contorno
    sale partido: se queda la mitad de arriba y se refleja. Coserlo desde el
    trozo suelto de la mitad de abajo daria un poligono con el orden de los
    puntos roto.
    """
    fig = plt.figure()
    cs = plt.contour(X, Y, psi, levels=[0.0])
    segs = [np.asarray(v) for v in cs.allsegs[0]]
    plt.close(fig)
    # El trozo del cuerpo es el de menor extension en x: el otro es el eje, que
    # cruza el dominio entero. Puede venir la mitad de arriba o la de abajo,
    # segun por donde corte el contorno, asi que se toma |y| y se refleja.
    acotados = [v for v in segs
                if len(v) > 20 and np.abs(v[:, 1]).max() > 1e-3]
    if not acotados:
        return None
    v = min(acotados, key=lambda a: np.ptp(a[:, 0]))
    # El lazo acotado recorre la mitad del cuerpo Y vuelve por el eje: entre el
    # morro y la cola, y=0 tambien cumple psi=0. Quitar ese tramo deja una sola
    # rama, y en su orden natural, que es lo que hace falta para el poligono
    # (ordenar por x mezcla las dos ramas y sale un serrucho).
    v = v[np.abs(v[:, 1]) > 1e-4]
    # El lazo empieza a media rama, asi que cerrarlo tal cual mete una cuerda
    # atravesando el cuerpo. Se rota para arrancar en el morro.
    v = np.roll(v, -int(np.argmin(v[:, 0])), axis=0)
    v = np.column_stack([v[:, 0], np.abs(v[:, 1])])
    return np.vstack([v, v[::-1] * [1.0, -1.0]])


def geometrias():
    Xf, Yf, uf, vf, pf = campo_fuselado()
    Xr, Yr, ur, vr, pr = campo_romo()

    poly_f = contorno_cuerpo(Xf, Yf, pf)
    R = 0.62
    th = np.linspace(0, 2 * np.pi, 400)
    poly_r = np.column_stack([R * np.cos(th), R * np.sin(th)])

    # Dentro del cuerpo no hay flujo: sin esto streamplot dibuja el campo
    # interior del dipolo, que es una espiral bonita y completamente falsa.
    def enmascara(X, Y, u, v, poly):
        dentro = MplPath(poly).contains_points(
            np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
        return np.where(dentro, np.nan, u), np.where(dentro, np.nan, v), dentro

    uf, vf, _ = enmascara(Xf, Yf, uf, vf, poly_f)
    ur, vr, _ = enmascara(Xr, Yr, ur, vr, poly_r)
    return (Xf, Yf, uf, vf, poly_f), (Xr, Yr, ur, vr, poly_r)


# ----------------------------------------------------------------------
# Variantes
# ----------------------------------------------------------------------
def _ejes(fig):
    ax = fig.subplots(2, 1)
    for a in ax:
        a.set_xlim(*XLIM); a.set_ylim(*VISTA_Y)
        a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    return ax


def _lineas(a, X, Y, u, v, color, lw, dens, arrowsize, semillas=None):
    a.streamplot(X[0], Y[:, 0], u, v, color=color, linewidth=lw,
                 density=dens, arrowsize=arrowsize, broken_streamlines=False)
    if semillas is not None:
        # broken_streamlines=True aqui a proposito: los torbellinos son lazos
        # cerrados y con la linea forzada a cruzar el dominio no se dibujan.
        # density baja y maxlength corto: con el sembrado denso las lineas se
        # apretujan dentro del nucleo y lo que sale es un borron negro, no un
        # torbellino.
        a.streamplot(X[0], Y[:, 0], u, v, color=color, linewidth=lw,
                     density=1.2, arrowsize=arrowsize, maxlength=3.0,
                     start_points=semillas, broken_streamlines=True)


def _cuerpo(ax, poly, cara, borde, lw=1.6):
    ax.add_patch(PathPatch(MplPath(poly), facecolor=cara, edgecolor=borde,
                           lw=lw, zorder=5))


def v1_lineas(datos, path):
    """Linea negra sobre blanco. Lo minimo que cuenta la historia."""
    (Xf, Yf, uf, vf, pf), (Xr, Yr, ur, vr, pr) = datos
    fig = plt.figure(figsize=(9, 7.2), layout="constrained")
    ax = _ejes(fig)
    for a, (X, Y, u, v, poly) in zip(ax, datos):
        _lineas(a, X, Y, u, v, "k", 1.0, (0.55, 0.42), 1.0,
                SEMILLAS_ESTELA if a is ax[1] else None)
        _cuerpo(a, poly, "white", "k", lw=2.0)
    ax[0].set_title("Eficiente — el flujo sigue la forma", fontsize=13, loc="left")
    ax[1].set_title("Ineficiente — el flujo se despega", fontsize=13, loc="left")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def _deficit(X, Y, x0, ancho0, ancho1, hondura):
    """Perdida de velocidad en la estela, como factor multiplicativo.

    El |u| del modelo potencial NO sirve de fondo tal cual: los nucleos de los
    torbellinos salen como puntos rapidos y la estela entera se pinta caliente,
    que es justo lo contrario de lo que hay que ensenar. Una estela es fluido
    LENTO. Esto es un dibujo, no una medida.
    """
    t = np.clip((X - x0) / (XLIM[1] - x0), 0, 1)
    ancho = ancho0 + (ancho1 - ancho0) * np.sqrt(t)
    # Arranque suave: con un escalon (X > x0) se ve una linea vertical dura
    # justo detras del cuerpo, que no significa nada.
    arranque = 1.0 / (1.0 + np.exp(-(X - x0) / 0.12))
    return 1.0 - hondura * np.exp(-(Y / ancho) ** 2) * arranque


def v2_campo(datos, path):
    """Velocidad de fondo mas lineas negras encima."""
    fig = plt.figure(figsize=(9, 7.2), layout="constrained")
    ax = _ejes(fig)
    deficits = ((1.57, 0.10, 0.24, 0.30), (0.62, 0.62, 1.25, 0.85))
    for a, (X, Y, u, v, poly), d in zip(ax, datos, deficits):
        spd = np.hypot(u, v) * _deficit(X, Y, *d)
        a.pcolormesh(X, Y, spd, cmap="Spectral_r", vmin=0.0, vmax=2.0,
                     shading="auto", zorder=0, rasterized=True)
        _lineas(a, X, Y, u, v, "k", 0.9, (0.55, 0.40), 0.9,
                SEMILLAS_ESTELA if a is ax[1] else None)
        _cuerpo(a, poly, "#f7f7f7", "k", lw=2.0)
    ax[0].set_title("Eficiente — el flujo sigue la forma", fontsize=13, loc="left")
    ax[1].set_title("Ineficiente — el flujo se despega", fontsize=13, loc="left")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def _banda_estela(ax, x0, y0, y1, color, alpha):
    """Estela como banda que se abre aguas abajo. Es un dibujo, no una medida:
    marcar la region por |u| < 0.5U da manchas sueltas alrededor de cada
    torbellino y no se lee como una estela."""
    xs = np.linspace(x0, XLIM[1], 60)
    t = (xs - x0) / (XLIM[1] - x0)
    ancho = y0 + (y1 - y0) * np.sqrt(np.clip(t, 0, 1))
    ax.fill_between(xs, -ancho, ancho, color=color, alpha=alpha, zorder=1,
                    linewidth=0)


def v3_anotada(datos, path):
    """Estela sombreada y anotaciones: la version para explicar en una charla."""
    fig = plt.figure(figsize=(9.5, 7.6), layout="constrained")
    ax = _ejes(fig)
    for a, (X, Y, u, v, poly) in zip(ax, datos):
        _lineas(a, X, Y, u, v, "k", 1.0, (0.55, 0.42), 1.0,
                SEMILLAS_ESTELA if a is ax[1] else None)
        _cuerpo(a, poly, "white", "k", lw=2.0)

    _banda_estela(ax[0], 1.57, 0.10, 0.22, "#2e7d5b", 0.16)
    _banda_estela(ax[1], 0.62, 0.62, 1.25, "#c0392b", 0.16)

    caja = dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.92)
    flecha = dict(arrowstyle="->", lw=1.2, shrinkB=2)

    ax[0].annotate("el flujo llega pegado a la cola", xy=(1.45, 0.16),
                   xytext=(2.5, 0.92), fontsize=11, bbox=caja,
                   arrowprops=flecha)
    ax[0].text(3.9, -1.18, "estela estrecha  →  poca resistencia",
               fontsize=12, color="#2e7d5b", ha="center", bbox=caja)
    ax[1].annotate("el flujo se despega", xy=(0.52, 0.42), xytext=(-1.1, 1.15),
                   fontsize=11, bbox=caja, arrowprops=flecha)
    ax[1].annotate("recirculación", xy=(1.22, -0.52), xytext=(2.6, -1.02),
                   fontsize=11, bbox=caja, arrowprops=flecha)
    ax[1].text(3.9, 1.22, "estela ancha  →  mucha resistencia",
               fontsize=12, color="#c0392b", ha="center", bbox=caja)
    ax[0].set_title("Eficiente", fontsize=14, loc="left", fontweight="bold")
    ax[1].set_title("Ineficiente", fontsize=14, loc="left", fontweight="bold")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def v4_duotono(datos, path):
    """Fondo oscuro, dos colores. Para portada o diapositiva."""
    fondo, tinta = "#11161c", "#eceff4"
    fig = plt.figure(figsize=(9, 7.2), layout="constrained", facecolor=fondo)
    ax = _ejes(fig)
    colores = ("#5bc8af", "#e8705a")
    for a, (X, Y, u, v, poly), c in zip(ax, datos, colores):
        a.set_facecolor(fondo)
        _lineas(a, X, Y, u, v, tinta, 1.1, (0.55, 0.42), 0.0,
                SEMILLAS_ESTELA if a is ax[1] else None)
        _cuerpo(a, poly, c, tinta, lw=1.4)
    ax[0].set_title("EFICIENTE", fontsize=15, loc="left", color=colores[0],
                    fontweight="bold")
    ax[1].set_title("INEFICIENTE", fontsize=15, loc="left", color=colores[1],
                    fontweight="bold")
    for a in ax:
        a.add_patch(FancyArrowPatch((-2.85, 1.28), (-1.95, 1.28),
                                    arrowstyle="-|>", mutation_scale=13,
                                    color=tinta, lw=1.2))
        a.text(-1.85, 1.28, "flujo", color=tinta, fontsize=10, va="center")
    fig.savefig(path, dpi=200, facecolor=fondo)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figuras_memoria")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    datos = geometrias()
    for nombre, fn in (("v1_lineas", v1_lineas), ("v2_campo", v2_campo),
                       ("v3_anotada", v3_anotada), ("v4_duotono", v4_duotono)):
        p = os.path.join(a.out, f"concepto_eficiencia_{nombre}.png")
        fn(datos, p)
        print(p)


if __name__ == "__main__":
    main()

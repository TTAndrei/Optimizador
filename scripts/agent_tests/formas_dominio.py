"""Contornos de dominio con verdad conocida, para verificar la geometria arbitraria.

A diferencia de `formas_geom`, que da CUERPOS normalizados a cuerda 1 para
sumergir en una caja, esto da CONTORNOS EXTERIORES en coordenadas fisicas: el
poligono encierra el fluido y todo lo de fuera es solido. Son los casos con
solucion analitica o con una ley de conservacion que se puede comprobar, que es
la unica forma de saber si un dominio importado esta bien resuelto.

Los tres cruzan el perimetro de la caja por izquierda y derecha a proposito: ahi
es donde el solver puede poner entrada y salida, y ahi es donde estaba el fallo
(las paredes que mueren en el borde del dominio).

Convenio de salida: poligono CERRADO, sentido antihorario, en metros.
"""
from __future__ import annotations

import os
import numpy as np


def canal(Lx=6.0, Ly=2.0, h=1.0, n=400):
    """Canal recto de altura h centrado en Ly/2. Poiseuille plano.

        u(y) = 1.5 * u_media * (1 - (2*(y-yc)/h)^2)
        dp/dx = -12 * mu * u_media / h^2

    El rectangulo se extiende un poco mas alla de la caja en x para que las
    paredes corten el borde limpiamente y no dejen una celda ambigua en la
    esquina.
    """
    yc = 0.5 * Ly
    y0, y1 = yc - 0.5 * h, yc + 0.5 * h
    x0, x1 = -0.1 * Lx, 1.1 * Lx
    return _rect(x0, y0, x1, y1, n)


def tobera(Lx=6.0, Ly=2.0, h_in=1.2, h_out=0.6, x_ini=1.5, x_fin=4.5, n=400):
    """Tobera convergente. Contraccion h_in/h_out (2:1 por defecto).

    Verdad: conservacion de masa. En incompresible, u_out/u_in = h_in/h_out
    exactamente, y el balance ha de cerrar por debajo del 1 por mil. Es la
    comprobacion primaria y no necesita solucion analitica de campo.
    """
    yc = 0.5 * Ly
    xs = np.concatenate([
        np.linspace(-0.1 * Lx, x_ini, n // 4, endpoint=False),
        np.linspace(x_ini, x_fin, n // 2, endpoint=False),
        np.linspace(x_fin, 1.1 * Lx, n // 4 + 1),
    ])
    # Semialtura: constante, coseno suavizado en la contraccion, constante.
    t = np.clip((xs - x_ini) / (x_fin - x_ini), 0.0, 1.0)
    hs = 0.5 * (h_in + (h_out - h_in) * 0.5 * (1.0 - np.cos(np.pi * t)))
    x = np.concatenate([xs, xs[::-1]])
    y = np.concatenate([yc + hs, (yc - hs)[::-1]])
    return _cierra(x, y)


def escalon(Lx=8.0, Ly=2.0, h=0.5, x_step=2.0, n=400):
    """Escalon hacia atras, razon de expansion 2 (entrada h, salida 2h).

    Verdad: longitud de reataque x_r/h contra Armaly et al. (1983) — ~3 a
    Re_h=100 y ~8 a Re_h=400. Es el caso que estresa la reparacion de puntos
    imagen en una esquina reentrante.
    """
    x0, x1 = -0.1 * Lx, 1.1 * Lx
    y_top = Ly
    # Antihorario. La entrada es un canal de altura h pegado al techo, desde x0
    # hasta x_step; despues el escalon abre a la altura completa.
    pts = [
        (x0, y_top - h), (x_step, y_top - h),   # suelo del canal de entrada
        (x_step, 0.0),                           # cara vertical del escalon
        (x1, 0.0),                               # suelo aguas abajo
        (x1, y_top),                             # borde derecho (salida)
        (x0, y_top),                             # techo
    ]
    x, y = _densifica(np.array([p[0] for p in pts]),
                      np.array([p[1] for p in pts]), n)
    return _cierra(x, y)


# ----------------------------------------------------------------------
def _rect(x0, y0, x1, y1, n):
    x, y = _densifica(np.array([x0, x1, x1, x0]),
                      np.array([y0, y0, y1, y1]), n)
    return _cierra(x, y)


def _densifica(px, py, n):
    """Reparte n puntos por el poligono proporcionalmente a la longitud de cada
    lado. Un lado con dos puntos y otro con doscientos rasteriza igual, pero la
    SDF exacta por segmentos no: ahi el coste va con el numero de segmentos y
    conviene que sea uniforme."""
    px, py = np.append(px, px[0]), np.append(py, py[0])
    d = np.hypot(np.diff(px), np.diff(py))
    total = d.sum()
    xs, ys = [], []
    for i, di in enumerate(d):
        k = max(2, int(round(n * di / total)))
        t = np.linspace(0.0, 1.0, k, endpoint=False)
        xs.append(px[i] + t * (px[i + 1] - px[i]))
        ys.append(py[i] + t * (py[i + 1] - py[i]))
    return np.concatenate(xs), np.concatenate(ys)


def _cierra(x, y):
    if np.hypot(x[0] - x[-1], y[0] - y[-1]) > 1e-12:
        x, y = np.append(x, x[0]), np.append(y, y[0])
    if area_con_signo(x, y) < 0:          # antihorario
        x, y = x[::-1], y[::-1]
    return x, y


def area_con_signo(x, y):
    return 0.5 * (np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))


DOMINIOS = {"canal": canal, "tobera": tobera, "escalon": escalon}


def escribe(nombre, path, **kw):
    x, y = DOMINIOS[nombre](**kw)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{nombre} contorno exterior (metros)\n")
        for xi, yi in zip(x, y):
            f.write(f"  {xi:.8f}  {yi:.8f}\n")
    return path


if __name__ == "__main__":
    for nombre, fn in DOMINIOS.items():
        x, y = fn()
        cerrado = np.hypot(x[0] - x[-1], y[0] - y[-1]) < 1e-9
        print(f"{nombre:8s} n={len(x):5d}  x[{x.min():+.3f},{x.max():+.3f}] "
              f"y[{y.min():+.3f},{y.max():+.3f}]  area={area_con_signo(x, y):+.5f}  "
              f"cerrado={cerrado}")

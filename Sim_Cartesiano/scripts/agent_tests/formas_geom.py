"""Geometrias romas canonicas en formato Selig, para el pipeline de perfiles.

Todas normalizadas a x en [0,1] y ordenadas como un perfil: se arranca en x
maximo (el "borde de salida"), se recorre el lado superior hasta x minimo (el
"borde de ataque") y se vuelve por el inferior. `load_solids_from_file` parte la
lista en `argmin(x)` y rasteriza por punto-en-poligono, asi que cualquier
poligono cerrado con ese orden vale.

Longitud caracteristica = 1 en las tres, que es lo que se pasa como `chord`, o
sea Cd y Cl salen referidos a 1 m.
"""
from __future__ import annotations

import os
import numpy as np


def circulo(n=1440):
    """Diametro 1. Cd experimental subcritico ~1.0-1.2, St ~0.2."""
    th = np.linspace(0.0, 2.0 * np.pi, n + 1)
    return 0.5 + 0.5 * np.cos(th), 0.5 * np.sin(th)


def cuadrado(n_lado=400):
    """Lado 1, una cara perpendicular al flujo. Cd experimental ~2.0-2.2."""
    t = np.linspace(0.0, 1.0, n_lado + 1)
    # (1,0) -> (1,0.5) -> (0,0.5) -> (0,-0.5) -> (1,-0.5) -> (1,0)
    tramos = [
        (np.ones_like(t[:-1]),            0.0 + 0.5 * t[:-1]),   # cara trasera, mitad sup
        (1.0 - t[:-1],                    np.full_like(t[:-1], 0.5)),   # cara superior
        (np.zeros_like(t[:-1]),           0.5 - t[:-1]),          # cara frontal
        (t[:-1],                          np.full_like(t[:-1], -0.5)),  # cara inferior
        (np.ones_like(t),                 -0.5 + 0.5 * t),        # cara trasera, mitad inf
    ]
    x = np.concatenate([a for a, _ in tramos])
    y = np.concatenate([b for _, b in tramos])
    return x, y


def gota(n=720, R=0.2):
    """Morro romo (semicirculo de radio R) y cola afilada en (1,0).

    Las dos rectas son las tangentes desde la punta de cola al circulo de morro,
    centrado en (R,0). Con R=0.2: espesor 0.4, esbeltez 2.5. Cd de un cuerpo
    fuselado de esta esbeltez ~0.05-0.1 referido a la longitud.
    """
    xc, d = R, 1.0 - R
    th_t = np.arcsin(R / d)               # semiangulo del cono tangente
    # Punto de tangencia superior, medido desde el centro del circulo
    phi_t = np.pi / 2 + th_t
    xt, yt = xc + R * np.cos(phi_t), R * np.sin(phi_t)

    s = np.linspace(0.0, 1.0, n // 2 + 1)
    # cola -> tangencia (lado superior)
    x1, y1 = 1.0 + s[:-1] * (xt - 1.0), 0.0 + s[:-1] * (yt - 0.0)
    # arco de morro, de la tangencia superior a la inferior pasando por x minimo
    phi = np.linspace(phi_t, 2 * np.pi - phi_t, n // 2 + 1)
    x2, y2 = xc + R * np.cos(phi), R * np.sin(phi)
    # tangencia inferior -> cola
    x3, y3 = xt + s * (1.0 - xt), -yt + s * (0.0 + yt)
    return np.concatenate([x1, x2[:-1], x3]), np.concatenate([y1, y2[:-1], y3])


FORMAS = {"circulo": circulo, "cuadrado": cuadrado, "gota": gota}

def area_poligono(x, y):
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# Area exacta, para verificar la rasterizacion contra solid.sum()*dx*dy
AREA = {n: area_poligono(*f()) for n, f in FORMAS.items()}


def escribe(forma, path):
    x, y = FORMAS[forma]()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{forma} L=1\n")
        for xi, yi in zip(x, y):
            f.write(f"  {xi:.8f}  {yi:.8f}\n")
    return path


if __name__ == "__main__":
    for nombre in FORMAS:
        x, y = FORMAS[nombre]()
        print(f"{nombre:9s} n={len(x):5d}  x[{x.min():.4f},{x.max():.4f}] "
              f"y[{y.min():+.4f},{y.max():+.4f}]  area={area_poligono(x, y):.6f}  "
              f"argmin_x={int(np.argmin(x))}  cerrado={np.hypot(x[0]-x[-1], y[0]-y[-1]) < 1e-9}")

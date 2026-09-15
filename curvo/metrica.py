r"""Campos metricos de una malla estructurada, a partir de sus vertices.

Convenio de indices. La malla son vertices `X[j,i]`, `Y[j,i]` de forma (M, N).
Las celdas son (M-1, N-1) = (ny, nx), con la celda (j,i) delimitada por los
vertices (j,i), (j,i+1), (j+1,i+1), (j+1,i).

    j+1 o-------o
        |       |        xi crece con i  ->
        | (j,i) |        eta crece con j  (hacia el campo lejano)
    j   o-------o
        i      i+1

- **Cara xi** (j,i): separa la celda (j,i-1) de la (j,i). Es el segmento del
  vertice (j,i) al (j+1,i). Hay N por fila => forma (ny, N).
- **Cara eta** (j,i): separa la celda (j-1,i) de la (j,i). Segmento del vertice
  (j,i) al (j,i+1). Forma (M, nx).

Los vectores de area salen del segmento recto entre vertices, `S = (dy, -dx)`.
Eso da conservacion geometrica **exacta**: la suma de las cuatro caras de una
celda es cero por telescopado del poligono cerrado, sin termino de correccion.
Calcularlos diferenciando centros de celda rompe esto (medido: |div(u_inf)|
pasa de 1e-16 a 2.4e-2 con celdas de area 2e-4).

El modulo funciona igual con numpy y con cupy: las metricas se construyen una
vez en CPU float64 junto con la malla, y se suben a GPU float32 para el solver.
Tener las dos rutas es lo que permite validar los operadores sin GPU.
"""

from __future__ import annotations

import numpy as np

try:                                            # cupy es opcional para la ruta CPU
    import cupy as _cp
except ImportError:                             # pragma: no cover
    _cp = None

_TIPOS_ARRAY = (np.ndarray,) if _cp is None else (np.ndarray, _cp.ndarray)

__all__ = ["xp_de", "Metrica"]


def xp_de(a):
    """numpy o cupy segun donde viva el array."""
    return _cp.get_array_module(a) if _cp is not None else np


def _pad_i(a, xp):
    """Replica la primera y ultima columna (indice de celda i)."""
    return xp.concatenate([a[:, :1], a, a[:, -1:]], axis=1)


def _pad_j(a, xp):
    """Replica la primera y ultima fila (indice de celda j)."""
    return xp.concatenate([a[:1, :], a, a[-1:, :]], axis=0)


class Metrica:
    """Geometria discreta de la malla: areas de cara, volumenes y coeficientes.

    Atributos principales (xp = numpy o cupy segun `X`):

    ===================  ==============  ==================================
    nombre               forma           que es
    ===================  ==============  ==================================
    ``Sx_xi, Sy_xi``     (ny, N)         vector de area de las caras xi
    ``Sx_eta, Sy_eta``   (M, nx)         vector de area de las caras eta
    ``J``                (ny, nx)        volumen (area 2D) de celda
    ``xc, yc``           (ny, nx)        centroide de celda
    ``a_xi, b_xi``       (ny, N)         coeficientes de flujo difusivo
    ``a_eta, b_eta``     (M, nx)         idem en eta
    ===================  ==============  ==================================

    Los coeficientes salen de escribir el gradiente en la base contravariante,
    ``grad(f) = f_xi * S_xi / J + f_eta * S_eta / J``, y proyectarlo sobre la
    cara::

        grad(f) . S_xi = a_xi * f_xi + b_xi * f_eta,
        a_xi = (S_xi . S_xi) / V_cara,   b_xi = (S_xi . S_eta) / V_cara

    `b` es el termino cruzado. **No es despreciable**: medido sobre las mallas
    generadas, |b|/sqrt(a_xi*a_eta) p99 vale 0.03-0.06 en perfiles reales pero
    0.33 de mediana sobre el historial del GA. Cualquier esquema que lo ignore
    (ADI, splitting direccional) se sostiene solo en los perfiles faciles.
    """

    def __init__(self, X, Y):
        xp = xp_de(X)
        self.xp = xp
        self.X = X
        self.Y = Y
        self.M, self.N = X.shape
        self.ny, self.nx = self.M - 1, self.N - 1
        if self.ny < 1 or self.nx < 1:
            raise ValueError(f"malla degenerada: {X.shape}")

        # --- vectores de area de cara, S = (dy, -dx) del segmento ---
        self.Sx_xi = Y[1:, :] - Y[:-1, :]
        self.Sy_xi = -(X[1:, :] - X[:-1, :])
        self.Sx_eta = -(Y[:, 1:] - Y[:, :-1])
        self.Sy_eta = X[:, 1:] - X[:, :-1]

        # --- volumen de celda por la formula del cordon ---
        xv = xp.stack([X[:-1, :-1], X[:-1, 1:], X[1:, 1:], X[1:, :-1]], axis=-1)
        yv = xp.stack([Y[:-1, :-1], Y[:-1, 1:], Y[1:, 1:], Y[1:, :-1]], axis=-1)
        self.J = 0.5 * sum(xv[..., k] * yv[..., (k + 1) % 4]
                           - xv[..., (k + 1) % 4] * yv[..., k] for k in range(4))
        self.xc = xv.mean(axis=-1)
        self.yc = yv.mean(axis=-1)

        # --- volumen visto desde cada cara (media de las celdas adyacentes) ---
        Jp_i = _pad_i(self.J, xp)                       # (ny, nx+2)
        Jp_j = _pad_j(self.J, xp)                       # (ny+2, nx)
        V_xi = 0.5 * (Jp_i[:, :-1] + Jp_i[:, 1:])       # (ny, N)
        V_eta = 0.5 * (Jp_j[:-1, :] + Jp_j[1:, :])      # (M, nx)
        eps = 1e-300
        V_xi = xp.maximum(V_xi, eps)
        V_eta = xp.maximum(V_eta, eps)

        # --- vector de area cruzado, promediado a la cara ---
        # En una cara xi (j,i) las cuatro caras eta vecinas son (j,i-1), (j,i),
        # (j+1,i-1) y (j+1,i); se replica en el borde.
        Sxe = _pad_i(self.Sx_eta, xp)                   # (M, nx+2)
        Sye = _pad_i(self.Sy_eta, xp)
        Sx_eta_f = 0.25 * (Sxe[:-1, :-1] + Sxe[:-1, 1:] + Sxe[1:, :-1] + Sxe[1:, 1:])
        Sy_eta_f = 0.25 * (Sye[:-1, :-1] + Sye[:-1, 1:] + Sye[1:, :-1] + Sye[1:, 1:])

        Sxx = _pad_j(self.Sx_xi, xp)                    # (ny+2, N)
        Syx = _pad_j(self.Sy_xi, xp)
        Sx_xi_f = 0.25 * (Sxx[:-1, :-1] + Sxx[:-1, 1:] + Sxx[1:, :-1] + Sxx[1:, 1:])
        Sy_xi_f = 0.25 * (Syx[:-1, :-1] + Syx[:-1, 1:] + Syx[1:, :-1] + Syx[1:, 1:])

        self.a_xi = (self.Sx_xi ** 2 + self.Sy_xi ** 2) / V_xi
        self.b_xi = (self.Sx_xi * Sx_eta_f + self.Sy_xi * Sy_eta_f) / V_xi
        self.a_eta = (self.Sx_eta ** 2 + self.Sy_eta ** 2) / V_eta
        self.b_eta = (self.Sx_eta * Sx_xi_f + self.Sy_eta * Sy_xi_f) / V_eta

    # ------------------------------------------------------------------
    def a(self, dtype):
        """Copia de las metricas en otro tipo (tipicamente float32 en GPU).

        Se filtra por `isinstance`, no por `hasattr`: el modulo numpy expone
        `np.shape` y `np.astype` como funciones, asi que `self.xp` pasaba
        cualquier prueba de pato y se intentaba convertir el propio modulo.
        """
        otro = object.__new__(Metrica)
        otro.__dict__.update(self.__dict__)
        for k, v in self.__dict__.items():
            if isinstance(v, _TIPOS_ARRAY):
                setattr(otro, k, v.astype(dtype))
        return otro

    def oblicuidad(self):
        """|b| / sqrt(a_xi * a_eta) en cada celda: peso de los terminos cruzados."""
        xp = self.xp
        b = 0.5 * (xp.abs(self.b_xi[:, :-1]) + xp.abs(self.b_xi[:, 1:]))
        d = xp.sqrt(xp.maximum(self.a_xi[:, :-1] * self.a_eta[:-1, :], 1e-300))
        return b / xp.maximum(d, 1e-300)

    def flujo_uniforme(self, u0, v0):
        """Flujos de cara de un campo de velocidad uniforme."""
        return (u0 * self.Sx_xi + v0 * self.Sy_xi,
                u0 * self.Sx_eta + v0 * self.Sy_eta)

    def __repr__(self):
        return (f"Metrica({self.ny}x{self.nx} celdas, "
                f"J en [{float(self.J.min()):.2e}, {float(self.J.max()):.2e}])")

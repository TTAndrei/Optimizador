"""Tests de metricas (F1) y operadores de volumenes finitos (F2).

Dos familias:

- **Exactitud en maquina**: corriente uniforme y campo lineal. No son tests de
  precision sino de consistencia de la metrica; si fallan, hay un error de signo
  o de estencil y no tiene sentido medir ordenes de convergencia.
- **Orden de convergencia**: solucion manufacturada sobre una malla
  deliberadamente distorsionada, refinando por dos.

La malla de orden es un cuadrado mapeado, no la malla C: asi el refinado es
exacto (h -> h/2) y el orden medido no se contamina con el cambio de forma del
campo lejano.
"""

import os

import numpy as np
import pytest

from curvo import malla as M
from curvo.metrica import Metrica
from curvo import operadores as op

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def malla_distorsionada(n, amp=0.10):
    """Cuadrado [0,1]^2 mapeado con una distorsion suave. Vertices (n+1, n+1).

    `sin(pi x)`, no `sin(2 pi x)`: con amplitud 0.18 y 2pi el jacobiano se hace
    negativo y los tests miden sobre una malla plegada (salian errores de 1e299).
    Con esta el jacobiano se mantiene por encima de ~0.37 hasta amp = 0.10.
    """
    t = np.linspace(0.0, 1.0, n + 1)
    E, C = np.meshgrid(t, t, indexing="xy")      # E = xi, C = eta
    s = np.sin(np.pi * E) * np.sin(np.pi * C)
    X, Y = E + amp * s, C + amp * s
    assert Metrica(X, Y).J.min() > 0.0, "la malla de prueba se pliega"
    return X, Y


def campo(x, y):
    return np.sin(2.1 * x) * np.cos(1.7 * y)


def lap_campo(x, y):
    return -(2.1 ** 2 + 1.7 ** 2) * campo(x, y)


@pytest.fixture(scope="module")
def met_naca():
    px, py = M.leer_dat(os.path.join(RAIZ, "profiles", "NACA_0012_sharp"))
    X, Y, info = M.generar_c(px, py)
    return Metrica(X, Y), info


# ---------------------------------------------------------------------------
# F1: metrica
# ---------------------------------------------------------------------------
def test_metrica_coincide_con_la_de_malla(met_naca):
    """`Metrica` y `malla.metricas` tienen que dar exactamente lo mismo."""
    met, _ = met_naca
    otra = M.metricas(met.X, met.Y)
    np.testing.assert_array_equal(met.Sx_xi, otra["Sx_xi"])
    np.testing.assert_array_equal(met.Sy_eta, otra["Sy_eta"])
    np.testing.assert_array_equal(met.J, otra["J"])


@pytest.mark.parametrize("alpha", [0.0, 0.0873, -0.35])
def test_corriente_uniforme_da_divergencia_nula(met_naca, alpha):
    met, _ = met_naca
    F_xi, F_eta = met.flujo_uniforme(np.cos(alpha), np.sin(alpha))
    d = op.divergencia(F_xi, F_eta)
    escala = np.abs(F_xi[:, 1:]) + np.abs(F_xi[:, :-1]) + np.abs(F_eta[1:, :]) + np.abs(F_eta[:-1, :])
    assert np.max(np.abs(d) / np.maximum(escala, 1e-300)) < 1e-13


def test_corriente_uniforme_en_float32(met_naca):
    """En f32 la conservacion tiene que aguantar hasta el epsilon de la precision."""
    met, _ = met_naca
    m32 = met.a(np.float32)
    F_xi, F_eta = m32.flujo_uniforme(np.float32(1.0), np.float32(0.0))
    d = op.divergencia(F_xi, F_eta)
    escala = np.abs(F_xi[:, 1:]) + np.abs(F_xi[:, :-1]) + np.abs(F_eta[1:, :]) + np.abs(F_eta[:-1, :])
    assert np.max(np.abs(d) / np.maximum(escala, 1e-30)) < 1e-5


def test_volumenes_positivos_y_suman_el_area(met_naca):
    met, _ = met_naca
    assert met.J.min() > 0.0
    x, y = met.X[0], met.Y[0]                    # contorno j=0, sentido horario
    area_interior = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    x2, y2 = met.X[-1], met.Y[-1]
    area_exterior = 0.5 * abs(np.sum(x2 * np.roll(y2, -1) - np.roll(x2, -1) * y2))
    assert met.J.sum() == pytest.approx(area_exterior - area_interior, rel=1e-9)


def test_oblicuidad_es_pequena_en_un_perfil_real(met_naca):
    """En un NACA 0012 la malla sale casi ortogonal. No asi en las del GA."""
    met, _ = met_naca
    assert np.percentile(met.oblicuidad(), 99) < 0.10


# ---------------------------------------------------------------------------
# F2: operadores
# ---------------------------------------------------------------------------
def test_gradiente_es_consistente_en_campo_lineal():
    X, Y = malla_distorsionada(48)
    met = Metrica(X, Y)
    a, b, c = 0.7, -1.3, 0.4
    phi = a + b * met.xc + c * met.yc
    borde = dict(
        oeste=a + b * 0.5 * (X[:-1, 0] + X[1:, 0]) + c * 0.5 * (Y[:-1, 0] + Y[1:, 0]),
        este=a + b * 0.5 * (X[:-1, -1] + X[1:, -1]) + c * 0.5 * (Y[:-1, -1] + Y[1:, -1]),
        sur=a + b * 0.5 * (X[0, :-1] + X[0, 1:]) + c * 0.5 * (Y[0, :-1] + Y[0, 1:]),
        norte=a + b * 0.5 * (X[-1, :-1] + X[-1, 1:]) + c * 0.5 * (Y[-1, :-1] + Y[-1, 1:]),
    )
    gx, gy = op.gradiente(met, phi, **borde)
    # Green-Gauss con interpolacion lineal entre centroides NO es exacto sobre
    # malla oblicua: el valor de cara cae fuera del segmento que une los dos
    # centroides. El error es de oblicuidad y es O(h^2) -- lo que se comprueba
    # es que sea pequeno y su orden se mide aparte.
    assert np.abs(gx - b).max() / abs(b) < 0.02
    assert np.abs(gy - c).max() / abs(c) < 0.02


def test_laplaciano_es_conservativo():
    """La suma del laplaciano integrado = flujo neto por la frontera, en maquina.

    Esta es la propiedad exacta del esquema, y es la que importa: lo que sale de
    una celda entra en la vecina por la misma cara. **No** se exige que el
    laplaciano de un campo lineal sea cero en maquina -- no lo es en un esquema
    de volumenes finitos sobre malla oblicua, porque el gradiente de cara se
    aproxima. Su consistencia se mide con el orden de convergencia.
    """
    X, Y = malla_distorsionada(48)
    met = Metrica(X, Y)
    b, c = -1.3, 0.4
    phi = b * met.xc + c * met.yc
    borde = dict(
        oeste=b * 0.5 * (X[:-1, 0] + X[1:, 0]) + c * 0.5 * (Y[:-1, 0] + Y[1:, 0]),
        este=b * 0.5 * (X[:-1, -1] + X[1:, -1]) + c * 0.5 * (Y[:-1, -1] + Y[1:, -1]),
        sur=b * 0.5 * (X[0, :-1] + X[0, 1:]) + c * 0.5 * (Y[0, :-1] + Y[0, 1:]),
        norte=b * 0.5 * (X[-1, :-1] + X[-1, 1:]) + c * 0.5 * (Y[-1, :-1] + Y[-1, 1:]),
    )
    F_xi, F_eta = op.flujos_difusivos(met, phi, 1.0, **borde)
    total = op.divergencia(F_xi, F_eta).sum()
    neto = (F_xi[:, -1].sum() - F_xi[:, 0].sum()
            + F_eta[-1, :].sum() - F_eta[0, :].sum())
    escala = np.abs(F_xi).sum() + np.abs(F_eta).sum()
    assert abs(total - neto) / escala < 1e-13


def test_laplaciano_converge_a_segundo_orden():
    """Solucion manufacturada sobre malla distorsionada, refinando por dos.

    Se mide sobre celdas interiores: el borde usa distancia de media celda y su
    error no dice nada del estencil interior, que es lo que se valida aqui.
    """
    errores = []
    for n in (32, 64, 128):
        X, Y = malla_distorsionada(n)
        met = Metrica(X, Y)
        phi = campo(met.xc, met.yc)
        borde = dict(
            oeste=campo(0.5 * (X[:-1, 0] + X[1:, 0]), 0.5 * (Y[:-1, 0] + Y[1:, 0])),
            este=campo(0.5 * (X[:-1, -1] + X[1:, -1]), 0.5 * (Y[:-1, -1] + Y[1:, -1])),
            sur=campo(0.5 * (X[0, :-1] + X[0, 1:]), 0.5 * (Y[0, :-1] + Y[0, 1:])),
            norte=campo(0.5 * (X[-1, :-1] + X[-1, 1:]), 0.5 * (Y[-1, :-1] + Y[-1, 1:])),
        )
        lap = op.laplaciano(met, phi, **borde)
        err = np.abs(lap - lap_campo(met.xc, met.yc))[2:-2, 2:-2]
        errores.append(np.sqrt((err ** 2).mean()))
    ordenes = [np.log2(errores[k] / errores[k + 1]) for k in range(len(errores) - 1)]
    assert min(ordenes) > 1.8, (errores, ordenes)


def test_gradiente_converge_a_segundo_orden():
    errores = []
    for n in (32, 64, 128):
        X, Y = malla_distorsionada(n)
        met = Metrica(X, Y)
        phi = campo(met.xc, met.yc)
        borde = dict(
            oeste=campo(0.5 * (X[:-1, 0] + X[1:, 0]), 0.5 * (Y[:-1, 0] + Y[1:, 0])),
            este=campo(0.5 * (X[:-1, -1] + X[1:, -1]), 0.5 * (Y[:-1, -1] + Y[1:, -1])),
            sur=campo(0.5 * (X[0, :-1] + X[0, 1:]), 0.5 * (Y[0, :-1] + Y[0, 1:])),
            norte=campo(0.5 * (X[-1, :-1] + X[-1, 1:]), 0.5 * (Y[-1, :-1] + Y[-1, 1:])),
        )
        gx, gy = op.gradiente(met, phi, **borde)
        gx_ex = 2.1 * np.cos(2.1 * met.xc) * np.cos(1.7 * met.yc)
        gy_ex = -1.7 * np.sin(2.1 * met.xc) * np.sin(1.7 * met.yc)
        err = np.hypot(gx - gx_ex, gy - gy_ex)[2:-2, 2:-2]
        errores.append(np.sqrt((err ** 2).mean()))
    ordenes = [np.log2(errores[k] / errores[k + 1]) for k in range(len(errores) - 1)]
    assert min(ordenes) > 1.8, (errores, ordenes)


def test_los_terminos_cruzados_importan():
    """Ignorarlos degrada la solucion en una malla no ortogonal.

    Justifica la metrica completa frente a un splitting direccional: no es una
    preferencia estetica, se mide.
    """
    X, Y = malla_distorsionada(64, amp=0.10)
    met = Metrica(X, Y)
    phi = campo(met.xc, met.yc)
    borde = dict(
        oeste=campo(0.5 * (X[:-1, 0] + X[1:, 0]), 0.5 * (Y[:-1, 0] + Y[1:, 0])),
        este=campo(0.5 * (X[:-1, -1] + X[1:, -1]), 0.5 * (Y[:-1, -1] + Y[1:, -1])),
        sur=campo(0.5 * (X[0, :-1] + X[0, 1:]), 0.5 * (Y[0, :-1] + Y[0, 1:])),
        norte=campo(0.5 * (X[-1, :-1] + X[-1, 1:]), 0.5 * (Y[-1, :-1] + Y[-1, 1:])),
    )
    exacto = lap_campo(met.xc, met.yc)
    con = op.laplaciano(met, phi, **borde)

    b_xi, b_eta = met.b_xi.copy(), met.b_eta.copy()
    met.b_xi[:] = 0.0
    met.b_eta[:] = 0.0
    sin = op.laplaciano(met, phi, **borde)
    met.b_xi, met.b_eta = b_xi, b_eta

    e_con = np.abs(con - exacto)[2:-2, 2:-2].max()
    e_sin = np.abs(sin - exacto)[2:-2, 2:-2].max()
    assert e_sin > 5.0 * e_con, (e_con, e_sin)


# ---------------------------------------------------------------------------
# Sobre la malla C real
# ---------------------------------------------------------------------------
def test_laplaciano_es_conservativo_sobre_malla_c(met_naca):
    """Lo mismo sobre la malla C real, con celdas de pared de area 2e-6."""
    met, _ = met_naca
    b, c = 0.9, -0.6
    phi = b * met.xc + c * met.yc
    X, Y = met.X, met.Y
    borde = dict(
        oeste=b * 0.5 * (X[:-1, 0] + X[1:, 0]) + c * 0.5 * (Y[:-1, 0] + Y[1:, 0]),
        este=b * 0.5 * (X[:-1, -1] + X[1:, -1]) + c * 0.5 * (Y[:-1, -1] + Y[1:, -1]),
        sur=b * 0.5 * (X[0, :-1] + X[0, 1:]) + c * 0.5 * (Y[0, :-1] + Y[0, 1:]),
        norte=b * 0.5 * (X[-1, :-1] + X[-1, 1:]) + c * 0.5 * (Y[-1, :-1] + Y[-1, 1:]),
    )
    F_xi, F_eta = op.flujos_difusivos(met, phi, 1.0, **borde)
    total = op.divergencia(F_xi, F_eta).sum()
    neto = (F_xi[:, -1].sum() - F_xi[:, 0].sum()
            + F_eta[-1, :].sum() - F_eta[0, :].sum())
    escala = np.abs(F_xi).sum() + np.abs(F_eta).sum()
    assert abs(total - neto) / escala < 1e-13

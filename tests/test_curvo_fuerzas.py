"""Fuerzas sobre la pared (F5).

Aqui es donde la malla adaptada tiene que pagar. El solver cartesiano da tres
estimadores de Cl con un 7 % de dispersion, una circulacion que cae de 0.65 a
0.29 al agrandar el lazo y un `dCp_TE` del doble de la tolerancia que necesita un
parche de Kutta explicito. Todo eso sale de que la superficie es una escalera
rasterizada y la longitud de arco es sintetica.

Dos familias:

- **Exactitud geometrica y de signo**, sin solver: la normal, el arco, que una
  presion uniforme no empuje, y que la fuerza viscosa valga lo que tiene que
  valer sobre un perfil de Poiseuille analitico. Estos son los que cazan un signo
  cambiado, y son instantaneos.
- **Propiedades del campo resuelto**: Cl nulo a alfa cero sobre la malla
  simetrica, y que el flujo potencial de arranque no tenga circulacion.
"""

import os

import numpy as np
import pytest

from curvo import malla as M
from curvo import fuerzas as fz
from curvo import operadores as op
from curvo.metrica import Metrica
from curvo.solver import Solver

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def perfil():
    px, py = M.leer_dat(os.path.join(RAIZ, "profiles", "NACA_0012_sharp"))
    X, Y, info = M.generar_c(px, py)
    return Metrica(X, Y), info, (X, Y)


def canal(n=64, m=24, largo=4.0, alto=1.0):
    """Canal plano uniforme. La pared de abajo hace de `perfil`."""
    x = np.linspace(0.0, largo, n + 1)
    y = np.linspace(0.0, alto, m + 1)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return Metrica(X, Y), {"perfil": (0, n + 1)}


# ---------------------------------------------------------------------------
# Geometria y signos
# ---------------------------------------------------------------------------
def test_las_normales_apuntan_hacia_el_fluido(perfil):
    """`S_eta[0]` es la normal exterior del cuerpo. Todo el convenio cuelga de esto."""
    met, info, _ = perfil
    g = fz.geometria_pared(met, info)
    centro = np.array([g["x"].mean(), g["y"].mean()])
    hacia_fuera = ((g["x"] - centro[0]) * g["nx"]
                   + (g["y"] - centro[1]) * g["ny"])
    assert (hacia_fuera > 0).all()


def test_la_superficie_del_perfil_es_cerrada(perfil):
    """La suma de los vectores de area sobre el contorno cerrado es cero.

    Es la version de la conservacion geometrica sobre la pared: sale exacta
    porque las caras son los segmentos rectos entre vertices del propio perfil,
    no una reconstruccion.
    """
    met, info, _ = perfil
    g = fz.geometria_pared(met, info)
    suma = np.hypot((g["nx"] * g["ds"]).sum(), (g["ny"] * g["ds"]).sum())
    assert suma / g["ds"].sum() < 1e-12


def test_una_presion_uniforme_no_empuja(perfil):
    """Consecuencia de lo anterior, y lo que el escalonado no puede dar."""
    met, info, _ = perfil
    ceros = np.zeros(met.J.shape)
    f = fz.fuerzas(met, ceros, ceros, np.full(met.J.shape, 3.7), info, nu=0.0)
    assert np.abs(f["F_presion"]).max() < 1e-12


def test_el_arco_es_el_del_poligono(perfil):
    """`ds = |S_eta[0]|` es la longitud real del segmento, no una estimacion."""
    met, info, (X, Y) = perfil
    i0, i1 = info["perfil"]
    lado = np.hypot(np.diff(X[0, i0:i1]), np.diff(Y[0, i0:i1]))
    assert np.abs(fz.geometria_pared(met, info)["ds"] - lado).max() < 1e-14


def test_la_fuerza_viscosa_vale_lo_que_tiene_que_valer():
    """Poiseuille plano analitico: la fuerza sobre la pared es `6 nu U L / h`.

    Se impone el perfil parabolico y se integra; no hace falta resolver nada.
    Es el test que fija **signo y magnitud** del flujo viscoso de pared, que es
    el que `flujos_difusivos` evalua con el mismo estencil que usa el momento.

    El error es de **primer orden** y vale exactamente `dy/(2h)`: el esquema
    evalua el gradiente de pared como `2*u[0]/dy`, y `u[0]` es la parabola en el
    centro de la primera celda. No es un defecto a corregir sino el precio de que
    la fuerza integrada sea la que el esquema quita al fluido; lo que se
    comprueba es que el error baje a la mitad al doblar la malla.
    """
    alto, largo, U, nu = 1.0, 4.0, 1.0, 0.01
    exacta = 6.0 * nu * U * largo / alto
    err = []
    for m in (24, 48, 96):
        met, info = canal(m=m, alto=alto, largo=largo)
        u = 6.0 * U * met.yc * (alto - met.yc) / alto ** 2
        f = fz.fuerzas(met, u, np.zeros_like(u), np.zeros_like(u), info, nu=nu)
        assert abs(f["F_viscosa"][1]) < 1e-12 * exacta
        err.append(abs(f["F_viscosa"][0] - exacta) / exacta)
        assert err[-1] < 0.6 / m                       # el dy/(2h) previsto
    for a, b in zip(err[:-1], err[1:]):
        assert 0.45 < b / a < 0.55


def test_la_circulacion_de_un_torbellino_es_su_intensidad():
    """`circulacion` sobre un vortice potencial devuelve su intensidad.

    Fija a la vez el signo y el factor: el vortice `u = G/(2 pi r) * (-sin, cos)`
    tiene circulacion `+G` en sentido antihorario, y el convenio aeronautico la
    cuenta con el signo cambiado.
    """
    t = np.linspace(-2.0, 2.0, 81)
    X, Y = np.meshgrid(t, t, indexing="xy")
    met = Metrica(X, Y)
    info = {"perfil": (0, met.nx + 1)}
    r2 = np.maximum(met.xc ** 2 + met.yc ** 2, 1e-12)
    G = 1.7
    u = -G * met.yc / (2 * np.pi * r2)
    v = G * met.xc / (2 * np.pi * r2)
    gamma = fz.circulacion(met, u, v, info, bc=dict(oeste=None, este=None),
                           sur=None)
    assert abs(-gamma - G) / G < 0.02


# ---------------------------------------------------------------------------
# Sobre el campo resuelto
# ---------------------------------------------------------------------------
def test_el_arranque_potencial_no_tiene_circulacion(perfil):
    """La proyeccion de la corriente libre es irrotacional: `Gamma = 0`.

    Sirve de linea base del estimador: lo que mida despues es circulacion que ha
    generado el solver, no un sesgo del estimador.
    """
    _, info, (X, Y) = perfil
    s = Solver(X, Y, info, nu=1e-3, alfa=5.0, dt=5e-3)
    g = fz.circulacion(s.met, s.u, s.v, info, bc=dict(oeste=None, este=None),
                       corte=s.corte)
    assert abs(g) < 5e-3


def test_cl_es_cero_a_alfa_cero(perfil):
    """Sobre la malla simetrica y a alfa = 0, la sustentacion es cero de verdad.

    No una tolerancia: la malla es simetrica a 1e-9 por construccion (F0) y el
    perfil es su propia imagen, asi que el unico modo de que salga Cl distinto de
    cero es que la discretizacion rompa la simetria. Es el test que el escalonado
    del IBM nunca pudo pasar.
    """
    _, info, (X, Y) = perfil
    s = Solver(X, Y, info, nu=1e-3, alfa=0.0, dt=5e-3)
    s.correr(40, cada=40)
    e = fz.estimadores_de_cl(s.met, s.u, s.v, s.p, info, s.nu, 1.0, 0.0,
                             bc=dict(oeste=None, este=None), corte=s.corte)
    assert abs(e["superficie"]) < 1e-9
    assert abs(e["delta_cp"]) < 1e-9
    assert abs(fz.delta_cp_te(s.met, s.p, info, 1.0)) < 1e-9

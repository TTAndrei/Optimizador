"""Spalart-Allmaras sobre la malla curvilinea (F5).

Dos cosas que comprobar, y ninguna necesita resolver un flujo:

- **El modelo**: en la capa logaritmica, con `nu_tilde = kappa u_t d`, produccion,
  destruccion y difusion se cancelan. Esa cancelacion **es** la definicion de
  `c_w1`, asi que es una identidad del modelo: si `S~`, `r`, `f_w` o algun signo
  estan mal, no cierra. Es el test que caza una transcripcion mala.
- **La distancia a pared**, que en malla adaptada es donde el solver cartesiano
  se equivoca un 35 %. Aqui es geometria exacta, y en la estela tiene que medir
  al **perfil**, no al corte de estela sobre el que se apoya la malla.
"""

import os

import numpy as np
import pytest

from curvo import malla as M
from curvo import turbulencia as tu
from curvo.metrica import Metrica
from curvo.solver import Solver

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = tu.CONSTANTES


@pytest.fixture(scope="module")
def perfil():
    px, py = M.leer_dat(os.path.join(RAIZ, "profiles", "NACA_0012_sharp"))
    return M.generar_c(px, py)


# ---------------------------------------------------------------------------
# El modelo
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nu, tope", [(1e-6, 0.2), (1e-8, 5e-3), (1e-12, 5e-3)])
def test_el_equilibrio_de_la_capa_logaritmica_cierra(nu, tope):
    """`P - D + difusion = 0` con el perfil de equilibrio de pared.

    El residuo que queda no es error: es el termino `f_v2`, que se va como `1/chi`
    y por eso el tope aflora al bajar `nu`. Con `chi` topado en 5e3 el suelo es
    2e-3, que es exactamente `4 * d(f_w)/dr * 1/chi_max`.
    """
    k, u_t = C["kappa"], 1.0
    d = np.logspace(-4, -1, 40)
    nu_tilde = k * u_t * d
    omega = u_t / (k * d)

    produccion, destruccion, _ = tu.terminos(nu_tilde, omega, d, nu)
    difusion = (k * u_t) ** 2 * (1.0 + C["cb2"]) / C["sigma"]
    assert np.abs((produccion - destruccion + difusion) / produccion).max() < tope


def test_cw1_es_la_combinacion_que_cierra_la_capa_log():
    assert C["cw1"] == pytest.approx(
        C["cb1"] / C["kappa"] ** 2 + (1.0 + C["cb2"]) / C["sigma"])


def test_la_viscosidad_turbulenta_tiende_a_nu_tilde():
    """`nu_t = nu_tilde f_v1`: vale `nu_tilde` lejos de la pared y cero pegado a ella."""
    nu = 1e-6
    nu_tilde = np.array([1e-9, 1e-6, 1e-3])
    nu_t = tu.viscosidad_turbulenta(nu_tilde, nu)
    assert nu_t[0] / nu_tilde[0] < 1e-8                  # chi = 1e-3, amortiguado
    assert nu_t[-1] / nu_tilde[-1] > 0.999               # chi = 1e3, sin amortiguar
    assert (nu_t <= nu_tilde).all()


# ---------------------------------------------------------------------------
# Distancia a pared
# ---------------------------------------------------------------------------
def test_la_distancia_a_pared_es_exacta():
    """Sobre una placa plana la distancia tiene que ser `|y|`, sin aproximacion."""
    x = np.linspace(-1.0, 2.0, 61)
    y = np.linspace(0.0, 1.0, 21)
    X, Y = np.meshgrid(x, y, indexing="xy")
    met = Metrica(X, Y)
    # la "pared" son los segmentos de j=0 con 0 <= x <= 1
    i0 = int(np.searchsorted(x, 0.0))
    i1 = int(np.searchsorted(x, 1.0)) + 1
    d = tu.distancia_a_pared(met, {"perfil": (i0, i1)})
    encima = (met.xc > 0.05) & (met.xc < 0.95)
    assert np.abs(d[encima] - met.yc[encima]).max() < 1e-12


def test_la_distancia_en_la_estela_mide_al_perfil(perfil):
    """En `j=0` de la estela, `d` es la distancia al borde de salida, no cero.

    Es el motivo de no usar el arco acumulado a lo largo de la linea eta, que
    seria lo barato: ahi `j=0` es el corte de estela, no una pared.
    """
    X, Y, info = perfil
    met = Metrica(X, Y)
    d = tu.distancia_a_pared(met, info)
    i0 = int(info["perfil"][0])
    estela = d[0, :i0]
    assert (estela > 0).all()
    assert np.all(np.diff(estela) < 0)                   # crece hacia la salida
    assert estela[-1] < 0.01                             # pegada al borde de salida
    pared = d[0, i0:int(info["perfil"][1]) - 1]
    assert pared.max() < 2e-3                            # media celda de pared


# ---------------------------------------------------------------------------
# Acoplado
# ---------------------------------------------------------------------------
def test_nu_tilde_no_se_vuelve_negativa(perfil):
    """La destruccion va implicita por puntos: divide, nunca resta.

    Se fuerza con un `dt` absurdo, cien veces el que usaria el solver. Con la
    destruccion explicita esto daria `nu_tilde` negativa y `nu_t` sin sentido.
    """
    X, Y, info = perfil
    s = Solver(X, Y, info, nu=1e-5, alfa=5.0, dt=0.5, turbulento=True)
    for _ in range(5):
        s.paso()
        assert float(s.nu_tilde.min()) >= 0.0
        assert np.isfinite(float(s.nu_tilde.max()))
    assert float(s.nu_t.min()) >= 0.0

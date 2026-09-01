"""Mapa de cuerpos y contorno exterior."""
import os
import sys

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import Simulador2D as S

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "agent_tests"))
import formas_dominio as fd


def _malla(nx=120, ny=60, Lx=6.0, Ly=3.0):
    return S.Mesh(Lx, Ly, 0.0, 1.0, 0.0, Lx / nx, Ly / ny)


def _circulo(cx, cy, r, n=360):
    th = np.linspace(0, 2 * np.pi, n + 1)
    return cx + r * np.cos(th), cy + r * np.sin(th)


def test_dos_cuerpos_se_distinguen():
    m = _malla()
    a = m.add_body(*_circulo(1.5, 1.5, 0.4), nombre="a")
    b = m.add_body(*_circulo(3.5, 1.5, 0.4), nombre="b")
    assert (a, b) == (1, 2)

    bid = cp.asnumpy(m.body_id)
    assert (bid == 1).sum() > 0 and (bid == 2).sum() > 0
    assert np.array_equal(bid > 0, cp.asnumpy(m.solid))
    assert m._geom_arbitrary is True     # dos cuerpos: ruta nueva


def test_un_solo_cuerpo_no_activa_la_ruta_nueva():
    m = _malla()
    m.add_body(*_circulo(1.5, 1.5, 0.4))
    assert m._geom_arbitrary is False


def test_area_rasterizada_de_un_circulo():
    """La mascara es par/impar sobre centros de celda, asi que el area
    rasterizada converge al area exacta. Si no, la rasterizacion esta mal."""
    m = _malla(nx=600, ny=300)
    m.add_body(*_circulo(3.0, 1.5, 0.5))
    area = float(cp.asnumpy(m.solid).sum()) * (6.0 / 600) * (3.0 / 300)
    assert area == pytest.approx(np.pi * 0.25, rel=0.01)


def test_contorno_exterior_deja_el_fluido_dentro():
    """Un canal: el fluido queda dentro del rectangulo y las paredes salen
    solidas arriba y abajo, tocando el borde del dominio."""
    m = _malla(nx=240, ny=120, Lx=6.0, Ly=2.0)
    x, y = fd.canal(Lx=6.0, Ly=2.0, h=1.0)
    m.add_body(x, y, rol="exterior", nombre="canal")

    solid = cp.asnumpy(m.solid)
    yc = np.asarray(m.Y_1d_f64)
    # En una columna del medio: solido fuera de [0.5, 1.5], fluido dentro.
    col = solid[:, solid.shape[1] // 2]
    assert col[(yc > 0.55) & (yc < 1.45)].sum() == 0
    assert col[yc < 0.45].all() and col[yc > 1.55].all()
    assert m._geom_arbitrary is True


def test_pared_del_conducto_llega_al_borde_del_dominio():
    """Es donde el IBM se apaga (los ghost excluyen los cuatro bordes) y donde
    apply_boundaries pisaba. La pared TIENE que existir ahi."""
    m = _malla(nx=240, ny=120, Lx=6.0, Ly=2.0)
    x, y = fd.canal(Lx=6.0, Ly=2.0, h=1.0)
    m.add_body(x, y, rol="exterior")

    izq = cp.asnumpy(m.solid[:, 0])
    yc = np.asarray(m.Y_1d_f64)
    assert izq[yc < 0.45].all(), "la pared no llega al borde izquierdo"
    assert izq[(yc > 0.55) & (yc < 1.45)].sum() == 0, "la entrada esta tapada"

"""Fronteras por parche: que el camino nuevo haga lo mismo que el viejo, y que
arregle lo que el viejo rompia.

El test 3 es el que importa para la compatibilidad: con los cuatro lados
uniformes, la ruta por celda tiene que dar los MISMOS campos que la ruta de
siempre, valor a valor. El test 4 es el que importa para los conductos: una
celda de borde que es solida no puede recibir velocidad de entrada encima.
"""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import Simulador2D as S


def _malla(nx=40, ny=30, Lx=1.0, Ly=0.75):
    return S.Mesh(Lx, Ly, 0.0, 1.0, 0.0, Lx / nx, Ly / ny)


def _fronteras_clasicas(m):
    m.set_boundary("left", "inflow", (1.0, 0.0))
    m.set_boundary("right", "outflow", 0.0)
    m.set_boundary("top", "slip")
    m.set_boundary("bottom", "slip")


def test_set_boundary_rellena_el_lado_entero():
    m = _malla()
    _fronteras_clasicas(m)
    assert bool(cp.all(m._bc_code["left"] == S.BC_INFLOW))
    assert bool(cp.all(m._bc_code["right"] == S.BC_OUTFLOW))
    assert bool(cp.all(m._bc_code["top"] == S.BC_SLIP))
    assert bool(cp.all(m._bc_u["left"] == 1.0))
    assert bool(cp.all(m._bc_p["right"] == 0.0))
    # Un caso clasico no debe activar la ruta nueva.
    assert m._geom_arbitrary is False


def test_parche_cubre_solo_su_tramo():
    m = _malla(Ly=1.0, ny=40)
    m.set_boundary("left", "noslip")
    m.set_boundary_patch("left", 0.4, 0.6, "inflow", (1.0, 0.0))

    y = np.asarray(m.Y_1d_f64)
    dentro = (y >= 0.4) & (y < 0.6)
    code = cp.asnumpy(m._bc_code["left"])
    assert np.all(code[dentro] == S.BC_INFLOW)
    assert np.all(code[~dentro] == S.BC_NOSLIP)
    assert m._geom_arbitrary is True


def test_ruta_por_celda_identica_a_la_clasica():
    """Compatibilidad estricta: mismo campo, valor a valor."""
    ma, mb = _malla(), _malla()
    for m in (ma, mb):
        _fronteras_clasicas(m)
        # Campo arbitrario pero reproducible, para que las copias de Neumann
        # tengan algo que copiar.
        rng = np.random.default_rng(0)
        m.u[:] = cp.asarray(rng.standard_normal(m.u.shape), dtype=m.u.dtype)
        m.v[:] = cp.asarray(rng.standard_normal(m.v.shape), dtype=m.v.dtype)
        m.p[:] = cp.asarray(rng.standard_normal(m.p.shape), dtype=m.p.dtype)

    ma.apply_boundaries()                    # ruta clasica
    mb._geom_arbitrary = True
    mb._rebuild_bc_masks()
    mb.apply_boundaries()                    # ruta por celda

    for campo in ("u", "v", "p"):
        a, b = cp.asnumpy(getattr(ma, campo)), cp.asnumpy(getattr(mb, campo))
        assert np.array_equal(a, b), f"{campo} difiere"


def test_borde_solido_no_recibe_inflow():
    """El bug. apply_boundaries corre DESPUES del IBM: si no se marca WALL, la
    celda de pared que muere en el borde recibe velocidad de corriente libre
    encima del cero que le acaba de poner el IBM, y el conducto fuga."""
    m = _malla(Ly=1.0, ny=40)
    m.set_boundary("left", "inflow", (1.0, 0.0))
    m.set_boundary_patch("left", None, None, "inflow", (1.0, 0.0))

    # Media columna izquierda solida, como la pared de un conducto.
    m.solid[:20, 0] = True
    m._rebuild_bc_masks()

    m.u[:] = 0.0
    m.v[:] = 0.0
    m.apply_boundaries()

    u0 = cp.asnumpy(m.u[:, 0])
    assert np.all(u0[:20] == 0.0), "el inflow ha pisado la pared"
    assert np.all(u0[20:] == 1.0), "el inflow no ha entrado por donde debia"


def test_dirichlet_de_presion_no_entra_en_la_pared():
    m = _malla()
    m.set_boundary("right", "outflow", 0.0)
    m.set_boundary_patch("right", None, None, "outflow", 0.0)
    m.solid[:10, -1] = True
    m._rebuild_bc_masks()

    fix = cp.asnumpy(m.fixed_pressure_mask_bc[:, -1])
    assert not fix[:10].any(), "presion fijada dentro de la pared"
    assert fix[10:].all()


def test_tramo_vacio_es_un_error():
    m = _malla()
    with pytest.raises(ValueError):
        m.set_boundary_patch("left", 0.5, 0.5, "inflow", 1.0)

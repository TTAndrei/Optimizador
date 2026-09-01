"""Dominios importados: canal de Poiseuille y tobera. Con verdad conocida.

Cortos a proposito (mallas de pocos miles de celdas, ~2000 iteraciones): lo que
se comprueba aqui es que la MECANICA es correcta —que las paredes existen, que
no fugan, que la entrada entra por donde debe— no la exactitud del solver, que
ya esta medida en otro sitio.

El balance de masa es el test que importa. Sin el arreglo del orden de
`apply_boundaries` (que corre despues del IBM y pisaba las celdas de pared del
borde con velocidad de corriente libre), el caudal de salida no cuadra con el de
entrada y este test falla ruidosamente.
"""
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


def _escena(x, y, parches, Lx, Ly, refinado):
    return {
        "contornos": [{"x": list(map(float, x)), "y": list(map(float, y)),
                       "rol": "exterior", "pared": "noslip",
                       "nombre": "dominio"}],
        "parches": parches,
        "refinado": refinado,
    }


def _correr(escena, Lx, Ly, dx, nu, iters, U=1.0):
    return S.main(
        filepath=None, escena=escena,
        Lx=Lx, Ly=Ly, dx_min=dx, factor_expansion=1.05,
        v0x=U, v0y=0.0, rho=1.0, nu=nu, chord=1.0, CFL=0.5,
        turb_model="none", transition_model="none",
        wall_treatment="consistent", advection_scheme="maccormack",
        mg_niveles_max=2, mg_max_outer=2, mg_cycles_per_outer=3,
        iteraciones=iters, guardado=max(1, iters // 4),
        stop_on_convergence=False, stop_on_clcd_convergence=False,
        graficos=False, live_view=False, mostrar_malla=False,
        shm_publish=False)


def _altura_libre(m, j):
    """Altura de fluido de la columna j, con los volumenes de control reales."""
    solid = cp.asnumpy(m.solid[:, j])
    return float(np.sum(cp.asnumpy(m.vol_y)[~solid]))


def _caudal(m, j):
    """∫u·dy en la columna j, con los volumenes de control de la malla."""
    u = cp.asnumpy(m.u[:, j])
    solid = cp.asnumpy(m.solid[:, j])
    dy = cp.asnumpy(m.vol_y)
    return float(np.sum(u * dy * (~solid)))


# ----------------------------------------------------------------------
def test_canal_conserva_el_caudal():
    """Canal recto. Las paredes mueren en el borde izquierdo y derecho, que es
    justo donde el IBM se apaga y donde apply_boundaries pisaba."""
    Lx, Ly, h, dx = 4.0, 2.0, 1.0, 0.05
    x, y = fd.canal(Lx=Lx, Ly=Ly, h=h)
    parches = [
        {"lado": "left", "tipo": "inflow", "valor": [1.0, 0.0],
         "desde": 0.5, "hasta": 1.5},
        {"lado": "right", "tipo": "outflow", "valor": 0.0,
         "desde": 0.5, "hasta": 1.5},
        {"lado": "top", "tipo": "noslip", "valor": None,
         "desde": None, "hasta": None},
        {"lado": "bottom", "tipo": "noslip", "valor": None,
         "desde": None, "hasta": None},
    ]
    m = _correr(_escena(x, y, parches, Lx, Ly, (0.0, Lx, 0.4, 1.6)),
                Lx, Ly, dx, nu=0.05, iters=2000)

    q_in = _caudal(m, 1)
    q_out = _caudal(m, m.nx - 2)
    assert q_in > 0.5 * h, f"apenas entra caudal: {q_in:.4f}"
    assert abs(q_out - q_in) / q_in < 0.02, \
        f"el canal fuga: entra {q_in:.5f}, sale {q_out:.5f}"


def test_canal_da_perfil_parabolico():
    """Poiseuille: u(y) = 1.5·ū·(1 − (2(y−yc)/h)²). Con 20 celdas de ancho y
    paredes IBM la tolerancia es del 10%; lo que se valida es la FORMA."""
    Lx, Ly, h, dx = 4.0, 2.0, 1.0, 0.05
    x, y = fd.canal(Lx=Lx, Ly=Ly, h=h)
    parches = [
        {"lado": "left", "tipo": "inflow", "valor": [1.0, 0.0],
         "desde": 0.5, "hasta": 1.5},
        {"lado": "right", "tipo": "outflow", "valor": 0.0,
         "desde": 0.5, "hasta": 1.5},
        {"lado": "top", "tipo": "noslip", "valor": None, "desde": None, "hasta": None},
        {"lado": "bottom", "tipo": "noslip", "valor": None, "desde": None, "hasta": None},
    ]
    m = _correr(_escena(x, y, parches, Lx, Ly, (0.0, Lx, 0.4, 1.6)),
                Lx, Ly, dx, nu=0.05, iters=4000)

    j = int(0.75 * m.nx)
    u = cp.asnumpy(m.u[:, j])
    solid = cp.asnumpy(m.solid[:, j])
    yc = np.asarray(m.Y_1d_f64)
    dentro = ~solid
    eta = 2.0 * (yc[dentro] - 1.0) / h
    perfil = u[dentro]
    teorico = 1.0 - eta ** 2
    # Se compara la FORMA: se escala el teorico al maximo medido.
    teorico = teorico * perfil.max() / max(teorico.max(), 1e-12)
    err = np.abs(perfil - teorico).max() / max(perfil.max(), 1e-12)
    assert perfil.max() > 0.5, f"el flujo no se ha desarrollado: max={perfil.max():.3f}"
    assert err < 0.10, f"el perfil no es parabólico: error máximo {100*err:.1f} %"


def test_tobera_acelera_el_flujo():
    """Contraccion 2:1 en incompresible: u_out/u_in = h_in/h_out = 2 exacto.
    Sale de la conservacion, no de ninguna correlacion."""
    Lx, Ly, dx = 4.0, 2.0, 0.05
    x, y = fd.tobera(Lx=Lx, Ly=Ly, h_in=1.2, h_out=0.6,
                     x_ini=1.0, x_fin=3.0)
    parches = [
        {"lado": "left", "tipo": "inflow", "valor": [1.0, 0.0],
         "desde": 0.4, "hasta": 1.6},
        {"lado": "right", "tipo": "outflow", "valor": 0.0,
         "desde": 0.7, "hasta": 1.3},
        {"lado": "top", "tipo": "noslip", "valor": None, "desde": None, "hasta": None},
        {"lado": "bottom", "tipo": "noslip", "valor": None, "desde": None, "hasta": None},
    ]
    m = _correr(_escena(x, y, parches, Lx, Ly, (0.0, Lx, 0.3, 1.7)),
                Lx, Ly, dx, nu=0.05, iters=3000)

    q_in = _caudal(m, 1)
    q_out = _caudal(m, m.nx - 2)
    assert abs(q_out - q_in) / q_in < 0.03, \
        f"la tobera no conserva masa: entra {q_in:.5f}, sale {q_out:.5f}"

    # La velocidad MEDIA es la que tiene que ir con la razon de alturas:
    # u_media = Q/h, y Q se conserva, asi que u_out/u_in = h_in/h_out = 2 exacto.
    # El maximo no vale: a la entrada el perfil es el plano que se impone y a la
    # salida ya esta desarrollado, donde u_max = 1.5*u_media, y eso mete un 1.5
    # espurio que no es la contraccion.
    h_in = _altura_libre(m, 1)
    h_out = _altura_libre(m, m.nx - 2)
    razon_geom = h_in / h_out
    razon_vel = (q_out / h_out) / (q_in / h_in)
    assert abs(razon_geom - 2.0) < 0.2, \
        f"la geometría no da una contracción 2:1: h_in={h_in:.3f}, h_out={h_out:.3f}"
    assert abs(razon_vel - razon_geom) / razon_geom < 0.05, \
        f"la aceleración ({razon_vel:.2f}) no sigue a la geometría ({razon_geom:.2f})"

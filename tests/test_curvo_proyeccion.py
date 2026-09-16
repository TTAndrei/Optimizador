"""Tests de la proyeccion de presion (F4).

Tres familias:

- **Exactitud en maquina**: que el operador del multigrid sea la divergencia del
  gradiente compacto, que una presion constante no mueva nada, que el corte de
  estela cierre. No son tests de precision: si fallan hay un error de signo o de
  estencil.
- **Desacoplo par-impar**: el defecto que el solver cartesiano arrastra (46.2 %
  de la divergencia residual en modo tablero, `docs/OPTIMIZACION_SOLVER.md`
  §2.1) y que el gradiente compacto tiene que quitar por construccion.
- **Casos con solucion conocida**: tobera convergente y canal curvo.
"""

import os

import numpy as np
import pytest

from curvo import malla as M
from curvo import conveccion as cv
from curvo import operadores as op
from curvo import proyeccion as pr
from curvo.metrica import Metrica
from curvo.multigrid import jerarquia, resolver, resolver_pcg

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Salida por el este, el resto sin flujo. Es la unica configuracion que no deja
# el Poisson singular, y es la de la malla C (plano de salida aguas abajo).
BC_SALIDA = dict(oeste=None, este=0.0, sur=None, norte=None)


def cuadrado(n, amp=0.10):
    """Cuadrado [0,1]^2 con la misma distorsion que usan F2 y F3."""
    t = np.linspace(0.0, 1.0, n + 1)
    E, C = np.meshgrid(t, t, indexing="xy")
    s = np.sin(np.pi * E) * np.sin(np.pi * C)
    return Metrica(E + amp * s, C + amp * s)


def tobera(n=48, m=24, largo=4.0, h0=1.0, h1=0.5):
    """Conducto plano que estrecha de `h0` a `h1`. Paredes en sur y norte."""
    x = np.linspace(0.0, largo, n + 1)
    t = np.linspace(0.0, 1.0, m + 1)
    h = h0 + (h1 - h0) * x / largo
    return Metrica(np.repeat(x[None, :], m + 1, axis=0), t[:, None] * h[None, :])


def anillo(n=48, m=16, r0=4.0, r1=5.0, ang=np.pi / 3):
    """Sector de corona circular: canal de hueco 1 curvado con radio medio 4.5.

    El radio es grande frente al hueco a proposito: asi el perfil desarrollado
    de Stokes se aparta poco de la parabola y la comparacion tiene sentido.
    """
    th = np.linspace(0.0, ang, n + 1)
    r = np.linspace(r1, r0, m + 1)
    return Metrica(r[:, None] * np.cos(th)[None, :], r[:, None] * np.sin(th)[None, :])


@pytest.fixture(scope="module")
def malla_c():
    px, py = M.leer_dat(os.path.join(RAIZ, "profiles", "NACA_0012_sharp"))
    X, Y, info = M.generar_c(px, py)
    met = Metrica(X, Y)
    return met, info, cv.corte_de_estela(info, met.nx)


def flujo_aleatorio(met, semilla=0):
    rng = np.random.default_rng(semilla)
    return (rng.standard_normal(met.a_xi.shape) * 0.01,
            rng.standard_normal(met.a_eta.shape) * 0.01)


def div_relativa(m_xi, m_eta):
    d = op.divergencia(m_xi, m_eta)
    esc = (np.abs(m_xi[:, 1:]) + np.abs(m_xi[:, :-1])
           + np.abs(m_eta[1:]) + np.abs(m_eta[:-1]))
    return float(np.max(np.abs(d) / np.maximum(esc, 1e-300)))


# ---------------------------------------------------------------------------
# El operador del multigrid es D.G
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("amp", [0.0, 0.10])
def test_el_operador_del_multigrid_es_la_divergencia_del_gradiente(amp):
    """`A.aplicar(p) = -div(grad(p).S) + frontera`, en maquina, no a 1e-6.

    El criterio del plan pedia 1e-6. Sale exacto porque la matriz **es** la del
    operador de conveccion-difusion con flujo nulo y nu=1, es decir, los mismos
    coeficientes de cara que evalua el gradiente. En el cartesiano la ruta
    `face_flux` y el suavizador usaban estenciles distintos y de ahi venia el
    46.2 % de divergencia residual en modo par-impar.
    """
    met = cuadrado(48, amp)
    A, b = pr.sistema_presion(met, BC_SALIDA)
    p = np.random.default_rng(0).standard_normal(met.J.shape)
    o_xi, o_eta, _, _ = pr._partes(met, p, pr._bc(BC_SALIDA), None)
    izq = A.aplicar(p)
    der = -op.divergencia(o_xi, o_eta)
    assert np.abs(izq - der).max() / np.abs(izq).max() < 1e-14


def test_una_presion_constante_no_mueve_nada():
    """El modo nulo del gradiente. Con todo Neumann el flujo de cara es cero."""
    met = cuadrado(32, 0.10)
    F_xi, F_eta = pr.flujos_de_presion(met, np.full(met.J.shape, 3.7))
    assert np.abs(F_xi).max() == 0.0 and np.abs(F_eta).max() == 0.0


def test_neumann_puro_no_se_admite():
    """Sin ninguna cara Dirichlet el Poisson es singular; se rechaza y no se ancla."""
    with pytest.raises(ValueError):
        pr.sistema_presion(cuadrado(16), dict(oeste=None, este=None,
                                              sur=None, norte=None))


# ---------------------------------------------------------------------------
# La proyeccion hace lo que dice
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("amp", [0.0, 0.05, 0.10])
def test_la_proyeccion_anula_la_divergencia(amp):
    met = cuadrado(64, amp)
    m_xi, m_eta = flujo_aleatorio(met)
    mx, me, _, _ = pr.proyectar(met, m_xi, m_eta, bc=BC_SALIDA, correcciones=24)
    d0 = np.abs(op.divergencia(m_xi, m_eta)).max()
    assert np.abs(op.divergencia(mx, me)).max() / d0 < 1e-12


def test_las_correcciones_cruzadas_convergen():
    """Los terminos cruzados van al termino independiente: hay que iterarlos.

    Con 0 iteraciones queda la divergencia que mete la oblicuidad de la malla.
    Converge geometricamente, y el ritmo lo fija la oblicuidad.
    """
    met = cuadrado(64, 0.10)
    m_xi, m_eta = flujo_aleatorio(met)
    d0 = np.abs(op.divergencia(m_xi, m_eta)).max()
    err = []
    for k in (0, 4, 8, 12):
        mx, me, _, _ = pr.proyectar(met, m_xi, m_eta, bc=BC_SALIDA, correcciones=k)
        err.append(np.abs(op.divergencia(mx, me)).max() / d0)
    assert err[0] > 1e-2
    razones = [b / a for a, b in zip(err[:-1], err[1:])]
    assert max(razones) < 0.05                  # factor < 0.5 por iteracion


def test_un_campo_solenoidal_no_se_mueve():
    """Flujos de una funcion de corriente: divergencia cero exacta, nada que hacer.

    `psi = 0` en todo el contorno, asi que no hay flujo por ninguna cara de
    frontera y el campo ya cumple la condicion de la pared.
    """
    met = cuadrado(48, 0.10)
    t = np.linspace(0.0, 1.0, met.nx + 1)
    psi = np.sin(np.pi * t)[None, :] * np.sin(np.pi * t)[:, None]
    m_xi, m_eta = cv.flujos_de_corriente(psi)
    mx, me, _, _ = pr.proyectar(met, m_xi, m_eta, bc=BC_SALIDA, correcciones=4)
    escala = np.abs(m_xi).max()
    assert np.abs(mx - m_xi).max() / escala < 1e-12
    assert np.abs(me - m_eta).max() / escala < 1e-12


def test_ida_y_vuelta_entre_velocidad_y_flujos():
    met = cuadrado(24, 0.10)
    u = np.full(met.J.shape, 2.0)
    v = np.full(met.J.shape, -0.7)
    uu, vv = pr.velocidad_de_flujos(met, *pr.flujos_de_velocidad(met, u, v))
    assert np.abs(uu - u).max() < 1e-14 and np.abs(vv - v).max() < 1e-14


# ---------------------------------------------------------------------------
# Desacoplo par-impar
# ---------------------------------------------------------------------------
def test_el_tablero_de_ajedrez_no_esta_en_el_nucleo():
    """`p_{i+1} - p_i` ve el tablero; `p_{i+1} - p_{i-1}`, el estencil ancho, no.

    Es la razon por la que la presion no se desacopla sin ningun termino de
    Rhie-Chow explicito: la correccion de cara usa la diferencia compacta.
    """
    met = cuadrado(32, 0.10)
    A, _ = pr.sistema_presion(met, BC_SALIDA)
    ny, nx = met.J.shape
    chk = (-1.0) ** (np.arange(ny)[:, None] + np.arange(nx)[None, :])
    assert np.abs(A.aplicar(chk)).max() / np.abs(A.aP).max() > 0.5


@pytest.mark.parametrize("n", [32, 64, 128])
def test_la_divergencia_residual_no_es_un_tablero(n):
    """El criterio de F4: menos del 5 % del residuo en modo par-impar."""
    met = cuadrado(n, 0.10)
    mx, me, _, _ = pr.proyectar(met, *flujo_aleatorio(met), bc=BC_SALIDA,
                                correcciones=8)
    assert pr.fraccion_par_impar(op.divergencia(mx, me)) < 0.05


# ---------------------------------------------------------------------------
# Multigrid
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [32, 64, 128])
def test_el_pcg_no_se_degrada_al_refinar(n):
    """F4 pide factor < 0.3. El ciclo V solo topa en 0.50 y el PCG lo baja a 0.05.

    0.50 es el techo de la aglomeracion sin suavizar, ya medido en F3: el
    coeficiente de cara gruesa sale x2 y el cociente de Rayleigh recupera la
    mitad que falta pero no el ancho de banda. El ciclo V sigue siendo un buen
    precondicionador aunque no sea un buen solver.
    """
    met = cuadrado(n, 0.10)
    A, b = pr.sistema_presion(met, BC_SALIDA)
    rhs = b + np.random.default_rng(1).standard_normal(A.aP.shape)
    niveles = jerarquia(A)
    A.b = rhs.copy()
    _, pcg = resolver_pcg(A, None, tol=1e-11, ciclos=80, niveles=niveles)
    A.b = rhs.copy()
    _, ciclo = resolver(A, None, tol=1e-11, ciclos=80, niveles=niveles)
    assert pcg["factor"] < 0.3
    assert ciclo["factor"] > 0.4               # el techo de la aglomeracion


def test_el_pcg_aguanta_la_malla_c(malla_c):
    """Con corte de estela y relacion de aspecto hasta 500, mismo factor que sin el.

    Es lo que fuerza a meter el corte dentro de la linea eta del suavizador: con
    `aC` retrasado (llega a la mitad de la diagonal) el factor sube de 0.10 a
    0.64 y el ciclo V solo se queda en 0.91, sin converger en 80 ciclos.
    """
    met, _, corte = malla_c
    rng = np.random.default_rng(3)
    factores = {}
    for nombre, c in (("con", corte), ("sin", None)):
        A, b = pr.sistema_presion(met, dict(oeste=0.0, este=0.0, sur=None,
                                            norte=None), c)
        A.b = b + rng.standard_normal(A.aP.shape) * met.J
        _, info = resolver_pcg(A, None, tol=1e-11, ciclos=80,
                               niveles=jerarquia(A))
        factores[nombre] = info["factor"]
    assert factores["con"] < 0.3
    assert factores["con"] < 1.5 * factores["sin"]


# ---------------------------------------------------------------------------
# Malla C
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("u0", [1.0, 10.0])
def test_la_malla_c_proyecta_la_corriente_libre(malla_c, u0):
    """Corriente uniforme con la pared cerrada: sale el flujo potencial.

    Es el caso de la etapa 1 completo -- corte de estela activo, relacion de
    aspecto hasta 500 -- y la divergencia tiene que irse a cero con la pared
    impermeable, no a pesar de ella.
    """
    met, _, corte = malla_c
    bc = dict(oeste=0.0, este=0.0, sur=None, norte=None)
    m_xi, m_eta = cv.flujos_de_campo(
        met, lambda x, y: (u0 * np.ones_like(x), np.zeros_like(x)))
    m_eta = m_eta.copy()
    m_eta[0] = np.where(corte, m_eta[0], 0.0)
    mx, me, p, _ = pr.proyectar(met, m_xi, m_eta, bc=bc, corte=corte,
                                correcciones=4)
    assert div_relativa(mx, me) < 1e-8
    # el corte es UNA cara: lo que sale por la rama baja entra por la alta
    assert np.abs(me[0][corte] + me[0][::-1][corte]).max() == 0.0


def test_el_perfil_simetrico_da_presion_simetrica(malla_c):
    """A alfa = 0 la presion sale simetrica en maquina, luego Cl = 0 exacto.

    Es la propiedad que el escalonado del IBM no podia dar y la que hace el test
    de sustentacion nula un test de verdad y no una tolerancia.
    """
    met, _, corte = malla_c
    bc = dict(oeste=0.0, este=0.0, sur=None, norte=None)
    m_xi, m_eta = cv.flujos_de_campo(
        met, lambda x, y: (np.ones_like(x), np.zeros_like(x)))
    m_eta = m_eta.copy()
    m_eta[0] = np.where(corte, m_eta[0], 0.0)
    _, _, p, _ = pr.proyectar(met, m_xi, m_eta, bc=bc, corte=corte,
                              correcciones=4)
    assert np.abs(p - p[:, ::-1]).max() / np.abs(p).max() < 1e-12


# ---------------------------------------------------------------------------
# Casos con solucion conocida
# ---------------------------------------------------------------------------
def test_la_tobera_acelera_la_razon_de_alturas():
    """Tobera 2:1. La proyeccion sola ya da el flujo potencial del conducto."""
    met = tobera()
    m_xi = np.zeros_like(met.a_xi)
    m_eta = np.zeros_like(met.a_eta)
    m_xi[:, 0] = met.Sx_xi[:, 0]                       # entrada uniforme, u = 1
    mx, me, _, _ = pr.proyectar(met, m_xi, m_eta, bc=BC_SALIDA, correcciones=8)

    entra = float(m_xi[:, 0].sum())
    sale = float(mx[:, -1].sum())
    assert abs(sale - entra) / entra < 0.01            # balance de masa
    assert np.abs(me[0]).max() == 0.0                  # paredes impermeables
    assert np.abs(me[-1]).max() == 0.0
    media = sale / 0.5                                 # altura de salida
    assert abs(media - 2.0) / 2.0 < 0.03


def test_poiseuille_en_canal_curvo():
    """Stokes estacionario en un sector de corona: parabola y balance de masa.

    Paso fraccionado incremental: momento viscoso implicito con la presion
    vieja, proyeccion de los flujos de cara, y la velocidad de celda corregida
    con el gradiente Green-Gauss. Sin conveccion -- lo que se valida aqui es el
    acoplamiento presion-velocidad, no el transporte, que ya lo hizo F3.
    """
    met = anillo()
    ny, nx = met.J.shape
    nu, U, dt = 0.01, 1.0, 2.0
    cero_xi = np.zeros_like(met.a_xi)
    cero_eta = np.zeros_like(met.a_eta)

    # entrada en theta = 0: la tangente es +y, asi que (u, v) = (0, U)
    bc_u = dict(oeste=np.zeros(ny), este=None, sur=0.0, norte=0.0)
    bc_v = dict(oeste=U * np.ones(ny), este=None, sur=0.0, norte=0.0)
    bc_p = dict(oeste=None, este=0.0, sur=None, norte=None)
    bc_vel = dict(oeste=(np.zeros(ny), U * np.ones(ny)), este=None,
                  sur=(0.0, 0.0), norte=(0.0, 0.0))
    sis_u = cv.sistema(met, cero_xi, cero_eta, nu=nu, dt=dt, bc=bc_u)
    sis_v = cv.sistema(met, cero_xi, cero_eta, nu=nu, dt=dt, bc=bc_v)
    sis_p = pr.sistema_presion(met, bc_p)

    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    for _ in range(90):
        gx, gy = op.gradiente(met, p, **bc_p)
        ue, _ = cv.avanzar(met, u, cero_xi, cero_eta, dt=dt, nu=nu, bc=bc_u,
                           fuente=-gx, correcciones=1, sis=sis_u)
        ve, _ = cv.avanzar(met, v, cero_xi, cero_eta, dt=dt, nu=nu, bc=bc_v,
                           fuente=-gy, correcciones=1, sis=sis_v)
        mx, me = pr.flujos_de_velocidad(met, ue, ve, bc_vel)
        mx, me, phi, _ = pr.proyectar(met, mx, me, dt=dt, bc=bc_p,
                                      correcciones=2, sis=sis_p)
        u, v = pr.corregir_velocidad(met, ue, ve, phi, dt, bc_p)
        p = p + phi

    assert abs(float(mx[:, -1].sum()) - float(mx[:, 0].sum())) < 0.01 * U

    th = np.arctan2(met.yc, met.xc)
    ut = -u * np.sin(th) + v * np.cos(th)
    s = np.hypot(met.xc, met.yc)[:, -1] - 4.0
    media = float(mx[:, -1].sum())
    assert np.abs(ut[:, -1] - 6.0 * media * s * (1.0 - s)).max() / ut[:, -1].max() < 0.05

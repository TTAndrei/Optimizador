"""El camino GPU da lo mismo que el de CPU, pieza a pieza.

Los modulos de `curvo/` trabajan con el `xp` del array que reciben, asi que la
misma funcion corre en numpy y en cupy. Lo que estos tests fijan es que eso no
es una promesa sino una igualdad medida: mismas metricas, misma matriz, mismo
engrosado, mismo suavizado, mismo ciclo y mismo resultado.

Merece la pena comprobarlo pieza a pieza y no solo al final: el suavizador y la
aglomeracion en GPU **no son las mismas operaciones de array** que en CPU, son
kernels escritos aparte (`_FUENTE` en `multigrid.py`). Un kernel con un signo
cambiado converge igual de bien a la solucion equivocada.

float32 va aparte. El solver corre en float32 porque la 3070 Ti hace fp64 a 1/64
de fp32; lo que se comprueba es que la respuesta es la misma dentro del epsilon
de la precision, no que sea identica.
"""

import os

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from curvo import malla as M                                    # noqa: E402
from curvo import conveccion as cv                              # noqa: E402
from curvo import operadores as op                              # noqa: E402
from curvo import proyeccion as pr                              # noqa: E402
from curvo.metrica import Metrica                               # noqa: E402
from curvo.multigrid import ciclo_v, jerarquia, resolver_pcg    # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    cp.cuda.runtime.getDeviceCount()
except cp.cuda.runtime.CUDARuntimeError:                        # pragma: no cover
    pytest.skip("sin GPU", allow_module_level=True)

BC_P = dict(oeste=0.0, este=0.0, sur=None, norte=None)
BC_PHI = dict(oeste=None, este=None, sur=1.0, norte=1.0)
NU = cv.AIRE["nu"]


def maxabs(a):
    return float(cp.abs(cp.asarray(a)).max())


@pytest.fixture(scope="module")
def par():
    """La misma malla C en CPU (float64) y en GPU (float64)."""
    px, py = M.leer_dat(os.path.join(RAIZ, "profiles", "NACA_0012_sharp"))
    X, Y, info = M.generar_c(px, py)
    cpu = Metrica(X, Y), cv.corte_de_estela(info, X.shape[1] - 1)
    gpu = (Metrica(cp.asarray(X), cp.asarray(Y)),
           cv.corte_de_estela(info, X.shape[1] - 1, xp=cp))
    return cpu, gpu


def sistemas(par, tipo="presion"):
    (mc, cc), (mg, cg) = par
    if tipo == "presion":
        return (pr.sistema_presion(mc, BC_P, cc)[0],
                pr.sistema_presion(mg, BC_P, cg)[0])
    corriente = lambda x, y: (x * 0 + 1.0, x * 0)               # noqa: E731
    a = cv.sistema(mc, *cv.flujos_de_campo(mc, corriente), nu=NU, dt=0.01,
                   bc=BC_PHI, corte=cc)[0]
    b = cv.sistema(mg, *cv.flujos_de_campo(mg, corriente), nu=NU, dt=0.01,
                   bc=BC_PHI, corte=cg)[0]
    return a, b


# ---------------------------------------------------------------------------
# Piezas
# ---------------------------------------------------------------------------
def test_las_metricas_coinciden(par):
    (mc, _), (mg, _) = par
    for nombre in ("Sx_xi", "Sy_xi", "Sx_eta", "Sy_eta", "J", "a_xi", "b_xi",
                   "a_eta", "b_eta"):
        a = getattr(mc, nombre)
        assert maxabs(getattr(mg, nombre) - cp.asarray(a)) / maxabs(a) < 1e-14


@pytest.mark.parametrize("tipo", ["presion", "conveccion"])
def test_la_matriz_coincide(par, tipo):
    a, b = sistemas(par, tipo)
    for nombre in ("aP", "aW", "aE", "aS", "aN", "aC"):
        x = getattr(a, nombre)
        assert maxabs(getattr(b, nombre) - cp.asarray(x)) / maxabs(x) < 1e-13


@pytest.mark.parametrize("tipo", ["presion", "conveccion"])
def test_el_engrosado_coincide_en_todos_los_niveles(par, tipo):
    """La aglomeracion en GPU es un kernel de recoleccion, no `bincount`.

    Cada hilo suma las <=4 celdas finas de su bloque aprovechando que los mapas
    son tramos contiguos. Es codigo distinto del de CPU, asi que se compara en
    los siete niveles, `aC` incluido: el corte es lo que mas facil se desalinea.
    """
    a, b = sistemas(par, tipo)
    nc, ng = jerarquia(a), jerarquia(b)
    assert len(nc) == len(ng) >= 5
    for (fc, _, _), (fg, _, _) in zip(nc, ng):
        assert fc.aP.shape == fg.aP.shape
        for nombre in ("aP", "aW", "aE", "aS", "aN", "aC"):
            x = getattr(fc, nombre)
            if x is None:
                assert getattr(fg, nombre) is None
                continue
            # `aS` y `aN` son todo ceros en el nivel mas grueso (ny = 1)
            assert maxabs(getattr(fg, nombre) - cp.asarray(x)) < 1e-11 * max(maxabs(x), 1.0)


@pytest.mark.parametrize("tipo", ["presion", "conveccion"])
def test_el_suavizador_coincide(par, tipo):
    """Incluye las lineas eta que cruzan el corte, que son un kernel aparte."""
    a, b = sistemas(par, tipo)
    rng = np.random.default_rng(0)
    rhs = rng.standard_normal(a.aP.shape)
    a.b, b.b = rhs.copy(), cp.asarray(rhs)
    pc = np.zeros_like(a.aP)
    pg = cp.zeros_like(b.aP)
    for _ in range(3):
        a.suavizar(pc, 1)
        b.suavizar(pg, 1)
        assert maxabs(pg - cp.asarray(pc)) / maxabs(pc) < 1e-12


def test_el_ciclo_v_coincide(par):
    a, b = sistemas(par, "presion")
    rng = np.random.default_rng(1)
    rhs = rng.standard_normal(a.aP.shape)
    a.b, b.b = rhs.copy(), cp.asarray(rhs)
    nc, ng = jerarquia(a), jerarquia(b)
    pc = np.zeros_like(a.aP)
    pg = cp.zeros_like(b.aP)
    for _ in range(3):
        ciclo_v(nc, pc, w=2.0)
        ciclo_v(ng, pg, w=2.0)
        assert maxabs(pg - cp.asarray(pc)) / maxabs(pc) < 1e-11


def test_el_pcg_converge_igual(par):
    a, b = sistemas(par, "presion")
    rng = np.random.default_rng(2)
    rhs = rng.standard_normal(a.aP.shape)
    a.b, b.b = rhs.copy(), cp.asarray(rhs)
    xc, ic = resolver_pcg(a, None, tol=1e-10, ciclos=80, niveles=jerarquia(a))
    xg, ig = resolver_pcg(b, None, tol=1e-10, ciclos=80, niveles=jerarquia(b))
    assert maxabs(xg - cp.asarray(xc)) / maxabs(xc) < 1e-8
    assert ig["factor"] < 0.3


# ---------------------------------------------------------------------------
# Extremo a extremo
# ---------------------------------------------------------------------------
def test_la_proyeccion_en_gpu_da_lo_mismo(par):
    (mc, cc), (mg, cg) = par
    corriente = lambda x, y: (x * 0 + 1.0, x * 0)               # noqa: E731
    fuera = []
    for met, corte in ((mc, cc), (mg, cg)):
        xp = met.xp
        m_xi, m_eta = cv.flujos_de_campo(met, corriente)
        m_eta = m_eta.copy()
        m_eta[0] = xp.where(corte, m_eta[0], 0.0)
        fuera.append(pr.proyectar(met, m_xi, m_eta, bc=BC_P, corte=corte,
                                  correcciones=4))
    (_, _, pc, _), (mxg, meg, pg, _) = fuera
    assert maxabs(pg - cp.asarray(pc)) / maxabs(pc) < 1e-9
    assert maxabs(pg - pg[:, ::-1]) / maxabs(pg) < 1e-12        # Cl = 0 exacto
    assert maxabs(meg[0][cg] + meg[0][::-1][cg]) == 0.0         # el corte cierra


@pytest.mark.parametrize("u0", [1.0, 10.0])
def test_la_corriente_libre_se_preserva_en_gpu(par, u0):
    (_, _), (met, corte) = par
    m_xi, m_eta = cv.flujos_de_campo(
        met, lambda x, y: (u0 * cp.ones_like(x), cp.zeros_like(x)))
    phi, _ = cv.avanzar(met, cp.ones_like(met.J), m_xi, m_eta, dt=0.01,
                        nu=NU, bc=BC_PHI, corte=corte)
    assert maxabs(phi - 1.0) < 1e-13


def test_las_fronteras_admiten_arrays_y_parejas(par):
    """Una frontera puede ser un escalar, un array por cara o la pareja `(u, v)`.

    La pareja es lo que usa `flujos_de_velocidad`, y en GPU hay que entrar en la
    tupla en vez de convertirla entera: `cupy.asarray` de una tupla de arrays de
    cupy revienta. Salio al montar la placa plana, donde la pared es movil aguas
    arriba del borde de ataque y por tanto la frontera sur es un array.
    """
    (_, _), (met, corte) = par
    perfil = cp.linspace(0.0, 1.0, met.nx)
    m_xi, m_eta = pr.flujos_de_velocidad(
        met, cp.ones_like(met.J), cp.zeros_like(met.J),
        dict(oeste=(1.0, 0.0), este=None, sur=(perfil, 0.0),
             norte=(1.0, 0.0)), corte)
    assert cp.isfinite(m_xi).all() and cp.isfinite(m_eta).all()
    phi, _ = cv.avanzar(met, cp.ones_like(met.J), m_xi, m_eta, dt=0.01, nu=NU,
                        bc=dict(oeste=None, este=None, sur=perfil, norte=1.0),
                        corte=corte)
    assert cp.isfinite(phi).all()


def test_un_termino_independiente_nulo_no_da_nan(par):
    """Un campo ya solenoidal es el arranque del solver, y tiene `b = 0`.

    Con `b = 0` el residuo inicial es cero, y el PCG hacia `beta = rz/rz` = 0/0.
    En CPU no salia porque comprueba la convergencia en cada iteracion y sale
    antes de llegar ahi; en GPU la comprueba cada dos para no sincronizar. Lo
    destapo la placa plana de Blasius, donde la corriente uniforme de arranque ya
    cumple el balance.
    """
    (_, _), (met, corte) = par
    t = cp.linspace(0.0, 1.0, met.nx + 1)
    psi = cp.sin(cp.pi * t)[None, :] * cp.sin(cp.pi * cp.linspace(
        0.0, 1.0, met.ny + 1))[:, None]
    m_xi, m_eta = cv.flujos_de_corriente(psi)
    mx, me, p, info = pr.proyectar(met, m_xi, m_eta, bc=BC_P, corte=corte,
                                   correcciones=2)
    assert cp.isfinite(p).all()
    assert cp.isfinite(mx).all() and cp.isfinite(me).all()


def test_float32_da_la_misma_respuesta():
    """La precision con la que corre el solver. Lo que se pide es el epsilon de f32.

    La 3070 Ti hace fp64 a 1/64 de fp32, asi que float64 en GPU es mas lento que
    la CPU y float32 es la unica opcion util. F1 ya midio que la conservacion
    aguanta (5.1e-8); aqui se comprueba que tambien aguantan la matriz, el
    multigrid y la proyeccion completa.
    """
    px, py = M.leer_dat(os.path.join(RAIZ, "profiles", "NACA_0012_sharp"))
    X, Y, info = M.generar_c(px, py)
    salida = {}
    for tipo in (np.float64, np.float32):
        met = Metrica(cp.asarray(X, dtype=tipo), cp.asarray(Y, dtype=tipo))
        corte = cv.corte_de_estela(info, met.nx, xp=cp)
        m_xi, m_eta = cv.flujos_de_campo(
            met, lambda x, y: (cp.ones_like(x), cp.zeros_like(x)))
        m_eta = m_eta.copy()
        m_eta[0] = cp.where(corte, m_eta[0], 0.0)
        mx, me, p, _ = pr.proyectar(met, m_xi, m_eta, bc=BC_P, corte=corte,
                                    correcciones=4)
        d = op.divergencia(mx, me)
        esc = (cp.abs(mx[:, 1:]) + cp.abs(mx[:, :-1])
               + cp.abs(me[1:]) + cp.abs(me[:-1]))
        salida[tipo] = p, float(cp.abs(d / cp.maximum(esc, 1e-30)).max())

    p64, div64 = salida[np.float64]
    p32, div32 = salida[np.float32]
    assert div64 < 1e-9
    assert div32 < 1e-3
    assert maxabs(p32.astype(cp.float64) - p64) / maxabs(p64) < 1e-4

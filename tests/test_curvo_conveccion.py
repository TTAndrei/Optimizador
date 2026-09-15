"""Tests de conveccion-difusion implicita conservativa y del multigrid (F3).

Cuatro familias, en este orden de importancia:

- **Conservacion**: es la propiedad por la que se cambia de esquema. Sobre un
  dominio cerrado la integral de la cantidad transportada no se mueve de la
  precision de maquina. El semi-Lagrangiano bilineal, en el mismo caso, pierde
  el 5 % en 50 pasos; esta medido aqui al lado.
- **Condiciones de frontera**: Dirichlet, gradiente nulo y el corte de estela,
  cada una con un caso donde la respuesta exacta se conoce.
- **Orden de convergencia**: solucion manufacturada sobre cuadrado mapeado, en
  regimen difusivo y en regimen convectivo con la viscosidad del aire.
- **Acotacion y difusion numerica**: el limitador TVD no inventa extremos, y el
  frente se difunde menos que con el esquema actual.

Las constantes del fluido son las del aire (`conveccion.AIRE`), y los casos de
transporte se corren a u = 1 y u = 10 m/s: con cuerda 1 m son Re = 6.8e4 y
6.8e5, y numeros de Peclet de celda de 1e3 a 1e4, que es donde un esquema
centrado se caeria y el upwind limitado tiene que aguantar.
"""

import os

import numpy as np
import pytest

from curvo import malla as M
from curvo import conveccion as cv
from curvo import multigrid as mg
from curvo import operadores as op
from curvo.metrica import Metrica

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NU = cv.AIRE["nu"]


def cuadrado(n, amp=0.0):
    """Cuadrado [0,1]^2, con una distorsion suave si `amp > 0`."""
    t = np.linspace(0.0, 1.0, n + 1)
    E, C = np.meshgrid(t, t, indexing="xy")
    s = np.sin(np.pi * E) * np.sin(np.pi * C)
    return Metrica(E + amp * s, C + amp * s)


def semilagrangiano(met, phi, u, v, dt, pasos):
    """Referencia: backtrace + interpolacion bilineal, el esquema actual.

    Solo vale sobre malla cartesiana uniforme, que es lo unico que hace falta
    para medir contra que se compara.
    """
    xc, yc = met.xc, met.yc
    n = met.nx
    h = float(xc[0, 1] - xc[0, 0])
    x0, y0 = float(xc[0, 0]), float(yc[0, 0])
    for _ in range(pasos):
        fi = np.clip((xc - u * dt - x0) / h, 0.0, n - 1.001)
        fj = np.clip((yc - v * dt - y0) / h, 0.0, n - 1.001)
        i, j = fi.astype(int), fj.astype(int)
        a, b = fi - i, fj - j
        phi = ((1 - a) * (1 - b) * phi[j, i] + a * (1 - b) * phi[j, i + 1]
               + (1 - a) * b * phi[j + 1, i] + a * b * phi[j + 1, i + 1])
    return phi


def integrar(met, phi, m_xi, m_eta, dt, pasos, nu=0.0, bc=None, bdf2=False,
             correcciones=3):
    """Avanza `pasos` pasos. Con `bdf2`, el primero es Euler atras (arranque)."""
    s_e = cv.sistema(met, m_xi, m_eta, nu=nu, dt=dt, bc=bc)
    s_2 = cv.sistema(met, m_xi, m_eta, nu=nu, dt=dt, bc=bc, bdf2=True) if bdf2 else None
    ant = phi
    for k in range(pasos):
        usa_bdf2 = bdf2 and k > 0
        nuevo, _ = cv.avanzar(met, phi, m_xi, m_eta, dt=dt, nu=nu, bc=bc,
                              correcciones=correcciones,
                              sis=s_2 if usa_bdf2 else s_e,
                              phi_ant=ant if usa_bdf2 else None)
        ant, phi = phi, nuevo
    return phi


# ---------------------------------------------------------------------------
# Conservacion
# ---------------------------------------------------------------------------
def test_dominio_cerrado_conserva_la_masa():
    """Campo solenoidal tangente al contorno: la integral no se mueve.

    Los flujos salen de una funcion de corriente en los vertices, asi que su
    divergencia discreta es cero **exacta** y no al orden del esquema: si la
    masa se moviera, seria del esquema y no del campo.
    """
    met = cuadrado(48, amp=0.10)
    psi = np.sin(np.pi * met.X) * np.sin(np.pi * met.Y)
    m_xi, m_eta = cv.flujos_de_corriente(psi)
    assert np.abs(op.divergencia(m_xi, m_eta)).max() < 1e-15

    phi = np.exp(-60.0 * ((met.xc - 0.3) ** 2 + (met.yc - 0.5) ** 2))
    m0 = cv.masa(met, phi)
    bc = dict(oeste=None, este=None, sur=None, norte=None)
    sis = cv.sistema(met, m_xi, m_eta, nu=0.0, dt=0.01, bc=bc)
    for _ in range(20):
        phi, _ = cv.avanzar(met, phi, m_xi, m_eta, dt=0.01, nu=0.0, bc=bc, sis=sis)
    assert abs(cv.masa(met, phi) - m0) / m0 < 1e-12


def test_el_semilagrangiano_no_conserva_y_este_si():
    """El motivo del cambio de esquema, medido en el mismo caso.

    50 pasos de un pulso en un campo de rotacion: el FV conservativo se queda en
    la precision de maquina y el semi-Lagrangiano bilineal pierde ~5 % de la
    cantidad transportada. No es un ajuste de parametros: el backtrace mas la
    interpolacion no forman un balance de flujos por cara.
    """
    met = cuadrado(96)
    psi = np.sin(np.pi * met.X) * np.sin(np.pi * met.Y)
    m_xi, m_eta = cv.flujos_de_corriente(psi)
    u = np.pi * np.sin(np.pi * met.xc) * np.cos(np.pi * met.yc)
    v = -np.pi * np.cos(np.pi * met.xc) * np.sin(np.pi * met.yc)

    ini = np.exp(-60.0 * ((met.xc - 0.3) ** 2 + (met.yc - 0.5) ** 2))
    m0 = cv.masa(met, ini)
    dt, pasos = 0.004, 50

    bc = dict(oeste=None, este=None, sur=None, norte=None)
    sis = cv.sistema(met, m_xi, m_eta, nu=0.0, dt=dt, bc=bc)
    phi = ini.copy()
    for _ in range(pasos):
        phi, _ = cv.avanzar(met, phi, m_xi, m_eta, dt=dt, nu=0.0, bc=bc, sis=sis)

    d_fv = abs(cv.masa(met, phi) - m0) / m0
    sl = ini.copy()
    xc, yc, h = met.xc, met.yc, 1.0 / met.nx
    for _ in range(pasos):
        fi = np.clip((xc - u * dt - xc[0, 0]) / h, 0.0, met.nx - 1.001)
        fj = np.clip((yc - v * dt - yc[0, 0]) / h, 0.0, met.ny - 1.001)
        i, j = fi.astype(int), fj.astype(int)
        a, b = fi - i, fj - j
        sl = ((1 - a) * (1 - b) * sl[j, i] + a * (1 - b) * sl[j, i + 1]
              + (1 - a) * b * sl[j + 1, i] + a * b * sl[j + 1, i + 1])
    d_sl = abs(cv.masa(met, sl) - m0) / m0

    assert d_fv < 1e-12
    assert d_sl > 1e-2                       # medido: 5.3e-2
    assert d_sl / d_fv > 1e9


# ---------------------------------------------------------------------------
# Condiciones de frontera
# ---------------------------------------------------------------------------
def test_dirichlet_da_el_perfil_lineal_exacto():
    """Difusion pura entre dos paredes a distinto valor: phi = y."""
    met = cuadrado(16)
    cero = np.zeros((met.ny, met.nx + 1)), np.zeros((met.ny + 1, met.nx))
    bc = dict(oeste=None, este=None, sur=0.0, norte=1.0)
    phi, _ = cv.avanzar(met, np.zeros((met.ny, met.nx)), *cero, dt=None, nu=1.0, bc=bc)
    assert np.abs(phi - met.yc).max() < 1e-11


def test_gradiente_nulo_no_deja_pasar_flujo():
    """Un solo Dirichlet y el resto gradiente nulo: el campo se hace uniforme.

    Hacen falta varias iteraciones externas aunque la respuesta sea trivial: en
    malla no ortogonal el termino cruzado de la frontera Dirichlet no es cero
    mientras el campo no lo sea, y va en la correccion diferida. Medido con 0,
    2 y 6 iteraciones: 7.8e-2, 1.8e-3, 2.3e-6.
    """
    met = cuadrado(16, amp=0.10)
    cero = np.zeros((met.ny, met.nx + 1)), np.zeros((met.ny + 1, met.nx))
    bc = dict(oeste=None, este=None, sur=None, norte=2.5)
    phi, _ = cv.avanzar(met, np.zeros((met.ny, met.nx)), *cero, dt=None, nu=1.0,
                        bc=bc, correcciones=12, ciclos=200)
    assert np.abs(phi - 2.5).max() < 1e-9


@pytest.mark.parametrize("u0", [1.0, 10.0])
def test_corriente_uniforme_no_se_altera(u0):
    """Entrada Dirichlet y salida libre: un campo uniforme se queda uniforme.

    Es el equivalente convectivo de la preservacion de corriente libre: solo se
    cumple si los flujos de cara tienen divergencia discreta nula y si el
    tratamiento de las cuatro fronteras es consistente entre si.
    """
    met = cuadrado(32, amp=0.10)
    m_xi, m_eta = cv.flujos_de_campo(
        met, lambda x, y: (u0 * np.ones_like(x), 0.3 * u0 * np.ones_like(x)))
    bc = dict(oeste=1.0, este=None, sur=1.0, norte=None)
    phi, _ = cv.avanzar(met, np.ones((met.ny, met.nx)), m_xi, m_eta,
                        dt=0.01, nu=NU, bc=bc)
    assert np.abs(phi - 1.0).max() < 1e-13


# ---------------------------------------------------------------------------
# Corte de estela
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def malla_c():
    px, py = M.leer_dat(os.path.join(RAIZ, "profiles", "NACA_0012_sharp"))
    X, Y, info = M.generar_c(px, py)
    met = Metrica(X, Y)
    return met, info, cv.corte_de_estela(info, met.nx)


def test_el_corte_ve_la_misma_cara_por_los_dos_lados(malla_c):
    """`m_eta[0,i] = -m_eta[0,nx-1-i]`: misma cara fisica, normal opuesta.

    Sale exacto, no aproximado, porque sobre el corte `Y = 0` y el vector de
    area de la cara se reduce a `(0, X[i+1]-X[i])`, que cambia de signo al
    recorrer la rama de arriba. Si no cerrara, el flujo que sale de la estela
    por abajo no seria el que entra por arriba y no habria conservacion.
    """
    met, _, corte = malla_c
    for u0 in (1.0, 10.0):
        _, m_eta = cv.flujos_de_campo(
            met, lambda x, y: (u0 * np.ones_like(x), np.zeros_like(x)))
        assert np.abs(m_eta[0][corte] + m_eta[0][::-1][corte]).max() == 0.0


@pytest.mark.parametrize("u0", [1.0, 10.0])
def test_malla_c_con_corte_preserva_la_corriente(malla_c, u0):
    """Sobre la malla C entera, con el corte activo y viscosidad del aire."""
    met, _, corte = malla_c
    m_xi, m_eta = cv.flujos_de_campo(
        met, lambda x, y: (u0 * np.ones_like(x), np.zeros_like(x)))
    bc = dict(oeste=None, este=None, sur=1.0, norte=1.0)
    phi, info = cv.avanzar(met, np.ones((met.ny, met.nx)), m_xi, m_eta,
                           dt=0.01, nu=NU, bc=bc, corte=corte)
    assert np.abs(phi - 1.0).max() < 1e-13


def test_el_espejo_del_corte_sobrevive_al_engrosado(malla_c):
    """En todos los niveles del multigrid el corte sigue emparejando `k <-> nc-1-k`.

    Es el riesgo 5 del plan. Se sostiene porque el engrosado en xi empareja desde
    los dos extremos hacia dentro; con el emparejamiento ingenuo desde i=0 y un
    numero impar de celdas el espejo se desalinea una celda por nivel.
    """
    met, _, corte = malla_c
    m_xi, m_eta = cv.flujos_de_campo(
        met, lambda x, y: (np.ones_like(x), np.zeros_like(x)))
    sis, _ = cv.sistema(met, m_xi, m_eta, nu=NU, dt=0.01,
                        bc=dict(sur=1.0, norte=1.0), corte=corte)
    for nivel, (s, _, _) in enumerate(mg.jerarquia(sis)):
        assert s.aC is not None, nivel
        np.testing.assert_allclose(s.aC, s.aC[::-1], rtol=1e-12, atol=0.0,
                                   err_msg=f"nivel {nivel}")


def test_el_engrosado_conserva_la_suma_de_las_filas(malla_c):
    """La ecuacion gruesa es la suma de las finas: la suma de cada fila tambien.

    Es lo que garantiza que la correccion gruesa no rompa el balance global, y
    por tanto que la conservacion aguante aunque el ciclo se corte pronto.
    """
    met, _, corte = malla_c
    m_xi, m_eta = cv.flujos_de_campo(
        met, lambda x, y: (np.ones_like(x), np.zeros_like(x)))
    sis, _ = cv.sistema(met, m_xi, m_eta, nu=NU, dt=0.01,
                        bc=dict(sur=1.0, norte=1.0), corte=corte)
    niveles = mg.jerarquia(sis)
    for k in range(len(niveles) - 1):
        f, _, _ = niveles[k]
        c, kj, ki = niveles[k + 1]
        suma_f = f.aP - f.aW - f.aE - f.aS - f.aN
        suma_f[0] -= f.aC if f.aC is not None else 0.0
        suma_c = c.aP - c.aW - c.aE - c.aS - c.aN
        suma_c[0] -= c.aC if c.aC is not None else 0.0
        idx = kj[:, None] * c.nx + ki[None, :]
        esperada = mg._sumar(idx, suma_f, (c.ny, c.nx))
        np.testing.assert_allclose(suma_c, esperada, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Orden de convergencia
# ---------------------------------------------------------------------------
_A, _B = 1.1, 0.7


def _manufacturada(n, nu):
    """Estacionario con fuente conocida sobre cuadrado mapeado. Devuelve el RMS."""
    met = cuadrado(n, amp=0.10)
    exacta = lambda x, y: np.exp(_A * x + _B * y)
    psi = met.X * met.Y + 0.2 * np.sin(np.pi * met.X) * np.sin(np.pi * met.Y)
    m_xi, m_eta = cv.flujos_de_corriente(psi)

    xf = 0.5 * (met.X[:-1, :] + met.X[1:, :])
    yf = 0.5 * (met.Y[:-1, :] + met.Y[1:, :])
    xg = 0.5 * (met.X[:, :-1] + met.X[:, 1:])
    yg = 0.5 * (met.Y[:, :-1] + met.Y[:, 1:])
    bc = dict(oeste=exacta(xf[:, 0], yf[:, 0]), este=exacta(xf[:, -1], yf[:, -1]),
              sur=exacta(xg[0], yg[0]), norte=exacta(xg[-1], yg[-1]))

    u = met.xc + 0.2 * np.pi * np.sin(np.pi * met.xc) * np.cos(np.pi * met.yc)
    v = -met.yc - 0.2 * np.pi * np.cos(np.pi * met.xc) * np.sin(np.pi * met.yc)
    ex = exacta(met.xc, met.yc)
    fuente = (u * _A + v * _B) * ex - nu * (_A * _A + _B * _B) * ex

    phi, _ = cv.avanzar(met, np.ones_like(ex), m_xi, m_eta, dt=None, nu=nu, bc=bc,
                        fuente=fuente, correcciones=8, tol=1e-13, ciclos=200)
    return np.sqrt((((phi - ex) ** 2) * met.J).sum() / met.J.sum())


@pytest.mark.parametrize("nu,etiqueta", [(1.0, "difusivo"), (NU, "convectivo")])
def test_orden_de_convergencia(nu, etiqueta):
    """Segundo orden en los dos regimenes. Medido: 2.00 difusivo, 2.02 convectivo.

    Con la viscosidad del aire el Peclet de celda pasa de 1e3, asi que el
    resultado depende por entero del upwind limitado y de la correccion
    diferida: con upwind de 1.er orden puro sale orden 1.
    """
    e = [_manufacturada(n, nu) for n in (24, 48, 96)]
    ordenes = [np.log2(e[k] / e[k + 1]) for k in range(len(e) - 1)]
    assert min(ordenes) > 1.8, (etiqueta, e, ordenes)


def test_sin_correccion_diferida_el_orden_baja_a_uno():
    """Contraste: el upwind de 1.er orden solo, con todo lo demas igual."""
    met24, met48 = cuadrado(24, 0.10), cuadrado(48, 0.10)
    errores = []
    for met in (met24, met48):
        exacta = lambda x, y: np.exp(_A * x + _B * y)
        psi = met.X * met.Y + 0.2 * np.sin(np.pi * met.X) * np.sin(np.pi * met.Y)
        m_xi, m_eta = cv.flujos_de_corriente(psi)
        xf = 0.5 * (met.X[:-1, :] + met.X[1:, :])
        yf = 0.5 * (met.Y[:-1, :] + met.Y[1:, :])
        xg = 0.5 * (met.X[:, :-1] + met.X[:, 1:])
        yg = 0.5 * (met.Y[:, :-1] + met.Y[:, 1:])
        bc = dict(oeste=exacta(xf[:, 0], yf[:, 0]), este=exacta(xf[:, -1], yf[:, -1]),
                  sur=exacta(xg[0], yg[0]), norte=exacta(xg[-1], yg[-1]))
        u = met.xc + 0.2 * np.pi * np.sin(np.pi * met.xc) * np.cos(np.pi * met.yc)
        v = -met.yc - 0.2 * np.pi * np.cos(np.pi * met.xc) * np.sin(np.pi * met.yc)
        ex = exacta(met.xc, met.yc)
        fuente = (u * _A + v * _B) * ex - NU * (_A * _A + _B * _B) * ex
        phi, _ = cv.avanzar(met, np.ones_like(ex), m_xi, m_eta, dt=None, nu=NU,
                            bc=bc, fuente=fuente, correcciones=0, ciclos=200)
        errores.append(np.sqrt((((phi - ex) ** 2) * met.J).sum() / met.J.sum()))
    assert np.log2(errores[0] / errores[1]) < 1.3


# ---------------------------------------------------------------------------
# Acotacion y difusion numerica
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("u0", [1.0, 10.0])
def test_el_limitador_no_inventa_extremos(u0):
    """Escalon advectado en diagonal: ni sobreimpulso ni valores negativos.

    Oblicuo a la malla a proposito: alineado con la malla cualquier esquema
    unidimensional se porta bien y el test no mide nada.

    La correccion diferida va en el termino independiente, asi que el esquema no
    es TVD **antes** de converger la iteracion externa: el sobreimpulso medido
    baja de -4.9e-4 con una iteracion a -3.2e-5 con dos y -1.5e-7 con cuatro.
    No es el limitador -- sale igual quitando la pendiente de borde.
    """
    met = cuadrado(64)
    c = u0 / np.sqrt(2.0)
    m_xi, m_eta = cv.flujos_de_campo(
        met, lambda x, y: (c * np.ones_like(x), c * np.ones_like(x)))
    s = (met.xc + met.yc) / np.sqrt(2.0)
    phi = (s < 0.25).astype(float)

    dt = 0.5 * (1.0 / met.nx) / u0
    bc = dict(oeste=1.0, sur=1.0, este=None, norte=None)
    phi = integrar(met, phi, m_xi, m_eta, dt, 30, bc=bc, correcciones=4)
    assert phi.min() > -1e-6
    assert phi.max() < 1.0 + 1e-12


def _pulso_advectado(n, cfl, bdf2, T=0.3, sigma=0.05):
    """Gaussiana 2D compacta transportada en diagonal. Devuelve (err_FV, err_SL, pico)."""
    met = cuadrado(n)
    u = v = 1.0 / np.sqrt(2.0)
    m_xi, m_eta = cv.flujos_de_campo(
        met, lambda x, y: (u * np.ones_like(x), v * np.ones_like(x)))
    g = lambda cx, cy: np.exp(-(((met.xc - cx) ** 2 + (met.yc - cy) ** 2)
                                / (2.0 * sigma ** 2)))
    ini, exacta = g(0.25, 0.25), g(0.25 + u * T, 0.25 + v * T)

    h = 1.0 / n
    pasos = int(round(T / (cfl * h)))
    dt = T / pasos
    bc = dict(oeste=0.0, sur=0.0, este=None, norte=None)
    phi = integrar(met, ini.copy(), m_xi, m_eta, dt, pasos, bc=bc, bdf2=bdf2)
    sl = semilagrangiano(met, ini.copy(), u, v, dt, pasos)
    return (np.abs(phi - exacta).mean(), np.abs(sl - exacta).mean(), phi.max())


def test_difunde_menos_que_el_semilagrangiano_bilineal():
    """Pulso suave advectado en diagonal, misma malla y mismo dt.

    Medido a CFL 0.5 con BDF2, error L1 frente al del semi-Lagrangiano
    bilineal: 0.31 (n=48), 0.15 (n=96), 0.074 (n=192). La ventaja crece al
    refinar porque el semi-Lagrangiano es de 1.er orden y no baja de ahi. El
    pico del pulso, que deberia valer 1, queda en 0.96 frente al 0.63 del
    semi-Lagrangiano a n=96.

    **Con Euler atras la ventaja se queda en 0.72** y no mejora al refinar: el
    error temporal de 1.er orden domina. Por eso el esquema tiene BDF2.
    """
    e_fv, e_sl, pico = _pulso_advectado(96, cfl=0.5, bdf2=True)
    assert e_fv < 0.25 * e_sl
    assert pico > 0.85


def test_bdf2_gana_orden_y_pierde_monotonia():
    """El precio de BDF2, medido: sobrepasa en un arranque en escalon.

    Ningun metodo lineal multipaso de orden mayor que uno es incondicionalmente
    monotono (Bolley-Crouzeix), y el coeficiente negativo de `phi^{n-1}` es lo
    que lo rompe. No se arregla iterando la correccion diferida: con 3 y con 8
    iteraciones sale el mismo pico. Es transitorio -- decae de 1.09 a 1.004 en
    60 pasos y baja a 1.013 con dt cinco veces menor -- pero para la `nu_tilde`
    de Spalart-Allmaras, que tiene que ser positiva, la eleccion es Euler atras.
    """
    met = cuadrado(48)
    m_xi, m_eta = cv.flujos_de_campo(
        met, lambda x, y: (np.ones_like(x), np.zeros_like(x)))
    bc = dict(oeste=1.0, este=None, sur=None, norte=None)
    dt = 1.0 / met.nx                                  # CFL = 1, arranque brusco
    cero = np.zeros((met.ny, met.nx))
    euler = integrar(met, cero.copy(), m_xi, m_eta, dt, 12, bc=bc, correcciones=4)
    bdf2 = integrar(met, cero.copy(), m_xi, m_eta, dt, 12, bc=bc, correcciones=4,
                    bdf2=True)
    assert euler.max() < 1.0 + 1e-12
    assert bdf2.max() > 1.0 + 1e-6


def test_bdf2_recupera_el_orden_que_euler_atras_se_come():
    """Con CFL fijo, `dt ~ h`: Euler atras deja el transporte en 1.er orden.

    Medido de 48 a 96 celdas: 0.80 con Euler atras, 1.81 con BDF2. Llevandolo a
    192 sale 0.85 y 1.84; no se deja en el test porque cuesta 90 s el solo.
    """
    ordenes = {}
    for bdf2 in (False, True):
        e = [_pulso_advectado(n, cfl=0.5, bdf2=bdf2)[0] for n in (48, 96)]
        ordenes[bdf2] = np.log2(e[0] / e[1])
    assert ordenes[True] > 1.7
    assert ordenes[False] < 1.2


@pytest.mark.parametrize("u0", [1.0, 10.0])
def test_el_pulso_viaja_a_la_velocidad_del_fluido(u0):
    """El centroide avanza `u*t`: el esquema transporta, no solo difunde."""
    met = cuadrado(96)
    m_xi, m_eta = cv.flujos_de_campo(
        met, lambda x, y: (u0 * np.ones_like(x), np.zeros_like(x)))
    phi = np.exp(-((met.xc - 0.2) / 0.05) ** 2)
    c0 = float((phi * met.xc * met.J).sum() / (phi * met.J).sum())

    dt = 0.5 * (1.0 / met.nx) / u0
    pasos = 40
    bc = dict(oeste=0.0, este=None, sur=None, norte=None)
    sis = cv.sistema(met, m_xi, m_eta, nu=NU, dt=dt, bc=bc)
    for _ in range(pasos):
        phi, _ = cv.avanzar(met, phi, m_xi, m_eta, dt=dt, nu=NU, bc=bc, sis=sis)
    c1 = float((phi * met.xc * met.J).sum() / (phi * met.J).sum())
    assert abs((c1 - c0) - u0 * dt * pasos) < 0.02 * u0 * dt * pasos


# ---------------------------------------------------------------------------
# Multigrid
# ---------------------------------------------------------------------------
def test_el_multigrid_no_se_degrada_al_refinar():
    """Difusion pura: el factor del ciclo V no depende de la malla.

    Sin el escalado del cociente de Rayleigh (ver `multigrid.ciclo_v`) este test
    falla: la aglomeracion sin suavizar da 0.70 / 0.85 / 0.93 para n = 32/64/128.
    """
    factores = []
    for n in (32, 64, 128):
        met = cuadrado(n, amp=0.10)
        cero = np.zeros((met.ny, met.nx + 1)), np.zeros((met.ny + 1, met.nx))
        sis, b = cv.sistema(met, *cero, nu=1.0, dt=None,
                            bc=dict(oeste=0.0, este=0.0, sur=0.0, norte=0.0))
        sis.b = b + met.J
        _, info = mg.resolver(sis, tol=1e-12, ciclos=40)
        factores.append(info["factor"])
    assert max(factores) < 0.65
    assert factores[-1] < 1.15 * factores[0]     # sin deriva con la malla


@pytest.mark.parametrize("u0", [1.0, 10.0])
def test_el_multigrid_aguanta_la_malla_c(malla_c, u0):
    """Malla C real, relacion de aspecto hasta ~500, Peclet de celda ~1e4."""
    met, _, corte = malla_c
    m_xi, m_eta = cv.flujos_de_campo(
        met, lambda x, y: (u0 * np.ones_like(x), np.zeros_like(x)))
    bc = dict(oeste=None, este=None, sur=0.0, norte=1.0)
    sis, b = cv.sistema(met, m_xi, m_eta, nu=NU, dt=0.05, bc=bc, corte=corte)
    sis.b = b + met.J * 1.0 / 0.05
    _, info = mg.resolver(sis, tol=1e-11, ciclos=40)
    assert info["residuos"][-1] <= 1e-11, info["residuos"][-1]
    assert info["ciclos"] <= 15

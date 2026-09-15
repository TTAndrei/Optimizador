"""Tests del generador de malla C body-fitted (`curvo/malla.py`).

Sin GPU. El test que manda es el de conservacion geometrica: si la divergencia
discreta de una corriente uniforme no sale cero en maquina, hay un error de
metrica y todo lo que se construya encima esta mal.
"""

import os

import numpy as np
import pytest

from curvo import malla as M

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERFILES = ["NACA_0012_sharp", "s1014.dat", "E387.dat", "SD7037.dat", "SG6043.dat",
            "AG24", "GM15"]


def _perfil(nombre):
    return M.leer_dat(os.path.join(RAIZ, "profiles", nombre))


@pytest.fixture(scope="module")
def malla_naca():
    px, py = _perfil("NACA_0012_sharp")
    return M.generar_c(px, py)


# ---------------------------------------------------------------------------
# Conservacion geometrica
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dn_pared", [2.0e-3, 1.0e-3, 1.9e-4])
@pytest.mark.parametrize("alpha", [0.0, 0.1, -0.35])
def test_corriente_uniforme_se_conserva(dn_pared, alpha):
    """div(u_inf) = 0 en maquina, a cualquier resolucion y cualquier angulo.

    Es exacto por construccion porque los vectores de area de cara salen del
    segmento recto entre vertices y telescopan sobre el cuadrilatero cerrado.
    Calcular las metricas diferenciando centros de celda rompe esto: medido,
    |div| pasa de 1e-16 a 2.4e-2 con celdas de area 2e-4.
    """
    px, py = _perfil("NACA_0012_sharp")
    X, Y, _ = M.generar_c(px, py, dn_pared=dn_pared)
    met = M.metricas(X, Y)
    div, escala = M.divergencia_uniforme(met, np.cos(alpha), np.sin(alpha))
    assert np.max(np.abs(div) / np.maximum(escala, 1e-300)) < 1e-13


def test_areas_de_cara_cierran_el_volumen(malla_naca):
    """Suma de los vectores de area de las 4 caras de cada celda = 0."""
    X, Y, _ = malla_naca
    m = M.metricas(X, Y)
    sx = m["Sx_xi"][:, 1:] - m["Sx_xi"][:, :-1] + m["Sx_eta"][1:, :] - m["Sx_eta"][:-1, :]
    sy = m["Sy_xi"][:, 1:] - m["Sy_xi"][:, :-1] + m["Sy_eta"][1:, :] - m["Sy_eta"][:-1, :]
    escala = np.abs(m["Sx_xi"][:, 1:]) + np.abs(m["Sy_xi"][:, 1:]) + 1e-300
    assert np.max(np.abs(sx) / escala) < 1e-13
    assert np.max(np.abs(sy) / escala) < 1e-13


# ---------------------------------------------------------------------------
# Validez geometrica
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nombre", PERFILES)
def test_perfiles_reales_dan_malla_utilizable(nombre):
    """Los perfiles reales del repo se mallan sin celdas cruzadas."""
    px, py = _perfil(nombre)
    X, Y, _ = M.generar_c(px, py)
    q = M.calidad(X, Y)
    assert q["j_negativos"] == 0, q["fallos"]
    assert q["valida"], q["fallos"]


def test_paso_de_pared_es_el_pedido(malla_naca):
    """Sobre el perfil, y solo sobre el perfil: en la estela `dn` crece a proposito."""
    X, Y, info = malla_naca
    q = M.calidad(X, Y, perfil=info)
    assert q["dn_pared_min"] == pytest.approx(info["dn_pared"], rel=0.25)
    assert q["dn_pared_max"] == pytest.approx(info["dn_pared"], rel=0.25)


@pytest.mark.parametrize("dn_pared", [2.0e-3, 1.9e-4])
def test_el_tope_de_aspecto_no_toca_el_perfil(dn_pared):
    """ASPECTO_MAX baja la relacion de aspecto sin aflojar la pared del cuerpo.

    La celda mala estaba en el corte de estela: `dn` seguia siendo el de pared 18
    cuerdas aguas abajo, donde `ds` ya vale 1.73. Medido a y+=1: 9142 sin tope,
    507 con tope, y el espaciado de pared sobre el perfil sin tocar.
    """
    px, py = _perfil("NACA_0012_sharp")
    kw = dict(dn_pared=dn_pared, n_capas_pared=15, crecimiento_pared=1.0)
    X0, Y0, i0 = M.generar_c(px, py, aspecto_max=np.inf, **kw)
    X1, Y1, i1 = M.generar_c(px, py, **kw)
    q0, q1 = M.calidad(X0, Y0, perfil=i0), M.calidad(X1, Y1, perfil=i1)

    assert q1["aspecto_max"] < 0.6 * q0["aspecto_max"]
    assert q1["dn_pared_max"] == pytest.approx(q0["dn_pared_max"], rel=1e-3)
    assert q1["j_negativos"] == 0
    assert q1["ortogonalidad_pared_min"] == pytest.approx(
        q0["ortogonalidad_pared_min"], abs=0.5)
    assert q1["valida"], q1["fallos"]


def test_los_pasos_por_columna_suman_lo_mismo():
    """Todas las columnas llegan al campo lejano: frente comun, sin cizallamiento."""
    ds = np.linspace(1e-3, 2.0, 200)
    pasos = M.pasos_por_columna(ds, dn_pared=2.0e-3, aspecto_max=100.0)
    total = pasos.sum(axis=0)
    np.testing.assert_allclose(total, total[0], rtol=1e-12)
    assert (pasos[0] >= 2.0e-3 * 0.9).all()
    assert pasos[0, -1] > 9.0 * pasos[0, 0]          # la estela arranca gruesa


@pytest.mark.parametrize("dn_max", [0.5, 0.2, 0.05])
def test_dn_max_acota_el_espaciado_normal(dn_max):
    """Con tope, ninguna capa lo supera, y la malla sigue llegando al campo lejano.

    Sin tope el espaciado normal crece x514 con los valores por defecto (de
    1.78e-3 a 0.914). No es patologico -- el crecimiento es autosemejante -- pero
    no estaba acotado por diseno. Poner el tope cuesta capas: el avance pasa de
    geometrico a lineal en cuanto se alcanza.
    """
    px, py = _perfil("NACA_0012_sharp")
    X, Y, info = M.generar_c(px, py, dn_max=dn_max)
    q = M.calidad(X, Y)
    # 5 % de holgura: el tope va sobre el paso prescrito y la marcha impone
    # ortogonalidad y volumen, no distancia. Medido, se pasa un 1.6 %.
    assert q["dn_max"] <= dn_max * 1.05
    assert q["j_negativos"] == 0
    assert np.abs(Y).max() > 0.9 * M.DISTANCIA_LEJOS      # llega al campo lejano


def test_dn_max_infinito_no_cambia_nada(malla_naca):
    """El defecto es sin tope: la malla tiene que salir identica."""
    X, Y, _ = malla_naca
    px, py = _perfil("NACA_0012_sharp")
    X2, Y2, _ = M.generar_c(px, py, dn_max=np.inf)
    np.testing.assert_array_equal(X, X2)
    np.testing.assert_array_equal(Y, Y2)


def test_plan_de_pasos_respeta_la_capa_de_pared():
    """Los primeros n_capas_pared pasos son constantes; despues crece."""
    pasos = M.plan_de_pasos(dn_pared=2.0e-3, n_capas_pared=15, crecimiento_pared=1.0)
    np.testing.assert_allclose(pasos[:15], 2.0e-3, rtol=0, atol=1e-15)
    assert pasos[16] > pasos[15]
    assert pasos.sum() >= M.DISTANCIA_LEJOS


def test_la_capa_de_pared_mete_mas_celdas_en_la_capa_limite():
    """Mas capas dentro de 0.07c (espesor medido de la capa limite a x/c=0.5).

    Y con el espaciado mas uniforme: sin capa de pared, dn ya ha crecido x4.3 al
    salir de la capa limite.
    """
    px, py = _perfil("NACA_0012_sharp")

    def dentro_de(dn_pared, **kw):
        X, Y, info = M.generar_c(px, py, **kw)
        i0, i1 = info["perfil"]
        l_eta = np.hypot(X[1:, :-1] - X[:-1, :-1], Y[1:, :-1] - Y[:-1, :-1])
        alt = np.cumsum(l_eta[:, i0:i1].mean(axis=1))
        n = int((alt < 0.07).sum())
        return n, l_eta[:n, i0:i1].mean(axis=1).max()

    n_sin, dn_sin = dentro_de(M.DN_PARED)
    n_con, dn_con = dentro_de(M.DN_PARED, n_capas_pared=15, crecimiento_pared=1.0)
    assert n_con > n_sin
    assert dn_con < dn_sin


def test_sin_capa_de_pared_la_malla_no_cambia(malla_naca):
    """El defecto es 0 capas: la malla tiene que salir identica."""
    X, Y, _ = malla_naca
    px, py = _perfil("NACA_0012_sharp")
    X2, Y2, _ = M.generar_c(px, py, n_capas_pared=0)
    np.testing.assert_array_equal(X, X2)
    np.testing.assert_array_equal(Y, Y2)


def test_crecimiento_normal_acotado(malla_naca):
    X, Y, info = malla_naca
    q = M.calidad(X, Y)
    assert q["crecimiento_max"] < 1.30 * info["crecimiento"]


# ---------------------------------------------------------------------------
# Topologia del corte de estela
# ---------------------------------------------------------------------------
def test_corte_de_estela_es_espejo_exacto(malla_naca):
    """En j=0, la celda i y la N-1-i de la estela son el mismo punto fisico.

    De esto depende que el emparejamiento del corte dentro del solver sea una
    permutacion de indices exacta y no una interpolacion.
    """
    X, Y, info = malla_naca
    a, b = info["estela_inf"]
    N = X.shape[1]
    np.testing.assert_allclose(X[0, a:b], X[0, N - b:N - a][::-1], rtol=0, atol=1e-14)
    np.testing.assert_allclose(Y[0, a:b], Y[0, N - b:N - a][::-1], rtol=0, atol=1e-14)


def test_la_estela_esta_en_la_linea_del_te(malla_naca):
    X, Y, info = malla_naca
    a, b = info["estela_inf"]
    y_te = Y[0, info["perfil"][0]]
    np.testing.assert_allclose(Y[0, a:b], y_te, atol=1e-14)
    assert X[0, a] == pytest.approx(info["x_out"])


# ---------------------------------------------------------------------------
# Simetria
# ---------------------------------------------------------------------------
def test_perfil_simetrico_da_malla_simetrica():
    """NACA 0012 a alpha=0: la malla tiene que ser simetrica respecto a y=0.

    En malla adaptada el angulo de ataque se aplica a la corriente, no a la
    geometria, asi que esta simetria es la que hace exacto el test de Cl = 0.
    """
    px, py = _perfil("NACA_0012_sharp")
    X, Y, info = M.generar_c(px, py)
    N = X.shape[1]
    esc = max(np.abs(X).max(), np.abs(Y).max())
    assert np.max(np.abs(X - X[:, ::-1])) / esc < 1e-9
    assert np.max(np.abs(Y + Y[:, ::-1])) / esc < 1e-9


# ---------------------------------------------------------------------------
# Borde de salida romo
# ---------------------------------------------------------------------------
def test_base_subcelda_se_afila_y_se_reporta():
    """Una base mas fina que el paso de pared no se puede resolver: se afila.

    Lo importante es que quede constancia: `te_afilado` es la bandera que
    distingue una geometria respetada de una modificada a proposito.
    """
    px, py = _perfil("NACA_0012_sharp")
    p = np.column_stack([px, py])
    p[0, 1] += 3.0e-4                            # abre una base de 6e-4
    p[-1, 1] -= 3.0e-4
    _, rangos = M.curva_inicial(p[:, 0], p[:, 1], dn_pared=2.0e-3)
    assert rangos["gap_te"] > 5.0e-4
    assert rangos["te_afilado"] is True
    assert rangos["n_base"] == 0


def test_te_afilado_no_se_marca_como_romo():
    px, py = _perfil("NACA_0012_sharp")
    _, rangos = M.curva_inicial(px, py)
    assert rangos["gap_te"] == 0.0
    assert rangos["te_afilado"] is False


# ---------------------------------------------------------------------------
# Distribucion de superficie
# ---------------------------------------------------------------------------
def test_la_superficie_pasa_por_la_geometria():
    """Los puntos redistribuidos caen sobre el contorno original.

    Con tolerancia de la flecha del spline: lo que se comprueba es que la
    redistribucion no deforma el perfil, no que interpole los puntos dados.
    """
    px, py = _perfil("E387.dat")
    sx, sy, _ = M.redistribuir_superficie(px, py, 256)
    a = np.column_stack([px, py])[:-1]
    b = np.column_stack([px, py])[1:]
    ab = b - a
    largo = np.maximum((ab * ab).sum(1), 1e-300)
    pt = np.column_stack([sx, sy])
    t = np.clip(((pt[:, None] - a[None]) * ab[None]).sum(2) / largo[None], 0.0, 1.0)
    proy = a[None] + t[..., None] * ab[None]
    d = np.min(np.hypot(*(pt[:, None] - proy).transpose(2, 0, 1)), axis=1)
    assert d.max() < 1.0e-3


def test_agrupamiento_en_le_y_te():
    px, py = _perfil("NACA_0012_sharp")
    sx, sy, info = M.redistribuir_superficie(px, py, 256)
    d = np.hypot(np.diff(sx), np.diff(sy))
    i_le = info["i_le"]
    assert d[0] < 0.4 * d.max()                  # TE agrupado
    assert d[i_le] < 0.4 * d.max()               # LE agrupado


# ---------------------------------------------------------------------------
# Coste
# ---------------------------------------------------------------------------
def test_generar_es_barato():
    """El GA paga ~12 min por evaluacion CFD: mallar tiene que ser ruido."""
    import time
    px, py = _perfil("NACA_0012_sharp")
    M.generar_c(px, py)                          # calentar scipy
    t0 = time.time()
    for _ in range(5):
        M.generar_c(px, py)
    assert (time.time() - t0) / 5.0 < 1.0


def test_dibujar_escribe_la_figura(tmp_path):
    """Humo: la figura se genera y no es un fichero vacio."""
    px, py = _perfil("NACA_0012_sharp")
    X, Y, info = M.generar_c(px, py)
    destino = tmp_path / "malla.png"
    M.dibujar(X, Y, str(destino), rangos=info, titulo="prueba")
    assert destino.exists() and destino.stat().st_size > 10_000

"""El criterio de parada por fuerzas, sobre series sinteticas y sobre las reales.

Las sinteticas cubren los tres modos de fallo que mataron a las tres versiones
del criterio del cartesiano: la deriva lenta y monotona (que la auto-similitud
daba por convergida), la oscilacion permanente (que el ruido crudo declaraba
eternamente no convergida) y la meseta con ruido blanco, que es lo unico que
todas acertaban.
"""

import os

import numpy as np
import pytest

from curvo import convergencia as cvg

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
TOL = 5e-4

# Las validaciones cerradas, como series de referencia: 401 muestras de t=0 a 20.
CASOS = ["v1_naca0012_re1e5_a5", "v2_dominio_24x12", "v3_ymas1_dn8e5",
         "v5_suavizador_pcr"]


def corre(t, cl, cd, **kw):
    """Pasa la serie entera por `ParadaFuerzas`. Devuelve (t_parada, resumen)."""
    p = cvg.ParadaFuerzas(**kw)
    for k in range(len(t)):
        if p.anotar(t[k], cl[k], cd[k])["ok"]:
            return t[k], p.resumen()
    return None, p.resumen()


def test_meseta_con_ruido_para():
    t = np.arange(0.0, 20.0, 0.05)
    r = np.random.default_rng(0).normal(0.0, 1e-4, len(t))
    t_p, res = corre(t, 0.5 + r, 0.02 + 0.1 * r, min_t=5.0)
    assert t_p is not None and t_p < 8.0
    assert abs(res["Cl"] - 0.5) < TOL


def test_deriva_lenta_no_para_antes_de_tiempo():
    """`A - B exp(-t/tau)`, que es la forma real de estas series.

    El error al parar tiene que quedarse en el orden de la tolerancia: es lo que
    la version 1 del cartesiano no cumplia (paraba en t~1-2 con hasta un 47 %).
    """
    t = np.arange(0.0, 60.0, 0.05)
    cl = 0.5 - 0.26 * np.exp(-t / 2.9)
    cd = 0.019 + 0.016 * np.exp(-t / 2.9)
    t_p, res = corre(t, cl, cd, min_t=5.0)
    assert t_p is not None
    assert abs(res["Cl"] - 0.5) < 2 * TOL
    assert abs(res["Cd"] - 0.019) < 2 * TOL
    # Y no para en el transitorio: a t=10 le quedan 8e-3 de Cl por recorrer.
    assert t_p > 12.0


def test_rampa_sin_meseta_no_para():
    """Una deriva que no decae no converge nunca, por pequena que sea."""
    t = np.arange(0.0, 40.0, 0.05)
    t_p, _ = corre(t, 0.5 + 2e-4 * t, 0.02 + 1e-5 * t, min_t=5.0)
    assert t_p is None


def test_oscilacion_permanente_para_por_la_media():
    """Ciclo limite de amplitud 20 veces la tolerancia sobre una media fija.

    La version 2 del cartesiano media `sigma/|media|` y aqui no habria abierto
    nunca: lo que se mide es la incertidumbre de la media, no la amplitud.
    """
    t = np.arange(0.0, 40.0, 0.02)
    osc = 1e-2 * np.sin(2 * np.pi * t / 0.3)
    t_p, res = corre(t, 0.5 + osc, 0.02 + 0.1 * osc, min_t=5.0)
    assert t_p is not None
    assert abs(res["Cl"] - 0.5) < TOL
    assert res["banda"]["Cl"] < TOL < 1e-2


def test_oscilacion_sobre_deriva_espera_a_la_deriva():
    t = np.arange(0.0, 60.0, 0.02)
    osc = 5e-3 * np.sin(2 * np.pi * t / 0.3)
    cl = 0.5 - 0.26 * np.exp(-t / 2.9) + osc
    t_p, _ = corre(t, cl, 0.02 + 0.1 * osc, min_t=5.0)
    assert t_p is not None and t_p > 12.0


def test_min_t_manda():
    t = np.arange(0.0, 20.0, 0.05)
    cl = np.full_like(t, 0.5)
    t_p, _ = corre(t, cl, np.full_like(t, 0.02), min_t=9.0)
    assert t_p is not None and t_p >= 9.0


def test_serie_corta_no_dice_nada():
    e = cvg.estado_serie([0.0, 0.1, 0.2], [0.5, 0.5, 0.5])
    assert not e["ok"] and np.isnan(e["banda"])


def test_nan_no_convergen():
    t = np.arange(0.0, 5.0, 0.05)
    cl = np.full_like(t, 0.5)
    cl[10] = np.nan
    assert not cvg.estado_serie(t, cl)["ok"]


def test_la_banda_no_confunde_amplitud_con_incertidumbre():
    """Sobre un ciclo limite la banda tiene que ser mucho menor que la amplitud.

    Es el fallo que las medias de bloque arreglan: `sigma` vale la amplitud del
    seno (7e-3) y la media de la ventana esta determinada a ~1e-5.
    """
    t = np.arange(0.0, 40.0, 0.02)
    s = 0.5 + 1e-2 * np.sin(2 * np.pi * t / 0.3)
    e = cvg.estado_serie(t, s)
    assert e["ok"] and e["banda"] < TOL
    assert e["banda"] < np.std(s) / 10.0


def test_la_banda_vale_lo_que_debe_en_ruido_blanco():
    """Con muestras independientes tiene que dar el CI95 de toda la vida."""
    g = np.random.default_rng(1)
    t = np.arange(0.0, 20.0, 0.05)
    s = 0.5 + g.normal(0.0, 1e-3, len(t))
    e = cvg.estado_serie(t, s, tol=1e-9)   # tol imposible: usa la serie entera
    ideal = 1.96 * 1e-3 / np.sqrt(len(t))
    assert 0.4 * ideal < e["banda"] < 2.5 * ideal


@pytest.mark.parametrize("caso", CASOS)
def test_sobre_las_validaciones_cerradas(caso):
    """Sobre las corridas reales: para tarde y con el error que promete.

    La referencia es la asintota del ajuste `A - B exp(-t/tau)` sobre la cola de
    la propia serie, que es el valor al que la corrida iba.
    """
    z = np.load(os.path.join(RAIZ, "validacion", caso, "historia.npz"),
                allow_pickle=True)
    d = z["datos"]
    t, cl, cd = d[:, 0], d[:, 1], d[:, 4]

    # Asintota por dos bloques lejanos: con decaimiento geometrico de razon r,
    # A = m2 + (m2-m1) r/(1-r). Evita depender de scipy.
    def asintota(s):
        m = [s[(t > a) & (t <= b)].mean()
             for a, b in ((11.0, 14.0), (14.0, 17.0), (17.0, 20.0))]
        r = (m[2] - m[1]) / (m[1] - m[0])
        return m[2] + (m[2] - m[1]) * r / (1.0 - r)

    t_p, res = corre(t, cl, cd, tol=1e-3, min_t=5.0)
    assert t_p is not None, "no para dentro de los 20 tiempos convectivos"
    assert t_p > 10.0
    assert abs(res["Cl"] - asintota(cl)) < 1.5e-3
    assert abs(res["Cd"] - asintota(cd)) < 1.5e-3
    # El extrapolado apunta mejor que la media de la ventana, que es para lo que
    # se guarda.
    assert (abs(res["extrapolado"]["Cl"] - asintota(cl))
            < abs(res["Cl"] - asintota(cl)))


def test_correr_para_por_fuerzas():
    """El enganche en `Solver.correr`: se para y deja el resumen escrito.

    Malla gruesa y tolerancia floja a proposito -- lo que se comprueba aqui es el
    cableado (que `coeficientes()` da numeros y que el lazo sale), no el
    criterio, que ya lo miden los tests de arriba.
    """
    from curvo import malla as M
    from curvo.solver import Solver

    px, py = M.leer_dat(os.path.join(RAIZ, "profiles", "NACA_0012_sharp"))
    X, Y, info = M.generar_c(px, py, n_sup=60, n_estela=40, dn_pared=2e-3,
                             n_capas_pared=0, distancia_lejos=6.0, x_out=6.0)
    s = Solver(X, Y, info, nu=1e-2, alfa=5.0, dt=5e-3)
    p = cvg.ParadaFuerzas(tol=1.0, ventana=0.05, min_t=0.0, sostenido=1)
    h = s.correr(200, cada=2, parada_fuerzas=p)

    assert p.resumen()["convergida"]
    assert len(h) < 200 // 2
    assert np.isfinite(s.coeficientes()).all()

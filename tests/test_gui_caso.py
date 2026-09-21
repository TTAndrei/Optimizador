from pathlib import Path

import pytest

from gui.caso import Caso, CasoError, previsualizar_malla


RAIZ = Path(__file__).parents[1]


def caso_pequeno():
    caso = Caso(perfil="NACA_0012_sharp")
    caso.malla.update(n_sup=32, n_estela=32, distancia_lejos=2.0, x_out=4.0,
                      n_capas_pared=2)
    caso.solver.update(turbulento=False)
    caso.ejecucion.update(backend="numpy", pasos=2)
    return caso


def test_caso_roundtrip_y_defaults(tmp_path):
    ruta = tmp_path / "caso.json"
    caso = caso_pequeno()
    caso.guardar(ruta)
    cargado = Caso.cargar(ruta)
    assert cargado.a_dict() == caso.a_dict()
    assert cargado.argumentos_malla()["dn_max"] == float("inf")


def test_caso_rechaza_perfil_inexistente():
    with pytest.raises(CasoError, match="no existe el perfil"):
        Caso(perfil="no-existe.dat").validar(RAIZ)


def test_preview_genera_malla_y_calidad():
    X, Y, info, calidad = previsualizar_malla(caso_pequeno(), RAIZ)
    assert X.shape == Y.shape
    assert X.ndim == 2
    assert info["perfil"][1] > info["perfil"][0]
    assert calidad["n_celdas"] == (X.shape[0] - 1) * (X.shape[1] - 1)


def test_preview_acepta_controles_avanzados_de_malla():
    caso = caso_pequeno()
    caso.malla.update(razon_le=0.12, razon_te=0.2, ancho_le=0.08,
                      ancho_te=0.07, mezcla_volumen=0.45,
                      disipacion=0.35, disipacion_curva=6.0)
    X, Y, _info, calidad = previsualizar_malla(caso, RAIZ)
    assert X.shape == Y.shape
    assert calidad["valida"]

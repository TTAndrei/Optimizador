"""Tests del importador DXF. Sin GPU: se pueden correr en cualquier sitio.

Los DXF de prueba se generan con el propio ezdxf, asi que el test no depende de
ningun fichero binario en el repo y cubre justamente los casos que rompen: una
sopa de segmentos sueltos, curvas que hay que aplanar, un contorno dentro de
otro y un lazo que no cierra.
"""
import math

import numpy as np
import pytest

ezdxf = pytest.importorskip("ezdxf")

from geom_import import (GeometriaAbierta, area_con_signo, cargar_dxf, coser)


def _doc():
    d = ezdxf.new("R2010")
    d.header["$INSUNITS"] = 6      # metros
    return d


def _dxf(tmp_path, doc, nombre="t.dxf"):
    p = tmp_path / nombre
    doc.saveas(p)
    return str(p)


# ----------------------------------------------------------------------
def test_circulo_area_y_tolerancia_cordal(tmp_path):
    """Un CIRCLE se aplana a poligono. El area del poligono inscrito queda por
    debajo de la del circulo, y la diferencia la acota la tolerancia de cuerda:
    es la comprobacion de que `flattening` hace lo que decimos que hace."""
    d = _doc()
    d.modelspace().add_circle((0, 0), radius=1.0)
    lazos = cargar_dxf(_dxf(tmp_path, d), tol_cordal=1e-3)

    assert len(lazos) == 1
    l = lazos[0]
    assert l["cerrado"]
    assert l["rol"] == "cuerpo"          # un solo lazo: no hay contorno exterior
    assert l["area"] == pytest.approx(math.pi, rel=2e-3)
    assert l["area"] > 0                  # antihorario
    assert l["area"] < math.pi            # poligono inscrito


def test_rectangulo_de_cuatro_lineas_sueltas(tmp_path):
    """Cuatro LINE inconexas y desordenadas: el cosido tiene que reconstruir el
    lazo. Es como salen la mayoria de los DXF de CAD."""
    d = _doc()
    msp = d.modelspace()
    esquinas = [(0, 0), (2, 0), (2, 1), (0, 1)]
    lineas = [(esquinas[i], esquinas[(i + 1) % 4]) for i in range(4)]
    for a, b in [lineas[2], lineas[0], lineas[3], lineas[1]]:   # desordenadas
        msp.add_line(a, b)

    lazos = cargar_dxf(_dxf(tmp_path, d), tol_cordal=1e-3)
    assert len(lazos) == 1
    assert lazos[0]["cerrado"]
    assert lazos[0]["area"] == pytest.approx(2.0, rel=1e-9)


def test_anidamiento_exterior_y_cuerpo(tmp_path):
    """Un rectangulo grande con un circulo dentro: el grande es el contorno
    exterior del dominio y el circulo un cuerpo sumergido."""
    d = _doc()
    msp = d.modelspace()
    msp.add_lwpolyline([(0, 0), (10, 0), (10, 4), (0, 4)], close=True)
    msp.add_circle((5, 2), radius=0.5)

    lazos = cargar_dxf(_dxf(tmp_path, d), tol_cordal=1e-3)
    roles = {l["rol"] for l in lazos}
    assert roles == {"exterior", "cuerpo"}
    ext = next(l for l in lazos if l["rol"] == "exterior")
    cue = next(l for l in lazos if l["rol"] == "cuerpo")
    assert ext["area"] == pytest.approx(40.0, rel=1e-9)
    assert cue["area"] == pytest.approx(math.pi * 0.25, rel=5e-3)
    assert ext["profundidad"] == 0 and cue["profundidad"] == 1


def test_dos_cuerpos_sin_exterior(tmp_path):
    """Dos circulos separados: ninguno contiene al otro, asi que no hay contorno
    exterior y los dos son cuerpos. El caso 'dos perfiles en la caja'."""
    d = _doc()
    d.modelspace().add_circle((0, 0), radius=1.0)
    d.modelspace().add_circle((5, 0), radius=1.0)

    lazos = cargar_dxf(_dxf(tmp_path, d), tol_cordal=1e-3)
    assert len(lazos) == 2
    assert all(l["rol"] == "cuerpo" for l in lazos)


def test_lazo_abierto_no_se_cierra_a_la_fuerza(tmp_path):
    """Tres lados de un rectangulo. En estricto tiene que fallar diciendo DONDE
    esta el hueco; en no estricto, devolverlo marcado para que la GUI lo pinte."""
    d = _doc()
    msp = d.modelspace()
    msp.add_line((0, 0), (2, 0))
    msp.add_line((2, 0), (2, 1))
    msp.add_line((2, 1), (0, 1))

    ruta = _dxf(tmp_path, d)
    with pytest.raises(GeometriaAbierta):
        cargar_dxf(ruta, tol_cordal=1e-3)

    lazos = cargar_dxf(ruta, tol_cordal=1e-3, estricto=False)
    assert len(lazos) == 1 and not lazos[0]["cerrado"]


def test_spline_se_aplana_y_cierra(tmp_path):
    """Una SPLINE cerrada: comprueba que la ruta de aplanado tambien vale para
    curvas libres, que es lo que sale de un CAD de verdad."""
    d = _doc()
    th = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts = [(2 * np.cos(t), np.sin(t)) for t in th]
    d.modelspace().add_spline(pts + [pts[0]])

    lazos = cargar_dxf(_dxf(tmp_path, d), tol_cordal=1e-3)
    assert len(lazos) == 1 and lazos[0]["cerrado"]
    assert lazos[0]["area"] > 0


def test_unidades_milimetros_pasan_a_metros(tmp_path):
    """$INSUNITS=4 son milimetros. Un CAD que exporta en mm y un solver que
    piensa en metros es la forma mas facil de simular algo mil veces mas grande
    de lo que se cree."""
    d = ezdxf.new("R2010")
    d.header["$INSUNITS"] = 4
    d.modelspace().add_lwpolyline([(0, 0), (1000, 0), (1000, 500), (0, 500)],
                                  close=True)

    lazos = cargar_dxf(_dxf(tmp_path, d), tol_cordal=1.0)
    assert lazos[0]["area"] == pytest.approx(0.5, rel=1e-9)   # 1 m x 0.5 m


def test_filtro_por_capa(tmp_path):
    d = _doc()
    msp = d.modelspace()
    msp.add_circle((0, 0), radius=1.0, dxfattribs={"layer": "PARED"})
    msp.add_circle((0, 0), radius=0.2, dxfattribs={"layer": "AUXILIAR"})

    lazos = cargar_dxf(_dxf(tmp_path, d), capa="PARED", tol_cordal=1e-3)
    assert len(lazos) == 1
    assert lazos[0]["capa"] == "PARED"


def test_area_con_signo():
    x = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    assert area_con_signo(x, y) == pytest.approx(1.0)
    assert area_con_signo(x[::-1], y[::-1]) == pytest.approx(-1.0)

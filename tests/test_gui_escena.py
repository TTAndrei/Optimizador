"""Camino completo sin ventana: DXF -> escena -> malla previsualizada.

Sin GPU y sin abrir pantalla (QT_QPA_PLATFORM=offscreen lo pone conftest si
hace falta). Cubre el enganche entre los tres modulos, que es donde se rompen
estas cosas: cada uno funciona solo y juntos no.
"""
import os
import sys

import numpy as np
import pytest

ezdxf = pytest.importorskip("ezdxf")
pytest.importorskip("cupy")          # ejes() importa Simulador2D

from gui.escena import Contorno, Escena, Parche


def _dxf_conducto(tmp_path):
    """Un conducto recto de 4 x 1 m con un cilindro dentro."""
    d = ezdxf.new("R2010")
    d.header["$INSUNITS"] = 6
    msp = d.modelspace()
    msp.add_lwpolyline([(0, 0.5), (4, 0.5), (4, 1.5), (0, 1.5)], close=True)
    msp.add_circle((1.5, 1.0), radius=0.15)
    p = tmp_path / "conducto.dxf"
    d.saveas(p)
    return str(p)


def test_dxf_a_escena(tmp_path):
    esc = Escena.desde_dxf(_dxf_conducto(tmp_path), Lx=4.0, Ly=2.0, dx_min=0.05)
    roles = sorted(c.rol for c in esc.contornos)
    assert roles == ["cuerpo", "exterior"]
    assert len(esc.parches) == 4
    # La banda fina sale del cilindro, no del conducto: el bbox del contorno
    # exterior es casi el dominio entero.
    x0, x1, y0, y1 = esc.banda_fina()
    assert x1 - x0 < 2.0 and y1 - y0 < 2.0


def test_diagnostico_de_malla(tmp_path):
    from gui.vista_malla import diagnostico
    esc = Escena.desde_dxf(_dxf_conducto(tmp_path), Lx=4.0, Ly=2.0, dx_min=0.05)
    esc.solver = {"nu": 0.05, "v0x": 1.0, "CFL": 0.5, "iteraciones": 1000}
    texto, avisos, X, Y, solid = diagnostico(esc)

    assert solid.any() and not solid.all()
    # Fluido dentro del conducto, sólido fuera.
    col = solid[:, solid.shape[1] // 2]
    assert col[(Y > 0.6) & (Y < 0.9)].sum() == 0
    assert col[Y < 0.4].all() and col[Y > 1.6].all()
    assert "Malla" in texto and "dt previsto" in texto


def test_escena_avisa_de_lo_que_rompe():
    esc = Escena(Lx=4.0, Ly=2.0)
    esc.parches = [Parche("left", "inflow", (1.0, 0.0))]     # sin salida
    avisos = " ".join(esc.avisos())
    assert "salida" in avisos

    esc.parches.append(Parche("right", "outflow", 0.0))
    esc.parches.append(Parche("top", "outflow", 5.0))        # dos presiones
    assert "presiones de salida distintas" in " ".join(esc.avisos())


def test_roundtrip_json(tmp_path):
    th = np.linspace(0, 2 * np.pi, 64)
    esc = Escena(Lx=4, Ly=2,
                 contornos=[Contorno(x=list(np.cos(th) + 2),
                                     y=list(np.sin(th) + 1), nombre="c")],
                 parches=[Parche("left", "inflow", (1.0, 0.0))])
    p = str(tmp_path / "e.json")
    esc.guardar(p)
    otra = Escena.cargar(p)
    assert otra.contornos[0].nombre == "c"
    assert otra.parches[0].tipo == "inflow"
    assert np.allclose(otra.contornos[0].x, esc.contornos[0].x)

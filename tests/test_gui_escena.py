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


def _dxf_tobera(tmp_path):
    """Contraccion 2:1: contorno exterior que NO es el rectangulo de la caja,
    asi que deja pared solida dentro del dominio."""
    d = ezdxf.new("R2010")
    d.header["$INSUNITS"] = 6
    d.modelspace().add_lwpolyline(
        [(0, 0.0), (2, 0.0), (4, 0.5), (4, 1.5), (2, 2.0), (0, 2.0)],
        close=True)
    p = tmp_path / "tobera.dxf"
    d.saveas(p)
    return str(p)


def _dxf_cilindro(tmp_path):
    """Solo el objeto, lejos del origen: el DXF no dice nada del dominio."""
    d = ezdxf.new("R2010")
    d.header["$INSUNITS"] = 6
    d.modelspace().add_circle((37.0, -12.0), radius=0.5)
    p = tmp_path / "cilindro.dxf"
    d.saveas(p)
    return str(p)


def test_dxf_a_escena(tmp_path):
    esc = Escena.desde_dxf(_dxf_conducto(tmp_path), Lx=4.0, Ly=2.0, dx_min=0.05)
    roles = sorted(c.rol for c in esc.contornos)
    assert roles == ["cuerpo", "exterior"]
    assert len(esc.parches) == 4
    # La banda fina sale del cilindro en x. En y cubre la seccion entera: en un
    # conducto la cortadura no es una capa pegada a la pared.
    x0, x1, y0, y1 = esc.banda_fina()
    assert x1 - x0 < 2.0
    assert (y0, y1) == (0.0, esc.Ly)


def test_el_dxf_con_paredes_manda_sobre_lx_ly(tmp_path):
    """El caso que se veia como una pared negra flotando: el DXF trae el
    contorno exterior y el usuario sube Ly a mano. La caja tiene que salir del
    contorno, y si no coincide hay que avisar."""
    esc = Escena.desde_dxf(_dxf_conducto(tmp_path), Lx=4.0, Ly=2.0, dx_min=0.05)
    assert esc.modo_dominio == "dxf"
    assert esc.Lx == pytest.approx(4.0) and esc.Ly == pytest.approx(1.0)
    # Arriba y abajo son pared del conducto, no campo lejano.
    tipos = {p.lado: p.tipo for p in esc.parches}
    assert tipos["top"] == "noslip" and tipos["bottom"] == "noslip"
    assert not esc.avisos()

    esc.Ly = 2.0
    assert any("no coincide con el contorno exterior" in a for a in esc.avisos())
    esc.ajustar_dominio()
    assert esc.Ly == pytest.approx(1.0) and not esc.avisos()


def test_el_dxf_de_solo_objeto_se_coloca_en_la_caja(tmp_path):
    """El otro caso: el DXF no dice nada del dominio, asi que lo ponemos
    nosotros y el objeto se coloca dentro, sin quedarse en el rincon."""
    esc = Escena.desde_dxf(_dxf_cilindro(tmp_path), dx_min=0.05)
    assert esc.modo_dominio == "caja"
    c = esc.contornos[0]
    assert c.centro() == pytest.approx((0.25 * esc.Lx, 0.5 * esc.Ly))
    assert not esc.avisos()

    # Moverlo mueve el offset, no los puntos del CAD.
    crudo = list(c.x)
    c.mover_centro_a(3.0, 0.5 * esc.Ly)
    assert c.centro() == pytest.approx((3.0, 0.5 * esc.Ly))
    assert c.x == crudo
    assert not esc.avisos()

    c.mover_centro_a(-5.0, 0.0)
    assert any("se sale de la caja" in a for a in esc.avisos())


def test_diagnostico_de_malla(tmp_path):
    """Conducto recto encajado a su bbox: sus paredes son las BC del borde, no
    solido inmerso. Lo unico solido es el cilindro."""
    from gui.vista_malla import diagnostico
    esc = Escena.desde_dxf(_dxf_conducto(tmp_path), dx_min=0.05)
    esc.solver = {"nu": 0.05, "v0x": 1.0, "CFL": 0.5, "iteraciones": 1000}
    texto, avisos, X, Y, solid = diagnostico(esc)

    assert solid.any() and not solid.all()
    # Las paredes del conducto caen sobre las filas del borde: solidas, que es
    # como se representa una pared en este solver.
    assert solid[0, :].all() and solid[-1, :].all()
    # Y la boca de entrada NO: solo las dos celdas de esquina, donde la pared la
    # corta. Si se tapona entera no entra flujo.
    assert solid[1:-1, 0].sum() == 0, "la columna de entrada ha quedado taponada"
    # El cilindro esta en (1.5, 1.0) del DXF, que tras encajar la caja es y=0.5.
    j = int(np.searchsorted(X, 1.5))
    assert solid[1:-1, j].any()
    assert "Malla" in texto and "dt previsto" in texto


def test_la_tobera_si_deja_pared_solida(tmp_path):
    """Un contorno exterior que no es el rectangulo de la caja SI deja solido
    dentro: es la diferencia con el conducto recto."""
    from gui.vista_malla import mascara, ejes
    esc = Escena.desde_dxf(_dxf_tobera(tmp_path), dx_min=0.05)
    # Un solo lazo no se puede clasificar por anidamiento, asi que el importador
    # lo da por cuerpo; marcarlo exterior es justo lo que hace la GUI a mano.
    assert esc.modo_dominio == "caja"
    esc.contornos[0].rol = "exterior"
    esc.autoconfigurar_dominio()
    assert esc.modo_dominio == "dxf"
    X, Y = ejes(esc)
    solid = mascara(esc, X, Y)
    assert solid.any(), "la contraccion tiene que dejar pared solida"
    # La garganta esta a la derecha: alli hay solido arriba y abajo.
    assert solid[0, -1] or solid[-1, -1]
    # Y la boca de entrada, que es toda seccion libre, salvo las esquinas.
    assert solid[1:-1, 0].sum() == 0


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


def test_no_se_lee_memoria_compartida_de_otra_corrida():
    """La shm POSIX sobrevive al proceso que la creo, asi que un bloque de una
    corrida anterior se abre igual de bien y se pinta con SU malla y SU dominio:
    es la mancha negra que no encaja con el DXF. El pid de la cabecera lo corta.
    """
    import struct
    from multiprocessing import shared_memory
    from gui.runner import COORDS, DATOS, META, Runner

    ny, nx = 39, 81
    tam = {META: 48, DATOS: ny * nx * 5, COORDS: (ny + nx) * 4}
    bloques = {}
    for nombre, n in tam.items():
        try:
            v = shared_memory.SharedMemory(name=nombre)
            v.close(); v.unlink()
        except FileNotFoundError:
            pass
        bloques[nombre] = shared_memory.SharedMemory(name=nombre, create=True, size=n)
    try:
        struct.pack_into("ii", bloques[META].buf, 0, ny, nx)
        struct.pack_into("i", bloques[META].buf, 40, 999999)

        r = Runner()
        assert r._abrir_shm() is False, "sin proceso no se lee nada"

        class Falso:
            pid = 4242
            def poll(self): return None
        r.proc = Falso()
        assert r._abrir_shm() is False, "el pid no es el nuestro: no se lee"
        assert not r._shm

        r.proc.pid = 999999                      # ahora si es nuestro
        assert r._abrir_shm() is True
        r._cerrar_shm()
    finally:
        for b in bloques.values():
            b.close()
            b.unlink()


def test_zona_de_refinado_a_mano_y_dy_propio(tmp_path):
    """La caja de refinado se puede fijar a mano y el eje y puede llevar su
    propio dx. Es lo que permite resolver una capa límite sin pagar la misma
    resolución a lo largo del flujo."""
    from gui.vista_malla import ejes
    esc = Escena.desde_dxf(_dxf_conducto(tmp_path), dx_min=0.02)
    esc.solver = {"ratio_max_malla": 8}

    auto = esc.banda_fina()
    esc.refinado = (1.0, 3.0, 0.1, 0.9)
    assert esc.banda_fina() == (1.0, 3.0, 0.1, 0.9)
    esc.refinado = None
    assert esc.banda_fina() == auto, "quitar la caja vuelve a la automática"

    X, Y = ejes(esc)
    assert np.diff(Y).min() == pytest.approx(esc.dx_min, rel=1e-6)
    esc.dy_min = 0.005
    assert esc.dy == pytest.approx(0.005)
    X2, Y2 = ejes(esc)
    assert np.diff(Y2).min() == pytest.approx(0.005, rel=1e-6)
    assert len(X2) == len(X), "cambiar dy no debe tocar el eje x"

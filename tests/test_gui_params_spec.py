"""Contrato entre `gui/params_spec.py` y `gui/caso.py`.

Lo que se protege es que la reorganizacion en pestanas no pierda parametros por
el camino: cada campo del caso tiene que tener control, y cada control tiene que
tener una frase de ayuda que explique a que afecta.
"""

from gui.caso import Caso
from gui.params_spec import PERFIL, PESTANAS, parametros


def test_todos_los_parametros_del_caso_tienen_control():
    """Ningun parametro del caso se queda sin editar desde la GUI."""
    en_spec = {clave for clave, *_ in parametros()} | {PERFIL[0]}
    caso = Caso()
    # `directorio` lo fija la GUI al lanzar, no el usuario.
    esperados = ({"perfil"} | set(caso.malla) | set(caso.solver)
                 | set(caso.ejecucion) - {"directorio"})
    assert esperados - en_spec == set()


def test_ningun_control_apunta_a_un_parametro_inexistente():
    caso = Caso()
    conocidos = set(caso.malla) | set(caso.solver) | set(caso.ejecucion)
    assert {clave for clave, *_ in parametros()} - conocidos == set()


def test_cada_parametro_explica_a_que_afecta():
    """La ayuda es el tooltip: una frase corta no sirve de nada."""
    for clave, _, _, ayuda, _ in parametros():
        assert ayuda and len(ayuda) >= 40, clave


def test_los_desplegables_traen_sus_opciones():
    for clave, _, tipo, _, opciones in parametros():
        if tipo == "c":
            assert opciones, clave
        else:
            assert opciones is None, clave


def test_no_hay_parametros_repetidos_entre_bloques():
    claves = [clave for clave, *_ in parametros()]
    assert len(claves) == len(set(claves))


def test_los_bloques_tienen_nombre_y_contenido():
    for pestana, bloques in PESTANAS:
        assert pestana and bloques
        for nombre, entradas in bloques:
            assert nombre and entradas, pestana

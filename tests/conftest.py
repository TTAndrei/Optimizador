"""Hace importable `curvo` desde los tests.

pytest mete en `sys.path` el directorio del test, no la raiz del repo, y el
paquete vive un nivel mas arriba. Los tests del solver cartesiano tienen su
propio conftest en `Sim_Cartesiano/tests/`.
"""
import os
import sys

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RAIZ not in sys.path:
    sys.path.insert(0, _RAIZ)

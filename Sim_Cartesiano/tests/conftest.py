"""Hace importables los módulos del solver cartesiano y de `scripts/`.

`_ROOT` es `Sim_Cartesiano/`, no la raíz del repo: desde la reorganización el
solver cartesiano y todo lo suyo viven aquí dentro.

`test_generacion_geometrica_ga.py` hace `import RunGA`, que vive en scripts/.
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Los tests de la GUI construyen widgets: sin esto Qt busca un servidor grafico
# y falla en CI o por SSH.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session", autouse=True)
def _cwd_del_cartesiano():
    """Sitúa el directorio de trabajo en `Sim_Cartesiano/` mientras corren.

    El solver escribe `sim_last.npz` y `polar_results.json` con rutas relativas
    al directorio actual, y varios tests piden `filepath="profiles/NACA_0012"`.
    Lanzados desde la raíz del repo eso ensuciaba la raíz y dependía de dónde se
    invocara pytest. Los tests de `curvo/` no se ven afectados: resuelven sus
    rutas desde `__file__`.
    """
    previo = os.getcwd()
    os.chdir(_ROOT)
    yield
    os.chdir(previo)

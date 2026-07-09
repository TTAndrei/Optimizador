"""Hace importables los módulos del repo (raíz) y de scripts/ desde los tests.

`test_generacion_geometrica_ga.py` hace `import RunGA`, que vive en scripts/.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from gui.caso import Caso
from gui import run_solver
from gui.run_solver import run


RAIZ = Path(__file__).parents[1]


def test_run_solver_numpy_escribe_outputs(tmp_path):
    caso = Caso(perfil="NACA_0012_sharp")
    caso.malla.update(n_sup=32, n_estela=32, distancia_lejos=2.0, x_out=4.0,
                      n_capas_pared=2)
    caso.solver.update(turbulento=False, dt=1.0e-3)
    caso.ejecucion.update(backend="numpy", pasos=1, cada_historia=1,
                          cada_campo=1, directorio=str(tmp_path / "corrida"))

    salida = run(caso, RAIZ)

    assert (salida / "caso.json").is_file()
    assert (salida / "malla.npz").is_file()
    assert (salida / "historia.npz").is_file()
    assert (salida / "ipc.json").is_file()
    campos = sorted((salida / "campos").glob("campo_*.npz"))
    assert campos
    with np.load(salida / "historia.npz") as historia:
        assert historia["datos"].shape[0] == 2
    with np.load(campos[-1]) as campo:
        assert campo["u"].ndim == 2
        assert campo["p"].shape == campo["u"].shape


def test_main_resuelve_raiz_desde_caso_en_runs(tmp_path):
    caso = Caso(perfil="NACA_0012_sharp")
    ruta = tmp_path / "runs" / "gui_prueba" / "caso.json"
    ruta.parent.mkdir(parents=True)
    caso.guardar(ruta)

    with patch.object(run_solver, "run") as ejecutar, patch.object(
            sys, "argv", ["gui.run_solver", str(ruta)]):
        run_solver.main()

    assert ejecutar.call_args.args[1] == RAIZ

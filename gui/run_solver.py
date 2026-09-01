"""Puente escena.json -> Simulador2D.main(). Corre en el proceso del solver.

Se ejecuta con el directorio de trabajo puesto en la carpeta de la escena: el
solver escribe sim_last.npz y polar_results.json con rutas relativas y no deben
pisar los del repo.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from gui.escena import Escena

# Coste medido en results/coste_salida/INFORME.md. Los volcados van
# submuestreados y el video se monta despues con render_videos.py: dibujar
# figuras dentro del solver cuesta ~20 s cada una y bloquea la GPU.
SALIDA = {
    "ninguno": {"shm_publish": False},
    "monitor": {"shm_publish": True},
    "grabar":  {"shm_publish": True, "dump_fields_max_nx": 900},
}


def main(ruta_escena):
    import Simulador2D

    esc = Escena.cargar(ruta_escena)
    carpeta = os.path.dirname(os.path.abspath(ruta_escena))

    kwargs = dict(esc.solver)
    # La malla la manda la escena, no el diccionario de solver: si no se pasan,
    # main() se queda con SUS defectos (dx_min=0.002) y sale una malla veinte
    # veces mas fina que la previsualizada.
    kwargs.setdefault("dx_min", float(esc.dx_min))
    kwargs.setdefault("factor_expansion", float(esc.factor_expansion))
    kwargs.update(
        Lx=float(esc.Lx), Ly=float(esc.Ly),
        filepath=None,                 # la geometria entra por la escena
        escena=esc.a_dict(),
        graficos=False, mostrar_malla=False,
    )
    kwargs.update(SALIDA.get(esc.modo_salida, SALIDA["monitor"]))
    if esc.modo_salida == "grabar":
        kwargs["dump_fields_dir"] = os.path.join(carpeta, "campos_t")

    os.chdir(carpeta)
    print(f"[solver] escena {ruta_escena}")
    print(f"[solver] {len(esc.contornos)} contorno(s), "
          f"{len(esc.parches)} parche(s) de frontera, salida={esc.modo_salida}")
    Simulador2D.main(**kwargs)

    if esc.modo_salida == "grabar":
        _montar_video(os.path.join(carpeta, "campos_t"), carpeta)


def _montar_video(campos, salida):
    """El render va aqui, fuera del bucle: ~0.2 s por figura en CPU frente a
    ~20 s dentro del solver, y sin robarle la GPU."""
    import glob
    if not glob.glob(os.path.join(campos, "f_*.npz")):
        print("[solver] no hay campos volcados, no se monta vídeo")
        return
    sys.path.insert(0, os.path.join(ROOT, "scripts", "agent_tests"))
    try:
        import render_videos as rv
        frames = os.path.join(salida, "frames")
        n = rv.frames_forma(campos, frames, "simulación", "velocidad")
        rv.make_video(frames, os.path.join(salida, "video.mp4"))
        print(f"[solver] {n} frames -> {os.path.join(salida, 'video.mp4')}")
    except Exception as e:
        print(f"[solver] el vídeo falló ({type(e).__name__}: {e}); "
              f"los campos están en {campos}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("uso: python -m gui.run_solver escena.json")
    main(sys.argv[1])

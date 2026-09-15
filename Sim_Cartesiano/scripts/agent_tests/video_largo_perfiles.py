"""
Dos corridas largas (ganador del AG y NACA 0012 sharp) en el dominio C
validado, volcando campos para montar videos de 30 s de la ZONA REFINADA y del
DOMINIO COMPLETO.

CONFIGURACION
    La de resultados_finales: dominio 24x16 con el perfil en cx=6, Re=1e5,
    alpha=4 (pico de L/D en los dos perfiles), dx=0.002, CFL=0.5, SA con
    transicion SA-BC, presupuesto de proyeccion 2x3 y las cinco optimizaciones
    de opt_solver. Lo unico que cambia es la ventana temporal: 52000
    iteraciones en vez de 26000, o sea t~40 en vez de t~20, para que la estela
    llegue a la salida y el video no sea todo transitorio de arranque.

    Sin parada por convergencia: un corte a mitad deja el video corto y con los
    dos perfiles parando en tiempos fisicos distintos.

POR QUE NO SE DIBUJA AQUI
    save_frame dentro del solver cuesta ~1.2 s por figura a dx=0.002 y bloquea
    la GPU: 722 frames x 2 ventanas x 2 perfiles serian ~2 h de matplotlib
    encima de las ~1.2 h de simulacion. Se vuelcan los campos submuestreados
    (~3 ms por evento sin comprimir, 20 ms comprimido) y se dibuja despues con
    --solo-render, en CPU y sin la GPU parada.

Uso:
    .venv/bin/python scripts/agent_tests/video_largo_perfiles.py
    .venv/bin/python scripts/agent_tests/video_largo_perfiles.py --perfil ganador_ag
    .venv/bin/python scripts/agent_tests/video_largo_perfiles.py --solo-render
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "agent_tests"))
os.chdir(ROOT)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import cupy as cp

import opt_solver
from Simulador2D import main as sim_main

OUT = os.path.join(ROOT, "results", "videos_largos")

PERFILES = {
    # Mismos .dat que resultados_finales. El del ganador se llama
    # NACA_0012_sharp_winner por la familia semilla de la que descendio, no
    # porque sea un NACA, y no es el AG24 de catalogo.
    "ganador_ag": "resultados_finales/perfiles/ganador_ag.dat",
    "naca0012": "resultados_finales/perfiles/naca0012.dat",
}

DOMINIO = {"Lx": 24.0, "Ly": 16.0, "cx": 6.0,
           "ancho_zona_fina_x": 1.5, "ancho_zona_fina_y": 1.0,
           "factor_expansion": 1.1}
ALPHA = 4.0
RE = 1e5
DX = 0.002
ITERS = 52000
FPS = 24
N_FRAMES = 722                    # 722 / 24 fps = 30.1 s
DUMP_CADA = ITERS // N_FRAMES     # 72
GUARDADO = DUMP_CADA // 2         # el bloque de salida cuelga de `it % guardado`
FLAGS = ("warm_start", "warm_start_filtered", "coarse_mask_majority",
         "fast_masks", "interp_float32")

# La banda fina que construye wake_refinement_mode="long_fine_x":
#   x de (cx + c/2) - ancho/2  a  eso + ancho + 1.5c   ->  [5.75, 8.75]
#   y centrada en cy=Ly/2 con ancho_zona_fina_y        ->  [7.50, 8.50]
# Se toma tal cual, sin margen: fuera de ella la malla ya esta estirada y el
# frame mezclaria dos resoluciones sin avisar.
XLIM_REF = (5.75, 8.75)
YLIM_REF = (7.50, 8.50)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def cfg_de(perfil):
    return dict(
        filepath=os.path.join(ROOT, PERFILES[perfil]),
        alpha_deg=ALPHA, chord=1.0, v0x=1.0, v0y=0.0, rho=1.0,
        nu=1.0 / RE, CFL=0.5, dx_min=DX, iteraciones=ITERS,
        turb_model="sa", wall_treatment="consistent",
        advection_scheme="maccormack", min_te_height_factor=1.0,
        transition_model="sa_bc", freestream_Tu=0.1,
        wake_refinement_mode="long_fine_x",
        mg_niveles_max=2, mg_max_outer=2, mg_cycles_per_outer=3,
        divergencia=0.02,
        stop_on_clcd_convergence=False, stop_on_convergence=False,
        guardado=GUARDADO,
        save_frames=False, graficos=False, live_view=False, mostrar_malla=False,
        dump_fields_dir=os.path.join(OUT, perfil, "campos_refinado"),
        dump_fields_dir_grueso=os.path.join(OUT, perfil, "campos_completo"),
        dump_fields_cada=DUMP_CADA,
        dump_fields_max_nx=900,
        dump_fields_max_nx_grueso=1100,
        save_frame_refined_xlim=XLIM_REF,
        save_frame_refined_ylim=YLIM_REF,
        **DOMINIO,
    )


def resumen(vec, descartar=0.30):
    arr = np.asarray(cp.asnumpy(vec), dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    cola = arr[int(descartar * arr.size):]
    return float(np.mean(cola)), float(np.std(cola))


def corre(perfil):
    cfg = cfg_de(perfil)
    d = os.path.join(OUT, perfil)
    os.makedirs(d, exist_ok=True)
    for k in ("dump_fields_dir", "dump_fields_dir_grueso"):
        os.makedirs(cfg[k], exist_ok=True)

    opt_solver.reset()
    opt_solver.enable(*FLAGS)
    assert opt_solver.active() == sorted(FLAGS), opt_solver.active()

    with open(os.path.join(d, "run_config.json"), "w") as f:
        json.dump({**{k: v for k, v in cfg.items()}, "optimizaciones": list(FLAGS)},
                  f, indent=2, default=str)

    log(f"{perfil}: Re={RE:.0e} alpha={ALPHA} dx={DX} iters={ITERS} "
        f"dump cada {DUMP_CADA} -> {N_FRAMES} frames x 2 ventanas")
    t0 = time.time()
    mesh = sim_main(**cfg)
    wall = time.time() - t0

    cl, cl_s = resumen(mesh.clvector)
    cd, cd_s = resumen(mesh.cdvector)
    meta = {"perfil": perfil, "Re": RE, "alpha": ALPHA, "dx": DX,
            "iteraciones": ITERS, "dump_cada": DUMP_CADA,
            "cl": cl, "cl_std": cl_s, "cd": cd, "cd_std": cd_s,
            "ld": cl / cd if abs(cd) > 1e-12 else None,
            "wall_s": round(wall, 1), "optimizaciones": list(FLAGS)}
    with open(os.path.join(d, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"{perfil}: hecho en {wall/60:.1f} min  Cl={cl:.4f} Cd={cd:.4f} "
        f"L/D={meta['ld']:.2f}")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--perfil", choices=(*PERFILES, "todos"), default="todos")
    ap.add_argument("--solo-render", action="store_true")
    args = ap.parse_args()

    perfiles = list(PERFILES) if args.perfil == "todos" else [args.perfil]
    if not args.solo_render:
        for p in perfiles:
            corre(p)

    import render_videos_largos as rv
    rv.render(perfiles, fps=FPS)


if __name__ == "__main__":
    main()

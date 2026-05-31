"""
script_barrido_conditions.py
=============================
Barrido sistemático de condiciones de vuelo para recolección de datos de IA.

Para cada combinación (perfil, alpha, velocidad) ejecuta el simulador CFD
y guarda los resultados en el mismo JSONL que usa el GA (aprendizaje_ML.jsonl).

Grid de condiciones:
  alphas     = [0, 2, 4, 6, 8, 10] grados
  velocidades = [0.5, 1, 2, 5, 10] m/s  → Re varía proporcional (chord=1, nu=1e-5)
  perfiles    = AG24, GM15, NACA_0012, OPTIMO_PARCIAL_2604_2 + extras de GA

Checkpoint automático:
  Guarda progreso en data/barrido_checkpoint.json tras cada simulación exitosa.
  Si se relanza con los mismos parámetros, reanuda desde el último punto completado.
  Casos fallidos NO se guardan en checkpoint y se reintentan en la siguiente run.

Uso:
    python scripts/script_barrido_conditions.py
    python scripts/script_barrido_conditions.py --perfiles profiles/AG24 profiles/GM15
    python scripts/script_barrido_conditions.py --alphas 0 4 8 --velocidades 1 5
    python scripts/script_barrido_conditions.py --dry-run   (muestra plan sin simular)
    python scripts/script_barrido_conditions.py --reset-checkpoint  (borra checkpoint y empieza de cero)
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cupy as cp
import numpy as np

import Simulador2D
from scripts.sim_defaults import PROJECTION_DEFAULTS, add_force_consistent_metrics, finite_or_nan

# Importar helpers de RunGA para parametrización geométrica y logging
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from RunGA import (
    DataLoggerML,
    cargar_perfil,
    descomponer_camber_espesor,
    estimar_radio_le,
    extraer_parametrizacion,
)

# ==========================================
# CONFIGURACIÓN
# ==========================================
ALPHAS_DEFAULT = [0, 2, 4, 6, 8, 10]
VELOCIDADES_DEFAULT = [0.5, 1.0, 2.0, 5.0, 10.0]
RHO = 1.0
NU = 1e-5
CHORD = 1.0

PERFILES_DEFAULT = [
    'profiles/AG24',
    'profiles/GM15',
    'profiles/NACA_0012',
    'profiles/OPTIMO_PARCIAL_2604_2.dat',
]

SIM_PARAMS_BASE = {
    'Lx': 12, 'Ly': 8,
    'CFL': 0.5,
    'iteraciones': 2000,
    'chord': CHORD,
    'dx_min': 0.001,
    'factor_expansion': 1.10,
    'ancho_zona_fina_x': 1.2,
    'ancho_zona_fina_y': 1.0,
    'graficos': False,
    'save_frames': False,
    'usar_wale': True,
    'stop_on_convergence': False,
    'live_view': False,
    'mostrar_malla': False,
    'mg_modo_turbo': True,
    'divergencia': 1e-1,
    **PROJECTION_DEFAULTS,
}

DESCARTE_FRAC = 0.20
OUTPUT_JSONL = 'data/aprendizaje_ML.jsonl'
CHECKPOINT_PATH = 'data/barrido_checkpoint.json'


# ==========================================
# CHECKPOINT
# ==========================================
def _checkpoint_key(perfil_path, v0x, alpha):
    """Clave única por caso: nombre de archivo (sin extensión) + condiciones."""
    nombre = os.path.splitext(os.path.basename(perfil_path))[0]
    return f"{nombre}|v{v0x:.4g}|a{alpha:.4g}"


def cargar_checkpoint(checkpoint_path, perfiles, alphas, velocidades):
    """
    Carga checkpoint si existe y los parámetros del grid coinciden.
    Retorna set de claves ya completadas, o set vacío si no hay checkpoint válido.
    """
    if not os.path.exists(checkpoint_path):
        return set()
    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Verificar que el grid es el mismo
        saved_perfiles = sorted(os.path.splitext(os.path.basename(p))[0] for p in data.get('perfiles', []))
        curr_perfiles  = sorted(os.path.splitext(os.path.basename(p))[0] for p in perfiles)
        if (saved_perfiles != curr_perfiles
                or sorted(data.get('alphas', [])) != sorted(alphas)
                or sorted(data.get('velocidades', [])) != sorted(velocidades)):
            print(" [checkpoint] Parámetros distintos → ignorando checkpoint anterior.")
            return set()

        completados = set(data.get('completados', []))
        print(f" [checkpoint] Reanudando: {len(completados)} casos ya completados.")
        return completados

    except Exception as e:
        print(f" [checkpoint] Error al leer checkpoint ({e}) → empezando de cero.")
        return set()


def guardar_checkpoint(checkpoint_path, perfiles, alphas, velocidades, completados):
    """Escribe checkpoint de forma atómica (write-then-rename)."""
    os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else '.', exist_ok=True)
    tmp = checkpoint_path + '.tmp'
    data = {
        'perfiles': [os.path.splitext(os.path.basename(p))[0] for p in perfiles],
        'alphas': alphas,
        'velocidades': velocidades,
        'completados': sorted(completados),
        'last_update': datetime.now().isoformat(),
    }
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, checkpoint_path)  # atómico en Windows y Linux


# ==========================================
# SIMULACIÓN
# ==========================================
def simular_punto(filepath, alpha_deg, v0x, params_base):
    """
    Ejecuta CFD para un punto (perfil, alpha, velocidad).
    Retorna dict con Cl, Cd, L/D y stats de convergencia, o None si falla.
    """
    mesh = None
    try:
        sim_params = dict(params_base)
        sim_params.update({
            'filepath': filepath,
            'alpha_deg': float(alpha_deg),
            'v0x': float(v0x),
            'v0y': 0.0,
            'rho': RHO,
            'nu': NU,
        })

        mesh = Simulador2D.main(**sim_params)

        if mesh.cdvector is None or len(mesh.cdvector) == 0:
            return None

        n = len(mesh.cdvector)
        i0 = max(1, int(n * DESCARTE_FRAC))

        cd_arr = cp.asnumpy(mesh.cdvector[i0:]).astype(float)
        cl_arr = cp.asnumpy(mesh.clvector[i0:]).astype(float)

        cd_val = float(np.mean(cd_arr))
        cl_val = float(np.mean(cl_arr))
        cd_std = float(np.std(cd_arr))
        cl_std = float(np.std(cl_arr))

        if not (np.isfinite(cd_val) and np.isfinite(cl_val)):
            return None

        ld = cl_val / cd_val if abs(cd_val) > 1e-6 else 0.0
        metric_row = {"Cl_final": cl_val}
        add_force_consistent_metrics(metric_row, mesh, sim_params)
        cl_force = finite_or_nan(metric_row.get("Cl_from_Cp_force_consistent"))
        cl_dataset = cl_force if np.isfinite(cl_force) else cl_val
        ld_dataset = cl_dataset / cd_val if abs(cd_val) > 1e-6 else 0.0

        return {
            'cl': round(cl_dataset, 6),
            'cl_raw': round(cl_val, 6),
            'cl_force_consistent': round(cl_force, 6) if np.isfinite(cl_force) else float("nan"),
            'cd': round(cd_val, 6),
            'ld': round(ld_dataset, 4),
            'ld_raw': round(ld, 4),
            'cl_std': round(cl_std, 6),
            'cd_std': round(cd_std, 6),
            'n_samples': int(len(cd_arr)),
        }

    except Exception as e:
        print(f"   [!] Error: {e}")
        return None

    finally:
        del mesh
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()


# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Barrido sistemático condiciones CFD para dataset IA')
    parser.add_argument('--perfiles', nargs='+', default=None, help='Rutas a archivos .dat de perfiles')
    parser.add_argument('--alphas', nargs='+', type=float, default=None, help='Ángulos de ataque (grados)')
    parser.add_argument('--velocidades', nargs='+', type=float, default=None, help='Velocidades m/s')
    parser.add_argument('--output', default=OUTPUT_JSONL, help='Ruta del archivo JSONL de salida')
    parser.add_argument('--checkpoint', default=CHECKPOINT_PATH, help='Ruta del archivo de checkpoint')
    parser.add_argument('--dry-run', action='store_true', help='Mostrar plan sin simular')
    parser.add_argument('--reset-checkpoint', action='store_true', help='Borrar checkpoint y empezar de cero')
    args = parser.parse_args()

    perfiles = args.perfiles or PERFILES_DEFAULT
    alphas = args.alphas or ALPHAS_DEFAULT
    velocidades = args.velocidades or VELOCIDADES_DEFAULT
    output_path = args.output
    checkpoint_path = args.checkpoint

    # Filtrar perfiles que existen
    perfiles_validos = [(p, os.path.exists(p)) for p in perfiles]
    perfiles = [p for p, ok in perfiles_validos if ok]
    for p, ok in perfiles_validos:
        if not ok:
            print(f" [!] Perfil no encontrado, ignorado: {p}")

    if not perfiles:
        print("ERROR: No hay perfiles válidos.")
        return

    total = len(perfiles) * len(alphas) * len(velocidades)
    print("\n" + "=" * 65)
    print(" BARRIDO SISTEMÁTICO DE CONDICIONES")
    print("=" * 65)
    print(f" Perfiles:    {len(perfiles)}")
    print(f" Alphas:      {alphas}")
    print(f" Velocidades: {velocidades} m/s")
    print(f" Total casos: {total}")
    print(f" Salida:      {output_path}")
    print(f" Checkpoint:  {checkpoint_path}")
    print("=" * 65)

    if args.dry_run:
        print("\n[DRY RUN] Plan de simulaciones:")
        for pf in perfiles:
            for v in velocidades:
                Re = v * CHORD / NU
                for a in alphas:
                    print(f"  {os.path.basename(pf):30s}  v={v:5.1f}  Re={Re:.0f}  alpha={a:+.1f}°")
        return

    # Checkpoint: reset si se pide, si no cargar progreso anterior
    if args.reset_checkpoint and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(" [checkpoint] Borrado. Empezando de cero.")
    completados = cargar_checkpoint(checkpoint_path, perfiles, alphas, velocidades)

    # Crear carpeta de salida si no existe
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    logger = DataLoggerML(output_path)

    t_total = time.time()
    n_ok = 0
    n_skip = 0
    n_fail = 0
    caso = 0

    for filepath in perfiles:
        puntos, _, le_idx, _ = cargar_perfil(filepath)
        if len(puntos) == 0:
            print(f" [!] No se pudo cargar {filepath}")
            continue

        param_geom = extraer_parametrizacion(puntos, le_idx)
        nombre_perfil = os.path.splitext(os.path.basename(filepath))[0]

        for v0x in velocidades:
            Re = v0x * CHORD / NU

            for alpha in alphas:
                caso += 1
                key = _checkpoint_key(filepath, v0x, alpha)

                # Saltar si ya completado en run anterior
                if key in completados:
                    n_skip += 1
                    print(f"\n[{caso}/{total}] {nombre_perfil}  v={v0x}  Re={Re:.0f}  alpha={alpha:+.1f}°  [SKIP]")
                    continue

                print(f"\n[{caso}/{total}] {nombre_perfil}  v={v0x}  Re={Re:.0f}  alpha={alpha:+.1f}°",
                      end='  ', flush=True)

                t0 = time.time()
                res = simular_punto(filepath, alpha, v0x, SIM_PARAMS_BASE)
                elapsed = time.time() - t0

                if res is None:
                    print("FALLO")
                    n_fail += 1
                    continue

                print(f"Cl={res['cl']:.4f}  Cd={res['cd']:.5f}  L/D={res['ld']:.2f}  ({elapsed:.0f}s)")
                n_ok += 1

                condiciones = {
                    'v0x': v0x,
                    'alpha_base': alpha,
                    'angulos_evaluados': [alpha],
                    'chord': CHORD,
                    'Re': round(Re, 1),
                    'rho': RHO,
                    'nu': NU,
                    'dx_min': SIM_PARAMS_BASE['dx_min'],
                    'iteraciones_cfd': SIM_PARAMS_BASE['iteraciones'],
                    'fitness_modo': 'mean',
                }

                resultados_limpios = {
                    f"{alpha:.1f}": {'cl': res['cl'], 'cd': res['cd'], 'ld': res['ld']}
                }

                conv = {
                    'cl_std_mean': res['cl_std'],
                    'cd_std_mean': res['cd_std'],
                    'n_samples': res['n_samples'],
                }

                logger.registrar(
                    perfil_puntos=puntos,
                    condiciones=condiciones,
                    resultados_por_angulo=resultados_limpios,
                    metadata={
                        'generacion': -1,
                        'individuo': -1,
                        'fitness': res['ld'],
                        'perfil_base': filepath,
                        'multi_angulo': False,
                        'fuente': 'barrido_conditions',
                        'parametrizacion': param_geom,
                        'convergencia': conv,
                    }
                )

                # Marcar como completado y persistir checkpoint
                completados.add(key)
                guardar_checkpoint(checkpoint_path, perfiles, alphas, velocidades, completados)

    t_elapsed = time.time() - t_total
    print(f"\n{'=' * 65}")
    print(f" BARRIDO COMPLETADO")
    print(f" OK: {n_ok}  |  Skip: {n_skip}  |  Fallos: {n_fail}  |  Tiempo: {t_elapsed/60:.1f} min")
    print(f" Registros en dataset: {logger.count}")
    print(f" Archivo: {output_path}")
    if n_ok + n_fail + n_skip == total and n_fail == 0:
        # Todos completados sin fallos: limpiar checkpoint
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f" [checkpoint] Barrido completo — checkpoint eliminado.")
    print('=' * 65)


if __name__ == '__main__':
    main()

"""
Estudio paramétrico del simulador 2D
- Ejecuta Simulador2D.main para cada combinación de (alpha_deg, dx_grueso).
- Guarda CD, CL y eficiencia (CL/CD) medios y finales por ángulo de ataque.
- Genera gráficos comparativos: una línea por dx_grueso.

Uso:
    python study_simulador.py
"""

import os
import csv
import time
import math
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import Simulador2D

# -----------------------------
# Configuración
# -----------------------------
CONFIG = {
    'iteraciones': 5000,
    'guardado': 50,
    'ALPHAS': [0, 1, 2,3, 4,5, 6,7, 8,9, 10,11,12],
    'DX_GRUESOS': [0.004,0.003,0.002,0.0018],
    'dx_fino': 0.0015, 
    'v0x': 5,
    'CFL': 0.5,
    'filepath': 'AG24',
    'chord': 1.0,
    'Lx': 7,
    'Ly': 6,
    'cx': 1,
    'rho': 1.225,
    'nu': 1.5e-5,
    'divergencia': 1e-1,
    'descartar_transitorio_frac': 0.10,
    'output_csv': 'estudio_alpha_resultados_AG24.csv',
    'out_png_prefix': 'estudio_alpha',
    'usar_wale': False,
    'usar_viscosidad_estela': False,
    'stop_on_convergence': False,
}


def ejecutar_simulacion(alpha_deg, dx_grueso):
    """Ejecuta una simulación y devuelve (mesh_fina, mesh_gruesa, geometria)."""
    mesh_fina, mesh_gruesa, geometria = Simulador2D.main(
        filepath=CONFIG['filepath'],
        iteraciones=CONFIG['iteraciones'],
        guardado=CONFIG['guardado'],
        v0x=CONFIG['v0x'],
        CFL=CONFIG['CFL'],
        alpha_deg=alpha_deg,
        chord=CONFIG['chord'],
        dx_fino=CONFIG['dx_fino'],
        dx_grueso=dx_grueso,
        Lx=CONFIG['Lx'],
        Ly=CONFIG['Ly'],
        cx=CONFIG['cx'],
        divergencia=CONFIG['divergencia'],
        graficos=False,
        usar_wale=CONFIG['usar_wale'],
        usar_viscosidad_estela=CONFIG['usar_viscosidad_estela'],
        stop_on_convergence=CONFIG['stop_on_convergence'],
    )
    return mesh_fina, mesh_gruesa, geometria


def calcular_coeficientes(mesh_gruesa, desc_frac=0.10):
    """Calcula CD, CL medios (descartando transitorio) y finales desde cdvector/clvector."""
    cdvec = mesh_gruesa.cdvector
    clvec = mesh_gruesa.clvector
    N = int(cdvec.size)
    inicio = max(1, int(math.floor(N * desc_frac)))

    cd_medio = float(cp.mean(cdvec[inicio:]).get())
    cl_medio = float(cp.mean(clvec[inicio:]).get())
    cd_final = float(cdvec[-1].get())
    cl_final = float(clvec[-1].get())

    ef_medio = cl_medio / cd_medio if abs(cd_medio) > 1e-12 else float('nan')
    ef_final = cl_final / cd_final if abs(cd_final) > 1e-12 else float('nan')

    return {
        'cd_medio': cd_medio, 'cl_medio': cl_medio, 'ef_medio': ef_medio,
        'cd_final': cd_final, 'cl_final': cl_final, 'ef_final': ef_final,
    }


def main():
    alphas = CONFIG['ALPHAS']
    dx_gruesos = CONFIG['DX_GRUESOS']

    csv_file = CONFIG['output_csv']
    with open(csv_file, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['dx_grueso', 'alpha_deg',
                         'cd_medio', 'cl_medio', 'ef_medio',
                         'cd_final', 'cl_final', 'ef_final'])

    # Estructura: resultados[dx_grueso] = {'alpha':[], 'cd_medio':[], ...}
    resultados = {}
    for dxg in dx_gruesos:
        resultados[dxg] = {
            'alpha': [], 'cd_medio': [], 'cl_medio': [], 'ef_medio': [],
            'cd_final': [], 'cl_final': [], 'ef_final': [],
        }

    total_sims = len(dx_gruesos) * len(alphas)
    sim_num = 0

    for dxg in dx_gruesos:
        for alpha in alphas:
            sim_num += 1
            print(f"\n{'='*60}")
            print(f"  Simulación {sim_num}/{total_sims}: dx_grueso={dxg}, alpha={alpha}°")
            print(f"{'='*60}")
            t0 = time.time()

            mesh_fina = mesh_gruesa = geo = None
            try:
                mesh_fina, mesh_gruesa, geo = ejecutar_simulacion(alpha, dxg)
                coefs = calcular_coeficientes(mesh_gruesa, CONFIG['descartar_transitorio_frac'])
            except Exception as e:
                print(f"  ERROR: {e}")
                coefs = {k: float('nan') for k in
                         ['cd_medio','cl_medio','ef_medio','cd_final','cl_final','ef_final']}

            for k, v in coefs.items():
                resultados[dxg][k].append(v)
            resultados[dxg]['alpha'].append(alpha)

            with open(csv_file, 'a', newline='') as cf:
                writer = csv.writer(cf)
                writer.writerow([dxg, alpha,
                                 coefs['cd_medio'], coefs['cl_medio'], coefs['ef_medio'],
                                 coefs['cd_final'], coefs['cl_final'], coefs['ef_final']])

            dt_elapsed = time.time() - t0
            print(f"  Cd_medio={coefs['cd_medio']:.6f}  Cl_medio={coefs['cl_medio']:.6f}  "
                  f"Ef_medio={coefs['ef_medio']:.3f}")
            print(f"  Cd_final={coefs['cd_final']:.6f}  Cl_final={coefs['cl_final']:.6f}  "
                  f"Ef_final={coefs['ef_final']:.3f}")
            print(f"  Tiempo: {dt_elapsed:.1f}s")

            # Liberar memoria GPU entre simulaciones
            del mesh_fina, mesh_gruesa, geo
            cp.get_default_memory_pool().free_all_blocks()

    # -----------------------------------------------------------------
    # Gráficos comparativos (una línea por dx_grueso)
    # -----------------------------------------------------------------
    prefijo = CONFIG['out_png_prefix']

    grupos = [
        ('cd_medio', 'Cd', 'Cd'),
        ('cl_medio', 'Cl', 'Cl'),
        ('ef_medio', 'Eficiencia', 'Cl/Cd'),
    ]

    for clave, titulo_grupo, ylabel in grupos:
        plt.figure(figsize=(9, 5))
        for dxg in dx_gruesos:
            plt.plot(resultados[dxg]['alpha'], resultados[dxg][clave],
                     '-o', label=f'dx_g={dxg}', markersize=5)
        plt.xlabel('Ángulo de ataque α [°]')
        plt.ylabel(ylabel)
        plt.title(f'{titulo_grupo} vs ángulo de ataque')
        plt.legend()
        plt.grid(True, alpha=0.3)
        out = f"{prefijo}_{titulo_grupo}.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"Guardado {out}")

    print(f"\nResultados en CSV: {csv_file}")
    print("Estudio completado.")


if __name__ == '__main__':
    main()

"""
build_dataset.py
=================
Procesa aprendizaje_ML.jsonl → dataset_ia.csv con features geométricas.

Columnas de salida:
  Condiciones: id, fuente, v0x, alpha, Re, chord
  Geometría:   camber_max, camber_pos, espesor_max, espesor_pos, le_radius, te_gap
  Targets:     cl, cd, ld
  Calidad:     cl_std, cd_std, n_samples

Los campos de parametrización se extraen del campo 'metadata.parametrizacion'
si existe (registros nuevos), o se calculan al vuelo desde la geometría (registros
antiguos sin ese campo).

Uso:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --input data/aprendizaje_ML.jsonl --output data/dataset_ia.csv
    python scripts/build_dataset.py --stats   (solo imprime estadísticas, no guarda)
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from RunGA import (
    cargar_perfil,
    descomponer_camber_espesor,
    estimar_radio_le,
    extraer_parametrizacion,
)

INPUT_DEFAULT = 'data/aprendizaje_ML.jsonl'
OUTPUT_DEFAULT = 'data/dataset_ia.csv'

COLUMNAS = [
    'id', 'fuente', 'perfil_base',
    'v0x', 'alpha', 'Re', 'chord',
    'camber_max', 'camber_pos', 'espesor_max', 'espesor_pos', 'le_radius', 'te_gap',
    'cl', 'cd', 'ld',
    'cl_std', 'cd_std', 'n_samples',
]


def cargar_jsonl(path):
    registros = []
    with open(path, 'r', encoding='utf-8') as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                try:
                    registros.append(json.loads(linea))
                except json.JSONDecodeError:
                    continue
    return registros


def extraer_fila(registro):
    """
    Extrae una fila del dataset a partir de un registro JSONL.
    Retorna dict con COLUMNAS o None si el registro es inválido.
    """
    try:
        meta = registro.get('metadata', {})
        cond = registro.get('condiciones', {})
        resultados = registro.get('resultados', {})

        if not resultados:
            return None

        # Tomar primer (o único) ángulo evaluado
        ang_key = sorted(resultados.keys(), key=float)[0]
        res = resultados[ang_key]

        cl = res.get('cl')
        cd = res.get('cd')
        ld = res.get('ld')

        if cl is None or cd is None or ld is None:
            return None
        if not (np.isfinite(cl) and np.isfinite(cd) and np.isfinite(ld)):
            return None
        if ld <= 0:
            return None

        # Condiciones de vuelo
        v0x = float(cond.get('v0x', 0))
        alpha = float(cond.get('alpha_base', float(ang_key)))
        Re = float(cond.get('Re', 0))
        chord = float(cond.get('chord', 1.0))

        # Parametrización geométrica — usar campo precalculado si existe
        param = meta.get('parametrizacion', {})
        if param and all(param.get(k) is not None for k in
                         ('camber_max', 'camber_pos', 'espesor_max', 'espesor_pos', 'le_radius', 'te_gap')):
            camber_max = param['camber_max']
            camber_pos = param['camber_pos']
            espesor_max = param['espesor_max']
            espesor_pos = param['espesor_pos']
            le_radius = param['le_radius']
            te_gap = param['te_gap']
        else:
            # Calcular al vuelo desde geometría
            perfil_raw = registro.get('perfil')
            if not perfil_raw:
                return None
            puntos = np.array(perfil_raw, dtype=float)
            if len(puntos) < 4:
                return None
            le_idx = int(np.argmin(puntos[:, 0]))
            p = extraer_parametrizacion(puntos, le_idx)
            if any(p.get(k) is None for k in
                   ('camber_max', 'camber_pos', 'espesor_max', 'espesor_pos', 'le_radius', 'te_gap')):
                return None
            camber_max = p['camber_max']
            camber_pos = p['camber_pos']
            espesor_max = p['espesor_max']
            espesor_pos = p['espesor_pos']
            le_radius = p['le_radius']
            te_gap = p['te_gap']

        # Stats de convergencia
        conv = meta.get('convergencia', {})
        cl_std = conv.get('cl_std_mean')
        cd_std = conv.get('cd_std_mean')
        n_samples = conv.get('n_samples')

        # Retrocompatibilidad: stats inline en resultados (format antiguo)
        if cl_std is None:
            cl_std = res.get('cl_std')
        if cd_std is None:
            cd_std = res.get('cd_std')
        if n_samples is None:
            n_samples = res.get('n_samples')

        fuente = meta.get('fuente', 'ga')
        perfil_base = meta.get('perfil_base', '')

        return {
            'id': registro.get('id', ''),
            'fuente': fuente,
            'perfil_base': perfil_base,
            'v0x': v0x,
            'alpha': alpha,
            'Re': Re,
            'chord': chord,
            'camber_max': camber_max,
            'camber_pos': camber_pos,
            'espesor_max': espesor_max,
            'espesor_pos': espesor_pos,
            'le_radius': le_radius,
            'te_gap': te_gap,
            'cl': cl,
            'cd': cd,
            'ld': ld,
            'cl_std': cl_std,
            'cd_std': cd_std,
            'n_samples': n_samples,
        }

    except Exception as e:
        return None


def imprimir_stats(filas):
    n = len(filas)
    print(f"\n{'=' * 55}")
    print(f" ESTADÍSTICAS DEL DATASET  ({n} filas válidas)")
    print('=' * 55)

    for col in ('v0x', 'alpha', 'Re', 'cl', 'cd', 'ld',
                'camber_max', 'espesor_max', 'le_radius', 'te_gap'):
        vals = [f[col] for f in filas if f.get(col) is not None and np.isfinite(f[col])]
        if vals:
            print(f"  {col:15s}: min={min(vals):+.4f}  max={max(vals):+.4f}  "
                  f"mean={np.mean(vals):+.4f}  std={np.std(vals):.4f}  n={len(vals)}")

    fuentes = Counter(f['fuente'] for f in filas)
    print(f"\n  Fuente:  {dict(fuentes)}")

    bases = Counter(os.path.basename(str(f['perfil_base'])) for f in filas)
    print(f"  Perfiles base: {dict(bases)}")

    alphas = sorted(set(f['alpha'] for f in filas))
    vels = sorted(set(f['v0x'] for f in filas))
    print(f"\n  Alphas únicos:      {alphas}")
    print(f"  Velocidades únicas: {vels}")
    print('=' * 55)


def main():
    parser = argparse.ArgumentParser(description='Construye dataset IA desde aprendizaje_ML.jsonl')
    parser.add_argument('--input', default=INPUT_DEFAULT)
    parser.add_argument('--output', default=OUTPUT_DEFAULT)
    parser.add_argument('--stats', action='store_true', help='Solo mostrar estadísticas')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: No se encontró {args.input}")
        return

    print(f" Cargando {args.input}...")
    registros = cargar_jsonl(args.input)
    print(f" {len(registros)} registros cargados.")

    print(" Extrayendo features...")
    filas = []
    n_skip = 0
    for reg in registros:
        fila = extraer_fila(reg)
        if fila is not None:
            filas.append(fila)
        else:
            n_skip += 1

    print(f" {len(filas)} filas válidas  |  {n_skip} descartadas")

    imprimir_stats(filas)

    if args.stats:
        return

    if not filas:
        print("ERROR: 0 filas válidas. No se genera CSV.")
        return

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(filas)

    print(f"\n Dataset guardado: {args.output}  ({len(filas)} filas, {len(COLUMNAS)} columnas)")


if __name__ == '__main__':
    main()

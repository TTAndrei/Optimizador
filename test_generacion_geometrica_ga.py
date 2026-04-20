"""
Prueba rápida de robustez geométrica del generador de hijos del GA.

No ejecuta CFD. Evalúa solo la creación de hijos con cruce + mutación
+ validación geométrica para detectar fallos de integridad física.

Uso:
    python test_generacion_geometrica_ga.py
    python test_generacion_geometrica_ga.py --muestras 800 --modo parametrica
"""

import argparse
import json
import os
import random
from collections import Counter

import numpy as np

import RunGA


class PadreDummy:
    """Objeto mínimo para reutilizar cruce() sin tocar el GA principal."""

    def __init__(self, genes, fitness):
        self.genes = genes
        self.fitness = fitness


def construir_pool_inicial(base, le_idx, te_indices, restricciones, config, n_pool):
    """Genera un conjunto inicial de padres geométricamente válidos."""
    pool = []
    intentos = 0
    max_intentos = max(n_pool * 25, int(config.get("max_intentos_geometria", 120)))

    while len(pool) < n_pool and intentos < max_intentos:
        genes, valido, _, _ = RunGA.mutar_y_validar_hijo(
            base.copy(),
            le_idx,
            te_indices,
            restricciones,
            config,
            aplicar_mutacion=True,
        )
        intentos += 1

        if not valido:
            continue

        pool.append(PadreDummy(genes=genes, fitness=random.random()))

    if not pool:
        pool.append(PadreDummy(genes=base.copy(), fitness=1.0))

    return pool, intentos


def ejecutar_prueba(muestras, semilla, modo, pool_size, output_json):
    random.seed(semilla)
    np.random.seed(semilla)

    config = dict(RunGA.CONFIG)
    config["modo_mutacion"] = modo
    config["reportar_diagnostico_geometria"] = False

    puntos, _, le_idx, te_indices = RunGA.cargar_perfil(config["archivo_base"])
    if len(puntos) == 0:
        raise RuntimeError("No se pudo cargar el perfil base.")

    restricciones = RunGA.construir_restricciones_geometricas(puntos, le_idx, config)
    pool, intentos_pool = construir_pool_inicial(
        puntos, le_idx, te_indices, restricciones, config, n_pool=pool_size
    )

    stats_modo = Counter()
    stats_fallo = Counter()

    n_validos = 0
    n_invalidos = 0

    min_te_gap = float("inf")
    max_te_gap = float("-inf")
    min_espesor = float("inf")
    min_radio_le = float("inf")

    for _ in range(muestras):
        if len(pool) >= 2:
            padre1, padre2 = random.sample(pool, 2)
        else:
            padre1 = pool[0]
            padre2 = pool[0]

        genes_hijo = RunGA.cruce(padre1, padre2, le_idx)
        aplicar_mutacion = random.random() < float(config["prob_mutacion"])

        genes_hijo, valido, modo_usado, diag = RunGA.mutar_y_validar_hijo(
            genes_hijo,
            le_idx,
            te_indices,
            restricciones,
            config,
            aplicar_mutacion=aplicar_mutacion,
        )

        stats_modo[modo_usado] += 1

        if not valido:
            n_invalidos += 1
            motivo = diag.get("motivo", "invalido_sin_motivo") if isinstance(diag, dict) else "invalido_sin_motivo"
            stats_fallo[motivo] += 1
            continue

        # Doble check defensivo para métricas consistentes.
        ok, diag_ok = RunGA.validar_geometria_perfil(
            genes_hijo, le_idx, restricciones, config
        )
        if not ok:
            n_invalidos += 1
            motivo = diag_ok.get("motivo", "fallo_post_validacion")
            stats_fallo[motivo] += 1
            continue

        n_validos += 1

        min_te_gap = min(min_te_gap, float(diag_ok["te_gap"]))
        max_te_gap = max(max_te_gap, float(diag_ok["te_gap"]))
        min_espesor = min(min_espesor, float(diag_ok["espesor_min"]))
        min_radio_le = min(min_radio_le, float(diag_ok["radio_le"]))

        # Mantiene diversidad de padres para testear cruces realistas.
        pool.append(PadreDummy(genes=genes_hijo, fitness=random.random()))
        if len(pool) > max(pool_size * 2, 40):
            del pool[random.randrange(len(pool))]

    resumen = {
        "muestras": int(muestras),
        "semilla": int(semilla),
        "modo": modo,
        "pool_size_inicial": int(pool_size),
        "intentos_pool_inicial": int(intentos_pool),
        "validos": int(n_validos),
        "invalidos": int(n_invalidos),
        "tasa_validez": (float(n_validos) / float(muestras)) if muestras > 0 else 0.0,
        "modos_usados": dict(stats_modo),
        "motivos_fallo": dict(stats_fallo),
        "restricciones": restricciones,
        "metricas_validos": {
            "te_gap_min": None if n_validos == 0 else float(min_te_gap),
            "te_gap_max": None if n_validos == 0 else float(max_te_gap),
            "espesor_min_global": None if n_validos == 0 else float(min_espesor),
            "radio_le_min": None if n_validos == 0 else float(min_radio_le),
        },
    }

    print("\n=== Test Geometrico GA ===")
    print(f"Modo: {modo}")
    print(f"Muestras: {muestras}")
    print(f"Validos: {n_validos} | Invalidos: {n_invalidos}")
    print(f"Tasa validez: {resumen['tasa_validez']:.3f}")

    if n_validos > 0:
        print(
            "TE gap [min, max]: "
            f"[{resumen['metricas_validos']['te_gap_min']:.6f}, "
            f"{resumen['metricas_validos']['te_gap_max']:.6f}]"
        )
        print(f"Espesor minimo global observado: {resumen['metricas_validos']['espesor_min_global']:.6f}")
        print(f"Radio LE minimo observado: {resumen['metricas_validos']['radio_le_min']:.6f}")

    if stats_fallo:
        print("Motivos de fallo:")
        for motivo, conteo in sorted(stats_fallo.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  - {motivo}: {conteo}")

    if output_json:
        out_dir = os.path.dirname(output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(resumen, f, indent=2, ensure_ascii=False)
        print(f"Resumen JSON: {output_json}")

    return resumen


def main():
    parser = argparse.ArgumentParser(description="Prueba de robustez geométrica del GA")
    parser.add_argument("--muestras", type=int, default=500, help="Cantidad de hijos a generar")
    parser.add_argument("--semilla", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument(
        "--modo",
        type=str,
        choices=["parametrica", "legacy"],
        default="parametrica",
        help="Modo de mutación a probar",
    )
    parser.add_argument("--pool-size", type=int, default=24, help="Tamaño inicial de pool de padres")
    parser.add_argument(
        "--output-json",
        type=str,
        default=os.path.join("resultados_ga", "diagnostico_generacion_geometrica.json"),
        help="Ruta de salida para resumen JSON",
    )

    args = parser.parse_args()
    ejecutar_prueba(
        muestras=args.muestras,
        semilla=args.semilla,
        modo=args.modo,
        pool_size=args.pool_size,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()

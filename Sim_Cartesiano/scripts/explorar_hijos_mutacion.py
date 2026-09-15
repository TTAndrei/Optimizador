"""
Explorador de hijos del GA y sensibilidad a parámetros de mutación.

Genera hijos sin CFD para distintos escenarios de parámetros, guarda muestras
visuales de perfiles y exporta rangos numéricos de características geométricas.

Uso:
    python explorar_hijos_mutacion.py
    python explorar_hijos_mutacion.py --hijos-validos 80 --muestras-plot 8
"""

import argparse
import csv
import json
import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

import RunGA


class PadreDummy:
    """Padre mínimo para reutilizar cruce() fuera del loop principal del GA."""

    def __init__(self, genes, fitness):
        self.genes = genes
        self.fitness = fitness


def construir_pool_inicial(base, le_idx, te_indices, restricciones, config, n_pool):
    """Genera padres válidos para iniciar cruces."""
    pool = []
    intentos = 0
    max_intentos = max(n_pool * 30, int(config.get("max_intentos_geometria", 120)))

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

    return pool


def area_poligono(puntos):
    """Área absoluta de polígono por fórmula de shoelace."""
    x = puntos[:, 0]
    y = puntos[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def extraer_metricas(genes, base, le_idx, restricciones):
    """Calcula características geométricas comparables entre hijos."""
    x_common, camber, espesor = RunGA.descomponer_camber_espesor(
        genes, le_idx, n_muestras=restricciones["n_muestras"]
    )
    if x_common is None:
        return None

    y_te_up = float(genes[0, 1])
    y_te_lo = float(genes[-1, 1])

    delta_y = genes[:, 1] - base[:, 1]

    return {
        "te_gap": abs(y_te_up - y_te_lo),
        "te_camber": 0.5 * (y_te_up + y_te_lo),
        "le_radius": float(RunGA.estimar_radio_le(genes, le_idx)),
        "thickness_min": float(np.min(espesor[1:])) if len(espesor) > 1 else float(espesor[0]),
        "thickness_max": float(np.max(espesor)),
        "thickness_mean": float(np.mean(espesor)),
        "camber_min": float(np.min(camber)),
        "camber_max": float(np.max(camber)),
        "camber_abs_max": float(np.max(np.abs(camber))),
        "delta_y_abs_max": float(np.max(np.abs(delta_y))),
        "delta_y_std": float(np.std(delta_y)),
        "area": float(area_poligono(genes)),
    }


def resumen_percentiles(valores):
    """Entrega min/p10/p50/p90/max para una lista numérica."""
    arr = np.asarray(valores, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def guardar_plot_muestras(path_png, base, muestras, titulo):
    """Guarda superposición base + muestras de hijos."""
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.plot(base[:, 0], base[:, 1], "k--", linewidth=1.2, alpha=0.8, label="Base")

    if muestras:
        for i, genes in enumerate(muestras):
            ax.plot(genes[:, 0], genes[:, 1], linewidth=1.0, alpha=0.75, label=f"Hijo {i + 1}")

    ax.set_title(titulo)
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path_png, dpi=170)
    plt.close(fig)


def generar_hijos_escenario(
    nombre,
    overrides,
    cfg_base,
    base,
    header,
    le_idx,
    te_indices,
    hijos_validos,
    max_intentos,
    muestras_plot,
    out_dir,
):
    """Genera hijos válidos para un escenario y guarda muestras."""
    cfg = dict(cfg_base)
    cfg.update(overrides)
    cfg["reportar_diagnostico_geometria"] = False

    restricciones = RunGA.construir_restricciones_geometricas(base, le_idx, cfg)
    pool = construir_pool_inicial(base, le_idx, te_indices, restricciones, cfg, n_pool=24)

    validos = []
    metricas = []
    stats_modo = Counter()
    fallos = Counter()

    intentos = 0
    while len(validos) < hijos_validos and intentos < max_intentos:
        if len(pool) >= 2:
            padre1, padre2 = random.sample(pool, 2)
        else:
            padre1 = pool[0]
            padre2 = pool[0]

        genes_hijo = RunGA.cruce(padre1, padre2, le_idx)
        aplicar_mut = random.random() < float(cfg["prob_mutacion"])
        genes_hijo, ok, modo_usado, diag = RunGA.mutar_y_validar_hijo(
            genes_hijo,
            le_idx,
            te_indices,
            restricciones,
            cfg,
            aplicar_mutacion=aplicar_mut,
        )
        intentos += 1
        stats_modo[modo_usado] += 1

        if not ok:
            motivo = diag.get("motivo", "invalido") if isinstance(diag, dict) else "invalido"
            fallos[motivo] += 1
            continue

        ok2, diag2 = RunGA.validar_geometria_perfil(genes_hijo, le_idx, restricciones, cfg)
        if not ok2:
            fallos[diag2.get("motivo", "fallo_post")] += 1
            continue

        m = extraer_metricas(genes_hijo, base, le_idx, restricciones)
        if m is None:
            fallos["sin_metricas"] += 1
            continue

        validos.append(genes_hijo)
        metricas.append(m)

        pool.append(PadreDummy(genes=genes_hijo, fitness=random.random()))
        if len(pool) > 56:
            del pool[random.randrange(len(pool))]

    scenario_dir = os.path.join(out_dir, nombre)
    os.makedirs(scenario_dir, exist_ok=True)

    n_save = min(len(validos), muestras_plot)
    muestras_guardadas = []
    for i in range(n_save):
        out_dat = os.path.join(scenario_dir, f"hijo_{i + 1:02d}.dat")
        RunGA.guardar_perfil(out_dat, validos[i], header)
        muestras_guardadas.append(validos[i])

    plot_path = os.path.join(scenario_dir, "muestras_hijos.png")
    guardar_plot_muestras(
        plot_path,
        base,
        muestras_guardadas,
        titulo=f"Escenario: {nombre} | validos={len(validos)}/{intentos}",
    )

    return {
        "nombre": nombre,
        "overrides": overrides,
        "intentos": int(intentos),
        "validos": int(len(validos)),
        "invalidos": int(intentos - len(validos)),
        "tasa_validez": (float(len(validos)) / float(intentos)) if intentos > 0 else 0.0,
        "stats_modo": dict(stats_modo),
        "fallos": dict(fallos),
        "metricas": metricas,
        "plot_path": plot_path,
        "scenario_dir": scenario_dir,
    }


def flatten_resumen_escenario(res):
    """Aplana resumen para CSV con percentiles de métricas."""
    row = {
        "escenario": res["nombre"],
        "intentos": res["intentos"],
        "validos": res["validos"],
        "invalidos": res["invalidos"],
        "tasa_validez": res["tasa_validez"],
    }

    for k, v in sorted(res["overrides"].items()):
        row[f"param_{k}"] = v

    if not res["metricas"]:
        return row

    keys = sorted(res["metricas"][0].keys())
    for key in keys:
        vals = [m[key] for m in res["metricas"]]
        pct = resumen_percentiles(vals)
        for p_name, p_val in pct.items():
            row[f"{key}_{p_name}"] = p_val

    return row


def guardar_csv_dicts(path_csv, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Muestra de hijos y sensibilidad de mutación")
    parser.add_argument("--semilla", type=int, default=42, help="Semilla para reproducibilidad")
    parser.add_argument("--hijos-validos", type=int, default=60, help="Hijos válidos objetivo por escenario")
    parser.add_argument("--max-intentos", type=int, default=320, help="Intentos máximos por escenario")
    parser.add_argument("--muestras-plot", type=int, default=6, help="Muestras guardadas por escenario")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("resultados_ga", "muestras_hijos_mutacion"),
        help="Directorio de salida",
    )

    args = parser.parse_args()

    random.seed(args.semilla)
    np.random.seed(args.semilla)

    cfg_base = dict(RunGA.CONFIG)
    cfg_base["modo_mutacion"] = "parametrica"

    base, header, le_idx, te_indices = RunGA.cargar_perfil(cfg_base["archivo_base"])
    if len(base) == 0:
        raise RuntimeError("No se pudo cargar el perfil base.")

    escenarios = [
        ("baseline", {}),
        ("camber_bajo", {"camber_mut_std": cfg_base["camber_mut_std"] * 0.5}),
        ("camber_alto", {"camber_mut_std": cfg_base["camber_mut_std"] * 1.8}),
        ("espesor_bajo", {"espesor_mut_std": cfg_base["espesor_mut_std"] * 0.5}),
        ("espesor_alto", {"espesor_mut_std": cfg_base["espesor_mut_std"] * 1.8}),
        ("te_camber_bajo", {"te_camber_shift_std": cfg_base["te_camber_shift_std"] * 0.5}),
        ("te_camber_alto", {"te_camber_shift_std": cfg_base["te_camber_shift_std"] * 1.8}),
        ("te_espesor_bajo", {"te_espesor_shift_std": cfg_base["te_espesor_shift_std"] * 0.5}),
        ("te_espesor_alto", {"te_espesor_shift_std": cfg_base["te_espesor_shift_std"] * 1.8}),
        ("prob_mut_baja", {"prob_mutacion": 0.45}),
        ("prob_mut_alta", {"prob_mutacion": 0.98}),
    ]

    os.makedirs(args.output_dir, exist_ok=True)

    resultados = []
    print("\n=== Exploracion de Hijos y Sensibilidad ===")
    print(f"Escenarios: {len(escenarios)}")
    print(f"Objetivo de hijos validos por escenario: {args.hijos_validos}\n")

    for nombre, overrides in escenarios:
        print(f"- Ejecutando escenario: {nombre}")
        res = generar_hijos_escenario(
            nombre=nombre,
            overrides=overrides,
            cfg_base=cfg_base,
            base=base,
            header=header,
            le_idx=le_idx,
            te_indices=te_indices,
            hijos_validos=args.hijos_validos,
            max_intentos=args.max_intentos,
            muestras_plot=args.muestras_plot,
            out_dir=args.output_dir,
        )
        resultados.append(res)
        print(
            f"  intentos={res['intentos']} | validos={res['validos']} | "
            f"tasa={res['tasa_validez']:.3f}"
        )

    rows_csv = [flatten_resumen_escenario(r) for r in resultados]
    out_csv = os.path.join(args.output_dir, "resumen_escenarios.csv")
    guardar_csv_dicts(out_csv, rows_csv)

    metricas_todas = []
    for r in resultados:
        metricas_todas.extend(r["metricas"])

    rango_global = {}
    if metricas_todas:
        for k in sorted(metricas_todas[0].keys()):
            vals = [m[k] for m in metricas_todas]
            rango_global[k] = resumen_percentiles(vals)

    params_info = {
        "camber_mut_std": {
            "base": cfg_base["camber_mut_std"],
            "rango_explorado": [cfg_base["camber_mut_std"] * 0.5, cfg_base["camber_mut_std"] * 1.8],
            "impacto_esperado": "Incrementa/reduce amplitud de curvatura global (camber).",
        },
        "espesor_mut_std": {
            "base": cfg_base["espesor_mut_std"],
            "rango_explorado": [cfg_base["espesor_mut_std"] * 0.5, cfg_base["espesor_mut_std"] * 1.8],
            "impacto_esperado": "Aumenta/disminuye variacion de espesor en el perfil.",
        },
        "te_camber_shift_std": {
            "base": cfg_base["te_camber_shift_std"],
            "rango_explorado": [cfg_base["te_camber_shift_std"] * 0.5, cfg_base["te_camber_shift_std"] * 1.8],
            "impacto_esperado": "Desplaza el camber en la zona del trailing edge (TE y!=0).",
        },
        "te_espesor_shift_std": {
            "base": cfg_base["te_espesor_shift_std"],
            "rango_explorado": [cfg_base["te_espesor_shift_std"] * 0.5, cfg_base["te_espesor_shift_std"] * 1.8],
            "impacto_esperado": "Modula espesor de salida del trailing edge romo.",
        },
        "prob_mutacion": {
            "base": cfg_base["prob_mutacion"],
            "rango_explorado": [0.45, 0.98],
            "impacto_esperado": "Controla frecuencia de hijos mutados frente a casi-clones.",
        },
    }

    resumen = {
        "semilla": args.semilla,
        "hijos_validos_objetivo": args.hijos_validos,
        "escenarios": [
            {
                "nombre": r["nombre"],
                "overrides": r["overrides"],
                "intentos": r["intentos"],
                "validos": r["validos"],
                "invalidos": r["invalidos"],
                "tasa_validez": r["tasa_validez"],
                "stats_modo": r["stats_modo"],
                "fallos": r["fallos"],
                "plot_path": r["plot_path"],
                "scenario_dir": r["scenario_dir"],
            }
            for r in resultados
        ],
        "rango_global_caracteristicas": rango_global,
        "parametros_mutacion": params_info,
        "resumen_csv": out_csv,
    }

    out_json = os.path.join(args.output_dir, "resumen_exploracion.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print("\n=== Archivos generados ===")
    print(f"- {out_json}")
    print(f"- {out_csv}")
    print(f"- Carpetas de muestras por escenario en: {args.output_dir}")


if __name__ == "__main__":
    main()

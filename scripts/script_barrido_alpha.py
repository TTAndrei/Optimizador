"""
Prueba: barrido de angulo de ataque.
Ejecuta main() secuencialmente para alpha = 0,2,4,6,8,10,
con 2000 iteraciones por caso, y compara Cd, Cl, eficiencia, etc.

Salida:
- CSV con resultados por angulo
- JSON con configuracion + resultados
- PNGs de curvas vs alpha
"""

import csv
import json
import math
import os
import sys
import time
from typing import TypedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np


class CommonParams(TypedDict):
    Lx: int
    Ly: int
    cx: int
    cy: int
    CFL: float
    iteraciones: int
    guardado: int
    polar_descarte: float
    divergencia: float
    v0x: int
    v0y: float
    rho: float
    nu: float
    filepath: str
    chord: float
    dx_min: float
    factor_expansion: float
    ancho_zona_fina_x: float
    ancho_zona_fina_y: float
    ratio_max_malla: int
    graficos: bool
    save_frames: bool
    frames_dir_grueso: str
    usar_wale: bool
    stop_on_convergence: bool
    live_view: bool
    mostrar_malla: bool
    mg_modo_turbo: bool


# Parametros comunes para todas las corridas
COMMON: CommonParams = {
    "Lx": 12,
    "Ly": 8,
    "cx": 2,
    "cy": 4,
    "CFL": 0.5,
    "iteraciones": 2000,
    "guardado": 50,
    "polar_descarte": 0.3,
    "divergencia": 1e-1,
    "v0x": 1,
    "v0y": 0.0,
    "rho": 1.0,
    "nu": 1 / 100000,
    "filepath": "profiles/AG24",
    "chord": 1.0,
    "dx_min": 0.001,
    "factor_expansion": 1.10,
    "ancho_zona_fina_x": 1.2,
    "ancho_zona_fina_y": 1.0,
    "ratio_max_malla": 100,
    "graficos": False,
    "save_frames": False,
    "frames_dir_grueso": "",
    "usar_wale": True,
    "stop_on_convergence": False,
    "live_view": False,
    "mostrar_malla": False,
    "mg_modo_turbo_hd": True,
}

# Barrido de alpha: 0 a 10 cada 2 grados
ALPHAS = list(range(-10, 11, 2))
DESCARTE_FRAC = 0.30

OUT_CSV = "barrido_ag24_resultados_bl.csv"
OUT_JSON = "barrido_ag24_resultados_bl.json"
OUT_PNG_PREFIX = "barrido_alpha_bl"


def resumen_vector(vec_gpu, iteraciones, guardado, descartar_frac):
    """Devuelve (mean, final, std, n_muestras) para el tramo estacionario."""
    vec = cp.asnumpy(vec_gpu).astype(float)
    n_esperado = max(1, int(iteraciones // guardado))
    vec = vec[:n_esperado]

    if vec.size == 0:
        return float("nan"), float("nan"), float("nan"), 0

    i0 = int(math.floor(vec.size * descartar_frac))
    i0 = min(max(i0, 0), vec.size - 1)
    tramo = vec[i0:]

    return float(np.mean(tramo)), float(vec[-1]), float(np.std(tramo)), int(tramo.size)


def guardar_csv(path_csv, rows):
    if not rows:
        return
    campos = list(rows[0].keys())
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(rows)


def guardar_json(path_json, payload):
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _safe_arr(rows, key):
    """Extrae columna como float array, NaN donde falte la clave."""
    return np.array([float(r.get(key, float("nan"))) for r in rows], dtype=float)


def graficar_curvas(rows, out_prefix):
    alphas = np.array([r["alpha_deg"] for r in rows], dtype=float)

    # ── valores BL corregidos (principales) ─────────────────────────────────
    cd_bl  = _safe_arr(rows, "Cd_bl")
    cl_bl  = _safe_arr(rows, "Cl_bl")
    ef_bl  = _safe_arr(rows, "Ef_bl")
    cd_p   = _safe_arr(rows, "Cd_p_bl")   # presion LES (fiable)
    cd_v   = _safe_arr(rows, "Cd_visc_bl") # friccion BL corregida

    # ── valores raw LES (referencia/diagnostico) ─────────────────────────────
    cd_raw = _safe_arr(rows, "Cd_mean")
    cl_raw = _safe_arr(rows, "Cl_mean")
    ef_raw = _safe_arr(rows, "Ef_mean")

    def _save(fig, path):
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    # ── Cd ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, cd_bl,  "o-",  linewidth=2,   markersize=6, color="#E53935", label="Cd_bl (corregido)")
    ax.plot(alphas, cd_raw, "s--", linewidth=1.5,  markersize=5, color="#90A4AE", label="Cd_mean (raw LES)")
    ax.plot(alphas, cd_p,   "^:",  linewidth=1.2,  markersize=4, color="#1E88E5", label="Cd_p (presion LES)")
    ax.plot(alphas, cd_v,   "v:",  linewidth=1.2,  markersize=4, color="#43A047", label="Cd_visc (BL corr.)")
    ax.set_xlabel("Angulo de ataque alpha [deg]")
    ax.set_ylabel("Cd")
    ax.set_title("Cd vs angulo de ataque")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, f"{out_prefix}_Cd.png")

    # ── Cl ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, cl_bl,  "o-",  linewidth=2,   markersize=6, color="#E53935", label="Cl_bl (corregido)")
    ax.plot(alphas, cl_raw, "s--", linewidth=1.5,  markersize=5, color="#90A4AE", label="Cl_mean (raw LES)")
    ax.set_xlabel("Angulo de ataque alpha [deg]")
    ax.set_ylabel("Cl")
    ax.set_title("Cl vs angulo de ataque")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5, linestyle=":")
    ax.legend()
    _save(fig, f"{out_prefix}_Cl.png")

    # ── Eficiencia ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, ef_bl,  "o-",  linewidth=2,   markersize=6, color="#E53935", label="Cl/Cd BL (corregido)")
    ax.plot(alphas, ef_raw, "s--", linewidth=1.5,  markersize=5, color="#90A4AE", label="Cl/Cd raw LES")
    ax.set_xlabel("Angulo de ataque alpha [deg]")
    ax.set_ylabel("Cl/Cd")
    ax.set_title("Eficiencia vs angulo de ataque")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5, linestyle=":")
    ax.legend()
    _save(fig, f"{out_prefix}_Eficiencia.png")

    # ── Polar Cl vs Cd ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(cd_bl,  cl_bl,  "o-",  linewidth=2,  markersize=6, color="#E53935", label="BL corregido")
    ax.plot(cd_raw, cl_raw, "s--", linewidth=1.5, markersize=5, color="#90A4AE", label="Raw LES")
    for i, a in enumerate(alphas):
        if i % 2 == 0:
            ax.annotate(f"{a:.0f}°", (cd_bl[i], cl_bl[i]),
                        textcoords="offset points", xytext=(5, 3), fontsize=7, color="#E53935")
    ax.set_xlabel("Cd")
    ax.set_ylabel("Cl")
    ax.set_title("Polar Cl vs Cd")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, f"{out_prefix}_Polar.png")

    # ── Componentes Cd ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alphas, cd_p, "o-",  linewidth=2,  markersize=5, color="#1E88E5", label="Cd_p (presion LES)")
    ax.plot(alphas, cd_v, "s--", linewidth=2,  markersize=5, color="#43A047", label="Cd_visc (BL corr.)")
    ax.plot(alphas, cd_bl,"^-",  linewidth=1.5, markersize=4, color="#E53935", label="Cd_bl = Cd_p + Cd_visc")
    ax.set_xlabel("Angulo de ataque alpha [deg]")
    ax.set_ylabel("Coeficiente")
    ax.set_title("Componentes Cd vs angulo de ataque")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, f"{out_prefix}_Componentes.png")


def imprimir_tabla(rows):
    print("\n" + "=" * 108)
    print(
        f"{'alpha':>6} {'nx':>6} {'ny':>6} {'Cd_mean':>11} {'Cl_mean':>11} {'Ef_mean':>11} "
        f"{'Cd_final':>11} {'Cl_final':>11} {'tiempo(s)':>10}"
    )
    print("-" * 108)
    for r in rows:
        print(
            f"{r['alpha_deg']:>6.1f} {r['nx']:>6} {r['ny']:>6} "
            f"{r['Cd_mean']:>11.5f} {r['Cl_mean']:>11.5f} {r['Ef_mean']:>11.5f} "
            f"{r['Cd_final']:>11.5f} {r['Cl_final']:>11.5f} {r['elapsed_s']:>10.1f}"
        )
    print("=" * 108)

    ref = rows[0]
    ref_cd = ref["Cd_mean"] if abs(ref["Cd_mean"]) > 1e-12 else 1e-12
    ref_cl = ref["Cl_mean"] if abs(ref["Cl_mean"]) > 1e-12 else 1e-12
    print("\nVariacion respecto a alpha=0 (medias):")
    for r in rows[1:]:
        d_cd = 100.0 * (r["Cd_mean"] - ref["Cd_mean"]) / abs(ref_cd)
        d_cl = 100.0 * (r["Cl_mean"] - ref["Cl_mean"]) / abs(ref_cl)
        print(f"  alpha={r['alpha_deg']:.1f} -> dCd={d_cd:+.1f}%  dCl={d_cl:+.1f}%")


def main():
    from Simulador2D import main as sim_main

    results = []
    v0x = float(COMMON["v0x"])
    v0y = float(COMMON["v0y"])
    U_inf = float(np.sqrt(v0x ** 2 + v0y ** 2))

    for alpha in ALPHAS:
        print("\n" + "=" * 70)
        print(f"  Barrido alpha: alpha = {alpha:.1f} deg")
        print("=" * 70)

        t0 = time.time()
        mesh = sim_main(**COMMON, alpha_deg=int(alpha))
        elapsed = time.time() - t0

        mu = COMMON["rho"] * COMMON["nu"]
        q = COMMON["rho"] * U_inf * U_inf * COMMON["chord"]

        forces = mesh.compute_drag_lift(mu, rho=COMMON["rho"], n_extrap_layers=5)

        cd_mean, cd_final, cd_std, n_cd = resumen_vector(
            mesh.cdvector, COMMON["iteraciones"], COMMON["guardado"], DESCARTE_FRAC
        )
        cl_mean, cl_final, cl_std, n_cl = resumen_vector(
            mesh.clvector, COMMON["iteraciones"], COMMON["guardado"], DESCARTE_FRAC
        )

        ef_mean = cl_mean / cd_mean if abs(cd_mean) > 1e-12 else float("nan")
        ef_final = cl_final / cd_final if abs(cd_final) > 1e-12 else float("nan")

        row = {
            "alpha_deg": float(alpha),
            "nx": int(mesh.nx),
            "ny": int(mesh.ny),
            "Cd_mean": float(cd_mean),
            "Cl_mean": float(cl_mean),
            "Ef_mean": float(ef_mean),
            "Cd_final": float(cd_final),
            "Cl_final": float(cl_final),
            "Ef_final": float(ef_final),
            "Cd_std": float(cd_std),
            "Cl_std": float(cl_std),
            "Cd_p": float(2.0 * forces["Drag_p"] / q),
            "Cd_v": float(2.0 * forces["Drag_v"] / q),
            "Cl_p": float(2.0 * forces["Lift_p"] / q),
            "Cl_v": float(2.0 * forces["Lift_v"] / q),
            "Drag": float(forces["Drag"]),
            "Lift": float(forces["Lift"]),
            "n_muestras_cd": int(n_cd),
            "n_muestras_cl": int(n_cl),
            "elapsed_s": float(elapsed),
        }

        try:
            from bl_correction import compute_corrected_forces
            bl = compute_corrected_forces(mesh, filepath=COMMON["filepath"],
                                          alpha_deg=float(alpha))
            row.update({
                "Cl_bl":         float(bl["Cl"]),
                "Cd_bl":         float(bl["Cd"]),
                "Cd_p_bl":       float(bl["Cd_p"]),
                "Cd_visc_bl":    float(bl["Cd_visc"]),
                "Ef_bl":         float(bl["Ef"]),
                "Cl_inviscid":   float(bl["Cl_inviscid"]),
                "trans_x_upper": float(bl["trans_x_upper"]),
                "trans_x_lower": float(bl["trans_x_lower"]),
                "regime_upper":  bl.get("regime_upper", ""),
                "regime_lower":  bl.get("regime_lower", ""),
                "x_sep_upper":   float(bl.get("x_sep_upper", 1.0)),
                "x_sep_lower":   float(bl.get("x_sep_lower", 1.0)),
            })
        except Exception as _e:
            row.update({"Cl_bl": float("nan"), "Cd_bl": float("nan"),
                        "Cd_p_bl": float("nan"), "Cd_visc_bl": float("nan"),
                        "Ef_bl": float("nan")})
            print(f"  [BL correction failed: {_e}]")

        results.append(row)

        bl_info = ""
        if not np.isnan(row.get("Cd_bl", float("nan"))):
            bl_info = (f"  |  Cd_bl={row['Cd_bl']:.5f}  Cl_bl={row['Cl_bl']:.5f}"
                       f"  [{row.get('regime_upper','?')}/{row.get('regime_lower','?')}]")
        print(
            f"  -> nx={mesh.nx}, ny={mesh.ny} | "
            f"Cd_mean={row['Cd_mean']:.5f}, Cl_mean={row['Cl_mean']:.5f} ({elapsed:.1f}s)"
            f"{bl_info}"
        )

        # Liberar memoria GPU entre corridas
        del mesh
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    imprimir_tabla(results)

    payload = {
        "config": {
            "COMMON": COMMON,
            "alphas": ALPHAS,
            "descartar_frac": DESCARTE_FRAC,
        },
        "resultados": results,
    }

    guardar_csv(OUT_CSV, results)
    guardar_json(OUT_JSON, payload)
    graficar_curvas(results, OUT_PNG_PREFIX)

    print("\nArchivos generados:")
    print(f"  - {OUT_CSV}")
    print(f"  - {OUT_JSON}")
    print(f"  - {OUT_PNG_PREFIX}_Cd.png")
    print(f"  - {OUT_PNG_PREFIX}_Cl.png")
    print(f"  - {OUT_PNG_PREFIX}_Eficiencia.png")
    print(f"  - {OUT_PNG_PREFIX}_Polar.png")
    print(f"  - {OUT_PNG_PREFIX}_Componentes.png")


if __name__ == "__main__":
    main()

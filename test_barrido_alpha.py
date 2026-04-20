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
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np


# Parametros comunes para todas las corridas
COMMON = dict(
    Lx=12,
    Ly=8,
    cx=2,
    cy=4,
    CFL=0.5,
    iteraciones=4000,
    guardado=50,
    polar_descarte=0.3,
    divergencia=1e-1,
    v0x=1.0,
    v0y=0.0,
    rho=1.0,
    nu=1 / 100000,
    filepath="NACA_0012",
    chord=1.0,
    dx_min=0.001,
    factor_expansion=1.10,
    ancho_zona_fina_x=1.2,
    ancho_zona_fina_y=1.0,
    ratio_max_malla=100.0,
    graficos=False,
    save_frames=False,
    frames_dir_grueso="",
    usar_wale=False,
    stop_on_convergence=False,
    live_view=False,
    mostrar_malla=False,
    mg_modo_turbo=True,
)

# Barrido de alpha: 0 a 10 cada 2 grados
ALPHAS = list(range(0, 11, 1))
DESCARTE_FRAC = 0.30

OUT_CSV = "barrido_alpha_resultados.csv"
OUT_JSON = "barrido_alpha_resultados.json"
OUT_PNG_PREFIX = "barrido_alpha"


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


def graficar_curvas(rows, out_prefix):
    alphas = np.array([r["alpha_deg"] for r in rows], dtype=float)

    def plot_doble(y_mean, y_final, ylabel, titulo, out_path):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(alphas, y_mean, "o-", linewidth=2, markersize=6, label=f"{ylabel} mean")
        ax.plot(alphas, y_final, "s--", linewidth=1.8, markersize=5, label=f"{ylabel} final")
        ax.set_xlabel("Angulo de ataque alpha [deg]")
        ax.set_ylabel(ylabel)
        ax.set_title(titulo)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    cd_mean = np.array([r["Cd_mean"] for r in rows], dtype=float)
    cd_final = np.array([r["Cd_final"] for r in rows], dtype=float)
    cl_mean = np.array([r["Cl_mean"] for r in rows], dtype=float)
    cl_final = np.array([r["Cl_final"] for r in rows], dtype=float)
    ef_mean = np.array([r["Ef_mean"] for r in rows], dtype=float)
    ef_final = np.array([r["Ef_final"] for r in rows], dtype=float)

    plot_doble(cd_mean, cd_final, "Cd", "Cd vs angulo de ataque", f"{out_prefix}_Cd.png")
    plot_doble(cl_mean, cl_final, "Cl", "Cl vs angulo de ataque", f"{out_prefix}_Cl.png")
    plot_doble(ef_mean, ef_final, "Cl/Cd", "Eficiencia vs angulo de ataque", f"{out_prefix}_Eficiencia.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alphas, [r["Cd_p"] for r in rows], "o-", linewidth=2, label="Cd_p")
    ax.plot(alphas, [r["Cd_v"] for r in rows], "o-", linewidth=2, label="Cd_v")
    ax.plot(alphas, [r["Cl_p"] for r in rows], "s--", linewidth=1.8, label="Cl_p")
    ax.plot(alphas, [r["Cl_v"] for r in rows], "s--", linewidth=1.8, label="Cl_v")
    ax.set_xlabel("Angulo de ataque alpha [deg]")
    ax.set_ylabel("Coeficiente")
    ax.set_title("Componentes de fuerzas vs angulo de ataque")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_Componentes.png", dpi=200)
    plt.close(fig)


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
    U_inf = float(np.sqrt(COMMON["v0x"] ** 2 + COMMON["v0y"] ** 2))

    for alpha in ALPHAS:
        print("\n" + "=" * 70)
        print(f"  Barrido alpha: alpha = {alpha:.1f} deg")
        print("=" * 70)

        t0 = time.time()
        mesh = sim_main(**COMMON, alpha_deg=float(alpha))
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
        results.append(row)

        print(
            f"  -> nx={mesh.nx}, ny={mesh.ny} | "
            f"Cd_mean={row['Cd_mean']:.5f}, Cl_mean={row['Cl_mean']:.5f}, Ef_mean={row['Ef_mean']:.5f} "
            f"({elapsed:.1f}s)"
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
    print(f"  - {OUT_PNG_PREFIX}_Componentes.png")


if __name__ == "__main__":
    main()

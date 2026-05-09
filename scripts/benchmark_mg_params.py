"""
Estudio de parámetros del solver multigrid.
Prueba 6 configuraciones × 500 iters y reporta: it/s, div_mean, div_max, Cl, Cd.
Salida: benchmark_mg_results.json + tabla en consola.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import cupy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Simulador2D import main as simular_main

OUT = Path(__file__).parent / "benchmark_mg_results.json"

# Configs base: combinaciones de max_outer / cycles / div
BASE = [
    {"id": "T1", "desc": "Turbo actual",            "max_outer": 4,  "cycles": 3, "div": 0.10},
    {"id": "T2", "desc": "Turbo + más ciclos",       "max_outer": 4,  "cycles": 5, "div": 0.05},
    {"id": "N1", "desc": "Normal actual",             "max_outer": 8,  "cycles": 5, "div": 0.10},
    {"id": "N2", "desc": "Equilibrio",                "max_outer": 6,  "cycles": 4, "div": 0.05},
    {"id": "N3", "desc": "Normal estricto",           "max_outer": 8,  "cycles": 5, "div": 0.01},
    {"id": "N4", "desc": "Muchos outers/pocos ciclos","max_outer": 12, "cycles": 3, "div": 0.01},
]

# Barrido de niveles MG: 0 = sólo RB-SOR, 1-2 = coarsening real (vertex-centered)
NIVELES = [0, 1, 2]

CONFIGS = [
    {**b, "id": f"{b['id']}_L{n}", "niveles": n}
    for b in BASE
    for n in NIVELES
]

CFG_BASE = dict(
    filepath="profiles/NACA_0012", chord=1.0, alpha_deg=10.0,
    Lx=12.0, Ly=8.0, cx=2.0,
    dx_min=0.001, factor_expansion=1.1,
    ancho_zona_fina_x=1.2, ancho_zona_fina_y=1.0,
    v0x=1.0, v0y=0.0, rho=1.0, p0=0.0,
    nu=1e-5,
    CFL=0.5, iteraciones=500, guardado=10,
    mg_modo_turbo=False,
    graficos=False, save_frames=False, live_view=False,
    mostrar_malla=False, stop_on_convergence=False,
    corregir_deriva_vertical=False,
    usar_wale=True, wale_Cw=0.15,
)

ITER_N = int(os.environ.get("ITER", "500"))


def run_config(cfg: dict) -> dict:
    kwargs = {**CFG_BASE,
              "iteraciones": ITER_N,
              "divergencia": cfg["div"],
              "mg_max_outer": cfg["max_outer"],
              "mg_cycles_per_outer": cfg["cycles"],
              "mg_niveles_max": cfg["niveles"]}

    t0 = time.time()
    mesh = simular_main(**kwargs)
    walltime = time.time() - t0
    its_per_sec = ITER_N / walltime if walltime > 0 else 0

    # Cl / Cd — últimos 30%
    cd_arr = cp.asnumpy(mesh.cdvector).flatten()
    cl_arr = cp.asnumpy(mesh.clvector).flatten()
    mask = (cd_arr != 0) | (cl_arr != 0)
    cd_arr = cd_arr[mask]; cl_arr = cl_arr[mask]
    n_ss = max(1, int(len(cd_arr) * 0.7))
    cd_ss = cd_arr[n_ss:]; cl_ss = cl_arr[n_ss:]

    # Divergencia — últimas 100 muestras
    div_mean_arr = cp.asnumpy(mesh.divvector).flatten()
    div_mean_arr = div_mean_arr[div_mean_arr > 0]
    div_max_arr = cp.asnumpy(mesh.divvector_max).flatten() if hasattr(mesh, 'divvector_max') else np.array([0.0])
    div_max_arr = div_max_arr[div_max_arr > 0]

    tail = 100
    div_mean_ss = div_mean_arr[-tail:] if len(div_mean_arr) >= tail else div_mean_arr
    div_max_ss  = div_max_arr[-tail:]  if len(div_max_arr)  >= tail else div_max_arr

    out = dict(
        id=cfg["id"], desc=cfg["desc"],
        max_outer=cfg["max_outer"], cycles_per_outer=cfg["cycles"],
        divergencia=cfg["div"], niveles=cfg["niveles"],
        its_per_sec=round(its_per_sec, 2),
        div_mean_norm=round(float(np.mean(div_mean_ss)), 4) if len(div_mean_ss) else None,
        div_max_norm=round(float(np.mean(div_max_ss)), 4)   if len(div_max_ss)  else None,
        Cl_mean=round(float(np.mean(cl_ss)), 4) if len(cl_ss) else None,
        Cl_std=round(float(np.std(cl_ss)),  4)  if len(cl_ss) else None,
        Cd_mean=round(float(np.mean(cd_ss)), 4) if len(cd_ss) else None,
        Cd_std=round(float(np.std(cd_ss)),  4)  if len(cd_ss) else None,
    )

    try:
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from bl_correction import compute_corrected_forces
        bl = compute_corrected_forces(mesh, filepath=CFG_BASE["filepath"],
                                      alpha_deg=float(CFG_BASE["alpha_deg"]))
        out.update({
            "Cl_bl":      round(float(bl["Cl"]), 4),
            "Cd_bl":      round(float(bl["Cd"]), 4),
            "Cd_p_bl":    round(float(bl["Cd_p"]), 4),
            "Cd_visc_bl": round(float(bl["Cd_visc"]), 4),
            "Ef_bl":      round(float(bl["Ef"]), 4),
        })
    except Exception as _e:
        out.update({"Cl_bl": None, "Cd_bl": None,
                    "Cd_p_bl": None, "Cd_visc_bl": None, "Ef_bl": None})
        print(f"  [BL correction failed: {_e}]", flush=True)

    return out


def print_table(results: list[dict]) -> None:
    header = f"{'ID':<7} {'Desc':<26} {'mo':>3} {'cpo':>4} {'div':>6} {'lvl':>3} {'it/s':>6} {'div_mean':>9} {'div_max':>9} {'Cl':>7} {'Cd':>7}"
    print("\n" + "="*len(header))
    print(header)
    print("-"*len(header))
    for r in results:
        print(f"{r['id']:<7} {r['desc']:<26} {r['max_outer']:>3} {r['cycles_per_outer']:>4} "
              f"{r['divergencia']:>6.3f} {r['niveles']:>3} {r['its_per_sec']:>6.1f} "
              f"{r['div_mean_norm'] or 0:>9.4f} {r['div_max_norm'] or 0:>9.3f} "
              f"{r['Cl_mean'] or 0:>7.4f} {r['Cd_mean'] or 0:>7.4f}")
    print("="*len(header))
    print("Columnas: mo=max_outer  cpo=cycles_per_outer  div=tol_div_rel  lvl=mg_niveles_max")
    print("div_mean/max normalizados por U_inf/chord\n")


def main() -> None:
    results = []
    for cfg in CONFIGS:
        print(f"\n[bench] Config {cfg['id']}: {cfg['desc']}  "
              f"(max_outer={cfg['max_outer']}, cycles={cfg['cycles']}, "
              f"div={cfg['div']}, niveles={cfg['niveles']})",
              flush=True)
        try:
            r = run_config(cfg)
            results.append(r)
            print(f"  -> it/s={r['its_per_sec']:.1f}  div_mean={r['div_mean_norm']:.4f}  "
                  f"div_max={r['div_max_norm']:.3f}  Cl={r['Cl_mean']:.4f}  Cd={r['Cd_mean']:.4f}",
                  flush=True)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}", flush=True)
            traceback.print_exc()
            results.append({"id": cfg["id"], "error": str(e)})

    print_table([r for r in results if "error" not in r])
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[save] {OUT}")


if __name__ == "__main__":
    main()

"""
Test diagnóstico: mide asimetría introducida por coarsening MG.
NACA_0012, alpha=0° → Cl debería ser ~0. Compara niveles_max=0 vs 1.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import cupy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Simulador2D import main as simular_main

CFG_BASE = dict(
    filepath="NACA_0012", chord=1.0, alpha_deg=0.0,
    Lx=12.0, Ly=8.0, cx=2.0,
    dx_min=0.001, factor_expansion=1.1,
    ancho_zona_fina_x=1.2, ancho_zona_fina_y=1.0,
    v0x=1.0, v0y=0.0, rho=1.0, p0=0.0,
    nu=1e-5,
    CFL=0.5, iteraciones=200, guardado=10,
    mg_modo_turbo=False,
    mg_max_outer=4, mg_cycles_per_outer=5,
    divergencia=0.05,
    graficos=False, save_frames=False, live_view=False,
    mostrar_malla=False, stop_on_convergence=False,
    corregir_deriva_vertical=False,
    usar_wale=True, wale_Cw=0.15,
)


def asimetria_p(p, X_1d, cx, chord, banda=2.0):
    """max|p(x,y) - p(2cx-x,y)| en franja X dentro [cx-banda, cx+banda]."""
    p_np = cp.asnumpy(p) if hasattr(p, 'get') else np.asarray(p)
    x = cp.asnumpy(X_1d) if hasattr(X_1d, 'get') else np.asarray(X_1d)
    x_lo, x_hi = cx - banda * chord, cx + banda * chord
    in_band = np.where((x >= x_lo) & (x <= x_hi))[0]
    p_norm = float(np.max(np.abs(p_np))) + 1e-30
    asim_max = 0.0
    for j in in_band:
        x_mirror = 2.0 * cx - x[j]
        j_m = int(np.argmin(np.abs(x - x_mirror)))
        d = float(np.max(np.abs(p_np[:, j] - p_np[:, j_m])))
        if d > asim_max:
            asim_max = d
    return asim_max, asim_max / p_norm


def run(label, niveles_max):
    cfg = {**CFG_BASE, "mg_niveles_max": niveles_max}
    print(f"\n[test] {label}: niveles_max={niveles_max}")
    t0 = time.time()
    mesh = simular_main(**cfg)
    dt = time.time() - t0

    cl_arr = cp.asnumpy(mesh.clvector).flatten()
    cd_arr = cp.asnumpy(mesh.cdvector).flatten()
    mask = (cl_arr != 0) | (cd_arr != 0)
    cl_arr = cl_arr[mask]; cd_arr = cd_arr[mask]
    n_ss = max(1, int(len(cl_arr) * 0.7))
    cl_ss = cl_arr[n_ss:]
    cd_ss = cd_arr[n_ss:]

    div_arr = cp.asnumpy(mesh.divvector).flatten()
    div_arr = div_arr[div_arr > 0]

    cl_mean = float(np.mean(cl_ss)) if len(cl_ss) else float('nan')
    cd_mean = float(np.mean(cd_ss)) if len(cd_ss) else float('nan')
    div_mean = float(np.mean(div_arr[-50:])) if len(div_arr) else float('nan')

    asim_abs, asim_rel = asimetria_p(mesh.p, mesh.X_1d,
                                     CFG_BASE["cx"], CFG_BASE["chord"])

    print(f"  Cl_mean = {cl_mean:+.6f}   (esperado ~0 para α=0°)")
    print(f"  Cd_mean = {cd_mean:+.6f}")
    print(f"  div_mean_norm = {div_mean:.4f}")
    print(f"  asimetria p (max abs) = {asim_abs:.4e}   (rel = {asim_rel:.2%})")
    print(f"  walltime = {dt:.1f}s   it/s = {200/dt:.2f}")
    return dict(label=label, niveles_max=niveles_max,
                Cl=cl_mean, Cd=cd_mean, div_mean_norm=div_mean,
                asim_abs=asim_abs, asim_rel=asim_rel, walltime=dt)


def main():
    results = []
    results.append(run("baseline (sin coarsening)", 0))
    results.append(run("coarsening 1 nivel", 1))

    print("\n" + "="*70)
    print(f"{'Label':<30} {'Cl':>10} {'div_mean':>10} {'asim_p':>12}")
    print("-"*70)
    for r in results:
        print(f"{r['label']:<30} {r['Cl']:>+10.5f} {r['div_mean_norm']:>10.4f} {r['asim_rel']:>11.2%}")
    print("="*70)

    cl_diff = abs(results[1]['Cl']) - abs(results[0]['Cl'])
    if abs(results[1]['Cl']) > 0.01:
        print(f"\n[BUG CONFIRMADO] coarsening introduce |Cl|={abs(results[1]['Cl']):.4f} (>0.01)")
    else:
        print(f"\n[OK] coarsening no rompe simetría: |Cl|={abs(results[1]['Cl']):.4f}")


if __name__ == "__main__":
    main()

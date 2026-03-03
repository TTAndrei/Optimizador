"""
Prueba sin trim del TE para demostrar que el trim ES la solucion del NaN.
Forzamos min_te_height=0 para desactivar el trim.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cupy as cp
from Simulador2D import Mesh

# Parametros identicos a study_simulador.py
dx_grueso = 0.004
Lx, Ly = 7, 6
v0x, v0y = 5.0, 0.0
p0 = 0.0
cx, cy = 1.0, Ly / 2.0
CFL = 0.5
nu = 1.5e-5
rho = 1.225

alphas = [0, 2, 5, 8, 10, 12]

for alpha in alphas:
    print(f"\n--- alpha={alpha} SIN TRIM ---")
    solver = Mesh(Lx, Ly, p0, v0x, v0y, dx_grueso, dx_grueso)
    # min_te_height=0 desactiva el trim
    solver.load_solids_from_file(
        filepath='AG24', chord=1.0,
        x_offset=cx, y_offset=cy,
        alpha_deg=alpha, fill=True, plot=False,
        min_te_height=0.0  # SIN TRIM
    )
    solver.set_boundary("left", "inflow", value=(v0x, v0y))
    solver.set_boundary("top", "slip", value=None)
    solver.set_boundary("bottom", "slip", value=None)
    solver.set_boundary("right", "outflow", value=p0)

    dt = CFL * dx_grueso / abs(v0x)
    failed = False
    for it in range(200):
        speed = cp.sqrt(solver.u**2 + solver.v**2)
        fluid_mask = ~solver.solid
        Umax = float(cp.max(speed[fluid_mask])) if cp.any(fluid_mask) else v0x
        Umax = max(Umax, 1e-12)
        dt_adv = CFL * dx_grueso / Umax
        dt_visc = 0.25 * (dx_grueso**2) / nu
        dt_use = min(dt_adv, dt_visc, dt * 2.0)
        if dt_use < 1e-8:
            dt_use = dt

        solver.advect_velocities(dt_use)
        solver.apply_boundaries(after_projection=False)
        solver.diffuse_velocity(nu, dt_use, usar_wale=False)
        solver.apply_boundaries(after_projection=False)
        solver.project_multigrid(rho, dt_use, tol_div=1e-1, verbose=False)
        solver.apply_boundaries(after_projection=True)
        solver.u = cp.clip(solver.u, -50, 50)
        solver.v = cp.clip(solver.v, -50, 50)

        has_nan = bool(cp.isnan(solver.u).any() or cp.isnan(solver.v).any() or cp.isnan(solver.p).any())
        vmax = float(cp.max(cp.abs(solver.u)))

        if has_nan:
            print(f"  NaN at iter {it}! dt_use={dt_use:.2e}")
            failed = True
            break
        if vmax > 45:
            print(f"  Blowup at iter {it}! vmax={vmax:.1f} dt_use={dt_use:.2e}")
            failed = True
            break
    if not failed:
        print(f"  OK: 200 iterations, no NaN. Umax_final={Umax:.2f}")

print("\nDone.")

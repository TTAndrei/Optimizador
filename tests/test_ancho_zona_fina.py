"""
Prueba: comparar fuerzas (Cd, Cl) con 3 valores distintos de ancho_zona_fina.
Ejecuta main() secuencialmente y almacena los resultados para comparar.
"""

import sys, time
import numpy as np

# Parámetros comunes para todas las corridas
COMMON = dict(
    Lx=12, Ly=6, cx=2,
    CFL=0.5,
    alpha_deg=5,
    iteraciones=600,       # Pocas iters para comparación rápida
    polar_descarte=0.3,
    divergencia=1e-1,
    v0x=1.0, v0y=0.0,
    rho=1.0, nu=1/100000,
    filepath="profiles/NACA_0012",
    chord=1.0,
    dx_min=0.002,
    factor_expansion=1.05,
    graficos=False,
    save_frames=False,
    frames_dir_grueso="",
    usar_wale=False,
    stop_on_convergence=False,
    live_view=False,
    mostrar_malla=False,
)

# Los 3 anchos en Y a comparar (en metros) — X se deja fijo
ANCHO_X = 1.2
ANCHOS_Y = [0.4, 0.8, 1.6]

results = []

for ancho in ANCHOS_Y:
    print("\n" + "=" * 60)
    print(f"  ancho_zona_fina_y = {ancho} m  (x fijo = {ANCHO_X} m)")
    print("=" * 60)

    from Simulador2D import main          # importar cada vez en caso de estado global
    t0 = time.time()

    mesh = main(
        **COMMON,
        ancho_zona_fina_x=ANCHO_X,
        ancho_zona_fina_y=ancho,
    )

    elapsed = time.time() - t0

    # Calcular fuerzas al final
    mu = COMMON['rho'] * COMMON['nu']
    forces = mesh.compute_drag_lift(mu, rho=COMMON['rho'], n_extrap_layers=5)
    U_inf  = COMMON['v0x']
    chord  = COMMON['chord']
    q      = COMMON['rho'] * U_inf**2 * chord

    Cd = 2.0 * forces['Drag'] / q
    Cl = 2.0 * forces['Lift'] / q

    results.append({
        'ancho': ancho,
        'nx': mesh.nx,
        'ny': mesh.ny,
        'Cd': Cd,
        'Cl': Cl,
        'elapsed_s': elapsed,
    })

    print(f"\n  → nx={mesh.nx}, ny={mesh.ny}  |  Cd={Cd:.4f}  Cl={Cl:.4f}  ({elapsed:.1f}s)")

# ── Tabla final ──────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"{'ancho_zona_fina_y':>20} {'nx':>6} {'ny':>6} {'Cd':>10} {'Cl':>10} {'tiempo(s)':>10}")
print("-" * 60)
for r in results:
    print(f"{r['ancho']:>20.2f} {r['nx']:>6} {r['ny']:>6} "
          f"{r['Cd']:>10.4f} {r['Cl']:>10.4f} {r['elapsed_s']:>10.1f}")
print("=" * 60)

# Variación relativa respecto al primer caso
ref_cd = results[0]['Cd'] or 1e-12
ref_cl = results[0]['Cl'] or 1e-12
print("\nVariación respecto al caso 1 (ancho_y más estrecho):")
for r in results[1:]:
    dCd = 100.0 * (r['Cd'] - results[0]['Cd']) / abs(ref_cd)
    dCl = 100.0 * (r['Cl'] - results[0]['Cl']) / abs(ref_cl)
    print(f"  ancho_y={r['ancho']:.2f}m  → ΔCd={dCd:+.1f}%  ΔCl={dCl:+.1f}%")

# RESUMEN — Optimizador CFD 2D

## Qué es
Simulador CFD 2D incompresible (Navier-Stokes) en GPU (CuPy, RTX 3070 Ti) para perfiles alares, base de un futuro optimizador aerodinámico. Todo el solver vive en `Simulador2D.py` (~8000 líneas, clase `Mesh` + `main()`).

- Malla cartesiana estirada (zona fina alrededor del perfil), advección semi-Lagrangiana, difusión (+WALE LES opcional), proyección de presión multigrid/CG.
- Sólidos por IBM (Immersed Boundary): máscara rasterizada + ghost-cell no-slip (`ibm_wall_mode="ghost_noslip"`).
- **Ejecutar SIEMPRE con `.venv/bin/python`** (el python3 del sistema no tiene CuPy).

## Estado actual (2026-07-02, rama Malla-Variable)

Caso de trabajo: NACA0012, α=5°, Re=1e5, Cl físico esperado ~0.55.

**Problema en curso**: sobre-circulación (meseta Cl~0.85-1.1), ΔCp_TE≈-0.15 (Kutta no cierra) y separación artificial tardía (~30k iters) que colapsa el Cl. Causa raíz identificada: el SDF por `distance_transform_edt` sobre la máscara raster da normales ruidosas en la cuña sub-celda del TE → ghost-cells con image points desviados → fuga de circulación.

**Diagnóstico cerrado (cadena de 3 capas, con evidencia cuantitativa en `results/agent_tests/`)**:
1. LSB laminar a Re≥1e4 (burst t≈5-7 convectivos, colapso). Fix: **modelo Spalart-Allmaras** (`turb_model="sa"`, nuevo) — elimina el colapso. WALE inerte a cualquier Cw.
2. Pared sub-resuelta: SA necesita dx≤0.002 (y+≲10) para flujo pegado.
3. **Inconsistencia IBM↔proyección** (causa raíz del Cl no-físico): la proyección legacy + `apply_ghost_cell_bc`/`reforzar_impermeabilidad` post-hoc absorben masa en la banda de pared (Q≈-0.06·U·c = succión parásita) → sobre-circulación ~2× (Cl 1.2-1.4 vs 0.55). La variante `projection_variant="compatible_flux"` elimina el sumidero (Cd físico 0.010) pero desangra la vorticidad de pared → Cl→0.1. **Ninguna de las dos da el valor físico; falta tratamiento de pared consistente (Neumann u·n en la Poisson o cut-cell) — próximo trabajo.**
- Métrica clave del sumidero: flujo neto ∮u·n dl por lazo cerrado alrededor del perfil (debe →0).
- Solver validado como físico a Re=1e3 (Kutta cierra sola, Cl estable).

**Añadido esta sesión**: monitor Kutta (`kutta_dcp_vector` + warning + subplot), SDF exacto de polígono (`ibm_sdf_source="polygon"`), SA completo (advección semi-Lagrangiana de ν̃ + destrucción point-implicit), Kutta explícita opcional (`kutta_enforce`), fix de acumulación de presión en `_project_multigrid_compatible_flux` (antes self.p = incremento del último outer), checkpoints (`save_state`/`analizar_checkpoint`), backup run 20k: `sim_last_20k_cfl025.npz.bak`.

## Tests
- `scripts/agent_tests/run_kutta_tests.py {smoke|smoke_polygon|baseline_a5|polygon_a5|compare|regression}` — runs coarse (dx=0.004, ~5-15 min) con criterios cuantitativos, resultados en `results/agent_tests/`.
- `pytest tests/ --ignore=tests/test_generacion_geometrica_ga.py` (5 passed; el de GA está roto pre-existente: importa `RunGA` inexistente).

## Diagnósticos clave (métodos de Mesh)
- `compute_circulation()` — Γ en lazos, Cl_circ=-2Γ/(U·c).
- `compute_cp_diagnostics(mu, rho)` — Cl(∮ΔCp), ΔCp_TE, pico de succión.
- `compute_kutta_dcp_instant(face_data)` — ΔCp_TE instantáneo para el monitor.
- `debug_te_report()` — ghosts/irreparables cerca del TE.
- `plot_streamlines()`, `plot_forces_over_time()`, `plot_cp_vs_chord()`.

## Config de referencia (fina, `__main__`)
Lx=12, Ly=8, dx_min=0.001 (malla 3243×1203, ~2 it/s), CFL=0.25, NACA_0012_sharp, min_te_height_factor=1.0, turbo_hd, wake long_fine_x. 20k iters ≈ 3h.

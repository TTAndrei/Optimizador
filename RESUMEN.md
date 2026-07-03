# RESUMEN — Optimizador CFD 2D

## Qué es
Simulador CFD 2D incompresible (Navier-Stokes) en GPU (CuPy, RTX 3070 Ti) para perfiles alares, base de un futuro optimizador aerodinámico. Todo el solver vive en `Simulador2D.py` (~8000 líneas, clase `Mesh` + `main()`).

- Malla cartesiana estirada (zona fina alrededor del perfil), advección semi-Lagrangiana, difusión (+WALE LES opcional), proyección de presión multigrid/CG.
- Sólidos por IBM (Immersed Boundary): máscara rasterizada + ghost-cell no-slip (`ibm_wall_mode="ghost_noslip"`).
- **Ejecutar SIEMPRE con `.venv/bin/python`** (el python3 del sistema no tiene CuPy).

## Estado actual (2026-07-03, rama Malla-Variable)

Caso de trabajo: NACA0012, α=5°, Re=1e5, Cl físico esperado ~0.55.

**FIX DE RAÍZ IMPLEMENTADO: `wall_treatment="consistent"`** (kwarg de `main()`, default "legacy" intacto). El sumidero IBM está eliminado. Componentes (todas reutilizan piezas existentes):
1. Divergencia por caras en la proyección (`divergence_form="face_flux"` → `_compute_flux_divergence_field_uv`): cara fluido-sólido = flujo 0 exacto → el Poisson ve el flujo contra la pared.
2. Gradiente de presión one-sided en celdas de pared (`wall_pressure_gradient_mode="one_sided"`, kernel ya existía).
3. `reforzar_impermeabilidad` desactivado (`disable_reforzar=True`) — era la cirugía que borraba masa (Q≈-0.06·U·c).
El Laplaciano masked ya era Neumann correcto; el problema era que divergencia y corrección anulaban la componente entera junto al sólido (proyección ciega en pared).

**Resultados validados** (JSON+npz en `results/agent_tests/wc_*`; commiteados en `a5c14b7`, npz >100MB excluidos por `.gitignore`):
- `wc_re1e3_a5` (Re=1e3, α=5): Cl=0.211±0.0004, Cd=0.148, Q_lazo=+0.02, ΔCp_TE=0.04. **Clava el DNS de referencia (Kurtulus Re=1000: Cl~0.22, Cd~0.146)**. Legacy daba 0.14 con Q=-0.06.
- `wc_sa_a0` (α=0, SA): Cl=0.0003, sin NaN — simetría perfecta.
- `wc_sa_a5_dx2` (α=5, Re=1e5, SA, dx=0.002, turbo_hd): **estable sin colapso**, Kutta cierra (ΔCp_TE=0.012), pero Cl=0.274. Q_lazo=+0.026 (MG turbo saturado).
- `wc_sa_a5_dx2_full` (sin turbo: div=0.02, outer=8, niveles=2, 24k iters): Q_lazo=0.006 ✓ pero Cl=0.312. **Hipótesis Q→Cl pendiente ~20 FALSIFICADA**: ΔQ=0.020 predecía ΔCl≈+0.4, medido +0.04.
- `wc_sa_a5_dx2_long` (div=0.01, outer=12, 48k iters, t=18.3 conv, 2h GPU): Q=0.0032, **Cl CONVERGE en plateau 0.32** (transitorio: pico 0.40 en t≈2, mínimo 0.312 en t≈10, luego estabiliza 0.320 con deriva 0.0025). cl_cp=0.383, cl_circ=0.281 (dispersión estimadores cerrando). x_succión mejoró a 0.045. **PROYECCIÓN EXONERADA: Q→0 no recupera el Cl.**
- **Diagnóstico del déficit (0.32 vs 0.55)**: SA sobre-difunde. Evidencia: BL en x/c=0.5 espesor ~0.07c (3× placa plana turbulenta Re=1e5), χ=nu_t/nu~40-50 en BL (teórico ~8), Cp_min=-0.81 (esperado ~-1.9), Cd=0.048 (alto). Causas candidatas: (a) SA fully-turbulent desde LE a Re=1e5 (flujo real laminar hasta ~x/c=0.5, penaliza succión), (b) primera celda a y+≈10 sin wall function (SA quiere y+≲1). Distancia de pared SA verificada correcta; constantes SA estándar.
- Hallazgo menor: `mg_modo_turbo_hd` pisaba `divergencia` (arreglado: `min(divergencia, 0.05)`); runs `wc_sa_a5_dx2` y `_t02` idénticos (el fix no cambió nada porque turbo satura outers igual).
- Añadido kwarg `sa_nu_tilde_factor` a `main()` (default 3.0; bajo ~0.1 retrasa transición SA).
- `wc_sa_a5_dx2_lam01` (sa_nu_tilde_factor=0.1, t=14.2 conv): **PEOR** — Cl=0.266, x_succión=0.14 (LSB reapareciendo), Cp_min=-0.66, Cd=0.056; χ_max=184 (transiciona igual pero tarde). **Falsificada hipótesis (a) fully-turbulent**; retrasar transición reactiva la burbuja laminar. Mantener sa_nu_tilde_factor=3.0.
- **Sospecha cuantificada: difusión numérica del semi-Lagrangiano bilineal.** nu_num≈α(1-α)·dx²/(2dt)~1e-4 cerca de pared = 10× nu molecular a Re=1e5 (a Re=1e3 era ~10% de nu → por eso el DNS clavó). Escala ∝dx (dt∝dx por CFL). Junto con y+≈10, motiva test de resolución.
- `wc_sa_a5_dx1` (dx=0.001, 60k iters, t=11 conv, 3.6h GPU): **Cl=0.389** (vs 0.32 a dx=0.002), Cd=0.039, Cp_min=-0.96, χ_max=105, estable. **Camino resolución confirmado, error ~1º orden en dx.** Extrapolando dx→0: Cl≈0.46. dx=0.0005 no cabe en la 3070 Ti.
- **Causa dominante identificada: difusión numérica del SL bilineal, nu_num≈u·dx/2 (independiente de dt)** — 1º orden, ~5-10× nu molecular cerca de pared a Re=1e5. A Re=1e3 era ~10% de nu (por eso clavó DNS).
- **MacCormack implementado** (`advection_scheme="maccormack"` en main(), default "sl"): corrección Selle 2008 (trace forward + φ*+(φ-φ**)/2) con limitador min/max del stencil y **banda de 2 celdas junto al sólido en SL puro** (sin banda, la corrección sobre el staircase mete pico espurio Cp=-11 en TE; con banda, α=0 perfecto Cp_min=-0.47). Validación Re=1e3 α=5: Cl=0.245/Cd=0.155 (SL: 0.211/0.148, DNS: 0.22/0.146) — ligero overshoot coherente con menos difusión en malla coarse, física intacta.
- `wc_mc_sa_a5_dx2` (dx=0.002, SA, MacCormack, t=11.5 conv): **Cl=0.452** (SL daba 0.32), Cp_min=-1.20, x_succión=0.015 (posición física del LE ✓), χ_max=73, Cd=0.040, std(Cl)=1e-5 (estacionario casi a máquina, deciles planos desde t≈4, sin sangrado). **Cuarta capa (nu_num del SL) CONFIRMADA y resuelta.** Coste MacCormack: ~4% vs SL.
- `wc_mc_sa_a5_dx1` (dx=0.001 + MacCormack, t=9.7 conv, 3.6h): **Cl=0.496 ✓ OBJETIVO CUMPLIDO** (criterio [0.4,0.7]), Cd=0.0335 (físico ~0.03 ✓), Cp_min=-1.24, x_succión=0.02 ✓, Q=0.005 ✓, deriva 0.001. **Richardson dx→0: Cl≈0.54 — clava el físico 0.55.** Convergencia de malla limpia (0.452@dx2 → 0.496@dx1).
- **CONFIG DE REFERENCIA DEL OPTIMIZADOR**: `wall_treatment="consistent"` + `turb_model="sa"` + `advection_scheme="maccormack"`. dx=0.002 para screening (Cl-0.1 vs físico, ~65 min/run), dx=0.001 para validación (Cl-0.05, ~3.6h).
- Regresión verde tras MacCormack (5 passed; defaults intactos, "sl" sigue siendo default).
- **POLAR VALIDADA** (`wc_mc_polar_a0/a2/a8` + `wc_mc_sa_a5_dx2`, dx=0.002 MacCormack): Cl(α)= 0(0.000), 2(0.185), 5(0.452), 8(0.708); Cd= 0.0305/0.0323/0.0404/0.0624. Monotonía ✓, pendiente 0.088/deg ✓, simetría α=0 exacta ✓, sublinealidad física en α=8 (pre-stall), Cp_min=-2.13 y succión x/c=0.015 en α=8. **DIAGNÓSTICO DE 4 CAPAS CERRADO COMPLETO.**
- **Todo commiteado** (`a5c14b7` "lift y drag solucionados": MacCormack + sa_nu_tilde_factor + resultados wc_*; `93b6c54`: script de barrido; `5979fab`: .gitignore para npz >100MB).
- **Script de barrido CFL/dx/α añadido**: `scripts/agent_tests/run_cfl_sweep.py` — cola resumible de 30 sims (α=0/2/5/8/10 × CFL=0.25/0.5/0.75 × dx=0.004/0.002) con la config de referencia, iteraciones auto-escaladas a t=12 convectivos, métricas JSON por sim (iteración/tiempo de convergencia de Cl/Cd), npz de estado, frames cada 1000 iters, streamlines finales, series temporales. Reanuda desde la última completada; CSV resumen + plots polar/convergencia/coste-vs-error. **Aún no ejecutado** (`results/cfl_sweep/` solo tiene `frames/` vacío).

**Diagnóstico histórico cerrado (4 capas, evidencia en `results/agent_tests/`)**:
1. LSB laminar a Re≥1e4 (burst t≈5-7 convectivos, colapso). Fix: **Spalart-Allmaras** (`turb_model="sa"`). WALE inerte.
2. Pared sub-resuelta: SA necesita dx≤0.002 (y+≲10).
3. Inconsistencia IBM↔proyección: legacy absorbía masa (sobre-circulación 2.2×); compatible_flux desangraba vorticidad (Cl→0.1). **Resuelta con wall_treatment="consistent"**.
4. Difusión numérica del SL bilineal (nu_num≈u·dx/2, 1º orden): BL 3× gruesa, Cl 0.32 vs 0.55, transitorios de 18 conv. **Resuelta con advection_scheme="maccormack"** (coste ~4%, estacionario en t≈4-5 conv).
- Métrica decisiva: Q_lazo=∮u·n dl (runner la guarda como `q_loop_025c/075c`).

**Añadido sesiones previas**: monitor Kutta (`kutta_dcp_vector`), SDF polígono (`ibm_sdf_source="polygon"`), SA completo, Kutta explícita opcional (`kutta_enforce`), fix acumulación de presión MG compatible_flux (CG ya estaba bien), checkpoints.

## Próximos pasos
1. **Ejecutar el barrido** `scripts/agent_tests/run_cfl_sweep.py` (30 sims, resumible) para caracterizar coste/error CFL×dx×α y fijar la config del optimizador genético.
2. Diseño del optimizador sobre config de referencia: dx=0.002 MacCormack, 18k iters (~30 min/punto, estable en t≈4-5 conv), ranking con sesgo consistente Cl−0.10; validación de ganadores a dx=0.001 (Richardson → valor físico).
3. Opcional barato: probar MacCormack+turbo_hd (~19 it/s esperado) para pre-screening si el sesgo por Q=0.026 resulta consistente entre geometrías.
4. Opcional física: sublinealidad α=8 (deriva -0.002, TE separación creciendo) — vigilar si el optimizador explora α altos; el barrido incluye α=10 para localizar stall numérico.
5. Limpieza menor: `RESUMEN.md.tmp` vacío sin trackear en la raíz (borrar).

## Tests
- `scripts/agent_tests/run_kutta_tests.py {smoke|smoke_polygon|baseline_a5|polygon_a5|compare|regression}` — runs coarse (dx=0.004, ~5-15 min) con criterios cuantitativos, resultados en `results/agent_tests/`.
- `scripts/agent_tests/run_cfl_sweep.py` — barrido CFL/dx/α resumible (ver Próximos pasos), resultados en `results/cfl_sweep/`.
- `pytest tests/ --ignore=tests/test_generacion_geometrica_ga.py` (5 passed; el de GA está roto pre-existente: importa `RunGA` inexistente).

## Diagnósticos clave (métodos de Mesh)
- `compute_circulation()` — Γ en lazos, Cl_circ=-2Γ/(U·c).
- `compute_cp_diagnostics(mu, rho)` — Cl(∮ΔCp), ΔCp_TE, pico de succión.
- `compute_kutta_dcp_instant(face_data)` — ΔCp_TE instantáneo para el monitor.
- `debug_te_report()` — ghosts/irreparables cerca del TE.
- `plot_streamlines()`, `plot_forces_over_time()`, `plot_cp_vs_chord()`.

## Config de referencia (fina, `__main__`)
Lx=12, Ly=8, dx_min=0.001 (malla 3243×1203, ~2 it/s), CFL=0.25, NACA_0012_sharp, min_te_height_factor=1.0, turbo_hd, wake long_fine_x. 20k iters ≈ 3h.

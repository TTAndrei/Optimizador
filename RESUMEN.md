# RESUMEN — Optimizador CFD 2D

## Qué es
Simulador CFD 2D incompresible (Navier-Stokes) en GPU (CuPy, RTX 3070 Ti) para perfiles alares, base de un futuro optimizador aerodinámico. Todo el solver vive en `Simulador2D.py` (~8000 líneas, clase `Mesh` + `main()`).

- Malla cartesiana estirada (zona fina alrededor del perfil), advección semi-Lagrangiana, difusión (+WALE LES opcional), proyección de presión multigrid/CG.
- Sólidos por IBM (Immersed Boundary): máscara rasterizada + ghost-cell no-slip (`ibm_wall_mode="ghost_noslip"`).
- **Ejecutar SIEMPRE con `.venv/bin/python`** (el python3 del sistema no tiene CuPy).

## Estado actual (2026-08-04, rama Malla-Variable, commit `8ab2461`)

**VERIFICACIÓN NUMÉRICA CERRADA — documento completo en [`docs/verificacion_numerica.md`](docs/verificacion_numerica.md) (10 secciones, es la fuente de verdad).** Leerlo antes de citar cualquier L/D. Guion del TFG en `docs/PROMPT_TFG.md`.

**Reorganización de resultados**: toda la campaña anterior está archivada en **`results/1eraGranOptimizacion/`** (`convergence_study/`, `verificacion_numerica/`, `agent_tests/`, `barridos/`, `metricas_ga/`, `comparativas_v1/`, `videos_ganadores/`, …). `results/verificacion_numerica/` en la raíz solo contiene lo nuevo (`criterio_parada.json/.png`, `series_dx004/`). Cualquier ruta de este documento o de scripts antiguos que apunte a `results/convergence_study/...` hay que leerla bajo `results/1eraGranOptimizacion/`. El `.gitignore` se reancló a patrones `**/` porque los anclados a `results/` dejaron de aplicar tras el archivado (metían 5585 ficheros y tres >100 MB al índice).

**El criterio de parada recalibrado por tolerancias fue RETIRADO y sustituido.** Historia completa, porque importa para no repetir el error:

- Defecto original: `_detect_series_convergence` comparaba cada serie con la media de su propio último 15 % — preguntaba "¿me parezco a mí mismo hace poco?" en vez de "¿he dejado de cambiar?". Una deriva lenta y monótona lo satisface siempre: paraba en t≈1–2 tiempos convectivos, midiendo flujo no desarrollado, con errores de hasta **47 %** en L/D. Lo enmascaraba el relleno de cola de `cdvector`/`clvector`, que daba σ=0.0 exacto (léase "no hay datos", no "convergencia perfecta"): el 81.8 % de las 3884 evaluaciones de `aprendizaje_ML.jsonl` tienen σ=0.
- El primer recalibrado (`clcd_tol_abs=0.001, clcd_tol_rel=0.01, window_conv_time=2.0`, `calibracion_earlystop.json`) daba 0.82 % de error, pero era **error de entrenamiento sobre 2 series**; al aplicarlo a casos nuevos llegaba a **−23.7 %** con signo impredecible. El defecto estaba en la pregunta, no en los umbrales.
- **Criterio nuevo (en producción)**: mide la pendiente de L/D en ventana móvil — `drift = |pendiente|/|media| < tol_drift`, `noise = std/|media| < tol_noise`, sostenido n chequeos y `t > min_t`. Se evalúa sobre **L/D** (la magnitud que optimiza el GA; Cl y Cd pueden derivar a la vez sin que su cociente lo haga), no sobre Cl y Cd por separado. Parámetros elegidos por **CV de 3 pliegues sobre 15 series completas** entre 864 combinaciones, verificados sobre **5 series apartadas que no intervienen en ninguna decisión**: err máx **1.14 %**, coste 0.56–0.67 del presupuesto. Valores: `clcd_tol_drift=0.005, clcd_tol_noise=0.05, clcd_window_conv_time=1.0, clcd_n_sostenido=1, clcd_min_t_fisico_before_check=5.0` (defaults en `main()` de `Simulador2D.py`; fuente de verdad en fichero: `results/verificacion_numerica/criterio_parada.json`, leído por `verificacion_numerica.criterio_calibrado()` y por `run_convergence_study.py`, que avisa en el log si no lo encuentra).
- **`min_t=5.0` es lo que arregla el modo de fallo restante**: sin él, dos series paraban en t≈4.1–4.4 con +10.6 % y +19.6 % — mesetas falsas previas al reataque de la burbuja laminar. La rejilla lo eligió sola pero la razón es física.
- **El residual de campo no se usa** (`tol_res=inf`): con `drift`+`min_t` no aportaba. Queda instrumentado (`resvector` con change_u/v/p) por si a otros Re importa. `change_v` se descarta como residual: se normaliza por la norma de v, pequeña en flujo casi horizontal, vale ~0.5 permanentemente y dominaría cualquier máximo sin significar nada.
- **Cableado verificado en vivo** (individuo 2019, dx=0.004): offline t_parada=7.013 / L/D=18.716 vs solver 7.0135 / 18.697 (error +0.84 % sobre el asintótico). Ahorro frente a presupuesto fijo: **33–45 %** — una evaluación a dx=0.004 baja de ~1150 s a ~700 s.
- **`clcd_tol_abs` y `clcd_tol_rel` se eliminaron** (no desactivados): cualquier script que los pase falla.

**¿Sobrevive el ranking del GA? Medido, y la respuesta es no en la zona que importa** (`revalidar_ranking.py`, `results/1eraGranOptimizacion/verificacion_numerica/revalidacion_ranking.json`). Muestra estratificada por deciles de n=20 individuos del historial (filtrada a los 2425 registros de dx=0.004; el historial mezcla otra población de 368 a dx=0.001 con mediana muy distinta), re-simulada a dx=0.002 sin parada y con presupuesto fijo:

- Spearman ρ global **0.7955**, Kendall τ 0.6421, Pearson 0.8341, cambio mediano **+65.3 %** (IQR 47–104).
- Separando por la mediana: ρ=0.612 en los 10 peores, **ρ=0.152 en los 10 mejores**. Solape top-4 = **0.25** (top-3 1/3, top-5 2/5, top-8 7/8).
- Casos ilustrativos: **2019** es el mejor real de la muestra (L/D 24.06) y es de **generación 1** — el GA lo rankeó octavo y lo descartó. **2312** (gen 17) es el único con Δ negativo (−10 %): la parada le regalaba Cd bajo; cae del puesto 3 al 13.
- **Aviso metodológico**: con n=12 la lectura parcial era la contraria (top-4 en orden exacto). Los 8 que faltaban eran la zona alta. Una muestra estratificada truncada a mitad de ejecución no es una muestra estratificada.
- Conclusión defendible: el GA **sí se movió al barrio correcto** (4 de 5 individuos de gen≥17 revalidan por encima de 20; ρ generación–L/D real 0.41, p=0.073) pero **eligió la casa con una señal que no discrimina a esa escala**.

**Y dx=0.004 SÍ sirve para rankear — el problema era la parada, no la malla** (`series_dx004.py`, mismos 20 individuos a dx=0.004 con presupuesto fijo y sin parada):

| señal de fitness | ρ global | ρ mitad alta | solape top-4 |
|---|---|---|---|
| fitness que usó el GA (dx=0.004 + parada sesgada) | 0.7955 | 0.152 | 0.25 |
| **dx=0.004, presupuesto fijo / criterio nuevo** | **0.9263** | **0.7455** | **1.00** |

El sesgo en valor absoluto a dx=0.004 es de **−44.9 % ± 68** y no importa para el orden (un desplazamiento común no reordena), pero **obliga a validar el ganador a dx=0.002**. Dos avisos: **1265** da L/D=**−1.29** a dx=0.004 (real 0.559) — el GA nunca vio fitness negativo porque la parada lo enmascaraba, y hay que revisar cómo se convierte fitness en probabilidad de selección; **2312** engaña a las dos mallas gruesas por igual, así que ahí el artefacto es geométrico, no de parada.

**GCI de Roache** (ganador Re=1e5, α=4°, dx=0.008/0.004/0.002/0.001, r=2, sin parada): Cl p=2.18 (coincide con el orden formal 2 del esquema) → **GCI fina ±1.43 %, la cifra defendible**; Richardson h→0: Cl=0.7487, L/D=25.34. **Cd no converge monótonamente** (0.03332→0.02877→0.02927), su GCI (±0.27 %) y el de L/D (±0.26 %) están sobrevalorados — incertidumbre real de Cd ≥ ±1.7 %; los p de 3.18 y 3.64 por encima del orden formal son otra señal de no estar plenamente en rango asintótico. dx=0.008 y dx=0.004 quedan fuera del rango asintótico (GCI grueso 3.3–6.7 %); dx=0.002 sí vale, con ~2.4 % de error de discretización. Validación cruzada del ganador por tres vías: serie larga 24.81, GCI a dx=0.002 24.69, reeval con criterio 24.50.

**Polar del ganador a dx=0.002** (α=0…10): L/D = 13.05 / 21.92 / **23.22** / 19.22 / 16.30 / 8.07, contra NACA0012 0.01 / 7.55 / 13.84 / 12.38 / 8.14 / 6.66. Máximo en α=4° (el punto de diseño), ventaja en todo el rango: mejora de banda ancha, no sobreajuste al ángulo.

**Validación externa contra XFOIL** (NACA0012 Re=1e5, Ncrit=9, airfoiltools; `validacion_xfoil.json`): **la sustentación es buena, la resistencia no.** MAE(Cl)=0.0912, pendiente dCl/dα=0.10211 vs XFOIL 0.11107 y teoría 2π 0.10966 (8 % por debajo, normal con espesor finito). MAE(Cd)=0.036, **error medio en Cd +123.8 %, monótono creciente con α** (−2.4 % a α=0 → +226 % a α=10). Causas plausibles por orden: resolución de capa límite (dx=0.002 con IBM ghost-cell da ~5–10 celdas dentro de la BL a Re=1e5), separación/reataque prematuro de la burbuja, y bidimensionalidad. **Consecuencia: el L/D absoluto NO es comparable con XFOIL ni con experimento, y como el error de Cd depende de α tampoco es un factor de escala divisible.** Lo que sí se sostiene: Cl, pendiente de sustentación, y comparación perfil-contra-perfil a igual α y misma malla — que es exactamente lo que hace el GA.

**Transición SA-BC implementada y validada** (commit `8e29bd5`): `transition_model="sa_bc"` + `freestream_Tu=0.1` en `main()` (default `"none"`, requiere `turb_model="sa"`). Intermitencia algebraica de Bas-Cakmakcıoğlu 2016: γ = 1−exp(−√T1−√T2) con Re_θc por la correlación de Menter, modulando **solo la producción** de SA; constantes `SA_BC_CHI1=0.002`, `SA_BC_CHI2=5.0`; γ expuesta como diagnóstico en `Mesh.sa_gamma`. Validada en `results/1eraGranOptimizacion/agent_tests/sabc_*` (Cd laminar correcto, LSB reataca, simetría α=0 intacta) y usada por las corridas de GA posteriores.

**Campaña de GA "1eraGranOptimizacion" completada** (commits `74add38`, `43d95e9`): modo `--mixto <tag>` en `run_convergence_study.py` (GA único de población mixta estilo Arm B, parada por estancamiento, pausable/reanudable, lanzador `scripts/agent_tests/mixto.sh`), 4 semillas nuevas de airfoiltools aptas a bajo Re (`SD7037`, `E387`, `SG6043`, `MH32`), flags `--multipunto/--delta-angulo/--fitness-modo` y `--transition-model/--freestream-tu`, `--run-tag` para no pisar corridas. Corridas: `mixto_masgen` (más generaciones, ganador `NACA_0012_sharp_Re100000_a4.0_LD18.18.dat`), `mixto_masperfiles` (8 semillas, LD16.13), `mixto_exprimir_re1e5`, `mixto_re1e3/re1e6/re1e7`, `islands_sabc/` (islas con SA-BC) y `refine_todos.json` con los refinados a dx=0.002. **Todos esos L/D son de la señal sesgada; no citar** (ver §10 del doc).

**Métricas del GA** (`scripts/agent_tests/metricas_ga.py`, sin GPU, ~1 min): 15 figuras + `INFORME.md`. Hallazgo: `usar_ia=False` en los 6 estudios, el surrogate RandomForest nunca filtró nada. Reentrenado offline sobre 2793 muestras Re=1e5: R²=0.850 (5-fold CV), MAE=1.07; a percentil 70 evitaría el 69 % del CFD perdiendo solo el 2.5 % de la élite (≈68 h-GPU desperdiciadas).

**Qué sigue siendo válido de la campaña anterior**: la metodología (GA de islas con migración, parametrización, solver, `wall_treatment="consistent"`, SA-BC), el coste (98.6 h-GPU, 3884 evaluaciones) y las conclusiones sobre el comportamiento del GA (papel de la migración, diversidad, presión selectiva, PCA: PC1+PC2 explican 75.5 % de la varianza geométrica). **Qué no**: todo valor absoluto de L/D anterior al recalibrado en sus tres capas (exploración dx=0.004, refinado dx=0.002 y los 3884 registros de `aprendizaje_ML.jsonl`), la afirmación de que el ranking se conserva, los ganadores como óptimos, y el L/D como magnitud comparable con la literatura.

**Cambios de código de esta tanda** (`8ab2461`): `Simulador2D.py` — `tvector` (eje de tiempo físico por muestra; el dt es adaptativo y el tiempo no se reconstruye desde el índice de iteración), `resvector`, `n_muestras_validas` + truncado real en lugar del relleno de cola, `_detect_series_convergence` reescrita sobre drift/noise aplicada a L/D. `scripts/RunGA.py` — `dump_series_path` vuelca t/Cl/Cd/(L/D) a `.npz`, ventana de promediado con mínimo de 10 muestras, `CI95`/`converged_clcd`/`t_conv_clcd`/`iters_efectivas`/`cl_cp_discrepancy` en el retorno. Scripts nuevos: `criterio_parada.py`, `revalidar_ranking.py`, `series_dx004.py`, `verificacion_numerica.py` (etapas `--calibrar/--reeval/--gci/--polar/--repolar/--validacion`, `criterio_calibrado()`, `T_TARGET_CASO`), `metricas_ga.py`, `video_ganador.py`, colas `cola.sh`, `cola_tfg.sh`, `cola_final.sh`. `T_TARGET_CASO` existe porque el horizonte no es universal: a Re=1e3 el L/D no baja del 1 % de error hasta t≈15 (26000 iters), a Re≥1e5 basta t=8.

**`cl_cp_discrepancy` cayó de 0.31–0.83 a 0.02–0.14** al medir sin transitorio: buena parte de la sobre-circulación diagnosticada era artefacto. Queda un residuo real, menor, que crece con Re.

**Árbol de trabajo limpio**: todo commiteado en `8ab2461`; solo queda `RESUMEN.md.tmp` sin seguir (artefacto del hook de regeneración). Los pendientes de limpieza de julio (rename `core_v0x`→`v0x_ic`, `--smoke`/Cl_cp en scripts de barrido, `prueba_inicialesdiferentes/`, `comparativas_v1/`, reorganización de `convergence_study/`) están resueltos vía el archivado en `results/1eraGranOptimizacion/`.

## Estado histórico (2026-07-12, rama Malla-Variable)

Caso de trabajo: NACA0012, α=5°, Re=1e5, Cl físico esperado ~0.55.

**FIX DE RAÍZ IMPLEMENTADO: `wall_treatment="consistent"`** (kwarg de `main()`, default "legacy" intacto). El sumidero IBM está eliminado. Componentes (todas reutilizan piezas existentes):
1. Divergencia por caras en la proyección (`divergence_form="face_flux"` → `_compute_flux_divergence_field_uv`): cara fluido-sólido = flujo 0 exacto → el Poisson ve el flujo contra la pared.
2. Gradiente de presión one-sided en celdas de pared (`wall_pressure_gradient_mode="one_sided"`, kernel ya existía).
3. `reforzar_impermeabilidad` desactivado (`disable_reforzar=True`) — era la cirugía que borraba masa (Q≈-0.06·U·c).
El Laplaciano masked ya era Neumann correcto; el problema era que divergencia y corrección anulaban la componente entera junto al sólido (proyección ciega en pared).

**Resultados validados** (JSON+npz en `results/1eraGranOptimizacion/agent_tests/wc_*`; commiteados en `a5c14b7`, npz >100MB excluidos por `.gitignore`):
- `wc_re1e3_a5` (Re=1e3, α=5): Cl=0.211±0.0004, Cd=0.148, Q_lazo=+0.02, ΔCp_TE=0.04. **Clava el DNS de referencia (Kurtulus Re=1000: Cl~0.22, Cd~0.146)**. Legacy daba 0.14 con Q=-0.06.
- `wc_sa_a0` (α=0, SA): Cl=0.0003, sin NaN — simetría perfecta.
- `wc_sa_a5_dx2` (α=5, Re=1e5, SA, dx=0.002, turbo_hd): **estable sin colapso**, Kutta cierra (ΔCp_TE=0.012), pero Cl=0.274. Q_lazo=+0.026 (MG turbo saturado).
- `wc_sa_a5_dx2_full` (sin turbo: div=0.02, outer=8, niveles=2, 24k iters): Q_lazo=0.006 ✓ pero Cl=0.312. **Hipótesis Q→Cl pendiente ~20 FALSIFICADA**: ΔQ=0.020 predecía ΔCl≈+0.4, medido +0.04.
- `wc_sa_a5_dx2_long` (div=0.01, outer=12, 48k iters, t=18.3 conv, 2h GPU): Q=0.0032, **Cl CONVERGE en plateau 0.32** (transitorio: pico 0.40 en t≈2, mínimo 0.312 en t≈10, luego estabiliza 0.320 con deriva 0.0025). cl_cp=0.383, cl_circ=0.281 (dispersión estimadores cerrando). x_succión mejoró a 0.045. **PROYECCIÓN EXONERADA: Q→0 no recupera el Cl.**
- **Diagnóstico del déficit (0.32 vs 0.55)**: SA sobre-difunde. Evidencia: BL en x/c=0.5 espesor ~0.07c (3× placa plana turbulenta Re=1e5), χ=nu_t/nu~40-50 en BL (teórico ~8), Cp_min=-0.81 (esperado ~-1.9), Cd=0.048 (alto). Causas candidatas: (a) SA fully-turbulent desde LE a Re=1e5 (flujo real laminar hasta ~x/c=0.5, penaliza succión), (b) primera celda a y+≈10 sin wall function (SA quiere y+≲1). Distancia de pared SA verificada correcta; constantes SA estándar.
- Hallazgo menor: `mg_modo_turbo_hd` pisaba `divergencia` (arreglado: `min(divergencia, 0.05)`); runs `wc_sa_a5_dx2` y `_t02` idénticos (el fix no cambió nada porque turbo satura outers igual).
- Añadido kwarg `sa_nu_tilde_factor` a `main()` (default 3.0; bajo ~0.1 retrasa transición SA).
- `wc_sa_a5_dx2_lam01` (sa_nu_tilde_factor=0.1, t=14.2 conv): **PEOR** — Cl=0.266, x_succión=0.14 (LSB reapareciendo), Cp_min=-0.66, Cd=0.056; χ_max=184 (transiciona igual pero tarde). **Falsificada hipótesis (a) fully-turbulent**; retrasar transición reactiva la burbuja laminar. Mantener sa_nu_tilde_factor=3.0. (La transición se resolvió después con SA-BC, ver Estado actual.)
- **Sospecha cuantificada: difusión numérica del semi-Lagrangiano bilineal.** nu_num≈α(1-α)·dx²/(2dt)~1e-4 cerca de pared = 10× nu molecular a Re=1e5 (a Re=1e3 era ~10% de nu → por eso el DNS clavó). Escala ∝dx (dt∝dx por CFL). Junto con y+≈10, motiva test de resolución.
- `wc_sa_a5_dx1` (dx=0.001, 60k iters, t=11 conv, 3.6h GPU): **Cl=0.389** (vs 0.32 a dx=0.002), Cd=0.039, Cp_min=-0.96, χ_max=105, estable. **Camino resolución confirmado, error ~1º orden en dx.** Extrapolando dx→0: Cl≈0.46. dx=0.0005 no cabe en la 3070 Ti.
- **Causa dominante identificada: difusión numérica del SL bilineal, nu_num≈u·dx/2 (independiente de dt)** — 1º orden, ~5-10× nu molecular cerca de pared a Re=1e5. A Re=1e3 era ~10% de nu (por eso clavó DNS).
- **MacCormack implementado** (`advection_scheme="maccormack"` en main(), default "sl"): corrección Selle 2008 (trace forward + φ*+(φ-φ**)/2) con limitador min/max del stencil y **banda de 2 celdas junto al sólido en SL puro** (sin banda, la corrección sobre el staircase mete pico espurio Cp=-11 en TE; con banda, α=0 perfecto Cp_min=-0.47). Validación Re=1e3 α=5: Cl=0.245/Cd=0.155 (SL: 0.211/0.148, DNS: 0.22/0.146) — ligero overshoot coherente con menos difusión en malla coarse, física intacta.
- `wc_mc_sa_a5_dx2` (dx=0.002, SA, MacCormack, t=11.5 conv): **Cl=0.452** (SL daba 0.32), Cp_min=-1.20, x_succión=0.015 (posición física del LE ✓), χ_max=73, Cd=0.040, std(Cl)=1e-5 (estacionario casi a máquina, deciles planos desde t≈4, sin sangrado). **Cuarta capa (nu_num del SL) CONFIRMADA y resuelta.** Coste MacCormack: ~4% vs SL.
- `wc_mc_sa_a5_dx1` (dx=0.001 + MacCormack, t=9.7 conv, 3.6h): **Cl=0.496 ✓ OBJETIVO CUMPLIDO** (criterio [0.4,0.7]), Cd=0.0335 (físico ~0.03 ✓), Cp_min=-1.24, x_succión=0.02 ✓, Q=0.005 ✓, deriva 0.001. **Richardson dx→0: Cl≈0.54 — clava el físico 0.55.** Convergencia de malla limpia (0.452@dx2 → 0.496@dx1).
- **CONFIG DE REFERENCIA DEL OPTIMIZADOR**: `wall_treatment="consistent"` + `turb_model="sa"` + `advection_scheme="maccormack"` (+ `transition_model="sa_bc"` desde julio 2026). dx=0.002 para validación del ganador, dx=0.004 legítimo para explorar/rankear (ver Estado actual §7 del doc).
- Regresión verde tras MacCormack (5 passed; defaults intactos, "sl" sigue siendo default).
- **POLAR VALIDADA** (`wc_mc_polar_a0/a2/a8` + `wc_mc_sa_a5_dx2`, dx=0.002 MacCormack): Cl(α)= 0(0.000), 2(0.185), 5(0.452), 8(0.708); Cd= 0.0305/0.0323/0.0404/0.0624. Monotonía ✓, pendiente 0.088/deg ✓, simetría α=0 exacta ✓, sublinealidad física en α=8 (pre-stall), Cp_min=-2.13 y succión x/c=0.015 en α=8. **DIAGNÓSTICO DE 4 CAPAS CERRADO COMPLETO.**
- **Script de barrido CFL/dx/α**: `scripts/agent_tests/run_cfl_sweep.py` — cola resumible de 30 sims (α=0/2/5/8/10 × CFL=0.25/0.5/0.75 × dx=0.004/0.002) con la config de referencia, iteraciones auto-escaladas a t=12 convectivos, métricas JSON por sim, npz de estado, frames, streamlines, series temporales. Reanuda desde la última completada; CSV resumen + plots. **Aún no ejecutado.**

**Diagnóstico histórico cerrado (4 capas, evidencia en `results/1eraGranOptimizacion/agent_tests/`)**:
1. LSB laminar a Re≥1e4 (burst t≈5-7 convectivos, colapso). Fix: **Spalart-Allmaras** (`turb_model="sa"`). WALE inerte.
2. Pared sub-resuelta: SA necesita dx≤0.002 (y+≲10).
3. Inconsistencia IBM↔proyección: legacy absorbía masa (sobre-circulación 2.2×); compatible_flux desangraba vorticidad (Cl→0.1). **Resuelta con wall_treatment="consistent"**.
4. Difusión numérica del SL bilineal (nu_num≈u·dx/2, 1º orden): BL 3× gruesa, Cl 0.32 vs 0.55, transitorios de 18 conv. **Resuelta con advection_scheme="maccormack"** (coste ~4%, estacionario en t≈4-5 conv).
- Métrica decisiva: Q_lazo=∮u·n dl (runner la guarda como `q_loop_025c/075c`).

**Añadido sesiones previas**: monitor Kutta (`kutta_dcp_vector`), SDF polígono (`ibm_sdf_source="polygon"`), SA completo, Kutta explícita opcional (`kutta_enforce`), fix acumulación de presión MG compatible_flux (CG ya estaba bien), checkpoints.

**GA de optimización de perfiles (`scripts/RunGA.py`)**: `simular_perfil` corre sobre consistent+SA+MacCormack, `main(config)` reutilizable con retorno dict, población mixta multi-semilla (`archivos_base` + `resamplear_a_grid` para cruce entre perfiles de distinto nº de puntos), parada por `tiempo_limite_s`, mutación con sigma adaptativa, parada por estancamiento (`parada_estancamiento`, `paciencia_generaciones`, `tol_mejora_fitness`), checkpoint de genes de toda la población en `estado_ga.json` (reanudación mid-corrida). Orquestador `scripts/agent_tests/run_convergence_study.py` (deadline-driven, resumible): Fase0 calibración dx, Arm A GA por-semilla, Arm B población mixta, S2 islas con migración (`--islands`), S3 refine top-k (`--refine`), `--aislado`, `--mixto`, `--multipunto`. Memoria de ML persistente: `cerebro_aerodinamico.pkl` + `aprendizaje_ML.jsonl`.

**Estudio de convergencia (CERRADO, `a0ee4f3`)**: Arm A no converge (4 semillas → 4 óptimos locales, CV=0.172); T1 mostró inversión de ranking dx=0.004↔dx=0.002 con la señal antigua (**releer a la luz de §7 del doc actual: con medida honesta dx=0.004 rankea bien, ρ=0.93**); el test aislado sin migración (28 h GPU, 4 semillas hasta estancamiento) da dist-forma media **0.0557** (igual/peor que la partida 0.0507) y spread L/D 2.44 → **la convergencia que producen las islas la fuerza la MIGRACIÓN, no una física de óptimo único**. Ganadores en `results/1eraGranOptimizacion/convergence_study/aislado_sin_migracion_convergido/`. Fixes asociados: `bf6824e` (distinguir semilla original del padre en `comparativa_perfiles.png`), `ac235b1` (organizar salidas en base_study/islands/tests), `9309f00` (S3 no sobrescribe `refine_dx002.json` cuando no hay refinados), `30c7cf6` (aislado pausable + `aislado.sh`).

## Próximos pasos

**Prioridad tras la verificación numérica (2026-08-04)**

1. **Relanzar la optimización con el criterio nuevo.** Consecuencia directa de §6: el GA anterior optimizó una señal que no ordena en la zona alta. Ya no hace falta rediseñar el presupuesto a ciegas — §7 justifica explorar a **dx=0.004** (ρ=0.93 global, 0.75 en la mitad alta) y §8 baja una evaluación a ~700 s. Escalas sobre la mesa (la campaña anterior fueron 98.6 h-GPU):

   | configuración | evaluaciones | h-GPU |
   |---|---|---|
   | pop 16 × 25 gen × 1 isla | 400 | 78 |
   | pop 16 × 25 gen × 4 islas | 1600 | 311 |
   | pop 24 × 40 gen × 4 islas | 3840 | 747 |

   El estudio de convergencia concluyó que la migración es el motor del GA → una isla sola probablemente no sirve. Validar siempre el ganador a dx=0.002.
2. **Manejar fitness negativo antes de lanzar**: el individuo 1265 da L/D=−1.29 a dx=0.004. El GA nunca vio negativos (la parada prematura los enmascaraba); revisar la conversión fitness→probabilidad de selección.
3. **Diagnosticar el exceso de Cd** (+226 % a α=10 frente a XFOIL): perfil de capa límite en pared a dx=0.001 vs dx=0.002, y posición de reataque de la burbuja frente a lo que predice XFOIL. Decide entre las hipótesis 1 y 2 de §5.2 (resolución de BL vs separación prematura).
4. **GA multipunto α∈{2,4,6}**: se descartó cuando una evaluación costaba ~17 min a dx=0.002 con presupuesto fijo; con dx=0.004 y parada por meseta baja a ~12 min por los tres ángulos, así que vuelve a estar sobre la mesa. Etapa `multipunto` ya cableada en `cola_tfg.sh` (fuera de la cola por defecto, hay que lanzarla a mano).
5. **Activar el surrogate** (`usar_ia=True`, umbral percentil 70): ahorro estimado del 69 % del CFD con pérdida del 2.5 % de la élite. Reentrenar sobre datos medidos con el criterio nuevo, no sobre el historial sesgado.

**Anteriores (siguen abiertos)**

6. Estudio de convergencia CERRADO: no relanzar corridas de diagnóstico de la misma pregunta. Direcciones abiertas: más diversidad/multi-arranque si se quiere un óptimo global real; revisar físicamente el óptimo de camber alto (Cl~1.07 a α=4, semilla s1014).
7. **Ejecutar el barrido** `scripts/agent_tests/run_cfl_sweep.py` (30 sims, resumible) para caracterizar coste/error CFL×dx×α — sigue pendiente, ninguna sim lanzada.
8. Investigar el error no fatal de carga de `cerebro_aerodinamico.pkl` ("inhomogeneous shape") visto en el log de "reforzado" — no bloquea (se reinicia la memoria), pero pierde el histórico.
9. Opcional barato: probar MacCormack+turbo_hd (~19 it/s esperado) para pre-screening si el sesgo por Q=0.026 resulta consistente entre geometrías.
10. Opcional física: sublinealidad α=8 (deriva -0.002, TE separación creciendo) — vigilar si el optimizador explora α altos.
11. Actualizar rutas en scripts/notebooks que apunten a `results/convergence_study/...` o `results/verificacion_numerica/{gci,polar,reeval_*}.json` — ahora viven bajo `results/1eraGranOptimizacion/`.

## Tests
- `scripts/agent_tests/cola_tfg.sh {run [etapa]|status}` — cola serie reanudable: `calibrar → reeval → gci → polar` (+ `multipunto`, fuera de la cola por defecto). Cada etapa cachea sus simulaciones en JSON; Ctrl+C y relanzar `run` continúa.
- `scripts/agent_tests/cola_final.sh` — encadena `repolar → validacion → revalidacion`. `scripts/agent_tests/cola.sh` — cola genérica de corridas de GA.
- `scripts/agent_tests/verificacion_numerica.py --calibrar|--reeval|--gci|--polar|--repolar|--validacion` — etapas sueltas; `criterio_calibrado()` y `T_TARGET_CASO` viven aquí.
- `scripts/agent_tests/criterio_parada.py` — ajuste por CV de 3 pliegues + verificación sobre conjunto retenido del criterio de parada → `results/verificacion_numerica/criterio_parada.{json,png}`.
- `scripts/agent_tests/series_dx004.py` — 20 series completas a dx=0.004 sin parar: referencia del criterio y test de ranking entre mallas.
- `scripts/agent_tests/revalidar_ranking.py --n 30` — muestreo estratificado del historial y correlación de rangos; cacheado y reanudable.
- `scripts/agent_tests/metricas_ga.py` — post-proceso del GA sin GPU (~1 min): 15 figuras + `INFORME.md`.
- `scripts/agent_tests/mixto.sh` — lanzador de los GA de población mixta (`--mixto <tag>`), pausable/reanudable.
- `scripts/agent_tests/video_ganador.py` — vídeo del campo del ganador. `plot_comparacion_transicion.py` — comparativa SA vs SA-BC. `validar_aislado_dx002.py` — revalidación a dx=0.002 de los ganadores del test aislado.
- `scripts/agent_tests/run_kutta_tests.py {smoke|smoke_polygon|baseline_a5|polygon_a5|compare|regression}` — runs coarse (dx=0.004, ~5-15 min) con criterios cuantitativos.
- `scripts/agent_tests/run_cfl_sweep.py` — barrido CFL/dx/α resumible (pendiente de ejecutar).
- `scripts/agent_tests/run_convergence_study.py --aislado` — test de convergencia por semilla sin migración (lanzador `aislado.sh`, subcomando `status`). **Estudio cerrado**; no relanzar salvo para direcciones nuevas.
- `scripts/comparativa_barrido_completo.py --smoke` — validación rápida del pipeline outer_sum vs XFoil (1 perfil, 1 Re, 3 alphas, malla gruesa, 50 iters) sin gastar GPU real.
- `pytest tests/ --ignore=tests/test_generacion_geometrica_ga.py` (5 passed; el de GA está roto pre-existente: importa `RunGA` inexistente).

## Diagnósticos clave (métodos de Mesh)
- `compute_circulation()` — Γ en lazos, Cl_circ=-2Γ/(U·c).
- `compute_cp_diagnostics(mu, rho)` — Cl(∮ΔCp), ΔCp_TE, pico de succión.
- `compute_kutta_dcp_instant(face_data)` — ΔCp_TE instantáneo para el monitor.
- `debug_te_report()` — ghosts/irreparables cerca del TE.
- `plot_streamlines()`, `plot_forces_over_time()`, `plot_cp_vs_chord()`.
- Series por muestra: `tvector` (tiempo físico), `clvector`/`cdvector`, `resvector` (change_u/v/p), `n_muestras_validas`.

## Config de referencia (fina, `__main__`)
Lx=12, Ly=8, dx_min=0.001 (malla 3243×1203, ~2 it/s), CFL=0.25, NACA_0012_sharp, min_te_height_factor=1.0, turbo_hd, wake long_fine_x, `wall_treatment="consistent"`, `turb_model="sa"`, `transition_model="sa_bc"`, `advection_scheme="maccormack"`. Parada por convergencia con los defaults nuevos (`clcd_tol_drift=0.005`, `clcd_tol_noise=0.05`, `clcd_window_conv_time=1.0`, `clcd_min_t_fisico_before_check=5.0`). **Nota**: el bloque `__main__` puede quedar como scratch de pruebas entre sesiones — usarlo como plantilla, no copiarlo literal.

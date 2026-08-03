# Graph Report - .  (2026-08-02)

## Corpus Check
- Large corpus: 5623 files · ~15,320,356 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 1295 nodes · 2678 edges · 73 communities (67 shown, 6 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 24 edges (avg confidence: 0.64)
- Token cost: 118,000 input · 9,500 output

## Community Hubs (Navigation)
- Verificación numérica y GCI
- Comparativa con XFOIL
- Tests de malla y proyección
- Diagnóstico de capa límite
- Diagnóstico presión-lift
- Validación de lift NACA0012
- Diagnóstico lift NACA0012
- Métricas del GA
- Presión y modelo Spalart-Allmaras
- Núcleo del algoritmo genético
- Editor interactivo de perfiles
- Divergencia y solver CG
- Visualización de campos
- Bucle principal y fronteras
- Runners de tests de agente
- Barrido de modos multigrid
- Corrección de capa límite
- GUI principal
- Solver cartesiano: núcleo
- Exploración de hijos por mutación
- Malla multirresolución
- Solver cartesiano: malla
- Canvas de visualización GUI
- Estudio de convergencia del GA
- Parametrización geométrica y dataset
- Diagnóstico de drag de presión
- Barrido de condiciones de vuelo
- Solver cartesiano: proyección
- Predicción y reconstrucción de perfiles
- Paneles de métricas GUI
- Widgets auxiliares GUI
- Validación de simetría
- Barrido de ángulo de ataque
- Capa límite y tratamiento de pared
- Paneles de parámetros GUI
- Mutación y proyección de perfiles
- Vídeo de simulación larga
- Entrenamiento de modelos IA
- Sólidos e IBM ghost-cell
- Tests de agente auxiliares
- Barrido de resolución
- Solver cartesiano: fuerzas
- Teoría numérica del solver
- Pipeline de IA predictiva
- Memoria compartida GUI
- Solver cartesiano: kernels
- Advección e interpolación
- Benchmark multigrid
- Solver cartesiano: fronteras
- Solver cartesiano: utilidades
- Oráculo aerodinámico Random Forest
- Tests de generación geométrica
- Teoría del optimizador genético
- Paneles de corrección BL
- Mutación paramétrica
- Solver cartesiano: métricas
- Fuerzas aerodinámicas y polar
- Divergencia enmascarada
- Cola de etapas del TFG
- Evaluación de población
- Visualización PCA del espacio de diseño
- Estudios de población mixta
- Validaciones de agente
- Visor en vivo
- Lector de logs GUI
- Test T1 de ranking por dx
- Estudio aislado sin migración
- Plots de agente
- Conversión polar-cartesiano
- Utilidades sueltas
- Utilidades sueltas

## God Nodes (most connected - your core abstractions)
1. `Mesh` - 96 edges
2. `main()` - 73 edges
3. `Mesh` - 55 edges
4. `MainWindow` - 28 edges
5. `main()` - 25 edges
6. `main()` - 24 edges
7. `compute_corrected_forces()` - 22 edges
8. `main()` - 20 edges
9. `main()` - 19 edges
10. `main()` - 18 edges

## Surprising Connections (you probably didn't know these)
- `Modelo surrogate forward (geometría → Cl, Cd)` --semantically_similar_to--> `OraculoAerodinamico`  [INFERRED] [semantically similar]
  Base Teorica.txt → scripts/RunGA.py
- `Configuración ideal de mutación (camber/espesor std)` --references--> `descomponer_camber_espesor()`  [INFERRED]
  docs/analisis_configuracion_ideal.md → scripts/RunGA.py
- `Relleno de cola: σ=0 enmascaraba el defecto` --rationale_for--> `simular_perfil()`  [EXTRACTED]
  docs/verificacion_numerica.md → scripts/RunGA.py
- `tvector: eje de tiempo físico de las series` --references--> `simular_perfil()`  [EXTRACTED]
  docs/verificacion_numerica.md → scripts/RunGA.py
- `Filtro predictivo Random Forest del GA` --implements--> `OraculoAerodinamico`  [EXTRACTED]
  Base Teorica.txt → scripts/RunGA.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Cadena de diagnóstico de la sobre-circulación (4 capas)** — resumen_burbuja_separacion_laminar, resumen_modelo_spalart_allmaras, resumen_wall_treatment_consistent, resumen_maccormack_adveccion, base_teorica_difusion_numerica, resumen_metrica_q_lazo [EXTRACTED 1.00]
- **Cadena de verificación: defecto → recalibrado → GCI → validación cruzada** — docs_verificacion_numerica_parada_anticipada_sesgada, docs_verificacion_numerica_sigma_cero_enmascarante, docs_verificacion_numerica_tvector_eje_tiempo, docs_verificacion_numerica_barrido_108_combinaciones, docs_verificacion_numerica_criterio_recalibrado, docs_verificacion_numerica_gci_roache, docs_verificacion_numerica_validacion_cruzada_tres_vias [EXTRACTED 1.00]
- **Paso temporal completo del solver (advección → difusión → proyección)** — base_teorica_adveccion_semilagrangiana, base_teorica_modelo_wale, base_teorica_multigrid_vcycle, base_teorica_ibm_ghost_cell, base_teorica_condicion_cfl [EXTRACTED 1.00]

## Communities (73 total, 6 thin omitted)

### Community 0 - "Verificación numérica y GCI"
Cohesion: 0.06
Nodes (53): Sensibilidad de Cl/Cd a la resolución de malla, Barrido offline de 108 combinaciones del criterio, Cola serie reanudable de verificación, Criterio de parada recalibrado, dx=0.004 fuera del rango asintótico: el GA exploró ahí, Extrapolación de Richardson, GCI de Roache (Grid Convergence Index), T_TARGET_CASO: el horizonte temporal no es universal (+45 more)

### Community 1 - "Comparativa con XFOIL"
Cohesion: 0.07
Nodes (50): fig_combinado(), fig_factor(), interp_xfoil(), load_sim(), load_xfoil(), main(), poly_fit(), print_resumen() (+42 more)

### Community 2 - "Tests de malla y proyección"
Cohesion: 0.05
Nodes (39): Condición CFL y paso temporal adaptativo, Malla cartesiana de densidad variable, Mesh, requires_cuda, make_video(), Path, Corre un perfil ganador (masgen Re=1e5 o re1e3) con guardado de frames y genera…, run() (+31 more)

### Community 3 - "Diagnóstico de capa límite"
Cohesion: 0.14
Nodes (45): Case, boundary_layer_audit(), build_base_config(), build_cases(), Case, clean(), cp_profile_audit(), create_videos() (+37 more)

### Community 4 - "Diagnóstico presión-lift"
Cohesion: 0.14
Nodes (45): _baseline_case_id(), build_cases(), build_le_comparison(), build_sensitivity_rows(), Case, _case_row(), classify_case(), configure_output_dir() (+37 more)

### Community 5 - "Validación de lift NACA0012"
Cohesion: 0.13
Nodes (42): build_cases(), build_config(), Case, diagnose(), find_row(), finite_or_nan(), history_rows(), main() (+34 more)

### Community 6 - "Diagnóstico lift NACA0012"
Cohesion: 0.14
Nodes (42): attach_hints(), attach_lift_diagnostics(), build_cases(), build_legacy_cases(), build_reduced_compare_templates(), Case, case_hint(), cuda_available() (+34 more)

### Community 7 - "Métricas del GA"
Cohesion: 0.11
Nodes (37): cargar_estudios(), cargar_evaluaciones(), col(), fig_convergencia(), fig_convergencia_cfd(), fig_convergencia_normalizada(), fig_correlaciones(), fig_coste() (+29 more)

### Community 8 - "Presión y modelo Spalart-Allmaras"
Cohesion: 0.08
Nodes (16): Mesh, Estado de los ghost cells cerca del TE: nº, irreparables y rango de d_ghost., points: lista de tuplas (i,j) en índices de celda (fila, columna). p_value:…, Aplica -(dt/rho)*grad(p^n) al campo de velocidad (paso predictor del método…, Constructor y basicos, IC/BC fully-turbulent: nu_tilde=3*nu en fluido, 0 en sólido.…, Distancia física a pared cacheada (geometría fija). sd en celdas (banda…, Fuentes (producción explícita, destrucción point-implicit) + difusión de… (+8 more)

### Community 9 - "Núcleo del algoritmo genético"
Cohesion: 0.09
Nodes (32): construir_restricciones_geometricas(), cruce(), generar_directorio_resultados(), generar_stem_optimizado(), guardar_estado_ga(), guardar_perfil(), guardar_perfil_con_metadata(), guardar_top_perfiles() (+24 more)

### Community 10 - "Editor interactivo de perfiles"
Cohesion: 0.09
Nodes (17): EditorPerfiles, main(), Editor Interactivo de Perfiles Aerodinámicos Permite cargar, editar y guardar…, Configurar la visualización interactiva con todos los puntos editables, Editor interactivo de perfiles aerodinámicos. Controles: - Click izquierdo +…, Actualizar título con información completa, Evento cuando se selecciona un punto, Evento de click del ratón (+9 more)

### Community 11 - "Divergencia y solver CG"
Cohesion: 0.09
Nodes (17): Pre-computa jerarquía multigrid: máscaras, h2, buffers por nivel., Anula componente normal de velocidad en la capa de fluido adyacente al sólido., Aplica condición IBM Ghost-Cell: impone u=0 en la posición exacta de la pared.…, Aplica dp/dn=0 en los bordes donde NO haya Dirichlet. Importante: la proyección…, Reserva y reutiliza buffers de caras para la proyección por flujo., Construye velocidades normales en caras para la ruta compatible por flujo. -…, Divergencia por balance de flujo en volúmenes de control centrados., Divergencia por flujo construyendo primero velocidades normales en caras. (+9 more)

### Community 12 - "Visualización de campos"
Cohesion: 0.07
Nodes (18): analizar_checkpoint(), generar_graficos_y_outputs(), Visualiza la magnitud de la velocidad en colores (sin flechas). - cmap:…, Visualiza la magnitud de velocidad y superpone flechas (quiver) con dirección y…, (x_le, x_te, y_lo, y_hi, chord, yc) del sólido en coords físicas., Bilinear de field_np (ny,nx) sobre malla tensor-product (x1d,y1d, crecientes)…, Circulación Γ = ∮ u·dl (sentido antihorario) en lazos rectangulares que rodean…, Líneas de corriente sobre fondo |u|, con varios encuadres para diagnosticar el… (+10 more)

### Community 13 - "Bucle principal y fronteras"
Cohesion: 0.08
Nodes (14): main(), SDF exacto punto-a-segmento al polígono _airfoil_polygon_x/y en una banda…, Cambia el ángulo de ataque modificando la dirección del flujo. Asume que la…, side: left/right/top/bottom ; bc_type: noslip/inflow/outflow/slip, Aplica condiciones de frontera. Parámetros: after_projection: Si True, no…, Difusión viscosa de u,v con modelo WALE de viscosidad turbulenta y viscosidad…, Calcula la viscosidad sub-malla ν_t según el modelo WALE (Nicoud & Ducros).…, Celdas fluidas en la franja aguas abajo del TE a lo largo de la bisectriz de la… (+6 more)

### Community 14 - "Runners de tests de agente"
Cohesion: 0.20
Nodes (24): analizar_convergencia(), calib_path(), calibrar(), _camber_espesor(), _cargar_ganador(), ejecutar_aislado(), ejecutar_estudio(), ejecutar_islas() (+16 more)

### Community 15 - "Barrido de modos multigrid"
Cohesion: 0.15
Nodes (24): fig_comparacion_modos(), fig_convergencia_malla(), fig_polar(), fig_resumen_grid(), fig_velocidad(), generate_figures(), get_curve(), load_results() (+16 more)

### Community 16 - "Corrección de capa límite"
Cohesion: 0.17
Nodes (21): _apply_pressure_debias(), _cf_head(), cl_from_delta_cp(), compute_geometric_pressure_forces(), _h1_from_h(), _h_from_h1(), _h_from_lam(), _head_rhs() (+13 more)

### Community 17 - "GUI principal"
Cohesion: 0.19
Nodes (3): QMainWindow, MainWindow, Guarda los parámetros actuales como defaults en gui_settings.json

### Community 18 - "Solver cartesiano: núcleo"
Cohesion: 0.12
Nodes (16): _generar_graficos_polar(), generar_graficos_y_outputs(), _guardar_punto_polar(), load_complete_checkpoint(), main(), Visualiza la magnitud de la velocidad en colores (sin flechas). - cmap:…, Grafica Cd y Cl con selector interactivo de rango para calcular la media. Usa…, Grafica el número de ciclos multigrid usados en cada paso temporal. Útil para… (+8 more)

### Community 19 - "Exploración de hijos por mutación"
Cohesion: 0.15
Nodes (19): area_poligono(), construir_pool_inicial(), extraer_metricas(), flatten_resumen_escenario(), generar_hijos_escenario(), guardar_csv_dicts(), guardar_plot_muestras(), main() (+11 more)

### Community 20 - "Malla multirresolución"
Cohesion: 0.10
Nodes (11): MallaMultiRes, Parámetros: Lx, Ly: dimensiones del dominio (m) dx_grueso: resolución de malla…, Convierte índices de malla fina a índices de malla gruesa. Retorna: (j_grueso,…, Retorna arrays 2D con índices gruesos para cada punto fino. Útil para…, Retorna límites físicos de malla fina., Ajusta coordenadas de sólido de sistema grueso a sistema fino. Retorna:…, Visualiza la configuración de mallas y máscaras. Parámetros: mostrar_pesos: si…, Kernel GPU para promediar malla fina → gruesa con filtrado de sólido y clamping. (+3 more)

### Community 21 - "Solver cartesiano: malla"
Cohesion: 0.14
Nodes (10): Aplica dp/dn=0 en los bordes donde NO haya Dirichlet. Importante: la proyección…, Calcula el campo completo de divergencia del^2 u = du/dx + dv/dy. Usa stencil…, Divergencia ∇·(u,v) usando kernel CUDA fusionado. Reemplaza ~20 operaciones…, Elimina el grado de libertad constante de la presión en la malla restando la…, Proyección incompresible usando Gradiente Conjugado (CG) precondicionado. CG…, Proyección incompresible con defect-correction iterativo + multigrid. Smoother:…, Pre-computa jerarquía multigrid: máscaras, h2, buffers por nivel., Anula componente normal de velocidad en la capa de fluido adyacente al sólido. (+2 more)

### Community 22 - "Canvas de visualización GUI"
Cohesion: 0.18
Nodes (4): FigureCanvas, compute_geometry_delta(), GAVisualizationCanvas, VisualizationCanvas

### Community 23 - "Estudio de convergencia del GA"
Cohesion: 0.19
Nodes (18): cmd_baseline_a5(), cmd_compare(), cmd_custom(), cmd_polygon_a5(), cmd_sa_a0(), cmd_sa_a2(), cmd_sa_a5(), cmd_sa_a8() (+10 more)

### Community 24 - "Parametrización geométrica y dataset"
Cohesion: 0.17
Nodes (17): Parametrización camber/espesor del perfil, Restricciones geométricas de mutación, Configuración ideal de mutación (camber/espesor std), cargar_jsonl(), extraer_fila(), imprimir_stats(), main(), build_dataset.py ================= Procesa aprendizaje_ML.jsonl →… (+9 more)

### Community 25 - "Diagnóstico de drag de presión"
Cohesion: 0.31
Nodes (17): audit_ibm(), Case, case_config(), main(), parse_cases(), Any, Namespace, Path (+9 more)

### Community 26 - "Barrido de condiciones de vuelo"
Cohesion: 0.16
Nodes (16): cargar_perfil(), Carga un perfil aerodinámico en formato Selig. Formato esperado: Línea 1:…, cargar_checkpoint(), _checkpoint_key(), guardar_checkpoint(), main(), script_barrido_conditions.py ============================= Barrido sistemático…, Carga checkpoint si existe y los parámetros del grid coinciden. Retorna set de… (+8 more)

### Community 27 - "Solver cartesiano: proyección"
Cohesion: 0.11
Nodes (9): Mesh, Constructor y basicos, Grafica la evolución de la media acumulada de Cd y Cl para mostrar…, Calcula la divergencia media absoluta del campo de velocidad. div(u) = du/dx +…, Grafica la evolución de la divergencia máxima a lo largo del tiempo. Para…, Guarda una imagen del estado actual en out_dir con nombre frame_XXXX.png. kind:…, Guarda el estado completo de la malla para poder reanudar la simulación.…, Carga el estado completo de la malla desde un archivo. Parámetros: filepath:… (+1 more)

### Community 28 - "Predicción y reconstrucción de perfiles"
Cohesion: 0.20
Nodes (15): cargar_modelo(), main(), mostrar_info(), predecir_geom(), predecir_perfil.py =================== CLI principal: dado un punto de…, verificar_surrogate(), _camber_parabola(), _espesor_naca() (+7 more)

### Community 29 - "Paneles de métricas GUI"
Cohesion: 0.18
Nodes (3): QWidget, GAMetricsPanel, MetricsPanel

### Community 30 - "Widgets auxiliares GUI"
Cohesion: 0.20
Nodes (9): CollapsibleSection, float_edit(), int_spin(), load_dat_profile(), main(), make_check_row(), make_row(), parse_gen_summary() (+1 more)

### Community 31 - "Validación de simetría"
Cohesion: 0.21
Nodes (15): _build_run_kwargs(), diff_runs(), _extract_metrics(), main(), validacion_simetria.py Banco de validacion para el simulador 2D LES…, Configura nu para alcanzar el Re objetivo con U=1, c=1., Context manager: silencia stdout (libera spam del simulador en logs)., Corre los 4 puntos de validacion: 2 Re x {WALE off, WALE on}. (+7 more)

### Community 32 - "Barrido de ángulo de ataque"
Cohesion: 0.31
Nodes (14): base_config(), finite_or_nan(), main(), parse_alpha_spec(), parse_args(), plot_curves(), print_table(), Any (+6 more)

### Community 33 - "Capa límite y tratamiento de pared"
Cohesion: 0.15
Nodes (14): Capa límite y condiciones de pared, Corrección de presión de fondo (sesgo espurio de lift), Extrapolación de presión en pared por regresión de capas, Immersed Boundary Method Ghost-Cell, Large Eddy Simulation (LES), Modelo WALE de viscosidad sub-malla, Refuerzo de impermeabilidad por proyección tangencial, Discrepancia Cp / integral de fuerza (+6 more)

### Community 34 - "Paneles de parámetros GUI"
Cohesion: 0.19
Nodes (3): QScrollArea, GAParamPanel, ParameterPanel

### Community 35 - "Mutación y proyección de perfiles"
Cohesion: 0.15
Nodes (14): _asegurar_x_estrictamente_creciente(), mutar_perfil_parametrico(), _perturbacion_suave(), proyectar_perfil_parametrico(), Interpolación suave en [0, 1] para crear envolventes sin quiebres., Evita problemas numéricos en interpolación si hay empates de X., Suavizado laplaciano 1D liviano para funciones c(x) y t(x)., Reproyecta un perfil `seed` sobre la rejilla X del perfil `ref` (mismo nº de… (+6 more)

### Community 36 - "Vídeo de simulación larga"
Cohesion: 0.33
Nodes (13): build_config(), main(), make_video(), output_dir(), parse_args(), Any, Namespace, Path (+5 more)

### Community 37 - "Entrenamiento de modelos IA"
Cohesion: 0.24
Nodes (13): cargar_dataset(), cargar_modelo(), entrenar_mlp_sklearn(), evaluar_modelos(), guardar_modelo(), main(), predecir_inverso(), predecir_surrogate() (+5 more)

### Community 38 - "Sólidos e IBM ghost-cell"
Cohesion: 0.16
Nodes (7): Devuelve máscara booleana (fluido) de celdas adyacentes 4-vecinas a sólido., Precomputar normales unitarias en interfaz sólido-fluido usando distancia (GPU)., Precomputa los datos necesarios para el método Ghost-Cell IBM. Identifica…, Retorna (media, máximo) de |u·n|/U_ref en la interfaz fluido-sólido., Mallas 2D de posiciones físicas (usa las almacenadas)., Añade un sólido circular centrado en (cx,cy) con radio 'radius' (unidades…, Añade un sólido rectangular delimitado por [x_min,x_max]×[y_min,y_max]…

### Community 39 - "Tests de agente auxiliares"
Cohesion: 0.29
Nodes (12): build_queue(), detect_convergence(), iters_for(), _loop_flux(), main_cli(), make_summary(), Sweep de caracterización del simulador para el optimizador genético: alpha ∈…, Convergencia = último instante en que la media móvil (ventana ~0.5 conv) se… (+4 more)

### Community 40 - "Barrido de resolución"
Cohesion: 0.32
Nodes (12): generate_figures(), get_curve(), load_results(), main(), plan_key(), ndarray, Barrido alpha × resolucion: perfil y modo configurables. Genera una linea por…, run_all() (+4 more)

### Community 41 - "Solver cartesiano: fuerzas"
Cohesion: 0.19
Nodes (6): Añade un sólido circular centrado en (cx,cy) con radio 'radius' (unidades…, Añade un sólido rectangular delimitado por [x_min,x_max]×[y_min,y_max]…, Carga un airfoil desde archivo de texto y lo rasteriza como sólido. Formato…, Devuelve máscara booleana (fluido) de celdas adyacentes 4-vecinas a sólido., Precomputar normales unitarias en interfaz sólido-fluido usando distancia (GPU)., Precomputa los datos necesarios para el método Ghost-Cell IBM. Identifica…

### Community 42 - "Teoría numérica del solver"
Cohesion: 0.18
Nodes (12): Advección semi-Lagrangiana (backtracing), Cuello de botella: ancho de banda de memoria GPU, Defect-correction iterativo, Difusión numérica del semi-Lagrangiano, Gradiente adjunto para simetría D·Dᵀ, Multigrid geométrico V-cycle, Ecuaciones de Navier-Stokes incompresibles 2D, Smoother Red-Black Gauss-Seidel SOR (+4 more)

### Community 43 - "Pipeline de IA predictiva"
Cohesion: 0.17
Nodes (9): Checkpoint atómico del barrido de condiciones, Pipeline de IA predictiva de perfiles, Predictor inverso (condiciones + targets → geometría), Problema one-to-many del predictor inverso, Modelo surrogate forward (geometría → Cl, Cd), DataLoggerML, Registra cada evaluación CFD exitosa en formato JSONL (JSON Lines). Cada línea…, Añade un registro de evaluación al archivo JSONL. Args: perfil_puntos: np.array… (+1 more)

### Community 44 - "Memoria compartida GUI"
Cohesion: 0.23
Nodes (4): QObject, MetricBox, sep(), SharedMemReader

### Community 45 - "Solver cartesiano: kernels"
Cohesion: 0.17
Nodes (6): Difusión viscosa de u,v con modelo WALE de viscosidad turbulenta y viscosidad…, Calcula la viscosidad sub-malla ν_t según el modelo WALE (Nicoud & Ducros).…, Calcula viscosidad turbulenta extra en la estela del perfil. Simula la…, Precomputa la máscara de estela basada en la posición del sólido y la dirección…, Calcula Drag y Lift usando múltiples capas de integración. Transforma de…, Obtiene el ángulo del flujo libre desde las condiciones inflow.

### Community 46 - "Advección e interpolación"
Cohesion: 0.18
Nodes (6): field: cp.array (ny, nx) x_idx, y_idx: coordenadas en *índices* (float) — j (x)…, Posición de partida x - u·dt en *índices* fraccionarios (j,i). dt>0 = backtrace…, Backtrace persistido en x_prev_idx/y_prev_idx (lo reusa advect_sa)., (min, max) de los 4 nodos del stencil bilineal en cada punto., Advección semi-Lagrangiana; steady=True activa under-relaxation para régimen…, Advección semi-Lagrangiana de nu_tilde reutilizando el backtrace persistido por…

### Community 47 - "Benchmark multigrid"
Cohesion: 0.24
Nodes (10): cd_visc_squire_young(), _cd_visc_surface(), compute_corrected_forces(), Cd_visc = 2*(θ_TE/c)*(Ue_TE/V∞)^((H_TE+5)/2), Squire-Young adaptado al régimen de separación. Adjunto (sep_idx=None): Squire-…, Calcula Cl y Cd corregidos post-simulación. mesh : objeto Mesh2D retornado por…, main(), print_table() (+2 more)

### Community 48 - "Solver cartesiano: fronteras"
Cohesion: 0.24
Nodes (5): Devuelve máscara de celdas de fluido adyacentes al sólido (frontera del sólido)., Calcula distancia signed del sólido y normales unitarias hacia el fluido., Integral de esfuerzos en la superficie del sólido (2D, steady): - T = -p n + mu…, Integral de esfuerzos en dos capas alrededor del sólido: - Capa 1: celdas de…, Integra fuerzas en N capas alrededor del sólido. - n_layers: número de capas…

### Community 49 - "Solver cartesiano: utilidades"
Cohesion: 0.20
Nodes (5): Visualiza campos de tracción en la superficie usando…, Versión definitiva de la integral de esfuerzos sobre el perfil. - Usa…, Muestrea Cp instantáneo en las caras frontera y actualiza la suma y contador…, Retorna (x_array, cp_mean_extrados, cp_mean_intrados) como numpy arrays. Si no…, Grafica Cp a lo largo de la cuerda del perfil usando datos de la máscara del…

### Community 50 - "Oráculo aerodinámico Random Forest"
Cohesion: 0.28
Nodes (5): OraculoAerodinamico, Filtro inteligente que predice el fitness de un perfil antes de simularlo. Usa…, Carga experiencia previa desde disco, Persiste experiencia acumulada a disco, Incorpora resultados de una generación para mejorar predicciones

### Community 51 - "Tests de generación geométrica"
Cohesion: 0.33
Nodes (7): construir_pool_inicial(), ejecutar_prueba(), main(), PadreDummy, Prueba rápida de robustez geométrica del generador de hijos del GA. No ejecuta…, Objeto mínimo para reutilizar cruce() sin tocar el GA principal., Genera un conjunto inicial de padres geométricamente válidos.

### Community 52 - "Teoría del optimizador genético"
Cohesion: 0.25
Nodes (8): Algoritmo genético de perfiles, Coeficientes aerodinámicos Cl, Cd, Cp y L/D, Filtro predictivo Random Forest del GA, Fitness L/D por evaluación CFD, Control interactivo por ficheros trigger, Sin migración el GA no converge a un óptimo único, GA de islas con migración, Optimización multipunto α∈{2,4,6}

### Community 53 - "Paneles de corrección BL"
Cohesion: 0.29
Nodes (8): panel_cp_inviscid(), _panel_geom(), Geometría auxiliar: log_r, dth (en marco local), cos_p, sin_p, ds., Velocidad (u,v) en (xi,yi) por panel VÓRTICE de fuerza γ=1/m. Local: u_loc =…, Velocidad (u,v) en (xi,yi) por panel FUENTE de fuerza σ=1/m. Local: u_loc =…, Cp invíscido por método de Hess-Smith (paneles fuente de fuerza variable + un…, _source_uv(), _vortex_uv()

### Community 54 - "Mutación paramétrica"
Cohesion: 0.25
Nodes (8): mutar_perfil(), mutar_y_validar_hijo(), Mutación suave por superposición de bumps gaussianos. Cada bump es una campana…, Aplica mutación según el modo configurado y valida restricciones geométricas.…, Suavizado laplaciano de coordenadas Y. Preserva leading edge, primer punto y…, Intenta reparar un LE agudo mezclando localmente con una referencia suave., reparar_leading_edge_agudo(), suavizar_perfil()

### Community 55 - "Solver cartesiano: métricas"
Cohesion: 0.29
Nodes (4): field: cp.array (ny, nx) x_idx, y_idx: coordenadas en *índices* (float) — j (x)…, Calcula la posición anterior de cada partícula en *índices* (j,i), listos para…, Advección semi-Lagrangiana; steady=True activa under-relaxation para régimen…, Aplica condición IBM Ghost-Cell: impone u=0 en la posición exacta de la pared.…

### Community 56 - "Fuerzas aerodinámicas y polar"
Cohesion: 0.25
Nodes (5): _guardar_punto_polar(), Obtiene el ángulo del flujo libre desde las condiciones inflow., Calcula Drag y Lift en el sistema aerodinámico (relativo al flujo libre). Hace…, Devuelve filas por cara usando exactamente la misma tracción de presión que…, Calcula Cd/Cl medio sobre el rango [iter_inicio, iter_fin), descartando la…

### Community 57 - "Divergencia enmascarada"
Cohesion: 0.25
Nodes (4): Calcula el campo completo de divergencia del^2 u = du/dx + dv/dy. Usa stencil…, Divergencia ∇·(u,v) usando kernel CUDA fusionado. Reemplaza ~20 operaciones…, Calcula la divergencia media absoluta del campo de velocidad. div(u) = du/dx +…, Divergencia máxima absoluta en celdas de fluido (detecta problemas locales…

### Community 58 - "Cola de etapas del TFG"
Cohesion: 0.53
Nodes (4): estado_de(), lanzar_etapa(), cola.sh script, status()

### Community 59 - "Evaluación de población"
Cohesion: 0.33
Nodes (5): calcular_fitness(), evaluar_poblacion(), Calcula fitness combinado a partir de resultados multi-ángulo. Modos: 'mean':…, Evalúa todos los individuos no evaluados de la población. Para cada individuo:…, Retorna True si el diseño merece ser simulado. Compara la predicción con un…

### Community 60 - "Visualización PCA del espacio de diseño"
Cohesion: 0.53
Nodes (5): load_history(), main(), pca_cupy(), plot_landscape(), Visualización PCA + Landscape de fitness (Cl/Cd). - Usa Cupy para calcular PCA…

### Community 61 - "Estudios de población mixta"
Cohesion: 0.80
Nodes (4): estado_de(), lanzar(), mixto.sh script, status()

### Community 62 - "Validaciones de agente"
Cohesion: 0.70
Nodes (4): ganador_dat(), iters_for(), log(), main()

### Community 63 - "Visor en vivo"
Cohesion: 0.50
Nodes (4): conectar_shm(), main(), Viewer en tiempo real para Simulador2D. Se ejecuta como proceso INDEPENDIENTE…, Intenta conectar a la memoria compartida del simulador.

### Community 65 - "Test T1 de ranking por dx"
Cohesion: 0.67
Nodes (3): main(), rank(), T1 — Consistencia de ranking dx=0.004 (estudio) vs dx=0.002 (refinado). Re-…

## Knowledge Gaps
- **14 isolated node(s):** `Ecuaciones de Navier-Stokes incompresibles 2D`, `Restricción de incompresibilidad`, `Large Eddy Simulation (LES)`, `Smoother Red-Black Gauss-Seidel SOR`, `Capa límite y condiciones de pared` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Bucle principal y fronteras` to `Verificación numérica y GCI`, `Tests de malla y proyección`, `Diagnóstico de capa límite`, `Diagnóstico presión-lift`, `Validación de lift NACA0012`, `Diagnóstico lift NACA0012`, `Presión y modelo Spalart-Allmaras`, `Divergencia y solver CG`, `Visualización de campos`, `Barrido de modos multigrid`, `Estudio de convergencia del GA`, `Diagnóstico de drag de presión`, `Validación de simetría`, `Barrido de ángulo de ataque`, `Vídeo de simulación larga`, `Sólidos e IBM ghost-cell`, `Tests de agente auxiliares`, `Barrido de resolución`, `Advección e interpolación`, `Benchmark multigrid`, `Fuerzas aerodinámicas y polar`, `Divergencia enmascarada`?**
  _High betweenness centrality (0.235) - this node is a cross-community bridge._
- **Why does `Mesh` connect `Presión y modelo Spalart-Allmaras` to `Tests de malla y proyección`, `Sólidos e IBM ghost-cell`, `Divergencia y solver CG`, `Visualización de campos`, `Bucle principal y fronteras`, `Advección e interpolación`, `Fuerzas aerodinámicas y polar`, `Divergencia enmascarada`?**
  _High betweenness centrality (0.111) - this node is a cross-community bridge._
- **Why does `load_cp_csv()` connect `Diagnóstico presión-lift` to `Métricas del GA`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **What connects `Ecuaciones de Navier-Stokes incompresibles 2D`, `Restricción de incompresibilidad`, `Large Eddy Simulation (LES)` to the rest of the system?**
  _14 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Verificación numérica y GCI` be split into smaller, more focused modules?**
  _Cohesion score 0.060496067755595885 - nodes in this community are weakly interconnected._
- **Should `Comparativa con XFOIL` be split into smaller, more focused modules?**
  _Cohesion score 0.07407407407407407 - nodes in this community are weakly interconnected._
- **Should `Tests de malla y proyección` be split into smaller, more focused modules?**
  _Cohesion score 0.052244897959183675 - nodes in this community are weakly interconnected._
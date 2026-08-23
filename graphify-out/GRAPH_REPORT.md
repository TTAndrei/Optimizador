# Graph Report - /home/ttandrei/Proyectos/Optimizador/Optimizador  (2026-08-21)

## Corpus Check
- 50 files · ~7,945,510 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1681 nodes · 3276 edges · 110 communities (97 shown, 13 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 57 edges (avg confidence: 0.77)
- Token cost: 447,666 input · 0 output

## Community Hubs (Navigation)
- Estudios asintóticos y polares
- Calibración frente a XFOIL
- Diagnóstico de presión y lift
- Auditoría de capa límite
- Diagnóstico de código del lift
- Diagnóstico de lift NACA 0012
- Constructor del documento TFG
- Núcleo Mesh del solver
- Métricas de convergencia del GA
- Motor del algoritmo genético
- Frontera inmersa y pared
- Editor interactivo de perfiles
- Turbulencia LES y Kutta
- Simulador cartesiano y polares
- Malla estirada y CFL
- Estudio de convergencia de malla
- Corrección de capa límite
- Barrido de modos numéricos
- Geometría de sólidos en malla
- Contorno y difusión de velocidad
- Fitness CFD y sesgo del cociente
- Ventana principal de la GUI
- Exploración de mutaciones
- Malla multirresolución
- Lienzo de visualización del GA
- Tests de condición de Kutta
- Registro de datos para ML
- Gráficas de Cp y convergencia
- Advección y proyección de presión
- Parametrización camber y espesor
- Viscosidad numérica y Re efectivo
- Diagnóstico de drag de presión
- Estimadores de Cl y veracidad
- Predictor inverso de perfiles
- Panel de métricas de la GUI
- Polar 2 grados dominio C
- Polar fina dominio C
- GUI del simulador
- Criterio de parada y coste
- Validación de simetría
- Estructura y moratoria del TFG
- Independencia de dominio
- Polar fina dominio del GA
- Barrido en ángulo de ataque
- Surrogate y oráculo aerodinámico
- Panel de parámetros del GA
- Mutación paramétrica y suavizado
- Vídeo de simulación larga
- Entrenamiento de la IA de perfiles
- Ghost cells e impermeabilidad
- Modelo de islas y restricciones
- GCI del perfil ganador
- Barrido de CFL
- Barrido de resolución
- Fuerzas corregidas y multigrid
- Modelos RANS y submalla
- Barra de estado de la GUI
- Calibración del criterio de parada
- Integración de fuerzas en superficie
- Perfil de Cp y tracción
- Logs de validación de runs
- Corrección de presión de fondo
- Poisson de proyección y cierre
- Tests de compatibilidad de flujo
- Series a dx=0.004
- Difusión numérica semi-Lagrangiana
- Mutación y reparación de perfiles
- Defecto del criterio de terminación
- Bibliografía y base del solver
- Cola de etapas del TFG
- Test de generación geométrica
- Cp invíscido por paneles
- Defecto de convergencia de series
- Viscosidad de estela
- Cálculo de drag y lift
- Campo de divergencia
- GCI y ventana común del fitness
- Estudio paramétrico del simulador
- Resumen del proyecto
- Acentuado de texto
- GCI de Roache y rankeo por malla
- Rendimiento GPU y presión
- Cola de trabajos
- Vídeo del perfil ganador
- Paisaje PCA del GA
- Jerarquía multigrid y métricas 1D
- Test de simetría del multigrid
- Variantes de acentuación
- Cola mixta
- Validación aislada dx=0.002
- Visor en tiempo real
- Sostenibilidad y huella energética
- Lector de log en hilo
- Script de gran optimización
- Consistencia de ranking por dx
- Ventana larga a dx=0.001
- Individuo del GA
- Advección semi-Lagrangiana
- Script aislado
- Comparación de transición
- Conversión polar a cartesiano
- Cola final
- Divergencia media
- Configuración de tests
- Sensibilidad a la resolución
- Control interactivo por ficheros
- Advección incondicionalmente estable
- Dependencias del proyecto

## God Nodes (most connected - your core abstractions)
1. `Mesh` - 96 edges
2. `main()` - 73 edges
3. `Mesh` - 55 edges
4. `MainWindow` - 28 edges
5. `main()` - 26 edges
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
- `Criterio nuevo: drift + noise sobre L/D` --semantically_similar_to--> `Criterio de parada por deriva y ruido`  [INFERRED] [semantically similar]
  docs/verificacion_numerica.md → RESUMEN.md
- `El defecto: parada anticipada en régimen transitorio` --semantically_similar_to--> `Defecto de _detect_series_convergence`  [INFERRED] [semantically similar]
  docs/verificacion_numerica.md → RESUMEN.md
- `El refinado de malla cambia al ganador (refuta el desplazamiento común)` --semantically_similar_to--> `El refinado a dx=0.002 cambia al ganador`  [INFERRED] [semantically similar]
  docs/tfg/resultados_finales_v2.txt → RESUMEN.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Cadena de cuatro defectos acoplados del solver** — resumen_diagnostico_cuatro_capas, docs_tfg_memoria_spalart_allmaras, resumen_wall_treatment_consistent, docs_tfg_difusion_numerica_viscosidad_numerica, resumen_maccormack, resumen_q_lazo [EXTRACTED 1.00]
- **Ambigüedad de la fuerza sobre cuerpos ultrafinos (quinta capa)** — docs_tfg_resultados_finales_v2_solapamiento_ghost_cells_te, docs_tfg_resultados_finales_v2_dispersion_estimadores_cl, docs_tfg_memoria_tres_estimadores_cl, docs_tfg_resultados_finales_v2_banda_ld_ganador, resumen_quinta_capa_te_ultrafino, docs_tfg_resultados_finales_v2_kutta_explicita [EXTRACTED 1.00]
- **Ciclo V&V del criterio de parada: defecto, recalibrado fallido, rediseño y límite de dominio** — docs_verificacion_numerica_defecto_parada, resumen_relleno_cola_sigma_cero, docs_verificacion_numerica_recalibrado_fallido, docs_verificacion_numerica_criterio_nuevo, docs_tfg_fitness_cfd_deriva_vs_dispersion, docs_tfg_resultados_finales_v2_criterio_fuera_de_dominio [EXTRACTED 1.00]
- **Campaña de validación a Re=1e5 con 500 iteraciones** — validacion_resultados_paso4_re100k_run_paso4_sdf, validacion_resultados_poly_f64_run_poly_f64, validacion_resultados_poly_radius_run_poly_radius, validacion_resultados_smooth0_run_smooth0 [INFERRED 0.85]
- **Diagnóstico de simetría de la máscara sólida** — validacion_resultados_poly_f64_run_poly_f64, validacion_resultados_poly_f64_asimetria_mascara, validacion_resultados_poly_radius_run_poly_radius, validacion_resultados_poly_radius_simetria_perfecta [INFERRED 0.85]
- **Evidencia de no convergencia a 500 iteraciones** — validacion_resultados_long4000_traza_cl_cd_4000, validacion_resultados_long4000_deriva_monotona_cd, validacion_resultados_long4000_salto_regimen_it2900, validacion_resultados_poly_f64_media_ultimo_70 [INFERRED 0.75]
- **Cadena de cuatro defectos acoplados de la sobre-circulación** — docs_tfg_tfg_spalart_allmaras, docs_tfg_tfg_inconsistencia_ibm_proyeccion, docs_tfg_tfg_q_lazo, docs_tfg_3_4_2_difusion_numerica_difusion_numerica [EXTRACTED 1.00]
- **Pipeline de corrección de presión de fondo** — docs_tfg_3_8_2_presion_de_fondo_indeterminacion_de_la_constante, docs_tfg_3_8_2_presion_de_fondo_residuo_de_cierre, docs_tfg_3_8_2_presion_de_fondo_fuerza_de_flotabilidad, docs_tfg_3_8_2_presion_de_fondo_ajuste_afin_de_fondo [EXTRACTED 1.00]
- **Verificación de la señal de fitness y rehecho de la campaña** — docs_tfg_3_10_3_evaluacion_fitness_criterio_de_parada_original, docs_tfg_3_10_3_evaluacion_fitness_criterio_deriva_dispersion, docs_tfg_3_10_3_evaluacion_fitness_calibracion_tolerancias, docs_tfg_tfg_verificacion_por_rangos_del_fitness, docs_tfg_tfg_campana_definitiva [EXTRACTED 1.00]
- **Paso temporal completo del solver (advección → difusión → proyección)** — base_teorica_adveccion_semilagrangiana, base_teorica_modelo_wale, base_teorica_multigrid_vcycle, base_teorica_ibm_ghost_cell, base_teorica_condicion_cfl [EXTRACTED 1.00]

## Communities (110 total, 13 thin omitted)

### Community 0 - "Estudios asintóticos y polares"
Cohesion: 0.05
Nodes (65): analisis(), cargar_serie(), figura(), main(), Asintótico temporal del perfil ganador del AG a alpha=4, Re=1e5, dx=0.002, en…, Qué habría reportado el run si hubiera acabado en t., tabla(), Coste por iteracion en funcion del tamano de dominio, a dx=0.002 fijo. El… (+57 more)

### Community 1 - "Calibración frente a XFOIL"
Cohesion: 0.07
Nodes (50): fig_combinado(), fig_factor(), interp_xfoil(), load_sim(), load_xfoil(), main(), poly_fit(), print_resumen() (+42 more)

### Community 2 - "Diagnóstico de presión y lift"
Cohesion: 0.13
Nodes (47): cl_from_delta_cp(), Cl = ∫(Cp_lower - Cp_upper) d(x/c) · cos(α) Rellena bins vacíos (valor 0.0) por…, _baseline_case_id(), build_cases(), build_le_comparison(), build_sensitivity_rows(), Case, _case_row() (+39 more)

### Community 3 - "Auditoría de capa límite"
Cohesion: 0.14
Nodes (45): Case, boundary_layer_audit(), build_base_config(), build_cases(), Case, clean(), cp_profile_audit(), create_videos() (+37 more)

### Community 4 - "Diagnóstico de código del lift"
Cohesion: 0.13
Nodes (42): build_cases(), build_config(), Case, diagnose(), find_row(), finite_or_nan(), history_rows(), main() (+34 more)

### Community 5 - "Diagnóstico de lift NACA 0012"
Cohesion: 0.14
Nodes (42): attach_hints(), attach_lift_diagnostics(), build_cases(), build_legacy_cases(), build_reduced_compare_templates(), Case, case_hint(), cuda_available() (+34 more)

### Community 6 - "Constructor del documento TFG"
Cohesion: 0.13
Nodes (22): Builder, construir(), empaquetar(), esc(), field(), main(), numero_auto(), para() (+14 more)

### Community 7 - "Núcleo Mesh del solver"
Cohesion: 0.07
Nodes (18): Mesh, Estado de los ghost cells cerca del TE: nº, irreparables y rango de d_ghost., points: lista de tuplas (i,j) en índices de celda (fila, columna). p_value:…, field: cp.array (ny, nx) x_idx, y_idx: coordenadas en *índices* (float) — j (x)…, Aplica -(dt/rho)*grad(p^n) al campo de velocidad (paso predictor del método…, Constructor y basicos, IC/BC fully-turbulent: nu_tilde=3*nu en fluido, 0 en sólido.…, Distancia física a pared cacheada (geometría fija). sd en celdas (banda… (+10 more)

### Community 8 - "Métricas de convergencia del GA"
Cohesion: 0.11
Nodes (37): cargar_estudios(), cargar_evaluaciones(), col(), fig_convergencia(), fig_convergencia_cfd(), fig_convergencia_normalizada(), fig_correlaciones(), fig_coste() (+29 more)

### Community 9 - "Motor del algoritmo genético"
Cohesion: 0.09
Nodes (35): calcular_fitness(), cruce(), evaluar_poblacion(), generar_directorio_resultados(), generar_stem_optimizado(), guardar_estado_ga(), guardar_perfil(), guardar_perfil_con_metadata() (+27 more)

### Community 10 - "Frontera inmersa y pared"
Cohesion: 0.08
Nodes (19): Capa límite y condiciones de pared, Immersed Boundary Method Ghost-Cell, Refuerzo de impermeabilidad por proyección tangencial, Anula componente normal de velocidad en la capa de fluido adyacente al sólido., Aplica condición IBM Ghost-Cell: impone u=0 en la posición exacta de la pared.…, Aplica dp/dn=0 en los bordes donde NO haya Dirichlet. Importante: la proyección…, Reserva y reutiliza buffers de caras para la proyección por flujo., Construye velocidades normales en caras para la ruta compatible por flujo. -… (+11 more)

### Community 11 - "Editor interactivo de perfiles"
Cohesion: 0.09
Nodes (17): EditorPerfiles, main(), Editor Interactivo de Perfiles Aerodinámicos Permite cargar, editar y guardar…, Configurar la visualización interactiva con todos los puntos editables, Editor interactivo de perfiles aerodinámicos. Controles: - Click izquierdo +…, Actualizar título con información completa, Evento cuando se selecciona un punto, Evento de click del ratón (+9 more)

### Community 12 - "Turbulencia LES y Kutta"
Cohesion: 0.08
Nodes (16): Large Eddy Simulation (LES), Modelo WALE de viscosidad sub-malla, main(), SDF exacto punto-a-segmento al polígono _airfoil_polygon_x/y en una banda…, Cambia el ángulo de ataque modificando la dirección del flujo. Asume que la…, side: left/right/top/bottom ; bc_type: noslip/inflow/outflow/slip, Aplica condiciones de frontera. Parámetros: after_projection: Si True, no…, Difusión viscosa de u,v con modelo WALE de viscosidad turbulenta y viscosidad… (+8 more)

### Community 13 - "Simulador cartesiano y polares"
Cohesion: 0.08
Nodes (20): _generar_graficos_polar(), generar_graficos_y_outputs(), _guardar_punto_polar(), load_complete_checkpoint(), main(), Visualiza la magnitud de la velocidad en colores (sin flechas). - cmap:…, Grafica Cd y Cl con selector interactivo de rango para calcular la media. Usa…, Grafica la evolución de la media acumulada de Cd y Cl para mostrar… (+12 more)

### Community 14 - "Malla estirada y CFL"
Cohesion: 0.08
Nodes (19): Condición CFL y paso temporal adaptativo, Malla cartesiana de densidad variable, analizar_checkpoint(), _generar_graficos_polar(), generar_malla_estirada(), generar_malla_estirada_intervalo(), Version con banda fina explicita, util para extender wake sin mover el LE., Genera un vector 1D de posiciones de nodo de 0 a L con espaciado variable: fino… (+11 more)

### Community 15 - "Estudio de convergencia de malla"
Cohesion: 0.19
Nodes (26): analizar_convergencia(), calib_path(), calibrar(), _camber_espesor(), _cargar_ganador(), _check_stop(), ejecutar_aislado(), ejecutar_estudio() (+18 more)

### Community 16 - "Corrección de capa límite"
Cohesion: 0.15
Nodes (23): _apply_pressure_debias(), cd_visc_squire_young(), _cd_visc_surface(), _cf_head(), compute_geometric_pressure_forces(), _h1_from_h(), _h_from_h1(), _h_from_lam() (+15 more)

### Community 17 - "Barrido de modos numéricos"
Cohesion: 0.15
Nodes (24): fig_comparacion_modos(), fig_convergencia_malla(), fig_polar(), fig_resumen_grid(), fig_velocidad(), generate_figures(), get_curve(), load_results() (+16 more)

### Community 18 - "Geometría de sólidos en malla"
Cohesion: 0.12
Nodes (11): Mesh, Constructor y basicos, Añade un sólido circular centrado en (cx,cy) con radio 'radius' (unidades…, Añade un sólido rectangular delimitado por [x_min,x_max]×[y_min,y_max]…, Carga un airfoil desde archivo de texto y lo rasteriza como sólido. Formato…, Guarda el estado completo de la malla para poder reanudar la simulación.…, Carga el estado completo de la malla desde un archivo. Parámetros: filepath:…, Devuelve máscara booleana (fluido) de celdas adyacentes 4-vecinas a sólido. (+3 more)

### Community 19 - "Contorno y difusión de velocidad"
Cohesion: 0.14
Nodes (11): Aplica dp/dn=0 en los bordes donde NO haya Dirichlet. Importante: la proyección…, Difusión viscosa de u,v con modelo WALE de viscosidad turbulenta y viscosidad…, Calcula el campo completo de divergencia del^2 u = du/dx + dv/dy. Usa stencil…, Divergencia ∇·(u,v) usando kernel CUDA fusionado. Reemplaza ~20 operaciones…, Elimina el grado de libertad constante de la presión en la malla restando la…, Proyección incompresible usando Gradiente Conjugado (CG) precondicionado. CG…, Proyección incompresible con defect-correction iterativo + multigrid. Smoother:…, Pre-computa jerarquía multigrid: máscaras, h2, buffers por nivel. (+3 more)

### Community 20 - "Fitness CFD y sesgo del cociente"
Cohesion: 0.10
Nodes (22): Auditoría de superficie (diagnóstico), Cociente de medias frente a media de cocientes, La evaluación CFD como función de fitness, Numerador y denominador de la misma ventana temporal, Análisis de ecuación modificada, Algoritmo genético (scripts/RunGA.py), Breiman, Random forests [28], DataLoggerML (+14 more)

### Community 21 - "Ventana principal de la GUI"
Cohesion: 0.19
Nodes (3): QMainWindow, MainWindow, Guarda los parámetros actuales como defaults en gui_settings.json

### Community 22 - "Exploración de mutaciones"
Cohesion: 0.15
Nodes (19): area_poligono(), construir_pool_inicial(), extraer_metricas(), flatten_resumen_escenario(), generar_hijos_escenario(), guardar_csv_dicts(), guardar_plot_muestras(), main() (+11 more)

### Community 23 - "Malla multirresolución"
Cohesion: 0.10
Nodes (11): MallaMultiRes, Parámetros: Lx, Ly: dimensiones del dominio (m) dx_grueso: resolución de malla…, Convierte índices de malla fina a índices de malla gruesa. Retorna: (j_grueso,…, Retorna arrays 2D con índices gruesos para cada punto fino. Útil para…, Retorna límites físicos de malla fina., Ajusta coordenadas de sólido de sistema grueso a sistema fino. Retorna:…, Visualiza la configuración de mallas y máscaras. Parámetros: mostrar_pesos: si…, Kernel GPU para promediar malla fina → gruesa con filtrado de sólido y clamping. (+3 more)

### Community 24 - "Lienzo de visualización del GA"
Cohesion: 0.18
Nodes (4): FigureCanvas, compute_geometry_delta(), GAVisualizationCanvas, VisualizationCanvas

### Community 25 - "Tests de condición de Kutta"
Cohesion: 0.19
Nodes (18): cmd_baseline_a5(), cmd_compare(), cmd_custom(), cmd_polygon_a5(), cmd_sa_a0(), cmd_sa_a2(), cmd_sa_a5(), cmd_sa_a8() (+10 more)

### Community 26 - "Registro de datos para ML"
Cohesion: 0.13
Nodes (16): cargar_perfil(), DataLoggerML, Registra cada evaluación CFD exitosa en formato JSONL (JSON Lines). Cada línea…, Añade un registro de evaluación al archivo JSONL. Args: perfil_puntos: np.array…, Carga todos los registros como lista de dicts (para entrenamiento ML), Carga un perfil aerodinámico en formato Selig. Formato esperado: Línea 1:…, cargar_checkpoint(), _checkpoint_key() (+8 more)

### Community 27 - "Gráficas de Cp y convergencia"
Cohesion: 0.11
Nodes (11): generar_graficos_y_outputs(), Visualiza la magnitud de la velocidad en colores (sin flechas). - cmap:…, Visualiza la magnitud de velocidad y superpone flechas (quiver) con dirección y…, Grafica Cd y Cl con selector interactivo de rango para calcular la media. Usa…, Grafica la evolución de la media acumulada de Cd y Cl para mostrar…, Grafica la evolución de la divergencia máxima a lo largo del tiempo. Para…, Retorna (x_array, cp_mean_extrados, cp_mean_intrados) como numpy arrays. Si no…, Grafica Cp a lo largo de la cuerda del perfil usando datos de la máscara del… (+3 more)

### Community 28 - "Advección y proyección de presión"
Cohesion: 0.11
Nodes (15): Advección semi-Lagrangiana (backtracing), Cuello de botella: ancho de banda de memoria GPU, Defect-correction iterativo, Difusión numérica del semi-Lagrangiano, Gradiente adjunto para simetría D·Dᵀ, Multigrid geométrico V-cycle, Ecuaciones de Navier-Stokes incompresibles 2D, Smoother Red-Black Gauss-Seidel SOR (+7 more)

### Community 29 - "Parametrización camber y espesor"
Cohesion: 0.16
Nodes (17): Parametrización camber/espesor del perfil, Restricciones geométricas de mutación, Configuración ideal de mutación (camber/espesor std), cargar_jsonl(), extraer_fila(), imprimir_stats(), main(), build_dataset.py ================= Procesa aprendizaje_ML.jsonl →… (+9 more)

### Community 30 - "Viscosidad numérica y Re efectivo"
Cohesion: 0.13
Nodes (18): Bibliografía mínima exigida, Número de Reynolds de celda, Reynolds efectivo y engrosamiento de la capa límite, Viscosidad numérica del semi-Lagrangiano, Anexo A: configuración de referencia del solver, Cakmakcioglu, Bas & Kaynak (2018), correlation-based algebraic transition model, Kurtulus (2015), NACA 0012 a Re=1000, Menter et al. (2006), correlation-based transition model (+10 more)

### Community 31 - "Diagnóstico de drag de presión"
Cohesion: 0.31
Nodes (17): audit_ibm(), Case, case_config(), main(), parse_cases(), Any, Namespace, Path (+9 more)

### Community 32 - "Estimadores de Cl y veracidad"
Cohesion: 0.13
Nodes (17): Reglas de veracidad del TFG, Drela (1989), XFOIL, Frontera inmersa con celdas fantasma, Mittal & Iaccarino (2005), Immersed boundary methods, Peskin (2002), The immersed boundary method, Recorte automático del borde de salida, Tres estimadores independientes de la sustentación, Altura mínima del borde de salida (+9 more)

### Community 33 - "Predictor inverso de perfiles"
Cohesion: 0.20
Nodes (15): cargar_modelo(), main(), mostrar_info(), predecir_geom(), predecir_perfil.py =================== CLI principal: dado un punto de…, verificar_surrogate(), _camber_parabola(), _espesor_naca() (+7 more)

### Community 34 - "Panel de métricas de la GUI"
Cohesion: 0.18
Nodes (3): QWidget, GAMetricsPanel, MetricsPanel

### Community 35 - "Polar 2 grados dominio C"
Cohesion: 0.24
Nodes (16): campo_path(), cargar(), csv_path(), _cuerpo(), _ejes(), escribir_csv(), figuras_campo(), figuras_polar() (+8 more)

### Community 36 - "Polar fina dominio C"
Cohesion: 0.24
Nodes (16): campo_path(), cargar(), csv_path(), _cuerpo(), _ejes(), escribir_csv(), figuras_campo(), figuras_polar() (+8 more)

### Community 37 - "GUI del simulador"
Cohesion: 0.20
Nodes (9): CollapsibleSection, float_edit(), int_spin(), load_dat_profile(), main(), make_check_row(), make_row(), parse_gen_summary() (+1 more)

### Community 38 - "Criterio de parada y coste"
Cohesion: 0.16
Nodes (16): Calibración de tolerancias por rejilla con series retenidas, Coste por evaluación CFD, Criterio de parada original (autosemejanza), Criterio de terminación por deriva y dispersión, Coste por evaluación y factor limitante de la campaña, Modo multipunto en ángulo de ataque, ASME V&V 20 [13], Ausencia de estado estacionario en la malla fina (+8 more)

### Community 39 - "Validación de simetría"
Cohesion: 0.21
Nodes (15): _build_run_kwargs(), diff_runs(), _extract_metrics(), main(), validacion_simetria.py Banco de validacion para el simulador 2D LES…, Configura nu para alcanzar el Re objetivo con U=1, c=1., Context manager: silencia stdout (libera spam del simulador en logs)., Corre los 4 puntos de validacion: 2 Re x {WALE off, WALE on}. (+7 more)

### Community 40 - "Estructura y moratoria del TFG"
Cohesion: 0.14
Nodes (15): Estructura obligatoria del TFG (7 capítulos), Excepción: la verificación de la función de fitness sí se escribe, Moratoria §3.0: qué no se escribe todavía, Narrativa del trabajo: V&V por encima del optimizador, Breiman (2001), Random forests, DataLoggerML: registro sin censura, Malla cartesiana estirada con stencils no uniformes, Clase Mesh y función main() (+7 more)

### Community 41 - "Independencia de dominio"
Cohesion: 0.21
Nodes (14): analizar(), diagnostico_contorno(), escribir_csv(), extrapolar(), factorial(), figuras(), geometria(), informe() (+6 more)

### Community 42 - "Polar fina dominio del GA"
Cohesion: 0.28
Nodes (14): campo_path(), cargar(), csv_path(), _cuerpo(), _ejes(), escribir_csv(), figuras_campo(), figuras_polar() (+6 more)

### Community 43 - "Barrido en ángulo de ataque"
Cohesion: 0.31
Nodes (14): base_config(), finite_or_nan(), main(), parse_alpha_spec(), parse_args(), plot_curves(), print_table(), Any (+6 more)

### Community 44 - "Surrogate y oráculo aerodinámico"
Cohesion: 0.16
Nodes (10): Checkpoint atómico del barrido de condiciones, Pipeline de IA predictiva de perfiles, Predictor inverso (condiciones + targets → geometría), Problema one-to-many del predictor inverso, Modelo surrogate forward (geometría → Cl, Cd), OraculoAerodinamico, Filtro inteligente que predice el fitness de un perfil antes de simularlo. Usa…, Carga experiencia previa desde disco (+2 more)

### Community 45 - "Panel de parámetros del GA"
Cohesion: 0.19
Nodes (3): QScrollArea, GAParamPanel, ParameterPanel

### Community 46 - "Mutación paramétrica y suavizado"
Cohesion: 0.15
Nodes (14): _asegurar_x_estrictamente_creciente(), mutar_perfil_parametrico(), _perturbacion_suave(), proyectar_perfil_parametrico(), Interpolación suave en [0, 1] para crear envolventes sin quiebres., Evita problemas numéricos en interpolación si hay empates de X., Suavizado laplaciano 1D liviano para funciones c(x) y t(x)., Reproyecta un perfil `seed` sobre la rejilla X del perfil `ref` (mismo nº de… (+6 more)

### Community 47 - "Vídeo de simulación larga"
Cohesion: 0.33
Nodes (13): build_config(), main(), make_video(), output_dir(), parse_args(), Any, Namespace, Path (+5 more)

### Community 48 - "Entrenamiento de la IA de perfiles"
Cohesion: 0.24
Nodes (13): cargar_dataset(), cargar_modelo(), entrenar_mlp_sklearn(), evaluar_modelos(), guardar_modelo(), main(), predecir_inverso(), predecir_surrogate() (+5 more)

### Community 49 - "Ghost cells e impermeabilidad"
Cohesion: 0.16
Nodes (7): Devuelve máscara booleana (fluido) de celdas adyacentes 4-vecinas a sólido., Precomputar normales unitarias en interfaz sólido-fluido usando distancia (GPU)., Precomputa los datos necesarios para el método Ghost-Cell IBM. Identifica…, Retorna (media, máximo) de |u·n|/U_ref en la interfaz fluido-sólido., Mallas 2D de posiciones físicas (usa las almacenadas)., Añade un sólido circular centrado en (cx,cy) con radio 'radius' (unidades…, Añade un sólido rectangular delimitado por [x_min,x_max]×[y_min,y_max]…

### Community 50 - "Modelo de islas y restricciones"
Cohesion: 0.18
Nodes (13): Algoritmo genético del optimizador, Jameson (1988), Aerodynamic design via control theory, Modelo de islas con migración, Parametrización camber/espesor con nodos de control, Restricciones geométricas del perfil, Skinner & Zare-Behtash (2018), State-of-the-art in aerodynamic shape optimisation, Ausencia de restricción estructural en el planteamiento, La deriva hacia perfiles delgados es física, no artefacto (+5 more)

### Community 51 - "GCI del perfil ganador"
Cohesion: 0.22
Nodes (11): criterio(), gci(), iters_for(), main(), GCI de tres mallas sobre el ganador de la campaña. El refinado del top-3 acabó…, f1 la malla fina. Devuelve orden observado, extrapolación y GCI de cada par., criterio(), geometria() (+3 more)

### Community 52 - "Barrido de CFL"
Cohesion: 0.29
Nodes (12): build_queue(), detect_convergence(), iters_for(), _loop_flux(), main_cli(), make_summary(), Sweep de caracterización del simulador para el optimizador genético: alpha ∈…, Convergencia = último instante en que la media móvil (ventana ~0.5 conv) se… (+4 more)

### Community 53 - "Barrido de resolución"
Cohesion: 0.32
Nodes (12): generate_figures(), get_curve(), load_results(), main(), plan_key(), ndarray, Barrido alpha × resolucion: perfil y modo configurables. Genera una linea por…, run_all() (+4 more)

### Community 54 - "Fuerzas corregidas y multigrid"
Cohesion: 0.24
Nodes (10): compute_corrected_forces(), Calcula Cl y Cd corregidos post-simulación. mesh : objeto Mesh2D retornado por…, main(), print_table(), Estudio de parámetros del solver multigrid. Prueba 6 configuraciones × 500…, run_config(), add_force_consistent_metrics(), Any (+2 more)

### Community 55 - "Modelos RANS y submalla"
Cohesion: 0.17
Nodes (12): Interferencia de ν_núm con el punto de trabajo de SA, Nicoud & Ducros (1999), WALE subgrid-scale model, Modelo RANS de Spalart-Allmaras, Spalart & Allmaras (1992), A one-equation turbulence model, Modelo submalla WALE, Cakmakcioglu, Bas y Kaynak, modelo algebraico de transición [6], Contraste con XFOIL (NACA 0012, Re=1e5), Drela, XFOIL [10] (+4 more)

### Community 56 - "Barra de estado de la GUI"
Cohesion: 0.23
Nodes (4): QObject, MetricBox, sep(), SharedMemReader

### Community 57 - "Calibración del criterio de parada"
Cohesion: 0.27
Nodes (11): asintotico(), barrer(), cargar_series(), evaluar(), figura(), main(), Criterio de parada nuevo, ajustado y validado sobre series separadas. El…, Error y coste de cada combinación. No parar no se penaliza como fallo: con tope… (+3 more)

### Community 58 - "Integración de fuerzas en superficie"
Cohesion: 0.21
Nodes (6): field: cp.array (ny, nx) x_idx, y_idx: coordenadas en *índices* (float) — j (x)…, Devuelve máscara de celdas de fluido adyacentes al sólido (frontera del sólido)., Calcula distancia signed del sólido y normales unitarias hacia el fluido., Integral de esfuerzos en la superficie del sólido (2D, steady): - T = -p n + mu…, Integral de esfuerzos en dos capas alrededor del sólido: - Capa 1: celdas de…, Integra fuerzas en N capas alrededor del sólido. - n_layers: número de capas…

### Community 59 - "Perfil de Cp y tracción"
Cohesion: 0.17
Nodes (6): Calcula la viscosidad sub-malla ν_t según el modelo WALE (Nicoud & Ducros).…, Visualiza campos de tracción en la superficie usando…, Versión definitiva de la integral de esfuerzos sobre el perfil. - Usa…, Muestrea Cp instantáneo en las caras frontera y actualiza la suma y contador…, Retorna (x_array, cp_mean_extrados, cp_mean_intrados) como numpy arrays. Si no…, Grafica Cp a lo largo de la cuerda del perfil usando datos de la máscara del…

### Community 60 - "Logs de validación de runs"
Cohesion: 0.27
Nodes (12): Corrección de offset en Fy, Deriva monótona del Cd, Salto de régimen en it≈2900, Traza Cl/Cd a 4000 iteraciones, Divergencia máxima residual (div_max), Run paso4_sdf a Re=1e5, Asimetría residual de la máscara sólida, Media de Cl/Cd sobre el último 70% (+4 more)

### Community 61 - "Corrección de presión de fondo"
Cohesion: 0.18
Nodes (11): Ajuste afín de fondo por mínimos cuadrados, Decaimiento 1/r del torbellino ligado en dominio finito, Corrección de presión de fondo, Fuerza de flotabilidad por pendiente de fondo, Análisis de ecuación modificada, Extrapolación de la presión a la pared por mínimos cuadrados, Ajuste afín del campo de presión de fondo, Fuerza de flotabilidad del gradiente de fondo (+3 more)

### Community 62 - "Poisson de proyección y cierre"
Cohesion: 0.20
Nodes (11): Poisson de proyección con Neumann en las cuatro fronteras, Indeterminación del nivel absoluto de presión, Residuo de cierre del contorno rasterizado, Brandt, Multi-level adaptive solutions [26], Inconsistencia entre frontera inmersa y proyección, Proyección de presión con multimalla geométrico, Flujo neto sobre lazos cerrados (Q_lazo), Rasterización de la geometría en doble precisión (+3 more)

### Community 63 - "Tests de compatibilidad de flujo"
Cohesion: 0.33
Nodes (8): Mesh, requires_cuda, _make_mesh(), _make_wall_mesh(), test_compatible_flux_reduces_wall_leak_on_simple_wall(), test_flux_faces_preserve_uniform_field_and_zero_divergence(), test_pressure_gradient_correction_is_uniform_on_nonuniform_mesh(), test_surface_force_audit_matches_pressure_force_integral()

### Community 64 - "Series a dx=0.004"
Cohesion: 0.31
Nodes (10): analizar(), correr(), escribir_dat(), geometrias(), iters_for(), _log(), Base de datos de series temporales a dx=0.004 para dos cosas a la vez. 1.…, Los 20 individuos de la revalidación, con su L/D a dx=0.002. (+2 more)

### Community 65 - "Difusión numérica semi-Lagrangiana"
Cohesion: 0.27
Nodes (10): Difusión numérica del esquema de advección, Interpolación bilineal del pie de característica, Reynolds efectivo y error de espesor de capa límite, Desplazamiento del punto de trabajo de Spalart-Allmaras, Superposición nu + nu_t + nu_num, Viscosidad numérica equivalente, Advección semi-Lagrangiana, Corrección de MacCormack (+2 more)

### Community 66 - "Mutación y reparación de perfiles"
Cohesion: 0.22
Nodes (10): mutar_perfil(), mutar_y_validar_hijo(), Intenta reparar un LE agudo mezclando localmente con una referencia suave., Mutación suave por superposición de bumps gaussianos. Cada bump es una campana…, Aplica mutación según el modo configurado y valida restricciones geométricas.…, Suavizado laplaciano de coordenadas Y. Preserva leading edge, primer punto y…, Valida integridad geométrica del perfil antes de IA/CFD., reparar_leading_edge_agudo() (+2 more)

### Community 67 - "Defecto del criterio de terminación"
Cohesion: 0.25
Nodes (9): Separación entre deriva y dispersión, El criterio de terminación fuera de su dominio de calibración, Acoplamiento por fichero del criterio de parada, Criterio nuevo: drift + noise sobre L/D, El defecto: parada anticipada en régimen transitorio, Recalibrado por tolerancias y su fallo de generalización, El residual de campo no se usa (tol_res=inf), T_TARGET_CASO: el horizonte no es universal (+1 more)

### Community 68 - "Bibliografía y base del solver"
Cohesion: 0.22
Nodes (9): Chorin (1968), Numerical solution of the Navier-Stokes equations, TFG: Simulador CFD 2D acelerado por GPU y optimización genética de perfiles, Chorin, Numerical solution of the Navier-Stokes equations [1], Okuta et al., CuPy [27], Implementación en GPU con CuPy y núcleos CUDA, Jameson, Aerodynamic design via control theory [14], Malla cartesiana estirada, Método de separación de Chorin (+1 more)

### Community 69 - "Cola de etapas del TFG"
Cohesion: 0.31
Nodes (3): hecho(), cola_tfg.sh script, status()

### Community 70 - "Test de generación geométrica"
Cohesion: 0.33
Nodes (7): construir_pool_inicial(), ejecutar_prueba(), main(), PadreDummy, Prueba rápida de robustez geométrica del generador de hijos del GA. No ejecuta…, Objeto mínimo para reutilizar cruce() sin tocar el GA principal., Genera un conjunto inicial de padres geométricamente válidos.

### Community 71 - "Cp invíscido por paneles"
Cohesion: 0.29
Nodes (8): panel_cp_inviscid(), _panel_geom(), Geometría auxiliar: log_r, dth (en marco local), cos_p, sin_p, ds., Velocidad (u,v) en (xi,yi) por panel VÓRTICE de fuerza γ=1/m. Local: u_loc =…, Velocidad (u,v) en (xi,yi) por panel FUENTE de fuerza σ=1/m. Local: u_loc =…, Cp invíscido por método de Hess-Smith (paneles fuente de fuerza variable + un…, _source_uv(), _vortex_uv()

### Community 72 - "Defecto de convergencia de series"
Cohesion: 0.25
Nodes (8): Criterio de parada por deriva y ruido, Defecto de _detect_series_convergence, Fitness negativo indistinguible de un fallo, Relleno de cola de clvector/cdvector y σ=0, Revalidación del ranking del GA, Series a dx=0.004 sin parada: la malla sí rankea, Simulador2D (solver CFD 2D en GPU), Validación externa contra XFOIL

### Community 73 - "Viscosidad de estela"
Cohesion: 0.25
Nodes (4): Calcula viscosidad turbulenta extra en la estela del perfil. Simula la…, Precomputa la máscara de estela basada en la posición del sólido y la dirección…, Calcula Drag y Lift usando múltiples capas de integración. Transforma de…, Obtiene el ángulo del flujo libre desde las condiciones inflow.

### Community 74 - "Cálculo de drag y lift"
Cohesion: 0.25
Nodes (5): _guardar_punto_polar(), Obtiene el ángulo del flujo libre desde las condiciones inflow., Calcula Drag y Lift en el sistema aerodinámico (relativo al flujo libre). Hace…, Devuelve filas por cara usando exactamente la misma tracción de presión que…, Calcula Cd/Cl medio sobre el rango [iter_inicio, iter_fin), descartando la…

### Community 75 - "Campo de divergencia"
Cohesion: 0.25
Nodes (4): Calcula el campo completo de divergencia del^2 u = du/dx + dv/dy. Usa stencil…, Divergencia ∇·(u,v) usando kernel CUDA fusionado. Reemplaza ~20 operaciones…, Calcula la divergencia media absoluta del campo de velocidad. div(u) = du/dx +…, Divergencia máxima absoluta en celdas de fluido (detecta problemas locales…

### Community 76 - "GCI y ventana común del fitness"
Cohesion: 0.29
Nodes (7): El fitness es cociente de medias, no media de cocientes, Numerador y denominador de la misma ventana temporal, Advertencia: las tres mallas del GCI no comparten definición de sustentación, GCI del perfil ganador: convergencia oscilatoria, Horizonte extendido a dx=0.001 y ausencia de estado estacionario, GCI de tres mallas sobre el ganador de tfg2, Ventana larga a dx=0.001 (t=71 físico)

### Community 77 - "Estudio paramétrico del simulador"
Cohesion: 0.38
Nodes (6): calcular_coeficientes(), ejecutar_simulacion(), main(), Estudio paramétrico del simulador 2D - Ejecuta Simulador2D.main para cada…, Ejecuta una simulación y devuelve (mesh_fina, mesh_gruesa, geometria)., Calcula CD, CL medios (descartando transitorio) y finales desde…

### Community 78 - "Resumen del proyecto"
Cohesion: 0.33
Nodes (6): Algoritmo genético de perfiles, Coeficientes aerodinámicos Cl, Cd, Cp y L/D, Corrección de presión de fondo (sesgo espurio de lift), Extrapolación de presión en pared por regresión de capas, Filtro predictivo Random Forest del GA, Fitness L/D por evaluación CFD

### Community 79 - "Acentuado de texto"
Cohesion: 0.53
Nodes (4): acentuar(), construir_mapa(), deacc(), main()

### Community 80 - "GCI de Roache y rankeo por malla"
Cohesion: 0.33
Nodes (6): ASME V&V 20-2009, Celik et al. (2008), Procedure for estimation of discretization uncertainty, El refinado de malla cambia al ganador (refuta el desplazamiento común), GCI de Roache sobre cuatro mallas, ¿Sirve dx=0.004 para rankear?, Validación cruzada por tres vías independientes

### Community 81 - "Rendimiento GPU y presión"
Cohesion: 0.33
Nodes (6): El cuello de botella es el ancho de banda de memoria, Núcleos CUDA propios del solver, Okuta et al. (2017), CuPy, Proyección de presión: ciclo en V multimalla, Indeterminación del nivel de presión, Residuo de cierre del contorno discreto

### Community 82 - "Cola de trabajos"
Cohesion: 0.53
Nodes (4): estado_de(), lanzar_etapa(), cola.sh script, status()

### Community 83 - "Vídeo del perfil ganador"
Cohesion: 0.47
Nodes (5): make_video(), Path, Corre un perfil ganador (masgen Re=1e5 o re1e3) con guardado de frames y genera…, run(), summarize()

### Community 84 - "Paisaje PCA del GA"
Cohesion: 0.53
Nodes (5): load_history(), main(), pca_cupy(), plot_landscape(), Visualización PCA + Landscape de fitness (Cl/Cd). - Usa Cupy para calcular PCA…

### Community 85 - "Jerarquía multigrid y métricas 1D"
Cohesion: 0.33
Nodes (4): calcular_metricas_1d(), Pre-computa jerarquía multigrid: máscaras, h2, buffers por nivel., Dada una secuencia monótona de posiciones (CuPy float32, longitud N), devuelve…, Parámetros ---------- Lx, Ly : dimensiones del dominio (m). p0 : presión…

### Community 86 - "Test de simetría del multigrid"
Cohesion: 0.47
Nodes (5): asimetria_p(), main(), Test diagnóstico: mide asimetría introducida por coarsening MG. NACA_0012,…, max|p(x,y) - p(2cx-x,y)| en franja X dentro [cx-banda, cx+banda]., run()

### Community 87 - "Variantes de acentuación"
Cohesion: 0.50
Nodes (4): candidatos(), cargar(), main(), Variantes con una tilde y/o virgulillas.

### Community 88 - "Cola mixta"
Cohesion: 0.80
Nodes (4): estado_de(), lanzar(), mixto.sh script, status()

### Community 89 - "Validación aislada dx=0.002"
Cohesion: 0.70
Nodes (4): ganador_dat(), iters_for(), log(), main()

### Community 90 - "Visor en tiempo real"
Cohesion: 0.50
Nodes (4): conectar_shm(), main(), Viewer en tiempo real para Simulador2D. Se ejecuta como proceso INDEPENDIENTE…, Intenta conectar a la memoria compartida del simulador.

### Community 91 - "Sostenibilidad y huella energética"
Cohesion: 0.50
Nodes (4): Análisis de sostenibilidad (ambiental, económica y social), El coste ambiental de los errores metodológicos, Huella energética y de carbono del trabajo, Surrogate RandomForest desactivado en toda la campaña

### Community 93 - "Script de gran optimización"
Cohesion: 1.00
Nodes (3): gran_optimizacion.sh script, status(), vivo()

### Community 94 - "Consistencia de ranking por dx"
Cohesion: 0.67
Nodes (3): main(), rank(), T1 — Consistencia de ranking dx=0.004 (estudio) vs dx=0.002 (refinado). Re-…

### Community 95 - "Ventana larga a dx=0.001"
Cohesion: 0.67
Nodes (3): criterio(), main(), ¿La oscilación del Cd en el GCI es espacial o de promediado temporal? El GCI de…

## Knowledge Gaps
- **59 isolated node(s):** `Ecuaciones de Navier-Stokes incompresibles 2D`, `Restricción de incompresibilidad`, `Large Eddy Simulation (LES)`, `Smoother Red-Black Gauss-Seidel SOR`, `Capa límite y condiciones de pared` (+54 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Turbulencia LES y Kutta` to `Estudios asintóticos y polares`, `Diagnóstico de presión y lift`, `Auditoría de capa límite`, `Diagnóstico de código del lift`, `Diagnóstico de lift NACA 0012`, `Núcleo Mesh del solver`, `Frontera inmersa y pared`, `Malla estirada y CFL`, `Barrido de modos numéricos`, `Tests de condición de Kutta`, `Gráficas de Cp y convergencia`, `Advección y proyección de presión`, `Diagnóstico de drag de presión`, `Validación de simetría`, `Barrido en ángulo de ataque`, `Vídeo de simulación larga`, `Ghost cells e impermeabilidad`, `Barrido de CFL`, `Barrido de resolución`, `Fuerzas corregidas y multigrid`, `Tests de compatibilidad de flujo`, `Cálculo de drag y lift`, `Campo de divergencia`, `Vídeo del perfil ganador`, `Test de simetría del multigrid`?**
  _High betweenness centrality (0.102) - this node is a cross-community bridge._
- **Why does `Mesh` connect `Núcleo Mesh del solver` to `Frontera inmersa y pared`, `Campo de divergencia`, `Turbulencia LES y Kutta`, `Cálculo de drag y lift`, `Malla estirada y CFL`, `Ghost cells e impermeabilidad`, `Jerarquía multigrid y métricas 1D`, `Gráficas de Cp y convergencia`, `Advección y proyección de presión`, `Tests de compatibilidad de flujo`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Why does `_detect_series_convergence()` connect `Estudios asintóticos y polares` to `Turbulencia LES y Kutta`, `Malla estirada y CFL`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **What connects `Ecuaciones de Navier-Stokes incompresibles 2D`, `Restricción de incompresibilidad`, `Large Eddy Simulation (LES)` to the rest of the system?**
  _59 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Estudios asintóticos y polares` be split into smaller, more focused modules?**
  _Cohesion score 0.0528169014084507 - nodes in this community are weakly interconnected._
- **Should `Calibración frente a XFOIL` be split into smaller, more focused modules?**
  _Cohesion score 0.07407407407407407 - nodes in this community are weakly interconnected._
- **Should `Diagnóstico de presión y lift` be split into smaller, more focused modules?**
  _Cohesion score 0.1347517730496454 - nodes in this community are weakly interconnected._
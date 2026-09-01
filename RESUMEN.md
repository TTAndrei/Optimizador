# RESUMEN — Optimizador CFD 2D

## Qué es
Simulador CFD 2D incompresible (Navier-Stokes) en GPU (CuPy, RTX 3070 Ti) para perfiles alares, base de un futuro optimizador aerodinámico. Todo el solver vive en `Simulador2D.py` (~8000 líneas, clase `Mesh` + `main()`).

- Malla cartesiana estirada (zona fina alrededor del perfil), advección semi-Lagrangiana, difusión (+WALE LES opcional), proyección de presión multigrid/CG.
- Sólidos por IBM (Immersed Boundary): máscara rasterizada + ghost-cell no-slip (`ibm_wall_mode="ghost_noslip"`).
- **Ejecutar SIEMPRE con `.venv/bin/python`** (el python3 del sistema no tiene CuPy).

## DOMINIOS ARBITRARIOS DESDE CAD + INTERFAZ (2026-09-01, sin commitear)

El solver ya no está atado a "un perfil alar dentro de una caja". Admite
**contornos importados de DXF**: un contorno exterior que define las paredes del
dominio (túnel, conducto, tobera, difusor) y N cuerpos internos, con las
condiciones de frontera asignadas **por tramo del perímetro** en vez de por lado
entero. Interfaz nueva en `gui/` (PyQt6 + pyqtgraph): importar, clasificar,
asignar fronteras con el ratón, previsualizar la malla y lanzar.

**Compatibilidad: bit a bit.** Todo lo nuevo va detrás de `Mesh._geom_arbitrary`,
que solo se activa con contorno exterior, con más de un cuerpo o con parches de
frontera. Comprobado con `scripts/agent_tests/equivalencia_bit.py`, que compara
`u`, `v`, `p`, `solid`, `cl` y `cd` con `np.array_equal` contra el
`Simulador2D.py` de HEAD — no con una puerta del 0.5 %, que es lo que hace
`bench_opt --compare` y por donde se cuela un cambio. **Idénticos.** Y sobre una
corrida de producción de 4000 iteraciones (`results/bench_opt/post_f23.json`
frente a `base_f0.json`): **Cl = 0.560895 y Cd = 0.029798 en las dos**, hasta el
último decimal. **32/32 tests.**

### Dos bugs, y el segundo importa más que la feature

1. **`apply_boundaries` corre DESPUÉS del IBM** en los seis caminos de paso
   (`apply_ghost_cell_bc` → `reforzar_impermeabilidad` → `apply_boundaries`) y
   escribía la columna o fila **entera** del borde. En cuanto una pared de
   conducto llega al perímetro de la caja, esa línea inyectaba velocidad de
   corriente libre en celdas de pared que el IBM acababa de poner a cero: fuente
   de masa dentro de la pared, y el conducto fugaba. Arreglado con el código
   `BC_WALL`, que marca las celdas de borde sólidas como "no escribir nada".
   Sin esto ningún caso analítico pasa.
2. **`generar_malla_estirada_intervalo` deja una celda residual diminuta en los
   extremos** cuando la banda fina no cuadra con el dominio, y como `Mesh.dx` es
   el **mínimo** de los espaciados, esa sola celda se lleva por delante el paso
   de tiempo: el límite viscoso va con `dx²`, así que `dx/12` en una celda divide
   el `dt` por **145**. Medido en la tobera de prueba: `dt` de 3.4e-5 a 5.0e-3.
   **Es el mismo mecanismo que hundió la malla de dx=0.001 del estudio** (ver más
   abajo), y ya está actuando en la ruta antigua: el estudio de coste pidió
   dx=0.004 y la malla salió con `dx_min=0.003452`, un 14 % menos. Arreglado con
   `absorber_slivers()`, **aplicado solo en la ruta nueva** para no mover ningún
   resultado publicado. Merece la pena mirar si conviene extenderlo.

### Coste de sacar frames — medido, no estimado

`scripts/agent_tests/coste_salida.py` → `results/coste_salida/INFORME.md`.
26 configuraciones (13 × 2 mallas) con la configuración de las polares finales
(dominio C, presupuesto 2×3, SA-BC, las cinco optimizaciones de `opt_solver`).
El bloque de salida está ahora desglosado en sub-cronómetros
(`guardado_fuerzas` / `_series` / `_shm` / `_frames` / `_dump`), porque los
cuatro caían en el mismo cubo y no se podía separar nada.

Lo que se mide es el **coste por evento**, que es lo único independiente de la
cadencia; el `it/s` entre corridas tiene ±3 % de ruido y no sirve.

| vía | ms/evento (dx=0.004) | ms/evento (dx=0.002) | a cadencia 50 |
|---|---|---|---|
| memoria compartida (vista en vivo) | 0.6 | 1.8 | **0.08 %** |
| ídem + vorticidad (`live_view`) | 1.3 | 3.8 | 0.2 % |
| `dump_fields` comprimido (`max_nx`=900) | 21 | 20 | 0.9 % |
| `dump_fields` **sin comprimir** | 3 | 3 | 0.1 % |
| `save_frame` dpi=600 **dentro del solver** | 753 | 1178 | **53 %** |

- **La vista en vivo es gratis.** No hay motivo para apagarla. Y hasta ahora se
  pagaba *siempre*, mirase alguien o no: el campo se publicaba incondicionalmente.
  Nuevo flag `shm_publish` (por defecto `True`, sin cambio de comportamiento) y
  `shm_cada` para bajar la cadencia.
- **El volcado a disco lo domina zlib**: el 85 % del coste. Quitarlo cuesta ×2.2
  de disco (`dump_fields_comprimir=False`).
- **Dibujar dentro del solver es ×60 el volcado equivalente.** Volcar campos y
  montar el vídeo después con `render_videos.py` da lo mismo por una fracción y
  sin ocupar la GPU. La GUI lo desaconseja con el número al lado.

### Qué hay nuevo

**Solver** (`Simulador2D.py`, +620/−199): `parse_geometry_file()` a nivel de
módulo (el recorte de borde de salida pasa a ser `trim_te=True`, opcional);
`Mesh.rasterize_polygon(region="inside"|"outside")`; `Mesh.add_body()` y
`Mesh.body_id`; `Mesh.set_boundary_patch()` con arrays de código por celda de
borde y `_apply_boundaries_por_celda()`; `Mesh.preparar_geometria()`;
`absorber_slivers()`; kwargs `escena`, `shm_publish`, `shm_cada`, `bench_sync`,
`dump_fields_comprimir`. Además se invalida `_opt_bc_cache` en
`_init_mg_hierarchy` — se indexaba **solo por nivel y no caducaba nunca**, así
que cualquier cambio posterior de geometría o de presión fijada lo dejaba
obsoleto y en silencio equivocado.

**Con geometría arbitraria se DESACTIVAN** la circulación, el diagnóstico de Cp y
el Kutta, y se dice por qué. `_airfoil_bbox` toma el bbox de *toda* la máscara
sólida: con un conducto devuelve el dominio entero y de ahí sale una "cuerda" que
envenena `Cl_circ`, los bins de Cp y el `Cp_min` sin dar error (se vio: Cd=10.4,
Cp_min=303). Un número plausible y equivocado es peor que ninguno.

**Módulos nuevos**: `geom_import.py` (DXF: aplanado con error de cuerda acotado,
cosido de segmentos sueltos por KDTree, anidamiento, orientación, `$INSUNITS`,
filtro por capa; **los lazos que no cierran se reportan con sus coordenadas y no
se cierran a la fuerza**). `gui/` (8 módulos): `escena` (modelo serializable),
`lienzo`, `panel` + `params_spec` (formulario declarativo), `vista_malla`
(previsualización con las comprobaciones baratas), `runner`, `run_solver`, `app`.

**Comprobaciones antes de gastar GPU** (`gui/vista_malla.py`), todas sobre la
máscara ya rasterizada en CPU: pared más fina que una celda (no existe, el fluido
se fuga), hueco de fluido por debajo de ~8 celdas (el IBM se atasca y la garganta
se tapona), cuerpo fuera de la banda fina (normales mal), sin entrada o sin
salida (Poisson singular), contorno que no cruza el perímetro, `dt` previsto
distinguiendo el límite convectivo del viscoso.

**Casos de prueba con verdad conocida** (`tests/test_dominio_arbitrario.py`):
canal de Poiseuille (balance de masa < 2 %, perfil parabólico < 10 %) y tobera
convergente 2:1 (balance < 3 %, aceleración = razón de alturas < 5 %). Geometrías
en `scripts/agent_tests/formas_dominio.py`. Ejemplos listos para abrir:
`.venv/bin/python scripts/agent_tests/demo_gui.py` → `results/gui_demo/`.

**Arrancar la interfaz**: `./lanzar_gui.sh`. Necesita `python3-pyside6.*`,
`python3-pyqtgraph` y `ezdxf`; el venv tiene ahora
`include-system-site-packages = true` (copia en `.venv/pyvenv.cfg.bak`).
**Ojo: la GUI usa PyQt6, no PySide6** — pyqtgraph elige PyQt6 cuando está
disponible, y cargar los dos bindings en el mismo proceso rompe los imports de
PySide6.

**Lo que queda fuera a propósito**: paredes `slip` por cuerpo; varias presiones de
salida distintas (`fixed_pressure_value` es un escalar); entradas que no caen en
el perímetro de la caja; bandas finas múltiples.

---

## RESULTADOS FINALES DEL TFG — CERRADOS 2026-08-26

**Estos son los resultados definitivos que se presentan.** Todo en
[`resultados_finales/`](resultados_finales/); leer su
[`README.md`](resultados_finales/README.md) y
[`richardson/INFORME.md`](resultados_finales/richardson/INFORME.md) antes de
citar nada. Inventario de las 744 figuras en
[`resultados_finales/FIGURAS.md`](resultados_finales/FIGURAS.md).

**72/72 puntos**: 2 perfiles × 4 mallas (dx = 0.008 / 0.006 / 0.004 / 0.002) × 9
ángulos (α = 0 a 8 **de grado en grado**), dominio 24×16, Re=1e5, t≈20 en todas,
ventana completa sin parada anticipada, ~10 h de GPU. Runner
`scripts/agent_tests/resultados_finales.py`, reanudable por punto.

**Terna del GCI: 0.008 / 0.004 / 0.002**, r=2 constante. La de 0.006 se calculó
como contraste y **perdió** sobre los mismos 54 casos: 35 casos con p utilizable
frente a 28, y GCI mediano 20.1 % frente a 44.2 %. Datos en
`richardson/comparativa_ternas.json`.

**La malla de 0.001 está descartada, y no por presupuesto: el dt colapsa.** De
0.002 a 0.001 el dt baja ×0.15 en vez de ×0.5 —sobra un factor 3.4— y sigue
cayendo durante todo el run (×6.8, sin estabilizar) mientras que a 0.002 es
plano. Las 52000 iteraciones se quedaban en **t≈6 en vez de t≈20**, así que esos
puntos se midieron en pleno transitorio y el Richardson habría comparado t=6
contra t=20.6 creyendo medir discretización espacial. Eso explica sin invocar
física todo lo raro de esa pata: CI95 del 95 % en el L/D, L/D no monótono en α=2
y 4, Cd inflado (0.030 vs 0.017) y el criterio de convergencia que no disparaba
nunca. Es la misma firma que precedía al blowup: `warm_start_filtered` evitó el
NaN pero no curó la inestabilidad de fondo. Los 4 puntos calculados quedan en
`resultados_finales/_archivo_malla_fina_descartada/` como evidencia.

**Punto de diseño α=4**, donde las cuatro mallas ponen el máximo de L/D:

| | Cl fina | Cl extrap | Cd fina | Cd extrap | L/D fina | L/D extrap |
|---|---|---|---|---|---|---|
| Ganador | 0.6965 | **0.6999** | 0.01692 | 0.01541 | **41.15** | 45.4 |
| NACA 0012 | 0.4978 | **0.5103** | 0.02746 | 0.02330 | **18.13** | 21.9 |

El Cl del ganador en α=4 es el caso mejor portado del estudio: **p=2.56, GCI
0.61 %**. **La ventaja del ganador se mantiene en los ocho ángulos con L/D
significativo, en las cuatro mallas y también extrapolada** — es el resultado más
robusto: no depende de la malla ni del método de extrapolación.

**Añadir los ángulos impares cambió la lectura del pico.** Con solo los pares
parecía que α=4 era un óptimo nítido; con α=3 dentro se ve que hay una meseta:
40.44 vs 41.15, un 1.7 %, y los intervalos se solapan. Además el salto de α=2 a
α=3 es enorme (21.6 → 40.4, casi el doble, con el Cd cayendo de 0.0212 a 0.0152):
hay una transición del flujo que los ángulos pares se saltaban por completo.

**Extrapolación robusta — dos correcciones, en `verificacion_numerica.py`.**
Richardson puro se dispara cuando el orden observado tiende a 0, porque el factor
es `1/(r^p − 1)`. Medido aquí: el Cl del ganador en α=6 daba **2.912** con la
malla fina en 0.879 (amplificación ×27.5, p real −0.05), y dos Cd cruzaban a
negativo (−0.0153 y −0.0116).

1. **Umbral `p ∈ [1, 4]`** (`extrapola_robusto`). Por debajo de p=1 la
   amplificación supera 1: el extrapolado se aleja de la malla fina más que el
   salto entero entre las dos mallas más finas. Con el umbral en 0.5, el Cd del
   ganador en α=3 colaba con p=0.762, bajaba un 37 % e **inventaba un pico de L/D
   en α=3 que ninguna de las tres mallas tiene**.
2. **El Cd se ajusta sobre `log(Cd)`**, así el extrapolado sale de un `exp()` y no
   puede cruzar cero.

`orden_con_signo()` expone el signo del orden observado, que la fórmula de Celik
devuelve en valor absoluto — y ese `abs()` escondía justo el caso que más importa
detectar, `p < 0`, que significa que los saltos **crecen** al refinar. **Sale
negativo en la mayoría de los Cd.** `gci_triplete()` y `orden_observado()` no se
tocaron, para que los estudios ya publicados sigan dando lo mismo.

**`warm_start_filtered`: la misma configuración en las cuatro mallas.** Sin él,
`warm_start` con presupuesto 2×3 revienta por blowup (iteración 1340 a dx=0.004,
~1740 a dx=0.001, todos los ángulos). La causa no era la magnitud de la semilla
sino su **estructura**: arrastraba el modo par-impar acumulado, que el multigrid
no reduce porque en malla colocada lo ve casi como núcleo. Un paso de Jacobi
ponderado con ω=½ lo aniquila (aniquilador exacto de ese modo, identidad sobre
los suaves). Coste nulo (387 s vs ~405 s sin `warm_start`), precio en exactitud
−0.25 % en el L/D del punto de diseño. **Se probó antes reescalar la semilla por
el cambio de dt y no funcionó** —solo movió el fallo de la iteración 1340 a la
1800—, y ese fallo es lo que demostró que el mecanismo era estructural.

**Lo que este estudio SÍ cierra**, frente a la versión anterior que quedaba en
negativo: la terna es homogénea en tiempo físico y en configuración de solver,
los dos perfiles se comparan en igualdad, y la ventaja del optimizado es robusta
a malla y a método. **Lo que sigue abierto**: el orden observado del Cd no es
utilizable en la mayoría de los ángulos, así que sus valores extrapolados son
estimaciones indicativas con p=2 nominal, no medidas del error de discretización.

---

## Estado anterior (2026-08-24) — contexto, ya no es el resultado

> Todo lo de esta seccion sigue siendo cierto como diagnostico y como
> historia del proyecto, pero **los valores que se presentan son los de
> `resultados_finales/`**, arriba. Donde haya conflicto, manda el bloque
> de resultados finales.

**SEGUNDA GRAN OPTIMIZACIÓN CERRADA** (11 épocas, refinado del top-3 y GCI de tres mallas del ganador). **La verificación del ganador cierra en negativo, y desde el 14 de agosto se sabe por qué en dos capas distintas:**

1. **El "Cl se movía un 6 % con la ventana" era un bug de mezcla de ventanas, no física** (`08af2c4`). `simular_perfil` dividía un Cl **instantáneo** (de `extract_surface_force_audit`, campo final y solo parte de presión) entre un Cd **promediado** (media de la cola de `cd_vector`, presión + viscoso). En flujo estacionario da igual; sobre el ganador a dx=0.001 —que oscila con periodo 0.20 y no llega a estacionario— el numerador dependía de en qué punto del ciclo cayó la última iteración: 0.7222 instantáneo frente a 0.6879 promediado. Corregido: el L/D usa `cl_val`, media temporal de la **misma** ventana que el Cd; la auditoría se conserva como diagnóstico en `cl_audit_inst`.
2. **Lo que sí sobrevive: el Cd oscila entre mallas y arrastra al L/D.** Y el estudio de Richardson sobre la polar completa (`835a874`, 5 ángulos × 3 mallas, ~48 h GPU) confirma que no es cosa de un ángulo desafortunado: **el Cl converge monótono en los cinco ángulos, el Cd rompe la monotonía en α=4 y α=6**. Ningún GCI de Cd ni de L/D es citable; solo la banda entre mallas.
3. **Y hay un tercer sesgo, medido en agosto y mayor que los dos anteriores: el dominio.** Toda la campaña del GA corrió en Lx=8, Ly=5 con fronteras top/bottom `slip` (túnel cerrado, no campo lejano). Ampliar a 24×16 sube el L/D del ganador de 27.70 a 31.70 a α=4. El **orden** entre individuos parece conservarse (medido en 2 de 3), el **valor absoluto** no.

**VERIFICACIÓN NUMÉRICA — documento en [`docs/verificacion_numerica.md`](docs/verificacion_numerica.md) (10 secciones).** Leerlo antes de citar cualquier L/D, pero **está desactualizado desde el 14 de agosto**: no incorpora la ventana larga, ni el fix del fitness, ni el estudio de dominio, ni el Richardson de la polar. Guion del TFG en `docs/PROMPT_TFG.md`; memoria en construcción en `docs/tfg/`, con el texto de sustitución de los capítulos 4-6 ya escrito en `docs/tfg/resultados_finales_v2.txt` (`99ffaa2`).

**Reorganización de resultados**: toda la campaña anterior está archivada en **`results/1eraGranOptimizacion/`** (`convergence_study/`, `verificacion_numerica/`, `agent_tests/`, `barridos/`, `metricas_ga/`, `comparativas_v1/`, `videos_ganadores/`, …). `results/verificacion_numerica/` en la raíz solo contiene lo nuevo. Cualquier ruta de este documento o de scripts antiguos que apunte a `results/convergence_study/...` hay que leerla bajo `results/1eraGranOptimizacion/`. El `.gitignore` se reancló a patrones `**/` porque los anclados a `results/` dejaron de aplicar tras el archivado.

### Optimización del solver de presión — 5.4× por polar, ACTIVA POR DEFECTO (2026-08-24, sin commitear)

Documento completo en [`docs/OPTIMIZACION_SOLVER.md`](docs/OPTIMIZACION_SOLVER.md). Polar completa (6 ángulos, dx=0.002, dominio C) de **7.73 h a 1.43 h**, con la divergencia residual **por debajo** de la que daba la configuración anterior.

**Vuelta atrás**: `OPT_SOLVER_OFF=1` más `mg_max_outer=8, mg_cycles_per_outer=5` reproduce el comportamiento previo, comprobado bit a bit.

**Estructura**: todo lo nuevo vive en `opt_solver.py`; `Simulador2D.py` sólo tiene 9 sustituciones línea por línea que delegan en él (**41 inserciones, 11 borrados** sobre 8866 líneas), cada una conservando la rama original. Con las opciones apagadas, el simulador reproduce el baseline con 14/14 métricas idénticas y trayectorias con delta 0.000e+00.

**Lo que está activo**, y su ganancia:

| opción | ganancia | ¿cambia resultados? |
|---|---|---|
| `fast_masks` | ×1.32 | **no, bit a bit idéntico** |
| `interp_float32` | ×1.17 | +0.012 % en Cl a t=20 |
| `warm_start` | ×1.88 | sí — mejor convergido |
| `coarse_mask_majority` | ×1.14 | sí — mejor factor multigrid |

Más el presupuesto de proyección: **8×5 → 2×3** (en `main()` y en `RunGA.CONFIG`).

**Hallazgos que importan más allá del rendimiento:**

1. **El multigrid nunca alcanzaba su tolerancia** (`frac_converge = 0.0`), y la causa es de discretización, no del solver: malla colocada, y en el modo tablero el gradiente centrado se anula (símbolo de L = −4/h², de D·G = 0). **El 46.2 % de la divergencia residual es modo par-impar**, que ninguna iteración puede eliminar. `_build_projection_faces` no escapa: interpolación lineal pura, sin Rhie-Chow.
2. **La divergencia media es mal indicador de convergencia.** Subir el presupuesto de 8×5 a 12×6 no mueve la divergencia (−0.7 %) pero cambia Cl un +4.00 %. Es justo lo que usa `tol_div` como criterio.
3. **La configuración anterior no estaba convergida en presupuesto de proyección.** Por debajo de 8×5 la degradación es monótona y fuerte; por encima hay ~4 % de dispersión en Cl. Es incertidumbre numérica real, del orden de la banda del GCI (Cl 4.25 %), y **no figura junto a la de malla en el estudio**.
4. **La malla es fuertemente anisótropa** (30 % de celdas con relación de aspecto > 8, anisotropía del operador hasta 2500) y eso rompe las **dos** mitades del multigrid: el suavizador punto a punto y el coarsening isótropo. Levantar el tope de niveles **empeora** (divergencia +62 %). La relajación por líneas arregla el suavizador y aun así pierde: cuesta 5.1× por ciclo.
5. **Fuga de float64**: `_bilinear_interpolate` promocionaba a doble precisión (`float32 − int32 → float64`), y `advect_sa` rebindeaba `nu_tilde`, con lo que **el modelo SA entero corría en float64** — a 1/64 de velocidad en esta GPU. Descartado como riesgo de precisión con medida: la cancelación en `fv2` empieza en chi > 1e4 y el chi real máximo es 50.4.

**Validación**: el control con la configuración antigua reproduce el punto publicado (Cl = 0.680389, Cd = 0.019986) **hasta el último decimal**. Polar completa en `results/polar_optimizada/`.

| α | ΔCl | ΔCd | ΔL/D |
|---|---|---|---|
| 0 | −7.89 % | −2.09 % | −5.92 % |
| 2 | +1.69 % | −7.23 % | +9.62 % |
| 4 | +2.54 % | **−15.38 %** | **+21.19 %** |
| 6 | +2.70 % | −3.03 % | +5.90 % |
| 8 | +2.22 % | −2.81 % | +5.17 % |

**Referencia externa**: el simulador tiene un sesgo documentado de sobrestimar Cd (+124 % contra XFOIL). La polar corregida **se acerca a XFOIL en los cinco ángulos, sin excepción**. Y desaparece una anomalía: el estudio publicado daba L/D = 34.04 para el perfil optimizado frente a 35.3 de XFOIL para un NACA0012 **sin** optimizar; con la corrección sale 41.26.

**Lo que NO se sostiene, y conviene no citar**: el mecanismo propuesto para el cambio en Cd (se predijo monótono con el ángulo y los datos lo desmienten — hay un pico agudo en α=4 y cae a ~−3 % en α=6 y 8); α=0 va en dirección contraria en Cl sin explicación; que 41.26 sea el L/D convergido no está demostrado; y la métrica `cl_cp_discrepancy` se retiró del análisis por una inconsistencia entre su valor y su bandera.

**Consecuencia**: las polares publicadas y el estudio de Richardson **no cambian** y siguen siendo reproducibles. Pero no se deben **mezclar** configuraciones: la diferencia es del orden de lo que el estudio de malla pretende medir.

### Criterio de parada — tercera versión, la puerta de ruido pasa a ser el CI95 (sin commitear)

Historia completa, porque las tres versiones anteriores fallaron por **medir la cosa equivocada** y conviene no repetirlo:

1. **Original**: comparaba cada serie con la media de su propio último 15 % — "¿me parezco a mí mismo hace poco?" en vez de "¿he dejado de cambiar?". Una deriva lenta y monótona lo satisface siempre: paraba en t≈1–2, errores hasta **47 %** en L/D. Lo enmascaraba el relleno de cola de `cdvector`/`clvector`, que daba σ=0.0 exacto (léase "no hay datos"): el 81.8 % de las 3884 evaluaciones de `aprendizaje_ML.jsonl` tienen σ=0.
2. **Recalibrado por tolerancias** (`clcd_tol_abs`/`clcd_tol_rel`): 0.82 % de error **de entrenamiento sobre 2 series**; sobre casos nuevos llegaba a **−23.7 %** con signo impredecible. El defecto estaba en la pregunta, no en los umbrales. Ambos kwargs **se eliminaron** (no desactivados): cualquier script que los pase falla.
3. **drift + ruido crudo** (`drift<0.005` y `σ/|media|<0.05` sobre ventana móvil de L/D, `8ab2461`): arregló el flujo estacionario y volvió el criterio **inaplicable con desprendimiento**. σ/|media| mide la amplitud de una oscilación física y permanente, no la incertidumbre del promedio: en el ganador a dx=0.001 vale 8.9 % con ventana 1 y **empeora a 11.7 %** al ensancharla, así que no abrió en 192000 iteraciones (t=71) mientras el drift ya pasaba holgado. También se descartó atar la ventana al periodo de desprendimiento: a dx=0.001 la estela no es periódica limpia (el pico de autocorrelación no llega a 0.2), no hay periodo que medir.
4. **En producción ahora** (`_detect_series_convergence` reescrita, calibrada por `scripts/agent_tests/criterio_ci95.py`): dos condiciones sobre una **ventana móvil que se ensancha sola** — se parte de `window_conv_time` y se dobla hasta que el CI95 baja del umbral o se agota la serie.
   - `ci95 = 1.96·σ/√N_ef / |media| < tol_ci95`, con **`N_ef` corregido por autocorrelación** (`_n_efectivo`, tiempo de autocorrelación integrado truncado en la primera autocorrelación no positiva). Con desprendimiento las muestras están correlacionadas y el CI95 ingenuo se cree N muestras cuando tiene unos pocos ciclos.
   - `drift = |pendiente|/|media| < tol_drift`, sobre esa misma ventana. Ensanchar de más no cuela un transitorio: un transitorio tiene pendiente y el drift lo caza.
   - Ensanchar hasta que el CI95 baje **no necesita que la señal sea periódica**: vale igual para un ciclo límite y para una estela de banda ancha, y no añade ningún parámetro nuevo.
   - Valores: **`clcd_tol_drift=0.002, clcd_tol_ci95=0.02, clcd_window_conv_time=1.0` (mínima, se ensancha), `clcd_n_sostenido=3, clcd_min_t_fisico_before_check=5.0`**.
   - **Calibrado sobre dos grupos, y los dos entran**: 20 series estacionarias a dx=0.004 del historial del GA + series con desprendimiento (polar del dominio C a dx=0.002, asintótico de α=4 hasta t=41.5, ventana larga a dx=0.001). Ajustar solo sobre las estacionarias es exactamente lo que produjo el criterio que no abre nunca. Resultados: CV peor error **1.15 %** (coste 0.72); grupo estacionario err máx 0.96 %, para en 15/16, coste 0.74; grupo desprendimiento err máx 1.15 %, para en 5/9, coste 0.78; conjunto retenido de 5 series err máx **0.43 %**, coste 0.88.
   - `criterio_ci95.py` **no reimplementa nada**: importa `_detect_series_convergence` del solver y la evalúa sobre las series guardadas, así que lo validado es el código que corre.
   - Fuente de verdad en fichero: `results/verificacion_numerica/criterio_parada.json` (**sobrescrito con el criterio CI95**; el ajuste completo está en `criterio_ci95.json`), leído por `verificacion_numerica.criterio_calibrado()`, `run_convergence_study.py`, `gci_ganador.py`, `test_espesor_malla.py` y `ventana_larga_dx001.py` — todos migrados de `tol_noise` a `tol_ci95`. `scripts/agent_tests/criterio_parada.py` queda marcado como **SUPERADO** en su docstring.
   - **`min_t=5.0` sigue siendo lo que arregla el modo de fallo restante**: sin él, dos series paraban en t≈4.1–4.4 con +10.6 % y +19.6 % — mesetas falsas previas al reataque de la burbuja laminar.
   - **Ojo antes de commitear**: el árbol tiene `_n_efectivo` **definida dos veces** en `Simulador2D.py` (líneas 7099 y 7121, idénticas). Borrar una.
- **El residual de campo no se usa** en el criterio de L/D (`tol_res=inf`). Pero **el guardián de estado estacionario `stop_on_convergence` sí actúa y ya ha estropeado datos**: truncó α=8 (10100 de 26000 iters) y α=10 (7950) en `polar_2grados_dom24x16`. En estudios de malla o de ventana hay que desactivar **las dos** paradas. `change_v` se descarta como residual: se normaliza por la norma de v, pequeña en flujo casi horizontal, vale ~0.5 permanentemente.
- **Cableado verificado en vivo** con la versión 3 (individuo 2019, dx=0.004): offline t_parada=7.013 / L/D=18.716 vs solver 7.0135 / 18.697. Ahorro frente a presupuesto fijo: 33–45 %.

**¿Sobrevive el ranking del GA? Medido, y la respuesta es no en la zona que importa** (`revalidar_ranking.py`, `results/1eraGranOptimizacion/verificacion_numerica/revalidacion_ranking.json`). Muestra estratificada por deciles de n=20 individuos del historial (filtrada a los 2425 registros de dx=0.004), re-simulada a dx=0.002 sin parada y con presupuesto fijo:

- Spearman ρ global **0.7955**, Kendall τ 0.6421, Pearson 0.8341, cambio mediano **+65.3 %** (IQR 47–104).
- Separando por la mediana: ρ=0.612 en los 10 peores, **ρ=0.152 en los 10 mejores**. Solape top-4 = **0.25**.
- Casos ilustrativos: **2019** es el mejor real de la muestra (L/D 24.06) y es de **generación 1** — el GA lo rankeó octavo y lo descartó. **2312** (gen 17) es el único con Δ negativo (−10 %): la parada le regalaba Cd bajo; cae del puesto 3 al 13.
- **Aviso metodológico**: con n=12 la lectura parcial era la contraria (top-4 en orden exacto). Una muestra estratificada truncada a mitad de ejecución no es una muestra estratificada.
- Conclusión defendible: el GA **sí se movió al barrio correcto** (ρ generación–L/D real 0.41, p=0.073) pero **eligió la casa con una señal que no discrimina a esa escala**.

**Y dx=0.004 SÍ sirve para rankear — el problema era la parada, no la malla** (`series_dx004.py`, mismos 20 individuos a dx=0.004 con presupuesto fijo):

| señal de fitness | ρ global | ρ mitad alta | solape top-4 |
|---|---|---|---|
| fitness que usó el GA (dx=0.004 + parada sesgada) | 0.7955 | 0.152 | 0.25 |
| **dx=0.004, presupuesto fijo / criterio nuevo** | **0.9263** | **0.7455** | **1.00** |

El sesgo en valor absoluto a dx=0.004 es de **−44.9 % ± 68** y no importa para el orden, pero **obliga a validar el ganador a dx=0.002**. Dos avisos: **1265** da L/D=**−1.29** a dx=0.004 (real 0.559) — el GA nunca vio fitness negativo porque la parada lo enmascaraba; **2312** engaña a las dos mallas gruesas por igual, ahí el artefacto es geométrico. **Matiz de agosto**: medido sobre geometrías del historial viejo; el ganador de `tfg2` (espesor 2.2 %) muestra que a esa escala de cuerpo fino el orden entre mallas deja de ser fiable.

**Validación externa contra XFOIL** (NACA0012 Re=1e5, Ncrit=9, airfoiltools; `validacion_xfoil.json`): **la sustentación es buena, la resistencia no.** MAE(Cl)=0.0912, pendiente dCl/dα=0.10211 vs XFOIL 0.11107 y teoría 2π 0.10966. MAE(Cd)=0.036, **error medio en Cd +123.8 %, monótono creciente con α** (−2.4 % a α=0 → +226 % a α=10). Causas plausibles: resolución de capa límite, separación/reataque prematuro de la burbuja, bidimensionalidad. **El L/D absoluto NO es comparable con XFOIL ni con experimento, y como el error de Cd depende de α tampoco es un factor de escala divisible.** Lo que sí se sostiene: Cl, pendiente de sustentación, y comparación perfil-contra-perfil a igual α, misma malla **y mismo dominio**.

**Transición SA-BC implementada y validada** (`8e29bd5`): `transition_model="sa_bc"` + `freestream_Tu=0.1` (default `"none"`, requiere `turb_model="sa"`). Intermitencia algebraica de Bas-Cakmakcıoğlu 2016: γ = 1−exp(−√T1−√T2) con Re_θc por la correlación de Menter, modulando **solo la producción** de SA; `SA_BC_CHI1=0.002`, `SA_BC_CHI2=5.0`; γ en `Mesh.sa_gamma`. Validada en `results/1eraGranOptimizacion/agent_tests/sabc_*`.

**Campaña de GA "1eraGranOptimizacion" completada** (`74add38`, `43d95e9`): modo `--mixto <tag>`, 4 semillas nuevas aptas a bajo Re (`SD7037`, `E387`, `SG6043`, `MH32`), flags `--multipunto/--delta-angulo/--fitness-modo`, `--transition-model/--freestream-tu`, `--run-tag`. **Todos esos L/D son de la señal sesgada; no citar.**

**Métricas del GA** (`scripts/agent_tests/metricas_ga.py`, sin GPU, ~1 min): 15 figuras + `INFORME.md`. Hallazgo: `usar_ia=False` en los 6 estudios, el surrogate RandomForest nunca filtró nada. Reentrenado offline sobre 2793 muestras Re=1e5: R²=0.850 (5-fold CV), MAE=1.07; a percentil 70 evitaría el 69 % del CFD perdiendo solo el 2.5 % de la élite. **Aún no pasado sobre `tfg2`.**

**Qué sigue siendo válido de la campaña anterior**: la metodología (GA de islas con migración, parametrización, solver, `wall_treatment="consistent"`, SA-BC), el coste (98.6 h-GPU, 3884 evaluaciones) y las conclusiones sobre el comportamiento del GA (migración, diversidad, presión selectiva, PCA: PC1+PC2 explican 75.5 % de la varianza geométrica). **Qué no**: todo valor absoluto de L/D anterior al recalibrado, la afirmación de que el ranking se conserva, los ganadores como óptimos, y el L/D como magnitud comparable con la literatura.

**`cl_cp_discrepancy` cayó de 0.31–0.83 a 0.02–0.14** al medir sin transitorio: buena parte de la sobre-circulación diagnosticada era artefacto. Queda un residuo real, menor, que crece con Re y que sobre cuerpos muy finos vuelve a levantar el flag (0.02–0.09 en las polares del dominio C).

### Segunda gran optimización (lanzada 2026-08-04, CERRADA 2026-08-13 en la época 10)

Lanzador `scripts/agent_tests/gran_optimizacion.sh {start|run|stop|status|snapshot|log}`; salidas en **`results/convergence_study/islands_tfg2/`**, log `results/convergence_study/gran_optimizacion_tfg2.log`, copias en `backups/tfg2_*`.

- **Presupuesto ejecutado**: 4 islas × pop 16 × 2 gen/época × **11 épocas** = 1232 evaluaciones nominales; **44 corridas GA cerradas y ~1160 evaluaciones CFD reales** (`aprendizaje_ML.jsonl` pasa de 3884 a **5044** registros), ~230 h-GPU. El plan original eran 13 épocas.
- **El recorte de 13 a 11 épocas fue conservador de más**: la ganancia por época había caído a +0.1…+0.5 con ruido entre épocas de ±1, pero las épocas 9 y 10 volvieron a dar saltos grandes (s1014 28.66, NACA 27.75). **Es la segunda vez que esta campaña finge convergencia**; la migración reordena la población con retardo y las mesetas de dos épocas no significan nada.
- **Config**: `--islands --run-tag tfg2 --island-epochs 11 --island-migrate-every 2 --island-pop 16 --dx 0.004 --t-target 12 --transition-model sa_bc --freestream-tu 0.1 --re 1e5 --refine-k 3 --stop-file results/convergence_study/STOP`. Dominio **8×5** — ver el estudio de dominio más abajo: es el sesgo más grande de toda la campaña.
- **Pausa/reanudación**: `stop` crea el centinela `STOP`; la corrida termina la evaluación en curso (≤12 min), guarda `estado_ga.json` y sale. `guardar_estado_ga` escribe a `.tmp` + fsync + `os.replace` (`2d2d639`). Se pausó/reanudó cinco veces sin perder trabajo pagado.

Mejor L/D por isla y época **a dx=0.004, dominio 8×5** (sirve para ordenar, no como valor final):

| época | GM15 | AG24 | NACA_0012_sharp | s1014 | spread |
|---|---|---|---|---|---|
| 0 | 16.96 | 15.61 | 14.71 | 7.14 | 9.82 |
| 1 | 19.90 | 19.28 | 17.70 | 8.03 | 11.87 |
| 2 | 21.81 | 21.49 | 19.82 | 21.77 | 2.00 |
| 3 | 22.90 | 22.09 | 22.65 | 22.71 | 0.81 |
| 4 | 23.13 | 22.70 | 22.66 | 23.29 | 0.63 |
| 5 | 24.85 | 23.06 | 23.22 | 25.27 | 2.21 |
| 6 | 25.44 | 24.73 | 24.95 | 25.66 | 0.93 |
| 7 | 25.75 | 25.58 | 24.98 | 25.47 | 0.78 |
| 8 | 25.49 | 24.57 | 25.47 | 25.15 | 0.92 |
| 9 | 25.98 | 24.79 | 26.55 | **28.66** | 3.88 |
| 10 | **26.84** | **25.00** | **27.75** | 25.92 | 2.75 |

- **Convergencia de la población**: distancia de forma media 0.0475 (época 0) → **0.0033** (época 10), spread de L/D 9.82 → 2.75. **La migración acerca las islas.** s1014 arrancaba en otra cuenca (distancia ~0.09 frente a ~0.006) y tardó tres épocas en integrarse.

**Refinado del top-3 a dx=0.002 (`refine_dx002.json`): el refinado CAMBIA al ganador.**

| individuo | L/D dx=0.004 | L/D dx=0.002 | Cl (ref) | Cd (ref) |
|---|---|---|---|---|
| **NACA_0012_sharp/epoch10** | 27.75 | **28.88** | 0.6965 | 0.02411 |
| GM15/epoch10 | 26.84 | 27.62 | 0.7725 | 0.02797 |
| s1014/epoch9 | **28.66** | 24.81 | 0.5791 | 0.02334 |

El líder de la exploración pierde 3.85 puntos al refinar mientras los otros dos suben. Ese perfil llevaba el espesor máximo al **84 % de la cuerda**. Lección: dos niveles de malla son necesarios, y el ranking a dx=0.004 deja de ser fiable justo en la zona de geometrías extremas a la que el GA tiende.

**GCI de tres mallas sobre el ganador** (`gci_ganador.py`, NACA_0012_sharp/epoch10, α=4°, Re=1e5, dominio 8×5, dx=0.004/0.002/0.001, r=2, FS=1.25). Geometría: t_max **2.16 %** en x/c=0.658, camber máx 5.64 %. **Recalculado en `08af2c4` con Cl y Cd de la misma ventana** (`gci_recalculado.json`, sin simular nada nuevo, a partir de las series guardadas; ventana t≥14, 3088 muestras, 285 ciclos de desprendimiento):

| magnitud | dx=0.004 | dx=0.002 | dx=0.001 | banda |
|---|---|---|---|---|
| Cl | 0.70144 | 0.68651 | 0.68783 | **0.19 %** |
| Cd | 0.02528 | 0.02388 | 0.02538 | **5.91 %** |
| L/D | 27.7455 | 28.7441 | 27.0971 | **6.08 %** |

- **El Cl queda prácticamente convergido entre las dos mallas finas** (0.6865 vs 0.6878). Ni el `p=1.52` ni el extrapolado 0.6841 de la lectura vieja son citables: con el punto corregido la razón de residuos es negativa y la convergencia es oscilatoria, pero la banda es del 0.19 %, o sea ruido.
- **Cd oscila** (baja y vuelve a subir, razón −0.93) y **arrastra al L/D** (razón −0.61). La banda citable del ganador se estrecha de 26.7–28.7 a **27.1–28.7**, y sigue siendo una banda, no un valor.
- **Pendiente barato**: dx=0.004 y dx=0.002 **todavía usan el Cl instantáneo**; allí el flujo llega a estacionario y la diferencia debería ser pequeña, pero hay que rehacerlas (~66 min) para que las tres mallas usen la misma definición.
- Fluctuación instantánea medida a dx=0.001: Cl **6.5 %**, Cd **15.5 %** — de ahí venía todo el ruido de estimador.
- **Estacionariedad por tramos** (cuatro tramos de t≈14 entre t=14 y t=71): L/D 27.01 / 27.15 / 27.18 / 27.05. El flujo es estadísticamente estacionario aunque nunca alcance estado estacionario.

**Ventana larga a dx=0.001** (`ventana_larga_dx001.py` → `results/verificacion_numerica/ventana_larga_dx001.json/.log/_series.npz`, `0263a13`). 192000 iteraciones de tope (t=71.0 físico alcanzado, cinco veces la corrida del GCI), **16.4 h de GPU**.

- **El criterio de la versión 3 no llegó a saltar** (`converged_clcd=false`, 192000 de 192000): esa fue la prueba que motivó la versión 4 (puerta de CI95). A esta resolución el perfil **no tiene estado estacionario**; las mallas gruesas sí convergen porque la difusión numérica amortigua el desprendimiento.
- **Cd: 0.025430 ± 0.000281 (CI95, 768 muestras) frente a 0.025507 con t=12 → 0.30 %, dentro del CI95.** El promediado temporal no era el problema: **la oscilación del Cd entre mallas es espacial y real**.
- **El Cl NO se movía un 6 %** — eso era el bug de fase del snapshot, corregido en `08af2c4`. La media temporal es 0.6878 con las dos ventanas.
- Queda un residuo real de dispersión entre estimadores sobre el mismo campo: Cl 0.6878 (media de superficie) frente a 0.6508 (circulación en lazo 0.15c) y 0.6419 (∮ΔCp en la cuerda) — **~7 %, no el 12 % que se creía**, pero el flag `cl_cp_discrepancy` sigue levantándose y `ΔCp_TE=−0.092` sigue doblando la tolerancia. Los lazos grandes bajan a Cl_circ 0.51 (0.40c y 0.80c) y 0.29 (1.50c): **la circulación no cierra**.
- **No repetir esta corrida**: el `.npz` de series está commiteado y el análisis se rehace offline.

**Ghost cells del borde de salida entre mallas** (`te_report_mallas.py` → `results/verificacion_numerica/te_report_mallas.json`, ~1 min sin simular): dx=0.004 → 11 ghosts en el TE, dx=0.002 → 14, dx=0.001 → 15; **irreparables en el TE = 0 en las tres** (1 irreparable en todo el dominio a dx=0.002). **La distribución es suave en dx**, así que la hipótesis "las ghost-cells de extradós e intradós se solapan y refinar lo empeora" **no está respaldada por el conteo**: el error de discretización no salta por ahí. La ambigüedad de fuerza sigue abierta, pero el sospechoso ya no es el reparto de ghosts.

### Estudio de independencia de dominio — el sesgo más grande de la campaña (sin commitear)

`scripts/agent_tests/estudio_dominio.py` → `results/verificacion_numerica/estudio_dominio/{INFORME.md,analisis.json,estudio_dominio.csv,png,campos/}`. Ganador de `islands_tfg2`, α=4°, dx=0.002, 13000 iters fijos; **solo cambia el dominio**.

Motivación: top/bottom son `slip` (v=0) — **paredes de túnel cerrado, no campo lejano** (`Simulador2D.py:7234-7235`). Con Ly=5 el perfil ve h=2.5c y la interferencia de sustentación vale σ=(π²/48)(c/h)²=0.033; el inflow fuerza v=0 a solo 2c, donde el vórtice ligado induciría un 2.7 % de U_inf. **Toda la campaña del GA se corrió ahí, así que el 27.75 es el del túnel, no el del perfil.** Los dos errores decaen con leyes distintas (lateral como (c/h)², longitudinal como c/d), así que el diseño separa los ejes: escalera A-B-C-D + factorial 2×2 A-E-F-C + caso G.

| | Lx×Ly | cx | σ | d_up | Cl | Cd | L/D | err Cl vs ∞ |
|---|---|---|---|---|---|---|---|---|
| A | 8×5 (el del GA) | 2 | 0.0329 | 2 | 0.6680 | 0.02411 | 27.70 | 2.36 % |
| B | 16×10 | 4 | 0.0082 | 4 | 0.6739 | 0.02118 | 31.81 | 1.50 % |
| **C** | **24×16** | **6** | **0.0032** | **6** | **0.6806** | **0.02147** | **31.70** | **0.52 % (dentro del CI95)** |
| D | 40×24 | 10 | 0.0014 | 10 | 0.6707 | 0.02201 | 30.47 | 1.96 % |
| E | 8×16 | 2 | 0.0032 | 2 | 0.6299 | 0.02699 | 23.34 | 7.93 % |
| F | 24×5 | 6 | 0.0329 | 6 | 0.6942 | 0.01928 | **36.01** | 1.46 % |
| G | 12×24 | 3 | 0.0014 | 3 | 0.6327 | 0.02553 | 24.78 | 7.52 % |

- **Extrapolación a dominio infinito** con `y = y_inf + k1·σ + k2·(c/d_up)`: **cl_inf = 0.6841**, **cd_inf = 0.01982** (residuo máx 0.0148 y 0.0016).
- **Manda Lx, no Ly.** E (8×16, ancho pero corto) es el peor caso de todos: alargar el dominio lateral sin alargar la estela empeora. G, igual. La estela truncada contra el `p=0` de salida es el error dominante (`u_out_min` 0.943 en A y E, 0.997 en C).
- **F (24×5) da el L/D más alto de todos (36.01) y hay que descartarlo**: `eps_top=0.0229` y `eps_bot=−0.0172` — es bloqueo de túnel puro, no física.
- **Los dos errores NO son aditivos**: el factorial da una interacción en Cl de +0.0245 sobre un efecto lateral de −0.014…−0.038 y longitudinal de +0.026…+0.051. **No vale corregir cada frontera por separado.**
- **Decisión: el dominio de referencia es C (Lx=24, Ly=16, cx=6)** para polares y simulaciones sueltas. Coste medido (`bench_dominio.py` → `bench_dominio.jsonl`): el estirado satura en dx_max, así que ampliar apenas añade celdas; el precio está en el Poisson — 7.18 it/s en 8×5 → 4.76 en 24×16 → 3.91 en 40×24. **C cuesta ~50 % más por iteración que el dominio del GA, no un factor.**

**¿Se conserva el ranking del GA al cambiar de dominio?** (`ranking_dominio.py` → `results/verificacion_numerica/ranking_dominio/`): top-1 y top-2 de la población final reevaluados con **el mismo presupuesto** (13000 iters, la campaña usó 10000/7750/5950 por el early-stop y con n distinto las barras no son comparables). top1 27.75 (fitness GA) → **31.81** en B; top2 27.23 → **29.51** en B. **El orden se conserva y la diferencia (2.30) supera el CI95 de cada uno (~1.1), pero por poco.** **Incompleto**: falta el top-3 y faltan las reevaluaciones en el dominio A con presupuesto igualado — sin ellas no se puede afirmar que el sesgo de dominio sea uniforme entre perfiles.

### Asintótico temporal y polares del dominio C (sin commitear)

**Cortar en t≈10 sobreestima el Cd ~5 %** (`asintotico_alpha4_domC.py` → `results/asintotico_alpha4_domC/`). Ganador, α=4, dx=0.002, dominio C, llevado a **t=41.5** (52000 iters, 3.4 h): el L/D de la cola sube 24.13 (t=5) → 32.33 (t=10) → **34.05 (t=20) y ahí se queda plano** (34.03 / 34.13 / 34.09 / 34.05 hasta t=41.5). **Toda la corrección está en el Cd** (0.02680 → 0.02099 → 0.02003); el Cl apenas se mueve (0.6818 desde t≈12). Doblar de t=20 a t=40 mueve el L/D 0.01.

**Extrapolación a t→∞** (`extrapolacion_polar.py`): **Richardson no vale en el tiempo** — la cola es un transitorio que decae y luego mesetea, no una ley de potencias. Contrastado contra el único punto con verdad medida (α=4, valor real 34.084): Richardson sobre T=5/10/20 da 39.41 (**+15.6 %**), el ajuste exponencial `f(T) = F∞ − A·e^(−T/τ)` sobre T, 2T y 4T da 34.66 (**+1.7 %**). El ajuste solo aplica si la razón de incrementos cae en (0,1); si no (desprendimiento sin meseta, o serie cortada por el guardián de residual) se marca **n/a** — ahí no hay valor asintótico que extraer.

**Polares disponibles, y NO son comparables entre sí punto a punto**:

| carpeta | dominio | α | t | notas |
|---|---|---|---|---|
| `results/verificacion_numerica/polar_{ganador_tfg2,naca0012}.*` | 8×5 (GA) | 0..10 de 2 en 2 | ~10 | con early-stop v3 |
| `results/polar_fina_1grado/` | 8×5 (GA) | 0..10 de 1 en 1 | ~10 | |
| `results/polar_fina_1grado_dom24x16/` | **C** | 1..10 de 1 en 1 | ~10 | el criterio no disparó ni una vez |
| `results/polar_2grados_dom24x16/` | **C** | 0..10 de 2 en 2 | ~20 | **early-stop desactivado**; α=8 y α=10 truncados por el guardián de residual |
| `results/richardson_polar_domC/` | **C** | 0..8 de 2 en 2 | ~20 | 3 mallas, paradas desactivadas |

Polar del ganador vs NACA0012 a dx=0.002 en el **dominio del GA** (L/D, α=0…10 de 2 en 2): 2.53 / 15.49 / **27.70** / 21.50 / 14.91 / 10.32 contra NACA 0.01 / 8.93 / 13.19 / 12.74 / 8.39 / 6.61.

Polar **corregida a t→∞ en el dominio C** (`figuras_analisis_polar.py` → `results/analisis_polar/`, columna `ld_fusion`; **no es simulación nueva**: es la polar de 1 grado a t≈10 multiplicada por el factor F∞/f(t=10) medido en los ángulos pares e interpolado, y solo se dibuja donde ese factor está acotado por dos ángulos con meseta):

| α | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| ganador | 10.37 | 18.50 | 32.40 | **32.67** | 29.01 | 22.01 | 17.79 | 14.36 | 11.72 | 10.09 |
| NACA0012 | 5.46 | 9.63 | 13.43 | **14.38** | 13.19 | 12.60 | 10.17 | 7.10 | 5.51 | 4.19 |

Máximo en el punto de diseño (α=3–4) y ventaja en todo el rango: **mejora de banda ancha, no sobreajuste al ángulo**. La ventaja es ~2.3× a α=4. **El valor absoluto no es citable** (ver Richardson abajo); la relación entre los dos perfiles, medida en el mismo dominio, misma malla y misma ventana, sí.

### Richardson sobre la POLAR — el Cd no converge en malla (`835a874`)

> **SUPERADO por los resultados finales del 2026-08-26** (ver arriba). Se
> conserva porque su diagnostico sigue siendo valido y porque el estudio
> final lo confirma: el Cd no converge en malla. Lo que cambia es la terna
> (0.008/0.004/0.002 en vez de 0.004/0.002/0.001, por el colapso de dt a
> 0.001), el numero de angulos (9 en vez de 5) y que ahora hay una
> extrapolacion robusta que no produce Cl de 2.91 ni Cd negativos.
> **No citar de aqui el 34.04 ni las bandas 19-55 %.**

`scripts/agent_tests/richardson_polar_domC.py` → `results/richardson_polar_domC/{richardson.json,polares.csv,polar_dx*.json,series/,campos/,figuras/,run.log}`. Tres mallas (dx=0.004/0.002/0.001, r=2) × cinco ángulos (0, 2, 4, 6, 8) en el dominio C, **~48 h de GPU** (13000/26000/52000 iters, t≈20 igual en las tres; la malla fina cuesta ~6 h por ángulo).

**Por qué la polar y no un ángulo suelto**: con un solo ángulo no hay forma de distinguir "Richardson no aplica" de "ese ángulo cayó mal". Cinco ángulos dan cinco estimaciones independientes de p.

**Las dos paradas van desactivadas a propósito** (criterio de L/D y guardián de residual): cortan a t distinto en cada malla y meten dependencia de ventana dentro de la diferencia entre mallas, que es justo lo que se quiere medir. El criterio calibrado **sí se pasa al solver para que registre su veredicto sin actuar**, y además se reproduce offline sobre la serie volcada de cada punto con la función del propio solver — así queda medido dónde habría parado sin pagar el sesgo de que parara.

| magnitud | monótonos | p medio | banda media entre mallas |
|---|---|---|---|
| Cl | **5/5** | 1.29 ± 0.32 | 26.5 % |
| Cd | 3/5 (rompe en α=4 y α=6) | 2.98 ± 1.73 | 39.2 % |
| L/D | 4/5 (rompe en α=4) | 2.28 ± 1.43 | 58.8 % |

- **El Cl es lo único citable**: monótono en los cinco ángulos, con p consistente y del orden esperado. GCI fino **1.1–5.7 % fuera de α=0** (α=2: 5.7 %, α=4: 4.2 %, α=6: 1.1 %, α=8: 2.3 %). **α=0 es un caso aparte**: Cl 0.074/0.151/0.187, GCI fino 21 % — a sustentación casi nula el error relativo explota y el punto no dice nada.
- **El Cd rompe justo en el punto de diseño**: a α=4 vale 0.02426 / 0.01999 / 0.02317 (no monótono, GCI fino 50 %), y a α=6 la razón de residuos casi se anula (p=6.44, GCI fino 0.006 % — un artefacto, no una medida).
- **El L/D lo hereda**: a α=4 da 27.62 / **34.04** / 30.17 — banda del **21 %**. Ese 34.04 es el número que sale de la polar del dominio C a dx=0.002, y **la malla fina no lo confirma**.
- **Conclusión operativa: ningún GCI de Cd ni de L/D es citable; solo la banda entre mallas (19–55 %).** Para el Cl, sí se puede citar el GCI fino.
- Los campos finales (355 MB de u/v/p) quedan fuera del índice; están en local para instrumentar el borde de salida. La corrida α=8 a dx=0.001 con ventana corta se archivó aparte como `DESCARTADO_ventana_corta_*` en vez de mezclarla con el resto.

**El adelgazamiento del perfil NO es artefacto de malla** (`test_espesor_malla.py` → `espesor_vs_malla.json`). El GA lleva el espesor máximo del 12 % (semilla) a **~1.9–2.2 %** con el L/D subiendo monótonamente. Cuatro ganadores del mismo linaje reevaluados a dx=0.002:

| individuo | t_max | L/D dx=0.004 | L/D dx=0.002 | Cl | Cd |
|---|---|---|---|---|---|
| e0_GM15 | 6.13 % | 16.96 | 22.01 | 1.048 | 0.0476 |
| e3_GM15 | 3.46 % | 22.90 | 27.22 | 0.730 | 0.0268 |
| e6_GM15 | 1.91 % | 25.44 | 28.82 | 0.601 | 0.0209 |
| e6_s1014 | 2.13 % | 25.66 | **30.17** | 0.728 | 0.0241 |

**El orden se conserva** entre mallas dentro de ese linaje y la mejora solo se encoge de 1.513× a 1.371×. El adelgazamiento es física de Re bajo — el Cd cae a la mitad mientras el Cl solo baja de 1.05 a 0.73. **Matices**: el linaje es homogéneo, y el caso s1014/epoch9 (espesor máximo al 84 % de cuerda) demuestra que fuera de él la malla gruesa sí reordena.

Tres defectos que hacían inviable una corrida de días, corregidos en `5213410`:

1. **Fitness negativo indistinguible de un fallo.** `calcular_fitness` filtraba `ld > 0`, así que un perfil con L/D<0 devolvía 0.0 — el mismo valor que un crasheo. Ahora hay centinela explícito `Individuo.evaluado`; el pool de reproducción pasa de `fitness > 0` a `ind.resultados`.
2. **Reanudar costaba una generación entera**: el checkpoint guardaba el fitness pero no lo restauraba → 16 re-simulaciones, 3.1 h por pausa. Ahora restaura fitness, resultados y `evaluado`.
3. **La pausa no paraba nada**: RunGA absorbía el Ctrl+C y `_run_ga` no lo miraba. Ahora se re-lanza como `KeyboardInterrupt` y las fases que llaman a `simular_perfil` consultan el centinela entre simulaciones.

Además: `parada_pedida()` acepta fichero centinela (las corridas de verificación aceptan `STOP_SIMULATION.trigger` en la raíz); **`guardar_memoria` ya no pisa el histórico** — volcaba `datos_X` tal cual, así que una corrida con memoria vacía lo borraba todo (una prueba de humo de 10 individuos se llevó 2347 experiencias); barra de progreso a 1 actualización/minuto cuando la salida no es TTY; flags `--dx` y `--t-target`; `sim_last.npz`, `_temp_gen*_ind*.dat` y `refine_tmp/` fuera del control de versiones.

**Cambios de código pendientes de commit**: `Simulador2D.py` — `_n_efectivo` (duplicada, limpiar) y `_detect_series_convergence` reescrita sobre CI95 con ventana auto-ensanchable, devuelve `(drift, ci95, converged)`; kwarg `clcd_tol_noise` → **`clcd_tol_ci95`** (cualquier script que pase el viejo falla). `scripts/RunGA.py` — **`dump_field_path`**: vuelca el campo final (u, v, p, máscara sólida y los ejes 1D de la malla estirada) en el instante de parada, sea por convergencia o por contador; de ahí salen los mapas de |u| y streamlines de las polares. `verificacion_numerica.simular(..., dump_field=...)`. Scripts nuevos: `criterio_ci95.py`, `estudio_dominio.py`, `bench_dominio.py`, `ranking_dominio.py`, `asintotico_alpha4_domC.py`, `polar_fina.py`, `polar_fina_dominio_c.py`, `polar_2grados_dom24x16.py`, `polar_naca0012.py`, `extrapolacion_polar.py`, `figuras_analisis_polar.py`.

### Memoria del TFG (`docs/tfg/`)

Pipeline propio: fuente en `memoria.txt` (marcas `#1..#4`, `$$latex$$`, `[FIG]/[TBL]/[TOC]/[PB]`, tablas `| a | b |`), `build_tfg.py` sustituye `word/document.xml` sobre la plantilla oficial EETAC `MaquetaTFG.docx`, `latex2omml.py` convierte el subconjunto de LaTeX usado a OMML (lanza `ValueError` si algo no está soportado, para que falle en generación y no en Word) y `acentuar.py`/`acentuar2.py` restauran tildes usando `/usr/share/dict/spanish`. Salida versionada: `TFG.docx`.

- `resultados_finales_v2.txt` (`99ffaa2`, no se compila): texto de sustitución para los capítulos 4, 5 y 6, con la misma marcación y una tabla de trazabilidad de cada número contra su fichero de resultados. Cae la restricción del capítulo 4 (la sección 4.6 ya se puede escribir), los resultados van con banda de incertidumbre en vez de valor puntual, el capítulo 5 gana tres limitaciones medidas y el 6 cambia todas las cifras de energía y emisiones.
- **Tres apartados reescritos y compilados como extractos sueltos** (sin seguir): `3.4.2` difusión numérica del esquema de advección, `3.8.2` corrección de presión de fondo, `3.10.3` la evaluación CFD como función de fitness — cada uno con su `.txt` fuente, `.docx` y `.pdf`.
- `figuras/diagrama_solver.{excalidraw,png,svg}` — diagrama del solver, sin seguir.
- El flujo está a medio migrar: hay `TFG.odt` y `TFG_v1_con_tus_ediciones.odt` sin seguir (edición en LibreOffice) junto al `TFG.docx` generado. **Decidir si se sigue con `build_tfg.py` o se pasa a edición manual.**

**Estado del árbol**: `835a874` es el último commit; **queda sin commitear todo el bloque de agosto 15-24** — recalibrado CI95 (`Simulador2D.py`, 138 líneas), `dump_field_path` en `RunGA.py`, los 11 scripts nuevos de dominio/polar/extrapolación, `results/analisis_polar/`, `results/asintotico_alpha4_domC/`, las tres carpetas de polares, `estudio_dominio/`, `ranking_dominio/`, `bench_dominio*`, `criterio_ci95.json`, y los apartados del TFG. **`polar_results.json` sale modificado**: lo escribe `main()` en cada corrida suelta, es salida automática, no un resultado curado. **`cerebro_aerodinamico.pkl` sale modificado y sigue fuera de los commits**: en disco está el truncado por la prueba de humo (74 experiencias frente a las 2347 de `8ab2461`) y mezcla genes de longitudes distintas entre islas, así que el oráculo no consigue cargarlo — se reconstruye aparte, **no commitear esa versión**. `RESUMEN.md.tmp` es artefacto del hook. `graphify-out/` sale modificado (grafo del repo regenerado).

## Estado histórico (2026-07-12, rama Malla-Variable)

Caso de trabajo: NACA0012, α=5°, Re=1e5, Cl físico esperado ~0.55.

**FIX DE RAÍZ IMPLEMENTADO: `wall_treatment="consistent"`** (kwarg de `main()`, default "legacy" intacto). El sumidero IBM está eliminado. Componentes:
1. Divergencia por caras en la proyección (`divergence_form="face_flux"` → `_compute_flux_divergence_field_uv`): cara fluido-sólido = flujo 0 exacto → el Poisson ve el flujo contra la pared.
2. Gradiente de presión one-sided en celdas de pared (`wall_pressure_gradient_mode="one_sided"`).
3. `reforzar_impermeabilidad` desactivado (`disable_reforzar=True`) — era la cirugía que borraba masa (Q≈-0.06·U·c).
El Laplaciano masked ya era Neumann correcto; el problema era que divergencia y corrección anulaban la componente entera junto al sólido.

**Resultados validados** (JSON+npz en `results/1eraGranOptimizacion/agent_tests/wc_*`; `a5c14b7`):
- `wc_re1e3_a5` (Re=1e3, α=5): Cl=0.211±0.0004, Cd=0.148, Q_lazo=+0.02, ΔCp_TE=0.04. **Clava el DNS de referencia (Kurtulus Re=1000: Cl~0.22, Cd~0.146)**. Legacy daba 0.14 con Q=-0.06.
- `wc_sa_a0` (α=0, SA): Cl=0.0003, sin NaN — simetría perfecta.
- `wc_sa_a5_dx2_long` (48k iters, t=18.3, 2h GPU): Q=0.0032, **Cl converge en plateau 0.32**. **PROYECCIÓN EXONERADA: Q→0 no recupera el Cl.** Hipótesis Q→Cl pendiente ~20 FALSIFICADA.
- **Diagnóstico del déficit (0.32 vs 0.55)**: SA sobre-difunde. BL en x/c=0.5 de espesor ~0.07c (3× placa plana turbulenta), χ=nu_t/nu~40-50 (teórico ~8), Cp_min=-0.81 (esperado ~-1.9).
- `wc_sa_a5_dx2_lam01` (sa_nu_tilde_factor=0.1): **PEOR** — Cl=0.266, LSB reapareciendo. **Falsificada la hipótesis fully-turbulent**; mantener `sa_nu_tilde_factor=3.0`. (La transición se resolvió después con SA-BC.)
- `wc_sa_a5_dx1` (dx=0.001): **Cl=0.389** (vs 0.32 a dx=0.002). **Camino resolución confirmado, error ~1º orden en dx.**
- **Causa dominante identificada: difusión numérica del SL bilineal, nu_num≈u·dx/2** (independiente de dt) — 1º orden, ~5-10× nu molecular cerca de pared a Re=1e5. A Re=1e3 era ~10 % de nu (por eso clavó el DNS).
- **MacCormack implementado** (`advection_scheme="maccormack"`, default "sl"): corrección Selle 2008 con limitador min/max del stencil y **banda de 2 celdas junto al sólido en SL puro** (sin banda, la corrección sobre el staircase mete pico espurio Cp=-11 en TE). Validación Re=1e3 α=5: Cl=0.245/Cd=0.155.
- `wc_mc_sa_a5_dx2`: **Cl=0.452** (SL daba 0.32), Cp_min=-1.20, x_succión=0.015, Cd=0.040. **Cuarta capa (nu_num del SL) CONFIRMADA y resuelta.** Coste ~4 %.
- `wc_mc_sa_a5_dx1`: **Cl=0.496 ✓ OBJETIVO CUMPLIDO**, Cd=0.0335, Cp_min=-1.24, Q=0.005. **Richardson dx→0: Cl≈0.54 — clava el físico 0.55.**
- **CONFIG DE REFERENCIA DEL OPTIMIZADOR**: `wall_treatment="consistent"` + `turb_model="sa"` + `advection_scheme="maccormack"` (+ `transition_model="sa_bc"`). dx=0.002 para validación, dx=0.004 legítimo para explorar/rankear dentro de un linaje homogéneo.
- **POLAR VALIDADA** (`wc_mc_polar_a0/a2/a8`, dx=0.002): Cl(α)= 0(0.000), 2(0.185), 5(0.452), 8(0.708); Cd= 0.0305/0.0323/0.0404/0.0624. Monotonía ✓, pendiente 0.088/deg ✓, simetría α=0 exacta ✓. **DIAGNÓSTICO DE 4 CAPAS CERRADO.** (Todo sobre NACA0012, espesor 12 %.)
- **Script de barrido CFL/dx/α**: `scripts/agent_tests/run_cfl_sweep.py` — 30 sims resumibles. **Aún no ejecutado.**

**Diagnóstico histórico cerrado (4 capas, evidencia en `results/1eraGranOptimizacion/agent_tests/`)**:
1. LSB laminar a Re≥1e4 (burst t≈5-7, colapso). Fix: **Spalart-Allmaras**. WALE inerte.
2. Pared sub-resuelta: SA necesita dx≤0.002 (y+≲10).
3. Inconsistencia IBM↔proyección: legacy absorbía masa (sobre-circulación 2.2×). **Resuelta con `wall_treatment="consistent"`**.
4. Difusión numérica del SL bilineal. **Resuelta con `advection_scheme="maccormack"`**.
- **Quinta capa abierta (agosto 2026)**: definición de fuerza sobre cuerpos ultrafinos — la circulación no cierra (Cl_circ cae de 0.65 a 0.29 al agrandar el lazo), ~7 % de dispersión entre estimadores de Cl, ΔCp_TE del doble de la tolerancia. **El conteo de ghost-cells del TE es suave entre mallas, así que el reparto de ghosts no es el mecanismo**; refinar tampoco lo arregla.
- Métrica decisiva: Q_lazo=∮u·n dl (`q_loop_025c/075c`).

**GA de optimización de perfiles (`scripts/RunGA.py`)**: `simular_perfil` corre sobre consistent+SA+MacCormack, `main(config)` reutilizable con retorno dict, población mixta multi-semilla (`archivos_base` + `resamplear_a_grid`), parada por `tiempo_limite_s`, mutación con sigma adaptativa, parada por estancamiento, checkpoint de genes en `estado_ga.json` (escritura atómica). Orquestador `scripts/agent_tests/run_convergence_study.py` (deadline-driven, resumible): Fase0 calibración dx, Arm A GA por-semilla, Arm B población mixta, S2 islas con migración (`--islands`), S3 refine top-k, `--aislado`, `--mixto`, `--multipunto`. Memoria de ML persistente: `cerebro_aerodinamico.pkl` + `aprendizaje_ML.jsonl`.

**Estudio de convergencia (CERRADO, `a0ee4f3`)**: Arm A no converge (4 semillas → 4 óptimos locales, CV=0.172); el test aislado sin migración (28 h GPU) da dist-forma media **0.0557** (igual/peor que la partida 0.0507) → **la convergencia que producen las islas la fuerza la MIGRACIÓN, no una física de óptimo único**. Fixes asociados: `bf6824e`, `ac235b1`, `9309f00`, `30c7cf6`.

## Próximos pasos

**Prioridad (2026-08-26, tras cerrar los resultados finales)**

1. **Commitear todo el bloque de agosto.** Son ~12 días sin seguir: recalibrado
   CI95, estudio de dominio, polares del dominio C, optimización del solver
   (`opt_solver.py`), `warm_start_filtered`, la extrapolación robusta y el
   estudio final completo. **Antes**: borrar la definición duplicada de
   `_n_efectivo` en `Simulador2D.py` (líneas 7099 y 7121, idénticas) y decidir
   qué entra de `results/` y de `resultados_finales/` (los `.npz` de campos son
   ~2 GB y el `.gitignore` ya los excluye).
2. **Escribir los capítulos 4-6 con los datos de `resultados_finales/`.** El
   texto de `docs/tfg/resultados_finales_v2.txt` está escrito contra el estudio
   anterior y hay que rehacerlo: cambia la terna, el número de ángulos, los
   valores y la conclusión (antes cerraba en negativo, ahora la ventaja del
   ganador es robusta). Las 9 figuras de `FIGURAS.md` §1 son las que van.
3. **Actualizar `docs/verificacion_numerica.md`**, congelado desde el 14 de
   agosto: no incorpora la ventana larga, el fix del fitness, el estudio de
   dominio, el Richardson de la polar ni nada del estudio final.
4. **Investigar el colapso de dt a dx=0.001.** Es el único obstáculo para una
   terna apoyada en las mallas finas, que sería mejor que la actual. La firma
   (dt cayendo ×6.8 sin estabilizar) apunta a la misma familia que el blowup del
   modo par-impar, que `warm_start_filtered` mitiga pero no cura.
5. **Diagnosticar el exceso de Cd** (+226 % a α=10 frente a XFOIL) y su no
   convergencia en malla, que es la incertidumbre dominante que queda. El GA
   converge a cuerpos de 2 % de espesor, así que un Cd sesgado sesga el óptimo de
   espesor.

**Anteriores (siguen abiertos)**

6. **Cerrar la definición de fuerza sobre cuerpos ultrafinos.** ~7 % entre
   estimadores de Cl, `ΔCp_TE = −0.092`, circulación que no cierra. El conteo de
   ghost-cells **ya está descartado** como mecanismo (`te_report_mallas.json`:
   11/14/15 ghosts, 0 irreparables en las tres mallas); el siguiente paso es
   comprobar si `kutta_enforce` cierra la circulación y decidir qué estimador
   vale. Misma familia que el punto 5.
7. **Terminar `ranking_dominio.py`**: falta el top-3 y las reevaluaciones en el
   dominio A con presupuesto igualado. Sin ellas no se puede afirmar que el sesgo
   de dominio sea uniforme entre perfiles, y de eso depende que la campaña del GA
   sea rescatable como *ranking*.
8. **Post-proceso de `tfg2`**: pasar `metricas_ga.py` sobre
   `results/convergence_study/islands_tfg2/` y **reconstruir
   `cerebro_aerodinamico.pkl`** — el de disco está truncado a 74 experiencias y
   mezcla genes de longitudes distintas entre islas; hay que resamplear a un grid
   común antes de reentrenar. Es la causa del error no fatal "inhomogeneous
   shape" que salía en cada arranque de isla, por el que el surrogate arrancaba
   en frío cada época.
9. **Activar el surrogate** (`usar_ia=True`, percentil 70): ahorro estimado del
   69 % del CFD perdiendo el 2.5 % de la élite. Reentrenar sobre los 1160
   registros nuevos de `tfg2`, no sobre el historial sesgado. Depende del 8.
10. **Si hay una tercera campaña de GA, correrla en el dominio C** — cuesta ~50 %
    más por iteración, no un factor, y el 8×5 mete un sesgo de +4 puntos de L/D en
    el valor absoluto. Alternativa barata: 8×5 para explorar, refinar el top-k
    en C.
11. **GA multipunto α ∈ {2,4,6}**: etapa `multipunto` ya cableada en
    `cola_tfg.sh`, fuera de la cola por defecto. **Antes conviene resolver 5 y 6**
    — si la fuerza no está bien definida en la geometría a la que converge el GA,
    multiplicar ángulos multiplica el problema.
12. Estudio de convergencia CERRADO: no relanzar corridas de diagnóstico de la
    misma pregunta. Direcciones abiertas: más diversidad/multi-arranque para un
    óptimo global real; revisar físicamente el óptimo de camber alto.
13. **Ejecutar el barrido** `scripts/agent_tests/run_cfl_sweep.py` (30 sims,
    resumible) — sigue pendiente, ninguna sim lanzada.
14. Opcional barato: probar MacCormack + turbo_hd (~19 it/s esperado) para
    pre-screening.
15. Actualizar rutas en scripts que apunten a `results/convergence_study/...` o
    `results/verificacion_numerica/{gci,polar,reeval_*}.json` — los de la campaña
    vieja viven bajo `results/1eraGranOptimizacion/`. **Cuidado**:
    `results/convergence_study/` contiene la campaña NUEVA (`islands_tfg2/`), así
    que una ruta antigua ya no falla: apunta a datos distintos.
16. `polar_results.json` en la raíz lo escribe `main()` en cada corrida suelta y
    se ensucia con puntos de verificación. No es una polar curada; no leerlo como
    resultado.

**Cerrados por el estudio final del 2026-08-26**, no rehacer:

- ~~Rehacer dx=0.004 y 0.002 del GCI del ganador con el Cl promediado~~ — el
  estudio final recalcula los 72 puntos de cero con la misma definición de fuerza
  en todas las mallas.
- ~~Dar bandas en vez del 28.88 y el 34.04~~ — sustituidos por los valores de
  `resultados_finales/`, con extrapolación robusta y GCI por magnitud.

## Tests

**Del estudio final (2026-08-26)** — ya ejecutados, `resultados_finales/`:

- `scripts/agent_tests/resultados_finales.py` — runner completo: 72 puntos,
  reanudable por punto, Richardson e informe. `--solo-analisis` y
  `--solo-figuras` rehacen sin simular. **~10 h GPU, ya ejecutado; no relanzar.**
- `scripts/agent_tests/rf_comparar_ternas.py` — compara 0.008 vs 0.006 sobre el
  mismo conjunto de casos. Sin GPU, segundos.
- `scripts/agent_tests/rf_richardson_figuras.py` — las 6 figuras individuales de
  Richardson, líneas suavizadas con PCHIP. Sin GPU.
- `scripts/agent_tests/rf_figuras_extra.py` — Cp de superficie (leído del
  contorno de la máscara, sin reproducir la rotación del perfil), polar de
  resistencia, malla y comparativa extrapolada. Sin GPU.
- `scripts/agent_tests/rf_progreso.py` — visor de progreso por memoria
  compartida; funciona con el runner redirigido a fichero.

**Anteriores**:

- `scripts/agent_tests/richardson_polar_domC.py` — Richardson de tres mallas × cinco ángulos en el dominio C, paradas desactivadas, presupuesto fijo t≈20 → `results/richardson_polar_domC/`. **~48 h GPU, ya ejecutado y commeteado (`835a874`); no relanzar.** Reanudable por punto.
- `scripts/agent_tests/te_report_mallas.py` — ghosts e irreparables en el TE para las tres mallas. Solo construye la malla y corre una iteración: ~1 min, sin GPU real → `results/verificacion_numerica/te_report_mallas.json`.
- `scripts/agent_tests/criterio_ci95.py` — recalibración del criterio de parada con puerta de CI95 sobre N efectivo; importa la función del solver y la evalúa sobre series guardadas (sin GPU) → `criterio_ci95.json` (+ sobrescribe `criterio_parada.json`). **Sustituye a `criterio_parada.py`, que queda como histórico.**
- `scripts/agent_tests/estudio_dominio.py [--solo A,E] [--analisis]` — independencia de dominio, 7 casos a dx=0.002 → `estudio_dominio/{INFORME.md,analisis.json,csv,png,campos/}`. **No se lanza solo.** ~5 h GPU.
- `scripts/agent_tests/bench_dominio.py --lx 12 --ly 8 --cx 3` — coste por iteración vs tamaño de dominio, un caso por invocación (subproceso limpio, sin fragmentar la GPU) → `bench_dominio.jsonl`.
- `scripts/agent_tests/ranking_dominio.py [--analisis]` — ¿el orden del GA sobrevive al cambio de dominio? Mismo presupuesto para todos. **Incompleto: falta el top-3 y el dominio A.**
- `scripts/agent_tests/asintotico_alpha4_domC.py [--analisis]` — α=4 del ganador hasta t≈41.5 en dominio C, vuelca serie completa y campo final → `results/asintotico_alpha4_domC/`. 3.4 h GPU, ya ejecutado.
- `scripts/agent_tests/polar_fina.py` (dominio GA) / `polar_fina_dominio_c.py` (dominio C, 1..10 de grado en grado) / `polar_2grados_dom24x16.py` (dominio C, t≈20, early-stop desactivado) / `polar_naca0012.py` — polares reanudables y pausables con `STOP_SIMULATION.trigger`; cada punto cachea su JSON y vuelca campo + streamlines.
- `scripts/agent_tests/extrapolacion_polar.py` — extrapola cada punto a t→∞ con ajuste exponencial sobre T/2T/4T (Richardson **no** vale en el tiempo). Marca n/a donde no hay meseta.
- `scripts/agent_tests/figuras_analisis_polar.py` — figuras del análisis de las dos polares del dominio C → `results/analisis_polar/` (incluye la polar corregida a 1 grado, que no es simulación nueva).
- `scripts/agent_tests/ventana_larga_dx001.py` — ganador a dx=0.001 con tope de 192000 iteraciones. **16.4 h GPU, ya ejecutado y commiteado; no relanzar.**
- `scripts/agent_tests/gci_ganador.py` — GCI de Roache de tres mallas sobre Cl, Cd y L/D; reanudable, cachea cada malla → `gci_ganador.json`. ~4.3 h GPU, de las que 3.3 h son la malla fina. **Pendiente: rehacer las dos mallas gruesas con el Cl promediado.**
- `scripts/agent_tests/test_espesor_malla.py` — reevalúa a dx=0.002 ganadores de distintas épocas de una isla → `espesor_vs_malla.json`.
- `scripts/agent_tests/gran_optimizacion.sh {start|run|stop|status|snapshot|log}` — segunda gran optimización (islas `tfg2`). **Campaña terminada**; queda como plantilla. `stop` deja el centinela `STOP` y espera al checkpoint (≤12 min).
- `scripts/agent_tests/cola_tfg.sh {run [etapa]|status}` — cola serie reanudable: `calibrar → reeval → gci → polar` (+ `multipunto`, fuera por defecto). `cola_final.sh` — `repolar → validacion → revalidacion`. `cola.sh` — cola genérica de GA.
- `scripts/agent_tests/verificacion_numerica.py --calibrar|--reeval|--gci|--polar|--repolar|--validacion` — etapas sueltas; `simular()` (con `dump`/`dump_field`), `criterio_calibrado()` y `T_TARGET_CASO` viven aquí.
- `scripts/agent_tests/series_dx004.py` — 20 series completas a dx=0.004 sin parar: referencia del criterio y test de ranking entre mallas.
- `scripts/agent_tests/revalidar_ranking.py --n 30` — muestreo estratificado del historial y correlación de rangos; cacheado y reanudable.
- `scripts/agent_tests/metricas_ga.py` — post-proceso del GA sin GPU (~1 min): 15 figuras + `INFORME.md`. **Pendiente de pasar sobre `tfg2`.**
- `scripts/agent_tests/mixto.sh` — lanzador de los GA de población mixta. `video_ganador.py` — vídeo del campo. `plot_comparacion_transicion.py` — SA vs SA-BC. `validar_aislado_dx002.py`, `t1_ranking_dx.py`.
- `scripts/agent_tests/run_kutta_tests.py {smoke|smoke_polygon|baseline_a5|polygon_a5|compare|regression}` — runs coarse (dx=0.004, ~5-15 min) con criterios cuantitativos. **Punto de entrada para el diagnóstico del TE** (punto 4).
- `scripts/agent_tests/run_cfl_sweep.py` — barrido CFL/dx/α resumible (pendiente de ejecutar).
- `scripts/agent_tests/run_convergence_study.py --aislado` — test sin migración. **Estudio cerrado**; no relanzar.
- `scripts/comparativa_barrido_completo.py --smoke` — validación rápida del pipeline outer_sum vs XFoil sin gastar GPU real.
- `pytest tests/ --ignore=tests/test_generacion_geometrica_ga.py` (5 passed; el de GA está roto pre-existente: importa `RunGA` inexistente).
- `docs/tfg/build_tfg.py [--plantilla RUTA] [--salida RUTA]` — regenera `TFG.docx` desde `memoria.txt` sobre la plantilla EETAC (sin GPU).

## Diagnósticos clave (métodos de Mesh)
- `compute_circulation()` — Γ en lazos, Cl_circ=-2Γ/(U·c). **Con el ganador de `tfg2` los lazos dan 0.65 / 0.51 / 0.51 / 0.29 (0.15c → 1.50c): la circulación no cierra.**
- `compute_cp_diagnostics(mu, rho)` — Cl(∮ΔCp), ΔCp_TE, pico de succión.
- `compute_kutta_dcp_instant(face_data)` — ΔCp_TE instantáneo para el monitor.
- `debug_te_report()` — ghosts/irreparables cerca del TE (usado por `te_report_mallas.py`).
- `plot_streamlines()`, `plot_forces_over_time()`, `plot_cp_vs_chord()`.
- Series por muestra: `tvector` (tiempo físico; el dt es adaptativo y el tiempo no se reconstruye desde el índice), `clvector`/`cdvector`, `resvector` (change_u/v/p), `n_muestras_validas`.
- Del retorno de `simular_perfil`: `cl_ci95`/`cd_ci95`/`ld_ci95` y `n_samples` (**comparar cambios entre corridas contra el CI95 antes de llamarlos reales**), `converged_clcd`, `t_conv_clcd`, `iters_efectivas`, `cl_cp_discrepancy` + `cl_cp_discrepancy_flag`, **`cl_audit_inst`** (Cl instantáneo de la auditoría de superficie: diagnóstico, **no** entra en el L/D desde `08af2c4`), `dump_series_path` (.npz de t/Cl/Cd/L·D) y `dump_field_path` (.npz del campo final).
- Diagnósticos de frontera leídos del campo final (`estudio_dominio.py`): `eps_top/eps_bot` (aceleración media en la pared slip = bloqueo medido), `v_in_max` (|v| justo dentro del inflow, que el BC fuerza a 0), `u_out_min` (estela viva en la salida si ≪1), `cp_wall_range` (si no es ~0, la pared aún ve el perfil).

## Config de referencia (fina, `__main__`)
Lx=12, Ly=8, dx_min=0.001 (malla 3243×1203, ~2 it/s), CFL=0.25, NACA_0012_sharp, min_te_height_factor=1.0, turbo_hd, wake long_fine_x, `wall_treatment="consistent"`, `turb_model="sa"`, `transition_model="sa_bc"`, `advection_scheme="maccormack"`. Parada por convergencia con los defaults nuevos (`clcd_tol_drift=0.005`, `clcd_tol_ci95=0.02`, `clcd_window_conv_time=1.0` mínima y auto-ensanchable, `clcd_min_t_fisico_before_check=5.0`; el fichero calibrado baja `tol_drift` a 0.002 y sube `n_sostenido` a 3).

**Dominio**: para polares y simulaciones sueltas usar **C — Lx=24, Ly=16, cx=6**, no el 8×5 del GA. Manda Lx (estela), no Ly. **En estudios de malla o de ventana, desactivar las DOS paradas** (criterio de L/D y guardián de residual `stop_on_convergence`).

**Nota**: el bloque `__main__` puede quedar como scratch de pruebas entre sesiones — usarlo como plantilla, no copiarlo literal.

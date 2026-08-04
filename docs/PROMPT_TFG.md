# Prompt: TFG — Simulador CFD 2D en GPU y optimización genética de perfiles aerodinámicos

> Este archivo **es el prompt**, no el trabajo. Se adjunta en una sesión nueva junto con el
> repositorio del simulador para que se redacte la memoria del TFG.

---

## 0. Cómo usar este prompt

Eres mi asistente de redacción para mi Trabajo de Fin de Grado. La conversación será larga:
iremos construyendo el documento capítulo a capítulo, no de una sola vez.

**Antes de escribir una sola línea del TFG, haz esto por orden:**

1. Lee `RESUMEN.md` en la raíz del repositorio. Es la fuente de verdad del estado del proyecto.
2. Lee `docs/verificacion_numerica.md` **entero**. Es el capítulo de verificación y contiene la
   corrección que invalida buena parte de los números antiguos. Nada se cita sin haberlo leído.
3. Lee `docs/memoria_tecnica.md`. Es un borrador técnico previo: sirve como material de partida
   para los capítulos 2 y 3, pero **sus capítulos 4, 5 y 6 están desactualizados** (fueron escritos
   antes de la verificación numérica y antes del cambio a Spalart-Allmaras y MacCormack).
4. Para cualquier pregunta sobre el código, usa el grafo de conocimiento en `graphify-out/`
   (`GRAPH_REPORT.md`, `graph.json`) antes de abrir `Simulador2D.py`, que tiene ~8000 líneas.

Después, propón un plan de redacción y espera confirmación antes de empezar el capítulo 1.

**Idioma:** español. **Tono:** memoria técnica de ingeniería, primera persona del plural o
impersonal, sin marketing. Las ecuaciones en LaTeX.

---

## 1. Material que se adjunta

**Repositorio completo**: `Proyectos/Optimizador/Optimizador/`

| Ruta | Qué es |
|---|---|
| `RESUMEN.md` | Estado del proyecto, cronología de hallazgos. Fuente de verdad. |
| `docs/verificacion_numerica.md` | Capítulo de V&V ya escrito: defecto, recalibrado, GCI, XFOIL, supervivencia del ranking. |
| `docs/memoria_tecnica.md` | Borrador técnico previo (teoría y método aprovechables; resultados obsoletos). |
| `Base Teorica.txt` | Notas teóricas extendidas (925 líneas), incluye la parte de IA predictiva. |
| `Simulador2D.py` | Solver CFD completo: clase `Mesh` + `main()`, kernels CUDA. |
| `scripts/RunGA.py` | Optimizador genético + `OraculoAerodinamico` (surrogate RandomForest) + `DataLoggerML`. |
| `scripts/agent_tests/verificacion_numerica.py` | Etapas `--calibrar/--reeval/--gci/--polar/--validacion`. |
| `scripts/agent_tests/revalidar_ranking.py` | Muestreo estratificado + correlación de rangos. |
| `scripts/agent_tests/run_convergence_study.py` | Orquestador del estudio de convergencia del GA (Arm A/B, islas, aislado). |
| `scripts/agent_tests/metricas_ga.py` | Post-proceso del GA sin GPU: 15 figuras + informe. |
| `results/verificacion_numerica/` | Datos y figuras de la verificación vigente. |
| `results/1eraGranOptimizacion/` | Todo el estudio grande: `verificacion_numerica/`, `metricas_ga/`, `convergence_study/`, `comparativas/`, `diagnosticos/`, `videos_ganadores/`. |
| `graphify-out/` | Grafo de conocimiento del repositorio (1295 nodos, 73 comunidades). |
| `profiles/` | Perfiles semilla: AG24, GM15, NACA_0012, NACA_0012_sharp, s1014, E387, MH32, SD7037, SG6043. |
| `aprendizaje_ML.jsonl` | 3884 evaluaciones CFD registradas (geometría + condiciones + Cl/Cd). |
| `cerebro_aerodinamico.pkl` | Memoria persistente del surrogate. |

**Pendiente de adjuntar por mí** (déjame hueco marcado en el texto, no te lo inventes):

- Plantilla oficial del TFG (formato, portada, estilo de citas).
- `[RELLENAR]` Límite de páginas del cuerpo (introducción → conclusiones).
- `[RELLENAR]` Estilo de citación exigido (IEEE / APA / otro).
- Figuras finales renderizadas que decida incluir.

---

## 2. Resumen del proyecto

Se ha desarrollado desde cero un **simulador CFD 2D incompresible acelerado por GPU** y un
**optimizador genético de perfiles aerodinámicos** que lo usa como función de fitness, más un
**modelo sustituto de aprendizaje automático** para abaratar la búsqueda.

El problema de fondo: el diseño preliminar de perfiles necesita evaluar miles de geometrías. El
túnel de viento es caro y lento; el CFD 3D de alta fidelidad requiere cientos de horas de clúster.
La hipótesis del trabajo es que una GPU de consumo (RTX 3070 Ti) con un solver 2D bien construido
puede cerrar ese hueco para exploración de formas.

Lo que se ha construido:

- **Solver**: malla cartesiana estirada, splitting de Chorin, advección semi-Lagrangiana con
  corrección MacCormack, difusión explícita con sub-stepping, proyección de presión por multigrid
  geométrico con defect-correction, sólidos por Immersed Boundary Method con ghost cells, modelos
  de turbulencia WALE (LES) y Spalart-Allmaras (RANS), cálculo de fuerzas por integración de
  esfuerzos en superficie. Todo en CuPy con kernels CUDA propios.
- **Optimizador**: algoritmo genético sobre la parametrización camber/espesor del perfil, con
  restricciones geométricas físicas, elitismo, torneo, sigma adaptativa, modelo de islas con
  migración, y filtro predictivo RandomForest.
- **Campaña experimental**: 3884 evaluaciones CFD, 98.6 horas de GPU, 6 estudios de optimización
  a Re ∈ {1e3, 1e5, 1e6, 1e7}.
- **Verificación y validación**: se encontró un defecto propio en el criterio de parada que
  invalidaba los valores absolutos de toda la campaña; se recalibró, se midió que el recalibrado
  **no bastaba**, se acotó la incertidumbre de malla con el GCI de Roache, se contrastó contra
  XFOIL y se midió si el ranking del GA sobrevivía a la corrección.

**La historia que cuenta el TFG no es "he construido un optimizador y funciona".** Es: he
construido un optimizador, he descubierto que mi propia función de fitness no ordenaba en la zona
donde el algoritmo selecciona, lo he demostrado con estadística de rangos, y he acotado
exactamente qué afirmaciones siguen siendo defendibles. Ese es el valor del trabajo y así debe
estructurarse la narrativa.

**Consecuencia directa: la campaña de optimización hay que rehacerla, y hasta entonces sus
resultados no se escriben.** Lee la moratoria de §3.0 antes de tocar el capítulo 4.

---

## 3. Reglas de veracidad — LEER ANTES DE ESCRIBIR NADA

### 3.0 Moratoria: qué NO se escribe todavía

Hay partes del proyecto que aún no existen en estado citable. **No las redactes, ni siquiera como
borrador provisional, ni las rellenes con los datos actuales del repositorio.**

**Prohibido escribir por ahora:**

1. **Resultados de optimización.** La campaña hay que rehacerla con presupuesto fijo y sin parada
   anticipada. Nada de curvas de convergencia del fitness, mejoras porcentuales por estudio, ni
   comparativas entre estudios.
2. **El perfil ganador y sus coeficientes.** No hay ganador validado. No lo describas, no lo
   dibujes, no des su geometría (espesor relativo, camber, posiciones) ni sus C_L, C_D o C_p como
   resultado.
3. **Cualquier tabla de L/D.** Los valores actuales salen de una función de fitness que no ordenaba
   en la zona alta. Esto incluye polares con columna L/D, tablas por Reynolds, tablas por α y
   cualquier L/D suelto citado en el cuerpo del texto.

**Qué hacer en su lugar:** escribe la sección afectada hasta donde el método y la metodología lo
permitan —cómo se mide, con qué configuración, con qué incertidumbre, qué figura irá ahí— y cierra
con `[EN ESPERA: campaña por rehacer]`. Una sección con el método descrito y el hueco marcado es
útil; una con números que habrá que borrar, no.

**Qué sí se puede escribir ya:** todo el capítulo 1, todo el 2, todo el 3 (método e
implementación, incluida la descripción del GA y del surrogate como *algoritmos*), la verificación
del solver que no dependa de la optimización (simetría, DNS de referencia, orden observado del
esquema, contraste con XFOIL sobre NACA 0012), el rendimiento computacional, el capítulo 6 y la
parte de las conclusiones que se refiere al método y a la verificación.

**Caso especial — la verificación de la función de fitness.** El estudio de supervivencia del
ranking (Spearman, Kendall, solape de top-k) **sí se escribe**: no es un resultado de optimización,
es la verificación que demuestra por qué hay que rehacer la campaña. Preséntalo con las estadísticas
de rango y sin publicar los L/D absolutos de los individuos. Es el argumento que justifica la
moratoria, así que tiene que estar.

### 3.1 Reglas generales

Estas reglas no son negociables. Un número mal citado aquí invalida un capítulo entero.

1. **Ningún valor absoluto de L/D anterior al recalibrado del criterio de parada es citable.**
   Esto incluye: los 3884 registros de `aprendizaje_ML.jsonl`, todo lo explorado a dx=0.004, y las
   tablas de `docs/memoria_tecnica.md` §4. Si un número no aparece en
   `docs/verificacion_numerica.md` o en `results/verificacion_numerica/`, pregúntame antes de usarlo.
2. **No afirmes que el ranking relativo se conserva.** Está medido y es falso en la zona alta:
   Spearman ρ = 0.80 global pero ρ = 0.15 en la mitad superior, con solape top-4 de 0.25
   (`verificacion_numerica.md` §6).
3. **No presentes los ganadores del GA como óptimos.** Son óptimos de una función de fitness que
   no discriminaba a esa escala.
4. **El L/D absoluto del solver no es comparable con XFOIL ni con datos experimentales.** El Cd se
   desvía hasta +226 % frente a XFOIL, y el error crece con α, así que tampoco es un factor de
   escala corregible. Lo que sí se sostiene: el Cl, la pendiente de sustentación, y las
   comparaciones perfil-contra-perfil a igual α y misma malla.
5. **Distingue siempre las dos definiciones de Cl**: `ld` (usa `Cl_from_Cp_force_consistent`) frente
   a `ld_raw` (integral de superficie). No las mezcles en una misma tabla sin decirlo.
6. **El criterio de parada recalibrado sirve para explorar, no para reportar.** Su error de
   generalización llega al −23.7 %. Todo resultado citado en el TFG debe venir de
   `stop_on_clcd_convergence=False` con presupuesto fijo.
7. **Si no hay dato, dilo.** Escribe `[PENDIENTE: descripción de lo que falta]` en el texto en vez
   de rellenar con un valor plausible. Los huecos marcados los completo yo.
8. **Cita la fuente de cada número** en un comentario o nota al pie interna mientras redactamos
   (ruta del JSON o sección del documento). Se limpian al final.

---

## 4. Formato y restricciones

- Cuerpo del documento: **introducción → conclusiones**, límite `[RELLENAR]` páginas.
- Ecuaciones numeradas, en LaTeX.
- Figuras y tablas con pie descriptivo y referencia cruzada en el texto. No incluyas una figura que
  no se comente.
- Todas las magnitudes con unidades; los coeficientes adimensionales, declarados como tales.
- Código: solo fragmentos cortos y solo cuando el algoritmo no se entienda sin ellos. Este es un
  TFG de ingeniería aeroespacial, no un manual de la API.
- Cuando cites resultados numéricos, incluye **incertidumbre**: CI95, GCI o desviación típica según
  corresponda. Una tabla sin barras de error no entra.

---

## 5. Estructura obligatoria y contenido capítulo a capítulo

Esta estructura la ha fijado mi tutor. Puede sufrir cambios menores; no la reorganices por tu
cuenta, pero avísame si detectas que algo importante no tiene sitio en ella.

---

### 1. Introducción

#### Motivación / problema / bibliografía de motivación

Plantea el problema: el coste de evaluar geometrías en diseño aerodinámico preliminar. Tres vías
—túnel de viento, CFD 3D de alta fidelidad, métodos de paneles tipo XFOIL— y qué le falta a cada
una: coste e infraestructura, horas de clúster, y fidelidad limitada en flujo separado
respectivamente. El hueco es una herramienta que resuelva las ecuaciones completas, sea barata por
evaluación y corra en hardware accesible.

Añade el contexto de la optimización de forma aerodinámica como disciplina (métodos adjuntos,
algoritmos evolutivos, modelos sustitutos) y por qué el número de evaluaciones es el factor
limitante en todos ellos.

**Instrucción:** busca tú la bibliografía, no me la pidas. Como mínimo debe aparecer: Chorin (1968)
para el método de proyección, Stam (1999) para la advección semi-Lagrangiana estable, Nicoud &
Ducros (1999) para WALE, Spalart & Allmaras (1992), Mittal & Iaccarino (2005) para IBM, Drela
(1989) para XFOIL, Jameson para optimización por adjunto, una revisión reciente del estado del arte
en optimización de forma aerodinámica, y Roache (1994) / Celik et al. (2008) / ASME V&V 20-2009
para verificación. Añade lo que consideres necesario.

#### Objetivos del trabajo

1. Desarrollar un solver CFD 2D incompresible transitorio completo, acelerado por GPU, capaz de
   evaluar perfiles aerodinámicos en minutos sobre hardware de consumo.
2. Implementar el cálculo de fuerzas aerodinámicas por integración de esfuerzos en superficie sobre
   geometría embebida (IBM), con precisión suficiente para comparar perfiles entre sí.
3. Acoplar el solver a un algoritmo genético que optimice la geometría del perfil, con
   parametrización y restricciones físicamente motivadas.
4. Reducir el coste de la búsqueda mediante un modelo sustituto entrenado con las propias
   evaluaciones.
5. **Verificar y validar el conjunto**: cuantificar el error de discretización, contrastar contra
   referencias externas, y determinar si la función de fitness empleada era apta para seleccionar.

Deja explícito que el objetivo 5 acabó siendo el que más peso tiene en las conclusiones.

#### Alcance y limitaciones

- Flujo **2D incompresible**. Sin efectos tridimensionales (estiramiento de vórtices, vórtices de
  punta), sin compresibilidad (válido Mach < 0.3).
- Régimen **transitorio**, sin hipótesis de estacionariedad.
- Rango de Reynolds explorado: 1e3 a 1e7, con el grueso del trabajo a **Re = 1e5**.
- Punto de diseño de la optimización: **α = 4°**, un solo ángulo (el multipunto quedó descartado
  por coste, ver §5 del documento de verificación).
- Hardware: una **RTX 3070 Ti**, 8 GB. Es una limitación dura del trabajo: dx = 0.0005 no cabe en
  memoria, así que el estudio de convergencia de malla se detiene en dx = 0.001.
- **Sin validación experimental propia.** La validación es contra DNS publicado (Kurtulus, Re=1000)
  y contra XFOIL.

---

### 2. Teoría

#### 2.1 Ecuaciones de Navier-Stokes

Conservación de cantidad de movimiento y de masa para flujo incompresible viscoso. Significado
físico de cada término: advección, difusión viscosa, gradiente de presión. Adimensionalización y
número de Reynolds como único parámetro de gobierno del problema.

#### 2.2 Restricción de incompresibilidad

∇·u = 0 como conservación exacta de masa. La presión no tiene dinámica propia: actúa como
multiplicador de Lagrange que proyecta el campo de velocidades sobre el espacio solenoidal. Explica
por qué esto obliga a resolver una ecuación de Poisson en cada paso temporal, y por qué ese paso es
el que domina el coste computacional (60–80 % del tiempo de iteración).

#### 2.3 Turbulencia

Definición del régimen turbulento, cascada de energía, escalas de Kolmogorov. **La particularidad
2D importa**: cascada inversa de energía y ausencia del mecanismo de estiramiento de vórtices; dilo
explícitamente porque es una limitación de fondo del trabajo.

Clasificación DNS / RANS / LES por qué escalas resuelve cada uno y a qué coste.

#### 2.4 Modelo WALE / Turbulencia

Formulación completa de WALE (Nicoud & Ducros): tensor de gradientes, parte simétrica desviatórica,
expresión de ν_t, y sus dos ventajas sobre Smagorinsky (comportamiento correcto en pared sin
funciones de amortiguamiento, y anulación natural en flujo laminar).

**Aviso importante que hay que contar con honestidad:** WALE está implementado y documentado, pero
en la práctica **resultó inerte** a los Reynolds de trabajo — no producía viscosidad turbulenta
significativa donde hacía falta, y la configuración de referencia del optimizador acabó usando
**Spalart-Allmaras**, un modelo RANS de una ecuación. Esta sección debe por tanto cubrir los dos
modelos, con la formulación de SA (ecuación de transporte de ν̃, producción, destrucción, distancia
a pared) y explicar el criterio por el que se pasó de uno a otro. Es un resultado del trabajo, no
un cambio de opinión: está documentado en `RESUMEN.md` (diagnóstico de 4 capas, capa 1).

Menciona también el rango y+ alcanzable con la malla usada y por qué eso condiciona lo que SA puede
predecir.

#### 2.5 Aerodinámica de perfiles

- **Qué es un perfil**: cuerda, extradós, intradós, camber, espesor, borde de ataque y de salida.
- **Fuerzas aerodinámicas**: sustentación y resistencia como componentes perpendicular y paralela a
  U∞; coeficientes C_L, C_D, C_p; eficiencia aerodinámica L/D; polar del perfil.
- **Conceptos básicos**: capa límite (laminar y turbulenta, espesor, y+), gradiente adverso de
  presión, separación, burbuja de separación laminar y reataque, entrada en pérdida, condición de
  Kutta.
- **Dependencia del rendimiento con la geometría**: efecto del camber sobre C_L, del espesor sobre
  la resistencia y sobre el comportamiento en pérdida, de la posición del espesor máximo sobre el
  gradiente de presión. Efecto del Reynolds: por qué un perfil óptimo a Re=1e5 no lo es a Re=1e7.
- **Dependencia con el tipo de cálculo**: por qué el mismo perfil da distintos C_D según se evalúe
  con teoría potencial + capa límite (XFOIL), RANS o LES, y por qué la resolución de malla cerca de
  la pared cambia la resistencia de fricción. Esto prepara el terreno para el capítulo 4.
- **Cómo funciona una optimización**: espacio de diseño, función objetivo, restricciones, mínimos
  locales frente a global, exploración frente a explotación. Introduce el algoritmo genético
  (población, fitness, selección, cruce, mutación, elitismo) y el concepto de modelo sustituto.
  Menciona por qué aquí no se puede usar un método adjunto (el solver no tiene adjunto implementado
  y la geometría es embebida, no ajustada al cuerpo).

---

### 3. Método / Aplicación

Este capítulo es el núcleo de ingeniería. Usa `docs/memoria_tecnica.md` §2–§4 como base pero
**actualízalo**: aquel documento describe la configuración antigua. La configuración de referencia
vigente es `wall_treatment="consistent"` + `turb_model="sa"` + `advection_scheme="maccormack"`.

Referencia las funciones y clases por nombre (`Mesh.project_multigrid`, `compute_wale_viscosity`,
`_precomputar_ghost_cell`, …), no por número de línea: las líneas se mueven.

#### 3.1 Arquitectura general del simulador

Diagrama de bloques: `main()` → construcción de malla → carga y rasterización de geometría →
condiciones de frontera → bucle temporal (dt adaptativo → advección → difusión → proyección →
condiciones de contorno → muestreo de fuerzas) → post-proceso. Clase `Mesh` como contenedor de
dominio, campos en GPU, métricas de malla y kernels compilados. Dimensión del código y estructura
de módulos (`Simulador2D.py`, `scripts/RunGA.py`, scripts de estudio).

#### 3.2 Método de splitting de Chorin

Los tres subproblemas por paso temporal y por qué el splitting permite usar un método optimizado
distinto para cada uno. Orden de precisión temporal del splitting y su error de división. Secuencia
real del código.

#### 3.3 Discretización espacial y temporal

Malla cartesiana **estirada**: espaciado mínimo dx_min en la zona del perfil, expansión geométrica
hacia los bordes con `factor_expansion`, tope `ratio_max_malla`. Stencils de 3 puntos no uniformes
obtenidos por interpolación de Lagrange, precomputados en float64 para preservar la simetría exacta
cuando el dominio es simétrico (detalle relevante: permite validar α=0° con C_L = 0 a precisión de
máquina). Almacenamiento de campos en float32 en GPU.

Paso temporal adaptativo: mínimo entre la restricción CFL advectiva y la restricción viscosa
explícita (C_visc = 0.25), con tope de 2× el dt nominal. Explica que al ser dt adaptativo el tiempo
físico no se reconstruye desde el índice de iteración — de ahí `tvector`, añadido durante la
verificación.

Introduce aquí el **tiempo convectivo** (t·U∞/c) como unidad de referencia; se usa en todo el
capítulo 4.

#### 3.4 Advección semi-Lagrangiana

Backtracing, interpolación bilineal sobre malla no uniforme mediante `searchsorted`, estabilidad
incondicional, paralelización trivial en GPU.

**Y su problema**: difusión numérica de primer orden, ν_num ≈ u·dx/2, independiente de dt. A Re=1e3
es ~10 % de la viscosidad molecular (por eso el caso de validación DNS salía bien); a Re=1e5 es
5–10× la molecular cerca de la pared, engrosaba la capa límite un factor 3 y hundía el C_L de 0.55
a 0.32.

**Solución implementada: corrección MacCormack** (`advection_scheme="maccormack"`), variante de
Selle et al. (2008): traza hacia adelante, corrige φ* + (φ − φ**)/2, con limitador min/max del
stencil. Detalle no obvio que hay que contar: fue necesaria una **banda de 2 celdas junto al sólido
donde se mantiene semi-Lagrangiano puro**, porque sobre el contorno escalonado del IBM la
corrección introducía un pico espurio de C_p ≈ −11 en el borde de salida. Coste: ~4 % de tiempo
adicional. Resultado: C_L de 0.32 a 0.452 a dx=0.002, y 0.496 a dx=0.001 (extrapolación de
Richardson: 0.54, frente al valor físico esperado 0.55).

#### 3.5 Difusión viscosa

Forward Euler explícito con sub-stepping automático, viscosidad efectiva ν_mol + ν_t espacialmente
variable, Laplaciano con stencil no uniforme, reimposición de máscaras y condiciones IBM tras cada
sub-paso. Justifica por qué se eligió explícito frente a implícito (coste del sistema lineal frente
a coste del sub-stepping).

#### 3.6 Proyección de presión

Ecuación de Poisson para la corrección de presión y bucle de **defect-correction** externo con
criterio de convergencia sobre la divergencia media normalizada por U∞/L.

**V-cycle multigrid geométrico**: pre-suavizado Red-Black Gauss-Seidel SOR, restricción
full-weighting 1-2-1⊗1-2-1/16, resolución en nivel grueso, prolongación bilineal vertex-centered
con pesos precomputados, post-suavizado. Explica la elección de RB-SOR (permite actualización
in-place sin conflictos de escritura en GPU: cada lanzamiento toca celdas de una sola paridad).

Coarsening de la máscara de sólidos vertex-centered con stencil 3×3 OR, y por qué preserva la
simetría. Protección de residuo: si el V-cycle amplifica el residuo por encima de 1.5×, se descarta
la corrección y se cae a Gauss-Seidel puro.

Menciona la ruta alternativa por gradiente conjugado precondicionado y el diseño del par
divergencia/gradiente como operadores adjuntos exactos (garantiza que D·Dᵀ sea simétrico definido
positivo).

Modos de configuración (turbo / turbo_hd / turbo_ultra) con su compromiso velocidad-divergencia;
tabúlalos aquí y remite al capítulo 4.4 para las cifras de rendimiento.

#### 3.7 Tratamiento de sólidos

Carga del perfil en formato Selig, escalado por cuerda, rotación por α alrededor del cuarto de
cuerda, rasterización con `matplotlib.path.Path.contains_points` en float64. Recorte automático del
borde de salida cuando queda por debajo de 2·dx_min.

**Ghost-Cell IBM**: identificación de celdas sólidas adyacentes a fluido, punto imagen simétrico
respecto a la pared, interpolación bilineal y asignación u_ghost = −u_imagen para imponer no-slip.

**El hallazgo importante de esta sección** (`wall_treatment="consistent"`, ver `RESUMEN.md`): el
tratamiento original era inconsistente entre el IBM y la proyección, y absorbía masa en la pared
(Q_lazo ≈ −0.06·U·c), lo que producía una sobre-circulación de factor 2.2. La corrección tiene tres
componentes: divergencia por flujo de caras (cara fluido-sólido = flujo exactamente nulo), gradiente
de presión one-sided en celdas de pared, y eliminación del "refuerzo de impermeabilidad" que borraba
masa. Cuenta también la métrica que permitió diagnosticarlo: **Q_lazo = ∮u·n dl** sobre lazos que
rodean el perfil.

Y cuenta el desenlace honesto: llevar Q a cero **no recuperó el C_L** — la hipótesis de que la
proyección era la culpable quedó falsada, y eso es lo que llevó a buscar la cuarta capa (§3.4).

#### 3.8 Cálculo de fuerzas aerodinámicas

Extrapolación de la presión a la pared por regresión lineal por mínimos cuadrados sobre varias capas
de fluido (más robusto que interpolar un solo punto con gradientes fuertes en el borde de salida).

**Corrección de presión de fondo**: ajuste de un plano afín a la presión en el far-field y sustracción,
más eliminación del offset residual ponderado sobre el contorno. Justifícala: sin ella aparece un
sesgo espurio de sustentación cuando el contorno discreto no cierra exactamente.

Esfuerzo viscoso a partir del tensor de deformación en superficie con la viscosidad efectiva local.
Integración sobre la frontera del sólido, transformación al sistema aerodinámico por rotación de α,
adimensionalización.

**Los tres estimadores de C_L y por qué existen tres**: integral de superficie (`ld_raw`), integral
de ∮ΔC_p (`cl_cp`), y circulación Γ en lazos (`cl_circ`). Su discrepancia mutua es el diagnóstico
de calidad más útil del solver: cayó de 0.31–0.83 a 0.02–0.14 al corregir el criterio de parada, lo
que reveló que buena parte de la supuesta sobre-circulación era artefacto del transitorio.

#### 3.9 Implementación en GPU

Tipos de kernel: `ElementwiseKernel` de CuPy frente a `RawKernel` en CUDA C, y por qué cada uno.
Enumera los kernels propios (`rb_gs_sor`, `restrict_2d`, `prolongate_add_2d`, `divergence_masked`,
`ghost_cell_bc_kernel`, `velocity_correction_kernel`, `adjoint_gradient_kernel`, `zero_solid_kernel`,
spreading de Peskin con `atomicAdd`).

Estrategia de minimización de transferencias CPU-GPU: solo escalares (C_L, C_D, divergencia) cada
`guardado` iteraciones; métricas de malla transferidas una vez. Warmup JIT antes del bucle principal
para no pagar 30–90 s de compilación en la primera iteración.

**El cuello de botella es el ancho de banda de memoria, no la capacidad aritmética**: los accesos a
vecinos en y saltan nx elementos y no coalescen. Justifica por qué eso limita el speedup frente a
CPU y por qué GPUs con más bandwidth escalarían mejor.

Sistema de memoria compartida GPU-CPU para el visor en tiempo real (`viewer_live.py`) y la GUI.

#### 3.10 Optimización geométrica

##### 3.10.1 Algoritmo genético

Ciclo evolutivo: evaluación de fitness, elitismo (4 de 12), selección por torneo (tamaño 4), cruce
por promediado de coordenadas, mutación paramétrica. Sigma adaptativa (exploración → refinamiento).
Parada por estancamiento (`paciencia_generaciones`, `tol_mejora_fitness`).

**Modelo de islas con migración**: varias poblaciones con semillas distintas que intercambian
individuos por épocas. Explica el propósito (mantener diversidad, escapar de óptimos locales) y
adelanta que el capítulo 4 mide si funciona.

Checkpointing de genes (`estado_ga.json`) para poder pausar y reanudar corridas de decenas de horas.

##### 3.10.2 Parametrización del perfil

Coordenadas Selig, descomposición en camber c(x) = (y_up + y_low)/2 y espesor t(x) = y_up − y_low.
Justifica por qué se muta en ese espacio y no en las coordenadas directas: son las funciones con
significado aerodinámico, y perturbarlas por separado evita generar geometrías cruzadas.

Perturbación suave por nodos de control (7 nodos), con las desviaciones típicas calibradas por
barrido (`camber_mut_std = 0.0035`, `espesor_mut_std = 0.0045`).

**Restricciones geométricas** y su motivación física: radio mínimo de borde de ataque, espesor
mínimo absoluto y relativo en el borde de salida, espesor mínimo global, atenuación de la mutación
en la zona del borde de ataque. Mecanismo de reintentos (hasta 120) y fallback.

Resamplado a rejilla común (`resamplear_a_grid`) para poder cruzar perfiles con distinto número de
puntos.

##### 3.10.3 Evaluación CFD como fitness

Cómo se lanza cada evaluación (`simular_perfil` → `Simulador2D.main()`), qué configuración usa, y
cómo se promedia el L/D. Aquí es **imprescindible** explicar el criterio de parada por convergencia
de C_L/C_D, porque es lo que falla: describe el mecanismo tal cual está en
`verificacion_numerica.md` §1.1, y remite al capítulo 4 para su análisis.

Coste por evaluación: ~91 s a dx=0.004 con el criterio original; ~460 s a dx=0.002; ~1740 s con
presupuesto fijo honesto a dx=0.002. Ese factor 19× es el que condiciona todo el diseño experimental.

Modo multipunto (α − δ, α, α + δ) con combinación mean/min/weighted: implementado, descartado por
coste.

##### 3.10.4 Filtro predictivo mediante IA

`OraculoAerodinamico`: RandomForestRegressor (100 árboles, profundidad 15) que predice el fitness a
partir de la geometría. Umbral dinámico (percentil sobre el máximo histórico), persistencia en
`cerebro_aerodinamico.pkl`, y `DataLoggerML` que registra toda evaluación en JSONL —incluidas las
descartadas— para no sesgar el entrenamiento futuro.

Describe también el resto del sistema de IA documentado en `Base Teorica.txt`: el surrogate directo
(geometría → C_L, C_D), el predictor inverso (condiciones + objetivos → geometría) y el problema
one-to-many que este último plantea.

**Y di la verdad sobre su uso**: `usar_ia=False` en los 6 estudios de la campaña. El filtro nunca
llegó a cribar nada. Su evaluación es por tanto retrospectiva y offline (capítulo 4.5).

---

### 4. Resultados

Todos los números de este capítulo salen de `docs/verificacion_numerica.md`,
`results/verificacion_numerica/` y `results/1eraGranOptimizacion/`. Ninguno de
`docs/memoria_tecnica.md` §4.

#### 4.1 Validación del solver

Estructúralo como una cadena de verificación, en este orden:

1. **Simetría**: α = 0° con dominio simétrico da C_L ≈ 3e-4 sin NaN. Verifica la cadena completa
   (métricas de malla, rasterización, multigrid) a precisión de máquina.
2. **Contraste con DNS publicado**: Re = 1000, α = 5°, C_L = 0.211 ± 0.0004 y C_D = 0.148 frente a
   Kurtulus (C_L ≈ 0.22, C_D ≈ 0.146). Es el caso donde la difusión numérica del semi-Lagrangiano
   es despreciable, y ahí el solver acierta.
3. **Divergencia residual** como métrica de calidad de la proyección, con sus umbrales.
4. **Convergencia de malla — GCI de Roache**: método (orden observado, extrapolación de Richardson,
   factor de seguridad 1.25), datos crudos de las cuatro mallas (dx = 0.008 / 0.004 / 0.002 / 0.001),
   y resultados. La cifra a defender es **p(C_L) = 2.18**, que coincide con el orden formal 2 del
   esquema, y **GCI(C_L) = ±1.43 %** en la malla fina.
   **Lectura crítica obligatoria**: C_D **no converge monótonamente** (0.03332 → 0.02877 → 0.02927),
   así que su GCI de ±0.27 % está sobrevalorado. Los órdenes observados por encima del orden formal
   son otra señal de que ese triplete no está plenamente en el rango asintótico. Y la consecuencia
   que arrastra todo el trabajo: **dx = 0.004 queda fuera del rango asintótico, y el GA exploró ahí**.
   **Restricción de la moratoria (§3.0)**: este estudio se corrió sobre una geometría procedente de
   la campaña antigua. Preséntala como "el perfil empleado en el estudio de convergencia", **sin
   identificarla como ganador ni presentar su rendimiento**, y da la tabla en C_L y C_D. **Sin
   columna de L/D.** El resultado que interesa aquí es una propiedad del esquema numérico, no del
   perfil.
5. **Validación cruzada por tres vías independientes** (serie larga sin parada, GCI a dx=0.002, y
   reevaluación con el criterio calibrado): describe el diseño del contraste y por qué tres rutas
   independientes que coinciden acotan el error mejor que cualquiera de ellas por separado. Las tres
   cifras son L/D del perfil ganador antiguo → `[EN ESPERA: campaña por rehacer]`. Si quieres cerrar
   la sección con números, reexprésalo en C_L y avísame antes.
6. **Contraste con XFOIL** (NACA 0012, Re = 1e5, Ncrit = 9): tabla completa de C_L y C_D. La
   pendiente de sustentación es el resultado sólido — 0.10211 /° frente a 0.11107 de XFOIL y 0.10966
   de la teoría de perfil delgado (2π), un 8 % por debajo, que es lo normal para espesor finito.
   El C_D no: se desvía de forma **monótona creciente con α**, de −2.4 % a α=0 hasta +226 % a α=10.
   Discute las tres hipótesis (resolución de capa límite, separación prematura, bidimensionalidad) y
   por qué un sesgo creciente con la carga apunta a mecanismo físico y no a calibración.

Cierra la sección con el enunciado exacto de lo que queda validado: **el solver sirve para comparar
perfiles entre sí a igual α y misma malla; no para predecir coeficientes absolutos.**

#### 4.2 Resultados fluidodinámicos

Campos de velocidad, presión y vorticidad. Estructura del flujo alrededor del perfil: punto de
remanso, aceleración en extradós, pico de succión, recuperación de presión, estela. Evolución del
transitorio en tiempos convectivos, desde el arranque impulsivo hasta el estacionario.

Casos con desprendimiento: a Re=1e3 el flujo desprende y los coeficientes oscilan; a Re≥1e5 con SA
el flujo se estaciona. Muestra la burbuja de separación laminar y su reataque, y la posición del
pico de succión (x/c ≈ 0.015–0.02 en la configuración correcta, frente a 0.14 cuando el modelo
turbulento fallaba).

Material disponible: `results/1eraGranOptimizacion/diagnosticos/` (casos NACA 0012, es la fuente
preferente) y `results/1eraGranOptimizacion/videos_ganadores/` (800 frames por caso, campos de
velocidad en dominio completo y en zona refinada, Re=1e3 y Re=1e5).

**Restricción de la moratoria (§3.0)**: prioriza los casos NACA 0012. Los frames de
`videos_ganadores/` solo pueden usarse como ilustración de fenomenología del flujo, **sin
identificar la geometría como perfil ganador y sin acompañarlos de coeficientes**. Si una figura
necesita rotularse con el perfil que la generó, usa un caso de referencia en su lugar.

#### 4.3 Resultados aerodinámicos

**Restricción de la moratoria (§3.0): esta sección se limita al perfil de referencia NACA 0012 y no
lleva tabla de L/D.** Nada de polares del ganador ni de comparativas ganador-frente-a-semilla.

Lo que sí se escribe ahora:

- Curva C_L(α) del NACA 0012 a Re = 1e5 y dx = 0.002, con su pendiente de sustentación y el
  contraste frente a XFOIL y a la teoría de perfil delgado (§4.1). Es el resultado aerodinámico
  sólido del trabajo.
- Curva C_D(α) del mismo caso, **presentada explícitamente como magnitud no validada**, con la
  discusión de su desviación creciente con α.
- Distribuciones de C_p en extradós e intradós para varios α, y su relación con la curva de
  sustentación: punto de remanso, pico de succión, recuperación de presión, y cómo se degrada al
  aumentar α.
- Sublinealidad a α altos y su interpretación física (pre-stall, separación creciente en el borde
  de salida).

`[EN ESPERA: campaña por rehacer]` — polar del perfil optimizado, comparación con la semilla, y la
comprobación de si la mejora es de banda ancha o está sobreajustada al ángulo de diseño. Deja el
hueco planteado como pregunta abierta y la figura prevista descrita, para rellenarlo cuando exista
la nueva campaña.

#### 4.4 Rendimiento computacional

- Iteraciones por segundo por modo de multigrid (turbo ~4.9 it/s, turbo_hd ~2.1, turbo_ultra ~1.4)
  y por resolución de malla (dx=0.001, malla 3243×1203, ~2 it/s).
- Reparto del tiempo por etapa: proyección 60–80 %, advección segunda, difusión la más barata,
  WALE ~10–15 % de overhead, MacCormack ~4 %.
- Coste por evaluación CFD según malla y criterio de parada, y el factor 19× entre la evaluación
  barata y la honesta.
- Coste total de la campaña: **3884 evaluaciones, 98.6 horas de GPU**, desglosado por estudio
  (tabla en `results/1eraGranOptimizacion/metricas_ga/INFORME.md`).
- Análisis de escalabilidad: por qué el cuello de botella es el ancho de banda, y qué implicaría
  correr en una GPU de mayor bandwidth o en varias.

#### 4.5 Resultados de optimización

**Esta es la sección más afectada por la moratoria (§3.0).** El grueso de lo que iría aquí —qué
perfil ganó, cuánto mejoró, con qué coeficientes— no se escribe todavía. Lo que sí se escribe es la
parte que explica **por qué** no se escribe: la verificación de la función de fitness. Esa parte no
es un resultado de optimización, es el resultado de verificación que obliga a rehacer la campaña, y
sin ella el hueco del capítulo no se entiende.

Orden de la sección:

1. **Diseño experimental de la campaña**, sin resultados: seis estudios a Re ∈ {1e3, 1e5, 1e6, 1e7},
   población, generaciones, operadores, presupuesto y coste (3884 evaluaciones, 98.6 h de GPU). Es
   descripción de lo hecho, no de lo obtenido, y sirve de referencia para dimensionar la campaña
   nueva.
2. **Estructura del espacio de diseño**: PCA sobre 3884 muestras de 256 coordenadas — PC1+PC2
   explican el **75.5 %** de la varianza geométrica. La dimensionalidad efectiva del problema es
   mucho menor que la nominal, y eso justifica que un GA con población pequeña pueda funcionar.
   Es una propiedad de la parametrización, no un resultado de rendimiento, así que entra. Advierte
   igualmente de que la muestra procede de la campaña antigua y habrá que recalcularlo.
3. **¿Converge el GA a un óptimo único?** Cuatro semillas (AG24, GM15, NACA_0012_sharp, s1014)
   corridas **solas, sin migración**, hasta estancamiento (7–25 generaciones, ~28 h de GPU):
   **acaban en geometrías distintas** — distancia de forma media 0.0557, igual o peor que la de
   partida (0.0507). Conclusión: **la convergencia que producen las islas la fuerza la migración, no
   una física de óptimo único**; hay varias cuencas locales.
   Escríbelo en términos de **distancia de forma**, sin dar los L/D de cada semilla, y añade el
   caveat honesto: el experimento usó la misma función de fitness defectuosa, así que la conclusión
   es sólida sobre la *dinámica del algoritmo* pero debe reconfirmarse con la campaña nueva.
4. **¿Seleccionó bien el GA?** El resultado central del trabajo, y el que justifica la moratoria.
   Muestreo estratificado por deciles del historial (n=20), re-simulado sin parada anticipada y con
   presupuesto fijo, comparado con el orden que vio el GA:

   - Spearman ρ = 0.7955 global, Kendall τ = 0.6421.
   - Separando por la mediana: ρ = 0.612 en la mitad inferior, **ρ = 0.152 en la mitad superior**.
   - Solape del top-k: 1/3 con k=3, **1/4 con k=4**, y no se recupera hasta que k deja de ser
     selectivo.
   - Sesgo mediano del fitness antiguo: **+65 %** (IQR 47–104), con dispersión de −10 % a +145 %.

   Preséntalo **con estadística de rangos y magnitudes de sesgo, sin publicar los L/D absolutos de
   los individuos** (§3.0). Los dos casos ilustrativos se cuentan por su posición, no por su valor:
   el mejor individuo real de la muestra era de **generación 1** y el GA lo rankeó octavo; otro
   individuo pasó del puesto 3 al 13 porque la parada anticipada le regalaba resistencia baja.

   Explica por qué el sesgo es sistemáticamente positivo (doble infraestimación: malla fuera del
   rango asintótico y transitorio) pero **no es un factor de escala común**, y por eso reordena.

   Enuncia la conclusión en sus dos mitades, sin suavizarla: *a favor*, la búsqueda sí se movió
   hacia una región de rendimiento real alto (4 de 5 individuos de generación ≥17 revalidan en la
   zona alta; correlación generación–rendimiento revalidado ρ = 0.41); *en contra*, el orden dentro
   de esa región no era fiable. El GA encontró el barrio correcto con una señal que distingue
   órdenes de magnitud, y eligió la casa con una señal que no distingue nada a esa escala.

   Cierra enlazando con la moratoria: por eso los resultados de optimización de este trabajo están
   pendientes de una campaña nueva, y por eso esa campaña necesita presupuesto fijo.

5. **Evaluación retrospectiva del surrogate**: `usar_ia=False` en los 6 estudios, el filtro nunca
   cribó nada. Reentrenado offline sobre 2793 muestras a Re=1e5: R² = 0.850 (validación cruzada
   5-fold), MAE = 1.07 en unidades de L/D. Punto de operación viable: umbral en el percentil 70
   evitaría el **69 % del CFD** perdiendo solo el **2.5 % de la élite** — unas 68 horas de GPU
   desperdiciadas por no haberlo activado.
   Son métricas de calidad del modelo, no rendimiento de ningún perfil, así que entran. Pero
   adviértelo: **está entrenado sobre las etiquetas defectuosas**, luego hay que reentrenarlo con la
   campaña nueva y estas cifras son una cota optimista de lo que puede esperarse.

6. `[EN ESPERA: campaña por rehacer]` — resultados de optimización propiamente dichos: convergencia
   del fitness, mejora conseguida, perfil resultante, su geometría y sus coeficientes, y el óptimo
   geométrico por Reynolds. Deja el esqueleto de la sección y las figuras previstas descritas
   (`results/.../metricas_ga/` tiene los scripts que las regeneran), sin datos.

---

### 5. Discusión

#### 5.1 Limitaciones físicas

Bidimensionalidad: sin estiramiento de vórtices, sin efectos de borde marginal, y a Re=1e5 el
desprendimiento real es tridimensional — justo donde el solver y XFOIL divergen. Incompresibilidad
y su límite de validez. Limitaciones del modelo de turbulencia: SA es RANS y fully-turbulent desde
el borde de ataque salvo ajuste, mientras que el flujo real a Re=1e5 es laminar sobre buena parte
de la cuerda; se probó retrasar la transición (`sa_nu_tilde_factor = 0.1`) y **empeoró** —
reapareció la burbuja laminar. WALE, inerte a estos Reynolds.

#### 5.2 Limitaciones numéricas

Difusión numérica del semi-Lagrangiano y en qué medida la resuelve MacCormack (no del todo: queda
la banda de 2 celdas junto al sólido en SL puro, justo donde más importa). Resolución de capa límite:
y+ ≈ 10 con dx=0.002 cuando SA quiere y+ ≲ 1, sin función de pared. Escalonado del IBM sobre malla
cartesiana. Error de discretización acotado: ±1.4 % en C_L a dx=0.001, ~2.4 % a dx=0.002, y **fuera
de rango asintótico a dx=0.004**.

**Y la limitación metodológica, que es la más importante**: el criterio de parada
`_detect_series_convergence` compara la señal contra la media de su propio último 15 %, es decir
pregunta si la señal se parece a su pasado reciente, no si ha dejado de cambiar. Una deriva lenta lo
satisface trivialmente. El recalibrado sube el umbral de disparo pero **no cambia la pregunta que
hace el test**, y por eso su error de generalización llega al −23.7 %. Ajustar tolerancias no
arregla un estimador mal planteado. Propón cómo debería plantearse (test sobre la pendiente de la
serie, o presupuesto fijo en tiempos convectivos con `T_TARGET` por régimen).

#### 5.3 Coste computacional

El compromiso precisión-velocidad está dominado por la proyección. Coste de la honestidad numérica:
una evaluación fiable cuesta 19× una barata, y eso hace inviable repetir la campaña a la misma
escala (3884 × 1740 s ≈ 1878 h de GPU). Discute las salidas: reducir población y generaciones, usar
una malla intermedia validada para *ranking* aunque no para valor absoluto, o activar el surrogate.
Cuantifica cada opción.

#### 5.4 Calidad de la optimización

Sensibilidad a los parámetros evolutivos; riesgo de convergencia prematura con población de 12 y
elitismo del 33 %; la migración como mecanismo de consenso artificial más que como descubrimiento de
un óptimo global; el sesgo que introduciría el surrogate si se activase sin control (descarta
geometrías alejadas de su dominio de entrenamiento).

Y por encima de todo: **la calidad de una optimización está acotada por la calidad de su función de
fitness**, y aquí la fitness no discriminaba en la zona alta. Cualquier discusión sobre operadores
genéticos es secundaria frente a eso.

Esta sección se escribe entera ahora: es discusión metodológica y no necesita resultados. Mantenla
en ese plano — **sin citar rendimiento de ningún perfil** (§3.0).

#### 5.5 Posibles mejoras

Ordénalas por relación beneficio/coste, no como lista de deseos:

- Rediseñar el criterio de parada (barato, y es el que más error elimina).
- Activar el surrogate con umbral calibrado (−69 % de coste CFD).
- Esquemas de advección de mayor orden en la banda junto al sólido (WENO/ENO), o IBM ajustado al
  cuerpo.
- Función de pared para SA, o refinamiento anisótropo en la capa límite para bajar y+.
- Malla adaptativa (AMR) guiada por estimadores de error.
- Optimización multiobjetivo (NSGA-II) y multipunto en α.
- Extensión a 3D reutilizando el núcleo GPU.
- Surrogate más potente (redes neuronales sobre grafo, PINN) sobre el dataset acumulado.
- Validación experimental en túnel de viento.

---

### 6. Análisis medioambiental

No hay material previo en el repositorio para este capítulo: **hay que construirlo**. Calcúlalo, no
lo cualifiques.

1. **Huella del propio trabajo**: 98.6 horas de GPU sobre una RTX 3070 Ti (TDP 290 W) más el resto
   del sistema. Estima el consumo total en kWh y conviértelo a kg de CO₂ equivalente con el factor
   de emisión del mix eléctrico español del año en curso — **búscalo tú** (Red Eléctrica publica el
   dato). Da también la incertidumbre del cálculo. El orden de magnitud esperado son unas pocas
   decenas de kWh; verifica.
2. **Comparación con las alternativas**: cuánto habría costado energéticamente la misma exploración
   en un túnel de viento (potencia del ventilador × horas de ensayo × número de geometrías) y en un
   clúster HPC con RANS 3D. Usa referencias publicadas para las estimaciones y declara las hipótesis.
   Este es el argumento fuerte del capítulo: el enfoque propuesto es entre dos y tres órdenes de
   magnitud más barato energéticamente por geometría evaluada.
3. **El coste de los errores**: las 68 horas de GPU que el surrogate habría ahorrado, y el coste de
   la campaña cuyos valores absolutos hubo que descartar. La eficiencia metodológica es una variable
   ambiental, no solo económica.
4. **Impacto potencial de la aplicación**: mejorar la eficiencia aerodinámica de un perfil se traduce
   en menor consumo. Cuantifícalo con una estimación honesta y acotada (por ejemplo, la relación
   entre L/D y consumo en la ecuación de Breguet para un caso de referencia), dejando claro que el
   trabajo no llega a demostrar una mejora aplicable — es potencial, no medida.
5. **Ciclo de vida del hardware**: menciona brevemente que la huella incorporada de fabricación de
   una GPU de consumo no es despreciable frente a su consumo operativo en un uso puntual como este.

Cierra con una valoración equilibrada: el trabajo tiene una huella modesta y su contribución
ambiental es indirecta y potencial, no demostrada.

---

### 7. Conclusiones

#### Resumen del trabajo realizado / sumario de lo conseguido frente a los objetivos

Recorre los cinco objetivos del capítulo 1 uno a uno y di, sin adornos, en qué grado se ha cumplido
cada uno:

- Objetivos 1 y 2 (solver y cálculo de fuerzas): **cumplidos y verificados**.
- Objetivos 3 y 4 (acoplamiento con el GA y modelo sustituto): **construidos y funcionales, con
  resultados pendientes de una campaña nueva**. Di exactamente eso, sin adornarlo ni ocultarlo.
- Objetivo 5 (V&V): **cumplido**, y su resultado es el que obliga a rehacer la campaña de los
  objetivos 3 y 4.

Esa es la conclusión honesta, y encaja con la moratoria de §3.0: el trabajo entrega herramienta
verificada más el diagnóstico de por qué los resultados anteriores no son publicables, no una
optimización cerrada.

Enuncia con precisión qué queda validado: **el solver sirve para comparar perfiles entre sí a igual
ángulo de ataque y misma malla, con una incertidumbre de discretización acotada en ±1.4 % en C_L;
no sirve para predecir coeficientes absolutos comparables con la literatura.** Es una afirmación más
estrecha que la que se hacía antes del capítulo de verificación, y es la defendible.

#### Principales aportaciones

- Solver CFD 2D transitorio completo acelerado por GPU con kernels CUDA propios, funcionando en
  hardware de consumo.
- Multigrid geométrico con defect-correction sobre geometría embebida, con preservación de simetría
  y protección automática contra divergencia del V-cycle.
- Diagnóstico y corrección de una cadena de cuatro defectos acoplados (burbuja laminar sin modelo
  turbulento → pared sub-resuelta → inconsistencia IBM-proyección → difusión numérica del
  semi-Lagrangiano), con la métrica Q_lazo como herramienta de diagnóstico. Cada capa se falsó o se
  confirmó con un experimento numérico, incluidas las hipótesis que resultaron falsas.
- Acoplamiento CFD + algoritmo genético con parametrización camber/espesor y restricciones
  geométricas físicas.
- **Un estudio de verificación y validación que encuentra un defecto propio, lo cuantifica, lo
  corrige, mide que la corrección no basta, y determina estadísticamente qué conclusiones
  sobreviven.** Esta es la aportación principal, y hay que decirlo.
- Evidencia experimental de que, en este problema, un GA sin migración no converge a un óptimo único
  — la convergencia observada en el modelo de islas es un artefacto del flujo de genes.
- Dataset de 3884 evaluaciones CFD (geometría, condiciones, coeficientes) e infraestructura de
  registro reutilizable para entrenar modelos sustitutos. Declara que **las etiquetas de este
  dataset están sesgadas** por el criterio de parada y que su valor actual es la geometría y el
  pipeline, no los coeficientes.

#### Trabajo a futuro

Lo de §5.5, priorizado, más explícitamente:

1. Relanzar la optimización con presupuesto fijo y sin parada anticipada, con el presupuesto
   rediseñado — es la consecuencia directa del capítulo 4.5 y la línea más importante.
2. Diagnosticar el exceso de C_D: perfil de capa límite en pared a dx=0.001 frente a dx=0.002 y
   posición del reataque frente a lo que predice XFOIL, para decidir entre las hipótesis 1 y 2 de
   §4.1.
3. GA multipunto en α, ya implementado y descartado solo por coste.
4. Validación experimental.

---

## 6. Bibliografía mínima que debes localizar y citar

No me pidas las referencias: búscalas. Como suelo mínimo:

- **Método numérico**: Chorin (1968), Stam (1999), Selle et al. (2008) para MacCormack,
  Brandt / Trottenberg para multigrid, Roache (1994), Celik et al. (2008), ASME V&V 20-2009.
- **Turbulencia**: Nicoud & Ducros (1999), Spalart & Allmaras (1992), Pope (*Turbulent Flows*).
- **IBM**: Peskin (2002), Mittal & Iaccarino (2005).
- **Aerodinámica**: Anderson (*Fundamentals of Aerodynamics*), Katz & Plotkin, Abbott & von Doenhoff
  (*Theory of Wing Sections*), Drela (1989) para XFOIL, Selig para perfiles a bajo Reynolds.
- **Referencia de validación**: Kurtulus, sobre el flujo alrededor del NACA 0012 a Re = 1000 (es la
  referencia contra la que valida el solver — localiza la cita exacta).
- **Optimización**: Jameson (métodos adjuntos), Goldberg o Eiben & Smith (algoritmos evolutivos),
  una revisión reciente de optimización de forma aerodinámica, Forrester et al. sobre modelos
  sustitutos.
- **GPU / computación**: documentación de CuPy, y alguna referencia sobre CFD en GPU.

---

## 7. Notas para ti (Claude) cuando se use este prompt

- **No empieces a escribir hasta haber leído** `RESUMEN.md`, `docs/verificacion_numerica.md` y
  `docs/memoria_tecnica.md`. Luego propón plan y espera confirmación.
- **Respeta la moratoria de §3.0 sin excepciones.** Antes de dar por cerrada cualquier sección,
  releela buscando: tablas o valores de L/D, coeficientes de un perfil optimizado, descripción del
  perfil ganador, y cifras de mejora del GA. Si aparece alguno, quítalo y sustitúyelo por
  `[EN ESPERA: campaña por rehacer]`. Si crees que un dato concreto debería ser excepción, pregunta
  antes de escribirlo.
- **Trabaja capítulo a capítulo.** Escribe uno, lo reviso, seguimos. No generes el documento entero
  de golpe.
- **Busca tú la información bibliográfica e histórica** en lugar de pedírmela.
- **Haz tú los cálculos** que el texto necesite (huella de carbono, estimaciones energéticas,
  conversión de unidades, verificación de tablas). No me dejes huecos aritméticos.
- **Verifica las cifras contra los ficheros de resultados** antes de escribirlas. Si una cifra
  aparece distinta en dos sitios, dímelo en lugar de elegir una.
- **Marca los huecos con `[PENDIENTE: ...]`** cuando falte material que tengo que aportar yo
  (figuras, plantilla, datos experimentales). No los rellenes con valores inventados.
- **Regla de oro de este TFG**: cuando un resultado sea peor de lo que me gustaría, escríbelo tal
  cual y explica por qué es un hallazgo. El capítulo de verificación vale más que una tabla de
  resultados sin barras de error, y la secuencia defecto → diagnóstico → recalibrado → validación
  cruzada por tres vías independientes es exactamente lo que se espera de un trabajo de V&V.
- Si detectas una contradicción entre lo que dice este prompt y lo que dicen los ficheros del
  repositorio, **manda el fichero** y avísame.

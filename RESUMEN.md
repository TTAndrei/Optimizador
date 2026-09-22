# RESUMEN — Optimizador CFD 2D

## ESTRUCTURA DEL REPO (reorganizado 2026-09-15)

El repo tiene ahora **dos solvers** y está partido en consecuencia. Mapa completo en `README.md`.

```
profiles/          perfiles .dat, COMPARTIDO por los dos
docs/              memoria y documentación del proyecto
curvo/  tests/     solver curvilíneo (malla C adaptada al cuerpo), F0–F4 cerradas
gui/               interfaz PyQt6 del curvo: caso, preview de malla, runner, visor
docs/figuras/      figuras de documentación y el script que las genera
validacion/        campañas de validación del curvo, una carpeta por caso
                   (v1_naca0012_re1e5_a5, v2_dominio_24x12, v3_ymas1_dn8e5):
                   cada una con correr.py, analizar.py, caso.json, malla.npz,
                   historia.npz, resultados.json, reparto.json y su README.md
Sim_Cartesiano/    el solver congelado y TODO lo suyo: Simulador2D.py, scripts/,
                   gui/, tests/, configs/, data/, results/, resultados_finales/,
                   plots/, figuras_memoria/, media/, lanzar_*.sh
```

- El cartesiano **se ejecuta desde `Sim_Cartesiano/`** (`cd Sim_Cartesiano && ../.venv/bin/python ...`):
  sus scripts escriben con rutas relativas a esa carpeta. Sus tests también pasan
  desde la raíz con `pytest Sim_Cartesiano/tests/`.
- `Sim_Cartesiano/profiles` es un enlace simbólico a `../profiles`, para que las
  rutas `ROOT/profiles/...` de su código sigan resolviendo **sin tocar ni una línea**.
- Los `lanzar_*.sh` son lo único suyo que cambió: `.venv/bin/python` → `../.venv/bin/python`.
- El curvo se ejecuta **desde la raíz y con `PYTHONPATH=.`**:
  `PYTHONPATH=. .venv/bin/python validacion/v1_naca0012_re1e5_a5/correr.py <carpeta_destino>`.
  Sin el `PYTHONPATH` el lanzamiento en segundo plano falla por no encontrar `curvo`.

## SOLVER CURVILÍNEO `curvo/` — F0–F5 cerradas, optimización cerrada ×1.99 (2026-09-18)

Migración a malla body-fitted, módulo aparte; `Simulador2D.py` queda congelado.
Plan completo en `~/.claude/plans/swift-soaring-seahorse.md`.

| fase | qué | estado |
|---|---|---|
| F0 | `malla.py` — generador C por marcha hiperbólica de Steger-Chaussee | ✅ 0.02 s/malla |
| F1 | `metrica.py` — métricas por vértices, numpy y cupy | ✅ `div(u∞)` < 1e-15 |
| F2 | `operadores.py` — divergencia, Green-Gauss, laplaciano de métrica completa | ✅ orden 1.99 |
| F3 | `conveccion.py`, `multigrid.py` — convección-difusión implícita conservativa + ACM | ✅ orden 2.02, conservación 1.3e-15 |
| F4 | `proyeccion.py` — proyección de presión, Poisson compacto + PCG | ✅ ver abajo |
| GPU | kernels CUDA + CuPy en todo el solver | ✅ **×15.9**, ver abajo |
| F5 | `solver.py`, `fuerzas.py`, `turbulencia.py` — NS completo, fuerzas y SA a Re=1e5 | ✅ los 5 criterios pasan (ver tabla) |
| F6 | etapa y⁺≈1, validación externa | y⁺<1 (v3) y dominio (v2) cerrados; **cilindro y transición pendientes** |
| opt 1 | reducir las 11 resoluciones de multigrid por paso | ⏳ pasos 1–4 de 5 hechos (`8e0e112`, `6a0f886`, `b2c9fe0`, `1918c80`), **167 → ~116 ms (−30 %)**; queda el paso 5, que es solo confirmación |
| opt 2 | latencia de lanzamiento: nivel grueso + fusión de kernels `u`/`v` | ✅ cerrada (`ecb8dc0`, `6e3aa65`, `63e1d13`), **115.64 → 85.79 ms** |
| opt 3 | el suavizador: sincronizaciones, líneas η y reducción cíclica paralela | ✅ **86.8 → 54.7 ms/paso, 11.5 → 18.3 it/s** (×1.59); confirmado a 8 000 pasos en `validacion/v5_suavizador_pcr` |

**F4 en una línea**: el operador del multigrid **es** `D·G` (2e-16, no 1e-6),
la divergencia residual en modo par-impar baja del 46.2 % del cartesiano a 1e-9,
y el factor de convergencia queda en 0.03–0.07 con PCG precondicionado por el
ciclo V. Sobre la malla C a α = 0 la presión sale simétrica a 3e-14 → **Cl = 0
exacto**, lo que el escalonado del IBM nunca dio.

Tres hallazgos que costaron: el PCG solo funciona si el ciclo V es simétrico
(`w = 2` fijo y post-suavizado invertido; con el peso de Rayleigh el
precondicionador no es lineal y el PCG empeora a 0.97); la línea η del suavizador
tiene que **cruzar el corte de estela** (si no, factor 0.64 en vez de 0.10); y el
término cruzado del corte hay que antisimetrizarlo o no cierra el balance (0.4 %).

**GPU**: los módulos trabajan con el `xp` del array que reciben, así que la misma
función corre en numpy y en cupy. **1.87 → 0.119 s por paso de tiempo** sobre la
malla de 21 065 celdas. Dos cosas que costaron encontrar:

- CuPy elemento a elemento salió **más lento que la CPU** (0.63 vs 0.43 s): con
  21 000 celdas un ciclo V lanza ~1700 kernels diminutos y está limitado por
  lanzamiento. Lo que gana es **fundir** cada barrido, la aglomeración y las
  transferencias entre niveles en un kernel cada uno (~112 lanzamientos por ciclo).
- **float32 es obligatorio**: la 3070 Ti hace fp64 a 1/64 de fp32. En float64 la
  GPU solo gana ×4.

```bash
.venv/bin/python -m pytest tests/ -q          # 171 tests, 6–7 min (15 piden GPU)
```

**170/171 pasan.** El que falla es `test_curvo_gpu.py::test_float32_da_la_misma_respuesta`,
`assert 3.3180796687645364e-09 < 1e-09` (`tests/test_curvo_gpu.py:251`): la
divergencia del camino float64 se queda 3.3× por encima de una tolerancia que es
estricta de más. **Verificado que es pre-existente en HEAD** (se comprobó guardando
el diff en la rama `paso1-wip` y relanzando el test sobre el árbol limpio), no
regresión de nada reciente, y **sigue igual tras `ecb8dc0`**. Aislado tarda 1.57 s
frente a los 250–376 s de la suite entera.

**F5, lo medido**. NACA 0012 a α = 5°, en los dos regímenes: laminar a Re = 1000
(donde el flujo 2D laminar *es* la física y el criterio mide la discretización) y
**Re = 1e5 con Spalart-Allmaras**, que es el punto de trabajo.

| criterio del plan | pedido | Re = 1000 | **Re = 1e5 + SA** |
|---|---|---|---|
| Cl superficie vs ΔCp | < 1 % (hoy 7 %) | 0.05–0.95 % | **0.45 %** |
| Cl superficie vs circulación | < 1 % | 1.0–5.0 % | **1.8 %** |
| Γ estable con el lazo | < 5 % (hoy 0.65→0.29) | 9–12 % | **6.1 %** |
| ΔCp_TE sin parche de Kutta | bajo tolerancia | −0.008 a −0.046 | **−0.025** (cartesiano −0.092 **con** parche) |
| Cl = 0 a α = 0 | máquina | **< 1e-9** en float64 | — |

A Re = 1e5 con SA el caso se asienta (Cl 0.4859, Cd 0.02125, `nu_t/nu` = 40,
divergencia 5.8e-8) en vez de oscilar 0.302 → 0.187 → 0.200 como hacía laminar.

## Defectos de malla: etapa 2 (y⁺ < 1 en TODA la pared)

Desde la validación 3, `curvo.malla` viene configurado para la **etapa 2**:

```python
DN_PARED = 8.0e-5        # era 1.9e-4; y antes 2.0e-3 (etapa 1, y+ ~ 10)
N_CAPAS_PARED = 15       # era 0
CRECIMIENTO_PARED = 1.0  # espaciado constante en esas 15 capas
```

`generar_c(px, py)` a secas da ya la malla de producción: 99×384 = **37 534
celdas**, `y⁺` del primer centro con **mediana 0.33, p95 0.69 y máximo 0.80**
sobre NACA 0012 a Re = 1e5. La etapa 1 se pide a mano con `dn_pared=2e-3,
n_capas_pared=0`.

Por qué 8.0e-5 y no 1.9e-4: `y⁺` no es uniforme, `u_tau` se dispara en el pico de
succión. Con 1.9e-4 la mediana era 0.49 pero el **máximo 1.88**, con `y⁺ > 1` en
el 2 % del arco alrededor del morro. Cuesta +7.7 % de celdas y **+2.2 % de tiempo
por paso**, y el refinado se concentra solo en morro y borde de salida porque
`ASPECTO_MAX` ya fija el suelo `ds/100 ≈ 1.2e-4` en el centro de la cuerda.
Las fuerzas no se mueven (Cl +0.01 %, Cd +0.02 %): ver `validacion/v3_ymas1_dn8e5/`.

En malla estructurada el número de capas lo fija la columna más fina, así que
**refinar solo el borde de ataque cuesta exactamente las mismas celdas** que
refinarlo todo: no hay ahorro que perseguir ahí.

**Cada corrida escribe su `figuras/malla.png`** con todos los parámetros de malla
y caso al pie (`M.dibujar(..., parametros=...)`), sin pedirlo. `malla_dominio()`
dibuja además la extensión real del dominio; los límites de vista de `analizar.py`
salen ahora de `caso.json` y no están cableados (era lo que hacía que v2 y v1
pareciesen la misma malla en las figuras).

**Aviso de tests**: dos tests de `test_curvo_malla.py` esperaban espaciado de
pared constante columna a columna. No lo es por diseño: `ASPECTO_MAX` varía `dn`
por columna. Los tests se reescribieron para generar la malla explícitamente y
comprobar lo que la malla promete, no lo que se suponía.

## Borde de salida romo: la cola de cierre (2026-09-21)

**Un perfil con base finita no se mallaba.** AG24 (gap 9.7e-4 c) y **el
NACA_0012 real (gap 2.4e-3 c), que tampoco mallaba y no lo sabíamos**, daban
jacobiano negativo y ortogonalidad de pared por debajo de 30°. El diagnóstico que
había escrito en el test canario culpaba al **morro** de AG24; está mal: las
celdas plegadas salen en `i = 64, 65, 329, 330`, **x ≈ 1.0002** — la unión
base-estela. El 87 % de los J≤0 del historial del GA están ahí; el 13 % restante,
en el morro (cúspides, problema aparte).

### El mecanismo

Colgar el corte de estela del punto medio de una base plana deja un nodo con
**ángulo interior de 90°**. La marcha le pide a la vez ser normal a la base (+x)
y normal al corte (+y), y un nodo no abre un abanico de 90°. No es negociable:
la base es perpendicular al flujo, la estela va con el flujo, y el corte es una
rendija de espesor cero cuyos dos lados son los **mismos puntos** (`[0, ::-1]`),
así que no puede salir a ±45° repartiendo el giro.

Medido, el mínimo de ortogonalidad está en **un solo nodo**: `i = 64`, 26.0°,
con 88-90° en sus vecinos y mediana 89.9° sobre el perfil.

### Lo que NO lo arregla — todo medido sobre 400 perfiles del historial del GA

| intento | GA válidas | AG24 |
|---|---|---|
| defectos (base plana) | **0.0 %** | NO, J≤0 6, ort 26.0° |
| rampa de espaciado en la estela | 0.5 % | — |
| base redondeada | 0.0 % | — |
| `MEZCLA_VOLUMEN` 0.3 → 0.9 | 0.0 % | — |
| más nodos en la base | 0.0 % | **empeora**: ver abajo |
| destrabado elíptico local | — | J≤0 0 pero **ort 1.4°** |
| afilar al punto medio | 19.0 % | BUENA, pero **NACA_0012 sigue NO (J≤0 10)** |
| **cola L = 4·gap** | **24.5 %** | **BUENA, J≤0 0, ort 83.1°** |

**Refinar la base no ayuda, y está medido.** Forzando el número de nodos de base
por encima del techo de `dn_pared`, la ortogonalidad **satura en ~28°** y el
plegado **empeora**: n_base 2 → J≤0 24 / 21.1°; 6 → 6 / 26.0°; 64 → **46** /
28.3°. Rayos paralelos saliendo de una base plana no pueden llenar la cuña de 90°
que se abre aguas abajo, y cuantas más columnas pongas, más se cruzan.

También se probó relajación elíptica global (Winslow con el paso normal
reimpuesto en cada barrido): preserva `dn` exacto pero J≤0 pasa de 6 a **1467**.
Winslow sin funciones de control iguala espaciados, y con aspecto 770 junto a la
pared eso es catastrófico.

### La solución: `COLA_TE = 4.0`

La base se cierra con una **cuña recta** desde las dos esquinas reales hasta una
punta a `COLA_TE · gap` aguas abajo. La punta es un borde de salida afilado
normal, con el corte de estela saliendo **colineal** (ángulo incluido 14.3°).
**No mueve ni un punto del perfil: solo añade** — afilar, que es la alternativa,
desplaza las dos esquinas reales gap/2 aguas arriba.

- La ventana útil tiene los dos lados: 1.5 deja la punta demasiado roma (J≤0 104)
  y 10 demasiado afilada en bases gruesas (J≤0 44). **De 2.5 a 6 todos los
  perfiles del repo salen BUENA; 4 es el centro.**
- Cola tangente (Hermite desde la esquina) va **peor**, 0 % — la recta al punto es
  lo que mantiene la colinealidad con la estela.
- Perfiles del repo: **9/10 BUENA** (antes 6/10). Los 6 de `gap = 0` salen
  **bit a bit idénticos** (la rama no se toca; hay test con `np.array_equal`).
- Simetría intacta: `|Y + Y[:, ::-1]| = 3.9e-13` en NACA_0012 romo, igual que el
  sharp → el Cl = 0 exacto a α = 0 sigue en pie. `div_uinf_rel` 1e-15.
- Historial del GA, 1200 perfiles: **0.0 % → 24.2 % válidas**, 10.4 % buenas.

### El precio, explícito

La cola es **cuerpo inventado**: AG24 0.39 % de cuerda, NACA_0012 0.96 %.
`info["cola_rel"]` la lleva y `calidad()` avisa por encima del 1 %
(`AVISOS["cola_rel_max"]`). `ganador_ag.dat` (gap 1.1e-2) dispara *"cola de
cierre del TE 4.3 % de cuerda"*, que es lo que se quiere ver **antes** de citar
unas fuerzas. Se pierde resistencia de base y desprendimiento desde la base.

**NO se ha medido el efecto en Cl/Cd.** Requiere correr el solver; pendiente.

### Por qué no se arregla la base plana de verdad

Se puede, pero no en esta topología, y son dos proyectos:

1. **Multibloque** — lo que hacen los códigos de producción. No quitan el codo:
   ponen ahí una **frontera de bloque** (bloque H detrás de la base), donde las
   líneas pueden no ser suaves sin contaminar nada. En monobloque el codo queda
   *dentro* de la malla. Coste: el corte deja de ser un espejo de índices y pasa a
   ser interfaz entre bloques → `operadores.py`, `proyeccion.py`, `multigrid.py`
   y `solver.py`.
2. **Generación elíptica con control de pared** (Sorenson/TTM). La marcha es un
   problema de valor inicial; una elíptica es de contorno y reparte la distorsión
   sobre muchas celdas. Monobloque, no toca el solver, pero reescribe el generador
   y obliga a revalidar el y⁺.

**Decisión: ninguna de las dos.** El gap de AG24 es 0.097 % de cuerda y su
resistencia de base a Re=1e5 es despreciable. El multibloque se justifica solo si
hace falta **física de base** (perfiles truncados, desprendimiento del borde romo).

`ganador_ag.dat` sigue NO VÁLIDA por **1 celda en `j=0, i=i_le−1`** — el morro,
no el TE. Es el problema conocido de cúspides del GA. Un barrido de 108 mallas
(1.3 s) encontró que `razon_le` 0.08 → 0.04 lo arregla: J≤0 1 → 0, válida.

## GUI curvilínea: parámetros por pestañas (2026-09-21)

`gui/params_spec.py` pasa de lista plana a **tres niveles**: pestañas → bloques
plegables → parámetros, cada uno con su frase de ayuda.

```
Malla      →  Pared · Superficie · Borde de salida · Estela y dominio · Marcha (avanzado)
Fisica     →  Corriente libre · Turbulencia · Integracion temporal
Ejecucion  →  Duracion y muestreo · Backend
```

Perfil y presets arriba, fijos. **Calidad de malla y Simulación fuera de las
pestañas, siempre visibles** y a dos columnas: son el resultado de tocar los
parámetros. El primer bloque de cada pestaña abre por defecto; los cerrados se
encogen al título.

**Tooltip en los 35 parámetros y en las 10 métricas de calidad.** Los textos
están **resumidos de los docstrings de `curvo/malla.py`**, o sea de lo medido, no
inventados. Si cambia un defecto o una medida, hay que tocar los dos sitios.

Dos cosas que faltaban: **`cola_te` no estaba en la GUI** y **`cola_rel` no se
mostraba**. Ahora `Borde de salida` lo edita y el panel de calidad lo muestra.

`tests/test_gui_params_spec.py` fija el contrato: ningún parámetro del caso sin
control, ningún control huérfano, ninguno sin ayuda, ningún desplegable sin
opciones.

### Auto-ajuste de malla — viable, no implementado

`generar_c` cuesta 32 ms y `calidad()` 9 ms: **81 mallas/s con 12 procesos**, sin
tocar el solver. Un barrido de 108 configuraciones sobre `ganador_ag.dat` dio
36 válidas en 1.3 s y encontró la que los defectos a mano no daban.

Si se implementa, dos reglas que no son opcionales: **`dn_pared`, `x_out` y
`distancia_lejos` no son palancas** (los fija el y⁺ objetivo y el estudio de
dominio — ajustarlos sube la puntuación cambiando la física a tus espaldas), y el
ajustador tiene que **declarar cuánta geometría inventada ha metido**.
`calidad()` es un veredicto **geométrico**: optimizar contra él compra validez con
física, que es exactamente lo que hace `cola_te`.

## Dónde se va el tiempo

`Solver(..., cronometro=True)` reparte el paso entre sus tres etapas y
`s.reparto()` lo devuelve en % y ms. Sincroniza la GPU antes de cada lectura del
reloj, así que mide trabajo y no encolado — la suma da el 100 % del reloj de
pared. Medido sobre 8 000 pasos, float32, RTX 3070 Ti, **línea base antes de la
optimización**:

| etapa | v3 (37 534 celdas) | v2 (33 704 celdas) |
|---|---|---|
| **momento** (2 ecuaciones) | **49.9 % — 83.30 ms** | 50.5 % — 81.64 ms |
| presión (Poisson + PCG) | 31.2 % — 52.16 ms | 30.3 % — 49.01 ms |
| turbulencia (SA) | 18.9 % — 31.58 ms | 19.2 % — 30.97 ms |
| total | 167.0 ms — 5.98 it/s | 161.6 ms — 6.18 it/s |

+11 % de celdas cuesta **+3.3 % de tiempo por paso**: a estos tamaños la GPU no
se satura y lo que se paga es latencia de lanzamiento, no trabajo.

El cuello **era** el número de resoluciones de multigrid por paso, no una etapa
suelta: **11** (momento 2 × 3, presión 3, SA 2) a ~15 ms cada una. Con los cuatro
pasos hechos son **7** (momento 2 × 2, presión 2, SA 1), y el reparto queda:

| etapa | línea base | tras pasos 1–3 | tras el paso 4 | **tras la fase 2** |
|---|---|---|---|---|
| momento | 83.30 ms (49.9 %) | 53.21 ms (41.7 %) | 53.22 ms (45.7 %) | **30.87 ms (36.8 %)** |
| presión | 52.16 ms (31.2 %) | 42.86 ms (33.6 %) | 42.99 ms (36.9 %) | **34.58 ms (41.2 %)** |
| turbulencia (SA) | 31.58 ms (18.9 %) | 31.69 ms (24.8 %) | 20.29 ms (17.4 %) | **18.49 ms (22.0 %)** |
| total | 167.0 ms — 5.98 it/s | 127.8 ms — 7.83 it/s | 116.5 ms — 8.58 it/s | **83.94 ms — 11.91 it/s** |

Las dos fases juntas son **×1.99** sobre la línea base, medido de corrida de 8 000
pasos a corrida de 8 000 pasos (`validacion/v3_ymas1_dn8e5` frente a
`validacion/v4_fundido`, mismo caso, misma malla, mismo `dt`). **Ese ×1.99 cruza
sesiones** y por tanto arrastra el ±4 % de ruido del aviso de abajo; el número
apretado es el de la cadena de A/B medidos seguidos dentro de la misma sesión:
**115.64 → 85.79 ms, ×1.35 solo para la fase 2**, con ±0.05 % de ruido.

**La presión pasa a ser la etapa más cara** (41.2 %). El momento ya no lo es: era
el 49.9 % de la línea base y ahora es el 36.8 %. Lo que queda en el momento ya no
es multigrid, es la construcción de la matriz y la jerarquía.

**Aviso al comparar totales entre sesiones**: el mismo código ha medido entre
107.7 y 116.5 ms/paso según la corrida y la longitud de la ventana. Dentro de un
A/B el ruido es ±0.05 %, pero **no se pueden cruzar totales de sesiones
distintas**: solo valen los pares base/optimizado medidos seguidos.

### Plan de optimización del curvo, fase 1 — 5 pasos, los 4 primeros hechos

Objetivo 167 → ~110 ms/paso **sin mover la física**. Criterio de aceptación
común: ΔCl y ΔCd < 0.5 % acumulados en un A/B de 800 pasos contra la línea base
de v3, `div` del mismo orden (5.68e-6), y cada paso en su propio commit para que
`git revert` baste.

**Paso 0 — determinismo, comprobado.** Dos corridas base idénticas dan resultados
**bit a bit iguales** en todas las columnas de física (máx. diferencia 0.0). Y el
ruido de medida es minúsculo: 164.49–164.57 ms de total entre corridas base
(±0.05 %), ±0.19 ms en momento. Sin esto no se puede llamar señal a nada.

**Paso 1 — `u` y `v` comparten matriz y jerarquía** (`8e0e112`, hecho). Los
coeficientes de la matriz de momento salen de los flujos, de `nu_ef` y de **qué**
caras son Dirichlet, no de su **valor**: construirla dos veces por paso no
compraba nada. `cv.sistema` acepta ahora una **lista** de fronteras y devuelve un
término independiente por cada una (con un `assert` de que la estructura de
frontera coincide, porque si no el término saldría mal en silencio), y `avanzar`
cachea la jerarquía en la propia matriz (`A._niveles`) como ya hacía `proyectar`.
Van en serie, no entrelazadas: `avanzar` reasigna `A.b` en cada corrección.

- **Bit a bit idéntico** sobre 800 pasos del caso v3 (`historia.npz`, máx. dif. 0.0).
- Momento **81.97 → 76.87 ms (−6.2 %)**, total **164.57 → 160.82 ms (−2.3 %)**,
  6.18 → 6.22 it/s. La mejora de momento es **27× el ruido** de la línea base.
- Presión +0.40 ms y SA +0.94 ms: dentro o al borde del ruido, despreciables.
- Micro-benchmark que lo motivó (`bench_0b.py`): `cv.sistema` cuesta 1.237 ms/paso
  y `jerarquia` 5.346 ms/paso; quitar una construcción entera de momento eran
  6.584 ms/paso = 3.8 % del total. Se ha capturado la mayor parte.

**Paso 2 — `correcciones` de momento 2 → 1** (`6a0f886`, hecho). Es el paso que
más ha dado, y el argumento es el que importa: **`correcciones` no fija dónde está
el punto fijo de la corrección diferida, solo la velocidad con que se llega.**
Cada paso de tiempo reanuda la corrección del anterior sobre un campo que apenas
se mueve, así que marchando en el tiempo se acumulan miles de iteraciones. El
default de `Solver` pasa a 1.

- **La primera medida decía lo contrario y estaba mal planteada.** Un barrido de
  0/1/2/3/8 correcciones sobre la solución manufacturada **desde frío** (cada
  malla arrancando de unos, con solo `correcciones` iteraciones) daba orden < 1.68
  por debajo de 8 y orden ~2.0 con 8. Eso mide la convergencia **hacia** el punto
  fijo, no la exactitud **del** punto fijo.
- **Encadenando 12 llamadas a `avanzar` reanudando desde el `phi` anterior** —que
  es como reanuda el solver— `correcciones=1` y `correcciones=2` dan el **mismo
  error a seis decimales**: razón de errores **1.000000** en las tres mallas
  (24, 48, 96) y en los dos regímenes, con **orden 2.00 difusivo y 2.02
  convectivo**. Por eso el test de orden de `tests/test_curvo_conveccion.py` sigue
  usando 8: mide desde frío a propósito.
- A/B de 800 pasos del caso v3: **Cl 0.392465 → 0.392468 (+0.001 %)**, **Cd
  0.0252215 → 0.0252215 (−0.000 %)**, Cm −0.00689269 → −0.00689294 (−0.004 %).
  Dentro del 0.5 % con margen de tres órdenes.
- Momento **77.49 → 53.99 ms (−30.3 %)**, total **162.5 → 138.8 ms (−14.6 %)**,
  **6.14 → 7.19 it/s (+17 %)**. Presión y SA sin cambio (−0.17 % y −0.06 %).
- **La divergencia no se degrada.** El máximo de la ventana sube de 2.68e-6 a
  8.00e-6 (+199 %) y eso asustó, pero mirando la serie completa oscila en los dos
  sentidos (a t=0.5 el paso 2 da 7.1e-6 frente a 1.6e-5 de la base; a t=1.0,
  1.3e-5 frente a 2.9e-6) y la **mediana es 6.90e-6**, comparable a la de v3
  (8.44e-6 sobre una ventana diez veces más larga). El máximo cayó en el último
  instante por azar. El transitorio inicial es idéntico (1.313e-02), así que la
  inicialización no se ha tocado. **No citar el +199 % como degradación.**

**Paso 3 — `correcciones_p` de presión 2 → 1** (`b2c9fe0`, hecho). El default de
`Solver` pasa a `correcciones_p=1`. Dos argumentos, y el segundo es el que no
estaba escrito antes: las correcciones cruzadas de la presión las fija la
**oblicuidad de la malla**, que sobre perfiles reales tiene p99 = **0.0065**
frente al 0.10 del cuadrado distorsionado del test —que sí necesita 4
(`curvo/proyeccion.py:130-133`)—, y lo que se resuelve cada paso no es la presión
sino su **incremento** `phi`, casi nulo en régimen asentado.

- A/B de 800 pasos del caso v3: **Cl +0.000 %**, **Cd −0.003 %**. Dos órdenes
  dentro de la puerta del 0.5 %.
- Presión **52.67 → 42.86 ms (−18.6 %)**; acumulado con el paso 2, total
  **162.5 → 127.8 ms (−21.4 %)**, **6.15 → 7.83 it/s**.
- **Lo que sí se paga: la divergencia residual se asienta más alta.** Mediana
  6.25e-6 → **1.34e-5 (+114 %)**, p95 1.92e-5 → 1.02e-4, máximo 4.58e-5 →
  1.86e-4. Sigue dentro de la tolerancia relativa del PCG en float32, y la
  mediana de v3 completo (8.41e-6, t=0..20) cae entre las dos. Los 23 tests de
  `test_curvo_proyeccion.py` pasan.
- **Y lo que se comprobó antes de aceptarlo: el modo par-impar NO crece.** Medido
  sobre el campo final, la fracción de tablero en `p` vale 7.818e-5 (base) /
  7.857e-5 (paso 2) / 7.849e-5 (paso 3), la de velocidad 1.033e-6 / 1.034e-6 /
  1.035e-6, y la oscilación del `Cp` de pared en el borde de salida sale
  **8.837e-2 idéntica en las tres**. La divergencia más alta no es inestabilidad
  espacial ni amplificación del Poisson.
- **Pendiente de este paso**: confirmarlo sobre una corrida completa a t\*=20, no
  solo sobre los 800 pasos del A/B. Va junto con la confirmación del paso 4.

**Paso 4 — SA: `correcciones` de 1 → 0** (`1918c80`, hecho). El default de
`avanzar_sa` en `curvo/turbulencia.py` pasa a 0, y el solver lo llama sin pasar
el argumento. **Lo que hay que entender antes de tocar nada: `correcciones = 0`
NO deja el transporte de `nu_tilde` en upwind de 1.er orden.** El bucle de
`avanzar` sigue evaluando la corrección diferida **una vez, sobre el campo del
paso anterior**: es Picard retrasado un paso, no Picard suprimido. Marchando en
el tiempo llega al mismo punto fijo, que es el mismo argumento del paso 2 llevado
a su conclusión.

- **Medido encadenando, no desde frío** (`punto_fijo.py`, scratchpad): con
  `corr ∈ {0, 1, 2}` los errores salen **idénticos a seis decimales** en las tres
  mallas y en los dos regímenes — difusivo 6.545e-04 / 1.639e-04 / 4.098e-05,
  convectivo 1.209e-03 / 2.961e-04 / 7.314e-05, con **orden 2.00 difusivo y
  2.02–2.03 convectivo** y razón `e(corr=1)/e(corr=2) = 1.000000`. El hallazgo
  no es solo del modelo de turbulencia: **vale para el solver de transporte
  entero**.
- A/B de 800 pasos del caso v3: **Cl +0.001 %**, **Cd −0.002 %**,
  **`nu_t/nu` +0.008 %** — la guardia que se puso al paso (vigilar `nu_t/nu`)
  no se mueve.
- Turbulencia **32.34 → 20.29 ms (−37 %)**; acumulado, total **162.5 → 116.5 ms
  (−28.3 %)**, **6.14 → 8.58 it/s**.
- Divergencia residual: +50.7 % relativo sobre el A/B, con el valor absoluto
  todavía pequeño. Es el mismo tipo de subida que el paso 3 y **entra en la misma
  confirmación pendiente a t\*=20**, no en una lectura aparte.

**Paso 5, pendiente — solo confirmación, no toca código**: rehacer el reparto
sobre los 8 000 pasos del caso v3 completo, y con él la comprobación a t\*=20 de
que la divergencia residual acumulada de los pasos 3 y 4 no deriva en una corrida
larga. Medir la **mediana**, no el máximo, y si sube de verdad medir el modo
par-impar antes de llamarlo degradación. **Ahora conviene hacerlo ya con
`ecb8dc0` dentro**, para no pagar dos corridas largas.

### Fase 3 de optimización — el suavizador: 86.8 → 54.7 ms/paso (×1.59)

**11.5 → 18.3 it/s**, con las fuerzas dentro del 0.01 %. A/B de 800 pasos contra
`dab73d1` (worktree, medidas seguidas) y confirmación de 8 000 pasos en
`validacion/v5_suavizador_pcr/`. La herramienta del A/B, que antes no existía, es
`scripts/ab_curvo.py` (`--correr` / `--comparar`).

**El diagnóstico, con nsys**: el paso gastaba 55.5 ms de GPU de 82 de pared, con
4654 lanzamientos y el **94 % del tiempo de GPU en el suavizador de líneas**, que
corría con **64–128 hilos de los 6144 núcleos**. Referencia de techo: el producto
matriz-vector sobre las 37 534 celdas cuesta **1.45 µs** y un barrido de líneas ξ
del nivel fino costaba **143 µs**. El problema no era ancho de banda: era una
línea por hilo con una recursión secuencial de 384 pasos.

**Paso 1 — las sincronizaciones (`fijos` + `activo` en la CPU)**. `resolver` y
`resolver_pcg` aceptan una cuenta fija de ciclos y no miran el residuo; `Solver`
la recalibra en el arranque y cada 50 pasos con un paso comprobado. Pero las
sincronizaciones caras **no eran esas**: eran 16 `aC.get()` por paso escondidas en
`Sistema._indices`, una por nivel de la jerarquía y por matriz construida, que
valían **13.46 ms de espera**. Se quitan pasando la máscara del corte ya en la
CPU (`Solver.corte_cpu`, `activo=` en `cv.sistema`). Verificado con un espía sobre
`cupy.ndarray.get`: **0 sincronizaciones por paso**. 85.9 → 67.8 ms (×1.27).

**Paso 2 — la convección solo relaja líneas η** (`Sistema.xi`, `BARRIDOS = 3`).
Medido sobre las matrices reales: el momento con η y tres barridos deja el
residuo en 3.8e-5 tras dos ciclos frente a 1.2e-4 del ADI, y cuesta 2.90 ms por
ciclo frente a 4.57 — más exacto y 1.58 veces más barato. **En el Poisson de
presión es al revés y por goleada**: el factor del PCG se va de 0.031 a 0.93. Por
eso `xi` es de la matriz y no del solver, y el defecto es `True`: el Poisson se
monta llamando a `conveccion.sistema` (`proyeccion.py:44`), y apagarlo ahí dentro
mandó el paso a **286 ms** con 81 iteraciones de PCG. 86.6 → 58.0 ms (×1.49).

**Paso 3 — reducción cíclica paralela**. Un bloque por línea y un hilo por
incógnita, en memoria compartida con doble buffer: `zebra_xi_pcr`,
`zebra_eta_pcr`, `zebra_eta_par_pcr` y los gemelos de dos campos, todos sobre
`pcr1`/`pcr2`. **Barrido de líneas ξ del nivel fino: 141.2 → 7.8 µs (×18)**;
suavizador completo ×12.3 en presión y ×7.3 en momento. Resuelve la misma
tridiagonal: en float64 coincide con Thomas en **1.2e-13**, y el 4.6e-5 de float32
es redondeo (factor del PCG 0.0223 con PCR frente a 0.0280 con Thomas).

**Dos cosas que costaron encontrar**:

- **`cronometro=True` falsea el A/B.** `_marca` sincroniza tres veces por paso y
  eso drena la tubería, que es justo lo que se optimiza: la mejora de la fase 1
  se veía como ×1.03 con cronómetro y era ×1.27 sin él. El `ms/paso` que se cita
  sale de 60 pasos con el cronómetro apagado; el reparto por etapas sigue
  valiendo para ver **dónde** se va el tiempo, pero su total no.
- **Con la GPU ya rápida, el cuello se mueve a la CPU.** La PCR salió *más lenta*
  de pared en el primer intento (59.8 frente a 58.0) porque el despacho construía
  `np.int32` y calculaba bloques y memoria compartida en cada uno de los ~2400
  lanzamientos por paso. Precalculado en el constructor: 55.1 ms.

**La jerarquía se para por razón, no por número de celdas** (`RAZON_GRUESA = 1024`):
lo que fija el factor es la razón entre el nivel más grueso y el fino. Recortar
niveles es tentador porque cada uno cuesta una tanda de lanzamientos, pero se
paga en escalabilidad, y eso sí se midió: con 1/64 el factor del ciclo V a n=256
se va a 0.641 (puerta 0.65) y el del PCG a 0.589 (puerta 0.3); con 1/256 pasan
pero el PCG queda a un 17 % de su puerta y empeorando al refinar. **1/1024 se
queda a un tercio de la puerta** y da 7 niveles en vez de 8 sobre la malla C. La
parada siguiente (1/256) daría 46.6 ms y es una decisión de riesgo, no de código.

**Confirmación larga** (8 000 pasos, t\*=20, `validacion/v5_suavizador_pcr`),
contra la misma corrida con `dab73d1`:

| magnitud | base | fase 3 | Δ % |
|---|---|---|---|
| Cl_sup | 0.498499936 | 0.498445969 | −0.0108 |
| Cl_circ | 0.487670134 | 0.487476501 | −0.0397 |
| Cd | 0.0193136418 | 0.0193119156 | −0.0089 |
| Cd_p | 0.00756678721 | 0.00756661707 | −0.0022 |
| Cd_v | 0.0117468545 | 0.0117452986 | −0.0132 |
| Cm | −0.00791676517 | −0.00791747149 | +0.0089 |
| ΔCp_TE | −0.027857814 | −0.0278565176 | −0.0047 |
| mediana de `div` | 7.446e-06 | 7.607e-06 | +2.15 |
| tablero en `p` | 1.3355e-04 | 1.3461e-04 | **+0.79** |
| ms/paso | 79.98 | **51.38** | ×1.557 |

El tablero merece la nota: en el A/B corto subía un 3 %, y en la corrida larga se
queda en +0.79 %. Era transitorio, no deriva. La suite: **151 pasan**, y el único
fallo sigue siendo el pre-existente `test_float32_da_la_misma_respuesta`.

**CUDA Graphs, descartado con dato**: la matriz del momento y la de SA se
reconstruyen en cada paso, así que los punteros que capturaría el grafo mueren
con ellas. Capturar exigiría escribir matrices y jerarquía in-place; el hueco de
lanzamiento se atacó por las otras dos vías.

### Fase 2 de optimización — latencia de lanzamiento: nivel grueso y kernels gemelos `u`/`v`

La fase 1 dejó el solver sin resoluciones de multigrid sobrantes. Lo que queda
por debajo es **latencia de lanzamiento de kernels**, no trabajo aritmético: con
37 534 celdas el suavizador usa ~128 hilos de los 6 144 núcleos de la 3070 Ti.

**Paso 1 — el nivel grueso, de 30 barridos a 4** (`ecb8dc0`, hecho). El nivel
final de la jerarquía es una malla de **tres celdas** y se relajaba 30 veces: 120
lanzamientos de kernel para tres incógnitas, **0.80 ms de los 4.15 que cuesta un
ciclo V (el 19 %)**, y lo pagaban momento, presión y turbulencia por igual. El
default de `ciclo_v` en `curvo/multigrid.py:696` pasa a `grueso=4`.

- **Medido sobre los sistemas reales del solver con término independiente
  aleatorio**, no sobre un problema de juguete: en convección el historial de
  residuos es **idéntico** con 1, 2, 4, 8 y 30 barridos (15 ciclos, factor 1.0,
  residuos 1.0e+00 → 1.2e-04 → 3.4e-07 en las cinco) — el operador es tan
  dominante en diagonal que la corrección gruesa no llega a intervenir. Y en el
  Poisson de presión el factor del PCG **no empeora al bajar, mejora**: 0.067 con
  30, **0.021 con 4**.
- A/B de 800 pasos del caso v3: **115.64 → 103.81 ms/paso (×1.11), 8.65 → 9.63
  it/s**. Por etapas: momento ×1.076, **presión ×1.202** (la que más gana),
  turbulencia ×1.048.
- **Las fuerzas se mueven menos del 0.005 %** (Cl, Cd, Cm). No es bit a bit
  idéntico —el cambio de barridos propaga diferencias de redondeo—, pero está
  tres órdenes dentro de la puerta del 0.5 %.
- **El modo par-impar se queda donde estaba**: fracción de tablero en `p`
  7.848e-5 → **7.843e-5**. Ese es el diagnóstico con sentido físico. La mediana
  de la divergencia sube **8.6 %** (1.54e-5 → 1.67e-5), dentro del mismo orden;
  el **máximo instantáneo marca +379 % (4.03e-06 → 1.93e-05) y es ruido de
  instante**, exactamente el mismo falso positivo que el +199 % del paso 2. **No
  citarlo como regresión de exactitud.**
- **El barrido de `minimo` (profundidad de la jerarquía) no dio nada**: 8 → 7 → 6
  → 5 niveles (`minimo` de 9 a 576) no marca tendencia clara de tiempo. El coste
  es de lanzamiento por nivel, no del número de niveles. **Descartado por ahora**,
  no merece la complejidad.

**Paso 2 — kernels gemelos de dos campos, hecho** (`6e3aa65`). `u` y `v` ya
compartían matriz y jerarquía desde la fase 1, pero **se resolvían en serie**: 4
ciclos V completos por paso y dos lecturas de los mismos coeficientes. Ahora el
hilo que resuelve una línea los lee una vez y arrastra las dos recursiones de
Thomas. Kernels nuevos en `curvo/multigrid.py`: `zebra_eta2`, `zebra_eta_par2`,
`zebra_xi2`, `aplicar2`, `restringir2`, **al lado** de los de un campo, que no se
tocan — presión y SA siguen por el camino de siempre y no pueden regresar.

- **El prototipo ingenuo (`float d[nc]` indexado por la variable del bucle) sale
  ×0.73, o sea 37 % MÁS LENTO** que las dos llamadas en serie: CUDA no puede
  probar las cotas en compilación y derrama el array a memoria local. Con
  escalares `d0`, `d1`: 36 registros por hilo (frente a 40 del kernel de un
  campo), cero memoria local, **×1.63 sobre `resolver` entero**. Esa es la razón
  de escribir gemelos y **no** generalizar a `nc` campos arbitrarios.
- **El margen viene de que el suavizador va famélico de paralelismo, no de ancho
  de banda**: el barrido de líneas η del nivel fino lanza **128 hilos sobre 6 144
  núcleos**, cada uno con una recursión secuencial de 98 pasos. El segundo campo
  añade instrucciones *independientes* que rellenan las burbujas de la cadena
  dependiente, así que casi no cuesta.
- Los coeficientes `c` de Thomas **no dependen del término independiente**, así
  que con dos campos se calculan una vez; solo `dp` va por campo, intercalado
  `(n, L, 2)`. Queda una división por celda en vez de dos por celda y campo.
- **Verificación kernel a kernel sobre la matriz de producción: identidad EXACTA
  (`err = 0.0e+00`)** en líneas η, líneas η emparejadas del corte, líneas ξ,
  suavizado directo e invertido y producto matriz-vector. `ciclo_v` se desvía
  **5.4e-9** — el peso de Rayleigh se reduce por ejes en vez de entero y cupy no
  garantiza el mismo orden de suma; son **22 veces menos que el ε de float32**.
- **Paran juntos pero cada campo se normaliza por SU término independiente.**
  Normalizar los dos por el máximo común relajaría al pequeño: a α=5° el término
  de `v` es un orden menor que el de `u` y se daría por convergido antes de
  tiempo. (La hipótesis previa se validó: en 800 pasos, `u` y `v` **nunca**
  convergen en número de ciclos distinto — 0 casos de 800.)
- A/B de 800 pasos: momento **49.11 → 34.67 ms (×1.42)**, total 103.81 → 88.48.
  Las 21 columnas de la historia coinciden con la corrida anterior en todas las
  cifras impresas.

**Paso 3 — corrección diferida por lotes, hecho** (`63e1d13`). Era lo único que
quedaba en serie del momento: 4 llamadas de 2 ms por paso, todo cupy elemento a
elemento, ~50 lanzamientos diminutos cada una. El coste no es aritmético, es de
lanzamiento, así que con los dos campos en un eje delante se divide por dos.

- Los operadores de cara (`dif_xi_caras`, `dif_eta_caras`, `a_caras_xi`,
  `a_caras_eta`, `divergencia`) y el limitador pasan a **indexar por el final**:
  los dos últimos ejes son la malla y delante puede ir un eje de campos. El flujo
  de masa y la métrica no lo llevan —son los mismos para todos— y se difunden.
- **La trampa que costó encontrar**: al apilar los valores de Dirichlet por campo
  hay que **no forzarles el tipo**. Un valor suelto llega como `float` de Python
  y al multiplicarlo por la cara promociona a doble precisión; redondearlo antes
  a float32 movía el término de las **dos filas pegadas a la frontera norte** en
  1.3e-4 relativo. Sin el cast, el resultado coincide con el de resolverlas
  sueltas en **1.3e-10**.
- Medido: **×1.80** sobre las dos llamadas por separado. A/B de 800 pasos:
  momento 35.14 → 31.34 ms (×1.12), total 89.77 → 85.79. **Las 21 columnas a
  0.0000 %**, la divergencia incluida.

**Confirmación larga de las dos fases** (`validacion/v4_fundido`, 8 000 pasos,
t\*=20, NACA 0012, Re=1e5, α=5°, float32, SA, 677 s). Contra
`validacion/v3_ymas1_dn8e5`:

| magnitud | v3 | v4 | Δ % |
|---|---|---|---|
| Cl_sup | 0.498465897 | 0.498477164 | +0.0023 |
| Cl_dcp | 0.495323287 | 0.495334385 | +0.0022 |
| Cl_circ | 0.487578861 | 0.487641881 | +0.0129 |
| Cd | 0.019315507 | 0.019314973 | −0.0028 |
| Cd_p | 0.007568798 | 0.007568033 | −0.0101 |
| Cd_v | 0.011746709 | 0.011746940 | +0.0020 |
| Cm | −0.007917233 | −0.007917298 | +0.0008 |
| dCp_TE | −0.027860325 | −0.027859703 | −0.0022 |
| nut_nu | 38.0383484 | 38.0397512 | +0.0037 |
| div (mediana) | 8.4393e-06 | **7.4526e-06** | **−11.7** |

**Peor desviación de una fuerza: 0.013 %**, dos órdenes dentro de la puerta del
0.5 %. La **divergencia residual baja**, no sube. El modo par-impar se queda
donde estaba: fracción de tablero en `p` 1.3379e-4 → **1.3347e-4**, oscilación
del Cp en el borde de salida 9.627e-3 → 9.614e-3.

**Aviso sobre `disp_gamma`**, que marca −0.469 % y es lo único cerca de la puerta:
**no es una fuerza**, es `ptp/media` de la circulación medida en cinco lazos de
radio distinto, o sea una medida de **dispersión**. Pasa de 6.554 % a 6.523 % de
dispersión — 0.03 puntos porcentuales. Las cinco circulaciones individuales se
mueven **≤0.032 %** (`gamma_0.15` +0.0009 %, `gamma_1.50` +0.0322 %). No citarlo
como desviación de exactitud.

- **El barrido de `minimo` (profundidad de la jerarquía) no dio nada**: 8 → 7 → 6
  → 5 niveles (`minimo` de 9 a 576) no marca tendencia clara de tiempo, y con
  `minimo=576` el PCG de presión empeora (43.3 ms frente a 29.4). El coste es de
  lanzamiento por nivel, no del número de niveles. **Descartado.**
- **Sigue fallando `test_curvo_gpu.py::test_float32_da_la_misma_respuesta`**
  (`div64 = 3.32e-09` contra una tolerancia de 1e-9). **Ya fallaba en la línea
  base `6ea1e7b`**, comprobado con `git stash`; no es de este trabajo. El resto de
  la suite: **131 pasan** (3 tests nuevos de dos campos).

### Lo que queda por probar en el curvo

- **Momento con `correcciones` 1 → 0.** El `punto_fijo.py` de la fase 1 demostró
  que el punto fijo de la corrección diferida **no depende** del número de
  iteraciones (orden 2.00 difusivo / 2.02 convectivo con 0, 1 y 2). Bajar a 0
  quitaría **la mitad** de las resoluciones de momento sin escribir una línea de
  CUDA. No se hizo porque cambia la física discreta y merece su propia
  validación.
- **La presión es ahora la etapa más cara** (34.58 ms, 41.2 %). Un solo campo, no
  hay nada que fundir; el margen está en el número de iteraciones del PCG.

## Parada por fuerzas: hasta que Cl y Cd fijan el tercer decimal (2026-09-22)

`curvo/convergencia.py` (nuevo). El criterio que había en el curvo era
`Solver.correr(parada=...)`, que mide el **campo**
(`max|u^{n+1}-u^n|/(u_inf·dt)`), y `t_final=20` a pelo en los scripts de
validación. Ninguno de los dos dice nada sobre **el número que se publica**.
Ahora `ParadaFuerzas` para cuando Cl y Cd dejan de moverse por encima de una
tolerancia **absoluta** (5e-4 por defecto = media unidad del tercer decimal).

Dos condiciones, las dos necesarias, sobre una ventana que **se dobla sola**:

| magnitud | qué mide | cómo |
|---|---|---|
| `banda` | la media está mal determinada porque la señal oscila | medias de 8 bloques: `1.96·s(medias)/√8` |
| `cola` | la media está bien determinada pero **sigue derivando** | `d1·r/(1−r)` con `r = d1/d0`, diferencias de tres bloques |

Dos decisiones que no son cosméticas:

- **La cola, no la pendiente**. Estas series son `A − B·exp(−t/τ)` limpias con
  **τ ≈ 2.9** tiempos convectivos (ajustado sobre v1, v2 y v3). Una pendiente
  pequeña multiplicada por una τ larga sigue siendo un cambio grande, así que lo
  que se compara con el umbral es **lo que le queda por recorrer**, estimado por
  decaimiento geométrico de las medias de bloque. Si `r ≥ 1` la serie no decae y
  no se para nunca; si la diferencia de bloque a bloque cae por debajo de la
  banda, no hay deriva que medir.
- **Medias de bloque, no `σ/√N_ef`**. El cartesiano corrige `N` por
  autocorrelación integrada (`_n_efectivo`), y eso **se pasa de conservador en el
  caso limpio**: sobre un seno de amplitud 1e-2 con 133 ciclos da una banda de
  6.8e-4 cuando el error real de la media es ~1e-5, o sea con umbral 5e-4 no
  abriría nunca. La media de un bloque de varios periodos sí promedia la
  oscilación.

**Medido sobre las cuatro validaciones cerradas** (401 muestras de t=0 a 20,
referencia = asíntota del ajuste exponencial de la propia serie):

| tol | dónde para | error de Cl al parar |
|---|---|---|
| 1e-3 | t = 14.6–18.6 (73–93 % del presupuesto) | 4.8e-4 a 1.1e-3 |
| 5e-4 | v2 en t=16.9, v5 en t=19.7; v1 y v3 **no llegan** dentro de t=20 | 1.8e-4 a 4.9e-4 |

O sea: **el error al parar sale del orden de la tolerancia pedida, y con
`t_final=20` el tercer decimal del Cl todavía no está fijado** — a t=20 a v1 le
quedan 7.9e-4 de cola (a v3 5.4e-4), ~1.3 tiempos convectivos más. El Cd sí:
su cola a t=20 es 3–5e-5. El ahorro es pequeño porque el caso de referencia está
dimensionado justo; lo que aporta el criterio es que el tiempo lo fija la señal y
no un número escrito a mano, que es lo que hace falta al barrer una polar.

`resumen()` guarda también `extrapolado = media + cola`, el valor de meseta, que
apunta 2–6 veces mejor que la media de la ventana (error 1e-4 a 4.6e-4 con
tol=1e-3). Se guarda como diagnóstico: **lo que se publica es la media**, que es
lo que se ha medido de verdad.

Enganches: `Solver.correr(..., parada_fuerzas=ParadaFuerzas())` y
`Solver.coeficientes()` (Cl, Cd ahora mismo; sincroniza la GPU, se llama cada N
pasos), `tol_fuerzas` en `validacion/v1.../correr.py` (0 = presupuesto entero) y
`ejecucion.tol_fuerzas` en la GUI, que vuelca `convergencia.json` al parar.
`tests/test_curvo_convergencia.py` (15 tests, 77 s) cubre los tres modos que
mataron a las versiones 1–3 del criterio del cartesiano: deriva lenta monótona,
rampa que no decae y oscilación permanente sobre media fija.

**En estudios de malla, dejarlo en 0**: si cada malla para en un tiempo físico
distinto, el orden observado mide esa diferencia y no la discretización.

## Validación 1: NACA 0012, α=5°, Re=1e5, contra XFOIL

`validacion/v1_naca0012_re1e5_a5/` — malla y⁺≈1 (34 853 celdas, 15 capas de pared
de espaciado constante), SA, hasta t\*=20, float32, **6.12 it/s**. `y⁺` del primer
centro: mediana 0.49, p95 1.62, **máximo 1.88** — de ahí la validación 3, que baja
el paso de pared a 8.0e-5 y deja el máximo en 0.80 sin mover las fuerzas.

| | nuestro | XFOIL Ncrit=5 | XFOIL Ncrit=9 |
|---|---|---|---|
| Cl | 0.4984 | 0.6026 (**−17.3 %**) | 0.6141 (−18.8 %) |
| Cd | 0.01931 | 0.01680 (**+15.0 %**) | 0.01674 (+15.4 %) |
| **Cd presión** | 0.00757 | 0.00766 (**−1.1 %**) | 0.00843 (−10.2 %) |
| Cd viscoso | 0.01174 | 0.00914 (+28 %) | 0.00831 (+41 %) |
| Cm (c/4) | −0.00794 | −0.0064 | −0.0077 |

**El Cd de presión cuadra al 1.1 %**: el campo de presión está bien y el error es
de estado de la capa límite. XFOIL transiciona en x/c=0.27–0.37; nuestro SA es
turbulento desde el borde de ataque, y eso predice exactamente los dos signos
(menos Cl por descambado, más fricción). **Lo siguiente es el modelo de
transición (SA-BC), no tocar el esquema.** El cartesiano iba **+124 % de Cd**.

Comprobaciones sin ningún ajuste: `Cp` de remanso **1.005** (exacto 1) en
x/c=0.0052 del intradós, pico de succión −1.65, `ΔCp_TE = −0.0278` **sin parche de
Kutta** (cartesiano −0.092 **con** parche), `u⁺=y⁺` hasta `y⁺≈10`, `Cf<0` solo en
x/c>0.966 (burbuja de borde de salida), Cd viscoso entre la placa plana laminar
(0.0084) y la turbulenta (0.0148).

**Defecto abierto**: tablero par-impar en el `Cp` de pared, amplitud 0.11 cerca
del TE y creciendo. No es precisión (f64 = f32) ni los cruzados diferidos
(`correcciones_p` 1 = 2 = 8; ver paso 3, la fracción de tablero no se mueve al
bajar de 2 a 1), y tampoco el nivel grueso (con 4 barridos sale igual que con 30).
Nace en `div(m*)`, que al ser el campo ya solenoidal a
1e-9 es ruido par-impar casi puro, y se acumula en `p += phi`. No toca a `u`, `v`
ni a las fuerzas integradas.

Dos comprobaciones independientes que salen bien y no llevan ningún ajuste:

- **Placa plana contra Blasius**: `Cf/Cf_Blasius` = 1.017 / 1.015 / 1.014 / 0.995.
  Cierra el riesgo 3 del plan — el limitador TVD no degrada la capa límite.
- **Cd viscoso a α = 0**: 0.0871 medido frente a 2·1.328/√1000 = **0.084** de la
  placa plana laminar de dos caras.
- **Cd viscoso a Re = 1e5 con SA**: 0.0132 frente a 2·0.074/Re^0.2 = **0.0148**
  de la placa plana turbulenta.

**Dos bugs que tapaban a SA** (los dos arreglados, ver `curvo/operadores.py` y
`curvo/turbulencia.py`):

1. `op.gradiente` aplicaba la condición de **pared sobre el corte de estela**. La
   cara `j=0` en la estela es cara interior, no frontera, y con `sur = 0` de no
   deslizamiento es un salto de `u=1` a `0` en media celda a lo largo de las 18
   cuerdas de estela: **|ω| = 495 a 1.35 cuerdas del perfil con u = 0.9923 a los
   dos lados**. SA se lo creía y fabricaba turbulencia en la estela desde cero.
   Con la media espejo, ω en el corte cae de 578 a 139 y a más de 0.5 cuerdas
   del perfil pasa a ser 0.
2. La **producción iba explícita y sin cota**: `c_b1 S~ dt = 0.45`, o sea +57 %
   por paso. Producción y destrucción se integran ahora juntas con la solución
   exacta de la logística `d(nt)/dt = a nt − b nt²`, positiva, monótona y acotada
   por el equilibrio local `a/b` con cualquier `dt`. Con eso `nu_t/nu` va de 0.6
   a 15.6 en 200 pasos en vez de llegar a 4685 en 20 y a NaN en 25.

```python
from curvo import malla as M, fuerzas as fz
from curvo.solver import Solver
import cupy as cp, numpy as np

px, py = M.leer_dat("profiles/NACA_0012_sharp")
X, Y, info = M.generar_c(px, py)
s = Solver(X, Y, info, nu=1e-3, alfa=5.0, dt=5e-3, xp=cp, dtype=np.float32)
s.correr(2000)
print(fz.estimadores_de_cl(s.met, s.u, s.v, s.p, info, s.nu, 1.0, 5.0,
                           bc=dict(oeste=None, este=None), corte=s.corte))
```

## Validación 2: independencia de dominio — 24 × 12 contra 26 × 17.5

`validacion/v2_dominio_24x12/` — mismo caso que v1 cambiando **solo el tamaño del
dominio**: 25.9×17.5 (borde lejano a 8.76 c) → **24.00×12.48** (6.24 c), de
34 853 a 33 704 celdas, un 27 % menos de área.

```bash
PYTHONPATH=. .venv/bin/python validacion/v1_naca0012_re1e5_a5/correr.py \
    validacion/v2_dominio_24x12 distancia_lejos=6.0 x_salida=18.39
```

El 12.00 exacto no es alcanzable: el número de capas sale del crecimiento
geométrico y está cuantizado (`distancia_lejos` de 5.8 a 6.2 da la misma malla).

| | v1 (grande) | v2 (24×12) | cambio |
|---|---|---|---|
| **Cl superficie** | 0.49840 | 0.49844 | **+0.01 %** |
| Cl por ∮ΔCp | 0.49527 | 0.49533 | +0.01 % |
| Cl por circulación | 0.48872 | 0.47871 | −2.05 % |
| **Cd** | 0.01931 | 0.01946 | **+0.77 %** |
| Cd presión | 0.00757 | 0.00770 | +1.71 % |
| Cm (c/4) | −0.00794 | −0.00783 | −1.33 % |
| ΔCp_TE | −0.02775 | −0.02778 | +0.11 % |
| dispersión de Γ | 6.34 % | **11.93 %** | +88 % |

**Las fuerzas no dependen del dominio.** El Cl se mueve **0.01 %** recortando un
27 % del área; el Cd, 0.77 %, y todo por la parte de presión. El campo lejano del
arco C está bien puesto: no hay bloqueo apreciable ni a 6.24 cuerdas. **Comparar
con el cartesiano, donde pasar de 8×5 a 24×16 movía el L/D de 27.70 a 31.70
(14 %)** — es otra liga. Contra XFOIL el Cd de presión sale incluso mejor:
+0.56 % frente a Ncrit=5 (v1 daba −1.14 %).

**Y explica la dispersión de la circulación**, que era la quinta capa abierta del
cartesiano: el cambio de Γ **escala monótonamente con el radio del lazo** (−0.23 %
a r=0.15c, −5.82 % a r=1.50c). O sea, la dispersión entre lazos es **artefacto
del dominio finito**, no defecto de la discretización: cuanto más cerca queda el
borde lejano, más recorta el lazo grande. El estimador de superficie, que es el
que se usa, no se entera.

## Qué es
Simulador CFD 2D incompresible (Navier-Stokes) en GPU (CuPy, RTX 3070 Ti) para perfiles alares, base de un futuro optimizador aerodinámico. El solver cartesiano vive en `Sim_Cartesiano/Simulador2D.py` (~8000 líneas, clase `Mesh` + `main()`); el curvilíneo, en `curvo/`.

- Malla cartesiana estirada (zona fina alrededor del perfil), advección semi-Lagrangiana, difusión (+WALE LES opcional), proyección de presión multigrid/CG.
- Sólidos por IBM (Immersed Boundary): máscara rasterizada + ghost-cell no-slip (`ibm_wall_mode="ghost_noslip"`).
- **Ejecutar SIEMPRE con `.venv/bin/python`** (el python3 del sistema no tiene CuPy).

## DOMINIOS ARBITRARIOS DESDE CAD + INTERFAZ (`bff0838`, `c104ba1`)

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
distinguiendo el límite convectivo del viscoso, y malla estirada dentro del
fluido con SA activo (ver abajo).

**SA y conductos, medido.** `_compute_sa_wall_distance` cuenta la distancia a
pared en **número de celdas** y la multiplica por el tamaño de **la celda local**,
así que solo es exacta si todas las celdas entre el punto y la pared miden lo
mismo. Con un perfil se cumple gratis: la capa límite vive dentro de la banda fina
uniforme. Con un conducto la cortadura ocupa **toda la sección**. Medido sobre un
canal de altura 1 en un dominio de 2: con la banda fina cubriendo la sección el
error en `d` es **0 % mediano y 3.1 % de pico**; con la banda solo en el centro y
un estiramiento de ×1.9 dentro del canal, **35 % mediano y 51 % de pico**. La
regla no es "no usar SA", es **la banda fina tiene que cubrir la sección**; la
previsualización lo comprueba y avisa.

**Casos de prueba con verdad conocida** (`tests/test_dominio_arbitrario.py`):
canal de Poiseuille (balance de masa < 2 %, perfil parabólico < 10 %) y tobera
convergente 2:1 (balance < 3 %, aceleración = razón de alturas < 5 %). Geometrías
en `scripts/agent_tests/formas_dominio.py`. Ejemplos listos para abrir:
`.venv/bin/python scripts/agent_tests/demo_gui.py` → `results/gui_demo/`.

**Arrancar la interfaz**: `./lanzar_gui.sh` desde `Sim_Cartesiano/`. Necesita
`python3-pyside6.*`, `python3-pyqtgraph` y `ezdxf`; el venv tiene
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
`Sim_Cartesiano/resultados_finales/`; leer su `README.md` y
`richardson/INFORME.md` antes de citar nada. Inventario de las 744 figuras en
`resultados_finales/FIGURAS.md`.

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

**VERIFICACIÓN NUMÉRICA — documento en `docs/verificacion_numerica.md` (10 secciones).** Leerlo antes de citar cualquier L/D, pero **está desactualizado desde el 14 de agosto**: no incorpora la ventana larga, ni el fix del fitness, ni el estudio de dominio, ni el Richardson de la polar. Guion del TFG en `docs/PROMPT_TFG.md`; memoria en construcción en `docs/tfg/`, con el texto de sustitución de los capítulos 4-6 ya escrito en `docs/tfg/resultados_finales_v2.txt` (`99ffaa2`).

**Reorganización de resultados**: toda la campaña anterior está archivada en **`results/1eraGranOptimizacion/`** (`convergence_study/`, `verificacion_numerica/`, `agent_tests/`, `barridos/`, `metricas_ga/`, `comparativas_v1/`, `videos_ganadores/`, …). `results/verificacion_numerica/` en la raíz solo contiene lo nuevo. Cualquier ruta de este documento o de scripts antiguos que apunte a `results/convergence_study/...` hay que leerla bajo `results/1eraGranOptimizacion/` **y ahora, además, bajo `Sim_Cartesiano/`**. El `.gitignore` se reancló a patrones `**/` porque los anclados a `results/` dejaron de aplicar tras el archivado.

### Optimización del solver de presión — 5.4× por polar, ACTIVA POR DEFECTO

Documento completo en `docs/OPTIMIZACION_SOLVER.md`. Polar completa (6 ángulos, dx=0.002, dominio C) de **7.73 h a 1.43 h**, con la divergencia residual **por debajo** de la que daba la configuración anterior.

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

1. **El multigrid nunca alcanzaba su tolerancia** (`frac_converge = 0.0`), y la causa es de discretización, no del solver: malla colocada, y en el modo tablero el gradiente centrado se anula (símbolo de L = −4/h², de D·G = 0). **El 46.2 % de la divergencia residual es modo par-impar**, que ninguna iteración puede eliminar. `_build_projection_faces` no escapa: interpolación lineal pura, sin Rhie-Chow. **Esto es exactamente lo que el curvo resuelve** (ver F4: el operador **es** D·G y la divergencia par-impar baja a 1e-9).
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

### Criterio de parada — cuarta versión, la puerta de ruido es el CI95

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
   - **Pendiente de limpieza**: `_n_efectivo` sigue **definida dos veces** en `Sim_Cartesiano/Simulador2D.py` (líneas **7775 y 7797**, idénticas). Borrar una. Es lo único que quedó del aviso de agosto tras los commits.
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

**Validación externa contra XFOIL** (NACA0012 Re=1e5, Ncrit=9, airfoiltools; `validacion_xfoil.json`): **la sustentación es buena, la resistencia no.** MAE(Cl)=0.0912, pendiente dCl/dα=0.10211 vs XFOIL 0.11107 y teoría 2π 0.10966. MAE(Cd)=0.036, **error medio en Cd +123.8 %, monótono creciente con α** (−2.4 % a α=0 → +226 % a α=10). Causas plausibles: resolución de capa límite, separación/reataque prematuro de la burbuja, bidimensionalidad. **El L/D absoluto NO es comparable con XFOIL ni con experimento, y como el error de Cd depende de α tampoco es un factor de escala divisible.** Lo que sí se sostiene: Cl, pendiente de sustentación, y comparación perfil-contra-perfil a igual α, misma malla **y mismo dominio**. **El curvo baja ese +124 % de Cd a +15 %** (validación 1), así que el sesgo era del IBM cartesiano, no del modelo.

**Transición SA-BC implementada y validada** (`8e29bd5`): `transition_model="sa_bc"` + `freestream_Tu=0.1` (default `"none"`, requiere `turb_model="sa"`). Intermitencia algebraica de Bas-Cakmakcıoğlu 2016: γ = 1−exp(−√T1−√T2) con Re_θc por la correlación de Menter, modulando **solo la producción** de SA; `SA_BC_CHI1=0.002`, `SA_BC_CHI2=5.0`; γ en `Mesh.sa_gamma`. Validada en `results/1eraGranOptimizacion/agent_tests/sabc_*`. **Aún no portada al curvo**, y es lo que explica su −17 % de Cl.

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
- Fluctuación instantánea medida a dx=0.001: Cl **6.5 %**, Cd **15.5 %** — de ahí venía todo el ruido de estimador.
- **Estacionariedad por tramos** (cuatro tramos de t≈14 entre t=14 y t=71): L/D 27.01 / 27.15 / 27.18 / 27.05. El flujo es estadísticamente estacionario aunque nunca alcance estado estacionario.

**Ventana larga a dx=0.001** (`ventana_larga_dx001.py` → `results/verificacion_numerica/ventana_larga_dx001.json/.log/_series.npz`, `0263a13`). 192000 iteraciones de tope (t=71.0 físico alcanzado, cinco veces la corrida del GCI), **16.4 h de GPU**.

- **El criterio de la versión 3 no llegó a saltar** (`converged_clcd=false`, 192000 de 192000): esa fue la prueba que motivó la versión 4 (puerta de CI95). A esta resolución el perfil **no tiene estado estacionario**; las mallas gruesas sí convergen porque la difusión numérica amortigua el desprendimiento.
- **Cd: 0.025430 ± 0.000281 (CI95, 768 muestras) frente a 0.025507 con t=12 → 0.30 %, dentro del CI95.** El promediado temporal no era el problema: **la oscilación del Cd entre mallas es espacial y real**.
- **El Cl NO se movía un 6 %** — eso era el bug de fase del snapshot, corregido en `08af2c4`. La media temporal es 0.6878 con las dos ventanas.
- Queda un residuo real de dispersión entre estimadores sobre el mismo campo: Cl 0.6878 (media de superficie) frente a 0.6508 (circulación en lazo 0.15c) y 0.6419 (∮ΔCp en la cuerda) — **~7 %**, y `ΔCp_TE=−0.092` dobla la tolerancia. Los lazos grandes bajan a Cl_circ 0.51 (0.40c y 0.80c) y 0.29 (1.50c): **la circulación no cierra**. **La validación 2 del curvo apunta a que buena parte de esa caída con el radio del lazo es truncamiento del dominio**, no defecto del esquema.
- **No repetir esta corrida**: el `.npz` de series está commiteado y el análisis se rehace offline.

**Ghost cells del borde de salida entre mallas** (`te_report_mallas.py` → `results/verificacion_numerica/te_report_mallas.json`, ~1 min sin simular): dx=0.004 → 11 ghosts en el TE, dx=0.002 → 14, dx=0.001 → 15; **irreparables en el TE = 0 en las tres** (1 irreparable en todo el dominio a dx=0.002). **La distribución es suave en dx**, así que la hipótesis "las ghost-cells de extradós e intradós se solapan y refinar lo empeora" **no está respaldada por el conteo**: el error de discretización no salta por ahí.

### Estudio de independencia de dominio — el sesgo más grande de la campaña

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
- **En el curvo esto casi desaparece** (validación 2): recortar el arco C de 8.76 a 6.24 cuerdas mueve el Cl un 0.01 %. La sensibilidad al dominio del cartesiano es del IBM + fronteras `slip`, no de la física.

**¿Se conserva el ranking del GA al cambiar de dominio?** (`ranking_dominio.py` → `results/verificacion_numerica/ranking_dominio/`): top-1 y top-2 de la población final reevaluados con **el mismo presupuesto** (13000 iters, la campaña usó 10000/7750/5950 por el early-stop y con n distinto las barras no son comparables). top1 27.75 (fitness GA) → **31.81** en B; top2 27.23 → **29.51** en B. **El orden se conserva y la diferencia (2.30) supera el CI95 de cada uno (~1.1), pero por poco.** **Incompleto**: falta el top-3 y faltan las reevaluaciones en el dominio A con presupuesto igualado.

### Asintótico temporal y polares del dominio C

**Cortar en t≈10 sobreestima el Cd ~5 %** (`asintotico_alpha4_domC.py` → `results/asintotico_alpha4_domC/`). Ganador, α=4, dx=0.002, dominio C, llevado a **t=41.5** (52000 iters, 3.4 h): el L/D de la cola sube 24.13 (t=5) → 32.33 (t=10) → **34.05 (t=20) y ahí se queda plano** (34.03 / 34.13 / 34.09 / 34.05 hasta t=41.5). **Toda la corrección está en el Cd** (0.02680 → 0.02099 → 0.02003); el Cl apenas se mueve (0.6818 desde t≈12). Doblar de t=20 a t=40 mueve el L/D 0.01.

**Extrapolación a t→∞** (`extrapolacion_polar.py`): **Richardson no vale en el tiempo** — la cola es un transitorio que decae y luego mesetea, no una ley de potencias. Contrastado contra el único punto con verdad medida (α=4, valor real 34.084): Richardson sobre T=5/10/20 da 39.41 (**+15.6 %**), el ajuste exponencial `f(T) = F∞ − A·e^(−T/τ)` sobre T, 2T y 4T da 34.66 (**+1.7 %**). El ajuste solo aplica si la razón de incrementos cae en (0,1); si no (desprendimiento sin meseta, o serie cortada por el guardián de residual) se marca **n/a**.

**Polares disponibles, y NO son comparables entre sí punto a punto**:

| carpeta | dominio | α | t | notas |
|---|---|---|---|---|
| `results/verificacion_numerica/polar_{ganador_tfg2,naca0012}.*` | 8×5 (GA) | 0..10 de 2 en 2 | ~10 | con early-stop v3 |
| `results/polar_fina_1grado/` | 8×5 (GA) | 0..10 de 1 en 1 | ~10 | |
| `results/polar_fina_1grado_dom24x16/` | **C** | 1..10 de 1 en 1 | ~10 | el criterio no disparó ni una vez |
| `results/polar_2grados_dom24x16/` | **C** | 0..10 de 2 en 2 | ~20 | **early-stop desactivado**; α=8 y α=10 truncados por el guardián de residual |
| `results/richardson_polar_domC/` | **C** | 0..8 de 2 en 2 | ~20 | 3 mallas, paradas desactivadas |

Polar del ganador vs NACA0012 a dx=0.002 en el **dominio del GA** (L/D, α=0…10 de 2 en 2): 2.53 / 15.49 / **27.70** / 21.50 / 14.91 / 10.32 contra NACA 0.01 / 8.93 / 13.19 / 12.74 / 8.39 / 6.61.

Polar **corregida a t→∞ en el dominio C** (`figuras_analisis_polar.py` → `results/analisis_polar/`, columna `ld_fusion`; **no es simulación nueva**):

| α | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| ganador | 10.37 | 18.50 | 32.40 | **32.67** | 29.01 | 22.01 | 17.79 | 14.36 | 11.72 | 10.09 |
| NACA0012 | 5.46 | 9.63 | 13.43 | **14.38** | 13.19 | 12.60 | 10.17 | 7.10 | 5.51 | 4.19 |

Máximo en el punto de diseño (α=3–4) y ventaja en todo el rango: **mejora de banda ancha, no sobreajuste al ángulo**. La ventaja es ~2.3× a α=4. **El valor absoluto no es citable**; la relación entre los dos perfiles, medida en el mismo dominio, misma malla y misma ventana, sí.

### Richardson sobre la POLAR — el Cd no converge en malla (`835a874`)

> **SUPERADO por los resultados finales del 2026-08-26** (ver arriba). Se
> conserva porque su diagnostico sigue siendo valido y porque el estudio
> final lo confirma: el Cd no converge en malla. Lo que cambia es la terna
> (0.008/0.004/0.002 en vez de 0.004/0.002/0.001, por el colapso de dt a
> 0.001), el numero de angulos (9 en vez de 5) y que ahora hay una
> extrapolacion robusta que no produce Cl de 2.91 ni Cd negativos.
> **No citar de aqui el 34.04 ni las bandas 19-55 %.**

`scripts/agent_tests/richardson_polar_domC.py` → `results/richardson_polar_domC/`. Tres mallas (dx=0.004/0.002/0.001, r=2) × cinco ángulos (0, 2, 4, 6, 8) en el dominio C, **~48 h de GPU**.

| magnitud | monótonos | p medio | banda media entre mallas |
|---|---|---|---|
| Cl | **5/5** | 1.29 ± 0.32 | 26.5 % |
| Cd | 3/5 (rompe en α=4 y α=6) | 2.98 ± 1.73 | 39.2 % |
| L/D | 4/5 (rompe en α=4) | 2.28 ± 1.43 | 58.8 % |

- **El Cl es lo único citable**: monótono en los cinco ángulos. GCI fino **1.1–5.7 % fuera de α=0**. **α=0 es un caso aparte**: GCI fino 21 % — a sustentación casi nula el error relativo explota.
- **El Cd rompe justo en el punto de diseño**: a α=4 vale 0.02426 / 0.01999 / 0.02317 (no monótono, GCI fino 50 %).
- **El L/D lo hereda**: a α=4 da 27.62 / **34.04** / 30.17 — banda del **21 %**.
- **Conclusión operativa: ningún GCI de Cd ni de L/D es citable; solo la banda entre mallas (19–55 %).**

**El adelgazamiento del perfil NO es artefacto de malla** (`test_espesor_malla.py` → `espesor_vs_malla.json`). El GA lleva el espesor máximo del 12 % (semilla) a **~1.9–2.2 %** con el L/D subiendo monótonamente. Cuatro ganadores del mismo linaje reevaluados a dx=0.002:

| individuo | t_max | L/D dx=0.004 | L/D dx=0.002 | Cl | Cd |
|---|---|---|---|---|---|
| e0_GM15 | 6.13 % | 16.96 | 22.01 | 1.048 | 0.0476 |
| e3_GM15 | 3.46 % | 22.90 | 27.22 | 0.730 | 0.0268 |
| e6_GM15 | 1.91 % | 25.44 | 28.82 | 0.601 | 0.0209 |
| e6_s1014 | 2.13 % | 25.66 | **30.17** | 0.728 | 0.0241 |

**El orden se conserva** entre mallas dentro de ese linaje y la mejora solo se encoge de 1.513× a 1.371×. El adelgazamiento es física de Re bajo. **Matices**: el linaje es homogéneo, y el caso s1014/epoch9 (espesor máximo al 84 % de cuerda) demuestra que fuera de él la malla gruesa sí reordena.

Tres defectos que hacían inviable una corrida de días, corregidos en `5213410`:

1. **Fitness negativo indistinguible de un fallo.** `calcular_fitness` filtraba `ld > 0`, así que un perfil con L/D<0 devolvía 0.0 — el mismo valor que un crasheo. Ahora hay centinela explícito `Individuo.evaluado`.
2. **Reanudar costaba una generación entera**: el checkpoint guardaba el fitness pero no lo restauraba → 16 re-simulaciones, 3.1 h por pausa.
3. **La pausa no paraba nada**: RunGA absorbía el Ctrl+C y `_run_ga` no lo miraba.

Además: `parada_pedida()` acepta fichero centinela; **`guardar_memoria` ya no pisa el histórico** (volcaba `datos_X` tal cual, y una prueba de humo de 10 individuos se llevó 2347 experiencias); barra de progreso a 1 actualización/minuto sin TTY; flags `--dx` y `--t-target`.

### Memoria del TFG (`docs/tfg/`)

Pipeline propio: fuente en `memoria.txt` (marcas `#1..#4`, `$$latex$$`, `[FIG]/[TBL]/[TOC]/[PB]`, tablas `| a | b |`), `build_tfg.py` sustituye `word/document.xml` sobre la plantilla oficial EETAC `MaquetaTFG.docx`, `latex2omml.py` convierte el subconjunto de LaTeX usado a OMML (lanza `ValueError` si algo no está soportado, para que falle en generación y no en Word) y `acentuar.py`/`acentuar2.py` restauran tildes usando `/usr/share/dict/spanish`. Salida versionada: `TFG.docx`.

**El TFG está presentado** (`dd1b04f`, `fadfa60`, `562f858`). Lo que sigue en este repo es trabajo posterior a la entrega: el solver curvilíneo.

- `resultados_finales_v2.txt` (`99ffaa2`): texto de sustitución para los capítulos 4, 5 y 6, con una tabla de trazabilidad de cada número contra su fichero de resultados.
- **Tres apartados reescritos y compilados como extractos sueltos**: `3.4.2` difusión numérica del esquema de advección, `3.8.2` corrección de presión de fondo, `3.10.3` la evaluación CFD como función de fitness.
- `figuras/diagrama_solver.{excalidraw,png,svg}` — diagrama del solver.

**Estado del árbol (2026-09-17)**: rama **`Geometria_Ajustada`**, HEAD **`ecb8dc0`**,
**árbol limpio** salvo este `RESUMEN.md` y `RESUMEN.md.tmp`, que es artefacto del
hook y no se commitea.
Todo el bloque de agosto que estuvo pendiente doce días (recalibrado CI95,
`opt_solver.py`, estudio de dominio, polares del dominio C, resultados finales)
**ya está commiteado**, junto con la geometría arbitraria (`bff0838`), la GUI
(`c104ba1`), la reorganización en dos solvers (`bdca39b`) y el curvo entero
(`87b56fe`, `6ea1e7b`) con los cuatro pasos de la fase 1 de optimización
(`8e0e112`, `6a0f886`, `b2c9fe0`, `1918c80`) y el primero de la fase 2
(`ecb8dc0`, nivel grueso 30 → 4 barridos).
Queda por limpiar la definición duplicada de `_n_efectivo`
(`Sim_Cartesiano/Simulador2D.py:7775` y `:7797`).
`Sim_Cartesiano/cerebro_aerodinamico.pkl` sigue siendo el truncado por la prueba
de humo (74 experiencias frente a las 2347 de `8ab2461`) y mezcla genes de
longitudes distintas entre islas, así que el oráculo no lo carga — hay que
reconstruirlo, no confiar en el que hay.

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
- `wc_sa_a5_dx2_long` (48k iters, t=18.3, 2h GPU): Q=0.0032, **Cl converge en plateau 0.32**. **PROYECCIÓN EXONERADA: Q→0 no recupera el Cl.**
- **Diagnóstico del déficit (0.32 vs 0.55)**: SA sobre-difunde. BL en x/c=0.5 de espesor ~0.07c (3× placa plana turbulenta), χ=nu_t/nu~40-50 (teórico ~8), Cp_min=-0.81 (esperado ~-1.9).
- `wc_sa_a5_dx2_lam01` (sa_nu_tilde_factor=0.1): **PEOR** — Cl=0.266, LSB reapareciendo. **Falsificada la hipótesis fully-turbulent**; mantener `sa_nu_tilde_factor=3.0`.
- `wc_sa_a5_dx1` (dx=0.001): **Cl=0.389** (vs 0.32 a dx=0.002). **Camino resolución confirmado, error ~1º orden en dx.**
- **Causa dominante identificada: difusión numérica del SL bilineal, nu_num≈u·dx/2** (independiente de dt) — 1º orden, ~5-10× nu molecular cerca de pared a Re=1e5. A Re=1e3 era ~10 % de nu (por eso clavó el DNS).
- **MacCormack implementado** (`advection_scheme="maccormack"`, default "sl"): corrección Selle 2008 con limitador min/max del stencil y **banda de 2 celdas junto al sólido en SL puro**. Validación Re=1e3 α=5: Cl=0.245/Cd=0.155.
- `wc_mc_sa_a5_dx2`: **Cl=0.452** (SL daba 0.32), Cp_min=-1.20, Cd=0.040. **Cuarta capa (nu_num del SL) CONFIRMADA y resuelta.** Coste ~4 %.
- `wc_mc_sa_a5_dx1`: **Cl=0.496 ✓ OBJETIVO CUMPLIDO**, Cd=0.0335, Q=0.005. **Richardson dx→0: Cl≈0.54 — clava el físico 0.55.**
- **CONFIG DE REFERENCIA DEL OPTIMIZADOR (cartesiano)**: `wall_treatment="consistent"` + `turb_model="sa"` + `advection_scheme="maccormack"` (+ `transition_model="sa_bc"`). dx=0.002 para validación, dx=0.004 legítimo para explorar/rankear dentro de un linaje homogéneo.
- **POLAR VALIDADA** (`wc_mc_polar_a0/a2/a8`, dx=0.002): Cl(α)= 0(0.000), 2(0.185), 5(0.452), 8(0.708). Monotonía ✓, pendiente 0.088/deg ✓, simetría α=0 exacta ✓. **DIAGNÓSTICO DE 4 CAPAS CERRADO.**
- **Script de barrido CFL/dx/α**: `scripts/agent_tests/run_cfl_sweep.py` — 30 sims resumibles. **Aún no ejecutado.**

**Diagnóstico histórico cerrado (4 capas, evidencia en `results/1eraGranOptimizacion/agent_tests/`)**:
1. LSB laminar a Re≥1e4 (burst t≈5-7, colapso). Fix: **Spalart-Allmaras**. WALE inerte.
2. Pared sub-resuelta: SA necesita dx≤0.002 (y+≲10).
3. Inconsistencia IBM↔proyección: legacy absorbía masa (sobre-circulación 2.2×). **Resuelta con `wall_treatment="consistent"`**.
4. Difusión numérica del SL bilineal. **Resuelta con `advection_scheme="maccormack"`**.
- **Quinta capa (definición de fuerza sobre cuerpos ultrafinos)**: en el cartesiano sigue abierta. **En el curvo ya no aparece**: los tres estimadores de Cl cuadran al 1.8 %, Γ varía 6.1 % con el lazo, `ΔCp_TE = −0.025` sin parche de Kutta, y la validación 2 muestra que lo que queda de dispersión entre lazos es truncamiento del dominio. La vía para cerrarla es la malla body-fitted, no seguir instrumentando el IBM.
- Métrica decisiva: Q_lazo=∮u·n dl (`q_loop_025c/075c`).

**GA de optimización de perfiles (`scripts/RunGA.py`)**: `simular_perfil` corre sobre consistent+SA+MacCormack, `main(config)` reutilizable con retorno dict, población mixta multi-semilla, parada por `tiempo_limite_s`, mutación con sigma adaptativa, checkpoint de genes en `estado_ga.json` (escritura atómica). Orquestador `scripts/agent_tests/run_convergence_study.py` (deadline-driven, resumible). Memoria de ML persistente: `cerebro_aerodinamico.pkl` + `aprendizaje_ML.jsonl`.

**Estudio de convergencia (CERRADO, `a0ee4f3`)**: Arm A no converge (4 semillas → 4 óptimos locales, CV=0.172); el test aislado sin migración (28 h GPU) da dist-forma media **0.0557** (igual/peor que la partida 0.0507) → **la convergencia que producen las islas la fuerza la MIGRACIÓN, no una física de óptimo único**.

## Próximos pasos

**Prioridad (2026-09-18) — el foco es el solver curvilíneo**

Las dos fases de optimización están **cerradas y confirmadas sobre 8 000 pasos**:
167.0 → **83.94 ms/paso, ×1.99, 5.99 → 11.91 it/s**, con la peor fuerza movida un
**0.013 %** y la divergencia residual **bajando un 11.7 %**. Ver "Dónde se va el
tiempo" y la sección de la fase 2. Lo siguiente:

0. **Medir qué le hace la cola del TE a Cl/Cd.** Es lo único que queda abierto
   del cambio del 21-sep: la malla es buena y la geometría añadida está acotada y
   avisada, pero **el efecto en las fuerzas no está medido**. AG24 añade 0.39 % de
   cuerda, NACA_0012 0.96 %. Un par de corridas contra el caso afilado lo cierra.
1. **La presión es ahora la etapa más cara** (34.58 ms, 41.2 % del paso). Es un
   solo campo, así que no hay nada que fundir: el margen está en el número de
   iteraciones del PCG, no en los kernels.
2. **Momento con `correcciones` 1 → 0.** Quitaría la mitad de las resoluciones de
   momento sin tocar CUDA. El `punto_fijo.py` de la fase 1 ya demostró que el
   punto fijo de la corrección diferida no depende del número de iteraciones
   (orden 2.00 difusivo / 2.02 convectivo con 0, 1 y 2), pero cambia la física
   discreta y necesita su propia validación larga.
3. **Portar SA-BC al curvo.** Sigue siendo la prioridad de física. Es la causa
   medida del −17 % de Cl y del +15 % de Cd contra XFOIL: el SA es turbulento
   desde el borde de ataque y XFOIL transiciona en x/c=0.27–0.37. El modelo ya
   está implementado y validado en el cartesiano (`transition_model="sa_bc"`, γ
   algebraica de Bas-Cakmakcıoğlu modulando solo la producción); hay que
   trasladarlo a `curvo/turbulencia.py`. **No tocar el esquema antes de esto.**
4. **Cerrar F6: validación externa del cilindro.** Es lo único de la fase que
   queda. Referencia a tener presente: el cartesiano 2D daba Cd 1.5–1.67 frente a
   1.1–1.2 experimental (+25–40 %) con St 0.20 correcto, así que un Cd alto ahí
   no es "más real".
5. **Arreglar o retirar la tolerancia de
   `test_curvo_gpu.py::test_float32_da_la_misma_respuesta`** (3.32e-09 contra
   1e-09, `tests/test_curvo_gpu.py:300`). Es pre-existente —falla igual en
   `6ea1e7b`, comprobado con `git stash`— y es el único fallo de la suite; o se
   justifica el umbral con la discretización que se usa, o se sube con razón
   escrita.
6. **Tablero par-impar en el `Cp` de pared** (amplitud 0.11 cerca del TE y
   creciendo). Nace en `div(m*)` y se acumula en `p += phi`; no toca ni a `u`, ni
   a `v`, ni a las fuerzas integradas, pero ensucia el Cp que se publica. **No es
   el número de correcciones cruzadas ni los barridos del nivel grueso, ni la
   fusión de campos**: con 1 y con 2 correcciones, con 4 y con 30 barridos, y en
   serie o fundido, la fracción de tablero sale igual a tres cifras (1.334e-4).
7. **Limpiar `_n_efectivo` duplicada** en `Sim_Cartesiano/Simulador2D.py`
   (líneas 7775 y 7797, idénticas).

**Del cartesiano — abiertos, pero ya no son la línea principal**

8. **Actualizar `docs/verificacion_numerica.md`**, congelado desde el 14 de
   agosto: no incorpora la ventana larga, el fix del fitness, el estudio de
   dominio, el Richardson de la polar ni el estudio final.
9. **Investigar el colapso de dt a dx=0.001.** Único obstáculo para una terna
   apoyada en las mallas finas. La firma (dt cayendo ×6.8 sin estabilizar) apunta
   a la misma familia que el blowup del modo par-impar, que `warm_start_filtered`
   mitiga pero no cura. **El curvo no tiene ese problema** (el operador es D·G),
   así que puede que la respuesta correcta sea no arreglarlo.
10. **El exceso de Cd** (+226 % a α=10 frente a XFOIL) y su no convergencia en
    malla es la incertidumbre dominante que le queda al cartesiano. El GA
    converge a cuerpos de 2 % de espesor, así que un Cd sesgado sesga el óptimo
    de espesor. En el curvo ese sesgo ya baja a +15 % y se explica por transición.
11. **Terminar `ranking_dominio.py`**: falta el top-3 y las reevaluaciones en el
    dominio A con presupuesto igualado. Sin ellas no se puede afirmar que el
    sesgo de dominio sea uniforme entre perfiles, y de eso depende que la campaña
    del GA sea rescatable como *ranking*.
12. **Post-proceso de `tfg2`**: pasar `metricas_ga.py` sobre
    `results/convergence_study/islands_tfg2/` y **reconstruir
    `cerebro_aerodinamico.pkl`** — el de disco está truncado a 74 experiencias y
    mezcla genes de longitudes distintas entre islas; hay que resamplear a un
    grid común antes de reentrenar. Es la causa del error no fatal "inhomogeneous
    shape" que salía en cada arranque de isla.
13. **Activar el surrogate** (`usar_ia=True`, percentil 70): ahorro estimado del
    69 % del CFD perdiendo el 2.5 % de la élite. Reentrenar sobre los 1160
    registros nuevos de `tfg2`, no sobre el historial sesgado. Depende del 12.
14. **Si hay una tercera campaña de GA, el candidato natural es el curvo**, no el
    cartesiano en dominio C: el curvo no tiene sesgo de dominio (0.01 % en Cl),
    el Cd está a +15 % en vez de +124 %, la malla se genera en 0.02 s y va ya a
    9.63 it/s. Lo que falta para eso es el generador sobre geometrías del GA — ahí
    el 15.9 % de éxito al mallar mide la geometría (cúspides con radio de morro
    mediano 3e-4 c), no el generador.
15. **GA multipunto α ∈ {2,4,6}**: etapa `multipunto` ya cableada en
    `cola_tfg.sh`, fuera de la cola por defecto.
16. **Ejecutar el barrido** `scripts/agent_tests/run_cfl_sweep.py` (30 sims,
    resumible) — sigue pendiente, ninguna sim lanzada.
17. Actualizar rutas en scripts que apunten a `results/convergence_study/...` o
    `results/verificacion_numerica/{gci,polar,reeval_*}.json` — los de la campaña
    vieja viven bajo `results/1eraGranOptimizacion/` y **todo el árbol del
    cartesiano cuelga ahora de `Sim_Cartesiano/`**. **Cuidado**:
    `results/convergence_study/` contiene la campaña NUEVA (`islands_tfg2/`), así
    que una ruta antigua ya no falla: apunta a datos distintos.
18. `polar_results.json` en la raíz de `Sim_Cartesiano/` lo escribe `main()` en
    cada corrida suelta y se ensucia con puntos de verificación. No es una polar
    curada; no leerlo como resultado.

**Cerrados, no rehacer:**

- ~~Paso 1 de la fase 1~~ — hecho (`8e0e112`), bit a bit idéntico.
- ~~Paso 2: `correcciones` de momento 2→1~~ — hecho (`6a0f886`), −14.6 % de tiempo
  con ΔCl +0.001 % y ΔCd −0.000 %. **No rehacer el barrido de orden desde frío**:
  esa medida es la que engañó, el punto fijo es el mismo con 1 y con 2.
- ~~Paso 3: `correcciones_p` de presión 2→1~~ — hecho (`b2c9fe0`), acumulado
  162.5 → 127.8 ms con ΔCl +0.000 % y ΔCd −0.003 %. **No volver a levantar la
  hipótesis del tablero**: se midió y la fracción par-impar en `p`, en la
  velocidad y la oscilación del Cp en el TE son idénticas con 1 y con 2.
- ~~Paso 4: SA `correcciones` 1→0~~ — hecho (`1918c80`), acumulado 162.5 →
  116.5 ms (−28.3 %, 8.58 it/s) con ΔCl +0.001 %, ΔCd −0.002 % y `nu_t/nu`
  +0.008 %. **No decir que el SA quedó en upwind de 1.er orden**: con 0 el bucle
  sigue evaluando la corrección diferida una vez sobre el campo anterior, es
  Picard retrasado un paso. Y **no rehacer el barrido de punto fijo**: con
  `corr ∈ {0,1,2}` los errores son idénticos a seis decimales en las tres mallas
  y en los dos regímenes, con orden 2.00 / 2.02.
- ~~Fase 2, paso 1: barridos del nivel grueso 30→4~~ — hecho (`ecb8dc0`),
  115.64 → 103.81 ms (×1.11, presión ×1.20) con las fuerzas dentro del 0.005 %.
  **No rehacer el barrido de `grueso`**: en convección el historial de residuos
  es idéntico con 1, 2, 4, 8 y 30, y en presión el factor del PCG mejora al bajar
  (0.067 → 0.021). Y **no leer el +379 % del máximo de divergencia como
  regresión**: el tablero no se mueve (7.848e-5 → 7.843e-5).
- ~~Bajar `minimo` (profundidad de la jerarquía)~~ — medido de 8 a 5 niveles
  (`minimo` 9 → 576): sin tendencia clara de tiempo. Descartado, el coste es por
  lanzamiento y no por número de niveles.
- ~~Fusión ingenua de `u` y `v` con `float d[nc]`~~ — medida y descartada: ×0.73
  por derrame a memoria local. La vía buena son escalares `d0`, `d1`.
- ~~Commitear el bloque de agosto~~ — hecho, el árbol está limpio.
- ~~Escribir los capítulos 4-6 del TFG~~ — el TFG está presentado.
- ~~Rehacer dx=0.004 y 0.002 del GCI del ganador con el Cl promediado~~ — el
  estudio final recalcula los 72 puntos de cero con la misma definición de fuerza
  en todas las mallas.
- ~~Dar bandas en vez del 28.88 y el 34.04~~ — sustituidos por los valores de
  `resultados_finales/`, con extrapolación robusta y GCI por magnitud.
- ~~Estudio de convergencia del GA~~ — cerrado, la convergencia la fuerza la
  migración. No relanzar corridas de diagnóstico de la misma pregunta.
- ~~El reparto de ghost-cells del TE como mecanismo del error de fuerza~~ —
  descartado con el conteo (11/14/15, 0 irreparables en las tres mallas).

## Tests

**Del curvo (2026-09-17)**:

- `.venv/bin/python -m pytest tests/ -q` — 139 tests, 4–6 min, 15 piden GPU.
  **138 pasan**; el fallo de `test_curvo_gpu.py::test_float32_da_la_misma_respuesta`
  (`3.32e-09 < 1e-09`) es pre-existente en HEAD — verificado con `git stash`,
  falla idéntico sin los cambios de la cola del TE.
- Los tests de GUI piden `QT_QPA_PLATFORM=offscreen` si no hay pantalla.
- **Se sustituyó el test canario `test_perfiles_que_solo_se_mallan_en_etapa_1[AG24]`**,
  que afirmaba que AG24 *debe* plegar y decía explícitamente que tenía que fallar
  cuando alguien lo arreglase. En su sitio hay tres tests que fijan el
  **mecanismo**, no el síntoma: que sin cola las celdas plegadas están en el TE y
  no en el morro, que la cola no mueve ningún punto del perfil, y que un TE
  afilado sale bit a bit idéntico con cualquier `cola_te`.
- `.venv/bin/python -m pytest tests/test_curvo_proyeccion.py -q` — **23/23 en
  50 s**. Es la puerta que hay que pasar al tocar `correcciones_p`.
- **El plan de la fase 2 menciona un `tests/test_curvo_multigrid.py` que no
  existe.** Al tocar `multigrid.py` la cobertura real son `test_curvo_gpu.py`,
  `test_curvo_conveccion.py` y `test_curvo_proyeccion.py`; si se quieren tests de
  multigrid aislados hay que escribir el fichero.
- **El test de orden de `test_curvo_conveccion.py` usa 8 correcciones a propósito**:
  mide desde frío, donde el número de correcciones sí marca el orden. El solver de
  producción reanuda del paso anterior y con **1 — y también con 0** llega al mismo
  punto fijo (razón de errores 1.000000 en 24/48/96, orden 2.00 difusivo y 2.02
  convectivo). No "arreglar" el test bajándolo.
- `validacion/v1_naca0012_re1e5_a5/correr.py <carpeta> [clave=valor ...]` — corre
  un caso completo de validación en la carpeta que se le pase, aceptando
  overrides por línea de órdenes (`distancia_lejos=6.0 x_salida=18.39` es lo que
  generó v2). Escribe `caso.json`, `malla.npz`, `historia.npz`, `pared_final.npz`,
  `resultados.json`, `reparto.json` y `figuras/malla.png`. **Lanzar siempre con
  `PYTHONPATH=.` desde la raíz.**
- `validacion/<caso>/analizar.py` — las figuras y el análisis del caso; límites de
  vista y número de celdas salen de `caso.json`, no cableados. `malla_dominio()`
  dibuja la extensión real del dominio.
- `bench_0b.py` — micro-benchmark de `cv.sistema` y `jerarquia` por paso; es de
  donde salió el techo de 3.8 % que motivó el paso 1.
- `punto_fijo.py` (scratchpad) — barrido de `correcciones ∈ {0,1,2}` **encadenando
  llamadas que reanudan**, sobre la solución manufacturada, en los dos regímenes y
  las tres mallas. Es la medida que cerró los pasos 2 y 4; la versión desde frío
  es la que engaña.
- **Protocolo del A/B de optimización**: 800 pasos del caso v3, mismo `dt` y misma
  malla, comparando `historia.npz` columna a columna y el `reparto.json`. Mirar la
  **mediana** de la divergencia, no su máximo: el máximo salta con el instante en
  que cae el último paso y ya produjo un falso +199 % (paso 2) y un falso +379 %
  (`ecb8dc0`). Y cuando la divergencia suba de verdad —como en los pasos 3 y 4—,
  **medir el modo par-impar** (fracción de tablero en `p` y en la velocidad,
  oscilación del Cp de pared en el TE) antes de llamarlo degradación: son cosas
  distintas. **Los totales solo son comparables dentro del mismo A/B**: entre
  sesiones el mismo código ha medido entre 107.7 y 116.5 ms.

**Del cartesiano, del estudio final (2026-08-26)** — ya ejecutados,
`Sim_Cartesiano/resultados_finales/`:

- `scripts/agent_tests/resultados_finales.py` — runner completo: 72 puntos,
  reanudable por punto, Richardson e informe. `--solo-analisis` y
  `--solo-figuras` rehacen sin simular. **~10 h GPU, ya ejecutado; no relanzar.**
- `scripts/agent_tests/rf_comparar_ternas.py` — compara 0.008 vs 0.006 sobre el
  mismo conjunto de casos. Sin GPU, segundos.
- `scripts/agent_tests/rf_richardson_figuras.py` — las 6 figuras individuales de
  Richardson, líneas suavizadas con PCHIP. Sin GPU.
- `scripts/agent_tests/rf_figuras_extra.py` — Cp de superficie, polar de
  resistencia, malla y comparativa extrapolada. Sin GPU.
- `scripts/agent_tests/rf_progreso.py` — visor de progreso por memoria
  compartida; funciona con el runner redirigido a fichero.

**Anteriores**:

- `scripts/agent_tests/richardson_polar_domC.py` — Richardson de tres mallas × cinco ángulos en el dominio C → `results/richardson_polar_domC/`. **~48 h GPU, ya ejecutado y commiteado (`835a874`); no relanzar.** Reanudable por punto.
- `scripts/agent_tests/te_report_mallas.py` — ghosts e irreparables en el TE para las tres mallas. ~1 min, sin GPU real.
- `scripts/agent_tests/criterio_ci95.py` — recalibración del criterio de parada con puerta de CI95 sobre N efectivo; importa la función del solver y la evalúa sobre series guardadas (sin GPU). **Sustituye a `criterio_parada.py`.**
- `scripts/agent_tests/estudio_dominio.py [--solo A,E] [--analisis]` — independencia de dominio, 7 casos a dx=0.002. **No se lanza solo.** ~5 h GPU.
- `scripts/agent_tests/bench_dominio.py --lx 12 --ly 8 --cx 3` — coste por iteración vs tamaño de dominio, un caso por invocación (subproceso limpio).
- `scripts/agent_tests/ranking_dominio.py [--analisis]` — ¿el orden del GA sobrevive al cambio de dominio? **Incompleto: falta el top-3 y el dominio A.**
- `scripts/agent_tests/asintotico_alpha4_domC.py [--analisis]` — α=4 del ganador hasta t≈41.5 en dominio C. 3.4 h GPU, ya ejecutado.
- `scripts/agent_tests/polar_fina.py` / `polar_fina_dominio_c.py` / `polar_2grados_dom24x16.py` / `polar_naca0012.py` — polares reanudables y pausables con `STOP_SIMULATION.trigger`.
- `scripts/agent_tests/extrapolacion_polar.py` — extrapola cada punto a t→∞ con ajuste exponencial sobre T/2T/4T (Richardson **no** vale en el tiempo). Marca n/a donde no hay meseta.
- `scripts/agent_tests/figuras_analisis_polar.py` — figuras del análisis de las dos polares del dominio C.
- `scripts/agent_tests/ventana_larga_dx001.py` — ganador a dx=0.001, tope 192000 iteraciones. **16.4 h GPU, ya ejecutado; no relanzar.**
- `scripts/agent_tests/gci_ganador.py` — GCI de Roache de tres mallas sobre Cl, Cd y L/D; reanudable, cachea cada malla.
- `scripts/agent_tests/test_espesor_malla.py` — reevalúa a dx=0.002 ganadores de distintas épocas de una isla.
- `scripts/agent_tests/gran_optimizacion.sh {start|run|stop|status|snapshot|log}` — segunda gran optimización (islas `tfg2`). **Campaña terminada**; queda como plantilla.
- `scripts/agent_tests/cola_tfg.sh {run [etapa]|status}` — cola serie reanudable. `cola_final.sh`, `cola.sh`.
- `scripts/agent_tests/verificacion_numerica.py --calibrar|--reeval|--gci|--polar|--repolar|--validacion` — etapas sueltas; `simular()`, `criterio_calibrado()` y `T_TARGET_CASO` viven aquí.
- `scripts/agent_tests/series_dx004.py` — 20 series completas a dx=0.004 sin parar.
- `scripts/agent_tests/revalidar_ranking.py --n 30` — muestreo estratificado del historial y correlación de rangos; cacheado y reanudable.
- `scripts/agent_tests/metricas_ga.py` — post-proceso del GA sin GPU (~1 min): 15 figuras + `INFORME.md`. **Pendiente de pasar sobre `tfg2`.**
- `scripts/agent_tests/equivalencia_bit.py` — compara `u`, `v`, `p`, `solid`, `cl`, `cd` con `np.array_equal` contra el `Simulador2D.py` de HEAD. **Es la puerta buena**, no el `--compare` del 0.5 %.
- `scripts/agent_tests/coste_salida.py` — coste por evento de cada vía de salida → `results/coste_salida/INFORME.md`.
- `scripts/agent_tests/demo_gui.py` — escenas de ejemplo para la interfaz → `results/gui_demo/`.
- `scripts/agent_tests/run_kutta_tests.py {smoke|smoke_polygon|baseline_a5|polygon_a5|compare|regression}` — runs coarse (dx=0.004, ~5-15 min) con criterios cuantitativos.
- `scripts/agent_tests/run_cfl_sweep.py` — barrido CFL/dx/α resumible (pendiente de ejecutar).
- `pytest Sim_Cartesiano/tests/` — 32/32 con la geometría arbitraria dentro.
- `docs/tfg/build_tfg.py [--plantilla RUTA] [--salida RUTA]` — regenera `TFG.docx` desde `memoria.txt` sobre la plantilla EETAC (sin GPU).

## Diagnósticos clave

**Del curvo** (`curvo/fuerzas.py`):
- `estimadores_de_cl(met, u, v, p, info, nu, U, alfa, bc=..., corte=...)` — devuelve los tres estimadores a la vez (superficie, ∮ΔCp, circulación) más `ΔCp_TE`. **Que cuadren entre sí es el criterio de F5**, y a Re=1e5 con SA cuadran al 1.8 %.
- Γ por lazos de radio creciente: la variación con el radio es **6.1 %**, y la validación 2 muestra que lo que queda escala con el tamaño del dominio.
- `calidad(..., perfil=...)` — veredicto de malla con avisos. **`perfil` va como argumento con nombre**, no posicional.
- `Solver(..., cronometro=True)` + `s.reparto()` — reparto del paso en % y ms, con sincronización de GPU antes de cada lectura. Es la medida de referencia de los A/B de optimización.
- **Fracción de tablero en `p` y en la velocidad, y oscilación del `Cp` de pared en el TE** — el diagnóstico que separa "más divergencia residual" de "modo par-impar creciendo". Valores de referencia sobre el campo final del A/B: 7.8e-5, 1.03e-6 y 8.837e-2, estables entre la base, los pasos 2 y 3 y el nivel grueso de `ecb8dc0`.
- **`nu_t/nu`** — la guardia del paso 4: con `correcciones=0` en SA se mueve +0.008 % sobre 800 pasos de v3. Si alguna vez se dispara, es ahí donde se ve primero.
- **Factor de convergencia del ciclo V / PCG** (`mg.resolver`, `mg.resolver_pcg`) — la guardia al tocar `grueso` o la jerarquía: momento converge en 15 ciclos con `grueso` de 1 a 30, presión da factor 0.020–0.067. Puerta: < 0.3.

**Del cartesiano** (métodos de `Mesh`):
- `compute_circulation()` — Γ en lazos, Cl_circ=-2Γ/(U·c). **Con el ganador de `tfg2` los lazos dan 0.65 / 0.51 / 0.51 / 0.29 (0.15c → 1.50c): la circulación no cierra.**
- `compute_cp_diagnostics(mu, rho)` — Cl(∮ΔCp), ΔCp_TE, pico de succión.
- `compute_kutta_dcp_instant(face_data)` — ΔCp_TE instantáneo para el monitor.
- `debug_te_report()` — ghosts/irreparables cerca del TE.
- `plot_streamlines()`, `plot_forces_over_time()`, `plot_cp_vs_chord()`.
- Series por muestra: `tvector` (tiempo físico; el dt es adaptativo y el tiempo no se reconstruye desde el índice), `clvector`/`cdvector`, `resvector`, `n_muestras_validas`.
- Del retorno de `simular_perfil`: `cl_ci95`/`cd_ci95`/`ld_ci95` y `n_samples` (**comparar cambios entre corridas contra el CI95 antes de llamarlos reales**), `converged_clcd`, `t_conv_clcd`, `iters_efectivas`, `cl_cp_discrepancy` + flag, **`cl_audit_inst`** (diagnóstico, **no** entra en el L/D desde `08af2c4`), `dump_series_path`, `dump_field_path`.
- Diagnósticos de frontera leídos del campo final: `eps_top/eps_bot` (bloqueo medido en la pared slip), `v_in_max`, `u_out_min` (estela viva en la salida si ≪1), `cp_wall_range`.

## Config de referencia

**Curvo (producción)**: `generar_c(px, py)` a secas — `DN_PARED=8.0e-5`,
`N_CAPAS_PARED=15`, `CRECIMIENTO_PARED=1.0`, **`COLA_TE=4.0`**, 99×384 = 37 534
celdas, y⁺ máximo 0.80 a Re=1e5. Con perfil de base roma la malla crece un poco
(AG24: 99×396 = 38 710) y `info["cola_rel"]` dice cuánta cola se ha añadido. `Solver(X, Y, info, nu=1e-3, alfa=5.0, dt=5e-3, xp=cp,
dtype=np.float32)`, SA activo, t\*=20. Defaults del solver tras la optimización:
**`correcciones=1`** (momento), **`correcciones_p=1`** (presión),
**`avanzar_sa(..., correcciones=0)`** (SA) y **`ciclo_v(..., grueso=4)`** — en los
tres primeros el punto fijo es el mismo que con 2 (pasos 2, 3 y 4), el 0 de SA
**no** es upwind de 1.er orden sino la corrección diferida retrasada un paso, y el
nivel grueso con 4 barridos da el mismo historial de residuos que con 30 (`ecb8dc0`).
**float32 obligatorio** en la 3070 Ti. Rendimiento actual: **103.81 ms/paso,
9.63 it/s** sobre la malla de v3, desde los 167 ms de partida.
Dominio: el arco C por defecto (~26×17.5) y el recortado (24×12.5) dan lo mismo
en fuerzas al 0.01 %, así que el tamaño se elige por coste, no por sesgo.

**Cartesiano (congelado)**: Lx=12, Ly=8, dx_min=0.001, CFL=0.25,
NACA_0012_sharp, `min_te_height_factor=1.0`, turbo_hd, wake long_fine_x,
`wall_treatment="consistent"`, `turb_model="sa"`, `transition_model="sa_bc"`,
`advection_scheme="maccormack"`. Parada por convergencia con los defaults nuevos
(`clcd_tol_drift=0.005`, `clcd_tol_ci95=0.02`, `clcd_window_conv_time=1.0`
mínima y auto-ensanchable, `clcd_min_t_fisico_before_check=5.0`; el fichero
calibrado baja `tol_drift` a 0.002 y sube `n_sostenido` a 3).

**Dominio del cartesiano**: para polares y simulaciones sueltas usar **C — Lx=24,
Ly=16, cx=6**, no el 8×5 del GA. Manda Lx (estela), no Ly. **En estudios de malla
o de ventana, desactivar las DOS paradas** (criterio de L/D y guardián de
residual `stop_on_convergence`).

**Nota**: el bloque `__main__` de `Simulador2D.py` puede quedar como scratch de
pruebas entre sesiones — usarlo como plantilla, no copiarlo literal.

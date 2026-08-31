# Resultados finales del TFG — verificación de malla y comparación de perfiles

**Estos son los resultados definitivos que se presentan.** Cerrados el
2026-08-26, 72/72 puntos completos.

Inventario de las 744 figuras en [`FIGURAS.md`](FIGURAS.md).
Tabla de GCI y avisos al citar en [`richardson/INFORME.md`](richardson/INFORME.md).

## Qué se calcula

| | |
|---|---|
| Perfiles | **Ganador de la campaña de optimización** y **NACA 0012 sharp**, la semilla de la que salió |
| Mallas | dx = 0.008 / 0.006 / 0.004 / 0.002 |
| Ángulos | α = 0 a 8 °, **de grado en grado** (9 ángulos) |
| Dominio | 24 × 16 cuerdas, perfil en cx = 6 (dominio C validado) |
| Régimen | Re = 1·10⁵, CFL = 0.5, t ≈ 20 en las cuatro mallas |
| Turbulencia | Spalart-Allmaras con transición SA-BC, Tu = 0.1 % |
| Puntos | 2 perfiles × 4 mallas × 9 ángulos = **72** |

Los dos perfiles tienen el mismo número de puntos y el mismo tratamiento del
borde de salida, así que la comparación entre ellos no arrastra diferencias de
discretización geométrica.

**Ojo con el nombre del ganador.** El fichero se llama `NACA_0012_sharp_winner`
porque esa fue la familia semilla de la que descendió, no porque sea un NACA. Y
**no** se le puede llamar "AG24": `AG24` es un perfil real de catálogo (AG24
Bubble Dancer DLG, de Mark Drela) que se usó como otra semilla del GA y compitió
contra él.

## La terna del GCI: 0.008 / 0.004 / 0.002

Razón de refinamiento **r = 2 constante** en los dos saltos, que es lo que pide
el procedimiento estándar (Roache; Celik et al. 2008).

La malla de **0.006 se calculó como terna alternativa de contraste y perdió**,
sobre los mismos 54 casos:

| terna | monótonas | p en rango | GCI mediano | GCI p90 |
|---|---|---|---|---|
| **0.008 / 0.004 / 0.002** (r = 2, 2) | **47/54** | **35/54** | **20.1 %** | **63.5 %** |
| 0.006 / 0.004 / 0.002 (r = 1.5, 2) | 46/54 | 28/54 | 44.2 % | 111.8 % |

Datos en `richardson/comparativa_ternas.json`, reproducible con
`scripts/agent_tests/rf_comparar_ternas.py`.

Por qué pierde el 0.006: rompe la simetría del NACA 0012 más que ninguna otra
malla (Cl = +0.0184 en α = 0, frente a +0.0040 del 0.008), rompe la monotonía en
α = 8 del ganador en las tres magnitudes a la vez, y con r = 1.5 el salto
0.006→0.004 es pequeño y la diferencia queda cerca del ruido.

## Por qué NO está la malla de 0.001

El plan original era 0.004 / 0.002 / 0.001, y era mejor por el lado teórico. Se
cayó por un motivo numérico, no de presupuesto: **a dx = 0.001 el paso de tiempo
colapsa.**

| dx | t_final alcanzado | dt medio |
|---|---|---|
| 0.004 | 15.9 – 19.6 | 1.4e-3 |
| 0.002 | 15.7 – 21.7 | 8.0e-4 |
| **0.001** | **5.2 – 6.9** | **1.2e-4** |

De 0.004 a 0.002 el dt baja ×0.57, lineal con la malla. De 0.002 a 0.001 baja
×0.15: sobra un factor 3.4. Las 52000 iteraciones se quedaban en t ≈ 6 en vez de
t ≈ 20, o sea que el Richardson habría comparado t = 6 contra t = 20.6 — el punto
fino aún en transitorio de arranque — y habría medido esa diferencia temporal
creyendo medir discretización espacial.

Además el dt no era solo pequeño: **seguía cayendo** durante todo el run (×6.8
sin estabilizar) mientras que a dx = 0.002 es plano. Es la misma firma que
precedía al blowup; `warm_start_filtered` evitó el NaN pero no curó la
inestabilidad de fondo.

Eso explica sin invocar física todo lo raro de esa pata: CI95 del 95 % en el L/D,
L/D no monótono en α = 2 y 4, Cd inflado (0.030 frente a 0.017) y el criterio de
convergencia que no disparaba nunca. El Cl **medio** sí coincidía con el de
dx = 0.002 (0.70 frente a 0.6977): lo que se rompía era la dispersión y el Cd.

Llevar dx = 0.001 hasta t = 20 pedía ~175000 iteraciones, 11 h por punto y 80 h
los que faltaban. Los 4 puntos que se llegaron a calcular están archivados en
`_archivo_malla_fina_descartada/` como evidencia del diagnóstico.

## Configuración del solver

Las cuatro mallas llevan **exactamente las mismas** optimizaciones y el mismo
presupuesto de proyección (`mg_max_outer=2, mg_cycles_per_outer=3`). Mezclar
configuraciones dentro de la terna sesgaría el orden aparente, que es justo la
señal que el Richardson mide.

```
warm_start   warm_start_filtered   coarse_mask_majority   fast_masks   interp_float32
```

**`warm_start_filtered` es lo que hace posible esa uniformidad.** Sin él,
`warm_start` con presupuesto 2×3 revienta por blowup de velocidad: iteración 1340
a dx = 0.004 y ~1740 a dx = 0.001, reproducible en todos los ángulos.

La causa no era la magnitud de la semilla sino su **estructura**: arrastraba el
modo par-impar (tablero) acumulado, que el multigrid no reduce porque en malla
colocada lo ve casi como núcleo. Un paso de Jacobi ponderado con ω = ½ lo aniquila
antes de reusarla (es un aniquilador exacto de ese modo e identidad sobre los
suaves). Coste nulo: 387 s frente a ~405 s sin `warm_start`. Precio en exactitud:
−0.25 % en el L/D del punto de diseño.

Se probó primero reescalar la semilla por el cambio de dt, y **no funcionó**:
solo movió el fallo de la iteración 1340 a la 1800. Ese fallo es lo que demostró
que el mecanismo era estructural.

## Extrapolación robusta — leer antes de citar

Richardson puro se dispara cuando el orden observado tiende a 0, porque el factor
de amplificación es `1/(r^p − 1)`. Casos medidos en este estudio:

| caso | p obs | amplificación | extrapolado | malla fina |
|---|---|---|---|---|
| Cl ganador α = 6 | −0.05 | **×27.5** | 2.912 | 0.879 |
| Cd ganador α = 6 | −0.10 | — | **−0.0153** | 0.0357 |
| Cd NACA α = 0 | −0.20 | — | **−0.0116** | 0.0143 |

Dos correcciones, ambas en `verificacion_numerica.extrapola_robusto()`:

1. **Umbral `p ∈ [1, 4]`.** Por debajo de p = 1 la amplificación supera 1, o sea
   que el extrapolado se aleja de la malla fina más que el salto entero entre las
   dos mallas más finas: deja de ser corrección. Con el umbral en 0.5 el Cd del
   ganador en α = 3 colaba con p = 0.762, bajaba un 37 % e **inventaba un pico de
   L/D en α = 3 que ninguna de las tres mallas tiene**. Fuera de rango se fuerza
   p = 2, el nominal del esquema.
2. **El Cd se ajusta sobre `log(Cd)`**, así el extrapolado sale de un `exp()` y
   no puede cruzar cero.

`orden_con_signo()` expone además el signo del orden observado, que la fórmula de
Celik devuelve en valor absoluto — y ese `abs()` escondía justo el caso que más
importa detectar, `p < 0`, que significa que los saltos **crecen** al refinar.

`gci_triplete()` y `orden_observado()` **no se tocaron**, para que los estudios
ya publicados sigan dando exactamente lo mismo.

## El resultado

Punto de diseño **α = 4**, donde las cuatro mallas ponen el máximo de L/D:

| | Cl fina | Cl extrap | Cd fina | Cd extrap | L/D fina | L/D extrap |
|---|---|---|---|---|---|---|
| Ganador | 0.6965 | **0.6999** | 0.01692 | 0.01541 | 41.15 | **45.4** |
| NACA 0012 | 0.4978 | **0.5103** | 0.02746 | 0.02330 | 18.13 | **21.9** |

El Cl del ganador en α = 4 es el caso mejor portado de todo el estudio: p = 2.56
y GCI del **0.61 %**.

**La ventaja del ganador se mantiene en los ocho ángulos con L/D significativo,
en las cuatro mallas y también extrapolada.** Es el resultado más robusto: no
depende de la malla ni del método de extrapolación.

## Estructura

```
resultados_finales/
  FIGURAS.md             inventario de las 744 figuras
  README.md              este fichero
  ESTADO.json            progreso legible por máquina
  PROGRESO.txt           progreso legible por humano
  metricas_todas.csv     todas las métricas de los 72 puntos, en plano
  runner.log             traza del runner
  perfiles/              los dos .dat usados + figura de geometría
  ganador_ag/  naca0012/
    metricas/            polar_dx*.json — 28 claves por punto
    series/              .npz con t, Cl(t), Cd(t), L/D(t) y residuos
    campos/              .npz con x, y, u, v, p y máscara de sólido
    figuras/campos/      velocidad y streamlines en zona fina Y dominio completo,
                         presión, vorticidad, vectores  (7 tipos × 36 puntos)
    figuras/historia/    fuerzas, eficiencia y residuos frente al tiempo
    figuras/polar/       Cl, Cd, L/D y Cl-Cd frente a α, cuatro mallas superpuestas
  richardson/            richardson.json, richardson_terna_alt.json,
                         tabla_gci.csv, comparativa_ternas.json, INFORME.md,
                         figuras/ (9)
  comparativa/           Cp de superficie, polar de resistencia, malla,
                         los dos perfiles en malla fina y extrapolados
  _archivo_malla_fina_descartada/   dx=0.001, NO usar (ver su LEEME.md)
  _archivo_sin_warmstart/           tanda anterior, histórico
```

Los `.npz` de campos no se versionan (≈2 GB); todo lo demás sí.

## Cómo reproducirlo

```bash
./lanzar_resultados_finales.sh            # primer plano, con barra de progreso
./lanzar_resultados_finales.sh --fondo    # desatendido, a resultados_finales/runner.log
```

Coste medido por punto: 189 s a dx = 0.008, 251 s a 0.006, 376 s a 0.004 y
1038 s a 0.002. Los 72 puntos son ~10 h de GPU.

**Ver el avance** desde otra terminal:

```bash
.venv/bin/python scripts/agent_tests/rf_progreso.py          # refresco continuo
.venv/bin/python scripts/agent_tests/rf_progreso.py --once   # una foto y sale
```

Muestra dos niveles: cuántos de los 72 puntos van hechos con la GPU consumida y
lo que queda, y en qué iteración va la simulación en curso con su Cl y Cd
instantáneos — esto último leyendo la memoria compartida que publica el
simulador, así que funciona aunque el runner esté redirigido a un fichero.

**Parar** limpiamente entre puntos (el punto en curso termina primero):

```bash
touch PARAR_ESTUDIO.trigger
```

**Reanudar**: volver a lanzar. Cada punto se escribe en cuanto termina y los ya
hechos se saltan por fichero.

## Rehacer solo el análisis o las figuras

Las horas de GPU no se repiten para retocar una leyenda:

```bash
.venv/bin/python scripts/agent_tests/resultados_finales.py --solo-analisis
.venv/bin/python scripts/agent_tests/resultados_finales.py --solo-figuras
.venv/bin/python scripts/agent_tests/rf_richardson_figuras.py   # las 6 individuales
.venv/bin/python scripts/agent_tests/rf_figuras_extra.py        # Cp, polar Cl-Cd, malla
.venv/bin/python scripts/agent_tests/rf_comparar_ternas.py      # 0.008 vs 0.006
```

`analisis()` corre **antes** que las figuras: el análisis es el resultado y las
figuras son cosmética. Al añadir dx = 0.008 un `KeyError` de color tumbó el cierre
de un estudio entero y dejó el `richardson.json` sin escribir.

## Ficheros de código

| fichero | qué hace |
|---|---|
| `scripts/agent_tests/resultados_finales.py` | configuración, runner, reanudación, Richardson e informe |
| `scripts/agent_tests/rf_figuras.py` | campos, historias, polares y convergencia |
| `scripts/agent_tests/rf_richardson_figuras.py` | las 6 figuras individuales de Richardson |
| `scripts/agent_tests/rf_figuras_extra.py` | Cp de superficie, polar de resistencia, malla, comparativa extrapolada |
| `scripts/agent_tests/rf_comparar_ternas.py` | compara las dos ternas candidatas |
| `scripts/agent_tests/rf_progreso.py` | visor de progreso |
| `scripts/agent_tests/verificacion_numerica.py` | `gci_triplete`, `extrapola_robusto`, `orden_con_signo` |

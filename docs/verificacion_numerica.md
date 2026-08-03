# Verificación numérica del solver

Fecha: 2026-08-02, revisado 2026-08-03. Datos y figuras en `results/verificacion_numerica/`.
La campaña anterior está archivada en `results/1eraGranOptimizacion/`.

Este documento recoge (1) un defecto en el criterio de parada de las simulaciones que
invalidaba los valores absolutos de todo el estudio previo, (2) un primer recalibrado
que no generalizó, (3) la re-evaluación de los ganadores, (4) el estudio de
convergencia de malla, (5) el contraste con XFOIL, (6) la comprobación de si el
ranking del GA sobrevive, (7) si una malla gruesa sirve para rankear y (8) el criterio
de parada definitivo. Es el material del capítulo de verificación del TFG.

---

## 1. El defecto: parada anticipada en régimen transitorio

### 1.1 Mecanismo

`Simulador2D.py` corta la simulación cuando considera que Cl y Cd se han estabilizado.
La decisión la toma `_detect_series_convergence`:

```python
ref = float(np.mean(s[-max(1, n * 15 // 100):]))   # media del último 15% de la propia serie
tol = max(tol_abs, tol_rel * abs(ref))
w   = max(3, int(n / t_fisico * window_conv_time))
sm  = np.convolve(s, np.ones(w) / w, mode="valid")
bad = np.where(np.abs(sm - ref) > tol)[0]
converged = (int(bad[-1] + w) if bad.size else 0) <= 0.9 * n
```

La referencia `ref` es la media del último 15 % **de la propia serie**. El test pregunta
si la señal se parece a su propio pasado reciente, no si ha dejado de cambiar. Una señal
que deriva de forma lenta y monótona lo satisface trivialmente: en cada instante se
parece mucho a donde estaba hace poco, aunque lleve camino de duplicarse.

Con los valores por defecto (`clcd_tol_abs=0.005`, `clcd_tol_rel=0.02`,
`clcd_window_conv_time=0.5`, `clcd_min_t_fisico_before_check=1.0`) disparaba en
t ≈ 1–2.7 tiempos convectivos.

### 1.2 Por qué pasó desapercibido

Un segundo defecto encadenado lo enmascaraba. Al salir por parada anticipada, el código
rellenaba la cola de `cdvector`/`clvector` con el último valor calculado. La media del
último 20 % de la serie caía entonces íntegra sobre datos constantes, y la desviación
típica salía **exactamente 0.0**.

Un σ = 0 se lee como convergencia perfecta. Significaba lo contrario: no había datos.
El 81.8 % de las 3884 evaluaciones de `aprendizaje_ML.jsonl` tienen σ = 0.

Corregido sustituyendo el relleno por `n_muestras_validas` y truncado real de los
vectores tras el bucle (`Simulador2D.py`).

### 1.3 Magnitud del error

Para medirlo hacía falta una referencia sin parada. Se corrieron dos casos a 30000
iteraciones con `stop_on_clcd_convergence=False`, volcando la serie completa
(`calib_re1e5.npz`, `calib_re1e3.npz`).

Esto obligó a añadir `tvector` al simulador: el paso temporal es adaptativo, así que el
tiempo físico de cada muestra no se puede reconstruir a posteriori desde el índice de
iteración. Sin eje de tiempo no hay análisis temporal posible.

Ganador Re=1e5, dx=0.002:

| t convectivo | L/D | error vs asintótico |
|---|---|---|
| 1.10 (parada por defecto) | 13.18 | 46.9 % |
| 2.23 | 17.69 | 28.7 % |
| 5.50 | 24.31 | 2.0 % |
| 10.9 – 21.7 | 24.85 | plano |

A t ≈ 1 el flujo lleva poco más de una longitud de cuerda desarrollándose: la capa
límite no está formada, la burbuja de recirculación no ha reatacado y la estela es un
transitorio. No es una medida imprecisa del L/D — es la medida de otro flujo.

**El presupuesto de iteraciones no era el problema.** 13000 iteraciones a dx=0.002
alcanzan t ≈ 9.4, de sobra. El fallo estaba solo en la parada.

---

## 2. Recalibrado del criterio

`verificacion_numerica.py --calibrar` reproduce offline la lógica exacta del solver
sobre las series de referencia, barriendo 108 combinaciones de
(`tol_abs`, `tol_rel`, `window_conv_time`, `min_t`). Para cada una calcula dónde habría
parado y qué L/D habría reportado, frente al valor asintótico de la serie completa.

Resultado del barrido:

- 60 combinaciones no llegan a parar dentro de la serie (criterio demasiado estricto).
- De las 48 que paran, la mediana del error máximo es **8.0 %**.
- Solo **3** bajan del 2 %.

| criterio | tol_abs | tol_rel | window | min_t | error máx | coste |
|---|---|---|---|---|---|---|
| por defecto | 0.005 | 0.02 | 0.5 | 1.0 | **46.9 %** | 0.11 |
| calibrado | 0.001 | 0.01 | 2.0 | 1.0 | **0.82 %** | 0.61 |

`coste` = fracción de la serie de referencia consumida antes de parar.

Las 3 combinaciones válidas comparten `tol_abs=0.001, tol_rel=0.01, window=2.0` y
difieren solo en `min_t` (1, 3 y 5), con resultados idénticos: exigir 2 tiempos
convectivos dentro de banda ya obliga a superar t≈6, así que el mínimo previo es
redundante. **El parámetro que importa es `window_conv_time`**, no las tolerancias ni
el tiempo mínimo.

Coste: las evaluaciones pasan a consumir ~5× más iteraciones. Antes eran baratas porque
no medían nada.

### 2.1 El 0.82 % es error de entrenamiento, no de generalización

**Advertencia importante.** Ese 0.82 % está medido sobre las **mismas dos series** que se
usaron para ajustar el criterio. Es error de entrenamiento sobre una muestra de tamaño 2,
y no dice nada sobre casos nuevos.

Al aplicar el criterio calibrado a puntos que no participaron en el ajuste (la polar del
NACA0012 y del ganador, sección 5) y comparar contra la misma simulación sin parada:

| caso | L/D con criterio calibrado | L/D sin parada | error | dónde paró |
|---|---|---|---|---|
| ganador α=8° | 15.557 | 16.298 | +4.8 % | t=4.10, n=27 |
| ganador α=10° | 8.486 | 8.074 | −4.9 % | no paró |
| NACA0012 α=6° | 11.228 | 12.379 | +10.2 % | **t=1.75, n=10** |
| NACA0012 α=8° | 10.071 | 8.136 | −19.2 % | t=3.40, n=23 |
| NACA0012 α=10° | 8.738 | 6.664 | **−23.7 %** | t=3.01, n=21 |

El error de generalización llega al **−23.7 %**, con signo impredecible. En el caso de
α=6° el criterio cortó en t=1.75 con 10 muestras: el mismo modo de fallo del criterio por
defecto, solo que más tarde.

El defecto de fondo (§1.1) no está resuelto: `_detect_series_convergence` sigue comparando
la señal con su propio pasado reciente. El recalibrado sube el umbral de disparo, pero no
cambia la pregunta que hace el test. Ajustar tolerancias no arregla un estimador mal
planteado.

`ganador α=10°` es el control: nunca llegó a dispararse, corrió el presupuesto entero, y
aun así se movió un −4.9 % al pasar de 11250 a 13000 iteraciones. A ese ángulo ni siquiera
el presupuesto completo alcanza el estado estacionario.

**Consecuencia práctica.** Todo lo medido en este documento usa
`stop_on_clcd_convergence=False` y presupuesto fijo.

**Este criterio recalibrado está retirado.** El barrido de tolerancias no podía
arreglarlo porque el defecto estaba en la pregunta que hacía el test, no en sus
umbrales. La §8 lo sustituye por uno que mide la tendencia de la serie, ajustado y
verificado sobre 20 series en vez de 2.

---

## 3. Re-evaluación de los ganadores

Los 4 ganadores del `convergence_study`, re-simulados a dx=0.002 con el criterio
calibrado (`reeval_ganadores.json`):

| caso | L/D previo | L/D corregido | CI95 | cambio | paró | iters |
|---|---|---|---|---|---|---|
| re1e3 | 5.013 | 4.184 | ±0.043 | −16.5 % | no | 26000 |
| re1e5 | 15.135 | 23.218 | ±0.896 | **+53.4 %** | sí | 8050 |
| re1e6 | 12.185 | 13.209 | ±0.019 | +8.4 % | sí | 8150 |
| re1e7 | 12.085 | 13.192 | ±0.310 | +9.2 % | no | 8250 |

Ambas columnas son **la misma malla, dx=0.002**. La única diferencia es cuándo se deja
de simular.

### 3.1 El sesgo no tiene signo constante

A Re=1e3 el flujo desprende y el L/D oscila; la parada caía en un pico de la oscilación
y **sobre**estimaba un 16.5 %. A Re≥1e5 el flujo es estacionario y la parada prematura
**infra**estimaba.

Esto invalida el argumento habitual de "los valores absolutos están mal pero el ranking
relativo se conserva". Con un sesgo que cambia de signo según el régimen de flujo, la
conservación del ranking hay que demostrarla, no suponerla.

### 3.2 Efecto sobre la discrepancia Cp / fuerza

`cl_cp_discrepancy` (diferencia entre el Cl de la integral de superficie y el Cl de
∮ΔCp) cae de 0.31–0.83 a **0.02–0.14**. Buena parte de lo que se atribuía a
sobre-circulación era artefacto del transitorio. Queda un residuo real, mucho menor,
que crece con el Reynolds.

### 3.3 Nota sobre las dos definiciones de Cl

`ld` usa `Cl_from_Cp_force_consistent`; `ld_raw` usa la integral de superficie, que es la
señal sobre la que opera el criterio de parada. Para comparar con el calibrado hay que
usar `ld_raw`:

| caso | ld_raw | referencia serie larga | error |
|---|---|---|---|
| re1e3 | 4.201 | 4.277 | 1.8 % |
| re1e5 | 24.502 | 24.810 | 1.3 % |

En estos dos casos el criterio calibrado cumple lo prometido — pero son precisamente los
dos casos sobre los que se ajustó (§2.1). La diferencia entre 23.22 y 24.50 en re1e5 es la
discrepancia Cp/fuerza del punto 3.2, no un fallo de la parada.

---

## 4. Convergencia de malla (GCI de Roache)

### 4.1 Método

El GCI (Celik et al., 2008; estándar ASME V&V 20) acota el error de discretización
espacial. Con tres mallas de refinamiento constante r se despejan el orden observado y
la solución de malla infinitamente fina, sin conocer ninguno de los dos de antemano:

```
p     = ln|ε₃₂/ε₂₁| / ln r                    orden observado
f_ext = (r^p·f₁ − f₂)/(r^p − 1)               extrapolación de Richardson
GCI   = Fs·|(f₁−f₂)/f₁| / (r^p − 1)           Fs = 1.25
```

El factor de seguridad Fs=1.25 es empírico, calibrado por Roache sobre cientos de casos.
El GCI se interpreta como una banda de confianza ~95 %.

**El estudio se corre con la parada anticipada desactivada.** Si unas mallas paran antes
que otras, el tiempo físico simulado deja de ser común y el orden observado mide esa
diferencia en lugar de la discretización espacial.

### 4.2 Datos crudos

Ganador Re=1e5, α=4°, r=2:

| dx | Cl | Cd | L/D | iters | muestras | wall |
|---|---|---|---|---|---|---|
| 0.008 | 0.4195 | 0.039999 | 10.488 | 4000 | 10 | 233 s |
| 0.004 | 0.5751 | 0.033324 | 17.258 | 7000 | 28 | 715 s |
| 0.002 | 0.7104 | 0.028769 | 24.692 | 13000 | 52 | 1743 s |
| 0.001 | 0.7402 | 0.029272 | 25.287 | 26000 | 104 | 7472 s |

El salto 0.004 → 0.002 vale +7.43 en L/D; el 0.002 → 0.001 solo +0.60. El régimen
asintótico empieza entre dx=0.004 y dx=0.002.

### 4.3 Resultados (triplete fino 0.001 / 0.002 / 0.004)

| magnitud | p obs. | malla fina | Richardson h→0 | GCI fina | GCI gruesa | monótona |
|---|---|---|---|---|---|---|
| Cl | 2.18 | 0.7402 | 0.7487 | ±1.43 % | ±6.74 % | sí |
| Cd | 3.18 | 0.029272 | 0.029334 | ±0.27 % | ±2.46 % | **no** |
| L/D | 3.64 | 25.287 | 25.339 | ±0.26 % | ±3.28 % | sí |

### 4.4 Lectura crítica

- **p(Cl)=2.18** es el resultado limpio: coincide con el orden formal 2 del esquema. Es
  la cifra a citar.
- **Cd no converge monótonamente** (0.03332 → 0.02877 → 0.02927: baja y rebota). Con
  no-monotonía el GCI deja de ser fiable — el propio Celik lo advierte. El ±0.27 % de Cd
  está sobrevalorado; su incertidumbre real es al menos el error aproximado entre las
  dos mallas finas, ±1.7 %.
- El GCI de L/D hereda el problema, porque L/D = Cl/Cd. Los p de 3.18 y 3.64, por encima
  del orden formal, son otra señal de que ese triplete no está plenamente en el rango
  asintótico.
- **Cifra de incertidumbre de malla defendible: ±1.4 % (la de Cl).**
- dx=0.008 y dx=0.004 quedan fuera del rango asintótico (GCI grueso 3.3–6.7 %). **El GA
  exploró a dx=0.004**, luego sus valores absolutos no son utilizables. dx=0.002 sí, con
  ~2.4 % de error de discretización frente al extrapolado.

### 4.5 Validación cruzada

Tres caminos independientes convergen al mismo valor para el ganador Re=1e5:

| vía | L/D |
|---|---|
| serie larga sin parada, 30000 iters | 24.81 |
| GCI a dx=0.002 | 24.69 |
| reeval con criterio calibrado (`ld_raw`) | 24.50 |

---

## 5. Polar y validación contra datos externos

### 5.1 Polar corregida

Polar α=0…10° a dx=0.002, ganador Re=1e5 y semilla NACA0012. Los cinco puntos de §2.1 se
recalcularon sin parada anticipada; el resto ya había corrido el presupuesto entero.

| α | ganador Cl | ganador Cd | ganador L/D | NACA0012 L/D |
|---|---|---|---|---|
| 0 | 0.2695 | 0.02065 | 13.05 | 0.01 |
| 2 | 0.5098 | 0.02325 | 21.92 | 7.55 |
| 4 | 0.6572 | 0.02831 | **23.22** | 13.84 |
| 6 | 0.8715 | 0.04534 | 19.22 | 12.38 |
| 8 | 1.0354 | 0.06353 | 16.30 | 8.14 |
| 10 | 1.1989 | 0.14849 | 8.07 | 6.66 |

El ganador mantiene la ventaja sobre la semilla en todo el rango, y el máximo de L/D cae
en α=4°, que es el punto sobre el que optimizó el GA. Nada indica sobreajuste al ángulo de
diseño: la mejora es de banda ancha.

### 5.2 Contraste con XFOIL

NACA0012 a Re=1e5, referencia de airfoiltools con Ncrit=9 (`validacion_xfoil.json`):

| α | Cl sim | Cl XFOIL | Cd sim | Cd XFOIL | error Cd |
|---|---|---|---|---|---|
| 0 | 0.0002 | 0.0000 | 0.01653 | 0.01693 | −2.4 % |
| 2 | 0.1659 | 0.3737 | 0.02199 | 0.01444 | +52.3 % |
| 4 | 0.4754 | 0.5365 | 0.03437 | 0.01520 | +126.1 % |
| 6 | 0.5777 | 0.6862 | 0.04667 | 0.01958 | +138.4 % |
| 8 | 0.7065 | 0.8471 | 0.08684 | 0.02869 | +202.7 % |
| 10 | 0.9954 | 0.9661 | 0.14937 | 0.04582 | +226.0 % |

MAE(Cl) = 0.0912. MAE(Cd) = 0.03598. Error medio en Cd: **+123.8 %**.

**La sustentación es buena; la resistencia no.** La pendiente de sustentación es el
resultado sólido:

| fuente | dCl/dα [1/°] |
|---|---|
| solver | 0.10211 |
| XFOIL | 0.11107 |
| teoría 2π | 0.10966 |

Un 8 % por debajo de la teoría delgada, que es lo normal para un perfil de espesor finito.

El Cd se desvía de forma **monótona creciente con α**: −2.4 % a α=0, +226 % a α=10. Un
sesgo constante apuntaría a calibración; uno que crece con la carga apunta al mecanismo
físico. Las causas plausibles, por orden:

1. **Resolución de capa límite.** dx=0.002 con IBM ghost-cell da ~5–10 celdas dentro de la
   capa límite a Re=1e5. El gradiente de velocidad en pared sale suavizado y la fricción
   sobreestimada. Empeora al aumentar α porque la capa límite se engrosa.
2. **Separación prematura.** Si la burbuja laminar no reataca donde debe, la resistencia de
   presión se dispara. Coherente con que la degradación se acelere a partir de α=6°.
3. **Bidimensionalidad.** XFOIL es 2D con corrección integral de capa límite; el solver
   resuelve las ecuaciones completas, y a Re=1e5 el desprendimiento real es tridimensional.
   Ninguno de los dos es la verdad, pero divergen justo donde eso importa.

**Consecuencia.** El L/D absoluto del solver **no es comparable con XFOIL ni con datos
experimentales**. Como el error de Cd depende de α, tampoco es un factor de escala que se
pueda dividir. Lo que sí se sostiene: el Cl, la pendiente de sustentación, y las
comparaciones perfil-contra-perfil **a igual α y misma malla**, que es la única operación
que hace el GA.

---

## 6. ¿Sobrevive el ranking del GA?

`revalidar_ranking.py` re-simula una muestra estratificada por deciles del historial a
Re=1e5, α=4°, dx=0.002, **sin parada anticipada y con presupuesto fijo**, y compara el
orden con el fitness original que vio el GA (dx=0.004, criterio sesgado).

Se estratifica por deciles en vez de muestrear al azar porque un muestreo simple se
concentraría en el grueso de la distribución, y el ranking importa en los extremos.

La muestra se filtra a los 2425 registros de dx=0.004: el historial mezcla esa población
con 368 evaluaciones a dx=0.001 de distribución muy distinta (mediana de L/D 10.04 frente
a 6.05), y mezclarlas metería la diferencia de malla dentro de la correlación de rangos.

### 6.1 Resultados (n=20)

| id | gen | L/D original | L/D revalidado | Δ | rango orig → reval |
|---|---|---|---|---|---|
| 1265 | 0 | 0.068 | 0.559 | +722 % | 20 → 20 |
| 1735 | 4 | 4.697 | 9.778 | +108 % | 19 → 19 |
| 1540 | 3 | 5.487 | 11.281 | +106 % | 18 → 17 |
| 902 | 1 | 6.261 | 15.362 | +145 % | 17 → 12 |
| 1311 | 3 | 6.947 | 11.417 | +64 % | 16 → 16 |
| 1559 | 5 | 7.523 | 15.808 | +110 % | 15 → 10 |
| 1363 | 9 | 8.063 | 12.696 | +58 % | 14 → 15 |
| 2103 | 1 | 8.521 | 11.135 | +31 % | 13 → 18 |
| 1375 | 10 | 9.009 | 13.620 | +51 % | 12 → 14 |
| 2140 | 4 | 9.727 | 15.476 | +59 % | 11 → 11 |
| 2146 | 4 | 10.464 | 17.850 | +71 % | 10 → 9 |
| 2159 | 5 | 11.037 | 20.064 | +82 % | 9 → 8 |
| **2019** | 1 | 11.815 | **24.061** | +104 % | **8 → 1** |
| 3100 | 27 | 12.548 | 21.844 | +74 % | 7 → 4 |
| 3097 | 27 | 13.261 | 22.038 | +66 % | 6 → 3 |
| 2317 | 17 | 13.919 | 20.642 | +48 % | 5 → 7 |
| 1968 | 1 | 14.590 | 20.782 | +42 % | 4 → 5 |
| **2312** | 17 | 15.485 | **13.893** | **−10 %** | **3 → 13** |
| 2681 | 4 | 16.352 | 20.675 | +26 % | 2 → 6 |
| 2958 | 19 | 18.478 | 22.946 | +24 % | 1 → 2 |

| métrica | valor |
|---|---|
| Spearman ρ | 0.7955 (p = 2.75e-05) |
| Kendall τ | 0.6421 (p = 2.51e-05) |
| Pearson r | 0.8341 |
| solape top-4 | **0.25** |
| cambio mediano | +65.3 % (IQR 47–104) |

### 6.2 El ranking global se conserva; la zona alta no

La ρ global de 0.80 es engañosa: viene casi entera de la mitad inferior. Separando la
muestra por la mediana:

| mitad | Spearman ρ |
|---|---|
| 10 peores | 0.612 |
| **10 mejores** | **0.152** |

Y el solape del top-k solo se recupera cuando k deja de ser selectivo:

| k | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| solape | 1/3 | 1/4 | 2/5 | 4/6 | 7/8 | 9/10 |

**El fitness que vio el GA separa bien lo bueno de lo malo, pero no ordena entre lo
bueno** — que es exactamente donde la selección decide. Dos casos lo ilustran:

- **2019** es el mejor real de la muestra (L/D 24.06) y es de **generación 1**. El GA lo
  rankeó octavo y lo descartó. Su Cd cae de 0.0447 a 0.0287 al medir bien. También el
  cuarto mejor original (1968, L/D real 20.78) es de generación 1.
- **2312** (generación 17) es el único con Δ negativo. Su Cd *sube* de 0.0307 a 0.0386
  mientras el del resto baja: la parada anticipada le regalaba resistencia baja. Pasa del
  puesto 3 al 13.

El cambio es sistemáticamente positivo (mediana +65 %) porque a dx=0.004 con parada
sesgada el L/D estaba infraestimado por partida doble: malla fuera del rango asintótico
(§4.4) y transitorio (§1.3). Pero la dispersión —de −10 % a +145 % descontando el
outlier— no es un factor de escala común, y por eso reordena.

### 6.3 Aviso metodológico

Con n=12 la lectura parcial era la contraria: los cuatro individuos de mayor L/D conservaban
el orden exacto y el factor de corrección crecía monótonamente. Parecía que la zona alta se
salvaba. Los 8 que faltaban eran precisamente la zona alta, y ahí es donde el orden rompe.
Una muestra estratificada truncada a mitad de ejecución no es una muestra estratificada.

**Conclusión.** No se puede afirmar que el GA seleccionó correctamente. La afirmación
defendible es más débil, y tiene dos partes.

*A favor:* la búsqueda **sí se movió hacia una región de L/D real alto**. De los cinco
individuos de generación ≥17 de la muestra, cuatro revalidan por encima de 20 (3100: 21.8;
3097: 22.0; 2317: 20.6; 2958: 22.9) y solo el 2312 falla. La correlación generación–L/D
revalidado es ρ = 0.41 (p = 0.073), positiva aunque no significativa con n=20.

*En contra:* el orden **dentro** de esa región no era fiable, y el mejor individuo de la
muestra estaba ya en la generación 1. El GA encontró el barrio correcto guiado por una
señal que distingue órdenes de magnitud, pero eligió la casa con una señal que no
distingue nada a esa escala.

---

## 7. ¿Sirve dx=0.004 para rankear?

El GCI (§4.4) dice que dx=0.004 está fuera del rango asintótico y sus valores
absolutos no valen. Pero el GA no necesita valores absolutos: necesita orden. Es una
pregunta distinta y hay que medirla aparte.

`series_dx004.py` re-simula los **mismos 20 individuos** de §6 a dx=0.004 con
presupuesto fijo (10000 iters, t=12) y sin parada anticipada, y compara el orden con
los valores de dx=0.002.

| señal de fitness | ρ global | **ρ mitad alta** | solape top-4 |
|---|---|---|---|
| fitness que usó el GA (dx=0.004 + parada sesgada) | 0.7955 | 0.1515 | 0.25 |
| **dx=0.004, presupuesto fijo** | **0.9263** | **0.7455** | **1.00** |

**El problema era la parada, no la malla.** A igualdad de malla, medir bien lleva la
mitad alta de ρ=0.15 a ρ=0.75 y el top-4 de 1/4 a 4/4.

El sesgo en valor absoluto es enorme —**−44.9 % ± 68**— y no importa para esto: es
casi un desplazamiento común, y un desplazamiento no reordena. Lo que sí obliga es a
**validar el ganador a dx=0.002**, porque a dx=0.004 el número no es citable.

Dos individuos se salen del patrón y conviene tenerlos presentes:

- **1265** da **L/D = −1.29** a dx=0.004 (real: 0.559). A malla gruesa ni genera
  sustentación positiva. No rompe el ranking —sigue último— pero el GA tiene que
  saber tratar fitness negativo, cosa que nunca hizo porque la parada prematura lo
  enmascaraba.
- **2312** sale **+21.5 %**, único positivo contra la tendencia general. Es el mismo
  que en §6.2 cayó del puesto 3 al 13. Engaña a las dos mallas gruesas por igual, así
  que no es artefacto de la parada sino de la geometría.

---

## 8. El criterio de parada nuevo

### 8.1 Qué mide

`_detect_series_convergence` se ha reescrito. La diferencia no está en las
tolerancias sino en la pregunta:

```python
w     = t >= (t[-1] - window)          # ventana móvil en tiempos convectivos
drift = |pendiente(s[w] vs t[w])| / |media(s[w])|   # tendencia
noise = std(s[w]) / |media(s[w])|                   # dispersión
converged = drift < tol_drift and noise < tol_noise
```

`drift` es lo que faltaba. Una señal que crece un 5 % por tiempo convectivo tiene
drift alto aunque se parezca mucho a su pasado inmediato — que era justo el caso que
el criterio viejo dejaba pasar (§1.1).

Se evalúa sobre **L/D**, no sobre Cl y Cd por separado. Es la magnitud que optimiza
el GA, y Cl y Cd pueden derivar a la vez sin que su cociente lo haga.

### 8.2 Cómo se eligió — que es donde falló la vez anterior

El error de §2.1 no fue el criterio, fue el método: ajustar sobre 2 series y reportar
el error de ajuste como si fuera de generalización.

Aquí hay tres bloques sobre las 20 series de §7, todas corridas sin parar:

1. **CV de 3 pliegues** sobre 15 series para elegir entre 864 combinaciones. Se
   puntúa por el **peor error en los pliegues retenidos**, no por el error de ajuste.
2. **5 series apartadas** que no intervienen en ninguna decisión.
3. Selección: la más barata entre las que cumplen err_max < 2 % en retenidos.

| conjunto | err máx | err medio | coste | para en |
|---|---|---|---|---|
| CV (15 series) | 1.83 % | 0.63 % | 0.56 | 14/15 |
| **TEST (5 series, nunca vistas)** | **1.14 %** | **0.65 %** | 0.67 | 5/5 |

Comparar con §2.1: el criterio anterior daba 0.82 % de ajuste y **−23.7 %** al
generalizar. Éste da 1.14 % sobre datos que no tocaron ninguna decisión.

Criterio elegido: `tol_drift=0.005, tol_noise=0.05, window=1.0, n_sostenido=1,
min_t=5.0`.

### 8.3 Dos cosas que hay que decir

**`min_t=5.0` es lo que arregla el modo de fallo.** Sin él, dos series de validación
paraban en t≈4.1–4.4 con errores de **+10.6 %** y **+19.6 %**: mesetas falsas
anteriores al reataque de la burbuja laminar. La rejilla lo eligió sola, pero la
razón es física y no estadística — por debajo de t≈5 el flujo todavía puede dar un
salto.

**El residual de campo no se usa** (`tol_res=inf`). La intención era replicar el
esquema de los solvers comerciales, que exigen tres condiciones: residuales bajos,
meseta del monitor y persistencia. Con `drift` + `min_t` la tercera no aportaba nada
en estos datos y el barrido la descartó. El sistema real son **dos** condiciones, no
tres. Queda instrumentado (`resvector`) por si a otros Reynolds importa.

Nota sobre `change_v`: se descarta como residual porque se normaliza por la norma de
v, pequeña en un flujo casi horizontal. Vale ~0.5 permanentemente y dominaría
cualquier máximo sin significar nada. Solo `change_u` y `change_p` son informativos.

### 8.4 Verificación del cableado

El criterio se validó offline sobre series ya calculadas; que el solver haga lo mismo
en vivo es una afirmación aparte. Individuo 2019, dx=0.004:

| | offline | solver |
|---|---|---|
| t de parada | 7.013 | 7.0135 |
| L/D reportado | 18.716 | 18.697 |

Asintótico 18.541 → error +0.84 %, coste 0.455.

Ahorro frente a presupuesto fijo: **33–45 %**. Una evaluación a dx=0.004 baja de
~1150 s a ~700 s.

---

## 9. Implementación

| fichero | cambio |
|---|---|
| `Simulador2D.py` | `tvector` (tiempo físico por muestra) y `resvector` (change_u/v/p) en las dos reservas, la escritura y el truncado; `n_muestras_validas` sustituye al relleno de cola; `_detect_series_convergence` reescrita sobre `drift`/`noise` y aplicada a L/D — los parámetros `clcd_tol_abs`/`clcd_tol_rel` se eliminan |
| `scripts/RunGA.py` | `dump_series_path` vuelca t/Cl/Cd/(L/D) a `.npz`; ventana de promediado con mínimo de 10 muestras; CI95, `converged_clcd`, `t_conv_clcd`, `iters_efectivas` y `cl_cp_discrepancy` en el retorno |
| `scripts/agent_tests/verificacion_numerica.py` | etapas `--calibrar`, `--reeval`, `--gci`, `--polar`, `--repolar`, `--validacion`; `criterio_calibrado()`; `T_TARGET_CASO`; GCI sin parada anticipada; `cargar_xfoil()` lee los CSV de `comparativas/` |
| `scripts/agent_tests/revalidar_ranking.py` | muestreo estratificado del historial y correlación de rangos, cacheado y reanudable |
| `scripts/agent_tests/series_dx004.py` | 20 series completas a dx=0.004 sin parar: base de referencia del criterio y test de ranking entre mallas |
| `scripts/agent_tests/criterio_parada.py` | ajuste por CV y verificación sobre conjunto retenido del criterio de parada |
| `scripts/agent_tests/run_convergence_study.py` | el GA lee el criterio calibrado al arrancar; flags `--multipunto`, `--delta-angulo`, `--fitness-modo` |
| `scripts/agent_tests/cola_tfg.sh` | cola serie reanudable: `calibrar → reeval → gci → polar` |
| `scripts/agent_tests/cola_final.sh` | encadena `repolar → validacion → revalidacion` |

El acoplamiento es por fichero: `results/verificacion_numerica/calibracion_earlystop.json`
es la única fuente de verdad del criterio. Quien no lo encuentra avisa en el log de que
está usando el criterio sesgado.

`T_TARGET_CASO` existe porque el horizonte temporal no es universal: a Re=1e3 el flujo
desprende y el L/D no baja del 1 % de error hasta t≈15 (26000 iteraciones), mientras que
a Re≥1e5 basta t=8.

---

## 10. Consecuencias para el TFG

**Qué sigue siendo válido**

- La metodología completa: GA de islas con migración, parametrización de perfiles,
  arquitectura del solver, tratamiento de pared consistente, transición SA-BC.
- El coste computacional del estudio: 98.6 h-GPU, 3884 evaluaciones CFD.
- Las conclusiones sobre el comportamiento del GA (papel de la migración, diversidad,
  presión selectiva, dimensionalidad efectiva del espacio de diseño vía PCA).

**Qué no**

- Todo valor absoluto de L/D anterior a este recalibrado, en las tres capas: exploración
  a dx=0.004, refinado a dx=0.002 y los 3884 registros de `aprendizaje_ML.jsonl`.
- La afirmación de que el ranking relativo se conserva. Medido (§6): ρ global 0.80 pero
  ρ = 0.15 en la mitad superior y solape top-4 de 0.25. El fitness no ordenaba en la zona
  donde el GA selecciona.
- Los ganadores como óptimos. Son óptimos de una función de fitness que no discriminaba a
  esa escala.
- El L/D como magnitud absoluta comparable con la literatura. El Cd se desvía hasta un
  +226 % frente a XFOIL, creciendo con α (§5.2).
- El criterio de parada recalibrado, retirado y sustituido (§8).

**Qué se ha recuperado**

- **Explorar a malla gruesa es legítimo**, al menos para rankear: dx=0.004 con medida
  honesta da ρ=0.93 y solape top-4 perfecto (§7). Lo que rompía el fitness era la
  parada, no la malla.
- **Parar por convergencia vuelve a ser viable**, con error verificado del 1.14 % sobre
  series retenidas y un ahorro del 33–45 % (§8). No hace falta presupuesto fijo salvo
  para producir referencias.

**Qué falta**

1. **Relanzar la optimización.** Consecuencia directa de §6: el GA anterior optimizó
   una señal que no ordena en la zona alta. Ya no hace falta rediseñar el presupuesto
   a ciegas — §7 justifica explorar a dx=0.004 (ρ=0.93 global, 0.75 en la mitad alta)
   y §8 baja el coste de una evaluación a ~700 s. Queda decidir la escala:

   | configuración | evaluaciones | h-GPU |
   |---|---|---|
   | pop 16 × 25 gen × 1 isla | 400 | 78 |
   | pop 16 × 25 gen × 4 islas | 1600 | 311 |
   | pop 24 × 40 gen × 4 islas | 3840 | 747 |

   La campaña anterior fueron 98.6 h. El estudio de convergencia concluyó que la
   migración entre islas es el motor del GA, así que una isla sola probablemente no
   sirve.

2. **Manejar fitness negativo.** El individuo 1265 da L/D = −1.29 a dx=0.004 (§7). El
   GA anterior nunca vio negativos porque la parada prematura los enmascaraba; hay que
   revisar cómo se convierte fitness en probabilidad de selección antes de lanzar.

3. Diagnosticar el exceso de Cd: perfil de capa límite en pared a dx=0.001 frente a
   dx=0.002, y posición del reataque de la burbuja frente a lo que predice XFOIL. Decide
   entre las hipótesis 1 y 2 de §5.2.
4. GA multipunto α∈{2,4,6}. Se descartó cuando una evaluación costaba ~17 min a
   dx=0.002 con presupuesto fijo. Con dx=0.004 y parada por meseta baja a ~12 min por
   los tres ángulos, así que vuelve a estar sobre la mesa si sobra máquina.

**Cómo presentarlo.** Un capítulo de verificación que encuentra y corrige un defecto
propio, lo cuantifica y acota su incertidumbre vale más que una tabla de resultados sin
barras de error. La secuencia defecto → diagnóstico → recalibrado → validación cruzada
por tres vías independientes es exactamente lo que se espera de un trabajo de V&V.

Y el resultado honesto de esa secuencia es que el recalibrado **no bastó**: arregló el
síntoma en los dos casos de ajuste y falló al generalizar. Eso es un hallazgo, no un
fracaso, siempre que se cuente. El solver, tal como está, sirve para comparar perfiles
entre sí a igual α y misma malla — que es lo que necesita el GA — y no para predecir
coeficientes absolutos. Es la afirmación defendible, y es más estrecha que la que se hacía
antes de este capítulo.

---

## Referencias

- Roache, P. J. (1994). *Perspective: A Method for Uniform Reporting of Grid Refinement
  Studies*. J. Fluids Eng. 116(3), 405–413.
- Celik, I. B. et al. (2008). *Procedure for Estimation and Reporting of Uncertainty Due
  to Discretization in CFD Applications*. J. Fluids Eng. 130(7), 078001.
- ASME V&V 20-2009. *Standard for Verification and Validation in Computational Fluid
  Dynamics and Heat Transfer*.

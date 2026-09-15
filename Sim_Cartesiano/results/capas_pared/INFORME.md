# Cuántas capas necesita la extrapolación de presión de pared

Estudio del número de capas `N` del ajuste que reconstruye la presión sobre la
superficie del perfil en el método de frontera inmersa. El solver usa `N=5` y ese
número nunca se había contrastado con una medida.

Script: `scripts/agent_tests/capas_pared.py`. Datos: `results/capas_pared/`.

## Qué se ha hecho

Barrido offline sobre los campos finales de `resultados_finales/naca0012/campos/`
(4 mallas × 9 ángulos). La malla se reconstruye desde cero con los parámetros de
generación de producción y **se comprueba** que salen idénticos los nodos `X`, `Y`
y la máscara de sólido antes de usarla, así que las normales, la máscara de
frontera y el `ds` son exactamente los del run original. Lo único que cambia es
`N`. Reproducir la reconstrucción fuera del solver se validó contra el propio
solver a `N=5`: coincidencia exacta en `Cl_p` y `Cd_p` (`validacion_n5.json`).

No hace falta simular: `N` solo entra en el post-proceso del campo de presión.

**Alcance.** Solo la parte de **presión** de las fuerzas. La tracción viscosa se
muestrea en un único punto a media celda y no depende de `N`; además los campos
volcados no guardan `nu_t`, así que el viscoso ni siquiera se puede reconstruir.
Los campos son instantáneos, así que la comparación **entre `N`** es exacta —mismo
campo, mismo instante— pero la comparación **entre mallas** arrastra la fase del
desprendimiento, igual que el estudio publicado.

## Hallazgo 0 — el argumento `n_extrap_layers` no hace nada

`Simulador2D.py:6024-6031`:

```python
reconstruction_layers = {"linear_5": 5, "linear_9": 9, "weighted_linear_9": 9,
                         "quadratic_9": 9, "robust_huber_9": 9}
n_effective_layers = int(reconstruction_layers.get(pressure_wall_reconstruction,
                                                   max(1, int(n_extrap_layers))))
```

Las cinco claves válidas de `pressure_wall_reconstruction` están todas en el
diccionario, así que el `.get` siempre acierta y el valor por defecto no se evalúa
nunca. El número de capas lo fija el **nombre del modo**, no el argumento. Todas
las llamadas del repositorio con `n_extrap_layers=5` pasan un parámetro inerte.
Esto no invalida ningún resultado publicado —todos se corrieron con 5 capas
igualmente— pero el barrido hay que hacerlo desde fuera, que es lo que hace el
script.

## Hallazgo 1 — la justificación de la varianza es falsa

La memoria dice: *«Promediar cinco capas en lugar de tomar una sola reduce además
la varianza del valor reconstruido»*. No es lo que hace el estimador.

`p_w` no es la media de las capas: es la **ordenada en el origen** de una recta
ajustada en `s = 0,5 … N−0,5`, es decir una extrapolación **fuera** del rango de
los datos. El intercepto es una combinación lineal de las capas con pesos que
cambian de signo y que **suman uno**. Con ruido blanco independiente por capa el
factor de amplificación exacto es

    var(p_w)/σ² = 1/N + s̄²/S_ss

que vale **1,00 en N=1**, **2,50 en N=2**, 1,46 en N=3, 1,05 en N=4 y **0,82 en
N=5**. O sea: cinco capas no bajan de una sola hasta el quinto punto, y aun
entonces reducen la desviación típica un 9 %, no la varianza a la quinta parte.

Y el ruido real no es blanco. Las capas comparten estarcido bilineal y su ruido
está correlacionado: medido sobre el contorno, `corr(capa 1, capa 2) = 0,90` a
dx=0,004 y `0,85` a dx=0,002. Con la covarianza empírica el factor real es:

| N | 1 | 2 | 3 | 4 | **5** | 6 | 7 | 8 | 9 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|
| teórico (ruido blanco) | 1,00 | 2,50 | 1,46 | 1,05 | **0,82** | 0,68 | 0,58 | 0,51 | 0,45 | 0,34 |
| empírico dx=0,004 | 1,00 | 1,19 | 1,22 | 1,20 | **1,16** | 1,10 | 1,05 | 0,99 | 0,94 | 0,81 |
| empírico dx=0,002 | 1,00 | 1,26 | 1,20 | 1,13 | **1,08** | 1,03 | 0,99 | 0,95 | 0,91 | 0,82 |

**Con cinco capas la varianza del valor reconstruido es un 8–16 % MAYOR que
tomando una sola.** No baja de 1 hasta N≈8, y aun a N=12 solo se queda en 0,82.

El caso de la fluctuación temporal, que es del que habla la frase de la memoria,
no se ha medido aquí —haría falta una serie temporal— pero está acotado por
arriba analíticamente: una fluctuación temporal coherente afecta a las cinco capas
a la vez, y como los pesos del intercepto suman uno, pasa **sin atenuar
ninguna**, con factor exactamente 1. Ninguna elección de `N` reduce la varianza de
una fluctuación de modo común. La afirmación no se sostiene en ningún límite.

Figura: `varianza_intercepto.png`.

## Hallazgo 2 — el ruido de `Cp` tampoco baja al añadir capas

Rugosidad de `Cp` a lo largo del contorno (RMS de la segunda diferencia,
normalizada por σ), α=4:

| N | 1 | 2 | 3 | 5 | 9 | 12 |
|---|---|---|---|---|---|---|
| dx=0,004 | 0,117 | 0,121 | 0,123 | **0,130** | 0,129 | 0,124 |
| dx=0,002 | 0,058 | 0,067 | 0,064 | **0,063** | 0,062 | 0,061 |

En las dos mallas finas el ruido a N=5 es **igual o peor** que a N=1. Solo en la
malla de 0,008 baja (0,227 → 0,171 de N=1 a N=12), que es donde el ajuste tiene
más margen porque el campo está mucho menos resuelto.

Y el mecanismo que la memoria invoca —el escalonado de la rasterización— no es el
que domina. Se midió directamente la correlación entre el residuo de alta
frecuencia de `Cp` y la distancia sub-celda del centro de la celda frontera a la
superficie, leída de la SDF (que el solver almacena ya en celdas). Sale **entre
0,05 y 0,14**, y **sube** con `N` en vez de bajar. El ruido de `Cp` no está
sincronizado con el escalonado.

También se descartó la contaminación por celdas sólidas: la fracción de puntos de
muestreo cuyo estarcido bilineal pisa el sólido es **0,000 en todas las capas, en
las cuatro mallas y los nueve ángulos**. Ni siquiera la capa de media celda toca
el cuerpo. Era el sospechoso más razonable para el ruido a `N` bajo y queda fuera.

Figuras: `ruido_N.png`, `estarcido_solido.png`.

## Hallazgo 3 — a `N` le da igual el `Cl`, pero mueve el `Cd` un 20 %

Rango de variación entre `N=1` y `N=12`, relativo al valor en `N=5`, mediana sobre
los nueve ángulos:

| | dx=0,008 | dx=0,004 | dx=0,002 |
|---|---|---|---|
| `Cl_p` | 1,6 % | 0,7 % | **0,5 %** |
| `Cd_p` | 6,4 % | 15,7 % | **17,7 %** |

Restringido al rango razonable `N=3…9` sigue siendo 0,2 % en `Cl_p` y **8–10 %**
en `Cd_p`. El `Cl` es insensible al número de capas; **el `Cd` de presión no**, y
además la sensibilidad **crece** al refinar la malla, que es lo contrario de lo
que uno esperaría de un detalle de post-proceso.

La deriva del `Cd_p` es monótona y creciente con `N`, y es **sesgo de curvatura**,
no ruido. Con un ajuste cuadrático sobre las mismas capas desaparece: el `Cd_p`
cuadrático tiene meseta en `N=7…9` mientras el lineal sigue subiendo
(dx=0,002, α=4: lineal 0,0185 → 0,0221 de N=3 a N=12; cuadrático 0,0193 → 0,0178
→ 0,0180, plano). La presión no es lineal en la normal más allá de unas pocas
celdas, y el residuo del ajuste lo confirma: crece monótonamente con `N` en todos
los casos.

Figuras: `sensibilidad_N.png`, `sesgo_curvatura.png`.

## Hallazgo 4 — el óptimo está en N=3–4, y el 5 cuesta un punto

No hay verdad de referencia para `p_w`, así que se usa una interna: la meseta del
ajuste cuadrático (media de N=7,8,9), que absorbe la curvatura que el lineal no
puede representar. Desvío del `Cd_p` lineal respecto a esa meseta, mediana sobre
los 27 casos de la terna 0,008/0,004/0,002:

| N | 1 | 2 | **3** | **4** | **5** | 6 | 7 | 9 | 12 |
|---|---|---|---|---|---|---|---|---|---|
| desvío mediano | 5,60 % | 2,39 % | **1,64 %** | **1,64 %** | **2,56 %** | 3,80 % | 5,32 % | 9,89 % | 14,06 % |

Curva en U limpia: ruido a `N` bajo, sesgo de curvatura a `N` alto, mínimo plano
en **N=3–4**. El `N` óptimo caso a caso tiene mediana 3 y está entre 2 y 5 en los
27 casos.

**El 5 es defendible: cuesta 0,9 puntos porcentuales sobre el óptimo, frente al
GCI mediano del 20 % y a la banda del `Cd` del 19–55 % del estudio de
convergencia.** La elección de `N` está un orden de magnitud por debajo de la
incertidumbre de malla ya publicada. Lo que no se sostiene es el **motivo** por el
que se eligió.

Figura: `criterio_N.png`.

## Hallazgo 5 — la malla de 0,006 tiene el campo de presión final roto

Fuera del objetivo del estudio, pero apareció y hay que dejarlo escrito. Los
campos de dx=0,006 dan `Cd_p ≈ −0,2 … −0,3` en ocho de los nueve ángulos, con
`Cl_p` muy por debajo de la media publicada. El campo de presión guardado tiene
`p ∈ [−7,8 , 16,5]` con media 0,81, frente a `p ∈ [−0,6 , 0,5]` con media ≈ 0 en
las otras tres mallas: un factor 30. La velocidad, en cambio, es normal.

La serie temporal lo confirma: en dx=0,006, α=4 el `Cd` de las últimas ocho
muestras va de −0,32 a +0,42, o sea que el punto no está en un desprendimiento
periódico establecido sino oscilando con amplitud diez veces su propia media.
Es la firma del crecimiento de presión suave documentado en el estudio del filtro
de tablero.

Esa pata no forma parte de la terna principal (0,008/0,004/0,002) y se calculó
como contraste, así que no toca ningún resultado citado. **Pero su campo final no
sirve para post-proceso** y sus puntos quedan excluidos de las conclusiones de
este informe.

## Lo que este estudio no cierra

- **La varianza temporal no se ha medido**, solo acotado analíticamente. Medirla
  pide una simulación instrumentada que calcule `N=1…10` en cada muestreo.
- **La meseta cuadrática no es la verdad**, es la referencia menos mala
  disponible. Dice dónde el modelo lineal deja de valer, no cuál es el `p_w` real.
- **El Richardson por `N` sale inconcluyente** (`gci_por_n.json`): con campos
  instantáneos de fases distintas solo 1 de 9 casos de `Cl_p` da un `p` utilizable,
  y eso pasa igual con cualquier `N`. La fase domina sobre el efecto que se
  buscaba. La comparación válida es la directa: sensibilidad a `N` frente a la
  banda GCI ya publicada, que es el hallazgo 4.

## Qué habría que cambiar en la memoria

1. Quitar la justificación por reducción de varianza. Es falsa y además es
   comprobable en dos líneas de álgebra, así que es mala apuesta dejarla escrita.
2. Sustituirla por la real, que es mejor argumento: el número de capas es un
   compromiso entre el ruido del muestreo, que pide `N` alto, y el sesgo de
   curvatura del modelo lineal, que pide `N` bajo, con óptimo medido en 3–4 y una
   penalización de menos de un punto porcentual en `N=5`.
3. Matizar la atribución del ruido al escalonado de la rasterización: medido, ese
   mecanismo explica poco (correlación 0,05–0,14) y el muestreo no toca celdas
   sólidas en ningún caso.
4. Se puede añadir el número que faltaba: el `Cl` no depende de `N` (0,5 % entre
   1 y 12 capas en la malla fina) y el `Cd` de presión sí (18 %), muy por debajo
   aun así de la incertidumbre de malla.

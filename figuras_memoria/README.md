# Figuras de la memoria (capítulos 1, 2, 3 y 4)

Doce figuras para el TFG. Cada una se regenera con un script propio en
`scripts/agent_tests/`. Los scripts que necesitan GPU guardan sus campos y series
al lado de la figura, de modo que retocar el dibujo no obliga a repetir la
corrida (`--solo-figura`).

| Figura | Fichero | Script | GPU |
|---|---|---|---|
| 1.A.1 Curva de sustentación Cl(α) | `fig_1A1_curva_cl_alpha.png` | `fig_mem_curvas_polar.py` | no |
| 1.A.2 Polar de resistencia Cd(Cl) | `fig_1A2_polar.png` | `fig_mem_curvas_polar.py` | no |
| 1.B Distribución de Cp de un perfil que sustenta | `fig_1B_cp_perfil.png` | `fig_mem_cp_perfil.py` | no |
| 1.C Capa límite, gradiente adverso y separación | `fig_1C_capa_limite.png` | `fig_mem_capa_limite.py` | no |
| 1.D Campo de presión y fuerzas sobre el perfil | `fig_1D_campo_presion.png` | `fig_mem_campo_presion.py` | no |
| 2.A Descomposición de Helmholtz-Hodge | `fig_2A_helmholtz_hodge.png` | `fig_mem_helmholtz.py` | no |
| 3.A Símbolos discretos del laplaciano y de div-grad | `fig_3A_simbolos_discretos.png` | `fig_mem_simbolos.py` | no |
| 3.B Modo de tablero antes y después del filtro | `fig_3B_modo_tablero.png` | `fig_mem_tablero.py` | sí |
| 4.A Convergencia de la proyección en un paso | `fig_4A_convergencia_proyeccion.png` | `fig_mem_proyeccion.py` | sí |
| 4.B Orden aparente frente al ángulo de ataque | `fig_4B_orden_aparente.png` | `fig_mem_orden.py` | no |
| 4.C Ventana larga hasta t* = 41.5 (Cl y Cd) | `fig_4C_ventana_larga.png` | `fig_mem_ventana_larga.py` | no |
| 5.A Cl(α) de los dos perfiles en tres mallas | `fig_5A_cl_alpha_mallas.png` | `fig_mem_cl_alpha_mallas.py` | no |

---

## 1.A.1 y 1.A.2 — Curva de sustentación y polar de resistencia

Figuras de definiciones para el capítulo introductorio, sin datos de ninguna
corrida: son el esqueleto sobre el que se leen después todos los resultados. Un
solo script escribe las dos.

**1.A.1**, Cl(α): tramo lineal con la pendiente 2π/rad de la teoría de perfil
delgado, α de sustentación nula desplazado a negativo por la curvatura, pérdida
de linealidad al avanzar la separación, Cl_max y caída. La saturación es
algebraica con exponente 6, para que el tramo recto llegue hasta cerca del
máximo en vez de despegarse a mitad de camino, que es lo que haría una tangente
hiperbólica.

**1.A.2**, polar Cd(Cl): parábola Cd = Cd0 + k·Cl² de referencia, cubeta de baja
resistencia, Cd mínimo, engorde al acercarse a la pérdida y la recta desde el
origen tangente a la polar. **El punto de tangencia es el de eficiencia
máxima, y no coincide ni con Cd mínimo ni con Cl máximo**: es el que optimiza el
algoritmo genético, y por eso la figura lo marca aparte.

Con los números de la figura, (Cl/Cd)max = 36.5 en Cl = 0.50, mientras que el Cd
mínimo cae en Cl = 0.15. Son valores representativos, no medidos.

## 1.B — Distribución de Cp de un perfil que sustenta

Define el pico de succión, el punto de remanso, la recuperación de presión del
extradós y el hecho de que la sustentación es el área encerrada entre las dos
ramas.

**No está dibujada a mano.** Sale de un método de paneles de Hess-Smith
—fuentes de intensidad constante por panel más un torbellino común, con la
condición de Kutta en el borde de salida— sobre un NACA 2412 con 200 paneles en
reparto coseno, α = 6°. Las influencias se derivan en ejes del panel (fuente ->
(log, β)/2π, torbellino -> (−β, log)/2π) y se proyectan sobre la normal y la
tangente del panel de control. Con los signos tomados de memoria el Cp sale
invertido y el perfil sustenta hacia abajo, así que el script comprueba lo que
calcula:

- Cp en el punto de remanso: 0.998 (exacto 1)
- pendiente a0 = 6.92/rad (2π = 6.28; con espesor sale algo por encima)
- α de sustentación nula = −2.13° frente al −2.1° tabulado del NACA 2412
- Cl por Kutta-Joukowski 0.9780 frente a 0.9778 por integración de la presión,
  y 0.891 por teoría de perfil delgado

Es flujo potencial, sin capa límite: vale como referencia de la **forma** de la
curva y de nada más, porque la separación que discute el capítulo siguiente es
justo lo que este modelo no ve. La silueta del perfil va superpuesta en el mismo
eje (eje insertado, para que conserve su proporción y siga cuadrando en x) con
el pico marcado sobre ella: a este ángulo el tramo de gradiente favorable se
reduce a la nariz (x/c = 0.006) y todo el resto del extradós está en gradiente
adverso, que es la banda rosa del fondo.

## 1.C — Capa límite, gradiente adverso y separación

Tres cosas en un dibujo, sobre el extradós estirado a línea recta: la capa
engorda aguas abajo, tras el pico de succión el gradiente se vuelve adverso y el
perfil de velocidad se vacía por abajo, y cuando el esfuerzo en la pared se
anula el flujo se invierte y la capa se separa.

Los perfiles de velocidad son soluciones de Falkner-Skan

```
f''' + f f'' + beta (1 - f'^2) = 0,   f(0) = f'(0) = 0,   f'(inf) = 1
```

integradas por disparo sobre f''(0), con beta = 2m/(m+1) para U_e ~ x^m. Los
cuatro perfiles que se dibujan y lo que da el disparo:

| beta | f''(0) | eta_99 | |
|---|---|---|---|
| +0.5 | 0.92768 | 2.750 | favorable, perfil lleno |
| 0 | 0.46960 | 3.472 | Blasius |
| −0.12 | 0.28176 | 3.866 | adverso, con punto de inflexión |
| −0.198838 | 0.00000 | 4.789 | separación, tau_w = 0 |

En la separación la raíz es exactamente f''(0) = 0, que es un extremo del
intervalo y el disparo no puede encerrar: se impone, y se comprueba que el
residuo f'(inf) − 1 queda por debajo de 1e-5.

El punto de inflexión marcado (○) es el cero de d2u/dy2, es decir de f''', que
es el **máximo** de f'' —no el cambio de signo de f'', que es el error fácil—.
Está ahí por la ecuación de cantidad de movimiento en la pared,
mu·d2u/dy2 = dp/dx: con gradiente adverso la curvatura en la pared es positiva,
aparece la inflexión, y con ella la inestabilidad que acaba desprendiendo la
capa.

El único trazo que no sale de una ecuación es el perfil invertido de detrás de
la separación, porque pasado ese punto la teoría de capa límite ya no tiene
solución; va dibujado como esquema, y así lo dice el script.

## 1.D — Campo de presión y fuerzas sobre el perfil

Figura de apertura, casi sin texto: corriente uniforme por la izquierda y el
campo de Cp alrededor de un NACA 2412 a α = 8°, para que se vean de un vistazo
la sobrepresión del borde de ataque y del intradós y la succión del extradós,
que es de donde sale la sustentación.

El campo sale del mismo método de paneles de Hess-Smith de la figura 1.B: con
las intensidades ya resueltas se evalúa la velocidad inducida en una rejilla de
900×460 puntos y se aplica Bernoulli, Cp = 1 − |V|²/U∞². El perfil se gira −α y
se resuelve con incidencia nula, así la corriente libre queda horizontal.
Comprobaciones del script: Cp = 1 en el remanso y Cl por Kutta-Joukowski igual
al de la integral de presión (1.2179 frente a 1.2178).

Sobre el campo van unas pocas líneas de corriente (integradas sólo aguas abajo,
sin semilla en y = 0 para no clavar una flecha en el punto de remanso) y, en la
pared, dos juegos de vectores pequeños: los grises son la presión, −Cp·n̂ del
propio método de paneles, hacia dentro donde comprime y hacia fuera donde
succiona; los naranjas son la fricción, y esos sí son cualitativos —módulo fijo,
dirección la del flujo en la pared—, porque el flujo potencial no tiene capa
límite. Por lo mismo la flecha D es cualitativa: sin viscosidad no hay
resistencia (d'Alembert), y está sólo para nombrar la descomposición de la
fuerza.

La escala se satura en Cp = −1: en la pared el pico de succión llega a −4 y sin
saturar el resto del campo saldría plano.

## 2.A — Descomposición de Helmholtz-Hodge

Tres paneles sobre el mismo rectángulo: un campo sintético **w**, su parte
solenoidal tangente a la frontera y el gradiente del potencial, con la relación
**w** = **u** + ∇φ escrita entre ellos.

El campo de partida **no** se construye sumando las dos piezas. Se define entero
—dos torbellinos, una fuente, un sumidero y una cizalla de fondo— y se
descompone resolviendo el mismo problema de Poisson con Neumann de la ecuación
(2.7), con volúmenes finitos y solver directo disperso. Los paneles 2 y 3 son
por tanto el resultado de una proyección de verdad.

El fondo de color es la divergencia de cada panel, con la misma escala en los
tres: toda la divergencia de **w** acaba en el gradiente y el panel central sale
blanco. Comprobado numéricamente al generar la figura: residuo de la suma 7e-17
y divergencia media de la parte solenoidal 2 % de la del campo de partida
(el resto es error de discretización del propio esquema centrado).

## 3.A — Símbolos discretos

Análisis local de Fourier del modo p_j = exp(i k x_j), con θ = k·h, normalizado
por el espaciado. Es analítico y no toca el solver:

- exacto: θ²
- laplaciano compacto de 5 puntos: 4 sin²(θ/2)
- div-grad con gradiente centrado (galga de 2h): sin²θ

Panel inferior: error relativo de cada símbolo respecto del exacto.

En θ = π —el modo de tablero, longitud de onda 2h— el gradiente centrado se
anula y con él la composición div-grad, mientras que el laplaciano compacto
alcanza ahí su máximo, 4/h². El modo par-impar es invisible para el operador que
la proyección resuelve, y de ahí que el multimalla no lo reduzca.

## 3.B — Modo de tablero antes y después del filtro

Dos corridas idénticas del punto que revienta (perfil optimizado, dx = 0.004,
α = 0, dominio 24×16, Re = 1e5, presupuesto 2×3, `warm_start` activo) que solo
se diferencian en `warm_start_filtered`. Las dos se paran en la iteración 700,
por debajo de la 1340 en la que la versión sin filtro aborta por blowup, para
que la comparación sea entre dos campos vivos.

**Se dibuja la divergencia residual tras la proyección, no la presión.** Es un
cambio deliberado respecto del enunciado original de la figura, y la razón es la
propia figura 3.A: el modo par-impar es casi invisible para el operador de
presión, así que no se acumula en p. Comprobado sobre los campos de estas dos
corridas: la parte par-impar de la presión es del orden del 0,04 % del campo y
los dos mapas de presión son indistinguibles a simple vista. Donde el modo vive
es en la divergencia que la proyección deja sin corregir —que es también donde
estaba medido, el 46 % de la divergencia residual— y ahí se ve celda a celda y
se ve desaparecer.

Medido sobre estas dos corridas: divergencia media 2.01e-2 → 1.69e-3 en todo el
dominio, doce veces menos, con el fondo suave intacto.

## 4.A — Convergencia de la proyección dentro de un paso temporal

Se lleva el caso de referencia (perfil optimizado, dx = 0.004, α = 4, dominio
24×16, Re = 1e5, las cinco optimizaciones de producción) hasta un estado
desarrollado y ahí se congela el campo. Desde ese mismo estado se repite **un
paso temporal completo** —advección, SA, difusión, proyección— variando solo el
número de iteraciones externas, de 1 a 12, con 3 ciclos en V por iteración
externa y el modo adaptativo desactivado para que el presupuesto sea el pedido.

De cada repetición se miden la divergencia media del campo resultante y el Cl,
este último con la misma integral de esfuerzos del bucle principal.

Resultado: de 1 a 12 iteraciones externas la divergencia media baja un 35 %
(2.01e-3 → 1.31e-3) con la pendiente ya casi plana, mientras el Cl se mueve un
10,9 % (0.6733 → 0.7514) y sigue subiendo en el último punto. Las dos curvas
responden al presupuesto a ritmos incomparables, así que la divergencia media no
certifica que el presupuesto baste para las fuerzas. El presupuesto de
producción, 2 iteraciones externas, queda marcado con la línea de puntos.

## 4.B — Orden aparente frente al ángulo de ataque

Sale directamente de `resultados_finales/richardson/tabla_gci.csv`, columna
`p_observado_con_signo`. Un punto por perfil, magnitud y ángulo, con la banda
admisible p ∈ [1, 4] sombreada y el orden nominal del esquema marcado.

Las series no monótonas —los dos saltos entre mallas con signo opuesto, donde el
orden no está definido— van en una banda propia bajo el eje para que no
desaparezcan del recuento.

Resumen de lo que dicen las tablas 4.6 y 4.12: de los 36 casos de Cl y Cd, el
orden es utilizable en 8, todos de sustentación; la resistencia no consigue
ninguno y sale con orden negativo en 10 de 18; los fallos de la sustentación se
concentran en los extremos del barrido.

## 4.C — Ventana larga hasta t* = 41.5

Prolongación del punto α = 4 del perfil del punto de diseño en la malla de
dx = 0.002 hasta t* = 41.5, más del doble del presupuesto t* ≈ 20 del resto del
capítulo, para acotar la sensibilidad a la ventana temporal. Las dos medias de
cola de cinco tiempos convectivos (t* = 15-20 y 36.5-41.5) coinciden dentro del
1 %: Cl 0.6796 → 0.6829 (+0.5 %), Cd 0.01994 → 0.02013 (+1.0 %), L/D 34.09 →
33.92 (−0.5 %). El presupuesto corto no sesga la polar.

Se recorta el arranque (t* < 2): el primer muestreo del impulso inicial vale
Cl = 19.9 y Cd = 5.53, dos órdenes por encima del régimen, y con él dentro las
dos curvas quedan aplastadas contra el cero — que es lo que le pasa a
`results/asintotico_alpha4_domC/evolucion_temporal.png`, la versión anterior de
esta misma figura. Datos ya calculados: `series.npz` de ese directorio.

## 5.A — Cl(α) de los dos perfiles en tres mallas

Un panel por perfil, una curva por malla, barras = CI95 de la media temporal.
Datos de `resultados_finales/metricas_todas.csv` (Re = 1e5, dominio 24×16).

Se dibuja la terna del GCI, dx = 0.008 / 0.004 / 0.002 (r = 2 en los dos
saltos). La cuarta malla del estudio, dx = 0.006, se omite: se calculó como
terna de contraste, no entra en el GCI, y su curva se solapa con las otras sin
añadir nada.

Las barras de CI95 son más pequeñas que los símbolos —a α = 4, ±0.0053 sobre
0.4978 en el NACA 0012 y ±0.0015 sobre 0.6965 en el ganador—, mientras que el
salto entre mallas en ese mismo punto vale 0.037 y 0.017. La incertidumbre
estadística de la ventana es un orden de magnitud menor que la de malla, que es
justo lo que la figura tiene que dejar ver.

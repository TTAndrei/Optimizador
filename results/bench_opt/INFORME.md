# Optimización del solver de presión — informe

Trabajo de la noche del 2026-08-24. Todo se mide sobre la config de producción
(`richardson_polar_domC`): perfil ganador de `islands_tfg2`, dominio C
(Lx=24, Ly=16, cx=6), **dx_min=0.002** → malla 1772×713 = 1.26 M celdas,
α=4°, Re=1e5, CFL=0.5, con las dos paradas desactivadas para que todas las
variantes recorran exactamente las mismas iteraciones.

## Qué se ha tocado

| Fichero | Qué es | Estado |
|---|---|---|
| `opt_solver.py` | **Nuevo.** Todas las optimizaciones, cada una con rama "off" que reproduce el código original | Desactivado por defecto |
| `Simulador2D.py` | Sustituciones línea por línea que llaman a `opt_solver` | **25 inserciones, 5 borrados** |
| `scripts/agent_tests/bench_opt.py` | **Nuevo.** Harness A/B con instrumentación del solver | — |
| `scripts/agent_tests/diag_divergencia.py` | **Nuevo.** Dónde y de qué tipo es la divergencia residual | — |
| `scripts/agent_tests/cola_*.sh` | **Nuevos** (`cola_noche.sh`, `cola_resto.sh`). No tocan `cola_final.sh` ni `cola_tfg.sh` | — |

Con todas las opciones apagadas el simulador se comporta igual que antes.
**Demostrado, no afirmado**: la variante `equiv` (4000 iters, todas las flags
apagadas) reproduce el baseline con 14/14 métricas idénticas —Cl, Cd, L/D, sus
desviaciones, ciclos multigrid totales, divergencia media y máxima, outers
gastados, fracción que converge— y con las trayectorias completas `Cl(t)`,
`Cd(t)`, `div(t)` y `cycles(t)` iguales **bit a bit** (delta máximo 0.000e+00
sobre 160 muestras). Lo único que varía es el cronometraje, +2.6 %, que es la
dispersión normal entre corridas.

Activación, sin tocar los scripts de campaña:

```bash
OPT_SOLVER=coarse_mask_majority,deep_levels .venv/bin/python scripts/...
```

## Punto de partida medido

| | valor |
|---|---|
| paso | 0.23 s (4.36 it/s) |
| **ocupación de GPU** | **40-42 %** → ~60 % del tiempo la GPU está parada |
| ciclos multigrid por paso | 31-35 |
| outers gastados por paso | **8.0 de 8** (siempre el máximo) |
| pasos que alcanzan la tolerancia | **0.0 %** |
| factor de convergencia del multigrid | 0.52 por V-cycle |
| kernel rojo-negro, nivel fino | 8.5 µs (en el límite del ancho de banda) |

## Hallazgos, en orden de importancia

### 1. El solver nunca converge, y por eso el presupuesto no compra nada

`frac_converge = 0.0`: ni un solo paso de 4000 baja de `tol_div`. El bucle
externo gasta siempre sus 8 outers. Esto cambia por completo la interpretación
del coste: no son 8 outers "por si acaso", son 8 outers que se pagan enteros
siempre.

La pregunta que decide todo es si esos 8 outers compran precisión. La curva de
presupuesto (`punto_o4c3`, `punto_o2c3`, `punto_o2c2`, `punto_o1c2`) la
responde, y es el candidato con mejor relación ganancia/riesgo porque **no toca
una línea de código**: es configuración.

### 1-bis. Por qué el solver no puede converger: desacoplamiento par-impar

Esto explica el hallazgo 1 en vez de sólo constatarlo, y es lo que sostiene toda
la recomendación.

La malla es **colocada**: u, v y p viven en el mismo punto. La proyección
resuelve `L p = D u*` y corrige `u = u* - G p`, así que la divergencia que queda
es `D u* - D·G p = (L - D·G) p`. Si `L` no es exactamente `D·G`, ese resto no es
cero por bien que se resuelva el Poisson.

Símbolos de Fourier en 1D uniforme (`L` compacto de 3 puntos frente a la
composición de dos derivadas centradas de 3 puntos):

| θ/π | símbolo L | símbolo D·G | \|L−D·G\|/\|L\| | modo |
|---|---|---|---|---|
| 0.05 | −0.0217 | −0.0215 | 0.5 % | suave |
| 0.15 | −0.2136 | −0.2022 | 5.3 % | suave |
| 0.30 | −0.8086 | −0.6451 | 20.2 % | medio |
| 0.49 | −1.9509 | −0.9994 | 48.8 % | medio |
| 0.89 | −3.8831 | −0.1135 | 97.1 % | casi tablero |
| **1.00** | **−4.0000** | **0.0000** | **100 %** | **tablero** |

En los modos suaves coinciden — de ahí que el método sea consistente y de
segundo orden. En el modo tablero el gradiente centrado **se anula**: hay
componentes de divergencia que ninguna presión puede eliminar.

Se comprobó si la ruta de flujo de cara escapa de esto, porque es la que usa
producción (`wall_treatment='consistent'` → `divergence_form="face_flux"`).
No escapa: `_build_projection_faces` (`Simulador2D.py:2221`) hace **interpolación
lineal pura, sin término de amortiguamiento tipo Rhie-Chow**, así que la
divergencia de cara es algebraicamente la diferencia centrada. Por eso las cuatro
combinaciones de gradiente y divergencia dan el mismo error en la medida directa.

**Consecuencia operativa — y aquí me equivoqué de predicción.** De esto deduje
que, si la tolerancia es inalcanzable por construcción, gastar 8 outers × 5
ciclos persiguiéndola sería trabajo tirado y recortar el presupuesto saldría casi
gratis. **La medida dice que no.** `punto_o4c3` (4 outers × 3 ciclos) va 2.21×
más rápido pero mueve Cl un −4.98 % y Cd un +12.1 %, y empeora la divergencia un
43 %: se sale de la puerta con holgura.

La deducción correcta es más estrecha. El desacuerdo entre `L` y `D·G` explica
por qué `frac_converge = 0.0` —la métrica de tolerancia incluye modos que ninguna
iteración puede eliminar— pero **no** implica que iterar no sirva: la componente
suave de la divergencia sí se elimina iterando, y es de ella de la que dependen
Cl y Cd. Que el criterio de parada sea inalcanzable no convierte en inútil el
trabajo que se hace antes de rendirse.

Así que no hay almuerzo gratis por el lado de la configuración, y el problema
vuelve a ser el que era: **conseguir la misma precisión con menos trabajo**, que
es exactamente lo que atacan el suavizador por líneas y los niveles profundos.
La curva de presupuesto pasa de ser el candidato principal a ser el eje de
referencia contra el que se miden: una variante sólo aporta algo si cae por
encima de esta curva de precisión frente a velocidad.

**Lo que esto NO significa.** No es un fallo del código ni algo que haya que
arreglar esta noche: es una propiedad conocida de las mallas colocadas, y el
remedio clásico (Rhie-Chow o malla desplazada) cambia la discretización y por
tanto invalidaría el GCI ya cerrado. Se documenta como el techo real del método,
no como una tarea pendiente.

### 2. La malla es fuertemente anisótropa y eso rompe el suavizador

Medido sobre la malla real:

```
relación de aspecto por celda:  p50 = 1.00   p75 = 28.1   p90..p100 = 50.0
  30.4 % de celdas con AR > 8
  24.5 % de celdas con AR > 32
anisotropía del operador (AR², que es lo que ve el suavizador):
  p90 = 2500        29.2 % de celdas con AR² > 100
```

El acoplamiento del laplaciano de 5 puntos va como 1/h², así que lo que degrada
al suavizador es el **cuadrado** de la relación de aspecto. Gauss-Seidel punto a
punto tiene factor de suavizado ≈ 1 en la dirección débil: en un tercio del
dominio no suaviza nada. Sin suavizado no hay multigrid.

Es la receta de libro (relajación por líneas o semi-coarsening) y explica por
qué el código tuvo que topar los niveles a 2 y añadir un guard anti-divergencia.

**Suavizador por líneas implementado** (`line_smoother`), con ordenación cebra y
Thomas. Verificado contra el rojo-negro en tres regímenes:

| caso | reducción de residuo por barrido | acuerdo entre métodos |
|---|---|---|
| isótropa | 1.9× mejor | 2.5e-04 |
| AR=8 | 10× mejor | 3.2e-04 |
| AR=50 (producción) | 19786× mejor | 8.4e-06 |

Sobre el problema real: factor MG 0.52 → **0.34**, divergencia residual
0.00327 → **0.00153**. Pero cuesta **3.2× más por paso** (49× por barrido; el
69 % de ese coste son las líneas X, que no coalescen). Por eso sólo puede ganar
tiempo a presupuesto recortado, que es como está en la cola.

### 3. El coarsening engorda el perfil — corregido, y sale gratis

`_coarsen_mask` hacía OR sobre un bloque 3×3: bastaba una celda fina sólida para
que la gruesa lo fuera. Con borde de salida afilado, dos niveles convierten un TE
de una celda en un cuerpo romo. La literatura de multigrid con fronteras
inmersas describe exactamente esto (pérdida de la frontera al engrosar, cambio
topológico, modos espurios).

`coarse_mask_majority` hace que la celda gruesa sea sólida sólo si lo es la
mayoría del bloque. Las estructuras finas desaparecen del nivel grueso en vez de
engordarlo, que es preferible: el nivel grueso sólo corrige error de baja
frecuencia y un Poisson sin el cuerpo sigue siendo buen precondicionador.

Resultado (400 iters): factor MG **0.52 → 0.43**, ciclos 31.4 → 28.0,
divergencia algo mejor, y **8 % más rápido**. La máscara se construye una vez en
el init, así que no cuesta nada en régimen. Es la mejora más limpia encontrada.

### 4. El número de niveles estaba topado a fuego

`nlvl_use = min(n_levels, 2)`. Aunque se suba `mg_niveles_max`, el V-cycle sigue
usando 2 niveles. La jerarquía de esta malla:

| nivel | dimensiones | celdas | |
|---|---|---|---|
| 0 | 1772 × 713 | 1.263.436 | |
| 1 | 886 × 356 | 315.416 | |
| 2 | 443 × 178 | **78.854** | ← tope actual del V-cycle |
| 3 | 221 × 89 | 19.669 | |
| 4 | 110 × 44 | 4.840 | |
| 5 | 55 × 22 | 1.210 | |
| 6 | 27 × 11 | **297** | ← con `deep_levels` |

Un multigrid funciona porque baja hasta una malla lo bastante pequeña como para
resolverla casi exactamente por cuatro duros. Aquí se para en **78.854 celdas**,
que no es una malla pequeña: los modos de longitud de onda larga no se corrigen
en ningún nivel. Es la explicación más simple de un factor de convergencia de
0.52 en lugar de 0.1-0.2.

Va acompañado de `coarse_iters`, porque la constante `coarse_solve_iters = 20`
también estaba fija: 20 barridos son una corrección parcial razonable sobre
78 k celdas —que es lo que el código pretendía— pero dejarían sin resolver el
nivel más grueso de una jerarquía profunda, y el experimento habría dado un
falso negativo por una constante mal escalada en vez de por el método.

Ese tope estaba puesto *por* los problemas 2 y 3. Corregidos ambos, se comprueba
si los niveles profundos ya suman.

**No funciona. Mi hipótesis era errónea.**

Con 4 niveles, `deep4` corría a 1.75 it/s, 2.5× **más lento** que el baseline: el
nivel más grueso todavía tiene 4840 celdas y resolverlo pide ~100 barridos que a
35 ciclos por paso no se amortizan. Se cortó esa prueba (5 variantes, ~2 h de
GPU) para hacer la buena, con 6 niveles, donde el nivel más grueso son 297 celdas
y 54 barridos lo resuelven casi exactamente.

Con 6 niveles y la máscara corregida, `deep6_maj`:

| | baseline | `deep6_maj` |
|---|---|---|
| it/s | 4.36 | **1.50 (0.34×)** |
| ciclos/paso | 35.0 | 40.0 (saturado) |
| **divergencia** | 0.00869 | **0.0141 (62 % PEOR)** |
| Cl | 0.5578 | 0.4641 (−16.8 %) |

No es que no ayude: **empeora**. Con más niveles la divergencia sube un 62 % y el
solver satura su presupuesto sin llegar a ningún sitio. Las tres variantes de 6
niveles restantes se cortaron: si falla a presupuesto completo, recortarlo no lo
va a rescatar.

**Lo que esto significa.** Yo supuse que el tope `min(n_levels, 2)` existía sólo
por la máscara dilatada, y que corrigiéndola se podría profundizar. El comentario
original del código —«con malla 20:1 no uniforme, niveles profundos son
inconsistentes»— tenía razón, y la causa es la misma anisotropía del hallazgo 2:
el coarsening isótropo 2×2 sobre celdas con relación de aspecto hasta 50 produce
un operador grueso que no representa al fino, y cada nivel compone el error.

Queda entonces un cuadro coherente: **la anisotropía rompe las dos mitades del
multigrid a la vez**, el suavizador y el coarsening. La relajación por líneas
arregla la primera pero cuesta demasiado en GPU; arreglar la segunda pedía
semi-coarsening, que no se ha implementado. Por eso el multigrid de este solver
se queda en un factor de 0.52 por ciclo y no hay forma barata de bajarlo.

### 5. El warm start no es neutro, y hay que decirlo

Como el solver nunca converge, arrancar el Poisson desde la presión del paso
anterior no da "la misma solución más rápido": tras N ciclos se llega a una
aproximación **distinta** (y mejor) de la misma solución. Los resultados cambian.
No es un fallo — es justo lo que la puerta del 0.5 % está para medir — pero
descarta describirlo como una optimización sin efecto sobre los números.

### 6. Vías descartadas con medida

| vía | resultado |
|---|---|
| `projection_variant="compatible_flux"` | Peor en todo: ciclos saturados a 40, divergencia 2.7× peor, Cd 34 % distinto |
| `usar_adjoint_correction=true` | **Blowup en la iteración 0** |
| `mg_apply_ibm_each_outer=false` | Bit-idéntico al baseline: el IBM por outer no es la causa |
| Portar a JAX | Techo 3.4× (el 60 % de GPU ociosa) por semanas de reescritura funcional de ~2000 líneas, con 8 GB de VRAM y preasignación del 75 %. CUDA Graphs da lo mismo sin reescribir |
| Coarse-Grid Projection | 2-30× en la literatura, pero cambia la discretización e **invalida el GCI ya cerrado** |
| Solver FFT | 30-60× pero exige malla uniforme separable y contornos simples. No aplica |

## Cuestión abierta que puede importar más que la velocidad

La curva de presupuesto degrada mucho más de lo esperado:

| presupuesto | ciclos/paso | ×vel | Cl | ΔCl | Cd | ΔCd | div |
|---|---|---|---|---|---|---|---|
| 8×5 (producción) | 35.0 | 1.00 | 0.5578 | — | 0.02863 | — | 0.00869 |
| 4×3 | 12.0 | 2.21 | 0.5300 | −4.98 % | 0.03211 | +12.1 % | 0.0124 |
| 2×3 | 6.0 | 3.22 | 0.4704 | −15.7 % | 0.03715 | +29.7 % | 0.0172 |

Para comparar: la banda del GCI a α=4 es **Cl 4.25 %, Cd 19.0 %, L/D 21.0 %**.
Es decir, bajar el presupuesto de la proyección a la mitad mueve Cl tanto como
**toda** la banda de refinamiento de malla del estudio.

Eso plantea una pregunta que el estudio de Richardson no contesta: **¿está
convergido en presupuesto de proyección el propio 8×5 con el que se calcularon
las polares?** Si no lo está, hay una incertidumbre numérica que no está
contabilizada junto a la de malla.

**Límite de lo que se puede concluir.** Estas medidas están a t≈2.5 (4000
iteraciones), en pleno transitorio: el propio Cl del baseline vale 0.558 aquí
frente a 0.680 convergido a t≈20. La sensibilidad en el transitorio podría ser
mayor que en régimen, así que **esto no demuestra que las polares publicadas
carguen ese error**. Es un motivo para medirlo, no una conclusión sobre el
estudio cerrado.

Ahora bien, la explicación tranquilizadora se puede contrastar con lo que ya hay
en disco, comparando las series `Cl(t)` de `baseline` y `punto_o4c3` por tramos:

| tramo de t | ΔCl | ΔCd | Cl del baseline |
|---|---|---|---|
| 0.03 – 0.54 | +3.81 % | +5.29 % | 0.4055 |
| 1.06 – 1.57 | −4.49 % | +10.29 % | 0.5121 |
| 2.09 – 2.60 | −2.59 % | +20.32 % | 0.5346 |
| 2.60 – 3.11 | −5.07 % | +13.42 % | 0.5611 |
| **primer tercio** | **−0.05 %** | **+5.44 %** | |
| **último tercio** | **−3.86 %** | **+16.81 %** | |

La diferencia **se ensancha** con el tiempo en vez de estrecharse. Sobre el rango
medido (t hasta 3.1, frente a t=20 en producción) no hay señal de que el
transitorio esté exagerando el efecto. Podría saturar más adelante, pero la
hipótesis cómoda no se sostiene con los datos disponibles.

### Respuesta, con la curva completa

| presupuesto | ciclos | divergencia | Cl | ΔCl | Cd | ΔCd |
|---|---|---|---|---|---|---|
| 1×2 | 2.0 | 0.0282 | 0.2825 | −49.36 % | 0.04919 | +71.80 % |
| 2×2 | 4.0 | 0.0203 | 0.4021 | −27.92 % | 0.04394 | +53.47 % |
| 2×3 | 6.0 | 0.0172 | 0.4704 | −15.68 % | 0.03715 | +29.73 % |
| 4×3 | 12.0 | 0.0124 | 0.5300 | −4.98 % | 0.03211 | +12.14 % |
| **8×5 (producción)** | 35.0 | 0.00869 | 0.5578 | — | 0.02863 | — |
| 12×6 | 40.9 | 0.00863 | 0.5801 | +4.00 % | 0.02969 | +3.70 % |
| 16×8 | 65.5 | 0.00764 | 0.5666 | +1.57 % | 0.02800 | −2.22 % |

**Por debajo de producción, estrictamente monótono**: Cl sube de 0.2825 a 0.5578
según se añade presupuesto. Falta de resolución clara y sistemática.

**Por encima, no**: 0.5578 → 0.5801 → 0.5666. Sube y baja. La dispersión es del
**3.93 %**, comparable a la banda del GCI para Cl (4.25 %), pero **no es un sesgo
en una dirección**.

**Conclusión, y modera lo que yo mismo había levantado.** El 8×5 de producción
está *en la rodilla* de la curva: por debajo se degrada de forma sistemática, por
encima hay una dispersión de ~4 % sin tendencia clara. Es una incertidumbre real
que no figura junto a la de malla en el estudio, y del orden de ella; pero **no
es el sesgo sistemático que sugerí antes de tener estos dos puntos**.

**Y refuta mi propia extrapolación.** Escribí más arriba que extrapolar Cl frente
a divergencia hacia cero daba Cl ≈ 0.701, y la marqué como no citable. La medida
directa a doble presupuesto da **0.5666**, no 0.70. La extrapolación estaba mal,
y la advertencia hizo su trabajo: conviene que quede como ejemplo de por qué no
se cita una extrapolación fuera del rango de los datos.

Queda pendiente sólo comprobar si esta dispersión del 4 % a t≈2.5 se mantiene, se
encoge o crece al presupuesto de producción (26000 iters, t≈20). Son ~3.3 h de
GPU y no cabían esta noche.

### Detalle que refuerza la recomendación

`warm`, con **13 ciclos**, deja menos divergencia (0.00668) que `punto_o16c8` con
**65 ciclos** (0.00764). Cinco veces menos trabajo y mejor convergido.

Su Cl (0.5973) queda por encima de todos los presupuestos del suavizador actual,
incluido el 16×8. Lo consistente con su divergencia es que esté mejor convergido
que todos ellos; pero conviene anotar que cae **fuera** de la banda de dispersión
de las corridas de presupuesto alto (0.5578-0.5801), así que esa lectura es
razonable, no demostrada.

### Comprobación original, para el registro

1. Encolado y adelantado en prioridad (`punto_o12c6`, `punto_o16c8`, a 4000
   iters): si subir el presupuesto por encima de 8×5 apenas mueve Cl y Cd, la
   curva ha aplanado ahí y el baseline vale como referencia.
2. Si sí se mueve, hace falta repetir la comparación al presupuesto de
   producción (26000 iters, t≈20). Son ~3.3 h de GPU y no cabían esta noche.

## Dónde vive la divergencia que queda

Había dos explicaciones incompatibles del suelo de divergencia y convenía
separarlas antes de invertir en el precondicionador:

- **A**: la métrica engaña. La proyección no corrige, por diseño, ni la interfaz
  sólido-fluido (el ghost-cell la reescribe después) ni las capas adyacentes a
  las salidas (su término independiente se pone a cero a propósito). Si casi toda
  la divergencia vive ahí, el solver hace su trabajo y el número es un artefacto
  de promediar sobre celdas que nadie va a corregir.
- **B**: hay divergencia real repartida por el interior, y entonces resolver bien
  el Poisson no la está quitando.

Medido (malla 1772×713, 400 iters):

| zona | % de \|div\| | % de celdas |
|---|---|---|
| borde del dominio (2 capas) | 0.04 % | 0.79 % |
| interfaz del perfil (1-3 celdas) | 21.50 % | **0.24 %** |
| cerca del perfil (4-12 celdas) | 15.15 % | 0.74 % |
| **interior lejano (>12 celdas)** | **63.31 %** | 98.24 % |
| borde + interfaz | 21.53 % | 1.03 % |

**Es el caso B.** El 63 % de la divergencia residual está en el interior lejano,
repartida por el 98 % de las celdas. No es un artefacto de promediar sobre celdas
intocables: el borde del dominio aporta un 0.04 % y la interfaz un 21.5 %.

Eso sí, la interfaz está **90 veces concentrada** (21.5 % de la divergencia en el
0.24 % de las celdas), así que es un foco real aunque no domine el total.

La lectura encaja con el desacoplamiento par-impar, que actúa en todo el dominio
y no sólo en los contornos, y descarta que el problema sea el tratamiento de la
frontera inmersa —lo cual es coherente con que `mg_apply_ibm_each_outer=false`
diera un resultado bit-idéntico.

## El techo, medido — y un modelo de coste que lo hace operativo

La curva de presupuesto degrada mucho y satura en velocidad:

| presupuesto | ciclos/paso | ms/paso | ×vel | ΔCl | ΔCd | div |
|---|---|---|---|---|---|---|
| 8×5 (producción) | 35.0 | 229.6 | 1.00 | — | — | 0.00869 |
| 4×3 | 12.0 | 104.1 | 2.21 | −4.98 % | +12.14 % | 0.0124 |
| 2×3 | 6.0 | 71.4 | 3.22 | −15.68 % | +29.73 % | 0.0172 |
| 2×2, pre/post=1 | 4.0 | 60.7 | 3.78 | −27.92 % | +53.47 % | 0.0203 |
| 1×2, pre/post=1 | 2.0 | 50.8 | 4.52 | −49.36 % | +71.80 % | 0.0282 |

El coste por paso resulta ser **afín en el número de V-cycles**, con un ajuste
exacto sobre los cinco puntos (R² = 1.0000):

```
ms/paso  =  39.2 ms fijos  +  5.43 ms por V-cycle
```

- La parte fija —advección, difusión, IBM, fuerzas— es **39 ms, el 17 %** del
  paso actual.
- **El techo real, con la proyección gratis, es 5.86×.**

**Corrección.** Antes escribí en este mismo informe que el techo era 3.58×,
calculado suponiendo que a 4 ciclos la proyección ya era residual. No lo era:
quedaba overhead por outer sin eliminar, y tomé un punto intermedio por
asíntota. `punto_o1c2` alcanza 4.52× y lo desmiente.

El desglose por componente que imprime el propio simulador (en ms por paso, no
en porcentaje: el porcentaje cambia sólo porque cambia la proyección y esconde
que el resto se queda igual) confirma el modelo y lo afina:

| variante | ms/paso | Advección | Difusión | Proyección | Guardado/fuerzas |
|---|---|---|---|---|---|
| baseline | 228.9 | 23.1 | 4.3 | **197.8** | 3.2 |
| `punto_o4c3` | 103.5 | 22.9 | 4.1 | 72.8 | 3.2 |
| `punto_o2c3` | 70.8 | 23.0 | 4.2 | 39.8 | 3.3 |
| `punto_o1c2` | 50.2 | 24.6 | 4.2 | 17.6 | 3.2 |

La advección son **23 ms y no se mueve** entre variantes, como debe. Con esto los
dos techos quedan separados y sin ambigüedad, porque responden a preguntas
distintas:

- **7.35×** si la proyección desapareciera del todo (quedan 31 ms de advección,
  difusión, fuerzas y demás).
- **5.86×** si los V-cycles fueran gratis pero se mantuviera la estructura de la
  proyección. La diferencia, ~8 ms, es coste fijo por paso dentro de la
  proyección: IBM por outer, divergencias, guard de residual, rollback — es
  decir, overhead que no escala con los ciclos y que también se puede atacar
  (variantes `singuard`, `sinrollback`, `ligero`, en cola).

Y marca dónde habría que ir después: si la proyección bajara a ~30 ms, el paso
quedaría en ~65 ms y **la advección pasaría a ser el 35 %**.

Lo útil del modelo es que reduce toda la pregunta a un solo número: **cuántos
V-cycles necesita un precondicionador para dar la precisión del baseline.**

| ganancia buscada | ciclos/paso necesarios | hoy |
|---|---|---|
| 1.5× | 21.0 | 35.0 |
| 2.0× | 13.9 | 35.0 |
| **2.78× (`warm_o4c3`)** | **8.0** | consigue **7.0** |
| 3.0× | 6.9 | 35.0 |
| 4.0× | 3.4 | 35.0 |

Con esto, cada variante se juzga sin ambigüedad. El suavizador por líneas cuesta
más por ciclo, así que tiene que bajar de 35 ciclos mucho más que
proporcionalmente; los niveles profundos cuestan casi lo mismo por ciclo, así que
les basta con reducir el conteo.

## Resultado principal: la semilla de presión

`warm_start` — arrancar el Poisson del outer 0 desde la presión del paso
anterior, en vez de desde cero.

| | baseline | `warm` |
|---|---|---|
| it/s | 4.36 | **8.21 (1.88×)** |
| ciclos multigrid por paso | 35.01 | **12.68** |
| divergencia media | 0.00869 | **0.00668 (23 % mejor)** |
| Cl | 0.5578 | 0.5973 (+7.08 %) |
| Cd | 0.02863 | 0.02881 (+0.63 %) |

Es 1.88× más rápido **dejando menos divergencia**. La comparación limpia, a
igualdad de trabajo, es todavía más clara:

| a ~12 ciclos por paso | divergencia |
|---|---|
| `punto_o4c3` (12.0 ciclos) | 0.0124 |
| `warm` (12.7 ciclos) | **0.00668** |

Con el mismo trabajo deja **1.85× menos divergencia**. No es un recorte de
presupuesto disfrazado: es el precondicionador haciendo más con lo mismo.

Y con presupuesto recortado la ganancia se compone:

| variante | ×vel | ciclos | divergencia | Cl | Cd |
|---|---|---|---|---|---|
| baseline (8×5) | 1.00 | 35.0 | 0.00869 | 0.5578 | 0.02863 |
| `punto_o4c3` (4×3, sin semilla) | 2.21 | 12.0 | 0.0124 | 0.5300 | 0.03211 |
| `warm` (8×5) | 1.88 | 12.7 | **0.00668** | 0.5973 | 0.02881 |
| **`warm_o4c3`** (4×3) | **2.78** | 7.0 | **0.00700** | 0.5781 | 0.02696 |

A igual presupuesto (4×3), la semilla baja la divergencia de 0.0124 a 0.0070
—1.77×— **y además** dispara la lógica adaptativa de ciclos, que cae de 12 a 7.
Por eso gana en velocidad y en convergencia a la vez.

Las dos configuraciones útiles, contra el 5.86× de techo:

- **`warm` + 4×3 → 2.78×**, con la divergencia todavía un 19 % mejor que la de
  producción actual.
- **`warm` + 8×5 → 1.88×**, con la divergencia un 23 % mejor.

### Seguridad de la semilla

La semilla deja de ser una conjetura razonable si el problema cambia, y entonces
arrancar desde ella es peor que arrancar de cero. Importa en concreto porque los
barridos polares cambian α a mitad de corrida (`plan_polar`). Verificado:

| situación | ¿usa semilla? | esperado |
|---|---|---|
| sin semilla previa | no | no |
| tras guardar una | sí | sí |
| **tras cambiar α** | **no** | **no** |
| vuelve a guardarse | sí | sí |
| malla de otra forma | no | no |
| en un outer distinto del 0 | no | no |
| con la opción apagada | no | no |

Se invalida explícitamente en `cambiar_angulo_ataque` y en el rollback por NaN.

Nota sobre el modelo: se reajustó tras repetir `punto_o2c2`, cuya medida original
estaba contaminada por el solape de simulaciones. La repetición dio Cl, Cd y
divergencia **bit-idénticos** y sólo el cronometraje distinto (16.49 frente a
15.60 it/s), lo que confirma que el solape afectaba al reloj y no a la física.
Con el dato limpio el ajuste pasa de R²=0.9996 a **R²=1.0000**.

### La puerta de aceptación estaba mal especificada, y hay que decirlo

El ΔCl de +7.08 % se sale de la puerta del 0.5 %, pero marcarlo "FUERA" sería
descartar justo lo que se busca. **La puerta suponía que el baseline es la
referencia correcta, y eso sólo vale para variantes que degradan.**

El signo lo delata. En la curva de presupuesto, menos convergencia da Cl más bajo
de forma monótona:

| ciclos | divergencia | Cl |
|---|---|---|
| 2.0 | 0.0282 | 0.2825 |
| 4.0 | 0.0203 | 0.4021 |
| 6.0 | 0.0172 | 0.4704 |
| 12.0 | 0.0124 | 0.5300 |
| 35.0 (baseline) | 0.00869 | 0.5578 |
| **12.7 (`warm`)** | **0.00668** | **0.5973** |

`warm` cae fuera de la curva por el lado bueno: menos divergencia y Cl más alto.
Está **mejor convergido que el baseline**, así que su ΔCl mide el error del
baseline, no el suyo. El comparador ahora lo marca `+conv` en vez de `FUERA`.

### Hasta dónde llega esto y hasta dónde no

Extrapolando linealmente Cl frente a divergencia hacia divergencia cero sale
Cl ≈ 0.701, lo que implicaría un −20 % en el baseline y un −15 % en `warm`.

**Ese número no es de fiar y no debe citarse.** Es una extrapolación lineal muy
por fuera del rango de los datos (de div=0.0087 hasta div=0), sobre una relación
que no hay motivo para creer lineal ahí, y a t≈2.5 en pleno transitorio. No
demuestra nada sobre las polares publicadas.

Lo que sí queda establecido, y es más estrecho: `warm` deja menos divergencia que
el baseline y da un Cl distinto, luego **el baseline no está convergido en
presupuesto de proyección**. Cuánto importa eso a t=20 lo dirán `punto_o12c6` y
`punto_o16c8`, encolados y adelantados en prioridad.

## El suavizador por líneas: la teoría se cumple, la GPU la mata

A igualdad exacta de trabajo (4 V-cycles), frente al suavizador punto a punto:

| a 4 ciclos | divergencia | Cl | ΔCl | ms por ciclo | ×vel |
|---|---|---|---|---|---|
| punto (`punto_o2c2`) | 0.0203 | 0.4021 | −27.92 % | 5.95 | 3.58 |
| líneas (`lines_o2c2`) | **0.0115** | **0.5602** | **+0.42 %** | **30.26** | 1.42 |

La convergencia hace exactamente lo que predice la teoría de multigrid
anisótropo: con 4 ciclos reproduce el Cl que el suavizador actual necesita 35
ciclos para alcanzar, con un ΔCl de +0.42 %. El diagnóstico del hallazgo 2 era
correcto.

Pero cuesta **5.1× más por ciclo** y sólo reduce la divergencia 1.77×. Para
compensar 5.1× de coste haría falta reducir los ciclos 5.1× a igual calidad, y
sólo los reduce 1.8×. **Pierde**: 1.42× de velocidad, que es lo mismo que se
consigue recortando presupuesto sin más.

La causa del coste está medida: un barrido de líneas cuesta 49× uno rojo-negro, y
**el 69 % de eso son las líneas X**, donde cada hilo recorre una fila y los hilos
vecinos acceden a direcciones separadas `nx` — ningún acceso coalesce. Las
líneas Y sí coalescen y cuestan un tercio. La variante `linesy_*` prueba si con
sólo Y (que cubre la estela, la zona anisótropa más extensa) el balance se da la
vuelta.

### Sólo líneas en Y: no vale, y de paso destapa un acoplamiento latente

`linesy_o2c2` sale a **0.55×, más lento que el baseline**, cuando el kernel medido
por separado dice que las líneas Y solas son 3.3× más baratas que Y+X. La
aritmética señala dónde se va el tiempo:

| | esperado | medido (descontando la parte fija) |
|---|---|---|
| Y+X, 8 barridos | 32 ms | 121 ms |
| **Y sola, 8 barridos** | **10 ms** | **377 ms** |

El exceso de ~360 ms no puede salir del suavizador. Sale del guard: cuando el
V-cycle amplifica el residuo, el solver lo descarta y cae a
`gs_sweeps = max(300, ...)` barridos en el nivel fino. Con el suavizador barato
eso cuesta 300 × 0.08 = 24 ms y pasa inadvertido; con líneas cuesta
300 × 1.2 = **360 ms**, que es justo el exceso observado.

Dos conclusiones:

1. **Las líneas sólo en Y no bastan.** Sin suavizar la dirección X el V-cycle
   diverge y el guard salta en cada paso. El diseño alternado Y/X era necesario,
   no un lujo.
2. **El guard es un multiplicador de coste oculto.** Esa constante 300 está
   calibrada para un suavizador barato. Cualquiera que cambie el suavizador tiene
   que revisarla, o pagará un coste que no aparece en ninguna cuenta de ciclos
   —de hecho `mg_cycles_media` sigue marcando 4.0 mientras el paso se multiplica
   por seis.

Confirmación directa, si se quiere cerrar del todo (no la he ejecutado por no
alargar la cola):

```bash
.venv/bin/python scripts/agent_tests/bench_opt.py --run linesy_singuard \
    --opt line_smoother_y --iters 400 --max-outer 2 --cycles 2 --pre 1 --post 1 \
    --set mg_guard_residual_every_outer=false
```

Si al quitar el guard el tiempo se desploma, queda demostrado.

### La lección, que vale más que el resultado

| | mejora de convergencia | coste por ciclo | balance |
|---|---|---|---|
| semilla de presión | 1.77× | **6.01 ms** (sin cambio) | **gana 2.78×** |
| líneas | 1.77× | 30.26 ms (5.1×) | pierde |

Las dos mejoran la convergencia lo mismo. Gana la que no encarece el ciclo. En
un solver que ya está limitado por lanzamiento de kernels, **una optimización
algorítmica sólo sirve si es casi gratis**, y eso descarta de entrada toda una
familia de mejoras "mejores" sobre el papel.

## La divergencia es un mal indicador de convergencia en este solver

`punto_o12c6` sube el presupuesto a 1.5× el de producción. El resultado separa
dos cosas que parecían la misma:

| | baseline (8×5) | `punto_o12c6` (12×6) | `warm` (8×5) |
|---|---|---|---|
| ciclos/paso | 35.0 | 40.9 | 12.7 |
| **divergencia** | 0.00869 | **0.00863 (−0.7 %)** | 0.00668 (−23 %) |
| **Cl** | 0.5578 | **0.5801 (+4.00 %)** | 0.5973 (+7.08 %) |
| Cd | 0.02863 | 0.02969 (+3.70 %) | 0.02881 (+0.63 %) |

Con un 17 % más de ciclos la divergencia **no se mueve** —está en su meseta— pero
Cl cambia un **+4.00 %**, que es del orden de toda la banda del GCI para Cl
(4.25 %).

Y encaja exactamente con el desacoplamiento par-impar del hallazgo 1-bis: la
divergencia medida está dominada por la componente de alta frecuencia, que
**ninguna iteración puede eliminar**, así que satura pronto y aparenta
convergencia. La componente suave sí sigue mejorando, y es de ella de la que
dependen Cl y Cd.

**Consecuencia práctica: la divergencia media no sirve como criterio de parada de
la proyección en este solver.** Es justo lo que usa `tol_div`, y explica por qué
`frac_converge = 0.0` convive con un campo que a efectos de fuerzas todavía
está lejos de converger. Un criterio útil tendría que mirar el cambio de Cl y Cd
entre outers, o filtrar el modo tablero antes de promediar.

Nótese también que `warm`, con **un tercio de los ciclos** de `punto_o12c6`,
queda mejor por las dos medidas: menos divergencia (0.00668 frente a 0.00863) y
Cl más alto (0.5973 frente a 0.5801).

## Cómo se juzga una variante

Recortar el presupuesto acelera y degrada a la vez, así que comparar dos
variantes por su `it/s` a secas no dice nada: siempre gana la que menos trabajo
hace. La pregunta útil es si una variante es más rápida **a igual precisión** que
el suavizador actual.

Por eso las variantes `punto_*` (mismo solver, distinto presupuesto) no son un
control secundario: son la **curva de referencia**. Cualquier otra variante se
sitúa contra ella interpolando en el error que consigue. Quedar por encima es
aportar algo; quedar por debajo significa que el mismo resultado se lograba antes
simplemente gastando menos.

```bash
.venv/bin/python scripts/agent_tests/bench_opt.py --pareto
```

| variante | error | ×vel | ×ref | ventaja |
|---|---|---|---|---|
| baseline (8 outers × 5 ciclos) | 0.00 % | 1.00 | 1.00 | — |
| `punto_o4c3` (4 × 3) | 12.14 % | 2.21 | 2.21 | 1.00 |

`error` = max(\|ΔCl\|, \|ΔCd\|) frente al baseline.

## Cómo activarlo en producción

Nada está activo por defecto. Dos formas, según convenga.

**Por entorno**, sin tocar ningún script de campaña:

```bash
OPT_SOLVER=warm_start .venv/bin/python scripts/agent_tests/richardson_polar_domC.py
```

**Por código**, en `scripts/RunGA.py`. La semilla se activa importando y
llamando a `opt_solver.enable("warm_start")` una vez al arrancar; el presupuesto
va en `CONFIG`:

```python
CONFIG = {
    ...
    'mg_max_outer': 4,          # era 8
    'sim_extra_params': {..., 'mg_cycles_per_outer': 3},   # el default es 5
}
```

`mg_cycles_per_outer` no está entre las claves explícitas que arma
`simular_perfil`, pero `sim_extra_params` se aplica encima de todas
(`RunGA.py:1315`), así que ese es el sitio.

Verificación de que quedó activo, en cualquier corrida:

```bash
.venv/bin/python -c "import opt_solver; print(opt_solver.active())"
```

### Advertencia antes de activarlo

**Cambia los resultados.** No es neutro y no debe presentarse como tal: la
semilla deja el campo mejor convergido, así que da un Cl distinto —y mejor— que
la configuración con la que se calcularon las polares publicadas.

Eso obliga a elegir, y la elección no es mía:

- **Para trabajo nuevo**: activarlo. Es más rápido y más convergido.
- **Para ampliar el estudio ya cerrado** (añadir ángulos o mallas a la polar de
  Richardson): **no mezclar**. Un punto calculado con semilla y otro sin ella no
  son comparables, y la diferencia (~4-7 % en Cl) es del orden de la banda de
  malla que el estudio pretende medir. O se rehace todo con la misma
  configuración, o se deja como está.

## Protocolo de validación

Puerta de aceptación, fijada por el usuario: **|ΔCl| y |ΔCd| ≤ 0.5 %** frente al
baseline, y la divergencia no puede empeorar más de un 5 %.

El 0.5 % es la sensibilidad de ventana temporal ya medida en este proyecto
(`results/asintotico_alpha4_domC`: L/D de cola 34.038 en t=20 frente a 34.049 en
t=41.5) y está un orden por debajo de la banda entre mallas del GCI
(Cd 6.4 %, L/D 7.6 %).

Además de la media de la ventana se compara la **trayectoria completa** Cl(t),
que es más estricta: detecta variantes que acaban en el mismo sitio recorriendo
un camino distinto.

Screening a 4000 iters; lo que pase el screening se confirma después al
presupuesto de producción (26000 iters, t≈20).

## Incidencia de método, para que conste

Al reordenar la cola maté el runner al que otro estaba esperando, y ese otro
arrancó de inmediato: durante unos 3 minutos hubo **dos simulaciones compartiendo
la GPU**. Eso no altera los resultados físicos, pero sí deprime el `it/s`, que es
la mitad de lo que se está midiendo. La variante afectada (`punto_o2c2`, medida a
15.11 it/s en vez de los ~16.5 que llevaba antes del solape) se **descarta y se
repite** al principio del último tramo.

Los runners ahora esperan al *proceso* del anterior, no a un patrón de comando:
esperar por patrón deja una ventana entre que un comando acaba y arranca el
siguiente, que es exactamente por donde se coló el solape.

Segunda trampa de la misma familia, por si alguien retoca estos scripts: un
runner que espera con `pgrep -f "bench_opt.py --queue"` **también se encuentra a
sí mismo reflejado** en cualquier otro proceso cuya línea de comando contenga esa
cadena — incluidos los bucles de vigilancia que uno lanza para seguir el
progreso. Estuvo parado un rato por eso: la cola había terminado, pero mis
propios `until [ -f ... ] || ! pgrep -f "bench_opt.py --queue"` mantenían viva la
coincidencia. Conviene esperar por el nombre del *script* runner
(`pgrep -f cola_resto.sh`), que no aparece en ningún otro sitio.

## Estado

Cola corriendo. Resultados en `results/bench_opt/*.json`, series en
`results/bench_opt/series/*.npz`, tabla con:

```bash
.venv/bin/python scripts/agent_tests/bench_opt.py --compare
```

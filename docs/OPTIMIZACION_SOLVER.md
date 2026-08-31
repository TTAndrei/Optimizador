# Optimización del solver de presión

Trabajo del 2026-08-24. El simulador pasa de **7.73 h a 1.43 h** por polar
completa (6 ángulos, dx=0.002, dominio C), con la divergencia residual **por
debajo** de la que daba la configuración anterior.

Las optimizaciones vienen **activadas por defecto**. Para volver al
comportamiento previo:

```bash
OPT_SOLVER_OFF=1 .venv/bin/python <script>
```

---

## 1. Estructura nueva

Todo lo añadido vive fuera del simulador. `Simulador2D.py` sólo tiene
**sustituciones línea por línea** que delegan en un módulo aparte, y cada una
conserva la rama original intacta.

| fichero | qué es |
|---|---|
| `opt_solver.py` | **Nuevo.** Las 8 opciones. Cada una con rama "off" idéntica al código original |
| `Simulador2D.py` | 9 puntos de sustitución. **41 inserciones, 11 borrados** sobre 8866 líneas |
| `scripts/RunGA.py` | Presupuesto de proyección en `CONFIG` + una clave que se perdía |
| `scripts/agent_tests/bench_opt.py` | **Nuevo.** Banco A/B con instrumentación del solver |
| `scripts/agent_tests/diag_divergencia.py` | **Nuevo.** Dónde y de qué tipo es la divergencia residual |
| `scripts/agent_tests/perfilar_bucle.py` | **Nuevo.** Perfilado con Nsight, sólo del bucle de tiempo |
| `scripts/agent_tests/polar_optimizada.py` | **Nuevo.** Polar de validación por la ruta de producción |
| `scripts/agent_tests/comparar_polares.py` | **Nuevo.** Comparación contra el estudio publicado |
| `scripts/agent_tests/desglose_componentes.py` | **Nuevo.** Reparto de tiempo por componente |

### Los 9 puntos de sustitución en `Simulador2D.py`

| línea | antes | ahora | opción |
|---|---|---|---|
| 16 | — | `import opt_solver` | — |
| 1092 | `_coarsen_mask(...)` | `opt_solver.coarsen_solid(...)` | `coarse_mask_majority` |
| 1695 | — | `invalidate_warm` en `cambiar_angulo_ataque` | seguridad |
| 3367 / 4121 | — | desvío al suavizador por líneas | `line_smoother` |
| 3388 / 4145 | cuerpo de `_aplicar_bc_mg` | `opt_solver.aplicar_bc_mg(...)` | `fast_masks` |
| 3428 / 4175 | `min(n_levels, 2)` | `opt_solver.mg_depth(...)` | `deep_levels` |
| 3430 / 4177 | `coarse_solve_iters = 20` | `opt_solver.coarse_iters(...)` | `deep_levels` |
| 3671 / 4351 | `p_corr.fill(0.0)` | `opt_solver.init_guess(...)` | `warm_start` |
| 3761 / 4401 | — | `store_guess` + `invalidate_warm` | `warm_start` |
| 1898 | `wx = x - j0` | `opt_solver.bilinear_weights(...)` | `interp_float32` |

Los pares (3xxx / 4xxx) son las dos variantes de proyección: `legacy_centered`
—la que usa producción— y `compatible_flux`. Se cablearon ambas para no dejar
una inconsistencia silenciosa.

### Qué queda activo y qué no

| opción | estado | ganancia | ¿cambia resultados? |
|---|---|---|---|
| `fast_masks` | **ON** | ×1.32 | **No. Bit a bit idéntico** |
| `interp_float32` | **ON** | ×1.17 | +0.012 % en Cl a t=20 |
| `warm_start` | **ON** | ×1.88 | Sí — deja el campo mejor convergido |
| `coarse_mask_majority` | **ON** | ×1.14 | Sí — mejor factor multigrid |
| `deep_levels` | OFF | **empeora** | divergencia +62 %, 0.34× |
| `line_smoother` | OFF | pierde | converge bien, cuesta 5.1× por ciclo |
| `line_smoother_y` | OFF | diverge | no suaviza la dirección X |
| `coarse_mask_none` | OFF | — | sólo cota experimental |

Más el presupuesto de proyección: **8 outers × 5 ciclos → 2 × 3**.

---

## 2. Hallazgos sobre el solver

### 2.1 El multigrid nunca alcanzaba su tolerancia

`frac_converge = 0.0`: ni un solo paso de 4000 bajaba de `tol_div`. El bucle
externo gastaba **siempre** sus 8 outers, y 40 V-cycles reducían la divergencia
sólo ×1.75.

La causa es de discretización, no del solver iterativo. La malla es **colocada**
—u, v y p en el mismo punto— así que la proyección resuelve `L p = D u*` y
corrige `u = u* − G p`, dejando `(L − D·G) p`. Los símbolos de Fourier en 1D
uniforme:

| θ/π | símbolo L | símbolo D·G | error |
|---|---|---|---|
| 0.05 | −0.0217 | −0.0215 | 0.5 % |
| 0.49 | −1.9509 | −0.9994 | 48.8 % |
| **1.00** | **−4.0000** | **0.0000** | **100 %** |

En el modo tablero el gradiente centrado **se anula**: hay divergencia que
ninguna presión puede eliminar. Medido sobre el campo real, **el 46.2 % de la
divergencia residual es modo par-impar**.

La ruta de flujo de cara no escapa: `_build_projection_faces`
(`Simulador2D.py:2221`) hace interpolación lineal pura, sin término de
amortiguamiento tipo Rhie-Chow, así que la divergencia de cara es
algebraicamente la diferencia centrada.

**No es un fallo a corregir.** Es una propiedad conocida de las mallas
colocadas, y el remedio clásico —Rhie-Chow o malla desplazada— cambiaría la
discretización e invalidaría el estudio de malla ya cerrado. Se documenta como
el techo del método.

### 2.2 La divergencia media es mal indicador de convergencia

Subir el presupuesto de 8×5 a 12×6 **no mueve la divergencia** (−0.7 %) pero
cambia Cl un **+4.00 %**. La divergencia medida está dominada por la componente
de alta frecuencia, que ninguna iteración puede eliminar, así que satura pronto
y aparenta convergencia; la componente suave sigue mejorando, y es de ella de la
que dependen Cl y Cd.

Eso es justo lo que usa `tol_div` como criterio de parada. Un criterio útil
tendría que mirar el cambio de Cl y Cd entre outers, o filtrar el modo tablero
antes de promediar.

### 2.3 La malla es fuertemente anisótropa

| relación de aspecto por celda | |
|---|---|
| p50 | 1.00 |
| p75 | 28.1 |
| p90 – p100 | 50.0 |

El 30.4 % de las celdas pasa de 8 y el 24.5 % pasa de 32. Lo que degrada al
suavizador es el **cuadrado** de esa relación, porque el acoplamiento del
laplaciano de 5 puntos va como 1/h²: la anisotropía efectiva del operador llega
a **2500**. Gauss-Seidel punto a punto tiene factor de suavizado ≈ 1 en la
dirección débil, así que **en un tercio del dominio no suaviza**.

Se implementó relajación por líneas alternada (cebra + Thomas), verificada
contra el rojo-negro en tres regímenes (acuerdo 2.5e-04 a 8.4e-06). Con 4 ciclos
reproduce el Cl que el suavizador actual necesita 35 ciclos para alcanzar. **Y
aun así pierde**: cuesta 5.1× por ciclo y sólo reduce la divergencia 1.77×.

La lección general, que vale más que el resultado:

| | mejora de convergencia | coste por ciclo | balance |
|---|---|---|---|
| semilla de presión | 1.77× | sin cambio | **gana 1.88×** |
| líneas | 1.77× | 5.1× | pierde |

En un solver limitado por lanzamiento de kernels, **una mejora algorítmica sólo
sirve si es casi gratis**.

### 2.4 El coarsening engordaba el perfil

`_coarsen_mask` hacía OR sobre un bloque 3×3: bastaba una celda fina sólida para
que la gruesa lo fuera. Una línea de una celda de espesor —como un borde de
salida afilado— pasa a **11 celdas sólidas** tras un nivel. `coarse_mask_majority`
exige que lo sea la mayoría del bloque, y entonces esa línea **desaparece** del
nivel grueso, que es preferible: el nivel grueso sólo corrige error de baja
frecuencia y un Poisson sin el cuerpo sigue siendo buen precondicionador,
mientras que un cuerpo deformado no lo es.

Verificado que el muestreo es idéntico al original en 240 casos aleatorios.

### 2.5 La profundidad del multigrid no se puede aumentar

El V-cycle tenía los niveles topados a fuego en `min(n_levels, 2)`. Con 1.26 M
celdas eso deja el nivel más grueso en **78 854 celdas**: el multigrid nunca
llega a una malla lo bastante pequeña para resolverla.

Levantar el tope **empeora**: con 6 niveles la divergencia sube un 62 % y el
solver va 3× más lento. La causa es la misma anisotropía — el coarsening isótropo
2×2 sobre celdas con relación de aspecto 50 produce un operador grueso que no
representa al fino, y cada nivel compone el error.

**La anisotropía rompe las dos mitades del multigrid a la vez**: el suavizador y
el coarsening. Arreglar la segunda pediría semi-coarsening, que no se ha
implementado.

---

## 3. Hallazgos sobre rendimiento

### 3.1 Modelo de coste

El tiempo por paso es afín en el número de V-cycles (R² = 1.0000):

```
ms/paso  =  39.2 ms fijos  +  5.43 ms por V-cycle
```

La parte fija —advección, difusión, IBM, fuerzas— es el 17 % del paso original.
Reduce cualquier optimización a un solo número: **cuántos V-cycles necesita para
dar la precisión del baseline**.

### 3.2 Fuga de float64 en la interpolación

`_bilinear_interpolate` calculaba los pesos como `wx = x - j0`, con `x` float32
y `j0` int32. La regla de promoción de NumPy y CuPy da **float64**, porque un
int32 no cabe exacto en un float32. A partir de ahí toda la fórmula bilineal se
evaluaba en doble precisión.

Los recuentos del perfil cuadran uno a uno con esa fórmula: 35 restas
`float32-int32` por iteración, 72 restas de `1−w`, 102 multiplicaciones, 71
sumas. **El 26 % del tiempo de GPU.**

Peor: `advect_sa` hace `self.nu_tilde = nt`, o sea rebindea, así que el campo se
quedaba en float64 **para el resto de la simulación**. Con eso el modelo
Spalart-Allmaras entero corría en doble precisión, incluidos `g**6` y `**(1/6)`,
que en una GeForce van a **1/64** de la velocidad de float32: 702 µs por
instancia, tres por iteración.

En el camino de las velocidades no compraba nada: el resultado se asignaba
dentro de arrays float32 y el doble se truncaba acto seguido.

**Riesgo evaluado y descartado con medida.** El float32 podría perder dígitos por
cancelación en `fv2 = 1 − chi/(1 + chi·fv1)`, pero sólo a partir de chi > 1e4.
La distribución real de chi en producción es p50 = 3.0, p99 = 12.0, **máximo
50.4**: el mecanismo no se activa. Confirmado a t=20, donde el cambio en Cl es
de **+0.012 %**.

**No se ha tocado ningún otro float64.** Los pesos de diferencias finitas se
siguen calculando en float64 desde `pos_1d_f64` y casteando al final —que es el
mecanismo que preserva la simetría del solver— y la rasterización del polígono
tampoco cambia. Además, por el lema de Sterbenz la resta `x − j0` es exacta en
float32, así que el peso sale bit a bit igual; lo que cambia es la acumulación
posterior.

### 3.3 El indexado booleano costaba el 23 % de la GPU

Escribir `p[mascara] = v` obliga a CuPy a convertir la máscara en índices: un
prefix sum sobre todo el array, un gather y un scatter. El perfil lo enseña en
crudo — `prepare_array_indexing`, `scan_naive`, `take`, `bsum_shfl` y
`scatter_update_mask` sumaban el **23 %** del tiempo de GPU, con **592 scans por
iteración**.

`cp.copyto(dst, src, where=mascara)` es un solo kernel elementwise. Medido sobre
la malla de producción: **6.1× más rápido** en la máscara 2D y 5.9× en las de
borde. Y el coste no lo marcan los datos sino la maquinaria: **142 µs para una
máscara de 713 elementos**.

Las máscaras de borde de `_aplicar_bc_mg` son además **constantes** —dependen de
la geometría, no del campo— y se recalculaban en cada llamada. Ahora se cachean
por nivel.

Es la misma operación escrita de otra forma: **resultado bit a bit idéntico**,
verificado en 8 métricas escalares y en las trayectorias completas.

### 3.4 Lo que no salió a cuenta

**Chebyshev: descartado con números.** El suavizador polinómico sustituiría al
rojo-negro con 1 lanzamiento en vez de 2 y todos los hilos activos en vez de la
mitad. Pero el kernel está **limitado por ancho de banda, no por ocupación**:
16.4 MB por lanzamiento en el nivel fino → 27 µs al límite del bus, medido ~40 µs.
El 50 % de hilos ociosos no es lo que lo frena. Lo que Chebyshev ahorra es
tráfico (2 lecturas + 1 escritura → 1 + 1), o sea **1.5× como mucho**, sobre un
kernel que es el 8.9 % del paso: **3.0 % de ahorro**, por debajo del umbral del
4 % fijado de antemano.

**CUDA Graphs: sin evaluar.** Se descartó primero con una cuenta mal hecha
(346 lanzamientos por paso estimados; Nsight midió **3199**), y luego quedó
desplazado por hallazgos mayores. Sigue pendiente y con margen: la configuración
actual corre al 68.5 % de ocupación de GPU.

**Otras vías, descartadas con medida**: `projection_variant="compatible_flux"`
(peor en todo), `usar_adjoint_correction` (blowup en la iteración 0),
`mg_apply_ibm_each_outer=false` (bit-idéntico, sin efecto), portar a JAX (techo
dentro del mismo margen, semanas de reescritura), Coarse-Grid Projection
(invalidaría el estudio de malla), solver FFT (exige malla uniforme separable).

---

## 4. Validación

### 4.1 Neutralidad del cableado

Con todas las opciones apagadas, la variante `equiv` reproduce el baseline con
**14/14 métricas idénticas** y las trayectorias completas `Cl(t)`, `Cd(t)`,
`div(t)` y `cycles(t)` iguales **bit a bit** (delta máximo 0.000e+00 sobre 160
muestras).

### 4.2 El control reproduce lo publicado

Una corrida a t=20 con la configuración anterior, por el banco de pruebas, da
**Cl = 0.680389 y Cd = 0.019986**: idénticos hasta el último decimal al punto
publicado. El andamiaje queda descartado como explicación de cualquier
diferencia.

### 4.3 Las dos rutas de código coinciden

En α=4, el banco de pruebas (`Simulador2D.main` directo) y la polar
(`vn.simular` → `RunGA.simular_perfil`) dan `|dif| = 0.00e+00` en Cl, Cd y L/D.

### 4.4 Polar completa

Perfil ganador del AG, dominio 24×16 cx=6, dx=0.002, Re=1e5, t≈20, sin paradas.

| α | Cl pub | Cl opt | ΔCl | Cd pub | Cd opt | ΔCd | L/D pub | L/D opt | ΔL/D |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.1509 | 0.1390 | −7.89 % | 0.03766 | 0.03687 | −2.09 % | 4.01 | 3.77 | −5.92 % |
| 2 | 0.4461 | 0.4536 | +1.69 % | 0.02326 | 0.02158 | −7.23 % | 19.18 | 21.02 | +9.62 % |
| 4 | 0.6804 | 0.6977 | +2.54 % | 0.01999 | 0.01691 | **−15.38 %** | 34.04 | **41.26** | **+21.19 %** |
| 6 | 0.8559 | 0.8790 | +2.70 % | 0.03658 | 0.03547 | −3.03 % | 23.40 | 24.78 | +5.90 % |
| 8 | 1.0165 | 1.0391 | +2.22 % | 0.06334 | 0.06156 | −2.81 % | 16.05 | 16.88 | +5.17 % |

**α=10 no es comparable**: su única referencia corrió 7950 de 26000 iteraciones.

Coste: **7.73 h → 1.43 h**, aceleración media **5.40×**.

### 4.5 Referencia externa

El simulador tiene un sesgo documentado de sobrestimar Cd (validación previa
contra XFOIL: **+124 % de media**). La polar corregida **se acerca a XFOIL en los
cinco ángulos, sin excepción**:

| α | Cd XFOIL | pub / XFOIL | opt / XFOIL |
|---|---|---|---|
| 0 | 0.01693 | 2.22 | **2.18** |
| 2 | 0.01444 | 1.61 | **1.49** |
| 4 | 0.01520 | 1.31 | **1.11** |
| 6 | 0.01958 | 1.87 | **1.81** |
| 8 | 0.02869 | 2.21 | **2.15** |

Es la evidencia más fuerte porque no depende de nada escrito aquí: XFOIL es
externo y la validación es de otra campaña. Con la reserva obligada de que es un
NACA0012, no el perfil optimizado, así que sólo vale la tendencia.

Dato de contexto: el estudio publicado da **L/D = 34.04 para el perfil
optimizado**, y XFOIL da **35.3 para un NACA0012 sin optimizar** — el perfil
optimizado saldría peor que el de partida, lo cual es raro. Con la corrección
sale 41.26 y esa anomalía desaparece.

---

## 5. Lo que no se sostiene

Por honestidad, y porque son las conclusiones que más fácilmente se citarían mal:

**El mecanismo propuesto para el cambio en Cd no queda demostrado.** La
hipótesis era que la divergencia espuria junto a la pared infla la resistencia y
que el efecto crecería con la carga del perfil. Con tres puntos parecía
cumplirse; los dos siguientes lo desmienten. El patrón real tiene un **pico
agudo en α=4** (−15.4 %) y cae a ~−3 % en α=6 y 8. Hay una explicación
alternativa —α=4 es a la vez el mínimo de la cubeta de resistencia y el punto de
diseño del AG— pero se formuló después de ver los datos y con estos puntos no se
pueden separar esas dos causas.

**α=0 va en dirección contraria** en Cl (−7.89 %, por encima de su CI95 del
4.5 %). Sin explicación.

**La métrica `cl_cp_discrepancy` se retiró del análisis.** Tiene valores con
`converged_clcd = False`, cuando el código sólo la escribe en el mismo bloque
que pone esa bandera a True. Hasta resolver esa inconsistencia no se puede
interpretar.

**Que 41.26 sea el L/D convergido no está demostrado.** Está mejor convergido
que 34.04 por la medida de divergencia, pero también se midió que la divergencia
es mal indicador. Falta correr el optimizado con más presupuesto a t=20 y ver si
se mueve.

---

## 6. Consecuencia para los resultados publicados

**Las polares publicadas y el estudio de Richardson no cambian**: siguen siendo
reproducibles con `OPT_SOLVER_OFF=1` y `mg_max_outer=8, mg_cycles_per_outer=5`.

Pero **la configuración anterior no estaba convergida en presupuesto de
proyección**. Por debajo de 8×5 la degradación es monótona y fuerte; por encima
hay ~4 % de dispersión en Cl sin tendencia clara. Es una incertidumbre numérica
real, del orden de la banda del GCI (Cl 4.25 %), y **no figura junto a la de
malla en el estudio**.

La decisión de qué hacer con eso —rehacer la polar, ampliar la discusión de
incertidumbre, o dejarlo— no es técnica.

**No mezclar configuraciones.** Un punto calculado con las optimizaciones y otro
sin ellas no son comparables, y la diferencia es del orden de lo que el estudio
de malla pretende medir.

---

## 7. `warm_start_filtered` — lo que hace usable esta configuración (2026-08-26)

Actualización posterior al documento original. **Sin esto, `warm_start` con el
presupuesto 2×3 no se puede usar**: revienta por blowup de velocidad en la
iteración **1340** a dx=0.004 y en la **~1740** a dx=0.001, de forma reproducible
en todos los ángulos probados.

### El cruce que lo aisló

Dos diagnósticos en `results/bench_opt/`:

- `diag_blowup.json` — cruce 2×2 de optimizaciones × presupuesto. El fallo
  **necesita las dos cosas a la vez**: optimizaciones ON con presupuesto 8×5 va
  bien; OFF con 2×3 va bien; **ON con 2×3 revienta**. O sea que una opción deja
  de ser inocua cuando la proyección se queda sin ciclos de sobra.
- `diag_flag.json` — leave-one-out. Quitar `warm_start` es lo único que lo evita;
  quitar cualquiera de las otras tres no cambia nada.

### La causa: estructura, no magnitud

`warm_start` no cambia la discretización, solo la semilla de un solver iterativo,
y sería irrelevante si la proyección convergiese. Con presupuesto fijo 2×3 no
converge —el desacoplamiento par-impar hace inalcanzable la tolerancia— así que
la solución **sí** depende de la semilla.

Se probó primero **reescalar la semilla por el cambio de dt**, con el argumento
de que el RHS del outer 0 lleva un término `(ρ/dt)·div(uⁿ)` que solo se anula si
el paso anterior quedó bien proyectado. **No funcionó**: solo movió el fallo de
la iteración 1340 a la 1800. Esa opción queda en `opt_solver.py` como
`warm_start_scaled`, marcada como PROBADO, NO FUNCIONA.

Ese fallo es lo que demostró que el mecanismo era **estructural**: la semilla
arrastra el **modo par-impar (tablero) acumulado**, que el multigrid no reduce
porque en malla colocada lo ve casi como núcleo del operador. Con solo 2×3 ciclos
no hay margen para disiparlo, así que se autoamplifica: divergencia residual →
velocidad espuria → el CFL encoge dt → más error → blowup.

### El arreglo

Un paso de **Jacobi ponderado con ω = ½** sobre la semilla antes de reusarla:

```python
buf[1:-1, 1:-1] = 0.5 * p[1:-1, 1:-1] + 0.125 * (
    p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:])
```

Es un **aniquilador exacto** del modo tablero (los cuatro vecinos suman −4p, así
que da p/2 − p/2 = 0) e **identidad sobre los modos suaves**. Verificado en
unidad: tablero → 0.0 exacto; rampa suave → cambio de 6e-8.

### Validación en GPU

| comprobación | sin filtro | con filtro |
|---|---|---|
| dx=0.004 α=0, 13000 it | **ABORTA it. 1340** | OK 13000/13000, 387 s |
| dx=0.001 α=4, 3000 it | **ABORTA it. ~1740** | OK, cero blowups |
| dx=0.001 α=0, 52000 it | (los 10 puntos fallaban) | **OK 52000/52000** |
| dx=0.002 α=4, exactitud | L/D 41.256 | L/D 41.155 (**−0.25 %**) |

**Coste nulo**: 387 s frente a ~405 s sin `warm_start`.

### Convergencia en presupuesto, medida

Contra lo que se sospechó en su momento, el 2×3 **sí está convergido**. L/D a
dx=0.002, α=4, con `warm_start`:

| presupuesto | ciclos | L/D |
|---|---|---|
| 2×3 | 6 | 41.256 |
| 4×3 | 12 | 41.994 |
| 8×5 | 40 | 42.051 |

Casi 7× el trabajo de proyección mueve el L/D un **1.9 %**. La curva es plana.

### Consecuencia

Con `warm_start_filtered` activo, **las cuatro mallas del estudio final llevan
exactamente la misma configuración de solver**, que es lo que exige un estudio de
Richardson: mezclar configuraciones dentro de la terna sesga el orden aparente,
que es justo la señal que se mide. Ver `resultados_finales/README.md`.

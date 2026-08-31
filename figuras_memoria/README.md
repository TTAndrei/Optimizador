# Figuras de la memoria (capítulos 2, 3 y 4)

Cinco figuras nuevas para el TFG. Cada una se regenera con un script propio en
`scripts/agent_tests/`. Los scripts que necesitan GPU guardan sus campos y series
al lado de la figura, de modo que retocar el dibujo no obliga a repetir la
corrida (`--solo-figura`).

| Figura | Fichero | Script | GPU |
|---|---|---|---|
| 2.A Descomposición de Helmholtz-Hodge | `fig_2A_helmholtz_hodge.png` | `fig_mem_helmholtz.py` | no |
| 3.A Símbolos discretos del laplaciano y de div-grad | `fig_3A_simbolos_discretos.png` | `fig_mem_simbolos.py` | no |
| 3.B Modo de tablero antes y después del filtro | `fig_3B_modo_tablero.png` | `fig_mem_tablero.py` | sí |
| 4.A Convergencia de la proyección en un paso | `fig_4A_convergencia_proyeccion.png` | `fig_mem_proyeccion.py` | sí |
| 4.B Orden aparente frente al ángulo de ataque | `fig_4B_orden_aparente.png` | `fig_mem_orden.py` | no |

---

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

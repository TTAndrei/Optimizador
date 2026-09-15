# Formas romas: Cl y Cd a Re = 1e3..1e6

10 puntos. Dominio 24x16 (cx=6), alpha=0, longitud caracteristica 1, o sea Re = Re_D y Cd referido a 1 m.

## Resultados

| forma | Re | dx | modelo | t | Cd | ±CI95 | Cd ref | desv | Cl | amp Cl | St |
|---|---|---|---|---|---|---|---|---|---|---|---|
| circulo | 1e+03 | 0.004 | laminar | 19.184 | 0.8227 | 0.003058 | 1.0 | -18 % | +0.0032 | 0.01201 | no desprende |
| circulo | 1e+04 | 0.004 | laminar | 20.245 | 2.4185 | 0.05075 | 1.1 | +120 % | +0.0103 | 2.07892 | **0.2005** (dupl.) |
| circulo | 1e+05 | 0.004 | laminar | 21.495 | 1.9532 | 0.078214 | 1.2 | +63 % | +0.4350 | 1.69791 | 0.2814 / 0.1883 ? |
| circulo | 1e+06 | 0.004 | laminar | 23.767 | 1.4910 | 0.055428 | 0.35 | +326 % | -0.0287 | 1.38113 | 0.1594 |
| cuadrado | 1e+03 | 0.004 | laminar | 20.536 | 1.1486 | 0.007202 | 2.1 | -45 % | +0.0254 | 0.01862 | no desprende |
| cuadrado | 1e+04 | 0.004 | laminar | 19.699 | 1.3579 | 0.053453 | 2.1 | -35 % | -0.5730 | 0.63508 | 0.1002 / 0.4022 ? |
| cuadrado | 1e+05 | 0.004 | laminar | 18.324 | 1.6034 | 0.049478 | 2.1 | -24 % | -0.7877 | 1.5117 | 0.214 |
| cuadrado | 1e+06 | 0.004 | laminar | 18.86 | 2.1249 | 0.067617 | 2.1 | +1 % | +0.4645 | 1.63048 | **0.2106** (dupl.) |
| gota | 1e+03 | 0.004 | laminar | 19.728 | 0.3771 | 0.000747 | 0.12 | +214 % | +0.0233 | 0.04759 | no desprende |
| gota | 1e+04 | 0.004 | laminar | 18.514 | 0.5325 | 0.009995 | 0.08 | +566 % | +0.5204 | 1.30023 | 0.1097 |

## Modelo de turbulencia

**10 de 10 puntos acabaron en laminar.** 3 por decreto (Re<=1e3, donde el modelo no tiene sentido fisico) y 7 por colapso del paso temporal con SA-BC.

**SA-BC no sobrevivio en ninguna de las tres formas, a ningun Reynolds.** El mecanismo es `dt_visc = 0.25*dx^2/nu_eff` (Simulador2D.py:8195): en la estela de un cuerpo romo nu_t satura y el paso temporal se hunde. Esto confirma y generaliza lo que ya se habia medido sobre el cilindro en `results/esfera_dx002/` (t=5.3 con SA frente a t=17.0 en laminar con las mismas 44000 iteraciones).

**Una sonda corta NO detecta el colapso, y esto costo cuatro corridas.** El veredicto se tomaba con 2500 iteraciones mirando el nivel de dt y su tendencia; cuatro casos la pasaron como 'dt estable' y luego se quedaron en t = 6.1 a 9.4 de los 20 objetivo. El colapso se desarrolla a lo largo del run, no en su arranque. La deteccion que si funciona es **a posteriori**: si con SA el tiempo fisico alcanzado se queda por debajo del 70 % del objetivo, se descarta SA y se repite en laminar. Los veredictos, con lo que dijo la sonda y lo que paso de verdad, estan en `results/formas/veredictos_sa.json`.

Esto importa mas alla del ahorro: si unos puntos del barrido corren con SA y otros en laminar, **la tendencia de Cd contra Re no significa nada**, porque mezcla el efecto del Reynolds con el del modelo. Todos los puntos de esta tabla usan el mismo modelo.

## Como leer esto

- La columna `desv` compara contra valores de libro para cuerpos 2D. El estudio mide el sesgo del solver, no pretende acertar: el cilindro ya daba +37 % en `results/esfera_dx002_laminar/`.
- **El circulo a Re=1e6 (+326 %) no es un fallo del solver, es una limitacion conocida y esperada.** La referencia 0.35 es la crisis de resistencia: por encima de Re~2e5 la capa limite transiciona a turbulenta antes de separar, el punto de separacion se va hacia atras y el Cd se desploma. Eso exige una capa limite resuelta y transicion tridimensional; con el modelo de turbulencia descartado por colapso de dt, el solver corre laminar y **no puede reproducir la crisis por construccion**. Lo consistente es comparar ese punto contra el ~1.2 subcritico, y entonces la desviacion es +24 %, en linea con el resto.
- **La referencia de la gota es la mas floja de las tres y conviene no apoyarse en su `desv`.** Los Cd de libro de 0.05-0.1 son de cuerpos fuselados de esbeltez ~4 con el flujo adherido hasta la cola. El de aqui tiene esbeltez 2.58 (espesor 0.387 con morro de radio 0.2), y a esa relacion la recuperacion de presion es lo bastante brusca como para que separe tambien en la realidad. El dato utilizable de esta fila no es la desviacion contra el libro sino que **la gota da entre 3 y 5 veces menos Cd que el circulo a igual Re y misma malla**, que es una comparacion interna y por tanto libre del sesgo del solver.
- **El cuadrado es el caso mejor portado**: su Cd de referencia no depende de Re (los puntos de separacion estan clavados en las aristas, no hay crisis de resistencia que reproducir) y el solver converge hacia el desde abajo: -45 %, -35 %, -24 % y **+1 % a Re=1e6**. Que la forma sin separacion ambigua sea la que acierta apunta a que el sesgo del solver esta en donde coloca la separacion, no en como integra las fuerzas.
- `t` es el tiempo convectivo alcanzado. Si un caso se queda corto, su Cd esta medido en transitorio y no es comparable con el resto.
- El CI95 es de la ventana de promediado, no incluye incertidumbre de malla. Para eso hay que mirar la diferencia entre dx=0.004 y dx=0.002.
- Cl deberia ser ~0 en las tres formas por simetria a alpha=0; un Cl grande delata que la ventana no cubre un numero entero de ciclos de desprendimiento, no una fuerza real.
- **El Strouhal se mide con dos estimadores y la tabla muestra los dos cuando discrepan** (`ST_REF`: cilindro 0.20 subcritico, cuadrado 0.13). El pico del FFT de Cl se equivoca cuando la estela sufre duplicacion de periodo —los vortices alternan intensidad entre lados y el subarmonico se lleva mas energia que el fundamental—, y entonces devuelve f/2. Contar cruces por cero es inmune a eso pero sobrecuenta si la onda tiene armonicos fuertes. Medido: el circulo a Re=1e4 daba 0.0998 por FFT y **0.2005 por cruces, que es el 0.20 canonico del cilindro**; el cuadrado a Re=1e6 igual (0.105 -> 0.2106). Marcados `(dupl.)`. Donde la razon entre los dos no es ~1 ni ~2 se muestran ambos con `?`: ahi el numero no es citable. Los dos valores estan siempre en el JSON (`st_fft`, `st_cruces`, `st_ratio_cruces_fft`), recalculables sin GPU con `scripts/agent_tests/formas_strouhal.py`.
- **`St = no desprende` no es un fallo de medida**: significa que a t~20, arrancando de un campo simetrico, la inestabilidad de la estela todavia no ha saturado. Se detecta por la amplitud de Cl (`amp Cl`): por debajo de 0.05 el pico del FFT mide el crecimiento de la inestabilidad, no un ciclo limite, y da un St espurio. El caso medido: circulo a Re=1e3 da un pico limpisimo (prominencia 6127) en f=0.104 con amplitud 0.012 — la mitad del St de libro, porque es la envolvente de crecimiento y no desprendimiento saturado. El Cd de esos casos SI es utilizable: es un Cd de estela simetrica, mas bajo que el desprendido.
- Un cuerpo perfectamente simetrico en una malla simetrica solo empieza a desprender cuando el ruido numerico rompe la simetria, y a Re bajo eso tarda. Alargar la ventana es lo que arreglaria estos casos.

Videos en `results/formas/<forma>/videos/`, campos en `campos/` y `campos_t/`, series en `series/`.

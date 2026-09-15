# Malla dx=0.001 — descartada, NO usar en la memoria

Estos puntos existen pero no forman parte del estudio. Se descartaron porque el
paso de tiempo colapsa a esa resolucion:

| dx    | t_final alcanzado | dt medio |
|-------|-------------------|----------|
| 0.004 | 15.9 - 19.6       | 1.4e-3   |
| 0.002 | 15.7 - 21.7       | 8.0e-4   |
| 0.001 | **5.2 - 6.9**     | 1.2e-4   |

De 0.004 a 0.002 el dt baja x0.57, lineal con la malla. De 0.002 a 0.001 baja
x0.15: sobra un factor 3.4. Las 52000 iteraciones se quedaban en t~6 en vez de
t~20, o sea que estos puntos se midieron en pleno transitorio de arranque y no
son comparables con los de las otras mallas.

Ademas el dt no era solo pequeno, seguia cayendo durante todo el run (x6.8 sin
estabilizar) mientras que a dx=0.002 es plano. Misma firma que precedia al
blowup: warm_start_filtered evito el NaN pero no curo la inestabilidad.

Se conservan como evidencia de ese diagnostico. La terna del estudio es
0.008/0.004/0.002.

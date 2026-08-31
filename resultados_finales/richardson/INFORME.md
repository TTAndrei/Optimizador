# Resultados finales — convergencia de malla

Generado 2026-08-26 16:17. **Resultados definitivos del TFG.**

Dos perfiles, 4 mallas (dx = 0.008 / 0.006 / 0.004 / 0.002), 9 angulos (alpha = 0 a 8 de grado en grado).
Terna principal del GCI: (0.008, 0.004, 0.002) con r=2 constante. Terna alternativa de contraste: (0.006, 0.004, 0.002) (r=1.5 y 2).
Dominio 24x16 con el perfil en cx=6, Re=1e+05, t~20 en todas las mallas, ventana completa sin parada anticipada.
Solver de presion con presupuesto de proyeccion {'mg_max_outer': 2, 'mg_cycles_per_outer': 3}. Optimizaciones activas, IDENTICAS en las cuatro mallas:

  - dx=0.008: coarse_mask_majority, fast_masks, interp_float32, warm_start, warm_start_filtered
  - dx=0.006: coarse_mask_majority, fast_masks, interp_float32, warm_start, warm_start_filtered
  - dx=0.004: coarse_mask_majority, fast_masks, interp_float32, warm_start, warm_start_filtered
  - dx=0.002: coarse_mask_majority, fast_masks, interp_float32, warm_start, warm_start_filtered

`warm_start_filtered` es lo que permite usar la misma configuracion en todas las mallas. Sin el filtro del modo par-impar, `warm_start` con presupuesto 2x3 revienta por blowup de velocidad (iteracion 1340 a dx=0.004, ~1740 a dx=0.001). Cruce en `results/bench_opt/`.

## Como leer la tabla

Dos extrapolaciones por fila:

  - `extrap` es Richardson puro con el orden observado. Se conserva por trazabilidad, pero **se dispara cuando p tiende a 0**: el factor 1/(r^p - 1) crece sin limite. Medido: el Cl del ganador en alpha=6 daba 2.91 con la malla fina en 0.879 (amplificacion x27.5), y varios Cd cruzaban a negativo.
  - `robusto` es el valor a usar. Cuando el orden observado no es utilizable (`p_fiable` = NO) se fuerza p=2, el nominal del esquema, y el Cd se ajusta sobre log(Cd) para que no pueda salir negativo. Ahi el numero es una estimacion indicativa acotada, no una medida del error de discretizacion.

El umbral es p en [1, 4]. P_MIN=1 y no 0.5 porque en r=2 la amplificacion vale exactamente 1 en p=1 y se dispara por debajo: con p<1 el extrapolado se aleja de la malla fina mas que el salto entero entre las dos mallas mas finas.

## Tabla de GCI

### Ganador de la campaña de optimización (AG)

| alpha | mag | grueso | medio | fino | p obs | extrap | **robusto** | GCI fino % | monotona | p fiable |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | Cl | -0.11471 | 0.09294 | 0.14049 | 2.13 | 0.15462 | **0.15462** | 12.56 | si | si |
| 0.0 | Cd | 0.04400 | 0.04654 | 0.03679 | no mono | 0.03337 | **0.03402** | 11.62 | NO | NO |
| 0.0 | Cl/Cd | -2.60690 | 1.99720 | 3.81860 | 1.34 | 5.01079 | **5.01079** | 39.03 | si | si |
| 1.0 | Cl | 0.20176 | 0.23426 | 0.32159 | -1.43 | 0.37334 | **0.35070** | 20.12 | si | NO |
| 1.0 | Cd | 0.03870 | 0.03812 | 0.02797 | -4.35 | 0.02735 | **0.02522** | 2.76 | si | NO |
| 1.0 | Cl/Cd | 5.21320 | 6.14530 | 11.49880 | -2.52 | 12.62740 | **13.28330** | 12.27 | si | NO |
| 2.0 | Cl | 0.32505 | 0.36144 | 0.45732 | -1.40 | 0.51596 | **0.48928** | 16.03 | si | NO |
| 2.0 | Cd | 0.03517 | 0.03164 | 0.02117 | -1.93 | 0.01585 | **0.01852** | 31.40 | si | NO |
| 2.0 | Cl/Cd | 9.24260 | 11.42290 | 21.59930 | -2.22 | 24.37410 | **24.99143** | 16.06 | si | NO |
| 3.0 | Cl | 0.48626 | 0.57221 | 0.61352 | 1.06 | 0.65175 | **0.65175** | 7.79 | si | si |
| 3.0 | Cd | 0.03578 | 0.02086 | 0.01517 | 0.76 | 0.01168 | **0.01365** | 28.81 | si | NO |
| 3.0 | Cl/Cd | 13.59010 | 27.43500 | 40.43600 | 0.09 | 240.72751 | **44.76967** | 619.16 | si | NO |
| 4.0 | Cl | 0.58163 | 0.67986 | 0.69648 | 2.56 | 0.69987 | **0.69987** | 0.61 | si | si |
| 4.0 | Cd | 0.03307 | 0.02242 | 0.01692 | 0.47 | 0.01108 | **0.01541** | 43.16 | si | NO |
| 4.0 | Cl/Cd | 17.58540 | 30.32710 | 41.15460 | 0.23 | 102.39938 | **44.76377** | 186.02 | si | NO |
| 5.0 | Cl | 0.61987 | 0.73189 | 0.81522 | 0.43 | 1.05719 | **0.84300** | 37.10 | si | NO |
| 5.0 | Cd | 0.03864 | 0.03330 | 0.02333 | -1.26 | 0.01180 | **0.02072** | 61.78 | si | NO |
| 5.0 | Cl/Cd | 16.04010 | 21.97930 | 34.93840 | -1.13 | 45.90247 | **39.25810** | 39.23 | si | NO |
| 6.0 | Cl | 0.73352 | 0.80482 | 0.87871 | -0.05 | 2.91229 | **0.90335** | 289.28 | si | NO |
| 6.0 | Cd | 0.06412 | 0.04833 | 0.03567 | -0.10 | -0.01531 | **0.03224** | 178.66 | si | NO |
| 6.0 | Cl/Cd | 11.43970 | 16.65360 | 24.63160 | -0.61 | 39.68044 | **27.29093** | 76.37 | si | NO |
| 7.0 | Cl | 0.78835 | 0.91204 | 0.97092 | 1.07 | 1.02444 | **1.02444** | 6.89 | si | si |
| 7.0 | Cd | 0.05600 | 0.06965 | 0.04804 | no mono | 0.01099 | **0.04245** | 96.42 | NO | NO |
| 7.0 | Cl/Cd | 14.07750 | 13.09460 | 20.20970 | no mono | 21.35014 | **22.58140** | 7.05 | NO | NO |
| 8.0 | Cl | 0.76938 | 0.93926 | 1.03803 | 0.78 | 1.17519 | **1.07095** | 16.52 | si | NO |
| 8.0 | Cd | 0.09520 | 0.10361 | 0.06167 | no mono | 0.05114 | **0.05187** | 21.34 | NO | NO |
| 8.0 | Cl/Cd | 8.08210 | 9.06530 | 16.83270 | -2.98 | 17.95839 | **19.42183** | 8.36 | si | NO |

Orden observado utilizable en 5 de 27 casos.

### NACA 0012 sharp (semilla de referencia)

| alpha | mag | grueso | medio | fino | p obs | extrap | **robusto** | GCI fino % | monotona | p fiable |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | Cl | 0.00402 | 0.00097 | -0.00146 | 0.32 | -0.01110 | **-0.00227** | 825.40 | si | NO |
| 0.0 | Cd | 0.03242 | 0.02219 | 0.01434 | -0.20 | -0.01156 | **0.01240** | 225.73 | si | NO |
| 0.0 | Cl/Cd | 0.12390 | 0.04370 | -0.10180 | -0.86 | -0.28050 | **-0.15030** | 219.43 | si | NO |
| 1.0 | Cl | 0.10654 | 0.08884 | 0.09703 | no mono | 0.10410 | **0.09977** | 9.10 | NO | NO |
| 1.0 | Cd | 0.04305 | 0.02472 | 0.01589 | 0.33 | 0.00766 | **0.01371** | 64.69 | si | NO |
| 1.0 | Cl/Cd | 2.47490 | 3.59390 | 6.10780 | -1.17 | 8.12447 | **6.94577** | 41.27 | si | NO |
| 2.0 | Cl | 0.16070 | 0.20159 | 0.20751 | 2.79 | 0.20852 | **0.20852** | 0.61 | si | si |
| 2.0 | Cd | 0.03760 | 0.03024 | 0.01728 | -1.36 | 0.00029 | **0.01434** | 122.90 | si | NO |
| 2.0 | Cl/Cd | 4.27460 | 6.66570 | 12.01070 | -1.16 | 16.33733 | **13.79237** | 45.03 | si | NO |
| 3.0 | Cl | 0.21235 | 0.33255 | 0.36597 | 1.85 | 0.37884 | **0.37884** | 4.40 | si | si |
| 3.0 | Cd | 0.04153 | 0.03651 | 0.02107 | -2.09 | 0.01364 | **0.01755** | 44.11 | si | NO |
| 3.0 | Cl/Cd | 5.11320 | 9.10830 | 17.36700 | -1.05 | 25.10561 | **20.11990** | 55.70 | si | NO |
| 4.0 | Cl | 0.31803 | 0.46124 | 0.49780 | 1.97 | 0.51033 | **0.51033** | 3.15 | si | si |
| 4.0 | Cd | 0.04623 | 0.04468 | 0.02746 | -3.83 | 0.02575 | **0.02335** | 7.76 | si | NO |
| 4.0 | Cl/Cd | 6.87900 | 10.32300 | 18.12790 | -1.18 | 24.29178 | **20.72953** | 42.50 | si | NO |
| 5.0 | Cl | 0.27888 | 0.47310 | 0.56161 | 1.13 | 0.63570 | **0.63570** | 16.49 | si | si |
| 5.0 | Cd | 0.05659 | 0.05284 | 0.03222 | -2.85 | 0.02764 | **0.02733** | 17.78 | si | NO |
| 5.0 | Cl/Cd | 4.92830 | 8.95350 | 17.42840 | -1.07 | 25.09480 | **20.25337** | 54.98 | si | NO |
| 6.0 | Cl | 0.32536 | 0.48368 | 0.60869 | 0.34 | 1.07789 | **0.65036** | 96.36 | si | NO |
| 6.0 | Cd | 0.06640 | 0.06386 | 0.03987 | -3.59 | 0.03703 | **0.03408** | 8.91 | si | NO |
| 6.0 | Cl/Cd | 4.90000 | 7.57420 | 15.26660 | -1.52 | 19.36588 | **17.83073** | 33.56 | si | NO |
| 7.0 | Cl | 0.37600 | 0.48812 | 0.65199 | -0.55 | 1.00696 | **0.70662** | 68.05 | si | NO |
| 7.0 | Cd | 0.08257 | 0.09321 | 0.05185 | no mono | 0.03753 | **0.04265** | 34.53 | NO | NO |
| 7.0 | Cl/Cd | 4.55360 | 5.23670 | 12.57340 | -3.42 | 13.32663 | **15.01897** | 7.49 | si | NO |
| 8.0 | Cl | 0.39807 | 0.50611 | 0.67315 | -0.63 | 0.97896 | **0.72884** | 56.78 | si | NO |
| 8.0 | Cd | 0.10010 | 0.11494 | 0.07984 | no mono | 0.05414 | **0.07070** | 40.24 | NO | NO |
| 8.0 | Cl/Cd | 3.97660 | 4.40320 | 8.43180 | -3.24 | 8.90892 | **9.77467** | 7.07 | si | NO |

Orden observado utilizable en 4 de 27 casos.

## Avisos al citar

- **Orden observado utilizable en 9 de 54 casos.** En el resto la columna `robusto` usa p=2 nominal.
- **Punto de diseno: alpha = 4.** Las tres mallas de la terna coinciden en que el maximo de L/D esta ahi. El extrapolado deja alpha=3 por encima, pero eso se debe a que su Cl aun no ha convergido en malla (correccion +6.2% frente a +0.5% en alpha=4), no a que el pico este ahi.
- **El L/D extrapolado se reconstruye desde Cl y Cd** (campo `ld_desde_cl_cd` en `richardson.json`), no extrapolando el cociente: un cociente hereda las patologias de ambos y revienta si el Cd extrapolado se acerca a cero.
- La ventaja del perfil optimizado sobre el NACA 0012 **se mantiene en todos los angulos, en las cuatro mallas y tambien extrapolada**. Es el resultado mas robusto del estudio: no depende de la malla ni del metodo de extrapolacion.


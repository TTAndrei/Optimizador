# Independencia de dominio — ganador islands_tfg2

alpha=4.0deg, Re=1e+05, dx=0.002, 13000 iters. Solo cambia el dominio.

Las fronteras top/bottom son `slip` (v=0): paredes de tunel cerrado, no
campo lejano. El error lateral decae como (c/h)^2 y el de entrada como c/d.

## Casos

| | Lx x Ly | cx | h | sigma | d_up | Cl | CI95 | Cd | CI95 | L/D | conv | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 8x5 | 2 | 2.5 | 0.0329 | 2 | 0.6680 | 0.0077 | 0.02411 | 0.00093 | 27.70 | False | 1788 |
| B | 16x10 | 4 | 5 | 0.0082 | 4 | 0.6739 | 0.0080 | 0.02118 | 0.00071 | 31.81 | False | 2537 |
| C | 24x16 | 6 | 8 | 0.0032 | 6 | 0.6806 | 0.0085 | 0.02147 | 0.00099 | 31.70 | False | 3030 |
| D | 40x24 | 10 | 12 | 0.0014 | 10 | 0.6707 | 0.0093 | 0.02201 | 0.00108 | 30.47 | False | 3316 |
| E | 8x16 | 2 | 8 | 0.0032 | 2 | 0.6299 | 0.0093 | 0.02699 | 0.00120 | 23.34 | False | 1913 |
| F | 24x5 | 6 | 2.5 | 0.0329 | 6 | 0.6942 | 0.0040 | 0.01928 | 0.00046 | 36.01 | False | 3096 |
| G | 12x24 | 3 | 12 | 0.0014 | 3 | 0.6327 | 0.0117 | 0.02553 | 0.00124 | 24.78 | False | 1999 |

## Diagnostico de fronteras (campo final)

`eps_top/bot` = aceleracion media en la pared slip (bloqueo medido).
`v_in_max` = |v| maxima justo dentro del inflow, que el BC fuerza a 0.
`u_out_min` = velocidad minima en el plano de salida (estela viva si << 1).
`cp_wall_range` = excursion de Cp en la pared: si no es ~0, aun ve el perfil.

| | eps_top | eps_bot | v_in_max | u_out_min | cp_wall_range | discrep. Cl |
|---|---|---|---|---|---|---|
| A | 0.00362 | 0.00067 | 0.00153 | 0.9440 | 0.06846 | 0.05434 |
| B | 0.00194 | -0.00043 | 0.00048 | 0.9825 | 0.03411 | 0.08568 |
| C | 0.00021 | -0.00021 | 0.00024 | 0.9967 | 0.02178 | 0.01583 |
| D | -0.00023 | 0.00042 | 0.00008 | 0.9998 | 0.01424 | 0.07993 |
| E | -0.00147 | 0.00157 | 0.00123 | 0.9434 | 0.02606 | 0.03061 |
| F | 0.02288 | -0.01718 | 0.00011 | 0.9997 | 0.06831 | 0.05166 |
| G | -0.00078 | 0.00089 | 0.00061 | 0.9598 | 0.01576 | 0.00762 |

## Extrapolacion a dominio infinito

Modelo `y = y_inf + k1*sigma + k2*(c/d_up)`, un termino por frontera.

- **cl_inf = 0.684137**  (k_lateral=+1.1113, k_longitud=-0.1148, residuo max=0.014786)
- **cd_inf = 0.019824**  (k_lateral=-0.1000, k_longitud=+0.0149, residuo max=0.001554)

| | Cl | err vs inf | CI95 | convergido en dominio |
|---|---|---|---|---|
| A | 0.6680 | 0.0161 (2.36%) | 0.0077 | no |
| B | 0.6739 | 0.0103 (1.50%) | 0.0080 | no |
| C | 0.6806 | 0.0036 (0.52%) | 0.0085 | SI |
| D | 0.6707 | 0.0134 (1.96%) | 0.0093 | no |
| E | 0.6299 | 0.0542 (7.93%) | 0.0093 | no |
| F | 0.6942 | 0.0100 (1.46%) | 0.0040 | no |
| G | 0.6327 | 0.0515 (7.52%) | 0.0117 | no |

## Factorial 2x2 (A=8x5, E=8x16, F=24x5, C=24x16)

Si `interaccion` no es ~0, los dos errores no son aditivos y no vale
corregirlos por separado.

| | lateral@Lx8 | lateral@Lx24 | longitud@Ly5 | longitud@Ly16 | interaccion | total A->C |
|---|---|---|---|---|---|---|
| cl | -0.03810 | -0.01357 | +0.02614 | +0.05068 | +0.02453 | +0.01258 |
| cd | +0.00287 | +0.00219 | -0.00484 | -0.00552 | -0.00068 | -0.00265 |
| ld | -4.36130 | -4.30180 | +8.30460 | +8.36410 | +0.05950 | +4.00280 |

# Verificación numérica

## Convergencia de malla (GCI) — Re=1e+05, α=4.0°

Mallas: dx = [0.001, 0.002, 0.004, 0.008]  |  factores de refinamiento r = [2.0, 2.0]

| magnitud | p observado | valor malla fina | Richardson h→0 | GCI fina | monótona |
|---|---|---|---|---|---|
| Cl | 2.18 | 0.7402 | 0.74865 | ±1.43% | sí |
| Cd | 3.18 | 0.029272 | 0.029334 | ±0.27% | **no** |
| L/D | 3.64 | 25.287 | 25.339 | ±0.26% | sí |

El GCI acota la incertidumbre de discretización de la malla más fina.

## Ganadores re-evaluados (estadística corregida)

| caso | Re | L/D | CI95 | σCl | σCd | n | early-stop Cl/Cd | iters |
|---|---|---|---|---|---|---|---|---|
| re1e3 | 1e+03 | 4.184 | ±0.0433 | 0.046442 | 0.007539 | 104 | False | 26000 |
| re1e5 | 1e+05 | 23.218 | ±0.8962 | 0.019758 | 0.003049 | 32 | True | 8050 |
| re1e6 | 1e+06 | 13.209 | ±0.0187 | 0.002808 | 0.000224 | 32 | True | 8150 |
| re1e7 | 1e+07 | 13.192 | ±0.3098 | 0.053936 | 0.003518 | 33 | False | 8250 |

## Polar

| α | L/D ganador_re1e5 | L/D NACA_0012 |
|---|---|---|
| 0.0 | 13.05 | 0.01 |
| 2.0 | 21.92 | 7.55 |
| 4.0 | 23.22 | 13.84 |
| 6.0 | 19.22 | 12.38 |
| 8.0 | 16.30 | 8.14 |
| 10.0 | 8.07 | 6.66 |

## Validación externa — NACA0012 vs XFOIL (airfoiltools, Ncrit=9)

MAE(Cl) = 0.0912  |  MAE(Cd) = 0.03598  |  error medio de Cd = +123.8%

Pendiente dCl/dα (α≤6°): solver 0.10211 · XFOIL 0.11107 · teoría 2π 0.10966 por grado

| α | Cl sim | Cl XFOIL | Cd sim | Cd XFOIL | L/D sim | L/D XFOIL | ΔCd |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.0002 | 0.0000 | 0.01653 | 0.01693 | 0.01 | 0.00 | -2.4% |
| 2.0 | 0.1659 | 0.3737 | 0.02199 | 0.01444 | 7.55 | 25.88 | +52.3% |
| 4.0 | 0.4754 | 0.5365 | 0.03436 | 0.01520 | 13.84 | 35.30 | +126.1% |
| 6.0 | 0.5777 | 0.6862 | 0.04667 | 0.01958 | 12.38 | 35.05 | +138.4% |
| 8.0 | 0.7065 | 0.8471 | 0.08684 | 0.02869 | 8.14 | 29.53 | +202.7% |
| 10.0 | 0.9954 | 0.9661 | 0.14937 | 0.04582 | 6.66 | 21.09 | +226.0% |

## Figuras

- `calibracion_earlystop.png`
- `gci.png`
- `polar.png`
- `reeval_ganadores.png`
- `validacion_xfoil.png`

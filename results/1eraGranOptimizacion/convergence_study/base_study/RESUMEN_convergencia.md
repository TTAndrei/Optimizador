# Estudio de convergencia de perfiles
Condiciones: α=4.0°, Re=1e5, CFL=0.5. Config CFD de referencia (consistent + SA + MacCormack).

## Veredicto
- Arm A: L/D NO converge (CV=0.172, spread=3.23) -> las semillas siguen en óptimos locales distintos.
- Distancia de forma media entre ganadores = 0.0375 (0 = misma forma).
- Arm B L/D=9.03 < mejor Arm A 9.95 (quizá pocas generaciones dentro del deadline).

## Arm A — GA independiente por semilla (diagnóstico)
| semilla | L/D | Cl | Cd |
|---|---|---|---|
| AG24 | 9.9469 | 0.662624 | 0.066616 |
| GM15 | 9.3051 | 0.657492 | 0.070659 |
| NACA_0012_sharp | 6.9534 | 0.414452 | 0.059604 |
| s1014 | 6.7127 | 0.50156 | 0.074718 |

Dispersión L/D: spread=3.234, CV=0.172 (mean=8.230).

## Arm B — población mixta + cruce (arreglo)
Ganador mixto: L/D=9.0306, Cl=0.533284, Cd=0.059053.
Población final: std L/D=0.006, spread=0.015 (n=3).

Distancia de forma media entre ganadores: 0.0375 (máx 0.0743).

## Figuras
- formas_ganadoras.png
- ld_por_semilla.png
- camber_espesor.png
- armB_convergencia.png
- distancia_forma.png

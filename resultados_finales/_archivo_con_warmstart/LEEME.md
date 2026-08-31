# Puntos de dx=0.002 calculados CON warm_start

Validos, pero de otra configuracion del solver: llevan `warm_start` activo, que
se retiro del estudio el 2026-08-25 al descubrir que provoca blowup de velocidad
con presupuesto 2x3 (los 10 puntos de dx=0.001 abortaron en la iteracion ~1740).

No mezclar con los de `resultados_finales/*/metricas/`: la terna del Richardson
exige la misma configuracion del solver en las tres mallas.

Los de `ag24` venian importados de `results/polar_optimizada/`, que sigue intacto.

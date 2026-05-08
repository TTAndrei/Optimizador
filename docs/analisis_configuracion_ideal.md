# Analisis de configuracion ideal de mutacion

## Configuracion recomendada
- modo_mutacion: parametrica
- camber_mut_std: 0.003500
- espesor_mut_std: 0.004500
- te_camber_shift_std: 0.001500
- te_espesor_shift_std: 0.001200
- prob_mutacion: 0.850000

## Criterio de seleccion
- Score = 0.72*validez_norm + 0.20*delta_y_abs_max_p50_norm + 0.08*abs(te_camber_p50)_norm
- Prioriza robustez geométrica y conserva exploración moderada.

## Seleccion por parametro
- camber_mut_std: 0.003500 (base | baseline | validez=0.741, delta_y_abs_max_p50=0.007544, |te_camber_p50|=0.000294, score=0.767)
- espesor_mut_std: 0.004500 (base | baseline | validez=0.741, delta_y_abs_max_p50=0.007544, |te_camber_p50|=0.000294, score=0.758)
- te_camber_shift_std: 0.001500 (base | baseline | validez=0.741, delta_y_abs_max_p50=0.007544, |te_camber_p50|=0.000294, score=0.723)
- te_espesor_shift_std: 0.001200 (base | baseline | validez=0.741, delta_y_abs_max_p50=0.007544, |te_camber_p50|=0.000294, score=0.853)
- prob_mutacion: 0.850000 (base | baseline | validez=0.741, delta_y_abs_max_p50=0.007544, |te_camber_p50|=0.000294, score=0.857)

## Ranking de escenarios por validez
1. baseline | validez=0.741 | delta_y_abs_max_p50=0.007544 | te_camber_p50=-0.000294 | te_gap_p50=0.001951
2. camber_bajo | validez=0.698 | delta_y_abs_max_p50=0.006629 | te_camber_p50=-0.000394 | te_gap_p50=0.001948
3. camber_alto | validez=0.698 | delta_y_abs_max_p50=0.010546 | te_camber_p50=0.000913 | te_gap_p50=0.001930
4. espesor_alto | validez=0.698 | delta_y_abs_max_p50=0.009591 | te_camber_p50=-0.000445 | te_gap_p50=0.002670
5. espesor_bajo | validez=0.686 | delta_y_abs_max_p50=0.007070 | te_camber_p50=-0.000960 | te_gap_p50=0.001884
6. te_espesor_alto | validez=0.682 | delta_y_abs_max_p50=0.007458 | te_camber_p50=-0.001489 | te_gap_p50=0.001887
7. te_camber_alto | validez=0.656 | delta_y_abs_max_p50=0.008178 | te_camber_p50=0.000682 | te_gap_p50=0.002622
8. te_camber_bajo | validez=0.649 | delta_y_abs_max_p50=0.007532 | te_camber_p50=-0.002031 | te_gap_p50=0.001547
9. te_espesor_bajo | validez=0.645 | delta_y_abs_max_p50=0.007603 | te_camber_p50=0.000026 | te_gap_p50=0.002155
10. prob_mut_alta | validez=0.645 | delta_y_abs_max_p50=0.008160 | te_camber_p50=-0.000459 | te_gap_p50=0.001443
11. prob_mut_baja | validez=0.609 | delta_y_abs_max_p50=0.006218 | te_camber_p50=0.001588 | te_gap_p50=0.002374

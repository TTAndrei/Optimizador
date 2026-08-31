# Puntos calculados SIN warm_start (2026-08-24/25)

Configuracion: coarse_mask_majority + fast_masks + interp_float32, presupuesto
2x3, sin warm_start. Se llego a ella para evitar el blowup, antes de descubrir
que la causa real era el modo de tablero en la semilla y que el filtro lo
arregla sin renunciar a warm_start.

Validos como datos, pero NO comparables con el estudio definitivo: sin
warm_start la proyeccion queda mucho menos convergida (el L/D del punto de
diseno sale 19.8 frente a 41.2).

Contiene las mallas dx=0.004 y dx=0.002 completas de los dos perfiles.
El directorio `ag24/` es el ganador del AG; se renombro a `ganador_ag/` en el
estudio definitivo porque AG24 es un perfil distinto (Drela) que fue otra
semilla del algoritmo.

# Organizacion de resultados

Esta carpeta separa salidas por tipo de estudio:

- `barridos/`: barridos parametricos de perfiles, angulo, resolucion y modos MG.
- `ga/`: corridas de optimizacion genetica y sus perfiles/records.
- `comparativas/`: comparaciones contra teoria, XFOIL o benchmarks externos.
- `diagnosticos/`: pruebas de depuracion numerica y fisica del solver.
- `calibracion/`: graficos o artefactos de ajuste/calibracion.
- `miscelanea/`: salidas puntuales que no pertenecen claramente a los grupos anteriores.

Para el diagnostico de deficit de lift en NACA0012, la salida esperada es:

```text
results/diagnosticos/diagnostico_lift_naca0012/
```

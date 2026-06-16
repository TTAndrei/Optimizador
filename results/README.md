# Organizacion de resultados

Esta carpeta separa salidas por tipo de estudio:

- `barridos/`: barridos parametricos de perfiles, angulo, resolucion y modos MG.
- `ga/`: corridas de optimizacion genetica y sus perfiles/records.
- `comparativas/`: comparaciones contra teoria, XFOIL o benchmarks externos.
- `diagnosticos/`: pruebas de depuracion numerica y fisica del solver.
- `calibracion/`: graficos o artefactos de ajuste/calibracion.
- `miscelanea/`: salidas puntuales que no pertenecen claramente a los grupos anteriores.

Para el diagnostico conservado de desprendimiento espurio en NACA0012, la
salida de referencia es:

```text
results/diagnosticos/naca0012_suite_desprendimiento/
```

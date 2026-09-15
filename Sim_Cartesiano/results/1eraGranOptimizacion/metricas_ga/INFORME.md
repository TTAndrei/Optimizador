# Métricas del optimizador genético

Evaluaciones CFD registradas: **3884**  |  estudios con historial: **6**
Coste acumulado en los estudios con historial: **98.6 GPU-horas**

## Estudios

| estudio | gens | L/D inicial | L/D final | mejora | GPU-h | evals | cribado surrogate | presión selectiva |
|---|---|---|---|---|---|---|---|---|
| exprimir_re1e5 | 28 | 17.007 | 18.478 | +8.6% | 19.0 | 666 | 0% | 0.39 |
| masgen | 24 | 13.974 | 18.177 | +30.1% | 11.2 | 381 | 0% | 0.46 |
| masperfiles | 12 | 13.245 | 16.128 | +21.8% | 7.9 | 281 | 0% | 0.42 |
| re1e3 | 19 | 3.715 | 5.013 | +34.9% | 35.7 | 297 | 0% | 0.33 |
| re1e6 | 16 | 8.628 | 11.310 | +31.1% | 7.9 | 249 | 0% | 0.36 |
| re1e7 | 30 | 8.541 | 11.714 | +37.2% | 16.9 | 473 | 0% | 0.42 |

## Óptimo geométrico por Reynolds

| Re | L/D | $t/c$ | $x_t/c$ | $f/c$ | $x_f/c$ |
|---|---|---|---|---|---|
| Re=1e3 | 5.01 | 0.07074 | 0.2125 | -0.07899 | 1.0 |
| Re=1e5 | 18.48 | 0.05823 | 0.3875 | 0.03259 | 0.4 |
| Re=1e6 | 11.92 | 0.05342 | 0.4125 | 0.01826 | 0.875 |
| Re=1e7 | 11.71 | 0.06848 | 0.35 | 0.0649 | 0.275 |

## Dimensionalidad del espacio de diseño

- 256 coordenadas por perfil, 3884 muestras.
- PC1+PC2 explican **75.5%** de la varianza geométrica.

## Estabilidad de las evaluaciones CFD

- σ = 0 exacto en **81.8%** de las evaluaciones históricas.
  Artefacto del relleno de la cola del vector al disparar el early-stop (`Simulador2D.py`); corregido: ahora se trunca a las muestras reales.
- Dispersión relativa mediana: $C_l$ 4.987%, $C_d$ 10.417%.

## Modelo sustituto (RandomForest)

- **`usar_ia=False` en los 6 estudios**: el surrogate nunca cribó ningún candidato.
- Reentrenado offline @ Re=1e5 con 2793 muestras: $R^2$ (5-fold CV) = **0.850**, MAE = 1.07.
- Punto de operación viable: evitar **69% del CFD** perdiendo solo 2.5% de la élite (umbral = percentil 70).

## Efecto del refinado de malla sobre el óptimo

| Re | L/D exploración | L/D refinado | Δ |
|---|---|---|---|
| Re=1e3 | 5.013 | 5.013 | +0.0% |
| Re=1e5 | 18.478 | 15.135 | -18.1% |
| Re=1e6 | 11.310 | 12.185 | +7.7% |
| Re=1e7 | 11.714 | 12.085 | +3.2% |

## Figuras

- `01_convergencia.png`
- `02_convergencia_normalizada.png`
- `03_presion_selectiva.png`
- `04_coste_computacional.png`
- `05_surrogate_y_operadores.png`
- `06_diversidad.png`
- `07_pca_espacio_diseno.png`
- `08_heatmaps_geometria.png`
- `09_correlaciones.png`
- `10_pareto_cl_cd.png`
- `11_historial_global.png`
- `12_ld_vs_reynolds.png`
- `13_perfiles_ganadores.png`
- `14_estabilidad_cfd.png`
- `15_surrogate_offline.png`

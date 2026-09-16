# Validación 3 — `y⁺ < 1` en TODA la pared: paso de pared 8.0e-5

Mismo caso que `../v1_naca0012_re1e5_a5/` (NACA 0012, α = 5°, Re = 1e5, SA,
t\* = 20, float32) cambiando **solo el paso de pared**, de 1.9e-4 a 8.0e-5.

```bash
PYTHONPATH=. .venv/bin/python validacion/v1_naca0012_re1e5_a5/correr.py validacion/v3_ymas1_dn8e5
PYTHONPATH=. .venv/bin/python validacion/v3_ymas1_dn8e5/analizar.py
```

## El problema

`y⁺` no es uniforme a lo largo del perfil: `u_tau` se dispara en el pico de
succión. Con 1.9e-4 la mediana era 0.49 — impecable — pero el **máximo 1.88**, y
`y⁺ > 1` en el 2 % del arco alrededor del morro (`|s − s_LE| < 0.016`,
x/c de 0 a 0.020). Mirar solo la mediana lo tapaba.

El máximo escala con `dn`, así que el paso que lo mete bajo 1 es 1.9e-4 / 1.88.
**8.0e-5** lo cumple con margen.

## Por qué en todo el contorno y no solo en el morro

En malla estructurada el número de capas lo fija la columna más fina, así que
refinar solo el borde de ataque **cuesta exactamente las mismas celdas** que
refinarlo todo: no hay ahorro que perseguir.

Pero el efecto **sí queda localizado**, y gratis, por `ASPECTO_MAX`: la regla es
`dn(ξ) = max(dn_pared, ds(ξ)/100)`, y a media cuerda `ds/100 ≈ 1.2e-4` ya es el
suelo. Esas columnas no se mueven. El refinado se concentra donde `ds` es
pequeño, es decir en el morro y el borde de salida, que es donde hacía falta.

| | v1 | v3 |
|---|---|---|
| paso de pared pedido | 1.9e-4 | **8.0e-5** |
| realizado sobre el perfil, mín → máx | 1.88e-4 → 1.92e-4 | 7.95e-5 → 1.34e-4 |
| celdas | 34 853 (92×384) | **37 534 (99×384)** — +7.7 % |
| veredicto de `calidad` | válida | **BUENA**, sin avisos |

## Resultado: `y⁺ < 1` en los 255 puntos de pared

| `y⁺` del primer centro | v1 | v3 |
|---|---|---|
| mediana | 0.491 | **0.329** |
| p95 | 1.621 | **0.689** |
| **máximo** | **1.880** | **0.798** |
| puntos con `y⁺ > 1` | 31 / 255 | **0 / 255** |

Predicho antes de correr, a partir del `cf` de v1: 0.797. Medido: 0.798.

Capa límite mejor resuelta de propina — celdas por debajo de `y⁺ = 15`: a media
cuerda de 16 a 23, en el morro de 4 a 10.

## Las fuerzas no se mueven

| | v1 | v3 | cambio |
|---|---|---|---|
| **Cl superficie** | 0.49840 | 0.49847 | **+0.01 %** |
| Cl por ∮ΔCp | 0.49527 | 0.49532 | +0.01 % |
| Cl por circulación | 0.48872 | 0.48758 | −0.23 % |
| **Cd** | 0.01931 | 0.01932 | **+0.02 %** |
| Cd presión | 0.00757 | 0.00757 | −0.05 % |
| Cd viscoso | 0.01174 | 0.01175 | +0.07 % |
| Cm (c/4) | −0.00794 | −0.00792 | −0.25 % |
| ΔCp_TE | −0.02775 | −0.02786 | +0.39 % |
| ν_t/ν máx | 37.32 | 38.04 | +1.92 % |

Contra XFOIL, sin cambio apreciable: Cl −17.28 % y Cd_p −1.19 % frente a Ncrit=5
(v1: −17.29 % y −1.14 %). **Que las fuerzas no se muevan es el resultado**: con
1.9e-4 ya estaban convergidas en `dn`, y el `y⁺ > 1` del morro no las
contaminaba. Lo que compra v3 es poder decir que la malla cumple el criterio en
toda la pared, no de media.

Aviso: el dominio no es idéntico (25.33 × 16.31 frente a 25.87 × 17.53), porque
el número de capas sale del plan de marcha y este arranca más fino. La
validación 2 midió que a estos tamaños las fuerzas no dependen del dominio
(0.01 % en Cl recortando un 27 % de área), así que la comparación se sostiene.

## Dónde se va el tiempo

`Solver(..., cronometro=True)`, 8 000 pasos, suma = 100 % del reloj de pared:

| etapa | v1 (34 853 celdas) | v3 (37 534 celdas) |
|---|---|---|
| momento (2 ecuaciones) | sin instrumentar | **49.9 % — 83.30 ms** |
| presión (Poisson + PCG) | sin instrumentar | **31.2 % — 52.16 ms** |
| turbulencia (SA) | sin instrumentar | **18.9 % — 31.58 ms** |
| total | 163.4 ms — 6.12 it/s | 167.0 ms — 5.98 it/s |

+7.7 % de celdas cuesta **+2.2 % de tiempo por paso**, no +7.7 %: a 37 k celdas
la GPU sigue sin saturarse y lo que se paga es latencia de lanzamiento, no
trabajo. El caso v2 (33 704 celdas) da el mismo reparto: 50.5 / 30.3 / 19.2 %.

El Poisson converge en **1 ciclo de PCG de mediana**, así que esos 52 ms no son
iteraciones: son las **3 resoluciones** de la corrección diferida más el montaje
de la jerarquía. Lo mismo en momento (6 resoluciones) y en SA (2). Son **11
resoluciones de multigrid por paso a ~15 ms**, y la palanca de optimización es el
número de correcciones externas, no ninguna etapa suelta.

## Figuras

`figuras/malla.png` la genera `correr.py` **en cada corrida**, con todos los
parámetros de malla y caso escritos al pie: perfil, α, Re, dt, t\*, `dn_pared`,
capas de pared, crecimiento, `dn_max`, `aspecto_max`, puntos de superficie,
`x_salida`, ortogonalidad, oblicuidad, aspecto real, J≤0 y dimensiones.

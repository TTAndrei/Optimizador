# Validación 2 — sensibilidad al dominio: 24 × 12

Mismo caso que `../v1_naca0012_re1e5_a5/` (NACA 0012, α = 5°, Re = 1e5, SA,
y⁺≈1, t\* = 20, float32) cambiando **solo el tamaño del dominio**.

```bash
.venv/bin/python validacion/v1_naca0012_re1e5_a5/correr.py \
    validacion/v2_dominio_24x12 distancia_lejos=6.0 x_salida=18.39
.venv/bin/python validacion/v2_dominio_24x12/analizar.py
```

| | v1 | v2 |
|---|---|---|
| dominio | 25.9 × 17.5 | **24.00 × 12.48** |
| celdas | 34 853 (92×384) | 33 704 (89×384) |
| borde lejano | 8.76 c | 6.24 c |

El 12.00 exacto no es alcanzable: el número de capas sale del crecimiento
geométrico y está cuantizado (`distancia_lejos` de 5.8 a 6.2 da la misma malla).
12.48 es el valor de la malla inmediatamente superior a 11.14.

## Resultado

| | v1 (grande) | v2 (24×12) | cambio |
|---|---|---|---|
| **Cl superficie** | 0.49840 | 0.49844 | **+0.01 %** |
| Cl por ∮ΔCp | 0.49527 | 0.49533 | +0.01 % |
| Cl por circulación | 0.48872 | 0.47871 | −2.05 % |
| **Cd** | 0.01931 | 0.01946 | **+0.77 %** |
| Cd presión | 0.00757 | 0.00770 | +1.71 % |
| Cd viscoso | 0.01174 | 0.01176 | +0.16 % |
| Cm (c/4) | −0.00794 | −0.00783 | −1.33 % |
| ΔCp_TE | −0.02775 | −0.02778 | +0.11 % |
| ν_t/ν máx | 37.32 | 37.33 | +0.01 % |
| dispersión de Γ | 6.34 % | **11.93 %** | +88 % |

**Las fuerzas no dependen del dominio.** El Cl se mueve **0.01 %** recortando el
dominio un 27 % en área; el Cd, 0.77 %, y todo él por la parte de presión. El
campo lejano del arco C está bien puesto: no hay bloqueo apreciable ni a 6.24
cuerdas. Comparado con el cartesiano, donde pasar de 8×5 a 24×16 movía el L/D de
27.70 a 31.70 (**14 %**), esto es otra liga.

Contra XFOIL el Cd de presión sale incluso mejor: **+0.56 %** frente a Ncrit=5
(v1 daba −1.14 %).

## Lo que sí se mueve: la circulación

| lazo | v1 | v2 | cambio |
|---|---|---|---|
| r = 0.15 c | 0.24986 | 0.24928 | **−0.23 %** |
| r = 0.30 c | 0.24887 | 0.24770 | −0.47 % |
| r = 0.60 c | 0.24629 | 0.24316 | −1.27 % |
| r = 1.00 c | 0.24240 | 0.23589 | −2.69 % |
| r = 1.50 c | 0.23437 | 0.22073 | **−5.82 %** |

El cambio **escala monótonamente con el radio del lazo**: los lazos pequeños dan
lo mismo en los dos dominios al 0.2 %, los grandes no. O sea, la dispersión de Γ
entre lazos es **artefacto del dominio**, no defecto de la discretización: cuanto
más cerca queda el borde lejano, más recorta el lazo grande.

**Consecuencia para el plan**: el criterio de F5 "Γ estable con el tamaño del lazo
dentro del 5 %" mide el dominio, no el esquema, igual que el de "Cl por
circulación al 1 %" mide Kutta-Joukowski y no la exactitud. Los dos hay que
reescribirlos. Lo que sí es criterio válido es que **superficie y ∮ΔCp coincidan**
(0.63 %) y que **las fuerzas no dependan del dominio** (0.01 % en Cl).

## Dónde se va el tiempo

`Solver(..., cronometro=True)`, suma = 100 % del reloj de pared:

| etapa | v1 | v2 |
|---|---|---|
| momento (2 ecuaciones) | 52.4 % — 82.6 ms | 50.5 % — 81.6 ms |
| presión (Poisson + PCG) | 27.7 % — 43.7 ms | 30.3 % — 49.0 ms |
| turbulencia (SA) | 19.9 % — 31.4 ms | 19.2 % — 31.0 ms |
| ritmo | 6.12 it/s | 6.18 it/s |

El Poisson converge en **1 ciclo de PCG de mediana**, así que sus 49 ms no son
iteraciones: son las **3 resoluciones** de la corrección diferida más el montaje
de la jerarquía. Lo mismo en el momento (6 resoluciones) y en SA (2). Son **11
resoluciones de multigrid por paso a ~14 ms**, y la palanca de optimización es el
número de correcciones externas, no ninguna etapa suelta.

## Figuras

`velocidad_dominio_final.png` mostraba una ventana fija (−2.5 ≤ x ≤ 5,
|y| ≤ 2.5) idéntica en v1 y v2, así que **las dos versiones salían iguales
aunque la malla sí cambia**. Corregido: ahora esa figura cubre el dominio
entero y la ventana de estela se guarda aparte.

| figura | qué es |
|---|---|
| `malla_dominio.png` | la malla completa, con Lx, Ly y los límites en el título |
| `velocidad_dominio_final.png` | \|u\| en **todo** el dominio |
| `velocidad_estela_final.png` | \|u\| en la ventana de estela (−2.5…5, ±2.5) |
| `velocidad_perfil_final.png` | \|u\| pegado al perfil |

La comprobación de que los dominios son distintos está en `malla.npz`:
v1 x∈[−7.87, 18.00], |y|≤8.76 (92×384); v2 x∈[−5.61, 18.39], |y|≤6.24 (89×384).

# Validación 1 — NACA 0012, α = 5°, Re = 1e5, Spalart-Allmaras

Primera validación del solver curvilíneo (`curvo/`) contra una referencia externa.
Todo lo de esta carpeta se regenera con:

```bash
.venv/bin/python validacion/v1_naca0012_re1e5_a5/correr.py     # ~22 min, GPU
.venv/bin/python validacion/v1_naca0012_re1e5_a5/analizar.py   # ~4 min, CPU
```

## Montaje

| | |
|---|---|
| perfil | `profiles/NACA_0012_sharp` (161 puntos) |
| α, Re, U∞ | 5°, 1e5, 1.0 (ν = 1e-5) |
| malla | C monobloque body-fitted, 92 × 384 = **34 853 celdas** |
| paso de pared | `dn = 1.9e-4` con **15 capas de espaciado constante** (`crecimiento_pared = 1.0`) |
| turbulencia | Spalart-Allmaras, `ν̃∞ = 3ν` |
| tiempo | `dt = 2.5e-3`, hasta `t* = 20` (8000 pasos) |
| precisión | float32 en GPU (RTX 3070 Ti) |

**El ángulo de ataque inclina la corriente, no la geometría**: la malla es la
simétrica y a α = 0 el Cl sale cero exacto.

Calidad de malla (`calidad.json`): 0 celdas cruzadas, ortogonalidad de pared
85.9° (p50 89.9°), crecimiento máximo 1.125, oblicuidad p99 0.033,
`div(u∞)` = **0 exacto**.

### Resolución de pared alcanzada

`y⁺` del primer centro de celda: **mediana 0.49, p95 1.62, máximo 1.88**. Con las
15 capas uniformes, la subcapa viscosa entera (`y⁺ < 5`) cae dentro de las
primeras 5 celdas y hay ~15 celdas por debajo de `y⁺ = 15`.

## Resultados

Convergido: entre `t* = 18` y `t* = 20` el Cl se mueve **3.9e-4** y el Cd
**2.7e-5**.

| | nuestro | XFOIL Ncrit=9 | XFOIL Ncrit=5 |
|---|---|---|---|
| **Cl** | 0.4984 | 0.6141 (**−18.8 %**) | 0.6026 (**−17.3 %**) |
| **Cd** | 0.01931 | 0.01674 (**+15.4 %**) | 0.01680 (**+15.0 %**) |
| **Cd presión** | 0.00757 | 0.00843 (−10.2 %) | 0.00766 (**−1.1 %**) |
| Cd viscoso | 0.01174 | 0.00831 (+41 %) | 0.00914 (+28 %) |
| **Cm** (c/4) | −0.00794 | −0.0077 | −0.0064 |
| transición | **ninguna** (turbulento desde el BA) | x/c = 0.366 | x/c = 0.268 |

Polar XFOIL de AirfoilTools, ya cacheada en el repo
(`Sim_Cartesiano/data/xfoil/`, `Sim_Cartesiano/data/ComparativasReales/`).

Otras salidas: `Cl` por ∮ΔCp 0.4953, por circulación 0.4887, ΔCp_TE = **−0.0278
sin parche de Kutta**, dispersión de Γ entre lazos 6.3 %, `ν_t/ν` máx 37.3,
divergencia relativa 1.6e-6, **5.94 it/s**.

### La discrepancia de Cl tiene una causa, y es falsable

**El Cd de presión coincide con XFOIL al 1.1 %.** El campo de presión, o sea la
forma del flujo exterior, está bien. Lo que no coincide es la **fricción, +28 %**,
y el Cl, −17 %. Los dos errores tienen el signo que predice una única causa:

> XFOIL transiciona en x/c = 0.27…0.37. Nuestro Spalart-Allmaras es **turbulento
> desde el borde de ataque**, porque no lleva modelo de transición.

Una capa límite turbulenta desde el BA es más gruesa (descamba el perfil → menos
sustentación) y arrastra más (→ más fricción). No hace falta invocar el esquema:
si fuera de discretización, el Cd de presión no cuadraría al 1 %.

Contraste con el solver cartesiano, que sobre el mismo perfil iba **+124 % de Cd
frente a XFOIL**. Aquí es +15 %, y de ese +15 % la parte explicada por la
transición ausente es toda.

**Lo siguiente es el modelo de transición** (SA-BC, ya validado en el solver
cartesiano), no tocar el esquema.

### Comprobaciones que no llevan ningún ajuste

- **Punto de remanso**: sale en `x/c = 0.0052, y = −0.0124` (intradós, como toca a
  α = 5°) con **`Cp = 1.005`**. El valor exacto es 1. **0.5 % de error.**
- **Pico de succión** `Cp = −1.65` en `x/c = 0.010`.
- **`Cf < 0` en el extradós solo desde `x/c = 0.966`**: burbuja de separación de
  borde de salida, que es lo que se espera a este Reynolds y ángulo.
- **`Cf < 0` en el intradós solo en `x/c ∈ [0, 0.0052]`**, aguas arriba del punto
  de remanso, donde el flujo va hacia delante de verdad.
- **`u⁺ = y⁺` hasta `y⁺ ≈ 10`** en las 7 estaciones: la subcapa viscosa está
  resuelta, no modelada.
- **Cd viscoso 0.0117** entre la placa plana laminar (2·1.328/√Re = **0.0084**) y
  la turbulenta (2·0.074/Re^0.2 = **0.0148**), que es donde tiene que estar una
  capa límite con `ν_t/ν` = 37.
- **ΔCp_TE = −0.0278 sin imponer Kutta**, frente a −0.092 del cartesiano **con**
  parche de Kutta.

### Sobre los tres estimadores de Cl

Superficie (0.4984) y ∮ΔCp (0.4953) coinciden al **0.63 %**: son la misma integral
de fuerza por dos caminos y tienen que coincidir. El de circulación (0.4887) va al
**1.9 %**, y **no tiene por qué cerrar**: es Kutta-Joukowski, `L = ρU∞Γ`, que
supone flujo no viscoso y estela delgada. A Re = 1e5 con estela real es una
comprobación de coherencia, no de exactitud. El criterio del plan de exigirle
< 1 % estaba mal puesto.

## Defecto abierto: tablero en la presión de pared

El `Cp` de la pared lleva una oscilación **par-impar pura** (la segunda diferencia
cambia de signo en el **100 %** de los índices; la malla no alterna — su `ds` solo
lo hace en el 8 %). Amplitud media en `Cp`: 0.006 a x/c = 0.2, 0.026 de media
sobre la cuerda, **0.11 cerca del borde de salida** (4 % del rango de `Cp`).

Lo medido sobre él:

- Vive **solo en `p`**. En `u` y `v` la componente alternante vale 9e-4 (0.09 % de
  U∞) y **bajó** respecto al arranque. La divergencia está en 1.6e-6.
- Es **uniforme en η**: idéntico de `j = 0` a `j = 20`, decae a partir de `j = 40`.
- **Sigue creciendo** despacio: 0.0140 (t\*=4) → 0.0186 → 0.0215 → 0.0238 →
  0.0258 (t\*=20). Se frena pero no se ha parado.

### Lo que NO es

| prueba | resultado |
|---|---|
| float64 en vez de float32 (t\*=3) | 1.196e-2 frente a **1.195e-2**. No es precisión. |
| `correcciones_p` = 8 en vez de 2 | **idéntico a 4 cifras**. No son los cruzados diferidos. |

O sea: el tablero **es la solución del sistema discreto**, no un error de
convergencia ni de redondeo.

### De dónde sale

Espiando la proyección (float64, 1200 pasos), componente alternante frente al
valor medio en la línea de pared:

| campo | alternante | \|valor\| medio | ratio |
|---|---|---|---|
| `div(m*)`, el término independiente del Poisson | 1.27e-9 | 1.06e-9 | **1.21** |
| `phi`, el incremento de presión del paso | 2.12e-6 | 2.64e-5 | 0.081 |
| `p` acumulada | 5.98e-3 | 1.93e-1 | 0.031 |

El campo provisional ya es solenoidal a **1e-9**, así que lo único que le queda al
término independiente *es* ruido par-impar: ratio 1.21, tablero casi puro. El
Poisson lo convierte fielmente en un `phi` con un 8 % de tablero, y `p = p + phi`
lo acumula 8000 veces. El origen está en `flujos_de_velocidad`, que reconstruye
los flujos de cara promediando velocidades de celda; ese mapa genera modo
par-impar sobre malla curva.

Por qué no contamina nada más: tanto la corrección de cara (diferencia compacta
`p_{i+1} − p_i`) como el gradiente Green-Gauss del momento promedian caras
adyacentes, y un tablero promediado da cero. Por eso `u`, `v` y los flujos salen
limpios, y al integrar el `Cp` sobre la pared el tablero se cancela — de ahí que
el Cd de presión cuadre al 1.1 % con XFOIL. Solo lo ve `fuerzas`, que lee
`p[0,i]` directo.

**Es cosmético para las fuerzas y real para el campo**, y crece, así que hay que
cerrarlo. Dos salidas, ninguna probada todavía: filtrar la componente par-impar
de `p` después de sumar `phi`, o sacar la presión de pared de la reconstrucción
por caras en vez de `p[0,i]`.

## Qué hay en la carpeta

```
correr.py            simula y vuelca; no dibuja nada
analizar.py          figuras, perfiles de capa límite y vídeo, desde lo volcado
caso.json            parámetros exactos del caso
malla.npz            X, Y (vértices) y el rango del perfil
calidad.json         métricas de calidad de la malla
historia.npz         401 filas x 17 columnas (ver `columnas`), cada t*=0.05
campos/              201 instantáneas de u, v, p, nu_t, cada t*=0.1
pared_final.npz      Cp, Cf, distancia del primer centro e y+ sobre la pared
resultados.json      tabla final + comparación con XFOIL, en limpio
figuras/             historia_coeficientes, ritmo, capa_limite_final,
                     pared_cp_ymas, velocidad_{dominio,perfil}_final
video/               velocidad_dominio.mp4, velocidad_perfil.mp4,
                     capa_limite.mp4  (201 marcos a 20 fps)
```

Las figuras se hacen **fuera del lazo**: dibujar dentro cuesta ~20 s por figura y
bloquea la GPU; desde los campos volcados son 0.6 s por marco.

## Coste

8000 pasos en **1307 s** = **6.12 it/s** de mediana (min 5.61, max 6.45) sobre
34 853 celdas, float32, RTX 3070 Ti. Ver `figuras/ritmo.png`.

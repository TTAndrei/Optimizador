# v5 — confirmación larga de la fase 3 de optimización (suavizador)

NACA 0012, α = 5°, Re = 1e5, SA, float32, malla C de producción (37 534 celdas),
dt = 2.5e-3, **8 000 pasos (t\* = 20)**. Generada con:

```bash
PYTHONPATH=. .venv/bin/python scripts/ab_curvo.py --correr validacion/v5_suavizador_pcr --pasos 8000
```

Es la corrida de referencia de los tres cambios de la fase 3: cuenta fija de
ciclos sin sincronizar, suavizador de la convección con líneas η solas, y
reducción cíclica paralela en las líneas del suavizador. La explicación y los
números intermedios están en `RESUMEN.md`, sección «Fase 3 de optimización».

## Contra la misma corrida con `dab73d1`

La base se corrió en un worktree de `dab73d1` en la misma sesión (el protocolo
pide medir base y variante seguidas: el mismo código ha dado 107.7–116.5 ms entre
sesiones distintas).

| magnitud | base | v5 | Δ % |
|---|---|---|---|
| Cl_sup | 0.498499936 | 0.498445969 | −0.0108 |
| Cl_circ | 0.487670134 | 0.487476501 | −0.0397 |
| Cl_dcp | 0.495356841 | 0.495303140 | −0.0108 |
| Cd | 0.0193136418 | 0.0193119156 | −0.0089 |
| Cd_p | 0.00756678721 | 0.00756661707 | −0.0022 |
| Cd_v | 0.0117468545 | 0.0117452986 | −0.0132 |
| Cm | −0.00791676517 | −0.00791747149 | +0.0089 |
| ΔCp_TE | −0.027857814 | −0.0278565176 | −0.0047 |
| mediana de `div` | 7.446e-06 | 7.607e-06 | +2.15 |
| máximo de `div` | 1.861e-04 | 1.798e-04 | −3.40 |
| tablero en `p` | 1.3355e-04 | 1.3461e-04 | +0.79 |
| tablero en `u` | 6.706e-07 | 6.595e-07 | −1.66 |
| ms/paso | 79.98 | **51.38** | **×1.557** |
| it/s | 12.50 | **19.46** | +55.7 |

Todas las fuerzas dentro del **0.05 %** acordado, y de hecho dentro del 0.011 %
salvo la circulación (−0.040 %, que es el estimador más ruidoso de los tres).

**El tablero, que es lo que había que vigilar**: en el A/B corto de 800 pasos
subía un 3 %, y aquí, con el campo asentado, se queda en **+0.79 %**. Era
transitorio, no deriva.

## Reparto por etapas

Con cronómetro, que **añade tres sincronizaciones por paso** y por eso su total
(53.4 ms) no coincide con el ms/paso citado:

| etapa | base | v5 | × |
|---|---|---|---|
| momento | 30.24 | 20.78 | 1.456 |
| presión | 33.79 | 18.39 | 1.838 |
| turbulencia | 18.17 | 14.24 | 1.276 |

## Ficheros

- `historia.npz` — las 21 columnas, una fila cada 20 pasos (401 filas)
- `metricas.json` — fuerzas finales, divergencia, tablero, reparto y ritmo
- `reparto.json` — el reparto por etapas

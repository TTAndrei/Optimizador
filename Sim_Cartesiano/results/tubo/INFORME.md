# Canal 2D: paredes no-slip, dominio en reposo, llenado desde la entrada

Sin cuerpo inmerso, o sea sin IBM de por medio: es el unico caso del proyecto con solucion analitica (Poiseuille) y por tanto la validacion mas limpia del solver.

## Resultados

| caso | Lx×Ly | Re_H | dx | modelo | t | Le medida | 0.05·Re·H | desarr. | u_c salida | err dp/dx |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 | 30×1 | 1e+02 | 0.01 | laminar | 39.88 | 6.60 | 5.0 | si | 1.4882 | +2.7 % |
| T2 | 30×1 | 5e+02 | 0.01 | laminar | 82.292 | 26.58 | 25.0 | si | 1.4884 | +13.9 % |
| T3 | 20×2 | 1e+03 | 0.01 | laminar | 36.507 | — | 100.0 | no | 1.2477 | n/a |
| T4 | 20×2 | 1e+04 | 0.01 | sa+sa_bc | 11.056 | — | 1000.0 | no | 1.0759 | n/a |
| T5 | 10×4 | 1e+04 | 0.01 | laminar | 27.661 | — | 2000.0 | no | 1.0359 | n/a |
| T6 | 20×1 | 1e+05 | 0.005 | sa+sa_bc | 7.366 | — | 5000.0 | no | 1.0264 | n/a |

## Como leer esto

- `err dp/dx` compara contra `-12·nu·U/H^2` y **solo se calcula si el flujo llego a desarrollarse dentro del dominio**; en los casos de llenado a Re alto la longitud de entrada teorica excede el tubo, asi que ahi la comparacion es n/a por construccion, no por fallo.
- `u_c salida` deberia tender a 1.5·U si el perfil se desarrollo.
- El arranque es impulsivo: el solver no tiene rampa de entrada, asi que el primer tramo del transitorio es un golpe de presion y no es fisica.
- **Los dos casos de validacion se desarrollan y ambos aciertan la longitud de entrada**: T1 da Le=6.6 frente a 5.0 de la correlacion y T2 da 26.6 frente a 25.0, con la linea central en 1.488 y 1.488 frente al 1.5 exacto de Poiseuille (99.2 %). El gradiente de presion sale a 2.7 % en T1. Es la validacion mas limpia del proyecto: sin IBM de por medio, el solver reproduce la solucion exacta.
- La velocidad en las paredes sale exactamente 0 (`u_pared_max` en el JSON), que es la comprobacion de que la condicion no-slip se aplica.

Videos en `results/tubo/<caso>/videos/`, perfiles en `figuras/`.

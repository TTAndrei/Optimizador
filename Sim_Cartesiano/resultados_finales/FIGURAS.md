# Inventario de figuras — resultados finales del TFG

Generado el 2026-08-26. **744 figuras** en el entregable, mas 248 archivadas que
NO deben usarse (ver el ultimo apartado).

Estudio: 2 perfiles x 4 mallas (dx = 0.008 / 0.006 / 0.004 / 0.002) x 9 angulos
(alpha = 0 a 8 de grado en grado), Re = 1e5, dominio 24x16 con el perfil en
cx = 6. Terna del GCI: **0.008 / 0.004 / 0.002**, con r = 2 constante.

---

## 1. Las que van en la memoria

Estas diez son las que sostienen el capitulo de resultados.

| Figura | Que enseña |
|---|---|
| `comparativa/cp_superficie_ganador_ag_a04.0.png` | **La mas explicativa.** Cp sobre extrados e intrados del ganador en el punto de diseño: meseta de succion lisa y recuperacion suave. Se lee junto a la del NACA 0012. |
| `comparativa/cp_superficie_naca0012_a04.0.png` | La misma magnitud para el NACA 0012: pico de succion mucho mas fuerte (-1.25) y oscilaciones violentas entre x/c 0.3 y 0.7. Ahi se ve **por que** el ganador dobla en eficiencia. |
| `comparativa/comparativa_extrapolada.png` | Los dos perfiles, Cl / Cd / L/D, malla fina frente a extrapolacion de Richardson. Es el resultado principal. |
| `comparativa/ganador_ag_vs_naca0012_dx0.0020.png` | Los dos perfiles en la malla fina con barras CI95. La comparacion sin extrapolar, la mas directa de defender. |
| `comparativa/polar_resistencia.png` | Polar Cl-Cd de los dos perfiles, con alpha etiquetado. |
| `richardson/figuras/richardson_ganador_ag_{cl,cd,ld}.png` | Las tres magnitudes del ganador: tres mallas + extrapolado, lineas suavizadas con PCHIP. |
| `richardson/figuras/richardson_naca0012_{cl,cd,ld}.png` | Lo mismo para el NACA 0012. |

## 2. Verificacion numerica

| Figura | Que enseña |
|---|---|
| `richardson/figuras/convergencia_ganador_ag_{cl,cd,ld}.png` | Una lamina por magnitud frente a dx, una linea por angulo. El tramo discontinuo entre la malla fina y la estrella en h=0 es la extrapolacion robusta, no un dato medido. Linea continua = orden observado utilizable; punteada = p forzado al nominal. |
| `richardson/figuras/convergencia_naca0012_{cl,cd,ld}.png` | Idem para el NACA 0012. |
| `richardson/figuras/richardson_resumen.png` | Las 6 curvas en una sola lamina 2x3. Util como resumen; el contenido es el mismo que las seis individuales. |
| `comparativa/malla_dx0.0020.png` | La malla variable: dominio completo y zoom al borde de ataque. Para el apartado de discretizacion. |
| `comparativa/cp_superficie_{ganador_ag,naca0012}_a00.0.png` | Cp de superficie a alpha = 0, una figura por perfil. Sirve de control de simetria del NACA 0012. |
| `perfiles/perfiles_comparados.png` | Geometria de los dos perfiles superpuesta. |

## 3. Polares por perfil

`{ganador_ag,naca0012}/figuras/polar/` — 4 figuras cada uno, 8 en total:
`polar_cl.png`, `polar_cd.png`, `polar_ld.png`, `polar_cl_cd.png`.
Cada una con las cuatro mallas superpuestas.

## 4. Campos (504 figuras)

`{perfil}/figuras/campos/dx<malla>_a<angulo>_<tipo>.png`
— 7 tipos x 36 puntos x 2 perfiles.

| Tipo | Contenido |
|---|---|
| `velocidad_zoom` | Modulo de velocidad, ventana sobre el perfil |
| `velocidad_dominio` | Modulo de velocidad, dominio 24x16 completo |
| `streamlines_zoom` | Lineas de corriente sobre el perfil |
| `streamlines_dominio` | Lineas de corriente, dominio completo |
| `presion_zoom` | Campo de presion (Cp) |
| `vorticidad_zoom` | Vorticidad |
| `vectores_zoom` | Vectores de velocidad |

## 5. Historias temporales (216 figuras)

`{perfil}/figuras/historia/dx<malla>_a<angulo>_<tipo>.png`
— 3 tipos x 36 puntos x 2 perfiles.

| Tipo | Contenido |
|---|---|
| `fuerzas` | Cl y Cd frente al tiempo fisico, con la ventana de promediado sombreada |
| `eficiencia` | L/D frente al tiempo |
| `residuos` | Residuo de la proyeccion, escala logaritmica |

## 6. Recuento

| Grupo | N |
|---|---|
| Campos | 504 |
| Historias | 216 |
| Polares por perfil | 8 |
| Richardson | 9 |
| Comparativa | 6 |
| Geometria | 1 |
| **Total entregable** | **744** |
| Archivadas, no usar | 248 |

Tamaño: 503 MB el entregable, 1.0 GB con el archivo.

---

## 7. Archivos que NO deben usarse

### `_archivo_malla_fina_descartada/` (50 ficheros, 40 figuras)

Malla dx = 0.001, descartada. El paso de tiempo colapsa a esa resolucion: de
0.002 a 0.001 el dt baja x0.15 en vez de x0.5, sobra un factor 3.4, y las 52000
iteraciones se quedaban en t~6 en vez de t~20. Esos puntos se midieron en pleno
transitorio y no son comparables con los de las otras mallas. Ver el `LEEME.md`
de esa carpeta.

### `_archivo_sin_warmstart/` (208 figuras)

Tanda anterior con una configuracion de solver distinta (sin `warm_start`) y con
el perfil ganador etiquetado como "ag24", que es un nombre equivocado: `AG24` es
un perfil real de catalogo (AG24 Bubble Dancer DLG, de Mark Drela) que se uso
como semilla del algoritmo genetico, no el ganador. Se conserva solo como
historico.

---

## 8. Como regenerar

```bash
# todas las figuras del estudio (campos, historias, polares, convergencia)
.venv/bin/python scripts/agent_tests/resultados_finales.py --solo-figuras

# solo el analisis de Richardson (richardson.json, tabla_gci.csv, INFORME.md)
.venv/bin/python scripts/agent_tests/resultados_finales.py --solo-analisis

# las 6 individuales de Richardson
.venv/bin/python scripts/agent_tests/rf_richardson_figuras.py

# Cp de superficie, polar de resistencia, malla y comparativa extrapolada
.venv/bin/python scripts/agent_tests/rf_figuras_extra.py

# comparacion de las dos ternas candidatas
.venv/bin/python scripts/agent_tests/rf_comparar_ternas.py
```

## 9. Avisos al citar

- **Punto de diseño: alpha = 4.** Las tres mallas coinciden en que el maximo de
  L/D esta ahi, y es el caso con el orden observado mas fiable del estudio
  (p = 2.56 en el Cl del ganador). El extrapolado deja alpha = 3 un 5% por
  encima, pero eso se debe a que su Cl aun no ha convergido en malla
  (correccion +6.2% frente a +0.5% en alpha = 4), no a que el pico este ahi.
- **El orden observado no es fiable en la mayoria de los casos**, sobre todo en
  el Cd. Donde `p_fiable = 0` en `tabla_gci.csv`, el valor extrapolado se
  calculo con el orden nominal p = 2 y es indicativo, no una medida del error de
  discretizacion.
- La malla de 0.006 se calculo como terna alternativa de contraste y **perdio**:
  35 casos buenos de 54 para 0.008 frente a 28 para 0.006, y GCI mediano 20.1%
  frente a 44.2%. Los datos estan en `richardson/comparativa_ternas.json`.
- La ventaja del ganador sobre el NACA 0012 **se mantiene en los ocho angulos,
  en las cuatro mallas y tambien extrapolada**. Es el resultado mas robusto del
  estudio: no depende de la malla ni del metodo de extrapolacion.

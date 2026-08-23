# RESUMEN — Optimizador CFD 2D

## Qué es
Simulador CFD 2D incompresible (Navier-Stokes) en GPU (CuPy, RTX 3070 Ti) para perfiles alares, base de un futuro optimizador aerodinámico. Todo el solver vive en `Simulador2D.py` (~8000 líneas, clase `Mesh` + `main()`).

- Malla cartesiana estirada (zona fina alrededor del perfil), advección semi-Lagrangiana, difusión (+WALE LES opcional), proyección de presión multigrid/CG.
- Sólidos por IBM (Immersed Boundary): máscara rasterizada + ghost-cell no-slip (`ibm_wall_mode="ghost_noslip"`).
- **Ejecutar SIEMPRE con `.venv/bin/python`** (el python3 del sistema no tiene CuPy).

## Estado actual (2026-08-24, rama Malla-Variable, commit `835a874` + trabajo sin commitear)

**SEGUNDA GRAN OPTIMIZACIÓN CERRADA** (11 épocas, refinado del top-3 y GCI de tres mallas del ganador). **La verificación del ganador cierra en negativo, y desde el 14 de agosto se sabe por qué en dos capas distintas:**

1. **El "Cl se movía un 6 % con la ventana" era un bug de mezcla de ventanas, no física** (`08af2c4`). `simular_perfil` dividía un Cl **instantáneo** (de `extract_surface_force_audit`, campo final y solo parte de presión) entre un Cd **promediado** (media de la cola de `cd_vector`, presión + viscoso). En flujo estacionario da igual; sobre el ganador a dx=0.001 —que oscila con periodo 0.20 y no llega a estacionario— el numerador dependía de en qué punto del ciclo cayó la última iteración: 0.7222 instantáneo frente a 0.6879 promediado. Corregido: el L/D usa `cl_val`, media temporal de la **misma** ventana que el Cd; la auditoría se conserva como diagnóstico en `cl_audit_inst`.
2. **Lo que sí sobrevive: el Cd oscila entre mallas y arrastra al L/D.** Y el estudio de Richardson sobre la polar completa (`835a874`, 5 ángulos × 3 mallas, ~48 h GPU) confirma que no es cosa de un ángulo desafortunado: **el Cl converge monótono en los cinco ángulos, el Cd rompe la monotonía en α=4 y α=6**. Ningún GCI de Cd ni de L/D es citable; solo la banda entre mallas.
3. **Y hay un tercer sesgo, medido en agosto y mayor que los dos anteriores: el dominio.** Toda la campaña del GA corrió en Lx=8, Ly=5 con fronteras top/bottom `slip` (túnel cerrado, no campo lejano). Ampliar a 24×16 sube el L/D del ganador de 27.70 a 31.70 a α=4. El **orden** entre individuos parece conservarse (medido en 2 de 3), el **valor absoluto** no.

**VERIFICACIÓN NUMÉRICA — documento en [`docs/verificacion_numerica.md`](docs/verificacion_numerica.md) (10 secciones).** Leerlo antes de citar cualquier L/D, pero **está desactualizado desde el 14 de agosto**: no incorpora la ventana larga, ni el fix del fitness, ni el estudio de dominio, ni el Richardson de la polar. Guion del TFG en `docs/PROMPT_TFG.md`; memoria en construcción en `docs/tfg/`, con el texto de sustitución de los capítulos 4-6 ya escrito en `docs/tfg/resultados_finales_v2.txt` (`99ffaa2`).

**Reorganización de resultados**: toda la campaña anterior está archivada en **`results/1eraGranOptimizacion/`** (`convergence_study/`, `verificacion_numerica/`, `agent_tests/`, `barridos/`, `metricas_ga/`, `comparativas_v1/`, `videos_ganadores/`, …). `results/verificacion_numerica/` en la raíz solo contiene lo nuevo. Cualquier ruta de este documento o de scripts antiguos que apunte a `results/convergence_study/...` hay que leerla bajo `results/1eraGranOptimizacion/`. El `.gitignore` se reancló a patrones `**/` porque los anclados a `results/` dejaron de aplicar tras el archivado.

### Criterio de parada — tercera versión, la puerta de ruido pasa a ser el CI95 (sin commitear)

Historia completa, porque las tres versiones anteriores fallaron por **medir la cosa equivocada** y conviene no repetirlo:

1. **Original**: comparaba cada serie con la media de su propio último 15 % — "¿me parezco a mí mismo hace poco?" en vez de "¿he dejado de cambiar?". Una deriva lenta y monótona lo satisface siempre: paraba en t≈1–2, errores hasta **47 %** en L/D. Lo enmascaraba el relleno de cola de `cdvector`/`clvector`, que daba σ=0.0 exacto (léase "no hay datos"): el 81.8 % de las 3884 evaluaciones de `aprendizaje_ML.jsonl` tienen σ=0.
2. **Recalibrado por tolerancias** (`clcd_tol_abs`/`clcd_tol_rel`): 0.82 % de error **de entrenamiento sobre 2 series**; sobre casos nuevos llegaba a **−23.7 %** con signo impredecible. El defecto estaba en la pregunta, no en los umbrales. Ambos kwargs **se eliminaron** (no desactivados): cualquier script que los pase falla.
3. **drift + ruido crudo** (`drift<0.005` y `σ/|media|<0.05` sobre ventana móvil de L/D, `8ab2461`): arregló el flujo estacionario y volvió el criterio **inaplicable con desprendimiento**. σ/|media| mide la amplitud de una oscilación física y permanente, no la incertidumbre del promedio: en el ganador a dx=0.001 vale 8.9 % con ventana 1 y **empeora a 11.7 %** al ensancharla, así que no abrió en 192000 iteraciones (t=71) mientras el drift ya pasaba holgado. También se descartó atar la ventana al periodo de desprendimiento: a dx=0.001 la estela no es periódica limpia (el pico de autocorrelación no llega a 0.2), no hay periodo que medir.
4. **En producción ahora** (`_detect_series_convergence` reescrita, calibrada por `scripts/agent_tests/criterio_ci95.py`): dos condiciones sobre una **ventana móvil que se ensancha sola** — se parte de `window_conv_time` y se dobla hasta que el CI95 baja del umbral o se agota la serie.
   - `ci95 = 1.96·σ/√N_ef / |media| < tol_ci95`, con **`N_ef` corregido por autocorrelación** (`_n_efectivo`, tiempo de autocorrelación integrado truncado en la primera autocorrelación no positiva). Con desprendimiento las muestras están correlacionadas y el CI95 ingenuo se cree N muestras cuando tiene unos pocos ciclos.
   - `drift = |pendiente|/|media| < tol_drift`, sobre esa misma ventana. Ensanchar de más no cuela un transitorio: un transitorio tiene pendiente y el drift lo caza.
   - Ensanchar hasta que el CI95 baje **no necesita que la señal sea periódica**: vale igual para un ciclo límite y para una estela de banda ancha, y no añade ningún parámetro nuevo.
   - Valores: **`clcd_tol_drift=0.002, clcd_tol_ci95=0.02, clcd_window_conv_time=1.0` (mínima, se ensancha), `clcd_n_sostenido=3, clcd_min_t_fisico_before_check=5.0`**.
   - **Calibrado sobre dos grupos, y los dos entran**: 20 series estacionarias a dx=0.004 del historial del GA + series con desprendimiento (polar del dominio C a dx=0.002, asintótico de α=4 hasta t=41.5, ventana larga a dx=0.001). Ajustar solo sobre las estacionarias es exactamente lo que produjo el criterio que no abre nunca. Resultados: CV peor error **1.15 %** (coste 0.72); grupo estacionario err máx 0.96 %, para en 15/16, coste 0.74; grupo desprendimiento err máx 1.15 %, para en 5/9, coste 0.78; conjunto retenido de 5 series err máx **0.43 %**, coste 0.88.
   - `criterio_ci95.py` **no reimplementa nada**: importa `_detect_series_convergence` del solver y la evalúa sobre las series guardadas, así que lo validado es el código que corre.
   - Fuente de verdad en fichero: `results/verificacion_numerica/criterio_parada.json` (**sobrescrito con el criterio CI95**; el ajuste completo está en `criterio_ci95.json`), leído por `verificacion_numerica.criterio_calibrado()`, `run_convergence_study.py`, `gci_ganador.py`, `test_espesor_malla.py` y `ventana_larga_dx001.py` — todos migrados de `tol_noise` a `tol_ci95`. `scripts/agent_tests/criterio_parada.py` queda marcado como **SUPERADO** en su docstring.
   - **`min_t=5.0` sigue siendo lo que arregla el modo de fallo restante**: sin él, dos series paraban en t≈4.1–4.4 con +10.6 % y +19.6 % — mesetas falsas previas al reataque de la burbuja laminar.
   - **Ojo antes de commitear**: el árbol tiene `_n_efectivo` **definida dos veces** en `Simulador2D.py` (líneas 7099 y 7121, idénticas). Borrar una.
- **El residual de campo no se usa** en el criterio de L/D (`tol_res=inf`). Pero **el guardián de estado estacionario `stop_on_convergence` sí actúa y ya ha estropeado datos**: truncó α=8 (10100 de 26000 iters) y α=10 (7950) en `polar_2grados_dom24x16`. En estudios de malla o de ventana hay que desactivar **las dos** paradas. `change_v` se descarta como residual: se normaliza por la norma de v, pequeña en flujo casi horizontal, vale ~0.5 permanentemente.
- **Cableado verificado en vivo** con la versión 3 (individuo 2019, dx=0.004): offline t_parada=7.013 / L/D=18.716 vs solver 7.0135 / 18.697. Ahorro frente a presupuesto fijo: 33–45 %.

**¿Sobrevive el ranking del GA? Medido, y la respuesta es no en la zona que importa** (`revalidar_ranking.py`, `results/1eraGranOptimizacion/verificacion_numerica/revalidacion_ranking.json`). Muestra estratificada por deciles de n=20 individuos del historial (filtrada a los 2425 registros de dx=0.004), re-simulada a dx=0.002 sin parada y con presupuesto fijo:

- Spearman ρ global **0.7955**, Kendall τ 0.6421, Pearson 0.8341, cambio mediano **+65.3 %** (IQR 47–104).
- Separando por la mediana: ρ=0.612 en los 10 peores, **ρ=0.152 en los 10 mejores**. Solape top-4 = **0.25**.
- Casos ilustrativos: **2019** es el mejor real de la muestra (L/D 24.06) y es de **generación 1** — el GA lo rankeó octavo y lo descartó. **2312** (gen 17) es el único con Δ negativo (−10 %): la parada le regalaba Cd bajo; cae del puesto 3 al 13.
- **Aviso metodológico**: con n=12 la lectura parcial era la contraria (top-4 en orden exacto). Una muestra estratificada truncada a mitad de ejecución no es una muestra estratificada.
- Conclusión defendible: el GA **sí se movió al barrio correcto** (ρ generación–L/D real 0.41, p=0.073) pero **eligió la casa con una señal que no discrimina a esa escala**.

**Y dx=0.004 SÍ sirve para rankear — el problema era la parada, no la malla** (`series_dx004.py`, mismos 20 individuos a dx=0.004 con presupuesto fijo):

| señal de fitness | ρ global | ρ mitad alta | solape top-4 |
|---|---|---|---|
| fitness que usó el GA (dx=0.004 + parada sesgada) | 0.7955 | 0.152 | 0.25 |
| **dx=0.004, presupuesto fijo / criterio nuevo** | **0.9263** | **0.7455** | **1.00** |

El sesgo en valor absoluto a dx=0.004 es de **−44.9 % ± 68** y no importa para el orden, pero **obliga a validar el ganador a dx=0.002**. Dos avisos: **1265** da L/D=**−1.29** a dx=0.004 (real 0.559) — el GA nunca vio fitness negativo porque la parada lo enmascaraba; **2312** engaña a las dos mallas gruesas por igual, ahí el artefacto es geométrico. **Matiz de agosto**: medido sobre geometrías del historial viejo; el ganador de `tfg2` (espesor 2.2 %) muestra que a esa escala de cuerpo fino el orden entre mallas deja de ser fiable.

**Validación externa contra XFOIL** (NACA0012 Re=1e5, Ncrit=9, airfoiltools; `validacion_xfoil.json`): **la sustentación es buena, la resistencia no.** MAE(Cl)=0.0912, pendiente dCl/dα=0.10211 vs XFOIL 0.11107 y teoría 2π 0.10966. MAE(Cd)=0.036, **error medio en Cd +123.8 %, monótono creciente con α** (−2.4 % a α=0 → +226 % a α=10). Causas plausibles: resolución de capa límite, separación/reataque prematuro de la burbuja, bidimensionalidad. **El L/D absoluto NO es comparable con XFOIL ni con experimento, y como el error de Cd depende de α tampoco es un factor de escala divisible.** Lo que sí se sostiene: Cl, pendiente de sustentación, y comparación perfil-contra-perfil a igual α, misma malla **y mismo dominio**.

**Transición SA-BC implementada y validada** (`8e29bd5`): `transition_model="sa_bc"` + `freestream_Tu=0.1` (default `"none"`, requiere `turb_model="sa"`). Intermitencia algebraica de Bas-Cakmakcıoğlu 2016: γ = 1−exp(−√T1−√T2) con Re_θc por la correlación de Menter, modulando **solo la producción** de SA; `SA_BC_CHI1=0.002`, `SA_BC_CHI2=5.0`; γ en `Mesh.sa_gamma`. Validada en `results/1eraGranOptimizacion/agent_tests/sabc_*`.

**Campaña de GA "1eraGranOptimizacion" completada** (`74add38`, `43d95e9`): modo `--mixto <tag>`, 4 semillas nuevas aptas a bajo Re (`SD7037`, `E387`, `SG6043`, `MH32`), flags `--multipunto/--delta-angulo/--fitness-modo`, `--transition-model/--freestream-tu`, `--run-tag`. **Todos esos L/D son de la señal sesgada; no citar.**

**Métricas del GA** (`scripts/agent_tests/metricas_ga.py`, sin GPU, ~1 min): 15 figuras + `INFORME.md`. Hallazgo: `usar_ia=False` en los 6 estudios, el surrogate RandomForest nunca filtró nada. Reentrenado offline sobre 2793 muestras Re=1e5: R²=0.850 (5-fold CV), MAE=1.07; a percentil 70 evitaría el 69 % del CFD perdiendo solo el 2.5 % de la élite. **Aún no pasado sobre `tfg2`.**

**Qué sigue siendo válido de la campaña anterior**: la metodología (GA de islas con migración, parametrización, solver, `wall_treatment="consistent"`, SA-BC), el coste (98.6 h-GPU, 3884 evaluaciones) y las conclusiones sobre el comportamiento del GA (migración, diversidad, presión selectiva, PCA: PC1+PC2 explican 75.5 % de la varianza geométrica). **Qué no**: todo valor absoluto de L/D anterior al recalibrado, la afirmación de que el ranking se conserva, los ganadores como óptimos, y el L/D como magnitud comparable con la literatura.

**`cl_cp_discrepancy` cayó de 0.31–0.83 a 0.02–0.14** al medir sin transitorio: buena parte de la sobre-circulación diagnosticada era artefacto. Queda un residuo real, menor, que crece con Re y que sobre cuerpos muy finos vuelve a levantar el flag (0.02–0.09 en las polares del dominio C).

### Segunda gran optimización (lanzada 2026-08-04, CERRADA 2026-08-13 en la época 10)

Lanzador `scripts/agent_tests/gran_optimizacion.sh {start|run|stop|status|snapshot|log}`; salidas en **`results/convergence_study/islands_tfg2/`**, log `results/convergence_study/gran_optimizacion_tfg2.log`, copias en `backups/tfg2_*`.

- **Presupuesto ejecutado**: 4 islas × pop 16 × 2 gen/época × **11 épocas** = 1232 evaluaciones nominales; **44 corridas GA cerradas y ~1160 evaluaciones CFD reales** (`aprendizaje_ML.jsonl` pasa de 3884 a **5044** registros), ~230 h-GPU. El plan original eran 13 épocas.
- **El recorte de 13 a 11 épocas fue conservador de más**: la ganancia por época había caído a +0.1…+0.5 con ruido entre épocas de ±1, pero las épocas 9 y 10 volvieron a dar saltos grandes (s1014 28.66, NACA 27.75). **Es la segunda vez que esta campaña finge convergencia**; la migración reordena la población con retardo y las mesetas de dos épocas no significan nada.
- **Config**: `--islands --run-tag tfg2 --island-epochs 11 --island-migrate-every 2 --island-pop 16 --dx 0.004 --t-target 12 --transition-model sa_bc --freestream-tu 0.1 --re 1e5 --refine-k 3 --stop-file results/convergence_study/STOP`. Dominio **8×5** — ver el estudio de dominio más abajo: es el sesgo más grande de toda la campaña.
- **Pausa/reanudación**: `stop` crea el centinela `STOP`; la corrida termina la evaluación en curso (≤12 min), guarda `estado_ga.json` y sale. `guardar_estado_ga` escribe a `.tmp` + fsync + `os.replace` (`2d2d639`). Se pausó/reanudó cinco veces sin perder trabajo pagado.

Mejor L/D por isla y época **a dx=0.004, dominio 8×5** (sirve para ordenar, no como valor final):

| época | GM15 | AG24 | NACA_0012_sharp | s1014 | spread |
|---|---|---|---|---|---|
| 0 | 16.96 | 15.61 | 14.71 | 7.14 | 9.82 |
| 1 | 19.90 | 19.28 | 17.70 | 8.03 | 11.87 |
| 2 | 21.81 | 21.49 | 19.82 | 21.77 | 2.00 |
| 3 | 22.90 | 22.09 | 22.65 | 22.71 | 0.81 |
| 4 | 23.13 | 22.70 | 22.66 | 23.29 | 0.63 |
| 5 | 24.85 | 23.06 | 23.22 | 25.27 | 2.21 |
| 6 | 25.44 | 24.73 | 24.95 | 25.66 | 0.93 |
| 7 | 25.75 | 25.58 | 24.98 | 25.47 | 0.78 |
| 8 | 25.49 | 24.57 | 25.47 | 25.15 | 0.92 |
| 9 | 25.98 | 24.79 | 26.55 | **28.66** | 3.88 |
| 10 | **26.84** | **25.00** | **27.75** | 25.92 | 2.75 |

- **Convergencia de la población**: distancia de forma media 0.0475 (época 0) → **0.0033** (época 10), spread de L/D 9.82 → 2.75. **La migración acerca las islas.** s1014 arrancaba en otra cuenca (distancia ~0.09 frente a ~0.006) y tardó tres épocas en integrarse.

**Refinado del top-3 a dx=0.002 (`refine_dx002.json`): el refinado CAMBIA al ganador.**

| individuo | L/D dx=0.004 | L/D dx=0.002 | Cl (ref) | Cd (ref) |
|---|---|---|---|---|
| **NACA_0012_sharp/epoch10** | 27.75 | **28.88** | 0.6965 | 0.02411 |
| GM15/epoch10 | 26.84 | 27.62 | 0.7725 | 0.02797 |
| s1014/epoch9 | **28.66** | 24.81 | 0.5791 | 0.02334 |

El líder de la exploración pierde 3.85 puntos al refinar mientras los otros dos suben. Ese perfil llevaba el espesor máximo al **84 % de la cuerda**. Lección: dos niveles de malla son necesarios, y el ranking a dx=0.004 deja de ser fiable justo en la zona de geometrías extremas a la que el GA tiende.

**GCI de tres mallas sobre el ganador** (`gci_ganador.py`, NACA_0012_sharp/epoch10, α=4°, Re=1e5, dominio 8×5, dx=0.004/0.002/0.001, r=2, FS=1.25). Geometría: t_max **2.16 %** en x/c=0.658, camber máx 5.64 %. **Recalculado en `08af2c4` con Cl y Cd de la misma ventana** (`gci_recalculado.json`, sin simular nada nuevo, a partir de las series guardadas; ventana t≥14, 3088 muestras, 285 ciclos de desprendimiento):

| magnitud | dx=0.004 | dx=0.002 | dx=0.001 | banda |
|---|---|---|---|---|
| Cl | 0.70144 | 0.68651 | 0.68783 | **0.19 %** |
| Cd | 0.02528 | 0.02388 | 0.02538 | **5.91 %** |
| L/D | 27.7455 | 28.7441 | 27.0971 | **6.08 %** |

- **El Cl queda prácticamente convergido entre las dos mallas finas** (0.6865 vs 0.6878). Ni el `p=1.52` ni el extrapolado 0.6841 de la lectura vieja son citables: con el punto corregido la razón de residuos es negativa y la convergencia es oscilatoria, pero la banda es del 0.19 %, o sea ruido.
- **Cd oscila** (baja y vuelve a subir, razón −0.93) y **arrastra al L/D** (razón −0.61). La banda citable del ganador se estrecha de 26.7–28.7 a **27.1–28.7**, y sigue siendo una banda, no un valor.
- **Pendiente barato**: dx=0.004 y dx=0.002 **todavía usan el Cl instantáneo**; allí el flujo llega a estacionario y la diferencia debería ser pequeña, pero hay que rehacerlas (~66 min) para que las tres mallas usen la misma definición.
- Fluctuación instantánea medida a dx=0.001: Cl **6.5 %**, Cd **15.5 %** — de ahí venía todo el ruido de estimador.
- **Estacionariedad por tramos** (cuatro tramos de t≈14 entre t=14 y t=71): L/D 27.01 / 27.15 / 27.18 / 27.05. El flujo es estadísticamente estacionario aunque nunca alcance estado estacionario.

**Ventana larga a dx=0.001** (`ventana_larga_dx001.py` → `results/verificacion_numerica/ventana_larga_dx001.json/.log/_series.npz`, `0263a13`). 192000 iteraciones de tope (t=71.0 físico alcanzado, cinco veces la corrida del GCI), **16.4 h de GPU**.

- **El criterio de la versión 3 no llegó a saltar** (`converged_clcd=false`, 192000 de 192000): esa fue la prueba que motivó la versión 4 (puerta de CI95). A esta resolución el perfil **no tiene estado estacionario**; las mallas gruesas sí convergen porque la difusión numérica amortigua el desprendimiento.
- **Cd: 0.025430 ± 0.000281 (CI95, 768 muestras) frente a 0.025507 con t=12 → 0.30 %, dentro del CI95.** El promediado temporal no era el problema: **la oscilación del Cd entre mallas es espacial y real**.
- **El Cl NO se movía un 6 %** — eso era el bug de fase del snapshot, corregido en `08af2c4`. La media temporal es 0.6878 con las dos ventanas.
- Queda un residuo real de dispersión entre estimadores sobre el mismo campo: Cl 0.6878 (media de superficie) frente a 0.6508 (circulación en lazo 0.15c) y 0.6419 (∮ΔCp en la cuerda) — **~7 %, no el 12 % que se creía**, pero el flag `cl_cp_discrepancy` sigue levantándose y `ΔCp_TE=−0.092` sigue doblando la tolerancia. Los lazos grandes bajan a Cl_circ 0.51 (0.40c y 0.80c) y 0.29 (1.50c): **la circulación no cierra**.
- **No repetir esta corrida**: el `.npz` de series está commiteado y el análisis se rehace offline.

**Ghost cells del borde de salida entre mallas** (`te_report_mallas.py` → `results/verificacion_numerica/te_report_mallas.json`, ~1 min sin simular): dx=0.004 → 11 ghosts en el TE, dx=0.002 → 14, dx=0.001 → 15; **irreparables en el TE = 0 en las tres** (1 irreparable en todo el dominio a dx=0.002). **La distribución es suave en dx**, así que la hipótesis "las ghost-cells de extradós e intradós se solapan y refinar lo empeora" **no está respaldada por el conteo**: el error de discretización no salta por ahí. La ambigüedad de fuerza sigue abierta, pero el sospechoso ya no es el reparto de ghosts.

### Estudio de independencia de dominio — el sesgo más grande de la campaña (sin commitear)

`scripts/agent_tests/estudio_dominio.py` → `results/verificacion_numerica/estudio_dominio/{INFORME.md,analisis.json,estudio_dominio.csv,png,campos/}`. Ganador de `islands_tfg2`, α=4°, dx=0.002, 13000 iters fijos; **solo cambia el dominio**.

Motivación: top/bottom son `slip` (v=0) — **paredes de túnel cerrado, no campo lejano** (`Simulador2D.py:7234-7235`). Con Ly=5 el perfil ve h=2.5c y la interferencia de sustentación vale σ=(π²/48)(c/h)²=0.033; el inflow fuerza v=0 a solo 2c, donde el vórtice ligado induciría un 2.7 % de U_inf. **Toda la campaña del GA se corrió ahí, así que el 27.75 es el del túnel, no el del perfil.** Los dos errores decaen con leyes distintas (lateral como (c/h)², longitudinal como c/d), así que el diseño separa los ejes: escalera A-B-C-D + factorial 2×2 A-E-F-C + caso G.

| | Lx×Ly | cx | σ | d_up | Cl | Cd | L/D | err Cl vs ∞ |
|---|---|---|---|---|---|---|---|---|
| A | 8×5 (el del GA) | 2 | 0.0329 | 2 | 0.6680 | 0.02411 | 27.70 | 2.36 % |
| B | 16×10 | 4 | 0.0082 | 4 | 0.6739 | 0.02118 | 31.81 | 1.50 % |
| **C** | **24×16** | **6** | **0.0032** | **6** | **0.6806** | **0.02147** | **31.70** | **0.52 % (dentro del CI95)** |
| D | 40×24 | 10 | 0.0014 | 10 | 0.6707 | 0.02201 | 30.47 | 1.96 % |
| E | 8×16 | 2 | 0.0032 | 2 | 0.6299 | 0.02699 | 23.34 | 7.93 % |
| F | 24×5 | 6 | 0.0329 | 6 | 0.6942 | 0.01928 | **36.01** | 1.46 % |
| G | 12×24 | 3 | 0.0014 | 3 | 0.6327 | 0.02553 | 24.78 | 7.52 % |

- **Extrapolación a dominio infinito** con `y = y_inf + k1·σ + k2·(c/d_up)`: **cl_inf = 0.6841**, **cd_inf = 0.01982** (residuo máx 0.0148 y 0.0016).
- **Manda Lx, no Ly.** E (8×16, ancho pero corto) es el peor caso de todos: alargar el dominio lateral sin alargar la estela empeora. G, igual. La estela truncada contra el `p=0` de salida es el error dominante (`u_out_min` 0.943 en A y E, 0.997 en C).
- **F (24×5) da el L/D más alto de todos (36.01) y hay que descartarlo**: `eps_top=0.0229` y `eps_bot=−0.0172` — es bloqueo de túnel puro, no física.
- **Los dos errores NO son aditivos**: el factorial da una interacción en Cl de +0.0245 sobre un efecto lateral de −0.014…−0.038 y longitudinal de +0.026…+0.051. **No vale corregir cada frontera por separado.**
- **Decisión: el dominio de referencia es C (Lx=24, Ly=16, cx=6)** para polares y simulaciones sueltas. Coste medido (`bench_dominio.py` → `bench_dominio.jsonl`): el estirado satura en dx_max, así que ampliar apenas añade celdas; el precio está en el Poisson — 7.18 it/s en 8×5 → 4.76 en 24×16 → 3.91 en 40×24. **C cuesta ~50 % más por iteración que el dominio del GA, no un factor.**

**¿Se conserva el ranking del GA al cambiar de dominio?** (`ranking_dominio.py` → `results/verificacion_numerica/ranking_dominio/`): top-1 y top-2 de la población final reevaluados con **el mismo presupuesto** (13000 iters, la campaña usó 10000/7750/5950 por el early-stop y con n distinto las barras no son comparables). top1 27.75 (fitness GA) → **31.81** en B; top2 27.23 → **29.51** en B. **El orden se conserva y la diferencia (2.30) supera el CI95 de cada uno (~1.1), pero por poco.** **Incompleto**: falta el top-3 y faltan las reevaluaciones en el dominio A con presupuesto igualado — sin ellas no se puede afirmar que el sesgo de dominio sea uniforme entre perfiles.

### Asintótico temporal y polares del dominio C (sin commitear)

**Cortar en t≈10 sobreestima el Cd ~5 %** (`asintotico_alpha4_domC.py` → `results/asintotico_alpha4_domC/`). Ganador, α=4, dx=0.002, dominio C, llevado a **t=41.5** (52000 iters, 3.4 h): el L/D de la cola sube 24.13 (t=5) → 32.33 (t=10) → **34.05 (t=20) y ahí se queda plano** (34.03 / 34.13 / 34.09 / 34.05 hasta t=41.5). **Toda la corrección está en el Cd** (0.02680 → 0.02099 → 0.02003); el Cl apenas se mueve (0.6818 desde t≈12). Doblar de t=20 a t=40 mueve el L/D 0.01.

**Extrapolación a t→∞** (`extrapolacion_polar.py`): **Richardson no vale en el tiempo** — la cola es un transitorio que decae y luego mesetea, no una ley de potencias. Contrastado contra el único punto con verdad medida (α=4, valor real 34.084): Richardson sobre T=5/10/20 da 39.41 (**+15.6 %**), el ajuste exponencial `f(T) = F∞ − A·e^(−T/τ)` sobre T, 2T y 4T da 34.66 (**+1.7 %**). El ajuste solo aplica si la razón de incrementos cae en (0,1); si no (desprendimiento sin meseta, o serie cortada por el guardián de residual) se marca **n/a** — ahí no hay valor asintótico que extraer.

**Polares disponibles, y NO son comparables entre sí punto a punto**:

| carpeta | dominio | α | t | notas |
|---|---|---|---|---|
| `results/verificacion_numerica/polar_{ganador_tfg2,naca0012}.*` | 8×5 (GA) | 0..10 de 2 en 2 | ~10 | con early-stop v3 |
| `results/polar_fina_1grado/` | 8×5 (GA) | 0..10 de 1 en 1 | ~10 | |
| `results/polar_fina_1grado_dom24x16/` | **C** | 1..10 de 1 en 1 | ~10 | el criterio no disparó ni una vez |
| `results/polar_2grados_dom24x16/` | **C** | 0..10 de 2 en 2 | ~20 | **early-stop desactivado**; α=8 y α=10 truncados por el guardián de residual |
| `results/richardson_polar_domC/` | **C** | 0..8 de 2 en 2 | ~20 | 3 mallas, paradas desactivadas |

Polar del ganador vs NACA0012 a dx=0.002 en el **dominio del GA** (L/D, α=0…10 de 2 en 2): 2.53 / 15.49 / **27.70** / 21.50 / 14.91 / 10.32 contra NACA 0.01 / 8.93 / 13.19 / 12.74 / 8.39 / 6.61.

Polar **corregida a t→∞ en el dominio C** (`figuras_analisis_polar.py` → `results/analisis_polar/`, columna `ld_fusion`; **no es simulación nueva**: es la polar de 1 grado a t≈10 multiplicada por el factor F∞/f(t=10) medido en los ángulos pares e interpolado, y solo se dibuja donde ese factor está acotado por dos ángulos con meseta):

| α | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| ganador | 10.37 | 18.50 | 32.40 | **32.67** | 29.01 | 22.01 | 17.79 | 14.36 | 11.72 | 10.09 |
| NACA0012 | 5.46 | 9.63 | 13.43 | **14.38** | 13.19 | 12.60 | 10.17 | 7.10 | 5.51 | 4.19 |

Máximo en el punto de diseño (α=3–4) y ventaja en todo el rango: **mejora de banda ancha, no sobreajuste al ángulo**. La ventaja es ~2.3× a α=4. **El valor absoluto no es citable** (ver Richardson abajo); la relación entre los dos perfiles, medida en el mismo dominio, misma malla y misma ventana, sí.

### Richardson sobre la POLAR — el Cd no converge en malla (`835a874`)

`scripts/agent_tests/richardson_polar_domC.py` → `results/richardson_polar_domC/{richardson.json,polares.csv,polar_dx*.json,series/,campos/,figuras/,run.log}`. Tres mallas (dx=0.004/0.002/0.001, r=2) × cinco ángulos (0, 2, 4, 6, 8) en el dominio C, **~48 h de GPU** (13000/26000/52000 iters, t≈20 igual en las tres; la malla fina cuesta ~6 h por ángulo).

**Por qué la polar y no un ángulo suelto**: con un solo ángulo no hay forma de distinguir "Richardson no aplica" de "ese ángulo cayó mal". Cinco ángulos dan cinco estimaciones independientes de p.

**Las dos paradas van desactivadas a propósito** (criterio de L/D y guardián de residual): cortan a t distinto en cada malla y meten dependencia de ventana dentro de la diferencia entre mallas, que es justo lo que se quiere medir. El criterio calibrado **sí se pasa al solver para que registre su veredicto sin actuar**, y además se reproduce offline sobre la serie volcada de cada punto con la función del propio solver — así queda medido dónde habría parado sin pagar el sesgo de que parara.

| magnitud | monótonos | p medio | banda media entre mallas |
|---|---|---|---|
| Cl | **5/5** | 1.29 ± 0.32 | 26.5 % |
| Cd | 3/5 (rompe en α=4 y α=6) | 2.98 ± 1.73 | 39.2 % |
| L/D | 4/5 (rompe en α=4) | 2.28 ± 1.43 | 58.8 % |

- **El Cl es lo único citable**: monótono en los cinco ángulos, con p consistente y del orden esperado. GCI fino **1.1–5.7 % fuera de α=0** (α=2: 5.7 %, α=4: 4.2 %, α=6: 1.1 %, α=8: 2.3 %). **α=0 es un caso aparte**: Cl 0.074/0.151/0.187, GCI fino 21 % — a sustentación casi nula el error relativo explota y el punto no dice nada.
- **El Cd rompe justo en el punto de diseño**: a α=4 vale 0.02426 / 0.01999 / 0.02317 (no monótono, GCI fino 50 %), y a α=6 la razón de residuos casi se anula (p=6.44, GCI fino 0.006 % — un artefacto, no una medida).
- **El L/D lo hereda**: a α=4 da 27.62 / **34.04** / 30.17 — banda del **21 %**. Ese 34.04 es el número que sale de la polar del dominio C a dx=0.002, y **la malla fina no lo confirma**.
- **Conclusión operativa: ningún GCI de Cd ni de L/D es citable; solo la banda entre mallas (19–55 %).** Para el Cl, sí se puede citar el GCI fino.
- Los campos finales (355 MB de u/v/p) quedan fuera del índice; están en local para instrumentar el borde de salida. La corrida α=8 a dx=0.001 con ventana corta se archivó aparte como `DESCARTADO_ventana_corta_*` en vez de mezclarla con el resto.

**El adelgazamiento del perfil NO es artefacto de malla** (`test_espesor_malla.py` → `espesor_vs_malla.json`). El GA lleva el espesor máximo del 12 % (semilla) a **~1.9–2.2 %** con el L/D subiendo monótonamente. Cuatro ganadores del mismo linaje reevaluados a dx=0.002:

| individuo | t_max | L/D dx=0.004 | L/D dx=0.002 | Cl | Cd |
|---|---|---|---|---|---|
| e0_GM15 | 6.13 % | 16.96 | 22.01 | 1.048 | 0.0476 |
| e3_GM15 | 3.46 % | 22.90 | 27.22 | 0.730 | 0.0268 |
| e6_GM15 | 1.91 % | 25.44 | 28.82 | 0.601 | 0.0209 |
| e6_s1014 | 2.13 % | 25.66 | **30.17** | 0.728 | 0.0241 |

**El orden se conserva** entre mallas dentro de ese linaje y la mejora solo se encoge de 1.513× a 1.371×. El adelgazamiento es física de Re bajo — el Cd cae a la mitad mientras el Cl solo baja de 1.05 a 0.73. **Matices**: el linaje es homogéneo, y el caso s1014/epoch9 (espesor máximo al 84 % de cuerda) demuestra que fuera de él la malla gruesa sí reordena.

Tres defectos que hacían inviable una corrida de días, corregidos en `5213410`:

1. **Fitness negativo indistinguible de un fallo.** `calcular_fitness` filtraba `ld > 0`, así que un perfil con L/D<0 devolvía 0.0 — el mismo valor que un crasheo. Ahora hay centinela explícito `Individuo.evaluado`; el pool de reproducción pasa de `fitness > 0` a `ind.resultados`.
2. **Reanudar costaba una generación entera**: el checkpoint guardaba el fitness pero no lo restauraba → 16 re-simulaciones, 3.1 h por pausa. Ahora restaura fitness, resultados y `evaluado`.
3. **La pausa no paraba nada**: RunGA absorbía el Ctrl+C y `_run_ga` no lo miraba. Ahora se re-lanza como `KeyboardInterrupt` y las fases que llaman a `simular_perfil` consultan el centinela entre simulaciones.

Además: `parada_pedida()` acepta fichero centinela (las corridas de verificación aceptan `STOP_SIMULATION.trigger` en la raíz); **`guardar_memoria` ya no pisa el histórico** — volcaba `datos_X` tal cual, así que una corrida con memoria vacía lo borraba todo (una prueba de humo de 10 individuos se llevó 2347 experiencias); barra de progreso a 1 actualización/minuto cuando la salida no es TTY; flags `--dx` y `--t-target`; `sim_last.npz`, `_temp_gen*_ind*.dat` y `refine_tmp/` fuera del control de versiones.

**Cambios de código pendientes de commit**: `Simulador2D.py` — `_n_efectivo` (duplicada, limpiar) y `_detect_series_convergence` reescrita sobre CI95 con ventana auto-ensanchable, devuelve `(drift, ci95, converged)`; kwarg `clcd_tol_noise` → **`clcd_tol_ci95`** (cualquier script que pase el viejo falla). `scripts/RunGA.py` — **`dump_field_path`**: vuelca el campo final (u, v, p, máscara sólida y los ejes 1D de la malla estirada) en el instante de parada, sea por convergencia o por contador; de ahí salen los mapas de |u| y streamlines de las polares. `verificacion_numerica.simular(..., dump_field=...)`. Scripts nuevos: `criterio_ci95.py`, `estudio_dominio.py`, `bench_dominio.py`, `ranking_dominio.py`, `asintotico_alpha4_domC.py`, `polar_fina.py`, `polar_fina_dominio_c.py`, `polar_2grados_dom24x16.py`, `polar_naca0012.py`, `extrapolacion_polar.py`, `figuras_analisis_polar.py`.

### Memoria del TFG (`docs/tfg/`)

Pipeline propio: fuente en `memoria.txt` (marcas `#1..#4`, `$$latex$$`, `[FIG]/[TBL]/[TOC]/[PB]`, tablas `| a | b |`), `build_tfg.py` sustituye `word/document.xml` sobre la plantilla oficial EETAC `MaquetaTFG.docx`, `latex2omml.py` convierte el subconjunto de LaTeX usado a OMML (lanza `ValueError` si algo no está soportado, para que falle en generación y no en Word) y `acentuar.py`/`acentuar2.py` restauran tildes usando `/usr/share/dict/spanish`. Salida versionada: `TFG.docx`.

- `resultados_finales_v2.txt` (`99ffaa2`, no se compila): texto de sustitución para los capítulos 4, 5 y 6, con la misma marcación y una tabla de trazabilidad de cada número contra su fichero de resultados. Cae la restricción del capítulo 4 (la sección 4.6 ya se puede escribir), los resultados van con banda de incertidumbre en vez de valor puntual, el capítulo 5 gana tres limitaciones medidas y el 6 cambia todas las cifras de energía y emisiones.
- **Tres apartados reescritos y compilados como extractos sueltos** (sin seguir): `3.4.2` difusión numérica del esquema de advección, `3.8.2` corrección de presión de fondo, `3.10.3` la evaluación CFD como función de fitness — cada uno con su `.txt` fuente, `.docx` y `.pdf`.
- `figuras/diagrama_solver.{excalidraw,png,svg}` — diagrama del solver, sin seguir.
- El flujo está a medio migrar: hay `TFG.odt` y `TFG_v1_con_tus_ediciones.odt` sin seguir (edición en LibreOffice) junto al `TFG.docx` generado. **Decidir si se sigue con `build_tfg.py` o se pasa a edición manual.**

**Estado del árbol**: `835a874` es el último commit; **queda sin commitear todo el bloque de agosto 15-24** — recalibrado CI95 (`Simulador2D.py`, 138 líneas), `dump_field_path` en `RunGA.py`, los 11 scripts nuevos de dominio/polar/extrapolación, `results/analisis_polar/`, `results/asintotico_alpha4_domC/`, las tres carpetas de polares, `estudio_dominio/`, `ranking_dominio/`, `bench_dominio*`, `criterio_ci95.json`, y los apartados del TFG. **`polar_results.json` sale modificado**: lo escribe `main()` en cada corrida suelta, es salida automática, no un resultado curado. **`cerebro_aerodinamico.pkl` sale modificado y sigue fuera de los commits**: en disco está el truncado por la prueba de humo (74 experiencias frente a las 2347 de `8ab2461`) y mezcla genes de longitudes distintas entre islas, así que el oráculo no consigue cargarlo — se reconstruye aparte, **no commitear esa versión**. `RESUMEN.md.tmp` es artefacto del hook. `graphify-out/` sale modificado (grafo del repo regenerado).

## Estado histórico (2026-07-12, rama Malla-Variable)

Caso de trabajo: NACA0012, α=5°, Re=1e5, Cl físico esperado ~0.55.

**FIX DE RAÍZ IMPLEMENTADO: `wall_treatment="consistent"`** (kwarg de `main()`, default "legacy" intacto). El sumidero IBM está eliminado. Componentes:
1. Divergencia por caras en la proyección (`divergence_form="face_flux"` → `_compute_flux_divergence_field_uv`): cara fluido-sólido = flujo 0 exacto → el Poisson ve el flujo contra la pared.
2. Gradiente de presión one-sided en celdas de pared (`wall_pressure_gradient_mode="one_sided"`).
3. `reforzar_impermeabilidad` desactivado (`disable_reforzar=True`) — era la cirugía que borraba masa (Q≈-0.06·U·c).
El Laplaciano masked ya era Neumann correcto; el problema era que divergencia y corrección anulaban la componente entera junto al sólido.

**Resultados validados** (JSON+npz en `results/1eraGranOptimizacion/agent_tests/wc_*`; `a5c14b7`):
- `wc_re1e3_a5` (Re=1e3, α=5): Cl=0.211±0.0004, Cd=0.148, Q_lazo=+0.02, ΔCp_TE=0.04. **Clava el DNS de referencia (Kurtulus Re=1000: Cl~0.22, Cd~0.146)**. Legacy daba 0.14 con Q=-0.06.
- `wc_sa_a0` (α=0, SA): Cl=0.0003, sin NaN — simetría perfecta.
- `wc_sa_a5_dx2_long` (48k iters, t=18.3, 2h GPU): Q=0.0032, **Cl converge en plateau 0.32**. **PROYECCIÓN EXONERADA: Q→0 no recupera el Cl.** Hipótesis Q→Cl pendiente ~20 FALSIFICADA.
- **Diagnóstico del déficit (0.32 vs 0.55)**: SA sobre-difunde. BL en x/c=0.5 de espesor ~0.07c (3× placa plana turbulenta), χ=nu_t/nu~40-50 (teórico ~8), Cp_min=-0.81 (esperado ~-1.9).
- `wc_sa_a5_dx2_lam01` (sa_nu_tilde_factor=0.1): **PEOR** — Cl=0.266, LSB reapareciendo. **Falsificada la hipótesis fully-turbulent**; mantener `sa_nu_tilde_factor=3.0`. (La transición se resolvió después con SA-BC.)
- `wc_sa_a5_dx1` (dx=0.001): **Cl=0.389** (vs 0.32 a dx=0.002). **Camino resolución confirmado, error ~1º orden en dx.**
- **Causa dominante identificada: difusión numérica del SL bilineal, nu_num≈u·dx/2** (independiente de dt) — 1º orden, ~5-10× nu molecular cerca de pared a Re=1e5. A Re=1e3 era ~10 % de nu (por eso clavó el DNS).
- **MacCormack implementado** (`advection_scheme="maccormack"`, default "sl"): corrección Selle 2008 con limitador min/max del stencil y **banda de 2 celdas junto al sólido en SL puro** (sin banda, la corrección sobre el staircase mete pico espurio Cp=-11 en TE). Validación Re=1e3 α=5: Cl=0.245/Cd=0.155.
- `wc_mc_sa_a5_dx2`: **Cl=0.452** (SL daba 0.32), Cp_min=-1.20, x_succión=0.015, Cd=0.040. **Cuarta capa (nu_num del SL) CONFIRMADA y resuelta.** Coste ~4 %.
- `wc_mc_sa_a5_dx1`: **Cl=0.496 ✓ OBJETIVO CUMPLIDO**, Cd=0.0335, Cp_min=-1.24, Q=0.005. **Richardson dx→0: Cl≈0.54 — clava el físico 0.55.**
- **CONFIG DE REFERENCIA DEL OPTIMIZADOR**: `wall_treatment="consistent"` + `turb_model="sa"` + `advection_scheme="maccormack"` (+ `transition_model="sa_bc"`). dx=0.002 para validación, dx=0.004 legítimo para explorar/rankear dentro de un linaje homogéneo.
- **POLAR VALIDADA** (`wc_mc_polar_a0/a2/a8`, dx=0.002): Cl(α)= 0(0.000), 2(0.185), 5(0.452), 8(0.708); Cd= 0.0305/0.0323/0.0404/0.0624. Monotonía ✓, pendiente 0.088/deg ✓, simetría α=0 exacta ✓. **DIAGNÓSTICO DE 4 CAPAS CERRADO.** (Todo sobre NACA0012, espesor 12 %.)
- **Script de barrido CFL/dx/α**: `scripts/agent_tests/run_cfl_sweep.py` — 30 sims resumibles. **Aún no ejecutado.**

**Diagnóstico histórico cerrado (4 capas, evidencia en `results/1eraGranOptimizacion/agent_tests/`)**:
1. LSB laminar a Re≥1e4 (burst t≈5-7, colapso). Fix: **Spalart-Allmaras**. WALE inerte.
2. Pared sub-resuelta: SA necesita dx≤0.002 (y+≲10).
3. Inconsistencia IBM↔proyección: legacy absorbía masa (sobre-circulación 2.2×). **Resuelta con `wall_treatment="consistent"`**.
4. Difusión numérica del SL bilineal. **Resuelta con `advection_scheme="maccormack"`**.
- **Quinta capa abierta (agosto 2026)**: definición de fuerza sobre cuerpos ultrafinos — la circulación no cierra (Cl_circ cae de 0.65 a 0.29 al agrandar el lazo), ~7 % de dispersión entre estimadores de Cl, ΔCp_TE del doble de la tolerancia. **El conteo de ghost-cells del TE es suave entre mallas, así que el reparto de ghosts no es el mecanismo**; refinar tampoco lo arregla.
- Métrica decisiva: Q_lazo=∮u·n dl (`q_loop_025c/075c`).

**GA de optimización de perfiles (`scripts/RunGA.py`)**: `simular_perfil` corre sobre consistent+SA+MacCormack, `main(config)` reutilizable con retorno dict, población mixta multi-semilla (`archivos_base` + `resamplear_a_grid`), parada por `tiempo_limite_s`, mutación con sigma adaptativa, parada por estancamiento, checkpoint de genes en `estado_ga.json` (escritura atómica). Orquestador `scripts/agent_tests/run_convergence_study.py` (deadline-driven, resumible): Fase0 calibración dx, Arm A GA por-semilla, Arm B población mixta, S2 islas con migración (`--islands`), S3 refine top-k, `--aislado`, `--mixto`, `--multipunto`. Memoria de ML persistente: `cerebro_aerodinamico.pkl` + `aprendizaje_ML.jsonl`.

**Estudio de convergencia (CERRADO, `a0ee4f3`)**: Arm A no converge (4 semillas → 4 óptimos locales, CV=0.172); el test aislado sin migración (28 h GPU) da dist-forma media **0.0557** (igual/peor que la partida 0.0507) → **la convergencia que producen las islas la fuerza la MIGRACIÓN, no una física de óptimo único**. Fixes asociados: `bf6824e`, `ac235b1`, `9309f00`, `30c7cf6`.

## Próximos pasos

**Prioridad (2026-08-24)**

1. **Commitear el bloque de agosto 15-24 antes de tocar nada más.** Son ~10 días de trabajo sin seguir: recalibrado CI95, estudio de dominio, polares del dominio C, asintótico, Richardson de la polar ya está commiteado pero el criterio que usa no. **Antes**: borrar la definición duplicada de `_n_efectivo` en `Simulador2D.py` y decidir qué entra del `results/` sin seguir (hay carpetas de campos pesadas; el `.gitignore` ya excluye los 355 MB del Richardson).
2. **Rehacer dx=0.004 y dx=0.002 del GCI del ganador con el Cl promediado** (~66 min de GPU). Es lo único que falta para que las tres mallas usen la misma definición de fuerza y la banda 27.1–28.7 sea homogénea. Barato y cierra un pendiente explícito de `gci_recalculado.json`.
3. **Terminar `ranking_dominio.py`**: falta el top-3 y faltan las reevaluaciones en el dominio A con presupuesto igualado. Sin ellas no se puede afirmar que el sesgo de dominio sea uniforme entre perfiles — y de eso depende que la campaña entera del GA sea rescatable como *ranking* o haya que rehacerla en dominio C.
4. **Cerrar la definición de fuerza sobre cuerpos ultrafinos.** Sigue abierta: ~7 % entre estimadores de Cl, `ΔCp_TE=−0.092`, circulación que no cierra. El conteo de ghost-cells **ya está descartado como mecanismo** (`te_report_mallas.json`: 11/14/15 ghosts, 0 irreparables en las tres mallas), así que el siguiente paso es comprobar si `kutta_enforce` cierra la circulación y decidir qué estimador es el válido. Los campos finales de las tres mallas están en local para instrumentarlo.
5. **Diagnosticar el exceso de Cd** (+226 % a α=10 frente a XFOIL, y no monótono en malla a α=4 y 6): perfil de capa límite en pared a dx=0.001 vs dx=0.002 y posición de reataque de la burbuja frente a XFOIL. **Es la misma familia de problema que el punto 4** — el GA converge a cuerpos de 2 % de espesor, así que un Cd sesgado sesga el óptimo de espesor.
6. **Post-proceso de `tfg2`**: pasar `metricas_ga.py` sobre `results/convergence_study/islands_tfg2/` (aún no hecho) y **reconstruir `cerebro_aerodinamico.pkl`** — el de disco está truncado a 74 experiencias y mezcla genes de longitudes distintas entre islas; hay que resamplear a un grid común antes de reentrenar (causa del error "inhomogeneous shape").
7. **Activar el surrogate** (`usar_ia=True`, umbral percentil 70): ahorro estimado del 69 % del CFD con pérdida del 2.5 % de la élite. Reentrenar sobre los **1160 registros nuevos de `tfg2`**, no sobre el historial sesgado. Depende del punto 6.
8. **Si hay una tercera campaña de GA, correrla en el dominio C** — cuesta ~50 % más por iteración, no un factor, y el 8×5 mete un sesgo de +4 puntos de L/D en el valor absoluto y una interacción no aditiva entre fronteras. Alternativa barata: mantener 8×5 para explorar y refinar el top-k en C.
9. **GA multipunto α∈{2,4,6}**: etapa `multipunto` ya cableada en `cola_tfg.sh` (fuera de la cola por defecto). **Antes conviene resolver 4 y 5** — si la fuerza no está bien definida en la geometría a la que converge el GA, multiplicar ángulos multiplica el problema.
10. **Terminar la memoria del TFG** (`memoria.txt` + `resultados_finales_v2.txt` → `build_tfg.py`). Cómo contarlo: (a) **no citar el 28.88 ni el 34.04**, dar bandas con la advertencia de que la incertidumbre dominante es la no-convergencia del Cd en malla; (b) el cambio de ganador en el refinado como resultado metodológico; (c) el estudio de dominio como el sesgo mayor y mejor medido de la campaña; (d) la ventana larga y el Richardson de la polar como verificaciones que **descartan** hipótesis (promediado temporal, "fue mala suerte del ángulo") — resultados negativos bien medidos, que es lo que pide una verificación numérica; (e) el bug de mezcla de ventanas del fitness como ejemplo de por qué una diferencia del 5-6 % puede no ser física. **Actualizar `docs/verificacion_numerica.md`**, que está congelado antes de todo esto.

**Anteriores (siguen abiertos)**

11. Error no fatal de carga de `cerebro_aerodinamico.pkl` ("inhomogeneous shape") — **causa identificada**: el `.pkl` mezcla genes de longitudes distintas entre islas. Se manifestaba en cada arranque de isla de `tfg2` y por eso el surrogate arrancaba en frío en cada época. Arreglo pendiente junto al punto 6.
12. Estudio de convergencia CERRADO: no relanzar corridas de diagnóstico de la misma pregunta. Direcciones abiertas: más diversidad/multi-arranque si se quiere un óptimo global real; revisar físicamente el óptimo de camber alto.
13. **Ejecutar el barrido** `scripts/agent_tests/run_cfl_sweep.py` (30 sims, resumible) — sigue pendiente, ninguna sim lanzada.
14. Opcional barato: probar MacCormack+turbo_hd (~19 it/s esperado) para pre-screening.
15. Actualizar rutas en scripts/notebooks que apunten a `results/convergence_study/...` o `results/verificacion_numerica/{gci,polar,reeval_*}.json` — los de la campaña vieja viven bajo `results/1eraGranOptimizacion/`. **Cuidado**: `results/convergence_study/` contiene la campaña NUEVA (`islands_tfg2/`), así que una ruta antigua ya no falla — apunta a datos distintos.
16. `polar_results.json` en la raíz lo escribe `main()` en cada corrida suelta y se ensucia con puntos de verificación. No es una polar curada; no leerlo como resultado.

## Tests
- `scripts/agent_tests/richardson_polar_domC.py` — Richardson de tres mallas × cinco ángulos en el dominio C, paradas desactivadas, presupuesto fijo t≈20 → `results/richardson_polar_domC/`. **~48 h GPU, ya ejecutado y commeteado (`835a874`); no relanzar.** Reanudable por punto.
- `scripts/agent_tests/te_report_mallas.py` — ghosts e irreparables en el TE para las tres mallas. Solo construye la malla y corre una iteración: ~1 min, sin GPU real → `results/verificacion_numerica/te_report_mallas.json`.
- `scripts/agent_tests/criterio_ci95.py` — recalibración del criterio de parada con puerta de CI95 sobre N efectivo; importa la función del solver y la evalúa sobre series guardadas (sin GPU) → `criterio_ci95.json` (+ sobrescribe `criterio_parada.json`). **Sustituye a `criterio_parada.py`, que queda como histórico.**
- `scripts/agent_tests/estudio_dominio.py [--solo A,E] [--analisis]` — independencia de dominio, 7 casos a dx=0.002 → `estudio_dominio/{INFORME.md,analisis.json,csv,png,campos/}`. **No se lanza solo.** ~5 h GPU.
- `scripts/agent_tests/bench_dominio.py --lx 12 --ly 8 --cx 3` — coste por iteración vs tamaño de dominio, un caso por invocación (subproceso limpio, sin fragmentar la GPU) → `bench_dominio.jsonl`.
- `scripts/agent_tests/ranking_dominio.py [--analisis]` — ¿el orden del GA sobrevive al cambio de dominio? Mismo presupuesto para todos. **Incompleto: falta el top-3 y el dominio A.**
- `scripts/agent_tests/asintotico_alpha4_domC.py [--analisis]` — α=4 del ganador hasta t≈41.5 en dominio C, vuelca serie completa y campo final → `results/asintotico_alpha4_domC/`. 3.4 h GPU, ya ejecutado.
- `scripts/agent_tests/polar_fina.py` (dominio GA) / `polar_fina_dominio_c.py` (dominio C, 1..10 de grado en grado) / `polar_2grados_dom24x16.py` (dominio C, t≈20, early-stop desactivado) / `polar_naca0012.py` — polares reanudables y pausables con `STOP_SIMULATION.trigger`; cada punto cachea su JSON y vuelca campo + streamlines.
- `scripts/agent_tests/extrapolacion_polar.py` — extrapola cada punto a t→∞ con ajuste exponencial sobre T/2T/4T (Richardson **no** vale en el tiempo). Marca n/a donde no hay meseta.
- `scripts/agent_tests/figuras_analisis_polar.py` — figuras del análisis de las dos polares del dominio C → `results/analisis_polar/` (incluye la polar corregida a 1 grado, que no es simulación nueva).
- `scripts/agent_tests/ventana_larga_dx001.py` — ganador a dx=0.001 con tope de 192000 iteraciones. **16.4 h GPU, ya ejecutado y commiteado; no relanzar.**
- `scripts/agent_tests/gci_ganador.py` — GCI de Roache de tres mallas sobre Cl, Cd y L/D; reanudable, cachea cada malla → `gci_ganador.json`. ~4.3 h GPU, de las que 3.3 h son la malla fina. **Pendiente: rehacer las dos mallas gruesas con el Cl promediado.**
- `scripts/agent_tests/test_espesor_malla.py` — reevalúa a dx=0.002 ganadores de distintas épocas de una isla → `espesor_vs_malla.json`.
- `scripts/agent_tests/gran_optimizacion.sh {start|run|stop|status|snapshot|log}` — segunda gran optimización (islas `tfg2`). **Campaña terminada**; queda como plantilla. `stop` deja el centinela `STOP` y espera al checkpoint (≤12 min).
- `scripts/agent_tests/cola_tfg.sh {run [etapa]|status}` — cola serie reanudable: `calibrar → reeval → gci → polar` (+ `multipunto`, fuera por defecto). `cola_final.sh` — `repolar → validacion → revalidacion`. `cola.sh` — cola genérica de GA.
- `scripts/agent_tests/verificacion_numerica.py --calibrar|--reeval|--gci|--polar|--repolar|--validacion` — etapas sueltas; `simular()` (con `dump`/`dump_field`), `criterio_calibrado()` y `T_TARGET_CASO` viven aquí.
- `scripts/agent_tests/series_dx004.py` — 20 series completas a dx=0.004 sin parar: referencia del criterio y test de ranking entre mallas.
- `scripts/agent_tests/revalidar_ranking.py --n 30` — muestreo estratificado del historial y correlación de rangos; cacheado y reanudable.
- `scripts/agent_tests/metricas_ga.py` — post-proceso del GA sin GPU (~1 min): 15 figuras + `INFORME.md`. **Pendiente de pasar sobre `tfg2`.**
- `scripts/agent_tests/mixto.sh` — lanzador de los GA de población mixta. `video_ganador.py` — vídeo del campo. `plot_comparacion_transicion.py` — SA vs SA-BC. `validar_aislado_dx002.py`, `t1_ranking_dx.py`.
- `scripts/agent_tests/run_kutta_tests.py {smoke|smoke_polygon|baseline_a5|polygon_a5|compare|regression}` — runs coarse (dx=0.004, ~5-15 min) con criterios cuantitativos. **Punto de entrada para el diagnóstico del TE** (punto 4).
- `scripts/agent_tests/run_cfl_sweep.py` — barrido CFL/dx/α resumible (pendiente de ejecutar).
- `scripts/agent_tests/run_convergence_study.py --aislado` — test sin migración. **Estudio cerrado**; no relanzar.
- `scripts/comparativa_barrido_completo.py --smoke` — validación rápida del pipeline outer_sum vs XFoil sin gastar GPU real.
- `pytest tests/ --ignore=tests/test_generacion_geometrica_ga.py` (5 passed; el de GA está roto pre-existente: importa `RunGA` inexistente).
- `docs/tfg/build_tfg.py [--plantilla RUTA] [--salida RUTA]` — regenera `TFG.docx` desde `memoria.txt` sobre la plantilla EETAC (sin GPU).

## Diagnósticos clave (métodos de Mesh)
- `compute_circulation()` — Γ en lazos, Cl_circ=-2Γ/(U·c). **Con el ganador de `tfg2` los lazos dan 0.65 / 0.51 / 0.51 / 0.29 (0.15c → 1.50c): la circulación no cierra.**
- `compute_cp_diagnostics(mu, rho)` — Cl(∮ΔCp), ΔCp_TE, pico de succión.
- `compute_kutta_dcp_instant(face_data)` — ΔCp_TE instantáneo para el monitor.
- `debug_te_report()` — ghosts/irreparables cerca del TE (usado por `te_report_mallas.py`).
- `plot_streamlines()`, `plot_forces_over_time()`, `plot_cp_vs_chord()`.
- Series por muestra: `tvector` (tiempo físico; el dt es adaptativo y el tiempo no se reconstruye desde el índice), `clvector`/`cdvector`, `resvector` (change_u/v/p), `n_muestras_validas`.
- Del retorno de `simular_perfil`: `cl_ci95`/`cd_ci95`/`ld_ci95` y `n_samples` (**comparar cambios entre corridas contra el CI95 antes de llamarlos reales**), `converged_clcd`, `t_conv_clcd`, `iters_efectivas`, `cl_cp_discrepancy` + `cl_cp_discrepancy_flag`, **`cl_audit_inst`** (Cl instantáneo de la auditoría de superficie: diagnóstico, **no** entra en el L/D desde `08af2c4`), `dump_series_path` (.npz de t/Cl/Cd/L·D) y `dump_field_path` (.npz del campo final).
- Diagnósticos de frontera leídos del campo final (`estudio_dominio.py`): `eps_top/eps_bot` (aceleración media en la pared slip = bloqueo medido), `v_in_max` (|v| justo dentro del inflow, que el BC fuerza a 0), `u_out_min` (estela viva en la salida si ≪1), `cp_wall_range` (si no es ~0, la pared aún ve el perfil).

## Config de referencia (fina, `__main__`)
Lx=12, Ly=8, dx_min=0.001 (malla 3243×1203, ~2 it/s), CFL=0.25, NACA_0012_sharp, min_te_height_factor=1.0, turbo_hd, wake long_fine_x, `wall_treatment="consistent"`, `turb_model="sa"`, `transition_model="sa_bc"`, `advection_scheme="maccormack"`. Parada por convergencia con los defaults nuevos (`clcd_tol_drift=0.005`, `clcd_tol_ci95=0.02`, `clcd_window_conv_time=1.0` mínima y auto-ensanchable, `clcd_min_t_fisico_before_check=5.0`; el fichero calibrado baja `tol_drift` a 0.002 y sube `n_sostenido` a 3).

**Dominio**: para polares y simulaciones sueltas usar **C — Lx=24, Ly=16, cx=6**, no el 8×5 del GA. Manda Lx (estela), no Ly. **En estudios de malla o de ventana, desactivar las DOS paradas** (criterio de L/D y guardián de residual `stop_on_convergence`).

**Nota**: el bloque `__main__` puede quedar como scratch de pruebas entre sesiones — usarlo como plantilla, no copiarlo literal.

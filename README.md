# Optimizador — dos solvers CFD 2D y el optimizador que los usa

```
.
├── RESUMEN.md            estado del proyecto; leer esto primero
├── requirements.txt
├── profiles/             perfiles .dat  — COMPARTIDO por los dos solvers
├── docs/                 memoria, analisis y documentacion del proyecto
├── graphify-out/         grafo del repo (generado)
│
├── curvo/                solver curvilineo sobre malla C adaptada al cuerpo
│   ├── malla.py          generador de malla C (marcha hiperbolica)
│   ├── metrica.py        areas de cara, volumenes, coeficientes (numpy o cupy)
│   ├── operadores.py     divergencia, gradiente, laplaciano de metrica completa
│   ├── multigrid.py      ACM + kernels CUDA del suavizador y la aglomeracion
│   ├── conveccion.py     conveccion-difusion implicita conservativa (TVD)
│   ├── proyeccion.py     proyeccion de presion, Poisson compacto + PCG
│   ├── solver.py         Navier-Stokes por paso fraccionado
│   └── fuerzas.py        Cl, Cd, Cm y Cp integrados sobre la pared
├── tests/                sus tests
│
└── Sim_Cartesiano/       solver cartesiano + IBM, CONGELADO, y todo lo suyo
    ├── Simulador2D.py    opt_solver.py  bl_correction.py  geom_import.py
    ├── scripts/          GA, barridos, diagnosticos, figuras
    ├── gui/              interfaz
    ├── tests/            sus tests
    ├── configs/  data/   configuracion y datos de entrada
    ├── results/  resultados_finales/  resultados_ga/  plots/  figuras_memoria/
    ├── media/  validacion_resultados/  prueba_inicialesdiferentes/
    ├── lanzar_*.sh       lanzadores (hacen `cd` a esta carpeta)
    └── profiles -> ../profiles
```

## Los dos solvers

| | `Sim_Cartesiano/` | `curvo/` |
|---|---|---|
| malla | cartesiana + frontera inmersa | C monobloque adaptada al cuerpo |
| superficie | rasterizada (escalonada) | **es** la linea `j=0` de la malla |
| adveccion | semi-Lagrangiano + MacCormack | volumenes finitos conservativos, TVD, implicito |
| GPU | CuPy + kernels CUDA | CuPy + kernels CUDA, o numpy sin GPU |
| celdas tipicas | 3.9 M | 21 k (misma resolucion de pared) |
| estado | **congelado**, bit a bit reproducible | en construccion (F0-F4 cerradas) |

`Simulador2D.py` no se toca: los resultados del TFG salieron de el y tienen que
poder reproducirse. El trabajo nuevo va en `curvo/`, que es un modulo aparte.

## Como se ejecuta

El entorno (`.venv/`) vive en la raiz. **Siempre `.venv/bin/python`**: el python
del sistema no tiene CuPy.

```bash
# --- solver curvilineo (desde la raiz) ---
.venv/bin/python -m pytest tests/ -q          # 13 de los tests piden GPU
.venv/bin/python -m curvo.malla --perfil profiles/NACA_0012_sharp --figura

# --- solver cartesiano (desde su carpeta) ---
cd Sim_Cartesiano
../.venv/bin/python -m pytest tests/ -q
./lanzar_gui.sh
../.venv/bin/python scripts/RunGA.py
```

El cartesiano se ejecuta desde `Sim_Cartesiano/` porque sus scripts escriben con
rutas relativas a esa carpeta (`results/`, `sim_last.npz`, `polar_results.json`).
Sus tests pasan igual desde la raiz: su `conftest.py` mueve el directorio de
trabajo a `Sim_Cartesiano/` mientras corren, para que no ensucien la raiz.

Invocar `../.venv/bin/python` desde dentro funciona pero saca un
`RuntimeWarning: Unexpected value in sys.prefix` (la ruta no es canonica); es
inocuo. Los `lanzar_*.sh` usan la ruta absoluta y no lo emiten.

### Campana de polares del solver curvo

La campaña reanudable de NACA 0012 sharp y AG24 se lanza desde la raiz:

```bash
.venv/bin/python scripts/campana_polares_curvo.py
```

Por defecto barre alfa de 0 a 15 grados en pasos de 1, usa los parametros de
malla y solver por defecto, y guarda cada corrida bajo
`resultados_polares_curvo/<perfil>/`. El estado global queda en
`estado.json`, por lo que se puede interrumpir con `Ctrl-C` y repetir el mismo
comando para continuar. Para una prueba corta:

```bash
.venv/bin/python scripts/campana_polares_curvo.py \
    --backend numpy --pasos 100 --cada 10
```

Cada angulo guarda historia, campos, pared, capa limite, calidad de malla y
metricas de `Cl`, `Cd`, `Cm`, `Cl` por tres estimadores, divergencia, `it/s`,
iteraciones de Poisson, `dCp` en el borde de salida, `nu_t/nu` y fraccion de
pared con `Cf < 0`. Las graficas quedan en
`<perfil>/graficas/`: polar, `Cl/Cd/L/D` frente a alfa, rendimiento y
divergencia, `Cp/Cf` de pared y perfiles de capa limite.

Si la calidad de malla de un perfil no es valida, la corrida se marca como
`fallida_malla`, se conserva su diagnostico y la campaña continua sin ejecutar
el solver sobre datos degenerados. Con los defaults actuales AG24 presenta
celdas con Jacobiano no positivo y queda marcado de esta forma hasta ajustar
su malla.

`profiles/` esta en la raiz porque lo usan los dos; dentro de `Sim_Cartesiano/`
hay un enlace simbolico para que las rutas `ROOT/profiles/...` de su codigo sigan
resolviendo sin tocar ni una linea.

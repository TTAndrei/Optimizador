#!/bin/bash
# Validacion al presupuesto de produccion: t~20, 26000 iters, dx=0.002.
#
# Hasta ahora todo se ha medido a 4000 iters (t~2.5), en pleno transitorio. La
# pregunta que queda es si las optimizaciones siguen dando lo mismo en regimen, y
# cuanto se mueven Cl y Cd frente al numero que ya esta publicado:
#
#   results/richardson_polar_domC/polar_dx0.0020.json, alpha=4
#   Cl = 0.680389   Cd = 0.019986   L/D = 34.0436   (26000 iters, 6171 s)
#
# Se corren TRES cosas, no una. Sin un control con la configuracion vieja
# ejecutado por este mismo harness, una diferencia frente al publicado podria ser
# del harness (guardado, ventana de promediado, orden de parametros) y no de la
# optimizacion. El control es lo que separa las dos explicaciones.
#
# Orden: primero las optimizadas, que son rapidas y dan senal temprana; el
# control va al final porque es el que tarda hora y media.
#
# guardado=50 para igualar al estudio publicado: la media se toma sobre el
# ultimo 20% de las muestras, asi que un guardado distinto cambia la ventana.
set -u
cd /home/ttandrei/Proyectos/Optimizador/Optimizador
PY=.venv/bin/python
B="scripts/agent_tests/bench_opt.py"
IT=26000

echo "=== t20: warm + maskmaj + 2x3  (la mas rapida, 3.45x) $(date +%H:%M:%S) ==="
$PY $B --run t20_warm_maskmaj_o2c3 --opt warm_start --opt coarse_mask_majority \
    --iters $IT --max-outer 2 --cycles 3 --set guardado=50

echo "=== t20: warm + maskmaj + 4x3  (la conservadora, 2.80x) $(date +%H:%M:%S) ==="
$PY $B --run t20_warm_maskmaj_o4c3 --opt warm_start --opt coarse_mask_majority \
    --iters $IT --max-outer 4 --cycles 3 --set guardado=50

echo "=== t20: CONTROL, configuracion actual sin tocar $(date +%H:%M:%S) ==="
$PY $B --run t20_control --iters $IT --set guardado=50

echo "=== VALIDACION t20 COMPLETA $(date +%H:%M:%S) ==="
$PY $B --compare t20_control t20_warm_maskmaj_o4c3 t20_warm_maskmaj_o2c3

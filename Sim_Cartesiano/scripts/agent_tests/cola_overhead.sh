#!/bin/bash
# Overhead por outer, rescatado del runner que se corto.
#
# Con 8 outers por paso cada uno de estos se paga ocho veces:
#   guard_residual_every_outer -> 2 laplacianos completos + 2 sincronizaciones
#   rollback_on_nan            -> copia entera de u y v + chequeo de NaN
#   compute_div_after          -> una divergencia extra al final
#
# El guard ademas resulto tener un coste oculto mucho mayor del que aparenta:
# cuando se dispara cae a `max(300, ...)` barridos del suavizador en el nivel
# fino, y eso con un suavizador caro multiplica el paso por seis sin que el
# contador de ciclos se entere. Quitarlo elimina esa red de seguridad, asi que
# solo vale si los resultados salen identicos.
set -u
cd /home/ttandrei/Proyectos/Optimizador/Optimizador
PY=.venv/bin/python
B="scripts/agent_tests/bench_opt.py"

while pgrep -f "cola_diag2.sh" > /dev/null; do sleep 30; done
echo "=== overhead por outer $(date +%H:%M:%S) ==="

$PY $B --run singuard --iters 4000 --set mg_guard_residual_every_outer=false
$PY $B --run sinrollback --iters 4000 --set mg_rollback_on_nan=false
$PY $B --run ligero --iters 4000 --set mg_guard_residual_every_outer=false \
    --set mg_rollback_on_nan=false --set mg_compute_div_after=false

echo "=== TODO COMPLETO $(date +%H:%M:%S) ==="
$PY $B --pareto
$PY $B --compare

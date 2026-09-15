#!/bin/bash
# Variantes del lote 1 que quedaron sin correr.
#
# El lote se corto en linesy_o4c3: iba a 1.06 it/s, o sea una hora de GPU para
# confirmar por segunda vez un fallo ya diagnosticado (las lineas solo en Y no
# suavizan la direccion X, el V-cycle diverge y salta el guard, que con un
# suavizador caro cuesta 300 barridos = 360 ms por paso). Se descarta esa
# variante y se recuperan aqui las cuatro siguientes, que si aportan:
#   maskmaj / masknone  aislan el efecto de la mascara gruesa a presupuesto
#                       completo, que es donde se ve limpio
#   todo_*              combinan lo que haya sobrevivido por separado
set -u
cd /home/ttandrei/Proyectos/Optimizador/Optimizador
PY=.venv/bin/python
B="scripts/agent_tests/bench_opt.py"

while pgrep -f "cola_final_noche.sh" > /dev/null; do sleep 30; done
echo "=== variantes pendientes del lote 1 $(date +%H:%M:%S) ==="

$PY $B --run maskmaj --opt coarse_mask_majority --iters 4000
$PY $B --run masknone --opt coarse_mask_none --iters 4000
$PY $B --run todo_o2c2 --opt line_smoother --opt coarse_mask_majority \
    --opt warm_start --iters 4000 --max-outer 2 --cycles 2 --pre 1 --post 1
$PY $B --run warm_maskmaj_o4c3 --opt warm_start --opt coarse_mask_majority \
    --iters 4000 --max-outer 4 --cycles 3
$PY $B --run warm_maskmaj_o2c3 --opt warm_start --opt coarse_mask_majority \
    --iters 4000 --max-outer 2 --cycles 3

echo "=== PENDIENTES COMPLETAS $(date +%H:%M:%S) ==="
$PY $B --pareto
$PY $B --compare

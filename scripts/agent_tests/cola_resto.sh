#!/bin/bash
# Todo lo que queda por medir, en un unico proceso secuencial.
#
# Antes esto eran tres scripts encadenados que se esperaban entre si con pgrep
# cada 30 s. Entre que uno acaba y el siguiente lo detecta hay una ventana en la
# que ninguno esta corriendo, y dos podian arrancar a la vez: dos simulaciones
# compartiendo la GPU no solo se pelean por la memoria, es que invalidan los
# cronometrajes, que son la mitad de lo que se esta midiendo.
set -u
cd /home/ttandrei/Proyectos/Optimizador/Optimizador
PY=.venv/bin/python
B="scripts/agent_tests/bench_opt.py"

while pgrep -f "bench_opt.py --queue" > /dev/null; do sleep 30; done
echo "=== lote 1 terminado $(date +%H:%M:%S) ==="

echo "=== reparto y estructura de la divergencia $(date +%H:%M:%S) ==="
$PY scripts/agent_tests/diag_divergencia.py --iters 400

echo "=== profundidad del multigrid $(date +%H:%M:%S) ==="
# El V-cycle tenia los niveles topados a 2 en el codigo. Con 1.26M celdas eso
# deja el nivel mas grueso en 78k, donde los modos de longitud de onda larga no
# se corrigen en ningun sitio: es la explicacion mas simple de un factor de
# convergencia de 0.52 en vez de 0.1-0.2. Ese tope estaba puesto porque el
# coarsening deformaba el solido (OR sobre 3x3) y porque el suavizador punto a
# punto no suaviza en malla anisotropa; corregidas las dos cosas, se comprueba
# si los niveles profundos ya suman.
$PY $B --run deep4 --opt deep_levels --iters 4000 --set mg_niveles_max=4
$PY $B --run deep4_maj --opt deep_levels --opt coarse_mask_majority --iters 4000 \
    --set mg_niveles_max=4
$PY $B --run deep4_maj_o4c3 --opt deep_levels --opt coarse_mask_majority \
    --iters 4000 --set mg_niveles_max=4 --max-outer 4 --cycles 3
$PY $B --run deep4_maj_o2c2 --opt deep_levels --opt coarse_mask_majority \
    --iters 4000 --set mg_niveles_max=4 --max-outer 2 --cycles 2 --pre 1 --post 1
$PY $B --run deep4_maj_lines_o2c2 --opt deep_levels --opt coarse_mask_majority \
    --opt line_smoother --iters 4000 --set mg_niveles_max=4 \
    --max-outer 2 --cycles 2 --pre 1 --post 1

echo "=== overhead por outer $(date +%H:%M:%S) ==="
# Con 8 outers por paso cada uno de estos se paga 8 veces: el guard son 2
# laplacianos completos mas 2 sincronizaciones, el rollback una copia entera de
# u y v mas un chequeo de NaN. Quitar el guard elimina ademas la red que detecta
# un V-cycle que diverge, asi que solo vale si los resultados salen identicos.
$PY $B --run singuard --iters 4000 --set mg_guard_residual_every_outer=false
$PY $B --run sinrollback --iters 4000 --set mg_rollback_on_nan=false
$PY $B --run ligero --iters 4000 --set mg_guard_residual_every_outer=false \
    --set mg_rollback_on_nan=false --set mg_compute_div_after=false

echo "=== TODO COMPLETO $(date +%H:%M:%S) ==="
$PY $B --compare

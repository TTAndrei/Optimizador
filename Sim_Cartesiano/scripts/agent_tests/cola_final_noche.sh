#!/bin/bash
# Ultimo tramo, reordenado por prioridad tras una medida nueva.
#
# Sobre las series ya volcadas, la diferencia entre 8x5 y 4x3 NO se estrecha con
# el tiempo fisico: dCd pasa de ~5% en el primer tercio a ~17% en el ultimo, y
# dCl de ~0% a -3.9%. La sensibilidad al presupuesto de proyeccion no es un
# artefacto del transitorio, que era la explicacion tranquilizadora.
#
# Eso asciende una pregunta por encima de la optimizacion: si el 8x5 con el que
# se calcularon las polares publicadas tampoco esta convergido, hay una
# incertidumbre numerica que no figura junto a la de malla en el estudio. Se mide
# primero; la jerarquia profunda va despues.
#
# Espera al PROCESO cola_resto.sh. Esperar por patron de comando deja una ventana
# entre que un comando acaba y arranca el siguiente, y dos simulaciones a la vez
# no solo se pelean por la GPU: invalidan los cronometrajes, que son la mitad de
# lo que se mide. Ya paso una vez esta noche.
set -u
cd /home/ttandrei/Proyectos/Optimizador/Optimizador
PY=.venv/bin/python
B="scripts/agent_tests/bench_opt.py"

while pgrep -f "cola_resto.sh" > /dev/null; do sleep 30; done
echo "=== ultimo tramo $(date +%H:%M:%S) ==="

# punto_o2c2 se midio mientras otro proceso compartia la GPU por un fallo de
# encadenado: su it/s esta deprimido y hay que repetirlo.
rm -f results/bench_opt/punto_o2c2.json results/bench_opt/series/punto_o2c2.npz
$PY $B --run punto_o2c2 --iters 4000 --max-outer 2 --cycles 2 --pre 1 --post 1

echo "=== esta convergido el baseline? $(date +%H:%M:%S) ==="
$PY $B --run punto_o12c6 --iters 4000 --max-outer 12 --cycles 6
$PY $B --run punto_o16c8 --iters 4000 --max-outer 16 --cycles 8

echo "=== jerarquia de 6 niveles $(date +%H:%M:%S) ==="
$PY $B --run deep6_maj --opt deep_levels --opt coarse_mask_majority \
    --iters 4000 --set mg_niveles_max=6
$PY $B --run deep6_maj_o4c3 --opt deep_levels --opt coarse_mask_majority \
    --iters 4000 --set mg_niveles_max=6 --max-outer 4 --cycles 3
$PY $B --run deep6_maj_o2c2 --opt deep_levels --opt coarse_mask_majority \
    --iters 4000 --set mg_niveles_max=6 --max-outer 2 --cycles 2 --pre 1 --post 1
$PY $B --run deep6_maj_o1c2 --opt deep_levels --opt coarse_mask_majority \
    --iters 4000 --set mg_niveles_max=6 --max-outer 1 --cycles 2 --pre 1 --post 1

echo "=== TODO COMPLETO $(date +%H:%M:%S) ==="
$PY $B --pareto
$PY $B --compare

#!/bin/bash
# Re-perfilado sobre la configuracion NUEVA.
#
# El perfil anterior decia que rb_gs_sor era el 10% del tiempo de GPU. Ese 10%
# es una fraccion de un total que ya no existe: fast_masks quito el 23% del
# indexado booleano e interp_float32 buena parte del 26% de float64. Al
# desaparecer esas partidas, el mismo trabajo absoluto del suavizador pasa a ser
# una fraccion mayor, y cuanto mayor no se sabe sin medirlo.
#
# De eso depende si Chebyshev merece la pena: sustituye el rojo-negro (2
# lanzamientos por barrido, la mitad de los hilos hace return inmediato) por un
# polinomico de 1 lanzamiento con todos los hilos trabajando. Si el suavizador ha
# pasado a pesar mucho, vale; si sigue marginal, no.
#
# Ya paso una vez en esta sesion: se descarto CUDA Graphs con una cuenta hecha
# sobre 346 lanzamientos por paso estimados, y nsys midio 3199.
set -u
cd /home/ttandrei/Proyectos/Optimizador/Optimizador
OUT=results/bench_opt/perfil
FLAGS=warm_start,coarse_mask_majority,fast_masks,interp_float32

while [ ! -f results/bench_opt/t20_full.json ]; do sleep 30; done
echo "=== t20_full terminado, re-perfilando $(date +%H:%M:%S) ==="

nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
    -o "$OUT/nuevo" .venv/bin/python scripts/agent_tests/perfilar_bucle.py 1000 "$FLAGS"

echo; echo "=== kernels, configuracion NUEVA ==="
nsys stats --report cuda_gpu_kern_sum --format table "$OUT/nuevo.nsys-rep" 2>/dev/null | grep -E "^\|" | head -16
echo; echo "=== API de CUDA, configuracion NUEVA ==="
nsys stats --report cuda_api_sum --format table "$OUT/nuevo.nsys-rep" 2>/dev/null | grep -E "^\|" | head -8
echo "=== REPERFILADO COMPLETO $(date +%H:%M:%S) ==="

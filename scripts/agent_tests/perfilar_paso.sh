#!/bin/bash
# Dónde se va el tiempo dentro de un paso, medido con Nsight Systems.
#
# POR QUE ESTO ANTES QUE LA FASE 5. La idea de la fase 5 era que el 41% de GPU
# ociosa fuese overhead de lanzamiento de kernels, y que CUDA Graphs lo
# recuperase. La cuenta no cuadra: con ~346 lanzamientos por paso y 26.4 ms de
# ocio salen 76 us de hueco por lanzamiento, y lanzar un kernel con CuPy cuesta
# 5-10 us. Sobra un orden de magnitud, asi que el ocio es otra cosa —lo mas
# probable, sincronizaciones con el host que vacian la cola— y CUDA Graphs no lo
# tocaria. Construirlo sin comprobar esto seria un paso en falso caro.
#
# Lo que hay que separar es:
#   tiempo de kernel        trabajo real en la GPU
#   cudaMemcpy / Synchronize  esperas del host, que vacian la tuberia
#   hueco sin nada           overhead de lanzamiento y trabajo de Python
#
# Solo si el tercero domina tiene sentido CUDA Graphs. Si domina el segundo, lo
# que hay que quitar son los `float(cp....)` del bucle. Y si domina el primero,
# no hay nada que rascar por esta via.
#
# Se perfilan 1000 iteraciones. Menos no vale: al terminar, main() ejecuta
# diagnosticos (auditoria de fuerzas de superficie, Cp, circulacion) que son un
# coste FIJO y con pocas iteraciones contaminarian el reparto. Con 1000 el bucle
# domina. Aun asi, al leer los resultados hay que recordar que la cola incluye
# esos diagnosticos y que sus kernels no son los del bucle.
# Las opciones van por --opt, NO por la variable OPT_SOLVER: run_variant empieza
# con opt_solver.reset() para aislar cada variante de la anterior, y eso borra lo
# que hubiera puesto el entorno. Se comprobo: con OPT_SOLVER puesto y sin --opt,
# este script habria perfilado la configuracion SIN optimizar. La variable de
# entorno si vale para los scripts de campana, que no resetean nada.
set -eu
cd /home/ttandrei/Proyectos/Optimizador/Optimizador
ITERS="${1:-1000}"
OUT=results/bench_opt/perfil
mkdir -p "$OUT"

echo "=== perfilando $ITERS iteraciones de la config ganadora ==="
nsys profile \
    --trace=cuda \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
    --output="$OUT/paso" \
    .venv/bin/python scripts/agent_tests/bench_opt.py \
        --run perfil_tmp --opt warm_start --opt coarse_mask_majority \
        --iters "$ITERS" --max-outer 2 --cycles 3 --set guardado=50

echo
echo "=== resumen de kernels (donde trabaja la GPU) ==="
nsys stats --report cuda_gpu_kern_sum --format table "$OUT/paso.nsys-rep" 2>/dev/null | head -25

echo
echo "=== resumen de la API de CUDA (donde espera el host) ==="
nsys stats --report cuda_api_sum --format table "$OUT/paso.nsys-rep" 2>/dev/null | head -25

echo
echo "=== transferencias de memoria (cada una es una sincronizacion) ==="
nsys stats --report cuda_gpu_mem_time_sum --format table "$OUT/paso.nsys-rep" 2>/dev/null | head -15

echo
echo "=== kernels + memops juntos, para el total de GPU ocupada ==="
nsys stats --report cuda_gpu_sum --format table "$OUT/paso.nsys-rep" 2>/dev/null | head -12

rm -f results/bench_opt/perfil_tmp.json results/bench_opt/series/perfil_tmp.npz
echo
echo "traza en $OUT/paso.nsys-rep"

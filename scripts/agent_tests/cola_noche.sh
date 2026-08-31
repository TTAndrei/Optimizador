#!/bin/bash
# Cola desatendida: espera a que acaben los tests de factor, lanza el
# diagnostico de reparto de divergencia y despues la cola de optimizaciones.
cd /home/ttandrei/Proyectos/Optimizador/Optimizador
PY=.venv/bin/python

while pgrep -f "bench_opt.py --run fac_" > /dev/null; do sleep 20; done
echo "=== factores terminados $(date +%H:%M:%S) ==="

echo "=== diagnostico de divergencia $(date +%H:%M:%S) ==="
$PY scripts/agent_tests/diag_divergencia.py --iters 400 2>&1 | tail -40

echo "=== cola de optimizaciones $(date +%H:%M:%S) ==="
$PY scripts/agent_tests/bench_opt.py --queue --iters 4000

echo "=== COLA COMPLETA $(date +%H:%M:%S) ==="

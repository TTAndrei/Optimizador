#!/usr/bin/env bash
# Escalera de malla del tubo a Re=1e3 (dx = 0.005 / 0.002 / 0.001) mas el
# llenado a Re=1e6 con dx=0.001. Mismo canal 16x0.2 en los cuatro, que es lo que
# hace comparable el refinado.
#
#   ./lanzar_tubo_fino.sh --fondo
#   touch PARAR_FORMAS.trigger      parada limpia (termina el caso en curso)
#
# Reanudable: cada caso escribe su tubo.json y se salta si ya existe.
set -u
cd "$(dirname "$0")"
if [ "${1:-}" = "--fondo" ]; then
    nohup "$0" --run >> results/tubo/runner_fino.log 2>&1 &
    echo "lanzado en segundo plano (pid $!)"
    exit 0
fi
PY=.venv/bin/python
for CASO in T7 T8 T10 T9; do
    echo "=== $CASO $(date +%H:%M:%S) ==="
    $PY -u scripts/agent_tests/tubo_canal.py --caso "$CASO"
done
echo "=== render $(date +%H:%M:%S) ==="
$PY -u scripts/agent_tests/render_videos.py --solo tubo
$PY -u scripts/agent_tests/formas_informe.py
echo "=== TODO COMPLETO $(date +%H:%M:%S) ==="

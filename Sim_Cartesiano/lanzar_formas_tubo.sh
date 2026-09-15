#!/usr/bin/env bash
# Estudio de formas romas + tubo. Un unico proceso secuencial: encadenar .sh con
# pgrep deja huecos donde dos simulaciones comparten GPU y arruinan los tiempos.
#
#   ./lanzar_formas_tubo.sh            primer plano
#   ./lanzar_formas_tubo.sh --fondo    nohup, log en results/formas/runner.log
#   touch PARAR_FORMAS.trigger         parada limpia (termina el punto en curso)
#
# Reanudable: relanzar continua donde lo dejo.
set -u
cd "$(dirname "$0")"
mkdir -p results/formas results/tubo

if [ "${1:-}" = "--fondo" ]; then
    nohup "$0" --run >> results/formas/runner.log 2>&1 &
    echo "lanzado en segundo plano (pid $!)"
    echo "  seguimiento: tail -f results/formas/runner.log"
    echo "  progreso:    cat results/formas/PROGRESO.txt"
    exit 0
fi

PY=.venv/bin/python
echo "=== FASE A: formas dx=0.004 $(date +%H:%M:%S) ==="
$PY -u scripts/agent_tests/formas_clcd.py --fase A --horas 1.0
echo "=== FASE B: tubo $(date +%H:%M:%S) ==="
$PY -u scripts/agent_tests/tubo_canal.py --horas 0.6
echo "=== FASE C: formas dx=0.002 $(date +%H:%M:%S) ==="
$PY -u scripts/agent_tests/formas_clcd.py --fase C --horas 1.8
echo "=== FASE D: videos e informes $(date +%H:%M:%S) ==="
$PY -u scripts/agent_tests/render_videos.py
$PY -u scripts/agent_tests/formas_informe.py
echo "=== TODO COMPLETO $(date +%H:%M:%S) ==="

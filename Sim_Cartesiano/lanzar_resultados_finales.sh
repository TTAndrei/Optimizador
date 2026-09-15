#!/bin/bash
# Estudio de resultados finales del TFG: Richardson de dos perfiles.
#
#   ./lanzar_resultados_finales.sh            en primer plano, con barra de tqdm
#   ./lanzar_resultados_finales.sh --fondo    desatendido, con log
#
# Ver el avance desde otra terminal:
#   .venv/bin/python scripts/agent_tests/rf_progreso.py
#
# Parar limpio entre puntos (no pierde el punto en curso hasta que termine):
#   touch PARAR_ESTUDIO.trigger
#
# Reanudar: volver a lanzar. Los puntos ya escritos se saltan.
set -eu
cd "$(dirname "$0")"
mkdir -p resultados_finales

if [ "${1:-}" = "--fondo" ]; then
    nohup .venv/bin/python scripts/agent_tests/resultados_finales.py \
        >> resultados_finales/runner.log 2>&1 &
    echo "lanzado en segundo plano (pid $!)"
    echo "log:      tail -f resultados_finales/runner.log"
    echo "progreso: .venv/bin/python scripts/agent_tests/rf_progreso.py"
else
    exec .venv/bin/python scripts/agent_tests/resultados_finales.py
fi

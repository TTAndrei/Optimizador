#!/usr/bin/env bash
# Encadena lo que queda tras la polar: repolar (si sigue) -> validacion XFOIL -> revalidacion del ranking.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
VER=results/verificacion_numerica

while pgrep -f "verificacion_numerica.py --repolar" >/dev/null; do sleep 30; done
echo ">>> repolar terminado"

$PY -u scripts/agent_tests/verificacion_numerica.py --validacion --re 1e5 2>&1 | tee -a "$VER/validacion.log"
echo ">>> validacion XFOIL terminada"

$PY -u scripts/agent_tests/revalidar_ranking.py --n 20 2>&1 | tee -a "$VER/revalidacion.log"
echo ">>> revalidacion del ranking terminada"

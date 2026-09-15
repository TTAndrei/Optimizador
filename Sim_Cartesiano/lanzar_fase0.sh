#!/usr/bin/env bash
# Fase 0: equivalencia bit a bit del refactor + estudio de coste de la salida.
# Todo secuencial: comparten GPU y solaparlos falsea el cronometraje.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
REF="${1:-$(pwd)/.ref_head/Simulador2D.py}"
LOG=results/coste_salida/fase0.log
mkdir -p results/coste_salida

{
  echo "===== $(date '+%F %T')  equivalencia bit a bit ====="
  $PY scripts/agent_tests/equivalencia_bit.py --ref "$REF" --iters 150 --dx 0.004
  EQ=$?
  echo "equivalencia: codigo $EQ"

  echo "===== $(date '+%F %T')  coste de salida, dx=0.004 ====="
  $PY scripts/agent_tests/coste_salida.py --todo --dx 0.004

  echo "===== $(date '+%F %T')  coste de salida, dx=0.002 ====="
  $PY scripts/agent_tests/coste_salida.py --todo --dx 0.002

  echo "===== $(date '+%F %T')  informe ====="
  $PY scripts/agent_tests/coste_salida.py --informe
  echo "===== $(date '+%F %T')  FIN ====="
} 2>&1 | tee -a "$LOG"

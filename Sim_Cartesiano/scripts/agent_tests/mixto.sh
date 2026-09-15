#!/usr/bin/env bash
# Estudios de población mixta — progreso en vivo, PAUSABLES y REANUDABLES.
#
#   ./scripts/agent_tests/mixto.sh masgen       -> 4 semillas clásicas, MÁS generaciones (pop 16, gen 40)
#   ./scripts/agent_tests/mixto.sh masperfiles  -> 8 semillas (4 + airfoiltools), MENOS generaciones (pop 24, gen 12)
#   ./scripts/agent_tests/mixto.sh status       -> estado de ambos estudios
#
# PAUSAR (liberar GPU/RAM):  Ctrl+C una vez. Termina la evaluación en curso,
#   guarda checkpoint (estado_ga.json) y sale SIN marcar la corrida completada.
# REANUDAR:  vuelve a lanzar el mismo comando; continúa desde la última
#   generación completada.
#
# Ajustar tope de horas:  DEADLINE=30 ./scripts/agent_tests/mixto.sh masgen
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

PY=.venv/bin/python
BASE=results/convergence_study

estado_de() {
  local d="$BASE/mixto_$1"
  if [[ -f "$d/estado_ga_final.json" ]]; then
    ld=$($PY -c "import json;print('%.3f'%json.load(open('$d/estado_ga_final.json'))['mejor_global']['fitness'])" 2>/dev/null || echo '?')
    printf "  %-12s  COMPLETADO   (L/D %s)\n" "$1" "$ld"
  elif [[ -f "$d/estado_ga.json" ]]; then
    g=$($PY -c "import json;print(json.load(open('$d/estado_ga.json'))['generacion_actual']+1)" 2>/dev/null || echo '?')
    ld=$($PY -c "import json;print('%.3f'%json.load(open('$d/estado_ga.json'))['mejor_global']['fitness'])" 2>/dev/null || echo '?')
    printf "  %-12s  en curso     (gen %s, mejor L/D %s)\n" "$1" "$g" "$ld"
  else
    printf "  %-12s  pendiente\n" "$1"
  fi
}

status() {
  echo "== Estudios mixtos — estado =="
  estado_de masgen
  estado_de masperfiles
}

lanzar() {
  local tag=$1; shift
  local LOG="$BASE/mixto_${tag}_run.log"
  echo "Progreso en vivo abajo. Log: $LOG"
  echo "PAUSAR (liberar GPU): Ctrl+C una vez  |  REANUDAR: relanza este comando"
  echo
  status
  echo
  mkdir -p "$BASE"
  $PY -u scripts/agent_tests/run_convergence_study.py --mixto "$tag" "$@" \
    --deadline-hours "$DEADLINE" 2>&1 | tee -a "$LOG"
  echo
  status
}

case "${1:-}" in
  masgen)
    DEADLINE="${DEADLINE:-24}"
    lanzar masgen --mixto-seeds 4 --mixto-pop 16 --mixto-gen-max 40 --mixto-paciencia 8 \
      --transition-model sa_bc --freestream-tu 0.1
    ;;
  masperfiles)
    DEADLINE="${DEADLINE:-14}"
    lanzar masperfiles --mixto-seeds 8 --mixto-pop 24 --mixto-gen-max 12 --mixto-paciencia 5 \
      --transition-model sa_bc --freestream-tu 0.1
    ;;
  status)
    status
    ;;
  *)
    echo "uso: $0 {masgen|masperfiles|status}"
    exit 1
    ;;
esac

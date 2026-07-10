#!/usr/bin/env bash
# Estudio aislado (sin migración) — progreso en vivo, PAUSABLE y REANUDABLE.
#
#   ./scripts/agent_tests/aislado.sh          -> corre en primer plano (progreso en vivo)
#   ./scripts/agent_tests/aislado.sh status   -> muestra estado por semilla
#
# PAUSAR (liberar GPU/RAM):  Ctrl+C una vez. Termina la evaluación en curso,
#   guarda checkpoint y sale SIN marcar la semilla como completada.
# REANUDAR:  vuelve a lanzar el mismo comando. Salta semillas ya convergidas y
#   continúa la semilla en curso desde su último checkpoint (gen completada).
#
# Ajustar tope de horas:  DEADLINE=60 ./scripts/agent_tests/aislado.sh
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

PY=.venv/bin/python
OUT=results/convergence_study/aislado_sin_migracion_convergido
LOG=results/convergence_study/aislado_run.log
SEEDS=(NACA_0012_sharp AG24 GM15 s1014)
DEADLINE="${DEADLINE:-40}"
ARGS=(--aislado --aislado-pop 12 --aislado-gen-max 30 --aislado-paciencia 6
      --deadline-hours "$DEADLINE")

status() {
  echo "== Aislado sin migración — estado =="
  for s in "${SEEDS[@]}"; do
    d="$OUT/$s"
    if [[ -f "$d/estado_ga_final.json" ]]; then
      ld=$($PY -c "import json;print('%.3f'%json.load(open('$d/estado_ga_final.json'))['mejor_global']['fitness'])" 2>/dev/null || echo '?')
      printf "  %-18s  CONVERGIDA   (L/D %s)\n" "$s" "$ld"
    elif [[ -f "$d/estado_ga.json" ]]; then
      g=$($PY -c "import json;print(json.load(open('$d/estado_ga.json'))['generacion_actual']+1)" 2>/dev/null || echo '?')
      ld=$($PY -c "import json;print('%.3f'%json.load(open('$d/estado_ga.json'))['mejor_global']['fitness'])" 2>/dev/null || echo '?')
      printf "  %-18s  en curso     (gen %s, mejor L/D %s)\n" "$s" "$g" "$ld"
    else
      printf "  %-18s  pendiente\n" "$s"
    fi
  done
}

case "${1:-run}" in
  status)
    status
    ;;
  run)
    echo "Progreso en vivo abajo. Log: $LOG"
    echo "PAUSAR (liberar GPU): Ctrl+C una vez  |  REANUDAR: relanza este comando"
    echo "Estado sin lanzar: ./scripts/agent_tests/aislado.sh status"
    echo
    status
    echo
    mkdir -p "$(dirname "$LOG")"
    $PY -u scripts/agent_tests/run_convergence_study.py "${ARGS[@]}" 2>&1 | tee -a "$LOG"
    echo
    status
    ;;
  *)
    echo "uso: $0 [run|status]"
    exit 1
    ;;
esac

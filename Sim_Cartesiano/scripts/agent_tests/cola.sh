#!/usr/bin/env bash
# Cola secuencial de estudios, pensada para dejar corriendo varios días sin
# supervisión (GPU única -> todo en serie). Cada estudio es un --mixto normal
# de run_convergence_study.py, checkpointeable con estado_ga.json.
#
#   ./scripts/agent_tests/cola.sh run          -> corre toda la cola en orden
#   ./scripts/agent_tests/cola.sh run exprimir_re1e5   -> corre solo esa etapa
#   ./scripts/agent_tests/cola.sh status       -> estado de todas las etapas
#
# PAUSAR (liberar GPU): Ctrl+C una vez. Guarda checkpoint y sale.
# REANUDAR: relanza `run` (mismo comando); salta etapas ya COMPLETADAS y
#   reanuda la que quedó a medias desde su último estado_ga.json.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

PY=.venv/bin/python
BASE=results/convergence_study
SCRIPT=scripts/agent_tests/run_convergence_study.py
WINNER_MASGEN="$BASE/mixto_masgen/NACA_0012_sharp_Re100000_a4.0_LD18.18.dat"
SEEDS8="profiles/NACA_0012_sharp,profiles/AG24,profiles/GM15,profiles/s1014.dat,profiles/SD7037.dat,profiles/E387.dat,profiles/SG6043.dat,profiles/MH32.dat"

# tag  re      seeds                    pop  gen  paciencia  deadline_h
ETAPAS=(
  "exprimir_re1e5|1e5|${WINNER_MASGEN},${SEEDS8}|24|60|12|40"
  "re1e3|1e3|${SEEDS8}|16|30|8|36"
  "re1e6|1e6|${SEEDS8}|16|30|8|36"
  "re1e7|1e7|${SEEDS8}|16|30|8|36"
)

dir_de() { echo "$BASE/mixto_$1"; }

estado_de() {
  local d; d=$(dir_de "$1")
  if [[ -f "$d/estado_ga_final.json" ]]; then
    ld=$($PY -c "import json;print('%.3f'%json.load(open('$d/estado_ga_final.json'))['mejor_global']['fitness'])" 2>/dev/null || echo '?')
    printf "  %-16s  COMPLETADO   (L/D %s)\n" "$1" "$ld"
  elif [[ -f "$d/estado_ga.json" ]]; then
    g=$($PY -c "import json;print(json.load(open('$d/estado_ga.json'))['generacion_actual']+1)" 2>/dev/null || echo '?')
    ld=$($PY -c "import json;print('%.3f'%json.load(open('$d/estado_ga.json'))['mejor_global']['fitness'])" 2>/dev/null || echo '?')
    printf "  %-16s  en curso     (gen %s, mejor L/D %s)\n" "$1" "$g" "$ld"
  else
    printf "  %-16s  pendiente\n" "$1"
  fi
}

status() {
  echo "== Cola de estudios — estado =="
  for etapa in "${ETAPAS[@]}"; do
    IFS='|' read -r tag _ <<< "$etapa"
    estado_de "$tag"
  done
}

lanzar_etapa() {
  IFS='|' read -r tag re seeds pop gen paciencia deadline <<< "$1"
  local d; d=$(dir_de "$tag")
  if [[ -f "$d/estado_ga_final.json" ]]; then
    echo "[skip] $tag ya completado"
    return
  fi
  local LOG="$BASE/mixto_${tag}_run.log"
  echo ">>> Etapa: $tag  (Re=$re, deadline ${deadline}h)"
  echo "Log: $LOG  |  Ctrl+C para pausar, relanzar 'run' para reanudar"
  $PY -u "$SCRIPT" --mixto "$tag" --re "$re" \
    --mixto-seeds "$seeds" --mixto-pop "$pop" --mixto-gen-max "$gen" \
    --mixto-paciencia "$paciencia" --transition-model sa_bc --freestream-tu 0.1 \
    --deadline-hours "$deadline" 2>&1 | tee -a "$LOG"
}

case "${1:-}" in
  run)
    if [[ -n "${2:-}" ]]; then
      for etapa in "${ETAPAS[@]}"; do
        IFS='|' read -r tag _ <<< "$etapa"
        [[ "$tag" == "$2" ]] && lanzar_etapa "$etapa"
      done
    else
      for etapa in "${ETAPAS[@]}"; do
        lanzar_etapa "$etapa"
      done
    fi
    echo
    status
    ;;
  status)
    status
    ;;
  *)
    echo "uso: $0 {run [etapa]|status}"
    echo "etapas: exprimir_re1e5, re1e3, re1e6, re1e7"
    exit 1
    ;;
esac

#!/usr/bin/env bash
# Cola de resultados para el TFG. GPU única -> todo en serie.
#
#   ./scripts/agent_tests/cola_tfg.sh run       -> ejecuta todas las etapas pendientes
#   ./scripts/agent_tests/cola_tfg.sh run gci   -> solo esa etapa
#   ./scripts/agent_tests/cola_tfg.sh status
#
# Etapas (en orden):
#   calibrar    series largas sin early-stop + barrido offline del criterio de parada
#   reeval      re-evaluación de los 4 ganadores con la estadística corregida
#   gci         convergencia de malla dx=0.008/0.004/0.002/0.001 (GCI de Roache)
#   polar       polar alpha=0..10 del ganador y de la semilla NACA0012
#   multipunto  GA optimizando sobre alpha={2,4,6}. FUERA de la cola por defecto:
#               con el criterio de parada recalibrado una evaluación cuesta ~17 min,
#               así que 3 angulos x 12 individuos = ~10 h por generación y no cabe en
#               un deadline razonable. Lanzarlo explícitamente si se decide asumirlo.
#
# PAUSAR: Ctrl+C. reeval/gci/polar cachean cada simulación en su JSON; el GA
# guarda estado_ga.json. Relanzar `run` reanuda donde quedó.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

PY=.venv/bin/python
VER=results/verificacion_numerica
BASE=results/convergence_study
WINNER="$BASE/mixto_masgen/NACA_0012_sharp_Re100000_a4.0_LD18.18.dat"
SEEDS8="profiles/NACA_0012_sharp,profiles/AG24,profiles/GM15,profiles/s1014.dat,profiles/SD7037.dat,profiles/E387.dat,profiles/SG6043.dat,profiles/MH32.dat"

etapa_calibrar() {
  $PY -u scripts/agent_tests/verificacion_numerica.py --calibrar 2>&1 | tee -a "$VER/calibrar.log"
}

etapa_reeval() {
  $PY -u scripts/agent_tests/verificacion_numerica.py --reeval 2>&1 | tee -a "$VER/reeval.log"
}

etapa_gci() {
  $PY -u scripts/agent_tests/verificacion_numerica.py --gci --re 1e5 2>&1 | tee -a "$VER/gci.log"
}

etapa_polar() {
  $PY -u scripts/agent_tests/verificacion_numerica.py --polar --re 1e5 2>&1 | tee -a "$VER/polar.log"
}

etapa_multipunto() {
  $PY -u scripts/agent_tests/run_convergence_study.py \
    --mixto multipunto_re1e5 --re 1e5 \
    --multipunto --delta-angulo 2.0 --fitness-modo mean \
    --mixto-seeds "$WINNER,$SEEDS8" --mixto-pop 16 --mixto-gen-max 25 \
    --mixto-paciencia 8 --transition-model sa_bc --freestream-tu 0.1 \
    --deadline-hours 60 2>&1 | tee -a "$BASE/mixto_multipunto_re1e5_run.log"
}

hecho() {
  case "$1" in
    calibrar)   [[ -f "$VER/calibracion_earlystop.json" ]] ;;
    reeval)     [[ -f "$VER/reeval_ganadores.json" ]] ;;
    gci)        [[ -f "$VER/gci.json" ]] ;;
    polar)      [[ -f "$VER/polar.png" ]] ;;
    multipunto) [[ -f "$BASE/mixto_multipunto_re1e5/estado_ga_final.json" ]] ;;
  esac
}

status() {
  echo "== Cola TFG =="
  for e in calibrar reeval gci polar multipunto; do
    if hecho "$e"; then printf "  %-12s COMPLETADO\n" "$e"
    else printf "  %-12s pendiente/en curso\n" "$e"; fi
  done
}

case "${1:-}" in
  run)
    mkdir -p "$VER"
    for e in ${2:-calibrar reeval gci polar}; do
      if hecho "$e"; then echo "[skip] $e ya completado"; continue; fi
      echo ">>> Etapa: $e"
      "etapa_$e"
    done
    echo; status
    ;;
  status) status ;;
  *) echo "uso: $0 {run [etapa]|status}"; echo "etapas: calibrar reeval gci polar multipunto"; exit 1 ;;
esac

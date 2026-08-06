#!/usr/bin/env bash
# Segunda gran optimización — GA de islas con el criterio de parada validado.
#
#   ./scripts/agent_tests/gran_optimizacion.sh start    -> lanza en segundo plano
#   ./scripts/agent_tests/gran_optimizacion.sh run      -> lanza en primer plano (Ctrl+C pausa)
#   ./scripts/agent_tests/gran_optimizacion.sh stop     -> pausa limpia y espera al checkpoint
#   ./scripts/agent_tests/gran_optimizacion.sh status   -> progreso por isla y época
#   ./scripts/agent_tests/gran_optimizacion.sh log      -> tail -f del log
#
# PAUSAR/REANUDAR: `stop` crea el fichero centinela STOP; la corrida termina la
# evaluación CFD en curso (<=12 min), guarda estado_ga.json y sale. `start` de
# nuevo continúa exactamente donde quedó: las épocas cerradas se saltan y los
# individuos ya simulados de la generación a medias no se vuelven a pagar.
#
# Presupuesto: 4 islas x pop 16 x 2 gen/época x 13 épocas = 1456 evaluaciones
# (16 la primera generación de cada época + 12 la segunda, con 4 élites).
# A ~700 s por evaluación a dx=0.004 son ~283 h, + ~2 h de calibración y ~3 h de
# refinado a dx=0.002: ~288 h dentro del tope de 311 h. Se dejan 13 épocas y no 14
# para que el deadline no trunque una época a medias.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

PY=.venv/bin/python
BASE=results/convergence_study
TAG=tfg2
ISLAS="$BASE/islands_$TAG"
STOP="$BASE/STOP"
LOG="$BASE/gran_optimizacion_$TAG.log"
PIDF="$BASE/gran_optimizacion_$TAG.pid"

EPOCAS=13
MIGRA=2
POP=16
DX=0.004
TTARGET=12
HORAS="${HORAS:-311}"

ARGS=(
  --islands --run-tag "$TAG"
  --island-epochs "$EPOCAS" --island-migrate-every "$MIGRA" --island-pop "$POP"
  --dx "$DX" --t-target "$TTARGET"
  --transition-model sa_bc --freestream-tu 0.1
  --re 1e5 --refine-k 3
  --deadline-hours "$HORAS"
  --stop-file "$STOP"
)

vivo() { [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

status() {
  echo "== Gran optimización ($TAG) =="
  if vivo; then echo "  estado: EN CURSO (pid $(cat "$PIDF"))"
  else echo "  estado: parada"; fi
  echo "  presupuesto: $EPOCAS épocas x 4 islas, pop $POP, dx $DX, tope ${HORAS}h"
  [[ -d "$ISLAS" ]] || { echo "  (sin épocas todavía)"; return; }
  for isla in "$ISLAS"/*/; do
    nm=$(basename "$isla")
    [[ -d "$isla" ]] || continue
    hechas=$(find "$isla" -maxdepth 2 -name estado_ga_final.json 2>/dev/null | wc -l)
    ult=$(ls -d "$isla"epoch* 2>/dev/null | sort -V | tail -1)
    ld='-'
    if [[ -n "$ult" && -f "$ult/estado_ga.json" ]]; then
      ld=$($PY -c "import json;d=json.load(open('$ult/estado_ga.json'));print('%.3f'%d['mejor_global']['fitness'])" 2>/dev/null || echo '?')
    fi
    printf "  %-20s %2s/%s épocas   mejor L/D %s\n" "$nm" "$hechas" "$EPOCAS" "$ld"
  done
  [[ -f "$ISLAS/refine_dx002.json" ]] && echo "  refinado dx=0.002: COMPLETADO"
}

case "${1:-}" in
  start)
    vivo && { echo "Ya está corriendo (pid $(cat "$PIDF"))."; exit 1; }
    mkdir -p "$BASE"; rm -f "$STOP"
    nohup $PY -u scripts/agent_tests/run_convergence_study.py "${ARGS[@]}" >>"$LOG" 2>&1 &
    echo $! >"$PIDF"
    echo "Lanzada (pid $(cat "$PIDF")). Log: $LOG"
    echo "Pausar: $0 stop   |   Ver: $0 log"
    ;;
  run)
    mkdir -p "$BASE"; rm -f "$STOP"
    $PY -u scripts/agent_tests/run_convergence_study.py "${ARGS[@]}" 2>&1 | tee -a "$LOG"
    ;;
  stop)
    vivo || { echo "No hay ninguna corrida activa."; exit 0; }
    touch "$STOP"
    echo "STOP creado. Esperando a que cierre la evaluación en curso (<=12 min)..."
    while vivo; do sleep 10; done
    rm -f "$STOP" "$PIDF"
    echo "Parada limpia. Reanudar con: $0 start"
    status
    ;;
  snapshot)
    # Copia consistente sin parar la corrida: solo las épocas cerradas
    # (estado_ga_final.json) son inmutables; la que está en curso se copia igual
    # y, si sale truncada por el corte, se descarta y se repite esa época.
    DEST="$BASE/backups/${TAG}_$(date +%Y%m%d_%H%M)"
    mkdir -p "$DEST"
    cp -a "$ISLAS" "$BASE/calibration.json" "$LOG" "$DEST"/ 2>/dev/null
    echo "Copia en $DEST ($(du -sh "$DEST" | cut -f1))"
    ;;
  status) status ;;
  log)    tail -f "$LOG" ;;
  *) echo "uso: $0 {start|run|stop|status|log|snapshot}"; exit 1 ;;
esac

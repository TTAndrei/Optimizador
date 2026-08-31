#!/bin/bash
# Segundo intento del diagnostico de divergencia.
#
# El primero dio la parte espacial —que ya contesta la pregunta: el 63% de la
# divergencia residual esta en el interior lejano, repartida por el 98% de las
# celdas, no concentrada en las celdas que la proyeccion no corrige por diseno—
# y murio despues por un bloque huerfano al refactorizar el test de operadores.
# Faltan por medir las dos partes que quedaron detras del fallo:
#   - compatibilidad L frente a D(G(p)) sobre un campo SUAVE, que es la medida
#     honesta (con ruido blanco el resultado sale enorme por construccion,
#     porque la composicion de dos derivadas centradas se anula en Nyquist)
#   - cuanto de la divergencia residual es modo tablero, que es la firma del
#     desacoplamiento par-impar de malla colocada
#
# Espera por el NOMBRE DEL SCRIPT anterior, no por el comando que ejecuta:
# esperar por el comando hace que el runner se encuentre reflejado en cualquier
# bucle de vigilancia que contenga esa misma cadena, y se queda parado para
# siempre. Paso esta noche.
set -u
cd /home/ttandrei/Proyectos/Optimizador/Optimizador

while pgrep -f "cola_pendientes.sh" > /dev/null; do sleep 30; done
echo "=== diagnostico de divergencia, segundo intento $(date +%H:%M:%S) ==="
.venv/bin/python scripts/agent_tests/diag_divergencia.py --iters 400
echo "=== DIAGNOSTICO COMPLETO $(date +%H:%M:%S) ==="

#!/usr/bin/env bash
# Interfaz del simulador. La GUI no toca la GPU: lanza el solver como proceso
# aparte y lo observa por memoria compartida.
cd "$(dirname "$0")"
exec "$(cd .. && pwd)/.venv/bin/python" -m gui "$@"

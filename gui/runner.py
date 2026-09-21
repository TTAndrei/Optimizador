"""Puente Qt entre la ventana y el proceso del solver curvilineo."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from PyQt6 import QtCore

from .caso import Caso
from .run_solver import CENTINELA
from .snapshot import Snapshot, SnapshotReader


class _Lector(QtCore.QThread):
    linea = QtCore.pyqtSignal(str)

    def __init__(self, proceso):
        super().__init__()
        self.proceso = proceso

    def run(self):
        for linea in iter(self.proceso.stdout.readline, ""):
            self.linea.emit(linea.rstrip("\n"))


class Runner(QtCore.QObject):
    log = QtCore.pyqtSignal(str)
    progreso = QtCore.pyqtSignal(dict)
    campo = QtCore.pyqtSignal(object)
    terminado = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proceso = None
        self.salida = None
        self._lector = None
        self._reader = None
        self._final_emitido = False
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._leer)

    def lanzar(self, caso: Caso, salida: str | Path) -> None:
        if self.activo():
            raise RuntimeError("ya hay una simulacion corriendo")
        self.salida = Path(salida).resolve()
        self._final_emitido = False
        self.salida.mkdir(parents=True, exist_ok=True)
        caso.ejecucion["directorio"] = str(self.salida)
        ruta = self.salida / "caso.json"
        caso.guardar(ruta)
        self.proceso = subprocess.Popen(
            [sys.executable, "-u", "-m", "gui.run_solver", str(ruta)],
            cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        self._lector = _Lector(self.proceso)
        self._lector.linea.connect(self.log)
        self._lector.start()
        self._timer.start(150)
        self.log.emit(f"[GUI] corrida: {self.salida}")

    def parar(self) -> None:
        if self.activo() and self.salida is not None:
            (self.salida / CENTINELA).touch()
            self.log.emit("[GUI] parada solicitada")

    def matar(self) -> None:
        if self.activo():
            self.proceso.terminate()

    def activo(self) -> bool:
        return self.proceso is not None and self.proceso.poll() is None

    def _leer(self) -> None:
        terminado = self.proceso is not None and self.proceso.poll() is not None
        if self._reader is None and self.salida is not None:
            ruta = self.salida / "ipc.json"
            if ruta.is_file():
                try:
                    nombres = tuple(json.loads(ruta.read_text())["names"])
                    self._reader = SnapshotReader(nombres)
                except (FileNotFoundError, KeyError, ValueError):
                    if not terminado:
                        return
        snapshot = self._reader.read() if self._reader is not None else None
        if snapshot is not None:
            self.campo.emit(snapshot)
            self.progreso.emit(snapshot.metrics)
        elif terminado and not self._final_emitido:
            self._emitir_campo_final()
        if terminado:
            self._timer.stop()
            self._cerrar_reader()
            self.terminado.emit(self.proceso.returncode)
            self.proceso = None

    def _emitir_campo_final(self) -> None:
        """Recupera el ultimo campo si el proceso termino antes de ser leido."""
        if self.salida is None:
            return
        archivos = sorted((self.salida / "campos").glob("campo_*.npz"))
        malla = self.salida / "malla.npz"
        if not archivos or not malla.is_file():
            return
        with np.load(malla) as geometria, np.load(archivos[-1]) as campo:
            datos = {"iter": int(archivos[-1].stem.split("_")[-1]),
                     "t": float(campo["t"])}
            historia = self.salida / "historia.npz"
            if historia.is_file():
                with np.load(historia) as serie:
                    if len(serie["datos"]):
                        columnas = list(serie["columnas"])
                        ultima = serie["datos"][-1]
                        datos.update({str(nombre): float(ultima[i])
                                      for i, nombre in enumerate(columnas)})
            snapshot = Snapshot(0, geometria["X"], geometria["Y"], campo["u"],
                                campo["v"], campo["p"], datos)
        self._final_emitido = True
        self.campo.emit(snapshot)
        self.progreso.emit(snapshot.metrics)

    def _cerrar_reader(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def close(self) -> None:
        self._timer.stop()
        self._cerrar_reader()
        if self._lector is not None:
            self._lector.wait(1000)
            self._lector = None

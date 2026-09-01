"""Lanza el solver como proceso aparte y lo observa por memoria compartida.

La GUI nunca toca la GPU. Si el solver revienta, la ventana sigue viva y el log
queda en pantalla; y una figura de matplotlib colgada en la GUI no puede
bloquear la simulacion.

El canal ya existe en el solver: `sim2d_meta` (40 bytes de progreso),
`sim2d_live` (|u|, mascara y opcionalmente vorticidad) y `sim2d_coords` (los dos
ejes fisicos, que hacen falta porque la malla es estirada y no se puede pintar
como una imagen uniforme).
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
from multiprocessing import shared_memory

import numpy as np
from PyQt6 import QtCore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(ROOT, ".venv", "bin", "python")
META, DATOS, COORDS = "sim2d_meta", "sim2d_live", "sim2d_coords"
# El solver ya mira este centinela en su bucle y para limpiamente.
CENTINELA = "STOP_SIMULATION.trigger"


class LectorLog(QtCore.QThread):
    linea = QtCore.pyqtSignal(str)

    def __init__(self, proc):
        super().__init__()
        self.proc = proc

    def run(self):
        for ln in iter(self.proc.stdout.readline, ""):
            self.linea.emit(ln.rstrip("\n"))


class Runner(QtCore.QObject):
    log = QtCore.pyqtSignal(str)
    progreso = QtCore.pyqtSignal(dict)          # iter, cl, cd, alpha
    campo = QtCore.pyqtSignal(object, object, object, object)  # speed, solid, x, y
    terminado = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc = None
        self.dir = None
        self._lector = None
        self._shm = {}
        self._forma = None
        self._t = QtCore.QTimer(self)
        self._t.timeout.connect(self._leer)

    # ------------------------------------------------------------------
    def lanzar(self, escena, carpeta=None):
        if self.activo():
            raise RuntimeError("ya hay una simulación corriendo")
        self.dir = carpeta or tempfile.mkdtemp(prefix="sim_gui_")
        os.makedirs(self.dir, exist_ok=True)
        ruta = os.path.join(self.dir, "escena.json")
        escena.guardar(ruta)
        self._quitar_centinela()
        self._tirar_shm_huerfana()

        self.proc = subprocess.Popen(
            [PYTHON, "-u", "-m", "gui.run_solver", ruta],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        self._lector = LectorLog(self.proc)
        self._lector.linea.connect(self.log)
        self._lector.start()
        self._t.start(150)
        self.log.emit(f"[GUI] lanzado en {self.dir}")

    def parar(self):
        """Centinela primero: deja que el solver cierre y guarde. El terminate
        es la red de seguridad, no el mecanismo."""
        if not self.activo():
            return
        open(os.path.join(self.dir, CENTINELA), "w").close()
        self.log.emit("[GUI] parada pedida (centinela)")

    def matar(self):
        if self.activo():
            self.proc.terminate()

    def activo(self):
        return self.proc is not None and self.proc.poll() is None

    # ------------------------------------------------------------------
    def _leer(self):
        if self.proc is not None and self.proc.poll() is not None:
            self._t.stop()
            self._cerrar_shm()
            self._quitar_centinela()
            self.terminado.emit(self.proc.returncode)
            self.proc = None
            return
        if not self._abrir_shm():
            return
        b = self._shm[META].buf
        ny, nx, it = struct.unpack_from("iii", b, 0)
        cd, cl, alpha = struct.unpack_from("fff", b, 12)
        self.progreso.emit({"iter": it, "cl": cl, "cd": cd, "alpha": alpha})

        n = ny * nx
        db = self._shm[DATOS].buf
        speed = np.frombuffer(db, dtype=np.float32, count=n).reshape(ny, nx).copy()
        solid = np.frombuffer(db, dtype=np.uint8, count=n,
                              offset=n * 4).reshape(ny, nx).astype(bool)
        cb = self._shm[COORDS].buf
        x = np.frombuffer(cb, dtype=np.float32, count=nx).copy()
        y = np.frombuffer(cb, dtype=np.float32, count=ny,
                          offset=nx * 4).copy()
        self.campo.emit(speed, solid, x, y)

    def _abrir_shm(self):
        """Solo acepta los bloques de NUESTRO proceso.

        La memoria compartida POSIX tiene nombre fijo y sobrevive al proceso que
        la creo, asi que un bloque huerfano de una corrida anterior —un test
        interrumpido, un blowup— se abre igual de bien y se pinta igual de bien,
        con la malla y el dominio de la OTRA simulacion. El solver publica su pid
        al final de la cabecera; sin coincidencia, no se lee nada.
        """
        if self._shm:
            return True
        if self.proc is None:
            return False
        try:
            for nombre in (META, DATOS, COORDS):
                self._shm[nombre] = shared_memory.SharedMemory(name=nombre)
        except FileNotFoundError:
            self._cerrar_shm()
            return False
        if self._shm[META].size < 44:
            self._cerrar_shm()               # cabecera vieja, sin pid
            return False
        pid, = struct.unpack_from("i", self._shm[META].buf, 40)
        if pid != self.proc.pid:
            self._cerrar_shm()
            return False
        return True

    def _tirar_shm_huerfana(self):
        """Borra los bloques que haya dejado una corrida anterior. Sin esto, el
        pid nunca coincide y no se ve nada; con esto, ademas, no se acumulan."""
        for nombre in (META, DATOS, COORDS):
            try:
                v = shared_memory.SharedMemory(name=nombre)
                v.close()
                v.unlink()
                self.log.emit(f"[GUI] tirado bloque huérfano {nombre}")
            except FileNotFoundError:
                pass

    def _cerrar_shm(self):
        for s in self._shm.values():
            try:
                s.close()
            except Exception:
                pass
        self._shm.clear()

    def _quitar_centinela(self):
        if not self.dir:
            return
        try:
            os.remove(os.path.join(self.dir, CENTINELA))
        except FileNotFoundError:
            pass

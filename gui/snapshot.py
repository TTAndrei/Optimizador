"""Snapshot IPC versionado para la visualizacion curvilinea."""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
import os
import struct
import uuid

import numpy as np


MAGIC = b"CURVOGUI1"
HEADER = struct.Struct("<9s7i7f")
# magic, pid, sequence, ny, nx, n_arrays, dtype_code, reserved,
# iter, t, Cl, Cd, divergencia, iteraciones/s, velocidad maxima
DTYPE_CODE = {np.dtype("float32"): 1, np.dtype("float64"): 2}


@dataclass(frozen=True)
class Snapshot:
    sequence: int
    X: np.ndarray
    Y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    metrics: dict[str, float | int]


class SnapshotWriter:
    """Publica snapshots completos en un bloque de memoria compartida."""

    def __init__(self, prefix: str | None = None):
        self.prefix = prefix or f"curvo_{uuid.uuid4().hex}"
        self._meta = None
        self._data = None
        self._shape = None
        self._dtype = None
        self._nbytes = 0
        self._sequence = 0

    @property
    def names(self) -> tuple[str, str]:
        return self.prefix + "_meta", self.prefix + "_data"

    def create(self, X: np.ndarray, Y: np.ndarray) -> None:
        X, Y = _arrays(X, Y)
        if X.shape != Y.shape or X.ndim != 2:
            raise ValueError("X e Y deben ser matrices 2D de igual forma")
        self._shape = X.shape
        self._dtype = X.dtype
        self._field_shape = (X.shape[0] - 1, X.shape[1] - 1)
        self._nbytes = 2 * X.nbytes + 3 * np.zeros(self._field_shape, dtype=X.dtype).nbytes
        self._meta = shared_memory.SharedMemory(name=self.names[0], create=True,
                                                 size=HEADER.size)
        self._data = shared_memory.SharedMemory(name=self.names[1], create=True,
                                                 size=self._nbytes)
        self._write_header(0, {})
        zeros = np.zeros(self._field_shape, dtype=self._dtype)
        self.publish(X, Y, zeros, zeros, zeros, {})

    def publish(self, X, Y, u, v, p, metrics) -> int:
        if self._data is None:
            self.create(X, Y)
            return self._sequence
        geometry = _arrays(X, Y)
        fields = _arrays(u, v, p)
        if any(a.shape != self._shape for a in geometry):
            raise ValueError("X e Y deben tener la forma de la malla")
        if any(a.shape != self._field_shape for a in fields):
            raise ValueError("u, v y p deben tener la forma de las celdas")
        if any(a.dtype != self._dtype for a in geometry + fields):
            raise ValueError("todos los campos deben tener el mismo dtype")
        self._sequence += 1
        self._write_header(self._sequence * 2 - 1, metrics)
        n_vertices = int(np.prod(self._shape))
        vertices = np.ndarray((2,) + self._shape, dtype=self._dtype,
                      buffer=self._data.buf, offset=0)
        vertices[...] = np.stack(geometry)
        fields_view = np.ndarray((3,) + self._field_shape, dtype=self._dtype,
                     buffer=self._data.buf, offset=2 * n_vertices * self._dtype.itemsize)
        fields_view[...] = np.stack(fields)
        self._write_header(self._sequence * 2, metrics)
        return self._sequence

    def _write_header(self, sequence: int, metrics: dict) -> None:
        valores = (float(metrics.get("iter", 0)), float(metrics.get("t", 0.0)),
                   float(metrics.get("Cl", 0.0)), float(metrics.get("Cd", 0.0)),
                   float(metrics.get("div", 0.0)), float(metrics.get("it_s", 0.0)),
                   float(metrics.get("speed_max", 0.0)))
        HEADER.pack_into(self._meta.buf, 0, MAGIC, os.getpid(), sequence,
                         *self._shape, 5, DTYPE_CODE[self._dtype], 0, *valores)

    def close(self, unlink: bool = True) -> None:
        for block in (self._meta, self._data):
            if block is not None:
                block.close()
                if unlink:
                    block.unlink()
        self._meta = self._data = None


class SnapshotReader:
    """Lee solo snapshots completos del writer asociado."""

    def __init__(self, names: tuple[str, str]):
        self.names = names
        self._meta = shared_memory.SharedMemory(name=names[0])
        self._data = shared_memory.SharedMemory(name=names[1])
        self.last_sequence = 0

    def read(self, metrics: dict[str, float | int] | None = None) -> Snapshot | None:
        first = HEADER.unpack_from(self._meta.buf)
        magic, _pid, sequence, ny, nx, count, dtype_code, _, it, t, cl, cd, div, it_s, speed_max = first
        if magic != MAGIC or sequence == 0 or sequence % 2:
            return None
        dtype = {value: key for key, value in DTYPE_CODE.items()}.get(dtype_code)
        if dtype is None or count != 5:
            return None
        field_shape = (ny - 1, nx - 1)
        n_vertices = ny * nx
        vertices = np.ndarray((2, ny, nx), dtype=dtype, buffer=self._data.buf,
                      offset=0).copy()
        fields = np.ndarray((3,) + field_shape, dtype=dtype, buffer=self._data.buf,
                    offset=2 * n_vertices * dtype.itemsize).copy()
        second_sequence = HEADER.unpack_from(self._meta.buf)[2]
        if sequence != second_sequence or sequence <= self.last_sequence:
            return None
        self.last_sequence = sequence
        datos = {"iter": int(it), "t": t, "Cl": cl, "Cd": cd,
             "div": div, "it_s": it_s, "speed_max": speed_max}
        datos.update(metrics or {})
        return Snapshot(sequence // 2, vertices[0], vertices[1], fields[0], fields[1],
                fields[2], datos)

    def close(self) -> None:
        self._meta.close()
        self._data.close()


def _arrays(*arrays):
    resultado = tuple(np.asarray(a) for a in arrays)
    if not resultado:
        raise ValueError("falta al menos un array")
    return resultado

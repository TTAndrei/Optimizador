"""Lienzo 2D: geometria, fronteras por click y campo en vivo.

pyqtgraph sobre PyQt6. Se usa PyQt6 y no PySide6 porque pyqtgraph elige PyQt6
cuando esta disponible, y cargar los dos bindings en el mismo proceso rompe los
imports de PySide6.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

# Un color por tipo de frontera. Los mismos en el lienzo y en la leyenda.
COLOR_BC = {
    "inflow":  "#2f8f4e",
    "outflow": "#c05a2a",
    "slip":    "#3a6ea5",
    "noslip":  "#8a8a8a",
}
COLOR_CUERPO = "#2b3440"
COLOR_EXTERIOR = "#5a4632"


class Lienzo(pg.PlotWidget):
    """Vista del dominio. Emite `parche_pedido(lado, desde, hasta)` cuando el
    usuario hace click en un tramo del perimetro."""

    parche_pedido = QtCore.pyqtSignal(str, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground("#f6f6f4")
        self.setAspectLocked(True)
        self.showGrid(x=True, y=True, alpha=0.15)
        self.setLabel("bottom", "x [m]")
        self.setLabel("left", "y [m]")

        self.escena = None
        self._items = []
        self._campo = pg.ImageItem()
        self._campo.setZValue(-10)
        self.addItem(self._campo)
        self._campo.hide()

        self.scene().sigMouseClicked.connect(self._click)

    # ------------------------------------------------------------------
    def dibujar(self, escena):
        self.escena = escena
        for it in self._items:
            self.removeItem(it)
        self._items.clear()

        self._add(self._rect(0, 0, escena.Lx, escena.Ly, "#b9b9b4", ancho=1))

        banda = escena.banda_fina()
        if banda:
            x0, x1, y0, y1 = banda
            self._add(self._rect(x0, y0, x1 - x0, y1 - y0, "#9ec5e8",
                                 ancho=1, guiones=True))

        for c in escena.contornos:
            x, y = c.arrays()
            color = COLOR_EXTERIOR if c.rol == "exterior" else COLOR_CUERPO
            relleno = pg.mkBrush(QtGui.QColor(color).lighter(160)) \
                if c.rol == "cuerpo" else None
            self._add(pg.PlotDataItem(x, y, pen=pg.mkPen(color, width=2),
                                      fillLevel=None, brush=relleno))

        for p, (lado, a, b) in self._tramos():
            x, y = self._coords_tramo(lado, a, b)
            self._add(pg.PlotDataItem(
                x, y, pen=pg.mkPen(COLOR_BC[p.tipo], width=7,
                                   cap=QtCore.Qt.PenCapStyle.FlatCap)))

        self.autoRange()

    def mostrar_campo(self, campo, x0, y0, x1, y1):
        """Pinta un campo escalar (ny, nx) estirado a la caja fisica."""
        if campo is None:
            self._campo.hide()
            return
        self._campo.setImage(np.asarray(campo).T, autoLevels=True)
        self._campo.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
        self._campo.show()

    # ------------------------------------------------------------------
    def _tramos(self):
        """(parche, (lado, desde, hasta)) con los None ya resueltos."""
        if self.escena is None:
            return
        for p in self.escena.parches:
            largo = (self.escena.Ly if p.lado in ("left", "right")
                     else self.escena.Lx)
            yield p, (p.lado,
                      0.0 if p.desde is None else p.desde,
                      largo if p.hasta is None else p.hasta)

    def _coords_tramo(self, lado, a, b):
        e = self.escena
        if lado == "left":
            return np.array([0.0, 0.0]), np.array([a, b])
        if lado == "right":
            return np.array([e.Lx, e.Lx]), np.array([a, b])
        if lado == "bottom":
            return np.array([a, b]), np.array([0.0, 0.0])
        return np.array([a, b]), np.array([e.Ly, e.Ly])

    def _click(self, ev):
        if self.escena is None or ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        pt = self.getViewBox().mapSceneToView(ev.scenePos())
        x, y, e = pt.x(), pt.y(), self.escena
        # Tolerancia del 3% de la dimension menor: en pantalla son unos milimetros.
        tol = 0.03 * min(e.Lx, e.Ly)
        cerca = {
            "left": abs(x - 0.0), "right": abs(x - e.Lx),
            "bottom": abs(y - 0.0), "top": abs(y - e.Ly),
        }
        lado = min(cerca, key=cerca.get)
        if cerca[lado] > tol:
            return
        s = y if lado in ("left", "right") else x
        largo = e.Ly if lado in ("left", "right") else e.Lx
        if not (-tol <= s <= largo + tol):
            return
        self.parche_pedido.emit(lado, max(0.0, s), largo)

    def _rect(self, x, y, w, h, color, ancho=1, guiones=False):
        pen = pg.mkPen(color, width=ancho)
        if guiones:
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        return pg.PlotDataItem([x, x + w, x + w, x, x],
                               [y, y, y + h, y + h, y], pen=pen)

    def _add(self, item):
        self.addItem(item)
        self._items.append(item)
        return item


class LeyendaBC(QtWidgets.QWidget):
    """Cuadradito de color por tipo de frontera. Sin esto, los tramos del
    perimetro son lineas de colores sin significado."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        for tipo, color in COLOR_BC.items():
            punto = QtWidgets.QLabel()
            punto.setFixedSize(12, 12)
            punto.setStyleSheet(f"background:{color};border-radius:2px;")
            lay.addWidget(punto)
            lay.addWidget(QtWidgets.QLabel(tipo))
            lay.addSpacing(10)
        lay.addStretch(1)

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
# Los solidos se pintan con este color plano encima del campo.
_RGBA_SOLIDO = np.array([0x2b, 0x34, 0x40, 255], dtype=np.uint8)
_LUT = pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256)
# Otro mapa para la malla: mirando la pantalla tiene que quedar claro si lo que
# se ve es el flujo o el tamaño de celda.
_LUT_MALLA = pg.colormap.get("magma").getLookupTable(0.0, 1.0, 256)


class Lienzo(pg.PlotWidget):
    """Vista del dominio. Emite `parche_pedido(lado, desde, hasta)` cuando el
    usuario hace click en un tramo del perimetro."""

    parche_pedido = QtCore.pyqtSignal(str, float, float)
    refinado_cambiado = QtCore.pyqtSignal(float, float, float, float)  # x0,x1,y0,y1

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
        self._nivel = None                 # escala de color, suavizada
        self.barra = BarraColor()          # la coloca la ventana

        # Caja de refinado arrastrable. Asas en las cuatro esquinas y en los
        # cuatro lados: una esquina reescala en las dos direcciones, un lado
        # mueve solo esa pared, que es como se ajusta el alto de una capa limite
        # sin tocar el largo.
        self.roi = pg.ROI([0, 0], [1, 1], pen=pg.mkPen("#1f77b4", width=2),
                          hoverPen=pg.mkPen("#1f77b4", width=3),
                          handlePen=pg.mkPen("#1f77b4"), movable=True,
                          rotatable=False, resizable=True)
        for pos, centro in (([0, 0], [1, 1]), ([1, 0], [0, 1]),
                            ([0, 1], [1, 0]), ([1, 1], [0, 0]),
                            ([0.5, 0], [0.5, 1]), ([0.5, 1], [0.5, 0]),
                            ([0, 0.5], [1, 0.5]), ([1, 0.5], [0, 0.5])):
            self.roi.addScaleHandle(pos, centro)
        self.roi.setZValue(20)
        self.addItem(self.roi)
        self.roi.hide()
        self.roi.sigRegionChangeFinished.connect(self._roi_movida)

        self.scene().sigMouseClicked.connect(self._click)

    # ------------------------------------------------------------------
    def dibujar(self, escena):
        self.escena = escena
        for it in self._items:
            self.removeItem(it)
        self._items.clear()

        self._add(self._rect(0, 0, escena.Lx, escena.Ly, "#b9b9b4", ancho=1))

        self.poner_refinado(escena.banda_fina())

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

    def poner_refinado(self, banda):
        """Coloca la caja de refinado sin reemitir la senal: si no, mover el
        cuadro de texto mueve la ROI, la ROI reescribe el cuadro y se entra en
        un ciclo de redondeos."""
        if not banda:
            self.roi.hide()
            return
        x0, x1, y0, y1 = banda
        self.roi.blockSignals(True)
        self.roi.setPos(x0, y0)
        self.roi.setSize((max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)))
        self.roi.blockSignals(False)
        self.roi.show()

    def _roi_movida(self):
        p, t = self.roi.pos(), self.roi.size()
        self.refinado_cambiado.emit(float(p.x()), float(p.x() + t.x()),
                                    float(p.y()), float(p.y() + t.y()))

    def mostrar_malla(self, mapa, solid, x, y, etiqueta, unidad):
        """Pinta el tamaño de celda. Escala fija (sin suavizado temporal) y
        limites reales, no percentiles: aqui interesa el maximo, que es el
        numero que dice si una zona resuelve algo o no."""
        self.mostrar_campo(mapa, solid, x, y, etiqueta=etiqueta, unidad=unidad,
                           lut=_LUT_MALLA, suavizar=False)

    def mostrar_campo(self, campo, solid, x, y, ancho=900,
                      etiqueta="|u|", unidad="m/s", lut=None, suavizar=True):
        """Pinta un campo escalar sobre el dominio fisico.

        `x` e `y` son los ejes REALES de la malla, que es estirada: pintar el
        array como una imagen uniforme deforma el campo, comprimiendo la zona
        fina y estirando la gruesa. Se remuestrea a una rejilla uniforme por
        vecino mas proximo (barato y sin inventar valores) usando esos ejes.

        Los solidos se pintan opacos en color de cuerpo en vez de dejarlos en el
        cero de la escala: si no, una pared y un remanso se ven igual.
        """
        if campo is None:
            self._campo.hide()
            self.barra.hide()
            return
        campo = np.asarray(campo, dtype=np.float32)
        ny, nx = campo.shape
        alto = max(16, int(round(ancho * (y[-1] - y[0]) / max(x[-1] - x[0], 1e-12))))
        ix = np.clip(np.interp(np.linspace(x[0], x[-1], ancho), x,
                               np.arange(nx)).round().astype(int), 0, nx - 1)
        iy = np.clip(np.interp(np.linspace(y[0], y[-1], alto), y,
                               np.arange(ny)).round().astype(int), 0, ny - 1)
        v = campo[np.ix_(iy, ix)]
        m = None if solid is None else np.asarray(solid)[np.ix_(iy, ix)]

        vv = v if m is None else v[~m]
        if not vv.size:
            vv = v.ravel()
        if suavizar:
            # Escala suavizada en el tiempo: recalcularla cada frame hace
            # parpadear el color entero cuando pasa un pico transitorio.
            pico = float(np.percentile(vv, 99.5))
            self._nivel = (pico if self._nivel is None
                           else 0.85 * self._nivel + 0.15 * pico)
            lo, hi = 0.0, max(self._nivel, 1e-9)
        else:
            lo, hi = float(vv.min()), float(vv.max())
            if hi - lo < 1e-12:
                hi = lo + 1e-12

        idx = np.clip((v - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
        rgba = np.empty(v.shape + (4,), dtype=np.uint8)
        rgba[..., :3] = (_LUT if lut is None else lut)[idx]
        rgba[..., 3] = 255
        if m is not None:
            rgba[m] = _RGBA_SOLIDO

        self._campo.setImage(rgba.transpose(1, 0, 2), autoLevels=False)
        self._campo.setRect(QtCore.QRectF(
            float(x[0]), float(y[0]), float(x[-1] - x[0]), float(y[-1] - y[0])))
        self._campo.show()
        self.barra.poner(lo, hi, unidad, etiqueta)
        self.barra.show()

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


class BarraColor(QtWidgets.QWidget):
    """Escala de color del campo, con sus dos extremos numerados.

    Sin ella el mapa es bonito y no dice nada: no se sabe si el amarillo son 2
    m/s o 200, ni si la escala se ha movido entre dos instantes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self._lo, self._hi, self._u = 0.0, 1.0, "m/s"
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        self.et_que = QtWidgets.QLabel("|u|")
        self.et_lo = QtWidgets.QLabel()
        self.et_hi = QtWidgets.QLabel()
        self.tira = QtWidgets.QFrame()
        self.tira.setMinimumWidth(160)
        self._pintar(_LUT)
        for w in (self.et_que, self.et_lo, self.tira, self.et_hi):
            lay.addWidget(w, 1 if w is self.tira else 0)
        lay.addStretch(1)
        self.poner(0.0, 1.0)

    def _pintar(self, lut):
        grad = ", ".join(
            f"stop:{i / 8:.3f} rgb({lut[i * 255 // 8][0]},"
            f"{lut[i * 255 // 8][1]},{lut[i * 255 // 8][2]})" for i in range(9))
        self.tira.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,{grad});"
            f"border:1px solid #b9b9b4;")

    def poner(self, lo, hi, unidad="m/s", etiqueta="|u|"):
        self._lo, self._hi, self._u = lo, hi, unidad
        self.et_que.setText(etiqueta)
        self._pintar(_LUT_MALLA if unidad in ("m", ":1") else _LUT)
        self.et_lo.setText(f"{lo:.3g}")
        self.et_hi.setText(f"{hi:.4g} {unidad}")

"""Formulario de parametros, generado desde params_spec."""
from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from gui.params_spec import GRUPOS


class PanelParametros(QtWidgets.QScrollArea):
    cambiado = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        cuerpo = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(cuerpo)
        lay.setContentsMargins(6, 6, 6, 6)

        self.campos = {}
        for titulo, params in GRUPOS:
            caja = QtWidgets.QGroupBox(titulo)
            form = QtWidgets.QFormLayout(caja)
            form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            for clave, etiqueta, tipo, defecto, ayuda in params:
                w = self._widget(tipo, defecto)
                if ayuda:
                    w.setToolTip(ayuda)
                self.campos[clave] = (w, tipo)
                form.addRow(etiqueta, w)
            lay.addWidget(caja)
        lay.addStretch(1)
        self.setWidget(cuerpo)

    def _widget(self, tipo, defecto):
        if tipo == "b":
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(defecto))
            w.stateChanged.connect(self.cambiado)
            return w
        if isinstance(tipo, tuple):
            w = QtWidgets.QComboBox()
            w.addItems(tipo[1])
            w.setCurrentText(str(defecto))
            w.currentTextChanged.connect(self.cambiado)
            return w
        w = QtWidgets.QLineEdit(_texto(defecto))
        w.editingFinished.connect(self.cambiado)
        return w

    # ------------------------------------------------------------------
    def valores(self):
        """dict listo para main(). Un campo vacio se omite y se queda con el
        valor por defecto del solver, que es mas seguro que colar un 0."""
        out = {}
        for clave, (w, tipo) in self.campos.items():
            if tipo == "b":
                out[clave] = w.isChecked()
            elif isinstance(tipo, tuple):
                out[clave] = w.currentText()
            else:
                t = w.text().strip()
                if not t:
                    continue
                out[clave] = int(float(t)) if tipo == "i" else float(t)
        return out

    def poner(self, valores):
        for clave, v in (valores or {}).items():
            par = self.campos.get(clave)
            if par is None:
                continue
            w, tipo = par
            if tipo == "b":
                w.setChecked(bool(v))
            elif isinstance(tipo, tuple):
                w.setCurrentText(str(v))
            else:
                w.setText(_texto(v))


def _texto(v):
    # None = campo vacio: `valores()` lo omite y manda el defecto del solver.
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)

"""Ventana principal.

Flujo: importar DXF -> clasificar contornos -> asignar fronteras por click ->
ajustar malla y parametros -> previsualizar -> lanzar.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gui.escena import Contorno, Escena, Parche, TIPOS_BC
from gui.lienzo import COLOR_BC, Lienzo, LeyendaBC
from gui.panel import PanelParametros
from gui.runner import Runner


class Ventana(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulador 2D — dominio y fronteras")
        self.resize(1500, 900)

        self.escena = Escena()
        self.runner = Runner(self)
        self.runner.log.connect(self._log)
        self.runner.progreso.connect(self._progreso)
        self.runner.campo.connect(self._campo)
        self.runner.terminado.connect(self._fin)

        self.lienzo = Lienzo()
        self.lienzo.parche_pedido.connect(self._nuevo_parche)
        self.panel = PanelParametros()

        centro = QtWidgets.QWidget()
        vc = QtWidgets.QVBoxLayout(centro)
        vc.setContentsMargins(0, 0, 0, 0)
        vc.addWidget(self.lienzo, 1)
        vc.addWidget(LeyendaBC())

        split = QtWidgets.QSplitter()
        split.addWidget(self._panel_izquierdo())
        split.addWidget(centro)
        split.addWidget(self._panel_derecho())
        split.setSizes([330, 800, 340])
        self.setCentralWidget(split)

        self._barra()
        self.statusBar().showMessage("Listo")
        self._refrescar()

    # ------------------------------------------------------------------ UI
    def _barra(self):
        b = self.addToolBar("principal")
        b.setMovable(False)
        for texto, fn in (("Importar DXF…", self.importar_dxf),
                          ("Abrir escena…", self.abrir),
                          ("Guardar escena…", self.guardar),
                          ("Previsualizar malla", self.previsualizar),
                          ("Lanzar", self.lanzar),
                          ("Parar", self.parar)):
            b.addAction(texto, fn)

    def _panel_izquierdo(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        caja = QtWidgets.QGroupBox("Dominio")
        f = QtWidgets.QFormLayout(caja)
        self.ed_lx = QtWidgets.QLineEdit(f"{self.escena.Lx:g}")
        self.ed_ly = QtWidgets.QLineEdit(f"{self.escena.Ly:g}")
        for ed in (self.ed_lx, self.ed_ly):
            ed.editingFinished.connect(self._leer_dominio)
        f.addRow("Lx [m]", self.ed_lx)
        f.addRow("Ly [m]", self.ed_ly)
        v.addWidget(caja)

        v.addWidget(QtWidgets.QLabel("Contornos"))
        self.lista = QtWidgets.QTreeWidget()
        self.lista.setColumnCount(3)
        self.lista.setHeaderLabels(["nombre", "rol", "pared"])
        self.lista.itemDoubleClicked.connect(self._editar_contorno)
        v.addWidget(self.lista, 1)

        v.addWidget(QtWidgets.QLabel("Fronteras (doble click para editar)"))
        self.lista_bc = QtWidgets.QTreeWidget()
        self.lista_bc.setColumnCount(4)
        self.lista_bc.setHeaderLabels(["lado", "tipo", "desde", "hasta"])
        self.lista_bc.itemDoubleClicked.connect(self._editar_parche)
        v.addWidget(self.lista_bc, 1)

        h = QtWidgets.QHBoxLayout()
        h.addWidget(self._boton("Fronteras por defecto", self._bc_por_defecto))
        h.addWidget(self._boton("Borrar frontera", self._borrar_parche))
        v.addLayout(h)
        return w

    def _panel_derecho(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        caja = QtWidgets.QGroupBox("Salida")
        f = QtWidgets.QFormLayout(caja)
        self.cb_salida = QtWidgets.QComboBox()
        self.cb_salida.addItems(["ninguno", "monitor", "grabar"])
        self.cb_salida.setCurrentText("monitor")
        self.cb_salida.setToolTip(
            "monitor: vista en vivo por memoria compartida, sin disco.\n"
            "grabar: además vuelca campos y monta el vídeo al terminar.\n"
            "El coste de cada modo está medido en results/coste_salida/.")
        f.addRow("Modo", self.cb_salida)
        v.addWidget(caja)

        self.metricas = QtWidgets.QLabel("—")
        self.metricas.setStyleSheet(
            "font-family:monospace;font-size:13px;padding:6px;")
        v.addWidget(self.metricas)

        v.addWidget(self.panel, 1)

        self.consola = QtWidgets.QPlainTextEdit()
        self.consola.setReadOnly(True)
        self.consola.setMaximumBlockCount(3000)
        self.consola.setStyleSheet("font-family:monospace;font-size:11px;")
        v.addWidget(self.consola, 1)
        return w

    def _boton(self, texto, fn):
        b = QtWidgets.QPushButton(texto)
        b.clicked.connect(fn)
        return b

    # ------------------------------------------------------------- acciones
    def importar_dxf(self):
        ruta, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Importar DXF", "", "DXF (*.dxf)")
        if not ruta:
            return
        from geom_import import GeometriaAbierta, cargar_dxf
        tol = self.panel.valores().get("dx_min", 0.004) / 4.0
        try:
            lazos = cargar_dxf(ruta, tol_cordal=tol)
        except GeometriaAbierta as e:
            QtWidgets.QMessageBox.warning(
                self, "Contorno abierto",
                f"{e}\n\nSe importa igualmente y los lazos abiertos quedan "
                f"marcados, pero no se puede simular así.")
            lazos = cargar_dxf(ruta, tol_cordal=tol, estricto=False)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error al leer el DXF", str(e))
            return

        for i, l in enumerate(lazos):
            self.escena.contornos.append(Contorno(
                x=list(map(float, l["x"])), y=list(map(float, l["y"])),
                rol=l["rol"], pared=l["pared"],
                nombre=f"{l['capa']}_{i}"))
        self._encajar_dominio()
        if not self.escena.parches:
            self._bc_por_defecto()
        self._log(f"[GUI] importados {len(lazos)} contorno(s) de "
                  f"{os.path.basename(ruta)}")
        self._refrescar()

    def abrir(self):
        ruta, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Abrir escena", "", "JSON (*.json)")
        if ruta:
            self.escena = Escena.cargar(ruta)
            self._volcar_al_panel()
            self.ed_lx.setText(f"{self.escena.Lx:g}")
            self.ed_ly.setText(f"{self.escena.Ly:g}")
            self.cb_salida.setCurrentText(self.escena.modo_salida)
            self._refrescar()

    def guardar(self):
        ruta, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Guardar escena", "escena.json", "JSON (*.json)")
        if ruta:
            self._recoger()
            self.escena.guardar(ruta)
            self._log(f"[GUI] escena guardada en {ruta}")

    def previsualizar(self):
        from gui.vista_malla import dialogo_malla
        self._recoger()
        dialogo_malla(self, self.escena)

    def lanzar(self):
        self._recoger()
        avisos = self.escena.avisos()
        if avisos:
            r = QtWidgets.QMessageBox.warning(
                self, "Avisos antes de lanzar", "\n\n".join(f"• {a}" for a in avisos)
                + "\n\n¿Lanzar igualmente?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No)
            if r != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        try:
            self.runner.lanzar(self.escena)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "No se pudo lanzar", str(e))

    def parar(self):
        self.runner.parar()

    # ------------------------------------------------------------- fronteras
    def _nuevo_parche(self, lado, desde, largo):
        tipo, ok = QtWidgets.QInputDialog.getItem(
            self, f"Frontera en «{lado}»", "Tipo:", list(TIPOS_BC), 0, False)
        if not ok:
            return
        valor = self._pedir_valor(tipo)
        if valor is False:
            return
        a, ok = QtWidgets.QInputDialog.getDouble(
            self, "Tramo", "desde:", desde, -1e6, 1e6, 4)
        if not ok:
            return
        b, ok = QtWidgets.QInputDialog.getDouble(
            self, "Tramo", "hasta:", largo, -1e6, 1e6, 4)
        if not ok:
            return
        self.escena.parches.append(Parche(lado, tipo, valor, a, b))
        self._refrescar()

    def _pedir_valor(self, tipo):
        if tipo == "inflow":
            u, ok = QtWidgets.QInputDialog.getDouble(
                self, "Entrada", "u [m/s]:", 1.0, -1e4, 1e4, 4)
            if not ok:
                return False
            v, ok = QtWidgets.QInputDialog.getDouble(
                self, "Entrada", "v [m/s]:", 0.0, -1e4, 1e4, 4)
            return (u, v) if ok else False
        if tipo == "outflow":
            p, ok = QtWidgets.QInputDialog.getDouble(
                self, "Salida", "presión:", 0.0, -1e6, 1e6, 4)
            return p if ok else False
        return None

    def _editar_parche(self, item, _col):
        i = self.lista_bc.indexOfTopLevelItem(item)
        p = self.escena.parches[i]
        tipo, ok = QtWidgets.QInputDialog.getItem(
            self, "Frontera", "Tipo:", list(TIPOS_BC),
            list(TIPOS_BC).index(p.tipo), False)
        if not ok:
            return
        v = self._pedir_valor(tipo)
        if v is False:
            return
        p.tipo, p.valor = tipo, v
        self._refrescar()

    def _borrar_parche(self):
        it = self.lista_bc.currentItem()
        if it is None:
            return
        del self.escena.parches[self.lista_bc.indexOfTopLevelItem(it)]
        self._refrescar()

    def _bc_por_defecto(self):
        self.escena.parches = [
            Parche("left", "inflow", (1.0, 0.0)),
            Parche("right", "outflow", 0.0),
            Parche("top", "slip"),
            Parche("bottom", "slip"),
        ]
        self._refrescar()

    def _editar_contorno(self, item, _col):
        i = self.lista.indexOfTopLevelItem(item)
        c = self.escena.contornos[i]
        rol, ok = QtWidgets.QInputDialog.getItem(
            self, c.nombre, "Rol:", ["cuerpo", "exterior"],
            0 if c.rol == "cuerpo" else 1, False)
        if not ok:
            return
        pared, ok = QtWidgets.QInputDialog.getItem(
            self, c.nombre, "Pared:", ["noslip", "slip"],
            0 if c.pared == "noslip" else 1, False)
        if not ok:
            return
        c.rol, c.pared = rol, pared
        self._refrescar()

    # ------------------------------------------------------------- utilidad
    def _leer_dominio(self):
        try:
            self.escena.Lx = float(self.ed_lx.text())
            self.escena.Ly = float(self.ed_ly.text())
        except ValueError:
            return
        self._refrescar()

    def _encajar_dominio(self):
        """Ajusta la caja a lo importado. Con contorno exterior, la caja es su
        bbox; con cuerpos sueltos, se deja sitio para la estela."""
        ext = [c for c in self.escena.contornos if c.rol == "exterior"]
        cajas = [c.bbox() for c in (ext or self.escena.contornos)]
        if not cajas:
            return
        x0 = min(b[0] for b in cajas); y0 = min(b[1] for b in cajas)
        x1 = max(b[2] for b in cajas); y1 = max(b[3] for b in cajas)
        if ext:
            self.escena.Lx, self.escena.Ly = x1 - x0, y1 - y0
            desplazar = (-x0, -y0)
        else:
            L = max(x1 - x0, y1 - y0)
            self.escena.Lx, self.escena.Ly = 24.0 * L, 16.0 * L
            desplazar = (6.0 * L - x0, 8.0 * L - 0.5 * (y0 + y1))
        for c in self.escena.contornos:
            c.x = [v + desplazar[0] for v in c.x]
            c.y = [v + desplazar[1] for v in c.y]
        self.ed_lx.setText(f"{self.escena.Lx:g}")
        self.ed_ly.setText(f"{self.escena.Ly:g}")

    def _volcar_al_panel(self):
        """dx_min y factor_expansion viven en la escena, no en el diccionario de
        solver. Sin esto, cargar una escena deja el panel con SUS valores y al
        lanzar se simula con una malla distinta de la que se ve."""
        self.panel.poner({**self.escena.solver,
                          "dx_min": self.escena.dx_min,
                          "factor_expansion": self.escena.factor_expansion})

    def _recoger(self):
        self.escena.solver = self.panel.valores()
        self.escena.dx_min = self.escena.solver.get("dx_min", self.escena.dx_min)
        self.escena.factor_expansion = self.escena.solver.get(
            "factor_expansion", self.escena.factor_expansion)
        self.escena.modo_salida = self.cb_salida.currentText()

    def _refrescar(self):
        self.lista.clear()
        for c in self.escena.contornos:
            QtWidgets.QTreeWidgetItem(self.lista, [c.nombre, c.rol, c.pared])
        self.lista_bc.clear()
        for p in self.escena.parches:
            it = QtWidgets.QTreeWidgetItem(self.lista_bc, [
                p.lado, p.tipo,
                "—" if p.desde is None else f"{p.desde:g}",
                "—" if p.hasta is None else f"{p.hasta:g}"])
            it.setForeground(1, QtGui.QBrush(QtGui.QColor(COLOR_BC[p.tipo])))
        self.lienzo.dibujar(self.escena)

    def _log(self, txt):
        self.consola.appendPlainText(txt)

    def _progreso(self, d):
        ld = d["cl"] / d["cd"] if abs(d["cd"]) > 1e-9 else float("nan")
        self.metricas.setText(
            f"iter {d['iter']:>7d}\nCl   {d['cl']:+.5f}\n"
            f"Cd   {d['cd']:+.5f}\nL/D  {ld:+.2f}")

    def _campo(self, speed, x, y):
        self.lienzo.mostrar_campo(speed, float(x[0]), float(y[0]),
                                  float(x[-1]), float(y[-1]))

    def _fin(self, code):
        self._log(f"[GUI] el solver terminó con código {code}")
        self.statusBar().showMessage(f"Terminado ({code})")

    def closeEvent(self, ev):
        self.runner.matar()
        super().closeEvent(ev)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    v = Ventana()
    v.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

"""Ventana principal.

Flujo: importar DXF -> clasificar contornos -> asignar fronteras por click ->
ajustar malla y parametros -> previsualizar -> lanzar.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gui.escena import (Contorno, Escena, MODOS_DOMINIO, Parche, TIPOS_BC,
                        parches_por_defecto)
from gui.lienzo import COLOR_BC, Lienzo, LeyendaBC
from gui.panel import PanelParametros
from gui.vista_malla import MODOS_MALLA
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
        self.lienzo.refinado_cambiado.connect(self._roi_refinado)
        self._malla_cache = None
        self.lienzo.scene().sigMouseMoved.connect(self._raton)
        self.panel = PanelParametros()

        malla = QtWidgets.QWidget()
        vc = QtWidgets.QVBoxLayout(malla)
        vc.setContentsMargins(0, 0, 0, 0)
        vc.addWidget(self.lienzo, 1)
        vc.addWidget(self.lienzo.barra)
        vc.addWidget(LeyendaBC())
        self.lienzo.barra.hide()

        # La convergencia va debajo del dominio y se puede agrandar arrastrando:
        # a veces interesa el campo y a veces la curva, y no hay un reparto que
        # valga para las dos cosas.
        centro = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        centro.addWidget(malla)
        centro.addWidget(self._zona_convergencia())
        centro.setStretchFactor(0, 3)
        centro.setSizes([620, 230])

        split = QtWidgets.QSplitter()
        split.addWidget(self._panel_izquierdo())
        split.addWidget(centro)
        split.addWidget(self._panel_derecho())
        split.setSizes([330, 800, 380])
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)

        self._barra()
        self.statusBar().showMessage("Listo")
        self._sinc_dominio()
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

        b.addSeparator()
        self.acc_malla = QtGui.QAction("Ver tamaño de celda", self)
        self.acc_malla.setCheckable(True)
        self.acc_malla.setToolTip(
            "Pinta el tamaño de celda sobre el dominio, antes de lanzar. "
            "Pasando el ratón se leen dx y dy en ese punto.")
        self.acc_malla.toggled.connect(self._ver_malla)
        b.addAction(self.acc_malla)
        self.cb_malla = QtWidgets.QComboBox()
        self.cb_malla.addItems(list(MODOS_MALLA))
        self.cb_malla.currentTextChanged.connect(
            lambda _: self._ver_malla(self.acc_malla.isChecked()))
        b.addWidget(self.cb_malla)

    def _panel_izquierdo(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        caja = QtWidgets.QGroupBox("Dominio")
        f = QtWidgets.QFormLayout(caja)
        self.cb_modo = QtWidgets.QComboBox()
        self.cb_modo.addItems(["caja: el DXF es solo el objeto",
                               "dxf: el DXF define las paredes"])
        self.cb_modo.setToolTip(
            "caja: el DXF trae el objeto y el dominio lo pones tú con Lx/Ly; "
            "el objeto se coloca donde quieras.\n"
            "dxf: el DXF trae el contorno exterior (conducto, tobera) y ES el "
            "dominio; Lx/Ly salen de él y no se editan.")
        self.cb_modo.currentIndexChanged.connect(self._cambiar_modo)
        f.addRow("Modo", self.cb_modo)
        self.ed_lx = QtWidgets.QLineEdit(f"{self.escena.Lx:g}")
        self.ed_ly = QtWidgets.QLineEdit(f"{self.escena.Ly:g}")
        for ed in (self.ed_lx, self.ed_ly):
            ed.editingFinished.connect(self._leer_dominio)
        f.addRow("Lx [m]", self.ed_lx)
        f.addRow("Ly [m]", self.ed_ly)
        f.addRow(self._boton("Encajar dominio", self._encajar_dominio))
        v.addWidget(caja)

        v.addWidget(self._caja_refinado())

        v.addWidget(QtWidgets.QLabel("Contornos"))
        self.lista = QtWidgets.QTreeWidget()
        self.lista.setColumnCount(5)
        self.lista.setHeaderLabels(["nombre", "rol", "pared", "x", "y"])
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

    def _caja_refinado(self):
        """Zona de malla fina: se arrastra en el lienzo o se teclea aqui.

        Una sola banda, no N. Varias bandas disjuntas exigen otro generador de
        malla y revalidar el detector de simetria del multigrid, `_bc_cache` y
        el `self.dx = min(diff)` del solver, que dan por hecho un unico minimo.
        """
        caja = QtWidgets.QGroupBox("Zona de refinado")
        f = QtWidgets.QFormLayout(caja)
        self.chk_ref = QtWidgets.QCheckBox("a mano")
        self.chk_ref.setToolTip(
            "Sin marcar, la banda sale del bbox de los cuerpos (y de toda la "
            "sección en y si el dominio viene del DXF).")
        self.chk_ref.toggled.connect(self._conmutar_refinado)
        f.addRow(self.chk_ref)

        self.ed_ref = {}
        for clave, etiqueta in (("x", "x [m]"), ("y", "y [m]"),
                                ("w", "ancho [m]"), ("h", "alto [m]")):
            ed = QtWidgets.QLineEdit()
            ed.editingFinished.connect(self._leer_refinado)
            self.ed_ref[clave] = ed
            f.addRow(etiqueta, ed)

        h = QtWidgets.QHBoxLayout()
        h.addWidget(self._boton("A los cuerpos", self._ref_cuerpos))
        h.addWidget(self._boton("Todo el dominio", self._ref_todo))
        f.addRow(h)
        return caja

    def _conmutar_refinado(self, activo):
        if activo:
            self.escena.refinado = tuple(self.escena.banda_fina()
                                         or (0.0, self.escena.Lx, 0.0, self.escena.Ly))
        else:
            self.escena.refinado = None
        self._refrescar()

    def _leer_refinado(self):
        """Los cuadros de texto mandan sobre la ROI."""
        try:
            x, y, w, h = (float(self.ed_ref[k].text()) for k in ("x", "y", "w", "h"))
        except ValueError:
            return
        if w <= 0 or h <= 0:
            return
        self.chk_ref.setChecked(True)
        self.escena.refinado = (x, x + w, y, y + h)
        self._refrescar()

    def _roi_refinado(self, x0, x1, y0, y1):
        """La ROI manda sobre los cuadros de texto."""
        self.chk_ref.setChecked(True)
        self.escena.refinado = (x0, x1, y0, y1)
        self._refrescar()

    def _ref_cuerpos(self):
        bb = self.escena.bbox_cuerpos()
        if bb is None:
            self._log("[GUI] no hay cuerpos a los que ajustar la zona fina")
            return
        self.chk_ref.setChecked(False)     # la automatica ya es esa
        self._refrescar()

    def _ref_todo(self):
        self.chk_ref.setChecked(True)
        self.escena.refinado = (0.0, self.escena.Lx, 0.0, self.escena.Ly)
        self._refrescar()

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

        self.barra_it = QtWidgets.QProgressBar()
        self.barra_it.setTextVisible(True)
        self.barra_it.setFormat("%v / %m iteraciones (%p %)")
        f.addRow(self.barra_it)
        v.addWidget(caja)

        self.consola = QtWidgets.QPlainTextEdit()
        self.consola.setReadOnly(True)
        self.consola.setMaximumBlockCount(3000)
        self.consola.setStyleSheet("font-family:monospace;font-size:11px;")
        caja_log = QtWidgets.QGroupBox("Log del solver")
        caja_log.setCheckable(True)
        caja_log.setChecked(False)
        vl = QtWidgets.QVBoxLayout(caja_log)
        vl.addWidget(self.consola)
        caja_log.toggled.connect(self.consola.setVisible)
        self.consola.hide()

        # Los parametros y el log reparten la altura arrastrando: son 26 campos
        # en cinco grupos y con el log fijo abajo no caben.
        pila = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        pila.addWidget(self.panel)
        pila.addWidget(caja_log)
        pila.setStretchFactor(0, 4)
        pila.setSizes([700, 120])
        v.addWidget(pila, 1)
        return w

    def _zona_convergencia(self):
        """Cl y Cd contra iteracion. Es lo que dice si ha convergido o si esta
        oscilando; un numero suelto en la consola no lo dice."""
        caja = QtWidgets.QGroupBox("Convergencia")
        lay = QtWidgets.QVBoxLayout(caja)
        lay.setContentsMargins(6, 4, 6, 4)
        self.metricas = QtWidgets.QLabel("—")
        self.metricas.setStyleSheet("font-family:monospace;font-size:13px;")
        lay.addWidget(self.metricas)
        self.grafica = pg.PlotWidget()
        self.grafica.setBackground("#f6f6f4")
        self.grafica.setMinimumHeight(90)
        self.grafica.showGrid(x=True, y=True, alpha=0.2)
        self.grafica.setLabel("bottom", "iteración")
        self.grafica.addLegend(offset=(-5, 5))
        self._c_cl = self.grafica.plot(pen=pg.mkPen("#2f8f4e", width=2), name="Cl")
        self._c_cd = self.grafica.plot(pen=pg.mkPen("#c05a2a", width=2), name="Cd")
        self._hist = {"it": [], "cl": [], "cd": []}
        lay.addWidget(self.grafica, 1)
        return caja

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
        msg = self.escena.autoconfigurar_dominio()
        self._ajustar_malla_al_modo()
        self._sinc_dominio()
        if not self.escena.parches:
            self._bc_por_defecto()
        self._log(f"[GUI] importados {len(lazos)} contorno(s) de "
                  f"{os.path.basename(ruta)}")
        self._log(f"[GUI] modo «{self.escena.modo_dominio}». {msg}")
        self._refrescar()

    def abrir(self):
        ruta, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Abrir escena", "", "JSON (*.json)")
        if ruta:
            self.escena = Escena.cargar(ruta)
            self._volcar_al_panel()
            self._sinc_dominio()
            self.cb_salida.setCurrentText(self.escena.modo_salida)
            self._refrescar()

    def guardar(self):
        ruta, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Guardar escena", "escena.json", "JSON (*.json)")
        if ruta:
            self._recoger()
            self.escena.guardar(ruta)
            self._log(f"[GUI] escena guardada en {ruta}")

    def _ver_malla(self, activo):
        """Pinta el tamaño de celda sobre el dominio. Mientras esté activo manda
        sobre el campo en vivo: comparten el mismo ImageItem y si no se pisan el
        uno al otro cada 150 ms."""
        self._malla_cache = None
        if not activo:
            self.lienzo.mostrar_campo(None, None, None, None)
            self.statusBar().showMessage("Listo")
            return
        self._recoger()
        try:
            from gui.vista_malla import mapa_celda, mascara, ejes
            X, Y = ejes(self.escena)
            solid = mascara(self.escena, X, Y)
        except Exception as e:
            self.acc_malla.setChecked(False)
            QtWidgets.QMessageBox.critical(
                self, "No se pudo generar la malla", f"{type(e).__name__}: {e}")
            return
        modo = self.cb_malla.currentText()
        etiqueta, unidad = MODOS_MALLA[modo]
        mapa = mapa_celda(X, Y, modo)
        self._malla_cache = (X, Y, mapa, unidad)
        self.lienzo.mostrar_malla(mapa, solid, X, Y, etiqueta, unidad)
        self.statusBar().showMessage(
            f"{len(X)}×{len(Y)} nodos · {etiqueta} de "
            f"{mapa.min():.4g} a {mapa.max():.4g} {unidad}")

    def _raton(self, pos):
        """Lee dx y dy bajo el cursor. Un mapa de color dice donde esta la malla
        fina; esto dice cuanto mide exactamente ahi."""
        if self._malla_cache is None:
            return
        X, Y, mapa, unidad = self._malla_cache
        vb = self.lienzo.getViewBox()
        if not self.lienzo.sceneBoundingRect().contains(pos):
            return
        p = vb.mapSceneToView(pos)
        i = int(np.clip(np.searchsorted(X, p.x()), 0, len(X) - 1))
        j = int(np.clip(np.searchsorted(Y, p.y()), 0, len(Y) - 1))
        hx = np.gradient(X)[i]
        hy = np.gradient(Y)[j]
        self.statusBar().showMessage(
            f"x={p.x():.4f}  y={p.y():.4f}   dx={hx:.5f} m  dy={hy:.5f} m   "
            f"aspecto {max(hx / hy, hy / hx):.1f}:1   "
            f"{self.cb_malla.currentText()}={mapa[j, i]:.5g} {unidad}")

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
        self._hist = {"it": [], "cl": [], "cd": []}
        self._c_cl.setData([], []); self._c_cd.setData([], [])
        self._total_it = int(self.escena.solver.get("iteraciones", 0) or 0)
        self.barra_it.setValue(0)
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
        self.escena.parches = parches_por_defecto(self.escena.modo_dominio)
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
        if rol == "cuerpo":
            cx, cy = c.centro()
            x, ok = QtWidgets.QInputDialog.getDouble(
                self, c.nombre, "centro x [m]:", cx, -1e6, 1e6, 5)
            if ok:
                y, ok2 = QtWidgets.QInputDialog.getDouble(
                    self, c.nombre, "centro y [m]:", cy, -1e6, 1e6, 5)
                if ok2:
                    c.mover_centro_a(x, y)
        self._refrescar()

    # ------------------------------------------------------------- utilidad
    def _leer_dominio(self):
        try:
            Lx, Ly = float(self.ed_lx.text()), float(self.ed_ly.text())
        except ValueError:
            return
        if self.escena.modo_dominio == "dxf":
            return                     # los campos estan deshabilitados
        self.escena.Lx, self.escena.Ly = Lx, Ly
        self._refrescar()

    def _cambiar_modo(self, i):
        self.escena.modo_dominio = MODOS_DOMINIO[i]
        # Cambiar de modo cambia quien es pared: en «dxf» los lados de arriba y
        # abajo son el conducto, en «caja» son campo lejano.
        self.escena.parches = parches_por_defecto(self.escena.modo_dominio)
        self._log(f"[GUI] modo de dominio «{self.escena.modo_dominio}». "
                  f"{self.escena.ajustar_dominio()}")
        self._ajustar_malla_al_modo()
        self._sinc_dominio()
        self._refrescar()

    def _encajar_dominio(self):
        """En «dxf» reescribe Lx/Ly a partir del contorno exterior; en «caja»
        recoloca el objeto en el cuarto delantero del dominio."""
        self._log(f"[GUI] {self.escena.ajustar_dominio()}")
        self._sinc_dominio()
        self._refrescar()

    def _ajustar_malla_al_modo(self):
        """En un conducto, dy queda fijo en dx_min por toda la seccion, asi que
        cada paso de estirado en x es relacion de aspecto directa. Con el 50 por
        defecto salen celdas 25:1, el multigrid deja de converger y revienta
        (medido: iteracion 790 en el conducto de 4.8x1). 8 aguanta."""
        if self.escena.modo_dominio != "dxf":
            return
        v = self.panel.valores()
        if v.get("ratio_max_malla", 50) > 8:
            self.panel.poner({**v, "ratio_max_malla": 8})
            self._log("[GUI] ratio máximo de malla bajado a 8: con contorno "
                      "exterior, estirar en x da celdas anisótropas y el "
                      "multigrid no converge.")

    def _sinc_dominio(self):
        """Deja los widgets del dominio de acuerdo con la escena. En modo «dxf»
        Lx/Ly los manda la geometria, asi que se muestran pero no se editan."""
        dxf = self.escena.modo_dominio == "dxf"
        self.cb_modo.blockSignals(True)
        self.cb_modo.setCurrentIndex(MODOS_DOMINIO.index(self.escena.modo_dominio))
        self.cb_modo.blockSignals(False)
        self.ed_lx.setText(f"{self.escena.Lx:g}")
        self.ed_ly.setText(f"{self.escena.Ly:g}")
        for ed in (self.ed_lx, self.ed_ly):
            ed.setEnabled(not dxf)
            ed.setToolTip("Lo fija el contorno exterior del DXF." if dxf else "")

    def _volcar_al_panel(self):
        """dx_min y factor_expansion viven en la escena, no en el diccionario de
        solver. Sin esto, cargar una escena deja el panel con SUS valores y al
        lanzar se simula con una malla distinta de la que se ve."""
        self.panel.poner({**self.escena.solver,
                          "dx_min": self.escena.dx_min,
                          "dy_min": self.escena.dy_min,
                          "factor_expansion": self.escena.factor_expansion})

    def _recoger(self):
        self.escena.solver = self.panel.valores()
        self.escena.dx_min = self.escena.solver.get("dx_min", self.escena.dx_min)
        self.escena.dy_min = self.escena.solver.get("dy_min")
        self.escena.factor_expansion = self.escena.solver.get(
            "factor_expansion", self.escena.factor_expansion)
        self.escena.modo_salida = self.cb_salida.currentText()

    def _refrescar(self):
        self.lista.clear()
        for c in self.escena.contornos:
            cx, cy = c.centro()
            QtWidgets.QTreeWidgetItem(self.lista, [
                c.nombre, c.rol, c.pared, f"{cx:.4g}", f"{cy:.4g}"])
        self.lista_bc.clear()
        for p in self.escena.parches:
            it = QtWidgets.QTreeWidgetItem(self.lista_bc, [
                p.lado, p.tipo,
                "—" if p.desde is None else f"{p.desde:g}",
                "—" if p.hasta is None else f"{p.hasta:g}"])
            it.setForeground(1, QtGui.QBrush(QtGui.QColor(COLOR_BC[p.tipo])))
        self._volcar_refinado()
        self.lienzo.dibujar(self.escena)
        # Rehacer el mapa, no apagarlo: arrastrar la caja de refinado y ver el
        # tamaño de celda cambiar es justo para lo que sirve.
        if getattr(self, "_malla_cache", None) is not None:
            self._ver_malla(True)

    def _volcar_refinado(self):
        banda = self.escena.banda_fina()
        self.chk_ref.blockSignals(True)
        self.chk_ref.setChecked(self.escena.refinado is not None)
        self.chk_ref.blockSignals(False)
        for ed in self.ed_ref.values():
            ed.setEnabled(self.escena.refinado is not None)
        if not banda:
            for ed in self.ed_ref.values():
                ed.setText("")
            return
        x0, x1, y0, y1 = banda
        for k, v in (("x", x0), ("y", y0), ("w", x1 - x0), ("h", y1 - y0)):
            ed = self.ed_ref[k]
            ed.blockSignals(True)
            ed.setText(f"{v:.4g}")
            ed.blockSignals(False)

    def _log(self, txt):
        self.consola.appendPlainText(txt)

    def _progreso(self, d):
        ld = d["cl"] / d["cd"] if abs(d["cd"]) > 1e-9 else float("nan")
        total = max(getattr(self, "_total_it", 0), 1)
        self.barra_it.setMaximum(total)
        self.barra_it.setValue(min(d["iter"], total))
        self.metricas.setText(
            f"Cl {d['cl']:+.5f}   Cd {d['cd']:+.5f}   "
            f"L/D {ld:+.2f}   α {d['alpha']:.2f}°")
        h = self._hist
        if not h["it"] or d["iter"] != h["it"][-1]:
            h["it"].append(d["iter"]); h["cl"].append(d["cl"]); h["cd"].append(d["cd"])
            self._c_cl.setData(h["it"], h["cl"])
            self._c_cd.setData(h["it"], h["cd"])

    def _campo(self, speed, solid, x, y):
        if self.acc_malla.isChecked():
            return          # el modo malla manda mientras este activo
        self.lienzo.mostrar_campo(speed, solid, x, y)

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

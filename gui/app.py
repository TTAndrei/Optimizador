"""Ventana principal de la GUI curvilinea."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

from PyQt6 import QtCore, QtGui, QtWidgets

from .caso import Caso, CasoError, previsualizar_malla
from .grafica import GraficaFuerzas
from .params_spec import PERFIL, PESTANAS
from .runner import Runner
from .visor import Visor


ROOT = Path(__file__).resolve().parents[1]


MALLA_RAPIDA = {
    "dn_pared": 2.0e-3,
    "crecimiento": 1.16,
    "distancia_lejos": 3.0,
    "n_sup": 96,
    "n_estela": 32,
    "x_out": 6.0,
    "n_capas_pared": 3,
    "aspecto_max": 80.0,
}


MALLA_PRODUCCION = Caso().malla.copy()


class Ventana(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Solver curvilineo | perfiles alares")
        self.resize(1480, 900)
        self.caso = Caso()
        self.runner = Runner(self)
        self.runner.campo.connect(self._campo)
        self.runner.progreso.connect(self._progreso)
        self.runner.log.connect(self._log)
        self.runner.terminado.connect(self._terminado)
        self.visor = Visor()
        self.grafica = GraficaFuerzas()
        self._campos = {}
        self._calidad_labels = {}
        self._metricas_labels = {}
        self._actualizando_ui = False
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._previsualizar)
        self._crear_ui()
        self._sincronizar_ui()
        self._previsualizar()

    def _crear_ui(self):
        central = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        central.setChildrenCollapsible(False)

        panel = self._crear_panel()
        central.addWidget(panel)

        derecha = QtWidgets.QWidget()
        derecha_layout = QtWidgets.QVBoxLayout(derecha)
        derecha_layout.setContentsMargins(10, 10, 10, 10)
        derecha_layout.setSpacing(10)
        derecha_layout.addLayout(self._crear_barra_visualizacion())
        derecha_layout.addWidget(self.visor, 1)
        derecha_layout.addWidget(self.grafica, 0)
        central.addWidget(derecha)
        central.setSizes((430, 1050))
        self.setCentralWidget(central)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f5f7fa; color: #18212f; }
            QGroupBox {
                border: 1px solid #c8d1dc;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 12px;
                font-weight: 600;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QGroupBox::indicator { width: 11px; height: 11px; }
            QTabWidget::pane {
                border: 1px solid #c8d1dc;
                border-radius: 6px;
                background: #ffffff;
            }
            QTabBar::tab {
                background: #e7ecf3;
                border: 1px solid #c8d1dc;
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                padding: 5px 13px;
                margin-right: 2px;
            }
            QTabBar::tab:selected { background: #ffffff; font-weight: 600; }
            QTabBar::tab:hover:!selected { background: #eef4ff; }
            QLineEdit, QComboBox, QPlainTextEdit {
                background: #ffffff;
                border: 1px solid #bac5d1;
                border-radius: 4px;
                padding: 4px 6px;
            }
            QLineEdit:focus, QComboBox:focus { border-color: #1f6feb; }
            QPushButton {
                background: #ffffff;
                border: 1px solid #aeb9c6;
                border-radius: 5px;
                padding: 6px 9px;
            }
            QPushButton:hover { background: #eef4ff; border-color: #6b93d6; }
            QPushButton:disabled { color: #798391; background: #edf0f4; }
        """)

    def _crear_panel(self):
        contenedor = QtWidgets.QWidget()
        exterior = QtWidgets.QVBoxLayout(contenedor)
        exterior.setContentsMargins(10, 10, 8, 10)
        exterior.setSpacing(8)

        titulo = QtWidgets.QLabel("Caso curvilineo")
        titulo.setStyleSheet("font-size: 18px; font-weight: 700;")
        exterior.addWidget(titulo)

        self._estado = QtWidgets.QLabel("Listo")
        self._estado.setWordWrap(True)
        self._estado.setStyleSheet("color: #334155;")
        exterior.addWidget(self._estado)

        exterior.addWidget(self._grupo_perfil())

        # Las pestanas se llevan los ~40 parametros; calidad y metricas se
        # quedan fuera, siempre a la vista, porque son el resultado de tocarlos.
        self._pestanas = QtWidgets.QTabWidget()
        self._pestanas.setMinimumHeight(330)
        for nombre, bloques in PESTANAS:
            self._pestanas.addTab(self._pestana(bloques), nombre)
        exterior.addWidget(self._pestanas, 1)

        exterior.addWidget(self._grupo_calidad())
        exterior.addWidget(self._grupo_metricas())

        acciones = QtWidgets.QGridLayout()
        self._boton_preview = QtWidgets.QPushButton("Previsualizar")
        self._boton_preview.clicked.connect(self._previsualizar)
        self._boton_lanzar = QtWidgets.QPushButton("Lanzar solver")
        self._boton_lanzar.clicked.connect(self._lanzar)
        self._boton_parar = QtWidgets.QPushButton("Parar")
        self._boton_parar.clicked.connect(self.runner.parar)
        self._boton_parar.setEnabled(False)
        acciones.addWidget(self._boton_preview, 0, 0)
        acciones.addWidget(self._boton_lanzar, 0, 1)
        acciones.addWidget(self._boton_parar, 0, 2)
        exterior.addLayout(acciones)

        self._log_view = QtWidgets.QPlainTextEdit(readOnly=True)
        self._log_view.setMaximumBlockCount(700)
        self._log_view.setFixedHeight(96)
        exterior.addWidget(self._log_view)
        return contenedor

    def _grupo_perfil(self):
        caja = QtWidgets.QGroupBox("Perfil y caso")
        layout = QtWidgets.QGridLayout(caja)
        layout.setColumnStretch(1, 1)

        self._perfil = QtWidgets.QComboBox()
        for ruta in sorted((ROOT / "profiles").iterdir()):
            if ruta.is_file():
                self._perfil.addItem(ruta.name)
        self._perfil.currentTextChanged.connect(self._perfil_cambiado)
        self._perfil.setToolTip(PERFIL[3])
        layout.addWidget(QtWidgets.QLabel("Perfil"), 0, 0)
        layout.addWidget(self._perfil, 0, 1, 1, 3)

        for columna, (texto, funcion) in enumerate((
                ("Rapida", lambda: self._aplicar_preset(MALLA_RAPIDA)),
                ("Produccion", lambda: self._aplicar_preset(MALLA_PRODUCCION)),
                ("Abrir", self._abrir_caso),
                ("Guardar", self._guardar_caso))):
            boton = QtWidgets.QPushButton(texto)
            boton.clicked.connect(funcion)
            layout.addWidget(boton, 1, columna)
            layout.setColumnStretch(columna, 1)
        return caja

    def _pestana(self, bloques):
        """Una pestana: scroll con los bloques plegables de esa familia."""
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        contenido = QtWidgets.QWidget()
        columna = QtWidgets.QVBoxLayout(contenido)
        columna.setContentsMargins(4, 6, 8, 6)
        columna.setSpacing(6)
        for indice, (nombre, entradas) in enumerate(bloques):
            # El primer bloque de cada pestana abierto: es el que se toca.
            columna.addWidget(self._bloque(nombre, entradas, abierto=indice == 0))
        columna.addStretch(1)
        scroll.setWidget(contenido)
        return scroll

    def _bloque(self, nombre, entradas, abierto):
        """Bloque plegable: el titulo es la casilla que lo abre y lo cierra."""
        caja = QtWidgets.QGroupBox(nombre)
        caja.setCheckable(True)
        caja.setChecked(abierto)
        form = QtWidgets.QFormLayout(caja)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setContentsMargins(10, 6, 10, 8)
        form.setSpacing(5)
        cuerpo = []
        for clave, etiqueta, tipo, ayuda, *resto in entradas:
            grupo = self._grupo_de_clave(clave)
            if grupo is None:
                continue
            editor = self._crear_editor(grupo, clave, tipo, resto[0] if resto else None)
            self._campos[(grupo, clave)] = editor
            titulo = QtWidgets.QLabel(etiqueta)
            for widget in (titulo, editor):
                widget.setToolTip(ayuda)
            titulo.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WhatsThisCursor))
            form.addRow(titulo, editor)
            cuerpo += [titulo, editor]
        def plegar(visible):
            for widget in cuerpo:
                widget.setVisible(visible)
            # Sin el tope, un bloque cerrado deja el marco vacio ocupando sitio.
            caja.setMaximumHeight(16777215 if visible else 26)

        caja.toggled.connect(plegar)
        plegar(abierto)
        return caja

    def _crear_editor(self, grupo, clave, tipo, opciones):
        if tipo == "b":
            editor = QtWidgets.QCheckBox()
            editor.stateChanged.connect(self._programar_preview)
            return editor
        if tipo == "c":
            editor = QtWidgets.QComboBox()
            editor.addItems(tuple(opciones))
            editor.currentTextChanged.connect(self._programar_preview)
            return editor
        editor = QtWidgets.QLineEdit()
        editor.setMinimumWidth(120)
        if tipo == "i":
            editor.setValidator(_IntValidador(editor))
        else:
            editor.setValidator(_FloatValidador(editor))
        if clave == "dn_max":
            editor.setPlaceholderText("inf")
        editor.textEdited.connect(self._programar_preview)
        editor.editingFinished.connect(self._previsualizar)
        return editor

    def _grupo_calidad(self):
        """Veredicto de la malla. Fuera de las pestanas: siempre a la vista."""
        caja = QtWidgets.QGroupBox("Calidad de malla")
        grid = QtWidgets.QGridLayout(caja)
        grid.setContentsMargins(10, 6, 10, 8)
        grid.setVerticalSpacing(3)
        claves = (
            ("n_celdas", "Celdas", "Celdas de la malla."),
            ("j_negativos", "J<=0", "Celdas plegadas. Cualquiera bloquea la malla."),
            ("ortogonalidad_pared_min", "Ort. pared",
             "Angulo minimo entre la pared y la primera capa, en grados. "
             "Bloquea por debajo de 30, avisa por debajo de 70."),
            ("ortogonalidad_p5", "Ort. p5",
             "Percentil 5 de la ortogonalidad en toda la malla, en grados."),
            ("crecimiento_max", "Crec. max",
             "Mayor salto de espaciado normal entre capas vecinas. Bloquea en 2.0."),
            ("aspecto_max", "Aspecto max",
             "Relacion de aspecto mayor. Alta junto a la pared es el OBJETIVO de "
             "la malla, no un defecto: con y+~1 sale de 10000."),
            ("oblicuidad_p99", "Oblic. p99",
             "Cuanto pesan los terminos cruzados del laplaciano curvilineo. "
             "Decide si la correccion diferida converge. Bloquea en 0.60."),
            ("dn_sobre_radio_le", "dn/r LE",
             "Paso de pared dividido por el radio del morro. Por encima de 0.5 la "
             "primera celda envuelve el borde de ataque entero."),
            ("cola_rel", "Cola TE [c]",
             "Longitud de la cola de cierre del borde de salida romo, en cuerdas. "
             "Es geometria inventada: avisa por encima del 1 % de cuerda."),
            ("div_uinf_rel", "div uniforme",
             "Divergencia discreta de una corriente uniforme. Tiene que ser cero "
             "en maquina; si no, hay un error de metrica."),
        )
        for indice, (clave, etiqueta, ayuda) in enumerate(claves):
            fila, columna = divmod(indice, 2)
            label = QtWidgets.QLabel("-")
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            self._calidad_labels[clave] = label
            titulo = QtWidgets.QLabel(etiqueta)
            for widget in (titulo, label):
                widget.setToolTip(ayuda)
            grid.addWidget(titulo, fila, 2 * columna)
            grid.addWidget(label, fila, 2 * columna + 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        self._avisos = QtWidgets.QLabel("-")
        self._avisos.setWordWrap(True)
        fila = (len(claves) + 1) // 2
        grid.addWidget(QtWidgets.QLabel("Avisos"), fila, 0)
        grid.addWidget(self._avisos, fila, 1, 1, 3)
        return caja

    def _grupo_metricas(self):
        caja = QtWidgets.QGroupBox("Simulacion")
        grid = QtWidgets.QGridLayout(caja)
        grid.setContentsMargins(10, 6, 10, 8)
        grid.setVerticalSpacing(3)
        for indice, (clave, etiqueta) in enumerate((
                ("iter", "Iteracion"),
                ("it_s", "Iter/s"),
                ("Cl", "Cl"),
                ("Cd", "Cd"),
                ("div", "Divergencia"),
                ("speed_max", "|u|max"))):
            fila, columna = divmod(indice, 2)
            label = QtWidgets.QLabel("-")
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            self._metricas_labels[clave] = label
            grid.addWidget(QtWidgets.QLabel(etiqueta), fila, 2 * columna)
            grid.addWidget(label, fila, 2 * columna + 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        return caja

    def _crear_barra_visualizacion(self):
        barra = QtWidgets.QHBoxLayout()
        barra.addWidget(QtWidgets.QLabel("Campo"))
        self._modo = QtWidgets.QComboBox()
        self._modo.addItems(("velocidad", "presion", "u", "v"))
        self._modo.currentTextChanged.connect(self._modo_campo)
        barra.addWidget(self._modo)
        barra.addSpacing(12)
        barra.addWidget(QtWidgets.QLabel("Vista"))
        self._vista = QtWidgets.QComboBox()
        self._vista.addItems(("perfil", "estela", "dominio"))
        self._vista.currentTextChanged.connect(self._vista_cambiada)
        barra.addWidget(self._vista)
        barra.addSpacing(12)
        self._vista_malla = QtWidgets.QPushButton("Ver malla")
        self._vista_malla.clicked.connect(self._previsualizar)
        barra.addWidget(self._vista_malla)
        barra.addStretch(1)
        return barra

    def _grupo_de_clave(self, clave):
        for grupo in ("malla", "solver", "ejecucion"):
            if clave in self.caso.__dict__[grupo]:
                return grupo
        return None

    def _sincronizar_ui(self):
        self._actualizando_ui = True
        self._perfil.setCurrentText(self.caso.perfil)
        for (grupo, clave), editor in self._campos.items():
            valor = self.caso.__dict__[grupo][clave]
            if isinstance(editor, QtWidgets.QCheckBox):
                editor.setChecked(bool(valor))
            elif isinstance(editor, QtWidgets.QComboBox):
                editor.setCurrentText(str(valor))
            else:
                editor.setText(_formatear_valor(valor))
        self._actualizando_ui = False

    def _leer_formulario(self):
        caso = Caso()
        caso.perfil = self._perfil.currentText()
        for grupo in ("malla", "solver", "ejecucion"):
            caso.__dict__[grupo].update(self.caso.__dict__[grupo])
        for (grupo, clave), editor in self._campos.items():
            actual = caso.__dict__[grupo][clave]
            if isinstance(editor, QtWidgets.QCheckBox):
                valor = editor.isChecked()
            elif isinstance(editor, QtWidgets.QComboBox):
                valor = editor.currentText()
            else:
                texto = editor.text().strip()
                if clave == "dn_max" and texto.lower() in {"", "inf", "infinito", "none"}:
                    valor = None
                elif isinstance(actual, int) and not isinstance(actual, bool):
                    valor = int(float(texto))
                else:
                    valor = float(texto)
            caso.__dict__[grupo][clave] = valor
        return caso

    def _aplicar_formulario(self):
        try:
            caso = self._leer_formulario()
            caso.validar(ROOT)
        except (ValueError, CasoError) as error:
            self._estado.setText(f"Parametro invalido: {error}")
            return False
        self.caso = caso
        return True

    def _programar_preview(self):
        if self._actualizando_ui:
            return
        self._preview_timer.start(550)

    def _perfil_cambiado(self):
        if self._actualizando_ui:
            return
        self.caso.perfil = self._perfil.currentText()
        self._programar_preview()

    def _previsualizar(self):
        self._preview_timer.stop()
        if not self._aplicar_formulario():
            return
        self._estado.setText("Generando preview de malla...")
        QtWidgets.QApplication.processEvents()
        try:
            X, Y, info, calidad = previsualizar_malla(self.caso, ROOT)
        except (CasoError, ValueError) as error:
            self._estado.setText(str(error))
            return
        self.visor.mostrar_malla(X, Y, info, calidad)
        self._actualizar_calidad(X, Y, info, calidad)

    def _actualizar_calidad(self, X, Y, info, calidad):
        for clave, label in self._calidad_labels.items():
            label.setText(_formatear_metrica(calidad.get(clave)))
        mensajes = calidad.get("fallos") or calidad.get("avisos") or ["sin avisos"]
        self._avisos.setText("; ".join(mensajes))
        veredicto = "valida" if calidad.get("valida") else "bloqueada"
        if calidad.get("buena"):
            veredicto = "buena"
        i0, i1 = info.get("perfil", (0, X.shape[1] - 1))
        self._estado.setText(
            f"Malla {veredicto}: {X.shape[1] - 1} x {X.shape[0] - 1} celdas | "
            f"perfil i={i0}:{i1} | avisos {len(calidad.get('avisos', []))}"
        )

    def _aplicar_preset(self, valores):
        self.caso.malla.update(valores)
        self._sincronizar_ui()
        self._previsualizar()

    def _abrir_caso(self):
        ruta, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Abrir caso", str(ROOT / "runs"), "Casos (*.json);;Todos (*)")
        if not ruta:
            return
        try:
            self.caso = Caso.cargar(ruta)
        except (CasoError, ValueError, OSError) as error:
            self._estado.setText(f"No se pudo abrir el caso: {error}")
            return
        self._sincronizar_ui()
        self._previsualizar()

    def _guardar_caso(self):
        if not self._aplicar_formulario():
            return
        ruta, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Guardar caso", str(ROOT / "runs" / "caso.json"),
            "Casos (*.json);;Todos (*)")
        if not ruta:
            return
        try:
            self.caso.guardar(ruta)
        except (CasoError, OSError) as error:
            self._estado.setText(f"No se pudo guardar el caso: {error}")
            return
        self._estado.setText(f"Caso guardado: {ruta}")

    def _lanzar(self):
        if not self._aplicar_formulario():
            return
        salida = ROOT / "runs" / datetime.now().strftime("gui_%Y%m%d_%H%M%S")
        try:
            self.caso.validar(ROOT)
            self.grafica.reiniciar()
            self.runner.lanzar(self.caso, salida)
            self._boton_lanzar.setEnabled(False)
            self._boton_parar.setEnabled(True)
        except (CasoError, RuntimeError) as error:
            self._log(f"[GUI] {error}")

    def _terminado(self, codigo):
        self._log(f"[GUI] fin: {codigo}")
        self._boton_lanzar.setEnabled(True)
        self._boton_parar.setEnabled(False)

    def _modo_campo(self, modo):
        self.visor.modo = modo
        self.visor.redibujar_ultimo()

    def _vista_cambiada(self, vista):
        self.visor.vista = vista
        self.visor.redibujar_ultimo()

    def _campo(self, snapshot):
        self.visor.mostrar_snapshot(snapshot)

    def _progreso(self, datos):
        if "Cl" in datos:
            self.grafica.actualizar(datos)
        for clave, label in self._metricas_labels.items():
            if clave in datos:
                label.setText(_formatear_metrica(datos[clave]))
        if "Cl" in datos:
            self._estado.setText("iter %s | %.2f it/s | Cl %.5f | Cd %.5f | div %.2e" %
                                 (datos.get("iter", 0), datos.get("it_s", 0.0),
                                  datos["Cl"], datos["Cd"], datos.get("div", 0.0)))

    def _log(self, texto):
        self._log_view.appendPlainText(texto)

    def closeEvent(self, event):
        self.runner.close()
        super().closeEvent(event)


def _FloatValidador(parent):
    expr = QtCore.QRegularExpression(r"^\s*(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|inf|infinito|none)?\s*$")
    return QtGui.QRegularExpressionValidator(expr, parent)


def _IntValidador(parent):
    expr = QtCore.QRegularExpression(r"^\s*[-+]?\d*\s*$")
    return QtGui.QRegularExpressionValidator(expr, parent)


def _formatear_valor(valor):
    if valor is None:
        return ""
    if isinstance(valor, float):
        return f"{valor:.8g}"
    return str(valor)


def _formatear_metrica(valor):
    if valor is None:
        return "-"
    if isinstance(valor, int):
        return f"{valor:d}"
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return str(valor)
    if abs(numero) >= 1.0e4 or (0 < abs(numero) < 1.0e-3):
        return f"{numero:.3e}"
    return f"{numero:.5g}"


def main():
    app = QtWidgets.QApplication(sys.argv)
    ventana = Ventana()
    ventana.show()
    return app.exec()

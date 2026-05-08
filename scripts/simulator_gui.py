#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""simulator_gui.py — Professional GUI launcher for Simulador2D.py"""

import sys
import os
import re
import json
import struct
import time
import tempfile
import subprocess
from pathlib import Path

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QSplitter, QScrollArea, QGroupBox, QLabel, QLineEdit,
    QCheckBox, QComboBox, QPushButton, QFileDialog, QTextEdit,
    QProgressBar, QFrame, QSizePolicy, QStatusBar, QSpinBox, QTabWidget,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize, QObject
from PyQt6.QtGui import QColor, QFont, QPalette, QIcon, QPixmap, QPainter, QImage

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

try:
    from multiprocessing import shared_memory as shm_mod
    SHM_AVAILABLE = True
except ImportError:
    SHM_AVAILABLE = False

SHM_NAME  = "sim2d_live"
META_NAME = "sim2d_meta"
SIM_DIR      = Path(__file__).parent
SETTINGS_FILE = SIM_DIR / "gui_settings.json"

# ─────────────────────────────  PALETTE  ────────────────────────────────────
C = dict(
    bg        = "#070b12",
    panel     = "#0c1018",
    panel2    = "#101520",
    border    = "#182030",
    border2   = "#1e2d42",
    accent    = "#00d4ff",
    accent2   = "#0099bb",
    amber     = "#f0a500",
    green     = "#22d47a",
    red       = "#e84040",
    text      = "#dce8f5",
    text2     = "#7a9ab8",
    text3     = "#3d556e",
    input_bg  = "#090d16",
    hover     = "#162030",
    select    = "#1a3050",
)

QSS = f"""
* {{ outline: 0; }}
QMainWindow, QWidget {{ background:{C['bg']}; color:{C['text']}; font:12px "Segoe UI"; }}
QScrollArea {{ border:none; background:transparent; }}
QScrollBar:vertical {{ background:{C['panel']}; width:5px; border-radius:2px; margin:0; }}
QScrollBar::handle:vertical {{ background:{C['border2']}; border-radius:2px; min-height:20px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
QScrollBar:horizontal {{ height:5px; }}

/* group box sections */
QGroupBox {{
    border:1px solid {C['border']}; border-radius:4px; margin-top:10px;
    padding:14px 6px 6px 6px; background:{C['panel']};
    font:11px "Segoe UI"; font-weight:600; color:{C['text2']};
}}
QGroupBox::title {{
    subcontrol-origin:margin; subcontrol-position:top left;
    left:10px; padding:0 4px; color:{C['accent']}; font-size:10px; letter-spacing:1.5px;
}}

QLabel {{ color:{C['text2']}; font-size:11px; }}
QLabel#metric_val {{ color:{C['text']}; font:bold 22px "Consolas"; }}
QLabel#metric_sub {{ color:{C['text3']}; font-size:10px; letter-spacing:1px; }}
QLabel#section_title {{ color:{C['accent']}; font:600 10px "Segoe UI"; letter-spacing:2px; padding:4px 0 2px 0; }}

QLineEdit, QSpinBox, QComboBox {{
    background:{C['input_bg']}; border:1px solid {C['border']};
    border-radius:3px; color:{C['text']}; padding:3px 6px;
    font:12px "Consolas"; selection-background-color:{C['select']};
}}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{ border:1px solid {C['accent2']}; }}
QLineEdit[readOnly="true"] {{ color:{C['text2']}; background:{C['panel']}; }}

QComboBox::drop-down {{ border:none; width:18px; }}
QComboBox::down-arrow {{
    border-left:4px solid transparent; border-right:4px solid transparent;
    border-top:5px solid {C['text2']}; margin-right:4px;
}}
QComboBox QAbstractItemView {{
    background:{C['panel2']}; border:1px solid {C['border2']};
    color:{C['text']}; selection-background-color:{C['select']};
}}

QSpinBox::up-button, QSpinBox::down-button {{ border:none; background:transparent; width:14px; }}
QSpinBox::up-arrow {{ border-left:4px solid transparent; border-right:4px solid transparent; border-bottom:5px solid {C['text2']}; }}
QSpinBox::down-arrow {{ border-left:4px solid transparent; border-right:4px solid transparent; border-top:5px solid {C['text2']}; }}

QCheckBox {{ color:{C['text']}; font-size:11px; spacing:6px; }}
QCheckBox::indicator {{ width:13px; height:13px; border:1px solid {C['border2']}; border-radius:2px; background:{C['input_bg']}; }}
QCheckBox::indicator:checked {{ background:{C['accent']}; border-color:{C['accent']}; image:none; }}

QPushButton {{
    border:1px solid {C['border2']}; border-radius:3px; padding:6px 14px;
    background:{C['panel2']}; color:{C['text']}; font-size:12px; font-weight:500;
}}
QPushButton:hover {{ background:{C['hover']}; border-color:{C['accent2']}; }}
QPushButton:pressed {{ background:{C['select']}; }}
QPushButton#btn_run {{
    background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #0b96af,stop:1 #077a91);
    border-color:{C['accent']}; color:#fff; font-weight:700; font-size:13px;
}}
QPushButton#btn_run:hover {{
    background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #10b8d4,stop:1 #0b96af);
}}
QPushButton#btn_stop {{
    background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #b82020,stop:1 #8c1818);
    border-color:{C['red']}; color:#fff; font-weight:700; font-size:13px;
}}
QPushButton#btn_stop:hover {{
    background:qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #d43030,stop:1 #b82020);
}}
QPushButton:disabled {{ color:{C['text3']}; border-color:{C['border']}; background:{C['panel']}; }}

QTextEdit {{
    background:{C['panel']}; border:1px solid {C['border']};
    color:{C['text2']}; font:11px "Consolas"; border-radius:3px;
}}
QProgressBar {{
    border:1px solid {C['border']}; border-radius:3px; background:{C['input_bg']};
    text-align:center; color:{C['text']}; font:11px "Consolas"; max-height:14px;
}}
QProgressBar::chunk {{
    background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {C['accent2']},stop:1 {C['accent']});
    border-radius:2px;
}}
QSplitter::handle {{ background:{C['border']}; }}
QSplitter::handle:horizontal {{ width:2px; }}
QSplitter::handle:vertical {{ height:2px; }}
QStatusBar {{ background:{C['panel']}; border-top:1px solid {C['border']}; color:{C['text2']}; font:11px "Consolas"; }}
QStatusBar::item {{ border:none; }}
QTabWidget::pane {{ border:1px solid {C['border']}; background:{C['bg']}; top:-1px; }}
QTabBar {{ background:{C['panel']}; }}
QTabBar::tab {{
    background:{C['panel2']}; color:{C['text2']}; border:1px solid {C['border']};
    border-bottom:none; padding:7px 20px; font:600 11px "Segoe UI"; letter-spacing:1px; min-width:150px;
}}
QTabBar::tab:selected {{ background:{C['bg']}; color:{C['accent']}; border-bottom:2px solid {C['accent']}; }}
QTabBar::tab:hover:!selected {{ background:{C['hover']}; color:{C['text']}; }}
"""

# ─────────────────────────────  HELPERS  ────────────────────────────────────

def sep():
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"background:{C['border']}; max-height:1px; border:none;")
    return f


def lbl(text, bold=False, color=None):
    l = QLabel(text)
    style = f"font-size:11px;"
    if bold:
        style += "font-weight:600;"
    if color:
        style += f"color:{color};"
    l.setStyleSheet(style)
    return l


def float_edit(default, placeholder=""):
    e = QLineEdit(str(default) if default is not None else "")
    if placeholder:
        e.setPlaceholderText(placeholder)
    e.setFixedHeight(24)
    return e


def int_spin(default, min_=0, max_=99999, step=1):
    s = QSpinBox()
    s.setRange(min_, max_)
    s.setValue(default)
    s.setSingleStep(step)
    s.setFixedHeight(24)
    return s


def make_row(label_text, widget, unit=""):
    row = QHBoxLayout()
    row.setContentsMargins(0, 1, 0, 1)
    l = QLabel(label_text)
    l.setFixedWidth(145)
    l.setStyleSheet(f"color:{C['text2']}; font-size:11px;")
    row.addWidget(l)
    row.addWidget(widget, 1)
    if unit:
        u = QLabel(unit)
        u.setStyleSheet(f"color:{C['text3']}; font-size:10px;")
        u.setFixedWidth(30)
        row.addWidget(u)
    return row


def make_check_row(label_text, widget):
    row = QHBoxLayout()
    row.setContentsMargins(0, 1, 0, 1)
    widget.setText(label_text)
    row.addWidget(widget)
    row.addStretch()
    return row


# ─────────────────────────────  COLLAPSIBLE SECTION  ────────────────────────

class CollapsibleSection(QWidget):
    def __init__(self, title, parent=None, collapsed=False):
        super().__init__(parent)
        self._collapsed = collapsed
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 4, 0, 0)
        outer.setSpacing(0)

        self.toggle_btn = QPushButton(f"  {'▶' if collapsed else '▼'}  {title}")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(not collapsed)
        self.toggle_btn.setFixedHeight(26)
        self.toggle_btn.setStyleSheet(f"""
            QPushButton {{
                text-align:left; border:none; border-bottom:1px solid {C['border']};
                background:{C['panel2']}; color:{C['text']}; font:600 11px "Segoe UI";
                letter-spacing:1px; padding-left:8px;
            }}
            QPushButton:hover {{ background:{C['hover']}; }}
            QPushButton:checked {{ color:{C['accent']}; }}
        """)
        self.toggle_btn.clicked.connect(self._toggle)

        self.content = QWidget()
        self.content.setVisible(not collapsed)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(8, 6, 4, 8)
        self.content_layout.setSpacing(3)

        outer.addWidget(self.toggle_btn)
        outer.addWidget(self.content)

    def _toggle(self):
        self._collapsed = not self._collapsed
        self.content.setVisible(not self._collapsed)
        arrow = '▶' if self._collapsed else '▼'
        txt = self.toggle_btn.text()
        import re
        new_txt = re.sub(r'^  [▶▼]  ', f'  {arrow}  ', txt)
        self.toggle_btn.setText(new_txt)

    def add_row(self, layout_or_widget):
        if isinstance(layout_or_widget, QWidget):
            self.content_layout.addWidget(layout_or_widget)
        else:
            self.content_layout.addLayout(layout_or_widget)


# ─────────────────────────────  PARAMETER PANEL  ────────────────────────────

class ParameterPanel(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumWidth(290)
        self.setMaximumWidth(340)

        container = QWidget()
        self.setWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 8)
        layout.setSpacing(2)

        # ── Perfil Aerodinámico ───────────────────────────────────────────
        s = CollapsibleSection("PERFIL AERODINÁMICO")
        self.e_filepath = float_edit("profiles/AG24", "nombre archivo")
        btn_browse = QPushButton("…")
        btn_browse.setFixedWidth(28)
        btn_browse.setFixedHeight(24)
        btn_browse.clicked.connect(self._browse_profile)
        row_fp = QHBoxLayout()
        row_fp.setContentsMargins(0,0,0,0)
        lbl_fp = QLabel("Perfil (filepath)")
        lbl_fp.setFixedWidth(145)
        lbl_fp.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        row_fp.addWidget(lbl_fp)
        row_fp.addWidget(self.e_filepath, 1)
        row_fp.addWidget(btn_browse)
        s.add_row(row_fp)
        self.e_chord   = float_edit(1.0);  s.add_row(make_row("Cuerda (chord)", self.e_chord, "m"))
        self.e_alpha   = float_edit(5.0);  s.add_row(make_row("Ángulo ataque (α)", self.e_alpha, "°"))
        layout.addWidget(s)

        # ── Dominio ───────────────────────────────────────────────────────
        s = CollapsibleSection("DOMINIO")
        self.e_Lx = float_edit(7.0);  s.add_row(make_row("Lx", self.e_Lx, "m"))
        self.e_Ly = float_edit(6.0);  s.add_row(make_row("Ly", self.e_Ly, "m"))
        self.e_cx = float_edit(2.0);  s.add_row(make_row("cx (pos. perfil x)", self.e_cx, "m"))
        self.e_cy = float_edit("",  "auto"); s.add_row(make_row("cy (pos. perfil y)", self.e_cy, "m"))
        layout.addWidget(s)

        # ── Fluido ────────────────────────────────────────────────────────
        s = CollapsibleSection("FLUIDO")
        self.e_rho  = float_edit(1.225);   s.add_row(make_row("Densidad ρ", self.e_rho, "kg/m³"))
        self.e_nu   = float_edit(1.5e-5);  s.add_row(make_row("Viscosidad ν", self.e_nu, "m²/s"))
        self.e_v0x  = float_edit(5.0);     s.add_row(make_row("Velocidad v₀ₓ", self.e_v0x, "m/s"))
        self.e_v0y  = float_edit(0.0);     s.add_row(make_row("Velocidad v₀ᵧ", self.e_v0y, "m/s"))
        self.e_p0   = float_edit(0.0);     s.add_row(make_row("Presión p₀", self.e_p0, "Pa"))
        self.e_div  = float_edit(1e-1);    s.add_row(make_row("Divergencia tol.", self.e_div, ""))
        layout.addWidget(s)

        # ── Malla ─────────────────────────────────────────────────────────
        s = CollapsibleSection("MALLA")
        self.e_dx_min = float_edit(0.01);   s.add_row(make_row("dx_min", self.e_dx_min, "m"))
        self.e_dy_min = float_edit("", "= dx_min"); s.add_row(make_row("dy_min", self.e_dy_min, "m"))
        self.e_fexp   = float_edit(1.05);   s.add_row(make_row("factor_expansion", self.e_fexp, ""))
        self.e_azfx   = float_edit("", "auto"); s.add_row(make_row("ancho_zona_fina_x", self.e_azfx, "m"))
        self.e_azfy   = float_edit("", "auto"); s.add_row(make_row("ancho_zona_fina_y", self.e_azfy, "m"))
        self.e_ratio  = int_spin(50, 1, 500); s.add_row(make_row("ratio_max_malla", self.e_ratio, ""))
        self.e_dx_max = float_edit("", "auto"); s.add_row(make_row("dx_max", self.e_dx_max, "m"))
        self.e_dy_max = float_edit("", "auto"); s.add_row(make_row("dy_max", self.e_dy_max, "m"))
        self.cb_wale  = QCheckBox(); s.add_row(make_check_row("Turbulencia WALE", self.cb_wale))
        layout.addWidget(s)

        # ── Simulación ────────────────────────────────────────────────────
        s = CollapsibleSection("SIMULACIÓN")
        self.e_CFL    = float_edit(0.8);   s.add_row(make_row("CFL", self.e_CFL, ""))
        self.e_iters  = int_spin(2000, 1, 500000); s.add_row(make_row("Iteraciones", self.e_iters, ""))
        self.e_guard  = int_spin(50, 1, 10000);    s.add_row(make_row("Guardado cada", self.e_guard, "iter"))
        self.cb_stop  = QCheckBox(); self.cb_stop.setChecked(True)
        s.add_row(make_check_row("Parar en convergencia", self.cb_stop))
        layout.addWidget(s)

        # ── Condiciones de Frontera ───────────────────────────────────────
        s = CollapsibleSection("CONDICIONES DE FRONTERA", collapsed=True)
        bc_types = ["inflow", "outflow", "slip", "noslip"]
        for side, default_type, default_val in [
            ("Izquierda (left)",  "inflow",  ""),
            ("Derecha (right)",   "outflow", "0.0"),
            ("Arriba (top)",      "slip",    ""),
            ("Abajo (bottom)",    "slip",    ""),
        ]:
            combo = QComboBox()
            combo.addItems(bc_types)
            combo.setCurrentText(default_type)
            combo.setFixedHeight(24)
            val_edit = float_edit(default_val, "valor (opt)")
            val_edit.setFixedWidth(70)
            row = QHBoxLayout()
            row.setContentsMargins(0,1,0,1)
            l = QLabel(side)
            l.setFixedWidth(115)
            l.setStyleSheet(f"color:{C['text2']};font-size:11px;")
            row.addWidget(l)
            row.addWidget(combo)
            row.addWidget(val_edit)
            s.add_row(row)
            attr = side.split("(")[1].rstrip(")")
            setattr(self, f"combo_bc_{attr}", combo)
            setattr(self, f"e_bc_{attr}_val", val_edit)
        layout.addWidget(s)

        # ── Multigrid ─────────────────────────────────────────────────────
        s = CollapsibleSection("MULTIGRID / PROYECCIÓN", collapsed=True)
        self.e_mg_outer   = int_spin(8, 1, 50);  s.add_row(make_row("mg_max_outer", self.e_mg_outer, ""))
        self.e_mg_cycles  = int_spin(5, 1, 50);  s.add_row(make_row("mg_cycles_per_outer", self.e_mg_cycles, ""))
        self.e_mg_pre     = int_spin(3, 0, 20);  s.add_row(make_row("mg_pre_suavizado", self.e_mg_pre, ""))
        self.e_mg_post    = int_spin(3, 0, 20);  s.add_row(make_row("mg_post_suavizado", self.e_mg_post, ""))
        self.cb_mg_turbo  = QCheckBox(); s.add_row(make_check_row("mg_modo_turbo", self.cb_mg_turbo))
        self.cb_mg_rapido = QCheckBox(); s.add_row(make_check_row("mg_modo_rapido", self.cb_mg_rapido))
        self.cb_mg_roll   = QCheckBox(); self.cb_mg_roll.setChecked(True)
        s.add_row(make_check_row("mg_rollback_on_nan", self.cb_mg_roll))
        self.cb_mg_ibm    = QCheckBox(); self.cb_mg_ibm.setChecked(True)
        s.add_row(make_check_row("mg_apply_ibm_each_outer", self.cb_mg_ibm))
        self.cb_mg_guard  = QCheckBox(); self.cb_mg_guard.setChecked(True)
        s.add_row(make_check_row("mg_guard_residual_every_outer", self.cb_mg_guard))
        self.cb_mg_adapt  = QCheckBox()
        s.add_row(make_check_row("mg_adaptive_outer0_cycles", self.cb_mg_adapt))
        self.cb_mg_divaft = QCheckBox(); self.cb_mg_divaft.setChecked(True)
        s.add_row(make_check_row("mg_compute_div_after", self.cb_mg_divaft))
        layout.addWidget(s)

        # ── Visualización ─────────────────────────────────────────────────
        s = CollapsibleSection("VISUALIZACIÓN / GUARDADO", collapsed=True)
        self.cb_graficos   = QCheckBox(); s.add_row(make_check_row("graficos (matplotlib al final)", self.cb_graficos))
        self.cb_save_frames= QCheckBox(); s.add_row(make_check_row("save_frames", self.cb_save_frames))
        self.e_frames_dir  = float_edit("", "directorio frames")
        btn_fd = QPushButton("…"); btn_fd.setFixedWidth(28); btn_fd.setFixedHeight(24)
        btn_fd.clicked.connect(self._browse_frames_dir)
        row_fd = QHBoxLayout(); row_fd.setContentsMargins(0,0,0,0)
        lf = QLabel("frames_dir"); lf.setFixedWidth(100); lf.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        row_fd.addWidget(lf); row_fd.addWidget(self.e_frames_dir, 1); row_fd.addWidget(btn_fd)
        s.add_row(row_fd)
        self.cb_live   = QCheckBox(); s.add_row(make_check_row("live_view (viewer externo)", self.cb_live))
        self.cb_malla  = QCheckBox(); s.add_row(make_check_row("mostrar_malla", self.cb_malla))
        layout.addWidget(s)

        # ── Análisis Polar ────────────────────────────────────────────────
        s = CollapsibleSection("ANÁLISIS POLAR", collapsed=True)
        self.e_polar_plan   = float_edit("", "ej: -5,-3,0,3,5,7,9")
        s.add_row(make_row("plan_polar (ángulos)", self.e_polar_plan, ""))
        self.e_polar_desc   = float_edit(0.3)
        s.add_row(make_row("polar_descarte (fracción)", self.e_polar_desc, ""))
        layout.addWidget(s)

        # ── Corrección Deriva ─────────────────────────────────────────────
        s = CollapsibleSection("CORRECCIÓN DERIVA VERTICAL", collapsed=True)
        self.cb_corr_der   = QCheckBox(); self.cb_corr_der.setChecked(True)
        s.add_row(make_check_row("corregir_deriva_vertical", self.cb_corr_der))
        self.e_umbral_der  = float_edit(1e-8)
        s.add_row(make_row("umbral_deriva_vertical", self.e_umbral_der, ""))
        self.e_corr_cada   = int_spin(1, 1, 100)
        s.add_row(make_row("corregir_deriva_cada", self.e_corr_cada, "iter"))
        self.e_factor_der  = float_edit(0.1)
        s.add_row(make_row("factor_deriva_vertical", self.e_factor_der, ""))
        layout.addWidget(s)

        # ── Diagnóstico ───────────────────────────────────────────────────
        s = CollapsibleSection("DIAGNÓSTICO", collapsed=True)
        self.cb_debug_spikes = QCheckBox(); s.add_row(make_check_row("debug_spikes", self.cb_debug_spikes))
        self.cb_diag_fuerzas = QCheckBox(); s.add_row(make_check_row("diagnostico_fuerzas", self.cb_diag_fuerzas))
        self.e_diag_bins     = int_spin(30, 1, 200)
        s.add_row(make_row("diagnostico_fuerzas_bins", self.e_diag_bins, ""))
        self.e_diag_te       = float_edit(0.85)
        s.add_row(make_row("diagnostico_fuerzas_te_start", self.e_diag_te, ""))
        self.e_diag_cada     = int_spin(500, 1, 10000)
        s.add_row(make_row("diagnostico_fuerzas_cada", self.e_diag_cada, "iter"))
        self.cb_diag_plot    = QCheckBox(); s.add_row(make_check_row("diagnostico_fuerzas_plot", self.cb_diag_plot))
        layout.addWidget(s)

        layout.addStretch()

    def _browse_profile(self):
        f, _ = QFileDialog.getOpenFileName(self, "Seleccionar perfil",
                                           str(SIM_DIR), "Todos (*)")
        if f:
            name = Path(f).stem
            self.e_filepath.setText(name)

    def _browse_frames_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Directorio frames", str(SIM_DIR))
        if d:
            self.e_frames_dir.setText(d)

    # ── param extraction ──────────────────────────────────────────────────

    def _f(self, w, default=None):
        txt = w.text().strip()
        if txt == "" or txt.lower() == "none":
            return default
        try:
            return float(txt)
        except ValueError:
            return default

    def _bc(self, side):
        combo = getattr(self, f"combo_bc_{side}")
        val_e = getattr(self, f"e_bc_{side}_val")
        bc_type = combo.currentText()
        val_txt = val_e.text().strip()
        val = float(val_txt) if val_txt else None
        return (bc_type, val)

    def set_params(self, p):
        def sf(widget, val):
            widget.setText("" if val is None else str(val))

        sf(self.e_filepath,  p.get('filepath', 'profiles/AG24'))
        sf(self.e_chord,     p.get('chord', 1.0))
        sf(self.e_alpha,     p.get('alpha_deg', 5.0))
        sf(self.e_Lx,        p.get('Lx', 7.0))
        sf(self.e_Ly,        p.get('Ly', 6.0))
        sf(self.e_cx,        p.get('cx', 2.0))
        sf(self.e_cy,        p.get('cy', ''))
        sf(self.e_rho,       p.get('rho', 1.225))
        sf(self.e_nu,        p.get('nu', 1.5e-5))
        sf(self.e_v0x,       p.get('v0x', 5.0))
        sf(self.e_v0y,       p.get('v0y', 0.0))
        sf(self.e_p0,        p.get('p0', 0.0))
        sf(self.e_div,       p.get('divergencia', 0.1))
        sf(self.e_dx_min,    p.get('dx_min', 0.01))
        sf(self.e_dy_min,    p.get('dy_min', ''))
        sf(self.e_fexp,      p.get('factor_expansion', 1.05))
        sf(self.e_azfx,      p.get('ancho_zona_fina_x', ''))
        sf(self.e_azfy,      p.get('ancho_zona_fina_y', ''))
        sf(self.e_dx_max,    p.get('dx_max', ''))
        sf(self.e_dy_max,    p.get('dy_max', ''))
        self.e_ratio.setValue(int(p.get('ratio_max_malla', 50)))
        self.cb_wale.setChecked(bool(p.get('usar_wale', False)))
        sf(self.e_CFL,       p.get('CFL', 0.8))
        self.e_iters.setValue(int(p.get('iteraciones', 2000)))
        self.e_guard.setValue(int(p.get('guardado', 50)))
        self.cb_stop.setChecked(bool(p.get('stop_on_convergence', True)))

        for side in ('left', 'right', 'top', 'bottom'):
            bc = p.get(f'boundary_{side}')
            if bc and isinstance(bc, (list, tuple)) and len(bc) == 2:
                getattr(self, f'combo_bc_{side}').setCurrentText(bc[0])
                getattr(self, f'e_bc_{side}_val').setText("" if bc[1] is None else str(bc[1]))

        self.e_mg_outer.setValue(int(p.get('mg_max_outer', 8)))
        self.e_mg_cycles.setValue(int(p.get('mg_cycles_per_outer', 5)))
        self.e_mg_pre.setValue(int(p.get('mg_pre_suavizado', 3)))
        self.e_mg_post.setValue(int(p.get('mg_post_suavizado', 3)))
        self.cb_mg_turbo.setChecked(bool(p.get('mg_modo_turbo', False)))
        self.cb_mg_rapido.setChecked(bool(p.get('mg_modo_rapido', False)))
        self.cb_mg_roll.setChecked(bool(p.get('mg_rollback_on_nan', True)))
        self.cb_mg_ibm.setChecked(bool(p.get('mg_apply_ibm_each_outer', True)))
        self.cb_mg_guard.setChecked(bool(p.get('mg_guard_residual_every_outer', True)))
        self.cb_mg_adapt.setChecked(bool(p.get('mg_adaptive_outer0_cycles', False)))
        self.cb_mg_divaft.setChecked(bool(p.get('mg_compute_div_after', True)))

        self.cb_graficos.setChecked(bool(p.get('graficos', False)))
        self.cb_save_frames.setChecked(bool(p.get('save_frames', False)))
        sf(self.e_frames_dir, p.get('frames_dir_grueso', ''))
        self.cb_live.setChecked(bool(p.get('live_view', False)))
        self.cb_malla.setChecked(bool(p.get('mostrar_malla', False)))

        plan = p.get('plan_polar')
        self.e_polar_plan.setText(",".join(str(a) for a in plan) if plan else "")
        sf(self.e_polar_desc, p.get('polar_descarte', 0.3))

        self.cb_corr_der.setChecked(bool(p.get('corregir_deriva_vertical', True)))
        sf(self.e_umbral_der, p.get('umbral_deriva_vertical', 1e-8))
        self.e_corr_cada.setValue(int(p.get('corregir_deriva_cada', 1)))
        sf(self.e_factor_der, p.get('factor_deriva_vertical', 0.1))

        self.cb_debug_spikes.setChecked(bool(p.get('debug_spikes', False)))
        self.cb_diag_fuerzas.setChecked(bool(p.get('diagnostico_fuerzas', False)))
        self.e_diag_bins.setValue(int(p.get('diagnostico_fuerzas_bins', 30)))
        sf(self.e_diag_te, p.get('diagnostico_fuerzas_te_start', 0.85))
        self.e_diag_cada.setValue(int(p.get('diagnostico_fuerzas_cada', 500)))
        self.cb_diag_plot.setChecked(bool(p.get('diagnostico_fuerzas_plot', False)))

    def get_params(self):
        polar_txt = self.e_polar_plan.text().strip()
        plan_polar = None
        if polar_txt:
            try:
                plan_polar = [float(x.strip()) for x in polar_txt.split(",")]
            except ValueError:
                plan_polar = None

        p = dict(
            CFL                          = self._f(self.e_CFL, 0.8),
            alpha_deg                    = self._f(self.e_alpha, 5.0),
            chord                        = self._f(self.e_chord, 1.0),
            filepath                     = self.e_filepath.text().strip() or "profiles/NACA_0012",
            Lx                           = self._f(self.e_Lx, 7.0),
            Ly                           = self._f(self.e_Ly, 6.0),
            dx_min                       = self._f(self.e_dx_min, 0.01),
            dy_min                       = self._f(self.e_dy_min, None),
            factor_expansion             = self._f(self.e_fexp, 1.05),
            ancho_zona_fina_x            = self._f(self.e_azfx, None),
            ancho_zona_fina_y            = self._f(self.e_azfy, None),
            ratio_max_malla              = self.e_ratio.value(),
            dx_max                       = self._f(self.e_dx_max, None),
            dy_max                       = self._f(self.e_dy_max, None),
            usar_wale                    = self.cb_wale.isChecked(),
            cx                           = self._f(self.e_cx, 2.0),
            cy                           = self._f(self.e_cy, None),
            p0                           = self._f(self.e_p0, 0.0),
            v0x                          = self._f(self.e_v0x, 5.0),
            v0y                          = self._f(self.e_v0y, 0.0),
            rho                          = self._f(self.e_rho, 1.225),
            nu                           = self._f(self.e_nu, 1.5e-5),
            divergencia                  = self._f(self.e_div, 1e-1),
            boundary_left                = self._bc("left"),
            boundary_right               = self._bc("right"),
            boundary_top                 = self._bc("top"),
            boundary_bottom              = self._bc("bottom"),
            guardado                     = self.e_guard.value(),
            iteraciones                  = self.e_iters.value(),
            mg_max_outer                 = self.e_mg_outer.value(),
            mg_cycles_per_outer          = self.e_mg_cycles.value(),
            mg_pre_suavizado             = self.e_mg_pre.value(),
            mg_post_suavizado            = self.e_mg_post.value(),
            mg_guard_residual_every_outer= self.cb_mg_guard.isChecked(),
            mg_adaptive_outer0_cycles    = self.cb_mg_adapt.isChecked(),
            mg_apply_ibm_each_outer      = self.cb_mg_ibm.isChecked(),
            mg_rollback_on_nan           = self.cb_mg_roll.isChecked(),
            mg_compute_div_after         = self.cb_mg_divaft.isChecked(),
            mg_modo_rapido               = self.cb_mg_rapido.isChecked(),
            mg_modo_turbo                = self.cb_mg_turbo.isChecked(),
            save_frames                  = self.cb_save_frames.isChecked(),
            frames_dir_grueso            = self.e_frames_dir.text().strip() or None,
            graficos                     = self.cb_graficos.isChecked(),
            stop_on_convergence          = self.cb_stop.isChecked(),
            plan_polar                   = plan_polar,
            polar_descarte               = self._f(self.e_polar_desc, 0.3),
            live_view                    = self.cb_live.isChecked(),
            mostrar_malla                = self.cb_malla.isChecked(),
            corregir_deriva_vertical     = self.cb_corr_der.isChecked(),
            umbral_deriva_vertical       = self._f(self.e_umbral_der, 1e-8),
            corregir_deriva_cada         = self.e_corr_cada.value(),
            factor_deriva_vertical       = self._f(self.e_factor_der, 0.1),
            debug_spikes                 = self.cb_debug_spikes.isChecked(),
            diagnostico_fuerzas          = self.cb_diag_fuerzas.isChecked(),
            diagnostico_fuerzas_bins     = self.e_diag_bins.value(),
            diagnostico_fuerzas_te_start = self._f(self.e_diag_te, 0.85),
            diagnostico_fuerzas_cada     = self.e_diag_cada.value(),
            diagnostico_fuerzas_plot     = self.cb_diag_plot.isChecked(),
        )
        return p


# ─────────────────────────────  VISUALIZATION CANVAS  ───────────────────────

class VisualizationCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(300)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        self.lbl_title = QLabel("Esperando simulación...")
        self.lbl_title.setStyleSheet(f"color:{C['text2']}; font-size:10px; padding:2px 4px;")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.lbl_title)

        self.field_label = QLabel()
        self.field_label.setMinimumHeight(350)
        self.field_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.field_label.setStyleSheet(
            f"background:{C['panel']}; border:1px solid {C['border2']}; border-radius:3px;"
        )
        v.addWidget(self.field_label, 5)

        self.fig = Figure(figsize=(8, 2.4), facecolor=C['bg'])
        self.coef_canvas = FigureCanvas(self.fig)
        self.coef_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.coef_canvas.setMinimumHeight(150)
        v.addWidget(self.coef_canvas, 2)

        self.ax_coefs = self.fig.add_subplot(111)
        self.ax_coefs.set_facecolor(C['panel'])
        self.ax_coefs.tick_params(colors=C['text3'], labelsize=8)
        for spine in self.ax_coefs.spines.values():
            spine.set_edgecolor(C['border2'])
        self.ax_coefs.set_xlabel("Iteración", color=C['text3'], fontsize=8)
        self.ax_coefs.set_ylabel("Coeficiente", color=C['text3'], fontsize=8)
        self.ax_coefs.grid(True, color=C['border2'], linewidth=0.5, alpha=0.8)

        self.line_cd, = self.ax_coefs.plot([], [], color='#ff5555', lw=1.2, label='Cd')
        self.line_cl, = self.ax_coefs.plot([], [], color='#55aaff', lw=1.2, label='Cl')
        self.ax_eff = self.ax_coefs.twinx()
        self.ax_eff.tick_params(colors=C['text3'], labelsize=7)
        self.ax_eff.set_ylabel('Cl/Cd', color='#77cc77', fontsize=8)
        self.line_eff, = self.ax_eff.plot([], [], color='#77cc77', lw=1, alpha=0.8, label='Cl/Cd', linestyle='--')
        self.ax_coefs.legend(fontsize=7, facecolor=C['panel2'], edgecolor=C['border'], labelcolor=C['text'])

        self._hist = dict(iter=[], cd=[], cl=[], eff=[])
        self._last_rgb = None
        self._palette = self._make_palette()
        self._x_1d = None
        self._y_1d = None
        self.coef_canvas.draw_idle()

    def set_coords(self, x_1d, y_1d):
        self._x_1d = np.asarray(x_1d, dtype=np.float32)
        self._y_1d = np.asarray(y_1d, dtype=np.float32)

    def _remap_to_physical(self, speed, solid):
        x_1d = self._x_1d
        y_1d = self._y_1d
        Lx = float(x_1d[-1])
        Ly = float(y_1d[-1])
        # Build target image with correct physical aspect ratio
        H = 600
        W = max(1, int(round(H * Lx / Ly)))
        W = min(W, 2400)
        xp = np.linspace(0.0, Lx, W, dtype=np.float32)
        yp = np.linspace(0.0, Ly, H, dtype=np.float32)
        ji = np.clip(np.searchsorted(x_1d, xp, side='right') - 1, 0, speed.shape[1] - 1)
        ii = np.clip(np.searchsorted(y_1d, yp, side='right') - 1, 0, speed.shape[0] - 1)
        return speed[np.ix_(ii, ji)], solid[np.ix_(ii, ji)]

    def _make_palette(self):
        # Paleta azul (lento) -> rojo (rapido) para render rapido en Qt.
        anchors = np.array([
            [0.0,   0,  24, 140],
            [0.25,  0, 110, 255],
            [0.50,  0, 190, 210],
            [0.75, 255, 150,   0],
            [1.0, 255,   0,   0],
        ], dtype=np.float32)
        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        pal = np.empty((256, 3), dtype=np.uint8)
        for cidx in range(3):
            pal[:, cidx] = np.interp(x, anchors[:, 0], anchors[:, cidx + 1]).astype(np.uint8)
        return pal

    def _render_field_to_label(self):
        if self._last_rgb is None:
            return
        h, w, _ = self._last_rgb.shape
        qimg = QImage(self._last_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        target = self.field_label.size()
        self.field_label.setPixmap(
            pix.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_field_to_label()

    def update_field(self, speed_2d, solid_2d):
        if speed_2d.size == 0:
            return
        speed = np.asarray(speed_2d, dtype=np.float32)
        solid = np.asarray(solid_2d, dtype=bool)

        if self._x_1d is not None and self._y_1d is not None:
            speed, solid = self._remap_to_physical(speed, solid)

        fluid = ~solid
        if np.any(fluid):
            vmax = float(np.percentile(speed[fluid], 99))
            vmax = max(vmax, 1e-3)
        else:
            vmax = 1.0

        norm = np.clip(speed / vmax, 0.0, 1.0)
        idx = (norm * 255.0).astype(np.uint8)
        rgb = self._palette[idx]
        rgb[solid] = np.array([32, 36, 44], dtype=np.uint8)
        rgb = np.ascontiguousarray(np.flipud(rgb))
        self._last_rgb = rgb
        self._render_field_to_label()

    def update_coefs(self, iteration, cd, cl):
        h = self._hist
        h['iter'].append(iteration)
        h['cd'].append(cd)
        h['cl'].append(cl)
        eff = cl / cd if abs(cd) > 1e-12 else 0.0
        h['eff'].append(eff)
        MAX = 3000
        if len(h['iter']) > MAX:
            for k in h:
                h[k] = h[k][-MAX:]
        self.line_cd.set_data(h['iter'], h['cd'])
        self.line_cl.set_data(h['iter'], h['cl'])
        self.line_eff.set_data(h['iter'], h['eff'])
        self.ax_coefs.relim()
        self.ax_coefs.autoscale_view()
        self.ax_eff.relim()
        self.ax_eff.autoscale_view()
        self.coef_canvas.draw_idle()

    def set_title(self, txt):
        self.lbl_title.setText(txt)

    def draw_idle(self):
        # Compatibilidad con llamadas existentes desde MainWindow.
        self.coef_canvas.draw_idle()

    def reset(self):
        self._hist = dict(iter=[], cd=[], cl=[], eff=[])
        for line in [self.line_cd, self.line_cl, self.line_eff]:
            line.set_data([], [])
        self.ax_coefs.relim()
        self.ax_coefs.autoscale_view()
        self.ax_eff.relim()
        self.ax_eff.autoscale_view()
        self.lbl_title.setText("Esperando simulación...")
        self._last_rgb = None
        self.field_label.clear()
        self.coef_canvas.draw_idle()


# ─────────────────────────────  METRICS PANEL  ──────────────────────────────

class MetricBox(QWidget):
    def __init__(self, label, unit="", parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(1)
        self.val_lbl = QLabel("—")
        self.val_lbl.setStyleSheet(f"color:{C['text']}; font:bold 18px 'Consolas'; ")
        self.val_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
        sub = QLabel(f"{label}  {unit}".strip())
        sub.setStyleSheet(f"color:{C['text3']}; font-size:10px; letter-spacing:1px;")
        v.addWidget(self.val_lbl)
        v.addWidget(sub)
        self.setStyleSheet(f"background:{C['panel2']}; border:1px solid {C['border']}; border-radius:4px;")

    def set(self, value):
        self.val_lbl.setText(str(value))


class MetricsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(200)
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)

        title = QLabel("MÉTRICAS")
        title.setStyleSheet(f"color:{C['accent']}; font:600 10px 'Segoe UI'; letter-spacing:2px; padding:4px 0;")
        v.addWidget(title)

        self.mb_cd   = MetricBox("Cd", "arrastre")
        self.mb_cl   = MetricBox("Cl", "sustentación")
        self.mb_eff  = MetricBox("Cl/Cd", "eficiencia")
        self.mb_alpha= MetricBox("α", "grados")
        self.mb_re   = MetricBox("Re", "Reynolds")
        self.mb_iter = MetricBox("Iter", "")
        self.mb_speed= MetricBox("Vel", "it/s")
        for mb in [self.mb_cd, self.mb_cl, self.mb_eff, self.mb_alpha, self.mb_re, self.mb_iter, self.mb_speed]:
            v.addWidget(mb)

        v.addWidget(sep())

        lbl_prog = QLabel("PROGRESO")
        lbl_prog.setStyleSheet(f"color:{C['accent']}; font:600 10px 'Segoe UI'; letter-spacing:2px; padding:4px 0;")
        v.addWidget(lbl_prog)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        v.addWidget(self.progress)
        self.lbl_prog_txt = QLabel("0 / 0")
        self.lbl_prog_txt.setStyleSheet(f"color:{C['text3']}; font:10px 'Consolas'; text-align:center;")
        self.lbl_prog_txt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.lbl_prog_txt)

        v.addWidget(sep())

        lbl_log = QLabel("LOG")
        lbl_log.setStyleSheet(f"color:{C['accent']}; font:600 10px 'Segoe UI'; letter-spacing:2px; padding:4px 0;")
        v.addWidget(lbl_log)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        v.addWidget(self.log, 1)

    def update_coefs(self, cd, cl, alpha):
        self.mb_cd.set(f"{cd:.5f}")
        self.mb_cl.set(f"{cl:.5f}")
        eff = cl / cd if abs(cd) > 1e-12 else 0.0
        self.mb_eff.set(f"{eff:.2f}")
        self.mb_alpha.set(f"{alpha:.1f}°")

    def update_iter(self, current, total, elapsed_time=None):
        pct = int(current / total * 100) if total > 0 else 0
        self.progress.setValue(pct)
        self.lbl_prog_txt.setText(f"{current} / {total}")
        self.mb_iter.set(str(current))
        
        # Calcular velocidad (it/s)
        if elapsed_time is not None and elapsed_time > 0:
            speed = current / elapsed_time
            self.mb_speed.set(f"{speed:.2f}")
        else:
            self.mb_speed.set("—")

    def set_re(self, re_val):
        if re_val:
            self.mb_re.set(f"{re_val:.0f}")

    def append_log(self, text):
        self.log.append(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def clear_log(self):
        self.log.clear()


# ─────────────────────────────  SUBPROCESS LOG READER  ──────────────────────

class LogReader(QThread):
    line_received = pyqtSignal(str)

    def __init__(self, process, parent=None):
        super().__init__(parent)
        self.process = process

    def run(self):
        for line in iter(self.process.stdout.readline, b''):
            try:
                self.line_received.emit(line.decode('utf-8', errors='replace').rstrip())
            except Exception:
                pass


# ─────────────────────────────  SHARED MEMORY READER  ───────────────────────

class SharedMemReader(QObject):
    frame_ready   = pyqtSignal(object, object, int, float, float, float)
    coords_ready  = pyqtSignal(object, object)  # x_1d, y_1d

    def __init__(self, parent=None):
        super().__init__(parent)
        self.shm_meta = None
        self.shm_data = None
        self.ny = 0
        self.nx = 0
        self._last_ts = 0.0
        self.x_1d = None
        self.y_1d = None

    def try_connect(self):
        if not SHM_AVAILABLE:
            return False
        try:
            shm_meta = shm_mod.SharedMemory(name=META_NAME, create=False)
            buf = shm_meta.buf
            ny = struct.unpack('i', bytes(buf[0:4]))[0]
            nx = struct.unpack('i', bytes(buf[4:8]))[0]
            if ny > 0 and nx > 0:
                shm_data = shm_mod.SharedMemory(name=SHM_NAME, create=False)
                self.shm_meta = shm_meta
                self.shm_data = shm_data
                self.ny = ny
                self.nx = nx
                self._last_ts = 0.0
                # Read physical coords if available
                try:
                    shm_c = shm_mod.SharedMemory(name="sim2d_coords", create=False)
                    x_1d = np.frombuffer(bytes(shm_c.buf[:nx * 4]), dtype=np.float32).copy()
                    y_1d = np.frombuffer(bytes(shm_c.buf[nx * 4:(nx + ny) * 4]), dtype=np.float32).copy()
                    shm_c.close()
                    self.x_1d = x_1d
                    self.y_1d = y_1d
                    self.coords_ready.emit(x_1d, y_1d)
                except Exception:
                    self.x_1d = None
                    self.y_1d = None
                return True
            shm_meta.close()
        except FileNotFoundError:
            pass
        return False

    def poll(self):
        if self.shm_meta is None:
            return
        try:
            buf = self.shm_meta.buf
            iteracion = struct.unpack('i', bytes(buf[8:12]))[0]
            cd_val    = struct.unpack('f', bytes(buf[12:16]))[0]
            cl_val    = struct.unpack('f', bytes(buf[16:20]))[0]
            alpha_val = struct.unpack('f', bytes(buf[20:24]))[0]
            timestamp = struct.unpack('d', bytes(buf[24:32]))[0]

            if timestamp <= self._last_ts:
                return
            self._last_ts = timestamp

            ny, nx = self.ny, self.nx
            n = ny * nx
            
            # Intentar leer datos volumétricos si existen
            if self.shm_data is not None:
                try:
                    speed = np.ndarray((ny, nx), dtype=np.float32,
                                       buffer=bytes(self.shm_data.buf[0:n * 4])).copy()
                    solid = np.ndarray((ny, nx), dtype=np.uint8,
                                       buffer=bytes(self.shm_data.buf[n * 4:n * 5])).copy().astype(bool)
                except Exception:
                    # Si falla la lectura de datos volumétricos, usar datos dummy
                    speed = np.zeros((ny, nx), dtype=np.float32)
                    solid = np.zeros((ny, nx), dtype=bool)
            else:
                # Sin live_view: usar datos dummy para visualización
                speed = np.zeros((ny, nx), dtype=np.float32)
                solid = np.zeros((ny, nx), dtype=bool)
            
            self.frame_ready.emit(speed, solid, iteracion, cd_val, cl_val, alpha_val)
        except Exception:
            pass

    def disconnect(self):
        if self.shm_meta:
            try: self.shm_meta.close()
            except Exception: pass
            self.shm_meta = None
        if self.shm_data:
            try: self.shm_data.close()
            except Exception: pass
            self.shm_data = None
        self.ny = 0; self.nx = 0


# ─────────────────────────────  GA HELPERS  ─────────────────────────────────

_RE_IND = re.compile(
    r'alpha=([\d.+\-]+)°?\s*->\s*Cl=\s*([\d.eE+\-]+)\s+Cd=\s*([\d.eE+\-]+)\s+L/D=\s*([\d.eE+\-]+)',
    re.IGNORECASE
)
_RE_GEN_DONE = re.compile(r'GEN\s+(\d+)\s+COMPLETADA.*?\((\d+)s\)', re.IGNORECASE)
_RE_MEJOR    = re.compile(r'Mejor\s+L/D:\s+([\d.eE+\-]+)', re.IGNORECASE)
_RE_MEDIA    = re.compile(r'Media\s+L/D:\s+([\d.eE+\-]+)', re.IGNORECASE)
_RE_PEOR     = re.compile(r'Peor\s+L/D:\s+([\d.eE+\-]+)', re.IGNORECASE)
_RE_EVAL     = re.compile(r'Evaluados:\s+(\d+)/(\d+)', re.IGNORECASE)
_RE_IA       = re.compile(r'IA\s+descartes:\s+(\d+)', re.IGNORECASE)
_RE_VSBASE   = re.compile(r'Δ\s+vs\s+perfil\s+base:\s+([+\-\d.eE]+)', re.IGNORECASE)


def parse_individual_line(line):
    m = _RE_IND.search(line)
    if m:
        try:
            return {'alpha': float(m.group(1)), 'cl': float(m.group(2)),
                    'cd': float(m.group(3)), 'ld': float(m.group(4))}
        except ValueError:
            pass
    return None


def parse_gen_summary(line):
    result = {}
    for pat, key in [(_RE_GEN_DONE, None), (_RE_MEJOR, 'mejor'), (_RE_MEDIA, 'media'),
                     (_RE_PEOR, 'peor'), (_RE_EVAL, None), (_RE_IA, 'ia_desc'), (_RE_VSBASE, 'vs_base')]:
        m = pat.search(line)
        if not m:
            continue
        if pat is _RE_GEN_DONE:
            result['gen_done'] = int(m.group(1))
            result['t_seg'] = float(m.group(2))
        elif pat is _RE_EVAL:
            result['evaluados'] = int(m.group(1))
            result['total_pop'] = int(m.group(2))
        else:
            try:
                result[key] = float(m.group(1))
            except ValueError:
                pass
    return result if result else None


def load_dat_profile(filepath):
    coords = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        x, y = float(parts[0]), float(parts[1])
                        if 0.0 <= x <= 1.5:
                            coords.append((x, y))
                    except ValueError:
                        pass
    except Exception:
        return None
    return np.array(coords) if len(coords) > 4 else None


def compute_geometry_delta(coords_base, coords_best, n_points=200):
    def split_profile(coords):
        le_idx = int(np.argmin(coords[:, 0]))
        upper = coords[:le_idx + 1][::-1]
        lower = coords[le_idx:]
        return upper, lower

    def interp_surface(surface, xg):
        idx = np.argsort(surface[:, 0])
        xs, ys = surface[idx, 0], surface[idx, 1]
        _, ui = np.unique(xs, return_index=True)
        return np.interp(xg, xs[ui], ys[ui])

    try:
        ub, lb = split_profile(coords_base)
        ut, lt = split_profile(coords_best)
        xg = np.linspace(0.01, 0.99, n_points)
        yb_up = interp_surface(ub, xg)
        yb_lo = interp_surface(lb, xg)
        yt_up = interp_surface(ut, xg)
        yt_lo = interp_surface(lt, xg)
        return xg, yt_up - yb_up, yt_lo - yb_lo, yb_up, yb_lo, yt_up, yt_lo
    except Exception:
        return None


# ─────────────────────────────  GA PARAMETER PANEL  ─────────────────────────

class GAParamPanel(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumWidth(290)
        self.setMaximumWidth(340)
        container = QWidget()
        self.setWidget(container)
        lay = QVBoxLayout(container)
        lay.setContentsMargins(4, 4, 4, 8)
        lay.setSpacing(2)

        # ── Perfil Base ──────────────────────────────────────────────────
        s = CollapsibleSection("PERFIL BASE")
        self.e_archivo_base = float_edit("OPTIMO_PARCIAL_2604_2.dat")
        btn_ab = QPushButton("…"); btn_ab.setFixedSize(28, 24)
        btn_ab.clicked.connect(self._browse_base)
        row_ab = QHBoxLayout(); row_ab.setContentsMargins(0, 0, 0, 0)
        l_ab = QLabel("Perfil base"); l_ab.setFixedWidth(145)
        l_ab.setStyleSheet(f"color:{C['text2']};font-size:11px;")
        row_ab.addWidget(l_ab); row_ab.addWidget(self.e_archivo_base, 1); row_ab.addWidget(btn_ab)
        s.add_row(row_ab)
        self.e_dir_res  = float_edit("resultados_ga"); s.add_row(make_row("Dir. resultados", self.e_dir_res))
        self.e_arch_ia  = float_edit("cerebro_aerodinamico.pkl"); s.add_row(make_row("Memoria IA", self.e_arch_ia))
        self.e_arch_ml  = float_edit("aprendizaje_ML.jsonl"); s.add_row(make_row("Datos ML", self.e_arch_ml))
        lay.addWidget(s)

        # ── Población ────────────────────────────────────────────────────
        s = CollapsibleSection("POBLACIÓN Y EVOLUCIÓN")
        self.e_pop    = int_spin(12, 2, 500);   s.add_row(make_row("Tamaño población", self.e_pop))
        self.e_gens   = int_spin(10, 1, 10000); s.add_row(make_row("Generaciones", self.e_gens))
        self.e_elites = int_spin(4, 0, 50);     s.add_row(make_row("Élites", self.e_elites))
        self.e_torneo = int_spin(4, 2, 50);     s.add_row(make_row("Tamaño torneo", self.e_torneo))
        lay.addWidget(s)

        # ── Mutación ─────────────────────────────────────────────────────
        s = CollapsibleSection("MUTACIÓN")
        self.e_prob_mut   = float_edit(0.85);  s.add_row(make_row("Prob. mutación", self.e_prob_mut))
        self.e_bumps_min  = int_spin(1, 1, 20); s.add_row(make_row("Bumps mín", self.e_bumps_min))
        self.e_bumps_max  = int_spin(4, 1, 20); s.add_row(make_row("Bumps máx", self.e_bumps_max))
        self.e_sigma_min  = float_edit(0.03);  s.add_row(make_row("σ_bump mín", self.e_sigma_min))
        self.e_sigma_max  = float_edit(0.15);  s.add_row(make_row("σ_bump máx", self.e_sigma_max))
        self.e_ampl_mut   = float_edit(0.008); s.add_row(make_row("Amplitud mut.", self.e_ampl_mut))
        self.e_prob_te    = float_edit(0.3);   s.add_row(make_row("Prob. mut. TE", self.e_prob_te))
        self.e_ampl_te    = float_edit(0.002); s.add_row(make_row("Amplitud TE", self.e_ampl_te))
        self.e_suavizado  = int_spin(2, 0, 20); s.add_row(make_row("Suavizado iter.", self.e_suavizado))
        self.combo_modo_mut = QComboBox(); self.combo_modo_mut.addItems(['parametrica', 'legacy'])
        self.combo_modo_mut.setFixedHeight(24); s.add_row(make_row("Modo mutación", self.combo_modo_mut))
        self.cb_legacy_fb = QCheckBox(); self.cb_legacy_fb.setChecked(True)
        s.add_row(make_check_row("Fallback a legacy", self.cb_legacy_fb))
        lay.addWidget(s)

        # ── Geom Paramétrica ─────────────────────────────────────────────
        s = CollapsibleSection("GEOM PARAMÉTRICA", collapsed=True)
        self.e_n_ctrl      = int_spin(7, 2, 30);    s.add_row(make_row("Nodos control", self.e_n_ctrl))
        self.e_camber_std  = float_edit(0.0035);     s.add_row(make_row("camber_mut_std", self.e_camber_std))
        self.e_esp_std     = float_edit(0.0045);     s.add_row(make_row("espesor_mut_std", self.e_esp_std))
        self.e_te_cam_std  = float_edit(0.0015);     s.add_row(make_row("te_camber_shift_std", self.e_te_cam_std))
        self.e_te_esp_std  = float_edit(0.0012);     s.add_row(make_row("te_espesor_shift_std", self.e_te_esp_std))
        self.e_le_prot     = float_edit(0.06);       s.add_row(make_row("le_proteccion_x", self.e_le_prot))
        lay.addWidget(s)

        # ── Restricciones Geom ───────────────────────────────────────────
        s = CollapsibleSection("RESTRICCIONES GEOM", collapsed=True)
        self.e_esp_min_g   = float_edit(2e-4);  s.add_row(make_row("espesor_min_global", self.e_esp_min_g))
        self.e_te_abs      = float_edit(3e-4);  s.add_row(make_row("te_esp_min_abs", self.e_te_abs))
        self.e_te_rel_min  = float_edit(0.65);  s.add_row(make_row("te_esp_rel_min", self.e_te_rel_min))
        self.e_te_rel_max  = float_edit(3.50);  s.add_row(make_row("te_esp_rel_max", self.e_te_rel_max))
        self.e_te_tol      = float_edit(1e-9);  s.add_row(make_row("te_validacion_tol", self.e_te_tol))
        self.e_le_fac      = float_edit(0.40);  s.add_row(make_row("le_radio_factor_min", self.e_le_fac))
        self.e_le_abs      = float_edit(2e-4);  s.add_row(make_row("le_radio_min_abs", self.e_le_abs))
        self.e_le_pts      = int_spin(5, 0, 30); s.add_row(make_row("le_puntos_preservar", self.e_le_pts))
        self.e_max_int     = int_spin(120, 1, 5000); s.add_row(make_row("max_intentos_geom", self.e_max_int))
        self.cb_diag_geom  = QCheckBox(); self.cb_diag_geom.setChecked(True)
        s.add_row(make_check_row("reportar_diag_geom", self.cb_diag_geom))
        lay.addWidget(s)

        # ── CFD Base ─────────────────────────────────────────────────────
        s = CollapsibleSection("SIMULACIÓN CFD")
        self.e_sim_iters = int_spin(2000, 100, 100000); s.add_row(make_row("Iteraciones CFD", self.e_sim_iters))
        self.e_ga_v0x    = float_edit(1.0);   s.add_row(make_row("Velocidad v₀ₓ", self.e_ga_v0x, "m/s"))
        self.e_ga_alpha  = float_edit(4.0);   s.add_row(make_row("Ángulo α base", self.e_ga_alpha, "°"))
        self.e_ga_chord  = float_edit(1.0);   s.add_row(make_row("Cuerda", self.e_ga_chord, "m"))
        self.e_ga_dx     = float_edit(0.001); s.add_row(make_row("dx_min", self.e_ga_dx, "m"))
        self.e_ga_cfl    = float_edit(0.5);   s.add_row(make_row("CFL", self.e_ga_cfl))
        self.e_ga_rho    = float_edit(1.0);   s.add_row(make_row("Densidad ρ", self.e_ga_rho, "kg/m³"))
        self.e_ga_nu     = float_edit(1e-5);  s.add_row(make_row("Viscosidad ν", self.e_ga_nu, "m²/s"))
        lay.addWidget(s)

        # ── Multi-ángulo ─────────────────────────────────────────────────
        s = CollapsibleSection("MULTI-ÁNGULO", collapsed=True)
        self.cb_multi_ang = QCheckBox(); s.add_row(make_check_row("Activar multi-ángulo", self.cb_multi_ang))
        self.e_delta_ang  = float_edit(1.0); s.add_row(make_row("Δα (±)", self.e_delta_ang, "°"))
        self.combo_fitness = QComboBox(); self.combo_fitness.addItems(['mean', 'min', 'weighted'])
        self.combo_fitness.setFixedHeight(24); s.add_row(make_row("Modo fitness", self.combo_fitness))
        self.e_peso_base  = float_edit(2.0); s.add_row(make_row("Peso ángulo base", self.e_peso_base))
        lay.addWidget(s)

        # ── IA / ML ──────────────────────────────────────────────────────
        s = CollapsibleSection("IA / ML", collapsed=True)
        self.cb_usar_ia   = QCheckBox(); s.add_row(make_check_row("Usar filtro IA", self.cb_usar_ia))
        self.e_umbral_ia  = float_edit(0.7); s.add_row(make_row("Umbral calidad", self.e_umbral_ia))
        lay.addWidget(s)

        # ── Sim Extra Params ─────────────────────────────────────────────
        s = CollapsibleSection("SIM EXTRA PARAMS", collapsed=True)
        hint = QLabel("JSON: parámetros extra para Simulador2D"); hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{C['text3']};font-size:10px;font-style:italic;"); s.add_row(hint)
        self.e_sim_extra = QTextEdit()
        self.e_sim_extra.setFixedHeight(100)
        self.e_sim_extra.setText(json.dumps(
            {'Lx': 12, 'Ly': 8, 'usar_wale': True, 'divergencia': 0.1,
             'ancho_zona_fina_x': 1.2, 'ancho_zona_fina_y': 1,
             'factor_expansion': 1.1, 'mg_modo_turbo': True}, indent=2))
        s.add_row(self.e_sim_extra)
        lay.addWidget(s)

        lay.addStretch()

    def _browse_base(self):
        f, _ = QFileDialog.getOpenFileName(self, "Perfil base", str(SIM_DIR), "DAT (*.dat);;All (*)")
        if f:
            self.e_archivo_base.setText(f)

    def _f(self, w, default=None):
        t = w.text().strip()
        if not t or t.lower() == 'none':
            return default
        try:
            return float(t)
        except ValueError:
            return default

    def get_config(self):
        extra = {}
        try:
            extra = json.loads(self.e_sim_extra.toPlainText())
        except Exception:
            pass
        return {
            'archivo_base':                     self.e_archivo_base.text().strip() or 'OPTIMO_PARCIAL_2604_2.dat',
            'directorio_resultados':            self.e_dir_res.text().strip() or 'resultados_ga',
            'archivo_memoria_ia':               self.e_arch_ia.text().strip() or 'cerebro_aerodinamico.pkl',
            'archivo_datos_ml':                 self.e_arch_ml.text().strip() or 'aprendizaje_ML.jsonl',
            'poblacion_tamano':                 self.e_pop.value(),
            'generaciones':                     self.e_gens.value(),
            'elites':                           self.e_elites.value(),
            'torneo_tamano':                    self.e_torneo.value(),
            'prob_mutacion':                    self._f(self.e_prob_mut, 0.85),
            'n_bumps_min':                      self.e_bumps_min.value(),
            'n_bumps_max':                      self.e_bumps_max.value(),
            'sigma_bump_min':                   self._f(self.e_sigma_min, 0.03),
            'sigma_bump_max':                   self._f(self.e_sigma_max, 0.15),
            'amplitud_mutacion':                self._f(self.e_ampl_mut, 0.008),
            'prob_mutar_trailing_edge':         self._f(self.e_prob_te, 0.3),
            'amplitud_trailing_edge':           self._f(self.e_ampl_te, 0.002),
            'suavizado_iteraciones':            self.e_suavizado.value(),
            'modo_mutacion':                    self.combo_modo_mut.currentText(),
            'usar_legacy_fallback':             self.cb_legacy_fb.isChecked(),
            'n_control_mutacion':               self.e_n_ctrl.value(),
            'camber_mut_std':                   self._f(self.e_camber_std, 0.0035),
            'espesor_mut_std':                  self._f(self.e_esp_std, 0.0045),
            'te_camber_shift_std':              self._f(self.e_te_cam_std, 0.0015),
            'te_espesor_shift_std':             self._f(self.e_te_esp_std, 0.0012),
            'le_proteccion_x':                  self._f(self.e_le_prot, 0.06),
            'espesor_min_global':               self._f(self.e_esp_min_g, 2e-4),
            'te_espesor_min_absoluto':          self._f(self.e_te_abs, 3e-4),
            'te_espesor_rel_min':               self._f(self.e_te_rel_min, 0.65),
            'te_espesor_rel_max':               self._f(self.e_te_rel_max, 3.50),
            'te_validacion_tol':                self._f(self.e_te_tol, 1e-9),
            'le_radio_factor_min':              self._f(self.e_le_fac, 0.40),
            'le_radio_min_absoluto':            self._f(self.e_le_abs, 2e-4),
            'le_puntos_preservar':              self.e_le_pts.value(),
            'max_intentos_geometria':           self.e_max_int.value(),
            'reportar_diagnostico_geometria':   self.cb_diag_geom.isChecked(),
            'simulacion_iteraciones':           self.e_sim_iters.value(),
            'v0x':                              self._f(self.e_ga_v0x, 1.0),
            'alpha_deg':                        self._f(self.e_ga_alpha, 4.0),
            'chord':                            self._f(self.e_ga_chord, 1.0),
            'dx_min':                           self._f(self.e_ga_dx, 0.001),
            'CFL':                              self._f(self.e_ga_cfl, 0.5),
            'rho':                              self._f(self.e_ga_rho, 1.0),
            'nu':                               self._f(self.e_ga_nu, 1e-5),
            'multi_angulo':                     self.cb_multi_ang.isChecked(),
            'delta_angulo':                     self._f(self.e_delta_ang, 1.0),
            'fitness_modo':                     self.combo_fitness.currentText(),
            'peso_angulo_base':                 self._f(self.e_peso_base, 2.0),
            'usar_ia':                          self.cb_usar_ia.isChecked(),
            'umbral_calidad':                   self._f(self.e_umbral_ia, 0.7),
            'sim_extra_params':                 extra,
        }

    def set_config(self, cfg):
        def sf(w, v): w.setText('' if v is None else str(v))
        sf(self.e_archivo_base, cfg.get('archivo_base', 'OPTIMO_PARCIAL_2604_2.dat'))
        sf(self.e_dir_res,      cfg.get('directorio_resultados', 'resultados_ga'))
        sf(self.e_arch_ia,      cfg.get('archivo_memoria_ia', 'cerebro_aerodinamico.pkl'))
        sf(self.e_arch_ml,      cfg.get('archivo_datos_ml', 'aprendizaje_ML.jsonl'))
        self.e_pop.setValue(int(cfg.get('poblacion_tamano', 12)))
        self.e_gens.setValue(int(cfg.get('generaciones', 10)))
        self.e_elites.setValue(int(cfg.get('elites', 4)))
        self.e_torneo.setValue(int(cfg.get('torneo_tamano', 4)))
        sf(self.e_prob_mut,   cfg.get('prob_mutacion', 0.85))
        self.e_bumps_min.setValue(int(cfg.get('n_bumps_min', 1)))
        self.e_bumps_max.setValue(int(cfg.get('n_bumps_max', 4)))
        sf(self.e_sigma_min,  cfg.get('sigma_bump_min', 0.03))
        sf(self.e_sigma_max,  cfg.get('sigma_bump_max', 0.15))
        sf(self.e_ampl_mut,   cfg.get('amplitud_mutacion', 0.008))
        sf(self.e_prob_te,    cfg.get('prob_mutar_trailing_edge', 0.3))
        sf(self.e_ampl_te,    cfg.get('amplitud_trailing_edge', 0.002))
        self.e_suavizado.setValue(int(cfg.get('suavizado_iteraciones', 2)))
        self.combo_modo_mut.setCurrentText(cfg.get('modo_mutacion', 'parametrica'))
        self.cb_legacy_fb.setChecked(bool(cfg.get('usar_legacy_fallback', True)))
        self.e_n_ctrl.setValue(int(cfg.get('n_control_mutacion', 7)))
        sf(self.e_camber_std,  cfg.get('camber_mut_std', 0.0035))
        sf(self.e_esp_std,     cfg.get('espesor_mut_std', 0.0045))
        sf(self.e_te_cam_std,  cfg.get('te_camber_shift_std', 0.0015))
        sf(self.e_te_esp_std,  cfg.get('te_espesor_shift_std', 0.0012))
        sf(self.e_le_prot,     cfg.get('le_proteccion_x', 0.06))
        sf(self.e_esp_min_g,   cfg.get('espesor_min_global', 2e-4))
        sf(self.e_te_abs,      cfg.get('te_espesor_min_absoluto', 3e-4))
        sf(self.e_te_rel_min,  cfg.get('te_espesor_rel_min', 0.65))
        sf(self.e_te_rel_max,  cfg.get('te_espesor_rel_max', 3.50))
        sf(self.e_te_tol,      cfg.get('te_validacion_tol', 1e-9))
        sf(self.e_le_fac,      cfg.get('le_radio_factor_min', 0.40))
        sf(self.e_le_abs,      cfg.get('le_radio_min_absoluto', 2e-4))
        self.e_le_pts.setValue(int(cfg.get('le_puntos_preservar', 5)))
        self.e_max_int.setValue(int(cfg.get('max_intentos_geometria', 120)))
        self.cb_diag_geom.setChecked(bool(cfg.get('reportar_diagnostico_geometria', True)))
        self.e_sim_iters.setValue(int(cfg.get('simulacion_iteraciones', 2000)))
        sf(self.e_ga_v0x,   cfg.get('v0x', 1.0))
        sf(self.e_ga_alpha, cfg.get('alpha_deg', 4.0))
        sf(self.e_ga_chord, cfg.get('chord', 1.0))
        sf(self.e_ga_dx,    cfg.get('dx_min', 0.001))
        sf(self.e_ga_cfl,   cfg.get('CFL', 0.5))
        sf(self.e_ga_rho,   cfg.get('rho', 1.0))
        sf(self.e_ga_nu,    cfg.get('nu', 1e-5))
        self.cb_multi_ang.setChecked(bool(cfg.get('multi_angulo', False)))
        sf(self.e_delta_ang,  cfg.get('delta_angulo', 1.0))
        self.combo_fitness.setCurrentText(cfg.get('fitness_modo', 'mean'))
        sf(self.e_peso_base,  cfg.get('peso_angulo_base', 2.0))
        self.cb_usar_ia.setChecked(bool(cfg.get('usar_ia', False)))
        sf(self.e_umbral_ia, cfg.get('umbral_calidad', 0.7))
        extra = cfg.get('sim_extra_params', {})
        if extra:
            try:
                self.e_sim_extra.setText(json.dumps(extra, indent=2))
            except Exception:
                pass


# ─────────────────────────────  GA VISUALIZATION CANVAS  ────────────────────

class GAVisualizationCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 7), facecolor=C['bg'])
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(300)

        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               hspace=0.42, wspace=0.38,
                               left=0.07, right=0.96, top=0.94, bottom=0.07)
        self.ax_conv = self.fig.add_subplot(gs[0, 0])
        self.ax_scat = self.fig.add_subplot(gs[0, 1])
        self.ax_geom = self.fig.add_subplot(gs[1, 0])
        self.ax_heat = self.fig.add_subplot(gs[1, 1])

        for ax in [self.ax_conv, self.ax_scat, self.ax_geom, self.ax_heat]:
            ax.set_facecolor(C['panel'])
            ax.tick_params(colors=C['text3'], labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor(C['border2'])

        # Convergence
        self.ax_conv.set_title("Convergencia L/D", color=C['text2'], fontsize=9, pad=3)
        self.ax_conv.set_xlabel("Generación", color=C['text3'], fontsize=7)
        self.ax_conv.set_ylabel("L/D", color=C['text3'], fontsize=7)
        self.ax_conv.grid(True, color=C['border2'], lw=0.4, alpha=0.6)
        self.line_mejor, = self.ax_conv.plot([], [], color=C['green'], lw=1.5, label='Mejor')
        self.line_media, = self.ax_conv.plot([], [], color=C['accent'], lw=1.2, linestyle='--', label='Media')
        self.line_peor,  = self.ax_conv.plot([], [], color=C['red'],   lw=1.0, linestyle=':', label='Peor')
        self.ax_conv.legend(fontsize=7, facecolor=C['panel2'], edgecolor=C['border'], labelcolor=C['text'])
        self._fill_conv = None
        self._hline_base = None

        # Scatter
        self.ax_scat.set_title("Población — Cd vs Cl", color=C['text2'], fontsize=9, pad=3)
        self.ax_scat.set_xlabel("Cd", color=C['text3'], fontsize=7)
        self.ax_scat.set_ylabel("Cl", color=C['text3'], fontsize=7)
        self.ax_scat.grid(True, color=C['border2'], lw=0.4, alpha=0.6)
        self._scat_artists = []
        self._scat_cbar = None

        # Profile
        self.ax_geom.set_title("Perfil: base vs mejor", color=C['text2'], fontsize=9, pad=3)
        self.ax_geom.set_xlabel("x/c", color=C['text3'], fontsize=7)
        self.ax_geom.set_ylabel("y/c", color=C['text3'], fontsize=7)
        self.ax_geom.grid(True, color=C['border2'], lw=0.3, alpha=0.5)
        self._geom_artists = []

        # Heatmap
        self.ax_heat.set_title("Δy geometría (mejor − base)", color=C['text2'], fontsize=9, pad=3)
        self.ax_heat.set_xlabel("x/c", color=C['text3'], fontsize=7)
        self._heat_cbar = None

        self.draw_idle()

    def _ax_style(self, ax):
        ax.set_facecolor(C['panel'])
        ax.tick_params(colors=C['text3'], labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(C['border2'])

    def update_convergence(self, hist_list, base_ld=None):
        if not hist_list:
            return
        gens  = [h['gen'] + 1 for h in hist_list]
        mejor = [h.get('mejor', 0) for h in hist_list]
        media = [h.get('media', mejor[-1]) for h in hist_list]
        peor  = [h.get('peor', 0) for h in hist_list]

        self.line_mejor.set_data(gens, mejor)
        self.line_media.set_data(gens, media)
        self.line_peor.set_data(gens, peor)

        if self._fill_conv:
            try: self._fill_conv.remove()
            except Exception: pass
        self._fill_conv = self.ax_conv.fill_between(
            gens, peor, mejor, alpha=0.10, color=C['accent'], zorder=1)

        if base_ld is not None and base_ld > 0:
            if self._hline_base:
                try: self._hline_base.remove()
                except Exception: pass
            self._hline_base = self.ax_conv.axhline(
                base_ld, color=C['amber'], lw=1, linestyle='--', alpha=0.7)

        self.ax_conv.relim()
        self.ax_conv.autoscale_view()
        self.draw_idle()

    def update_scatter(self, individuals):
        for a in self._scat_artists:
            try: a.remove()
            except Exception: pass
        self._scat_artists = []
        if self._scat_cbar:
            try: self._scat_cbar.remove()
            except Exception: pass
            self._scat_cbar = None

        if not individuals:
            self.draw_idle()
            return

        cds  = [d['cd'] for d in individuals]
        cls_ = [d['cl'] for d in individuals]
        lds  = [d['ld'] for d in individuals]

        sc = self.ax_scat.scatter(cds, cls_, c=lds, cmap='viridis', s=50, alpha=0.85,
                                   zorder=3, edgecolors=C['border2'], linewidths=0.5)
        self._scat_artists.append(sc)
        try:
            cb = self.fig.colorbar(sc, ax=self.ax_scat, shrink=0.75)
            cb.set_label('L/D', color=C['text2'], fontsize=7)
            cb.ax.yaxis.set_tick_params(color=C['text3'], labelsize=6)
            self._scat_cbar = cb
        except Exception:
            pass

        best_i = int(np.argmax(lds))
        star = self.ax_scat.scatter([cds[best_i]], [cls_[best_i]], s=120,
                                     color=C['green'], marker='*', zorder=5)
        ann = self.ax_scat.annotate(
            f'L/D={lds[best_i]:.2f}', (cds[best_i], cls_[best_i]),
            xytext=(6, 4), textcoords='offset points', fontsize=6, color=C['green'])
        self._scat_artists += [star, ann]

        self.ax_scat.relim()
        self.ax_scat.autoscale_view()
        self.draw_idle()

    def update_geometry(self, coords_base, coords_best):
        result = compute_geometry_delta(coords_base, coords_best)
        if result is None:
            return
        xg, d_up, d_lo, yb_up, yb_lo, yt_up, yt_lo = result

        # ── Profile comparison ────────────────────────────────────────────
        for a in self._geom_artists:
            try: a.remove()
            except Exception: pass
        self._geom_artists = []

        lb, = self.ax_geom.plot(xg, yb_up, color='#607080', lw=1, linestyle='--', alpha=0.7, label='Base')
        self.ax_geom.plot(xg, yb_lo, color='#607080', lw=1, linestyle='--', alpha=0.7)
        lt, = self.ax_geom.plot(xg, yt_up, color=C['accent'], lw=1.5, label='Mejor')
        self.ax_geom.plot(xg, yt_lo, color=C['accent'], lw=1.5)

        fu = self.ax_geom.fill_between(xg, yb_up, yt_up, where=d_up > 0,  alpha=0.22, color=C['green'])
        fd = self.ax_geom.fill_between(xg, yb_up, yt_up, where=d_up < 0,  alpha=0.22, color=C['red'])
        fl = self.ax_geom.fill_between(xg, yb_lo, yt_lo, where=d_lo < 0,  alpha=0.18, color=C['green'])
        fr = self.ax_geom.fill_between(xg, yb_lo, yt_lo, where=d_lo > 0,  alpha=0.18, color=C['red'])
        self._geom_artists = [lb, lt, fu, fd, fl, fr]

        try:
            self.ax_geom.set_aspect('equal', adjustable='datalim')
        except Exception:
            pass
        self.ax_geom.legend(fontsize=6, facecolor=C['panel2'], edgecolor=C['border'], labelcolor=C['text'])

        # ── Delta heatmap ─────────────────────────────────────────────────
        self.ax_heat.cla()
        self._ax_style(self.ax_heat)
        self.ax_heat.set_title("Δy geometría (mejor − base)", color=C['text2'], fontsize=9, pad=3)
        self.ax_heat.set_xlabel("x/c", color=C['text3'], fontsize=7)

        if self._heat_cbar:
            try: self._heat_cbar.remove()
            except Exception: pass
            self._heat_cbar = None

        hmap = np.vstack([d_up, d_lo])
        vmax = max(float(np.abs(hmap).max()), 1e-8)
        im = self.ax_heat.imshow(hmap, aspect='auto', cmap='RdYlGn',
                                  vmin=-vmax, vmax=vmax,
                                  extent=[0, 1, -0.5, 1.5],
                                  origin='lower', interpolation='bilinear')
        self.ax_heat.set_yticks([0, 1])
        self.ax_heat.set_yticklabels(['Inferior', 'Superior'], fontsize=6, color=C['text2'])
        try:
            cb = self.fig.colorbar(im, ax=self.ax_heat, shrink=0.75)
            cb.set_label('Δy', color=C['text2'], fontsize=7)
            cb.ax.yaxis.set_tick_params(color=C['text3'], labelsize=6)
            self._heat_cbar = cb
        except Exception:
            pass

        self.draw_idle()

    def reset(self):
        for ln in [self.line_mejor, self.line_media, self.line_peor]:
            ln.set_data([], [])
        if self._fill_conv:
            try: self._fill_conv.remove()
            except Exception: pass
            self._fill_conv = None
        if self._hline_base:
            try: self._hline_base.remove()
            except Exception: pass
            self._hline_base = None
        for a in self._scat_artists:
            try: a.remove()
            except Exception: pass
        self._scat_artists = []
        if self._scat_cbar:
            try: self._scat_cbar.remove()
            except Exception: pass
            self._scat_cbar = None
        self.ax_geom.cla(); self._ax_style(self.ax_geom)
        self.ax_heat.cla(); self._ax_style(self.ax_heat)
        self._geom_artists = []
        if self._heat_cbar:
            try: self._heat_cbar.remove()
            except Exception: pass
            self._heat_cbar = None
        self.draw_idle()


# ─────────────────────────────  GA METRICS PANEL  ───────────────────────────

class GAMetricsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(200)
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)

        title = QLabel("GA MÉTRICAS")
        title.setStyleSheet(f"color:{C['accent']}; font:600 10px 'Segoe UI'; letter-spacing:2px; padding:4px 0;")
        v.addWidget(title)

        self.mb_best  = MetricBox("Best L/D", "fitness")
        self.mb_mean  = MetricBox("Mean L/D", "generación")
        self.mb_delta = MetricBox("Δ L/D base", "absoluto")
        self.mb_time  = MetricBox("T gen.", "s")
        self.mb_evals = MetricBox("Evaluados", "ind.")
        self.mb_ia    = MetricBox("IA descartes", "ind.")
        for mb in [self.mb_best, self.mb_mean, self.mb_delta, self.mb_time, self.mb_evals, self.mb_ia]:
            v.addWidget(mb)

        v.addWidget(sep())
        lbl_p = QLabel("PROGRESO")
        lbl_p.setStyleSheet(f"color:{C['accent']}; font:600 10px 'Segoe UI'; letter-spacing:2px; padding:4px 0;")
        v.addWidget(lbl_p)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        v.addWidget(self.progress)
        self.lbl_gen = QLabel("Gen 0 / 0")
        self.lbl_gen.setStyleSheet(f"color:{C['text3']}; font:10px 'Consolas';")
        self.lbl_gen.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.lbl_gen)

        v.addWidget(sep())
        lbl_l = QLabel("LOG GA")
        lbl_l.setStyleSheet(f"color:{C['accent']}; font:600 10px 'Segoe UI'; letter-spacing:2px; padding:4px 0;")
        v.addWidget(lbl_l)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(100)
        v.addWidget(self.log, 1)

    def update_gen(self, gen, total, stats=None):
        pct = int(gen / total * 100) if total > 0 else 0
        self.progress.setValue(pct)
        self.lbl_gen.setText(f"Gen {gen} / {total}")
        if not stats:
            return
        if 'mejor' in stats:
            self.mb_best.set(f"{stats['mejor']:.4f}")
        if 'media' in stats:
            self.mb_mean.set(f"{stats['media']:.4f}")
        if 'vs_base' in stats:
            v = stats['vs_base']
            color = C['green'] if v >= 0 else C['red']
            self.mb_delta.val_lbl.setStyleSheet(f"color:{color}; font:bold 18px 'Consolas';")
            self.mb_delta.set(f"{v:+.4f}")
        if 't_seg' in stats:
            self.mb_time.set(f"{stats['t_seg']:.0f}s")
        if 'evaluados' in stats:
            self.mb_evals.set(str(stats['evaluados']))
        if 'ia_desc' in stats:
            self.mb_ia.set(str(stats['ia_desc']))

    def append_log(self, text):
        tl = text.lower()
        if 'generaci' in tl or ('gen ' in tl and 'completada' in tl):
            color = C['accent']
        elif 'mejor' in tl or 'record' in tl:
            color = C['green']
        elif 'error' in tl or 'crítico' in tl:
            color = C['red']
        elif 'warning' in tl:
            color = C['amber']
        else:
            color = None
        if color:
            self.log.append(f'<span style="color:{color}">{text}</span>')
        else:
            self.log.append(text)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def clear_log(self):
        self.log.clear()

    def reset(self):
        self.progress.setValue(0)
        self.lbl_gen.setText("Gen 0 / 0")
        for mb in [self.mb_best, self.mb_mean, self.mb_delta, self.mb_time, self.mb_evals, self.mb_ia]:
            mb.set("—")
        self.clear_log()


# ─────────────────────────────  MAIN WINDOW  ────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulador 2D CFD")
        self.resize(1360, 820)
        self.setStyleSheet(QSS)

        self._process = None
        self._log_reader = None
        self._total_iters = 2000
        self._run_start_time = 0.0

        self._ga_process = None
        self._ga_log_reader = None
        self._ga_tmp_script = None
        self._ga_current_inds = []
        self._ga_pending_stats = {}
        self._ga_gen_num = 0
        self._ga_scatter_t = 0.0

        self._build_ui()
        self._build_status_bar()

        # shared memory polling timer
        self.shm_reader = SharedMemReader()
        self.shm_reader.frame_ready.connect(self._on_frame)
        self.shm_reader.coords_ready.connect(lambda x, y: self.canvas.set_coords(x, y))
        self.shm_timer = QTimer()
        self.shm_timer.setInterval(40)  # ~25 fps
        self.shm_timer.timeout.connect(self._poll_shm)

        # connect timer to try connect until found
        self.connect_timer = QTimer()
        self.connect_timer.setInterval(500)
        self.connect_timer.timeout.connect(self._try_shm_connect)

        # restore last session params
        self._load_settings()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        sim_tab = QWidget()
        self._build_sim_tab(sim_tab)
        self.tabs.addTab(sim_tab, "◈  SIMULADOR CFD")

        ga_tab = QWidget()
        self._build_ga_tab(ga_tab)
        self.tabs.addTab(ga_tab, "⟳  OPTIMIZADOR GA")

    def _build_sim_tab(self, parent):
        root = QHBoxLayout(parent)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left: parameter panel + controls ─────────────────────────────
        left_widget = QWidget()
        left_widget.setFixedWidth(310)
        left_widget.setStyleSheet(f"background:{C['panel']}; border-right:1px solid {C['border']};")
        left_v = QVBoxLayout(left_widget)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(0)

        # header
        header = QWidget()
        header.setFixedHeight(42)
        header.setStyleSheet(f"""
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #0a1020, stop:0.7 #0c1828, stop:1 #0a1020);
            border-bottom:1px solid {C['border2']};
        """)
        hh = QHBoxLayout(header)
        hh.setContentsMargins(12, 0, 12, 0)
        logo = QLabel("◈")
        logo.setStyleSheet(f"color:{C['accent']}; font-size:18px; font-weight:bold;")
        title = QLabel("CFD SIMULATOR 2D")
        title.setStyleSheet(f"color:{C['text']}; font:700 12px 'Segoe UI'; letter-spacing:2px;")
        hh.addWidget(logo)
        hh.addWidget(title)
        hh.addStretch()
        left_v.addWidget(header)

        self.param_panel = ParameterPanel()
        left_v.addWidget(self.param_panel, 1)

        # control buttons - 2 rows for better layout
        ctrl = QWidget()
        ctrl.setStyleSheet(f"background:{C['panel']}; border-top:1px solid {C['border']};")
        ctrl_v = QVBoxLayout(ctrl)
        ctrl_v.setContentsMargins(8, 6, 8, 6)
        ctrl_v.setSpacing(4)
        
        # Row 1: EJECUTAR, DETENER
        row1 = QHBoxLayout()
        row1.setSpacing(4)
        self.btn_run = QPushButton("▶ EJECUTAR")
        self.btn_run.setObjectName("btn_run")
        self.btn_run.setFixedHeight(32)
        self.btn_stop = QPushButton("■ DETENER")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setFixedHeight(32)
        self.btn_stop.setEnabled(False)
        row1.addWidget(self.btn_run, 1)
        row1.addWidget(self.btn_stop, 1)
        
        # Row 2: RESET, GUARDAR
        row2 = QHBoxLayout()
        row2.setSpacing(4)
        self.btn_reset = QPushButton("↺ RESET")
        self.btn_reset.setFixedHeight(32)
        self.btn_reset.setToolTip("Limpiar visualización")
        self.btn_save_defaults = QPushButton("💾 GUARDAR")
        self.btn_save_defaults.setFixedHeight(32)
        self.btn_save_defaults.setToolTip("Guardar parámetros como defaults")
        row2.addWidget(self.btn_reset, 1)
        row2.addWidget(self.btn_save_defaults, 1)
        
        ctrl_v.addLayout(row1)
        ctrl_v.addLayout(row2)
        left_v.addWidget(ctrl)

        self.btn_run.clicked.connect(self._run)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_reset.clicked.connect(self._reset_view)
        self.btn_save_defaults.clicked.connect(self._save_defaults)

        # ── Center: visualization ─────────────────────────────────────────
        self.canvas = VisualizationCanvas()
        center_widget = QWidget()
        center_v = QVBoxLayout(center_widget)
        center_v.setContentsMargins(4, 4, 4, 4)
        center_v.setSpacing(0)
        center_v.addWidget(self.canvas)

        # ── Right: metrics panel ──────────────────────────────────────────
        self.metrics = MetricsPanel()

        # ── Splitter ──────────────────────────────────────────────────────
        root.addWidget(left_widget)
        root.addWidget(center_widget, 1)
        root.addWidget(self.metrics)

    def _build_ga_tab(self, parent):
        root = QHBoxLayout(parent)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left: GA params + buttons ─────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(310)
        left.setStyleSheet(f"background:{C['panel']}; border-right:1px solid {C['border']};")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)

        hdr = QWidget(); hdr.setFixedHeight(42)
        hdr.setStyleSheet(f"""
            background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0a1020,stop:0.7 #0c1828,stop:1 #0a1020);
            border-bottom:1px solid {C['border2']};""")
        hh = QHBoxLayout(hdr); hh.setContentsMargins(12, 0, 12, 0)
        logo = QLabel("⟳"); logo.setStyleSheet(f"color:{C['accent']};font-size:18px;font-weight:bold;")
        ttl  = QLabel("OPTIMIZADOR GENÉTICO")
        ttl.setStyleSheet(f"color:{C['text']};font:700 11px 'Segoe UI';letter-spacing:2px;")
        hh.addWidget(logo); hh.addWidget(ttl); hh.addStretch()
        lv.addWidget(hdr)

        self.ga_panel = GAParamPanel()
        lv.addWidget(self.ga_panel, 1)

        ctrl = QWidget(); ctrl.setFixedHeight(64)
        ctrl.setStyleSheet(f"background:{C['panel']};border-top:1px solid {C['border']};")
        ch = QHBoxLayout(ctrl); ch.setContentsMargins(8, 8, 8, 8); ch.setSpacing(6)
        self.ga_btn_run = QPushButton("▶  OPTIMIZAR")
        self.ga_btn_run.setObjectName("btn_run"); self.ga_btn_run.setFixedHeight(40)
        self.ga_btn_stop = QPushButton("■  DETENER")
        self.ga_btn_stop.setObjectName("btn_stop"); self.ga_btn_stop.setFixedHeight(40)
        self.ga_btn_stop.setEnabled(False)
        self.ga_btn_reset = QPushButton("↺"); self.ga_btn_reset.setFixedSize(40, 40)
        ch.addWidget(self.ga_btn_run, 2); ch.addWidget(self.ga_btn_stop, 2); ch.addWidget(self.ga_btn_reset, 0)
        lv.addWidget(ctrl)

        self.ga_btn_run.clicked.connect(self._run_ga)
        self.ga_btn_stop.clicked.connect(self._stop_ga)
        self.ga_btn_reset.clicked.connect(self._reset_ga)

        # ── Center: GA canvas ─────────────────────────────────────────────
        self.ga_canvas = GAVisualizationCanvas()
        cw = QWidget(); cv = QVBoxLayout(cw); cv.setContentsMargins(4, 4, 4, 4); cv.setSpacing(0)
        cv.addWidget(self.ga_canvas)

        # ── Right: GA metrics ─────────────────────────────────────────────
        self.ga_metrics = GAMetricsPanel()

        root.addWidget(left)
        root.addWidget(cw, 1)
        root.addWidget(self.ga_metrics)

    def _build_status_bar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.lbl_status = QLabel("◉  IDLE")
        self.lbl_status.setStyleSheet(f"color:{C['text3']}; padding:0 8px;")
        self.lbl_iter_sb = QLabel("")
        self.lbl_iter_sb.setStyleSheet(f"color:{C['text2']}; font:11px 'Consolas'; padding:0 8px;")
        self.lbl_fps = QLabel("")
        self.lbl_fps.setStyleSheet(f"color:{C['text3']}; font:11px 'Consolas'; padding:0 8px;")
        sb.addPermanentWidget(self.lbl_status)
        sb.addWidget(self.lbl_iter_sb)
        sb.addPermanentWidget(self.lbl_fps)
        self._fps_count = 0
        self._fps_t0 = time.time()
        self._last_fps = 0.0

    # ── simulation control ────────────────────────────────────────────────

    def _run(self):
        if self._process and self._process.poll() is None:
            return
        params = self.param_panel.get_params()
        self._total_iters = params.get('iteraciones', 2000)
        self._run_start_time = time.time()  # Registrar tiempo de inicio

        # compute Re
        v0x = params.get('v0x', 5.0) or 5.0
        chord = params.get('chord', 1.0) or 1.0
        nu = params.get('nu', 1.5e-5) or 1.5e-5
        re_val = v0x * chord / nu
        self.metrics.set_re(re_val)

        script = self._make_script(params)
        tf = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                         dir=str(SIM_DIR), prefix='_sim_run_')
        tf.write(script)
        tf.close()
        self._tmp_script = tf.name

        try:
            self._process = subprocess.Popen(
                [sys.executable, tf.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(SIM_DIR),
            )
        except Exception as e:
            self.metrics.append_log(f"[ERROR] No se pudo iniciar: {e}")
            return

        self._log_reader = LogReader(self._process)
        self._log_reader.line_received.connect(self.metrics.append_log)
        self._log_reader.start()

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("● RUNNING")
        self.lbl_status.setStyleSheet(f"color:{C['green']}; padding:0 8px; font-weight:600;")
        self.metrics.clear_log()
        self.metrics.append_log("[INFO] Simulación iniciada...")

        self.connect_timer.start()
        self.shm_reader.disconnect()

        # poll process for completion
        self._done_timer = QTimer()
        self._done_timer.setInterval(500)
        self._done_timer.timeout.connect(self._check_done)
        self._done_timer.start()

    def _stop(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self.metrics.append_log("[INFO] Señal de terminación enviada.")
        self._set_idle()

    def _set_idle(self):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("◉  IDLE")
        self.lbl_status.setStyleSheet(f"color:{C['text3']}; padding:0 8px;")
        self.connect_timer.stop()
        self.shm_timer.stop()
        self.shm_reader.disconnect()
        if hasattr(self, '_done_timer'):
            self._done_timer.stop()

    def _check_done(self):
        if self._process and self._process.poll() is not None:
            self.metrics.append_log(f"[INFO] Proceso terminado (código {self._process.returncode}).")
            self._set_idle()
            # cleanup temp script
            try:
                os.unlink(self._tmp_script)
            except Exception:
                pass

    def _try_shm_connect(self):
        if self.shm_reader.try_connect():
            self.connect_timer.stop()
            self.shm_timer.start()
            self.metrics.append_log(f"[INFO] Memoria compartida conectada ({self.shm_reader.ny}x{self.shm_reader.nx}).")

    def _poll_shm(self):
        self.shm_reader.poll()

    def _on_frame(self, speed, solid, iteration, cd, cl, alpha):
        self.canvas.update_field(speed, solid)
        if cd != 0.0 or cl != 0.0:
            self.canvas.update_coefs(iteration, cd, cl)
        title = f"|V|  —  iter {iteration}   α={alpha:.1f}°   Cd={cd:.5f}   Cl={cl:.5f}"
        self.canvas.set_title(title)

        self.metrics.update_coefs(cd, cl, alpha)
        elapsed = time.time() - self._run_start_time
        self.metrics.update_iter(iteration, self._total_iters, elapsed)

        self.lbl_iter_sb.setText(
            f"iter {iteration}/{self._total_iters}   Cd={cd:.5f}   Cl={cl:.5f}   α={alpha:.1f}°"
        )

        # FPS
        self._fps_count += 1
        dt = time.time() - self._fps_t0
        if dt >= 1.0:
            self._last_fps = self._fps_count / dt
            self._fps_count = 0
            self._fps_t0 = time.time()
        self.lbl_fps.setText(f"{self._last_fps:.0f} FPS")

    def _reset_view(self):
        self.canvas.reset()
        self.metrics.clear_log()
        self.metrics.update_iter(0, 0)

    def _save_defaults(self):
        """Guarda los parámetros actuales como defaults en gui_settings.json"""
        try:
            params = self.param_panel.get_params()
            # Convertir tuples a lists para JSON serialization
            for key in ('boundary_left', 'boundary_right', 'boundary_top', 'boundary_bottom'):
                if key in params and isinstance(params[key], tuple):
                    params[key] = list(params[key])
            ga_cfg = self.ga_panel.get_config()
            data = {'sim': params, 'ga': ga_cfg}
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            self.metrics.append_log("[OK] Parámetros guardados como defaults.")
        except Exception as e:
            self.metrics.append_log(f"[ERROR] No se pudo guardar: {e}")

    # ── GA control ────────────────────────────────────────────────────────

    def _run_ga(self):
        if self._ga_process and self._ga_process.poll() is None:
            return
        cfg = self.ga_panel.get_config()
        script = self._make_ga_script(cfg)
        tf = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                          dir=str(SIM_DIR), prefix='_ga_run_')
        tf.write(script); tf.close()
        self._ga_tmp_script = tf.name
        try:
            self._ga_process = subprocess.Popen(
                [sys.executable, tf.name],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(SIM_DIR))
        except Exception as e:
            self.ga_metrics.append_log(f"[ERROR] {e}")
            return

        self._ga_current_inds = []
        self._ga_pending_stats = {}
        self._ga_gen_num = 0
        self._ga_scatter_t = 0.0

        self._ga_log_reader = LogReader(self._ga_process)
        self._ga_log_reader.line_received.connect(self._on_ga_log_line)
        self._ga_log_reader.start()

        self.ga_btn_run.setEnabled(False)
        self.ga_btn_stop.setEnabled(True)
        self.ga_metrics.reset()
        self.ga_metrics.append_log("[INFO] GA iniciado...")
        self.ga_metrics.update_gen(0, cfg.get('generaciones', 10))

        self._ga_poll_timer = QTimer()
        self._ga_poll_timer.setInterval(2000)
        self._ga_poll_timer.timeout.connect(self._poll_ga_files)
        self._ga_poll_timer.start()

    def _stop_ga(self):
        if self._ga_process and self._ga_process.poll() is None:
            self._ga_process.terminate()
            self.ga_metrics.append_log("[INFO] Señal de terminación enviada.")
        self._set_ga_idle()

    def _set_ga_idle(self):
        self.ga_btn_run.setEnabled(True)
        self.ga_btn_stop.setEnabled(False)
        if hasattr(self, '_ga_poll_timer'):
            self._ga_poll_timer.stop()

    def _reset_ga(self):
        self.ga_canvas.reset()
        self.ga_metrics.reset()

    def _on_ga_log_line(self, line):
        self.ga_metrics.append_log(line)

        ind = parse_individual_line(line)
        if ind:
            self._ga_current_inds.append(ind)
            now = time.time()
            if now - self._ga_scatter_t > 0.8:
                self.ga_canvas.update_scatter(self._ga_current_inds)
                self._ga_scatter_t = now
            return

        summary = parse_gen_summary(line)
        if not summary:
            return
        self._ga_pending_stats.update(summary)
        total = self.ga_panel.get_config().get('generaciones', 10)

        if 'gen_done' in summary:
            self._ga_gen_num = summary['gen_done']
            self.ga_metrics.update_gen(self._ga_gen_num, total, self._ga_pending_stats)
            if self._ga_current_inds:
                self.ga_canvas.update_scatter(self._ga_current_inds)
            self._ga_current_inds = []
            self._ga_pending_stats = {}
        else:
            self.ga_metrics.update_gen(self._ga_gen_num, total, self._ga_pending_stats)

    def _poll_ga_files(self):
        if self._ga_process and self._ga_process.poll() is not None:
            self.ga_metrics.append_log(f"[INFO] GA terminado (código {self._ga_process.returncode}).")
            self._set_ga_idle()
            try: os.unlink(self._ga_tmp_script)
            except Exception: pass

        cfg = self.ga_panel.get_config()
        results_dir = SIM_DIR / cfg['directorio_resultados']
        hist_file = results_dir / 'historial_fitness.json'
        if hist_file.exists():
            try:
                with open(hist_file, 'r', encoding='utf-8') as f:
                    hist = json.load(f)
                if isinstance(hist, list) and hist:
                    self.ga_canvas.update_convergence(hist)
                    last = hist[-1]
                    total = cfg.get('generaciones', 10)
                    self.ga_metrics.update_gen(last.get('gen', 0) + 1, total, {
                        'mejor':    last.get('mejor', 0),
                        'media':    last.get('media', 0),
                        't_seg':    last.get('tiempo_seg', 0),
                        'evaluados': last.get('evaluados', 0),
                        'ia_desc':  last.get('descartados_ia', 0),
                        'vs_base':  last.get('vs_base', 0) or 0,
                    })
            except Exception:
                pass

        optimo_file = results_dir / 'OPTIMO_PARCIAL.dat'
        base_str = cfg['archivo_base']
        base_file = Path(base_str) if Path(base_str).is_absolute() else SIM_DIR / base_str
        if optimo_file.exists() and base_file.exists():
            try:
                cb = load_dat_profile(str(base_file))
                ct = load_dat_profile(str(optimo_file))
                if cb is not None and ct is not None:
                    self.ga_canvas.update_geometry(cb, ct)
            except Exception:
                pass

    def _make_ga_script(self, config):
        lines = [
            "import sys",
            f"sys.path.insert(0, r{repr(str(SIM_DIR))})",
            "import RunGA",
            "",
            "RunGA.CONFIG.update({",
        ]
        for k, v in config.items():
            lines.append(f"    {repr(k)}: {repr(v)},")
        lines += ["}", "", "RunGA.main()"]
        return "\n".join(lines)

    # ── script generation ─────────────────────────────────────────────────

    def _make_script(self, params):
        lines = [
            "import sys",
            f"sys.path.insert(0, r{repr(str(SIM_DIR))})",
            "from Simulador2D import main",
            "",
            "main(",
        ]
        for k, v in params.items():
            lines.append(f"    {k}={repr(v)},")
        lines.append(")")
        return "\n".join(lines)

    def _save_settings(self):
        try:
            params = self.param_panel.get_params()
            for key in ('boundary_left', 'boundary_right', 'boundary_top', 'boundary_bottom'):
                if key in params and isinstance(params[key], tuple):
                    params[key] = list(params[key])
            ga_cfg = self.ga_panel.get_config()
            data = {'sim': params, 'ga': ga_cfg}
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_settings(self):
        if not SETTINGS_FILE.exists():
            return
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Support both old format (flat dict) and new format ({'sim':..,'ga':..})
            if 'sim' in data:
                params = data['sim']
                ga_cfg = data.get('ga', {})
            else:
                params = data
                ga_cfg = {}
            for key in ('boundary_left', 'boundary_right', 'boundary_top', 'boundary_bottom'):
                if key in params and isinstance(params[key], list):
                    params[key] = tuple(params[key])
            self.param_panel.set_params(params)
            if ga_cfg:
                self.ga_panel.set_config(ga_cfg)
        except Exception:
            pass

    def closeEvent(self, event):
        self._save_settings()
        self._stop()
        self._stop_ga()
        if self._log_reader:
            self._log_reader.quit()
        if self._ga_log_reader:
            self._ga_log_reader.quit()
        event.accept()


# ─────────────────────────────  ENTRY POINT  ────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Simulador 2D CFD")
    app.setStyle("Fusion")

    # Apply dark palette as base so native widgets inherit it
    pal = QPalette()
    bg = QColor(C['bg'])
    pal.setColor(QPalette.ColorRole.Window, bg)
    pal.setColor(QPalette.ColorRole.WindowText, QColor(C['text']))
    pal.setColor(QPalette.ColorRole.Base, QColor(C['input_bg']))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(C['panel']))
    pal.setColor(QPalette.ColorRole.Text, QColor(C['text']))
    pal.setColor(QPalette.ColorRole.Button, QColor(C['panel2']))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(C['text']))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(C['select']))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(C['text']))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

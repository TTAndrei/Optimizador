"""Visor de malla y campos sobre la malla C curvilinea."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure


class Visor(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 5), tight_layout=True, facecolor="#f8fafc")
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self._mesh = None
        self._cbar = None
        self._X = self._Y = None
        self._info = None
        self._ultimo_snapshot = None
        self.modo = "velocidad"
        self.vista = "perfil"

    def mostrar_malla(self, X, Y, info=None, calidad=None):
        self._X, self._Y = np.asarray(X), np.asarray(Y)
        self._info = info or {}
        self._ultimo_snapshot = None
        self._limpiar()
        self._mesh = None

        lineas = _lineas_decimadas(self._X, self._Y)
        coleccion = LineCollection(lineas, colors="#64748b", linewidths=0.45, alpha=0.42)
        self.ax.add_collection(coleccion)

        i0, i1 = self._rango_perfil()
        self.ax.plot(self._X[0, i0:i1], self._Y[0, i0:i1], color="#05070a", lw=1.4,
                     solid_capstyle="round", label="perfil")
        self._dibujar_estela_tenue()
        self._estilo_ejes("Preview de malla C")
        self._aplicar_limites(self._X, self._Y)
        self.ax.legend(loc="upper right", frameon=True, framealpha=0.9)
        self.draw_idle()

    def mostrar_snapshot(self, snapshot):
        self._ultimo_snapshot = snapshot
        self._X, self._Y = np.asarray(snapshot.X), np.asarray(snapshot.Y)
        campo, titulo, mapa, divergente = self._campo(snapshot)
        if self._mesh is None:
            self._limpiar()
            self._mesh = self.ax.pcolormesh(
                self._X,
                self._Y,
                np.asarray(campo),
                shading="auto",
                cmap=mapa,
                edgecolors="none",
                linewidth=0.0,
                antialiased=False,
                rasterized=True,
            )
            self._cbar = self.fig.colorbar(self._mesh, ax=self.ax, fraction=0.032, pad=0.015)
        else:
            self._mesh.set_array(np.asarray(campo).ravel())
            self._mesh.set_cmap(mapa)
        vmin, vmax = _limites_robustos(campo, divergente=divergente)
        self._mesh.set_clim(vmin, vmax)
        if self._cbar is not None:
            self._cbar.set_label(titulo)
            self._cbar.update_normal(self._mesh)
        self._dibujar_perfil_sobre_campo()
        self._estilo_ejes(f"{titulo} | iteracion {snapshot.metrics.get('iter', 0)}")
        self._aplicar_limites(self._X, self._Y)
        self.draw_idle()

    def redibujar_ultimo(self):
        if self._ultimo_snapshot is not None:
            self._mesh = None
            self.mostrar_snapshot(self._ultimo_snapshot)
        elif self._X is not None and self._Y is not None:
            self.mostrar_malla(self._X, self._Y, self._info)

    def _campo(self, snapshot):
        if self.modo == "presion":
            return snapshot.p, "Presion", "coolwarm", True
        if self.modo == "u":
            return snapshot.u, "Componente u", "viridis", False
        if self.modo == "v":
            return snapshot.v, "Componente v", "coolwarm", True
        return np.hypot(snapshot.u, snapshot.v), "Velocidad", "turbo", False

    def _limpiar(self):
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except ValueError:
                pass
            self._cbar = None
        self.ax.clear()

    def _rango_perfil(self):
        if isinstance(self._info, dict) and "perfil" in self._info:
            i0, i1 = self._info["perfil"]
            return int(i0), int(i1)
        return 0, self._X.shape[1]

    def _dibujar_estela_tenue(self):
        if not isinstance(self._info, dict):
            return
        for clave in ("estela_inf", "estela_sup"):
            if clave not in self._info:
                continue
            i0, i1 = self._info[clave]
            self.ax.plot(self._X[0, i0:i1], self._Y[0, i0:i1], color="#94a3b8",
                         lw=0.8, ls="--", alpha=0.7)

    def _dibujar_perfil_sobre_campo(self):
        if self._X is None or self._Y is None:
            return
        i0, i1 = self._rango_perfil()
        for linea in list(self.ax.lines):
            linea.remove()
        self.ax.plot(self._X[0, i0:i1], self._Y[0, i0:i1], color="#0b0f17", lw=1.1,
                     solid_capstyle="round")

    def _estilo_ejes(self, titulo):
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title(titulo)
        self.ax.set_xlabel("x/c")
        self.ax.set_ylabel("y/c")
        self.ax.grid(color="#d5dde7", alpha=0.55, linewidth=0.5)
        self.ax.set_facecolor("#ffffff")

    def _aplicar_limites(self, X, Y):
        if self.vista == "dominio":
            self.ax.set_xlim(float(np.nanmin(X)), float(np.nanmax(X)))
            self.ax.set_ylim(float(np.nanmin(Y)), float(np.nanmax(Y)))
            return
        i0, i1 = self._rango_perfil()
        xp = X[0, i0:i1]
        yp = Y[0, i0:i1]
        if self.vista == "estela":
            x0 = max(float(np.nanmin(xp)) - 0.2, float(np.nanmin(X)))
            self.ax.set_xlim(x0, float(np.nanmax(X)))
            alto = max(0.55, min(float(np.nanmax(Y)), 1.5))
            self.ax.set_ylim(-alto, alto)
            return
        margen_x = max(0.12, 0.08 * (float(np.nanmax(xp)) - float(np.nanmin(xp))))
        margen_y = max(0.18, 3.0 * (float(np.nanmax(yp)) - float(np.nanmin(yp)) + 1e-6))
        self.ax.set_xlim(float(np.nanmin(xp)) - margen_x, float(np.nanmax(xp)) + margen_x)
        self.ax.set_ylim(float(np.nanmin(yp)) - margen_y, float(np.nanmax(yp)) + margen_y)


def _lineas_decimadas(X, Y, max_i=90, max_j=65):
    paso_i = max(1, int(np.ceil(X.shape[1] / max_i)))
    paso_j = max(1, int(np.ceil(X.shape[0] / max_j)))
    indices_j = sorted(set(range(0, X.shape[0], paso_j)) | {X.shape[0] - 1})
    indices_i = sorted(set(range(0, X.shape[1], paso_i)) | {X.shape[1] - 1})
    lineas = []
    for j in indices_j:
        lineas.append(np.column_stack([X[j, :], Y[j, :]]))
    for i in indices_i:
        lineas.append(np.column_stack([X[:, i], Y[:, i]]))
    return lineas


def _limites_robustos(campo, divergente=False):
    valores = np.asarray(campo, dtype=float)
    valores = valores[np.isfinite(valores)]
    if valores.size == 0:
        return 0.0, 1.0
    bajo, alto = np.percentile(valores, [2.0, 98.0])
    if divergente:
        escala = max(abs(float(bajo)), abs(float(alto)), 1e-12)
        return -escala, escala
    if alto <= bajo:
        centro = float(alto)
        delta = max(abs(centro) * 0.05, 1e-9)
        return centro - delta, centro + delta
    return float(bajo), float(alto)

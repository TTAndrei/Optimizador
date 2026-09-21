"""Historia de fuerzas y rendimiento durante la simulacion."""
from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class GraficaFuerzas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(7, 2.8), tight_layout=True, facecolor="#f8fafc")
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self._t = []
        self._cl = []
        self._cd = []
        self._line_cl = None
        self._line_cd = None
        self._preparar()

    def actualizar(self, metricas):
        if "Cl" not in metricas:
            return
        self._t.append(float(metricas.get("t", 0.0)))
        self._cl.append(float(metricas["Cl"]))
        self._cd.append(float(metricas["Cd"]))
        if self._line_cl is None or self._line_cd is None:
            self._preparar()
        self._line_cl.set_data(self._t, self._cl)
        self._line_cd.set_data(self._t, self._cd)
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw_idle()

    def reiniciar(self):
        self._t.clear()
        self._cl.clear()
        self._cd.clear()
        self._preparar()
        self.draw_idle()

    def _preparar(self):
        self.ax.clear()
        self._line_cl, = self.ax.plot(self._t, self._cl, color="#1d4ed8", label="Cl")
        self._line_cd, = self.ax.plot(self._t, self._cd, color="#c2410c", label="Cd")
        self.ax.set_xlabel("t")
        self.ax.set_ylabel("coeficiente")
        self.ax.grid(color="#d5dde7", alpha=0.7, linewidth=0.5)
        self.ax.set_facecolor("#ffffff")
        self.ax.legend(loc="best")

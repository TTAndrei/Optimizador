"""Modelo serializable de una corrida del solver curvilineo."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from curvo import malla


class CasoError(ValueError):
    """Configuracion de corrida invalida."""


@dataclass
class Caso:
    """Configuracion completa que comparten la GUI y el proceso solver."""

    perfil: str = "NACA_0012_sharp"
    malla: dict[str, Any] = field(default_factory=lambda: {
        "dn_pared": 8.0e-5,
        "crecimiento": 1.12,
        "distancia_lejos": 8.0,
        "n_sup": 256,
        "n_estela": 64,
        "x_out": 18.0,
        "razon_le": 0.08,
        "razon_te": 0.15,
        "ancho_le": 0.05,
        "ancho_te": 0.05,
        "ajustar_dn_pared": 0.0,
        "cola_te": 4.0,
        "dn_max": None,
        "n_capas_pared": 15,
        "crecimiento_pared": 1.0,
        "aspecto_max": 100.0,
        "mezcla_volumen": 0.30,
        "disipacion": 0.30,
        "disipacion_curva": 8.0,
    })
    solver: dict[str, Any] = field(default_factory=lambda: {
        "nu": 1.0e-5,
        "u_inf": 1.0,
        "alfa": 5.0,
        "dt": 2.5e-3,
        "bdf2": False,
        "correcciones": 1,
        "correcciones_p": 1,
        "turbulento": True,
        "nu_tilde_inf": 3.0,
    })
    ejecucion: dict[str, Any] = field(default_factory=lambda: {
        "pasos": 1000,
        "cada_historia": 20,
        "cada_campo": 40,
        "tol_fuerzas": 0.0,
        "backend": "cupy",
        "dtype": "float32",
        "salida": "monitor",
        "directorio": "",
    })

    def validar(self, raiz: str | Path | None = None) -> None:
        """Valida la configuracion sin generar la malla ni inicializar la GPU."""
        errores: list[str] = []
        ruta = self.ruta_perfil(raiz)
        if not ruta.is_file():
            errores.append(f"no existe el perfil: {ruta}")

        positivos = (
            ("malla.dn_pared", self.malla.get("dn_pared")),
            ("malla.crecimiento", self.malla.get("crecimiento")),
            ("malla.distancia_lejos", self.malla.get("distancia_lejos")),
            ("malla.n_sup", self.malla.get("n_sup")),
            ("malla.n_estela", self.malla.get("n_estela")),
            ("malla.x_out", self.malla.get("x_out")),
            ("malla.razon_le", self.malla.get("razon_le")),
            ("malla.razon_te", self.malla.get("razon_te")),
            ("malla.ancho_le", self.malla.get("ancho_le")),
            ("malla.ancho_te", self.malla.get("ancho_te")),
            ("malla.n_capas_pared", self.malla.get("n_capas_pared")),
            ("malla.crecimiento_pared", self.malla.get("crecimiento_pared")),
            ("malla.aspecto_max", self.malla.get("aspecto_max")),
            ("malla.disipacion", self.malla.get("disipacion")),
            ("solver.nu", self.solver.get("nu")),
            ("solver.u_inf", self.solver.get("u_inf")),
            ("solver.dt", self.solver.get("dt")),
            ("ejecucion.pasos", self.ejecucion.get("pasos")),
        )
        for nombre, valor in positivos:
            if valor is None or valor <= 0:
                errores.append(f"{nombre} debe ser positivo")
        for nombre in ("cada_historia", "cada_campo"):
            valor = self.ejecucion.get(nombre)
            if valor is None or int(valor) != valor or valor <= 0:
                errores.append(f"ejecucion.{nombre} debe ser un entero positivo")
        tol_f = self.ejecucion.get("tol_fuerzas")
        if tol_f is None or tol_f < 0:
            errores.append("ejecucion.tol_fuerzas debe ser mayor o igual que cero")
        for nombre in ("n_sup", "n_estela", "n_capas_pared"):
            valor = self.malla.get(nombre)
            if valor is None or int(valor) != valor or valor <= 0:
                errores.append(f"malla.{nombre} debe ser un entero positivo")
        for nombre in ("correcciones", "correcciones_p"):
            valor = self.solver.get(nombre)
            if valor is None or int(valor) != valor or valor <= 0:
                errores.append(f"solver.{nombre} debe ser un entero positivo")
        dn_max = self.malla.get("dn_max")
        if dn_max is not None and dn_max <= 0:
            errores.append("malla.dn_max debe ser positivo o vacio")
        for nombre in ("ajustar_dn_pared", "mezcla_volumen", "disipacion_curva",
                       "cola_te"):
            valor = self.malla.get(nombre)
            if valor is None or valor < 0:
                errores.append(f"malla.{nombre} debe ser mayor o igual que cero")
        if self.ejecucion.get("backend") not in {"numpy", "cupy"}:
            errores.append("ejecucion.backend debe ser numpy o cupy")
        if self.ejecucion.get("dtype") not in {"float32", "float64"}:
            errores.append("ejecucion.dtype debe ser float32 o float64")
        if self.ejecucion.get("salida") not in {"monitor", "grabar", "ninguno"}:
            errores.append("ejecucion.salida no es valida")
        if errores:
            raise CasoError("; ".join(errores))

    def ruta_perfil(self, raiz: str | Path | None = None) -> Path:
        base = Path(raiz) if raiz is not None else Path(__file__).resolve().parents[1]
        ruta = Path(self.perfil)
        if ruta.is_absolute():
            return ruta
        if ruta.is_file():
            return ruta
        en_profiles = base / "profiles" / ruta
        if en_profiles.is_file():
            return en_profiles
        return base / ruta

    def argumentos_malla(self) -> dict[str, Any]:
        """Devuelve solo argumentos aceptados por `generar_c`."""
        valores = dict(self.malla)
        if valores.get("dn_max") is None:
            valores["dn_max"] = np.inf
        return valores

    def a_dict(self) -> dict[str, Any]:
        valores = asdict(self)
        if valores["malla"].get("dn_max") is None:
            valores["malla"]["dn_max"] = None
        return valores

    def guardar(self, ruta: str | Path) -> None:
        self.validar()
        Path(ruta).write_text(json.dumps(self.a_dict(), indent=2), encoding="utf-8")

    @classmethod
    def cargar(cls, ruta: str | Path) -> "Caso":
        datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
        base = cls()
        if "malla" in datos:
            base.malla.update(datos["malla"])
        if "solver" in datos:
            base.solver.update(datos["solver"])
        if "ejecucion" in datos:
            base.ejecucion.update(datos["ejecucion"])
        if "perfil" in datos:
            base.perfil = datos["perfil"]
        caso = base
        caso.validar()
        return caso


def previsualizar_malla(caso: Caso, raiz: str | Path | None = None):
    """Genera la malla y su diagnostico para la vista previa de la GUI."""
    caso.validar(raiz)
    px, py = malla.leer_dat(caso.ruta_perfil(raiz))
    X, Y, info = malla.generar_c(px, py, **caso.argumentos_malla())
    calidad = malla.calidad(X, Y, perfil=info)
    return X, Y, info, calidad

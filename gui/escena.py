"""Modelo de datos de una escena: geometria, fronteras, malla y salida.

Es lo que la GUI edita y lo que el solver lee. Deliberadamente sin PySide6 ni
CuPy: se serializa a JSON, se puede versionar, se puede escribir a mano y se
puede testear sin abrir una ventana ni tocar la GPU.

Por que una escena y no veinte kwargs mas: `main()` ya tiene ~200 argumentos.
Lo que la geometria arbitraria anade —N contornos con su rol y su tipo de pared,
y una lista de parches de frontera por tramo del perimetro— no cabe como
argumentos sueltos sin volverlo ilegible. Entra por `escena=` y nada mas.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

TIPOS_BC = ("inflow", "outflow", "slip", "noslip")
LADOS = ("left", "right", "top", "bottom")


@dataclass
class Contorno:
    """Un poligono cerrado, en metros.

    rol="exterior" -> el fluido queda DENTRO y todo lo de fuera es solido (un
    tunel, un conducto, una tobera). rol="cuerpo" -> solido dentro del poligono
    (un perfil, un cilindro, un obstaculo).
    """
    x: list
    y: list
    rol: str = "cuerpo"              # "cuerpo" | "exterior"
    pared: str = "noslip"            # "noslip" | "slip"
    nombre: str = ""
    es_perfil: bool = False          # habilita Kutta, Cp y cuerda; ver plan §5

    def arrays(self):
        return np.asarray(self.x, float), np.asarray(self.y, float)

    def bbox(self):
        x, y = self.arrays()
        return float(x.min()), float(y.min()), float(x.max()), float(y.max())


@dataclass
class Parche:
    """Un tramo del perimetro de la caja con su condicion de frontera.

    `desde`/`hasta` van en coordenadas FISICAS a lo largo del lado (y para
    left/right, x para top/bottom). None = el extremo del lado. Asi el parche no
    depende de la malla y sobrevive a un cambio de dx.
    """
    lado: str
    tipo: str
    valor: object = None             # inflow: (u,v) o u; outflow: presion
    desde: float = None
    hasta: float = None

    def __post_init__(self):
        if self.lado not in LADOS:
            raise ValueError(f"lado {self.lado!r} no es uno de {LADOS}")
        if self.tipo not in TIPOS_BC:
            raise ValueError(f"tipo {self.tipo!r} no es uno de {TIPOS_BC}")


@dataclass
class Escena:
    Lx: float = 24.0
    Ly: float = 16.0
    dx_min: float = 0.004
    factor_expansion: float = 1.1
    # Caja de refinado dibujada a mano (x0, x1, y0, y1). None = derivarla del
    # bbox de los cuerpos. Manda sobre lo automatico: en una tobera lo que
    # interesa es la garganta, no el bbox del contorno.
    refinado: tuple = None
    contornos: list = field(default_factory=list)
    parches: list = field(default_factory=list)
    # Resto de kwargs de main() (Re, turbulencia, presupuesto, ...).
    solver: dict = field(default_factory=dict)
    modo_salida: str = "monitor"     # "ninguno" | "monitor" | "grabar"

    # ------------------------------------------------------------------
    def bbox_cuerpos(self):
        """bbox de los cuerpos sumergidos. El contorno exterior NO cuenta: su
        bbox es practicamente el dominio entero y refinar todo eso hace explotar
        el numero de celdas."""
        cajas = [c.bbox() for c in self.contornos if c.rol == "cuerpo"]
        if not cajas:
            return None
        return (min(b[0] for b in cajas), min(b[1] for b in cajas),
                max(b[2] for b in cajas), max(b[3] for b in cajas))

    def banda_fina(self):
        """(x0, x1, y0, y1) de la zona de malla fina.

        La extension aguas abajo (1.5 veces la dimension mayor) es la misma que
        ya usa `wake_refinement_mode="long_fine_x"` en el solver: la estela es
        lo que hay que resolver, no el cuerpo.
        """
        if self.refinado is not None:
            return tuple(self.refinado)
        bb = self.bbox_cuerpos()
        if bb is None:
            return None
        x0, y0, x1, y1 = bb
        L = max(x1 - x0, y1 - y0)
        return (x0 - 0.25 * L, x1 + 1.5 * L, y0 - 0.5 * L, y1 + 0.5 * L)

    def avisos(self):
        """Comprobaciones baratas, antes de gastar GPU. No mira la malla: las
        que necesitan la mascara rasterizada van en la vista de malla."""
        av = []
        tipos = {p.tipo for p in self.parches}
        if "inflow" not in tipos:
            av.append("No hay ningun parche de entrada (inflow): el flujo no "
                      "entra por ningun sitio.")
        if "outflow" not in tipos:
            av.append("No hay ningun parche de salida (outflow): sin presion "
                      "fijada en ninguna frontera, el Poisson es singular.")
        presiones = {p.valor for p in self.parches if p.tipo == "outflow"}
        if len(presiones) > 1:
            av.append(f"Hay {len(presiones)} presiones de salida distintas. El "
                      "solver solo admite una: fixed_pressure_value es escalar.")
        exteriores = [c for c in self.contornos if c.rol == "exterior"]
        if len(exteriores) > 1:
            av.append(f"Hay {len(exteriores)} contornos exteriores; solo puede "
                      "haber uno.")
        for c in exteriores:
            x0, y0, x1, y1 = c.bbox()
            if x0 > 0 and x1 < self.Lx:
                av.append(f"El contorno exterior '{c.nombre or '?'}' no corta "
                          "el perimetro de la caja: el dominio queda cerrado y "
                          "no hay por donde entrar ni salir.")
        for c in self.contornos:
            x, y = c.arrays()
            if np.hypot(x[0] - x[-1], y[0] - y[-1]) > 1e-9:
                av.append(f"El contorno '{c.nombre or '?'}' no esta cerrado.")
        if exteriores and self.solver.get("turb_model") == "sa":
            av.append("Con contorno exterior y modelo SA, la distancia a pared "
                      "sale mal fuera de la banda fina (se cuenta en indices de "
                      "celda sobre una malla estirada). Usar wale o none.")
        return av

    # ------------------------------------------------------------------
    def a_dict(self):
        d = asdict(self)
        d["contornos"] = [
            {**asdict(c), "x": list(map(float, c.x)), "y": list(map(float, c.y))}
            for c in self.contornos]
        d["parches"] = [asdict(p) for p in self.parches]
        return d

    def guardar(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.a_dict(), f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def desde_dict(cls, d):
        d = dict(d)
        d["contornos"] = [Contorno(**c) for c in d.get("contornos", [])]
        d["parches"] = [Parche(**p) for p in d.get("parches", [])]
        if d.get("refinado") is not None:
            d["refinado"] = tuple(d["refinado"])
        return cls(**d)

    @classmethod
    def cargar(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.desde_dict(json.load(f))

    @classmethod
    def desde_dxf(cls, path_dxf, **kw):
        """Atajo: importa el DXF y arma una escena con los cuatro lados por
        defecto (entrada a la izquierda, salida a la derecha, slip arriba y
        abajo), que es lo que quiere el 90% de los casos."""
        from geom_import import cargar_dxf
        esc = cls(**kw)
        tol = kw.get("dx_min", esc.dx_min) / 4.0
        for i, l in enumerate(cargar_dxf(path_dxf, tol_cordal=tol)):
            esc.contornos.append(Contorno(
                x=list(l["x"]), y=list(l["y"]), rol=l["rol"],
                pared=l["pared"], nombre=f"{l['capa']}_{i}"))
        esc.parches = [
            Parche("left", "inflow", (1.0, 0.0)),
            Parche("right", "outflow", 0.0),
            Parche("top", "slip"),
            Parche("bottom", "slip"),
        ]
        return esc

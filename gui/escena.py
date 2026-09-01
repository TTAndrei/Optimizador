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

# Quien define las paredes del dominio. Son dos casos de uso distintos y no se
# pueden mezclar sin que salga geometria colgando dentro de la caja:
#   "dxf"  -> el DXF trae el contorno exterior (tunel, conducto, tobera). La
#             caja ES su bbox y Lx/Ly no son del usuario.
#   "caja" -> el DXF trae solo el objeto (perfil, cilindro). La caja la ponemos
#             nosotros y el objeto se coloca dentro donde se quiera.
MODOS_DOMINIO = ("caja", "dxf")


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
    # Traslacion respecto de las coordenadas del DXF, en metros. Se guarda
    # aparte y no se suma a x/y para que el original siga siendo el del CAD y
    # mover el cuerpo sea reversible y legible en el JSON.
    off_x: float = 0.0
    off_y: float = 0.0

    def arrays(self):
        return (np.asarray(self.x, float) + self.off_x,
                np.asarray(self.y, float) + self.off_y)

    def bbox(self):
        x, y = self.arrays()
        return float(x.min()), float(y.min()), float(x.max()), float(y.max())

    def centro(self):
        x0, y0, x1, y1 = self.bbox()
        return 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    def mover_centro_a(self, cx, cy):
        x, y = self.centro()
        self.off_x += float(cx) - x
        self.off_y += float(cy) - y


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
    dy_min: float = None             # None = igual que dx_min
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
    modo_dominio: str = "caja"       # ver MODOS_DOMINIO

    # ------------------------------------------------------------------
    @property
    def dy(self):
        return float(self.dy_min) if self.dy_min else float(self.dx_min)

    def exteriores(self):
        return [c for c in self.contornos if c.rol == "exterior"]

    def bbox_exterior(self):
        cajas = [c.bbox() for c in self.exteriores()]
        if not cajas:
            return None
        return (min(b[0] for b in cajas), min(b[1] for b in cajas),
                max(b[2] for b in cajas), max(b[3] for b in cajas))

    def autoconfigurar_dominio(self):
        """Elige el modo al importar y encaja la caja. Devuelve un texto.

        Si el DXF trae contorno exterior, ese contorno ES el dominio. Si solo
        trae cuerpos, la caja se dimensiona a 24x16 veces el cuerpo, que es la
        proporcion del dominio C con el que estan hechas las polares."""
        if self.exteriores():
            self.modo_dominio = "dxf"
        else:
            self.modo_dominio = "caja"
            bb = self.bbox_cuerpos()
            if bb is not None:
                L = max(bb[2] - bb[0], bb[3] - bb[1])
                self.Lx, self.Ly = 24.0 * L, 16.0 * L
        return self.ajustar_dominio()

    def ajustar_dominio(self):
        """Encaja geometria y caja segun el modo. Devuelve un texto de lo hecho.

        En modo "dxf" reescribe Lx/Ly: cualquier otro valor deja pared solida
        colgando dentro del dominio y las BC del perimetro escritas por fuera de
        ella, que es como se ve un conducto que fuga por arriba.
        En modo "caja" no toca Lx/Ly, solo recoloca el objeto."""
        if self.modo_dominio == "dxf":
            bb = self.bbox_exterior()
            if bb is None:
                return ("Modo dxf sin contorno exterior: marca uno como "
                        "exterior o pasa a modo caja.")
            for c in self.contornos:
                c.off_x -= bb[0]
                c.off_y -= bb[1]
            self.Lx, self.Ly = bb[2] - bb[0], bb[3] - bb[1]
            return f"Caja fijada por el DXF: Lx={self.Lx:g}, Ly={self.Ly:g}."
        return self.centrar_cuerpos()

    def centrar_cuerpos(self, fx=0.25, fy=0.5):
        """Lleva el conjunto de cuerpos a (fx*Lx, fy*Ly) manteniendo sus
        posiciones relativas. Un cuarto de dominio por delante y tres cuartos de
        estela es el reparto del caso del TFG."""
        bb = self.bbox_cuerpos()
        if bb is None:
            return "No hay cuerpos que colocar."
        dx = fx * self.Lx - 0.5 * (bb[0] + bb[2])
        dy = fy * self.Ly - 0.5 * (bb[1] + bb[3])
        for c in self.contornos:
            if c.rol == "cuerpo":
                c.off_x += dx
                c.off_y += dy
        return (f"Objeto centrado en ({fx * self.Lx:g}, {fy * self.Ly:g}) "
                f"de una caja {self.Lx:g}x{self.Ly:g}.")

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
            # Conducto sin cuerpos: no hay nada que centrar, todo fino.
            return (0.0, self.Lx, 0.0, self.Ly) if self.modo_dominio == "dxf" else None
        x0, y0, x1, y1 = bb
        L = max(x1 - x0, y1 - y0)
        banda = (x0 - 0.25 * L, x1 + 1.5 * L, y0 - 0.5 * L, y1 + 0.5 * L)
        if self.modo_dominio == "dxf":
            # En un conducto la cortadura ocupa TODA la seccion, no una capa
            # pegada a la pared, asi que la banda fina tiene que cubrirla
            # entera. Es lo mismo que exige la distancia a pared de SA, y ademas
            # es lo que evita que dy se estire hasta la pared del conducto.
            banda = (banda[0], banda[1], 0.0, self.Ly)
        return banda

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
        ext = self.exteriores()
        if self.modo_dominio not in MODOS_DOMINIO:
            av.append(f"Modo de dominio {self.modo_dominio!r} desconocido.")
        elif self.modo_dominio == "dxf":
            if not ext:
                av.append("Modo «dxf» pero ningun contorno esta marcado como "
                          "exterior: no hay paredes que definan la caja. "
                          "Marca uno como exterior o pasa a modo «caja».")
            if len(ext) > 1:
                av.append(f"Hay {len(ext)} contornos exteriores; solo puede "
                          "haber uno.")
            bb = self.bbox_exterior()
            # El aviso que faltaba: subir Ly a mano con un conducto importado
            # deja la pared del conducto flotando dentro de la caja, con las BC
            # del perimetro escritas por encima. Se ve como una pared negra que
            # no frena el flujo.
            if bb is not None:
                tol = max(1e-9, 1e-6 * max(self.Lx, self.Ly))
                if (abs(bb[0]) > tol or abs(bb[1]) > tol
                        or abs(bb[2] - self.Lx) > tol
                        or abs(bb[3] - self.Ly) > tol):
                    av.append(
                        f"La caja ({self.Lx:g} x {self.Ly:g}) no coincide con "
                        f"el contorno exterior ([{bb[0]:g}, {bb[2]:g}] x "
                        f"[{bb[1]:g}, {bb[3]:g}]). En modo «dxf» las paredes "
                        f"son las del DXF: pulsa «Encajar dominio» o pasa a "
                        f"modo «caja».")
        elif ext:
            av.append(
                f"Modo «caja» con {len(ext)} contorno(s) marcados como "
                f"exterior. En este modo el dominio lo define Lx/Ly, asi que "
                f"o los pasas a «cuerpo» o cambias a modo «dxf».")

        for c in self.contornos:
            if c.rol != "cuerpo":
                continue
            x0, y0, x1, y1 = c.bbox()
            if x0 < 0 or y0 < 0 or x1 > self.Lx or y1 > self.Ly:
                av.append(
                    f"El cuerpo '{c.nombre or '?'}' ([{x0:g}, {x1:g}] x "
                    f"[{y0:g}, {y1:g}]) se sale de la caja {self.Lx:g} x "
                    f"{self.Ly:g}. Muevelo o agranda el dominio.")
        for c in self.contornos:
            x, y = c.arrays()
            if np.hypot(x[0] - x[-1], y[0] - y[-1]) > 1e-9:
                av.append(f"El contorno '{c.nombre or '?'}' no esta cerrado.")
        return av

    # ------------------------------------------------------------------
    def a_dict(self):
        d = asdict(self)
        d["contornos"] = [
            {**asdict(c), "x": list(map(float, c.x)), "y": list(map(float, c.y))}
            for c in self.contornos]
        d["parches"] = [asdict(p) for p in self.parches]
        return d

    def contornos_para_malla(self):
        """[(contorno, x, y)] con las coordenadas finales que ve el rasterizador.

        No muta nada: se llama tantas veces como haga falta y da lo mismo.

        Lo unico que cambia respecto de `arrays()` es que en modo "dxf" las
        BOCAS del contorno exterior se sacan fuera de la caja. La caja se encaja
        al bbox del contorno, asi que sus tramos verticales caen exactamente
        sobre las columnas del borde; y como una celda cuyo centro cae sobre el
        contorno es pared, esas columnas se rasterizarian solidas y no entraria
        flujo. Solo en x: en y el contorno SI es pared y tiene que coincidir con
        el borde. Es la misma convencion que ya usa formas_dominio.canal().
        """
        salida = []
        eps = 1e-3 * max(self.Lx, self.Ly)
        tol = 1e-9 * max(1.0, self.Lx)
        for c in self.contornos:
            x, y = c.arrays()
            if self.modo_dominio == "dxf" and c.rol == "exterior":
                x = np.where(x <= tol, -eps, x)
                x = np.where(x >= self.Lx - tol, self.Lx + eps, x)
            salida.append((c, x, y))
        return salida

    def dict_solver(self):
        """Como `a_dict()` pero con las coordenadas que rasteriza el solver ya
        resueltas: offsets sumados y bocas extendidas. El solver rasteriza
        `c["x"]` tal cual; en el JSON se guarda el original para poder seguir
        moviendo el objeto."""
        d = self.a_dict()
        for cd, (c, x, y) in zip(d["contornos"], self.contornos_para_malla()):
            cd["x"], cd["y"] = x.tolist(), y.tolist()
            cd["off_x"] = cd["off_y"] = 0.0
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
        esc.autoconfigurar_dominio()
        esc.parches = parches_por_defecto(esc.modo_dominio)
        return esc


def parches_por_defecto(modo_dominio="caja"):
    """Entrada a la izquierda, salida a la derecha. Arriba y abajo depende del
    caso: con contorno exterior importado esos lados SON pared del conducto y
    ponerlos slip deja escapar el flujo por ellos."""
    tapa = "noslip" if modo_dominio == "dxf" else "slip"
    return [
        Parche("left", "inflow", (1.0, 0.0)),
        Parche("right", "outflow", 0.0),
        Parche("top", tapa),
        Parche("bottom", tapa),
    ]

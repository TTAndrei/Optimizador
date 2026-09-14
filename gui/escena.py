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
TIPOS_PARED = ("noslip", "slip")
LADOS = ("left", "right", "top", "bottom")

# Giro maximo, en grados, para que dos segmentos consecutivos del DXF sigan
# siendo la misma arista. El DXF llega poligonizado —un arco son doscientos
# segmentos— asi que la unidad que se pincha y a la que se le pone condicion no
# puede ser el segmento: tiene que ser el tramo recto que un humano llamaria
# "la pared inclinada" o "el techo". 8 grados separa esquinas reales sin partir
# un arco flatteneado con tol_cordal en mil aristas.
ANG_ARISTA = 8.0

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
    # Excepciones a `pared`, por arista: {"3": "slip"}. Las claves son str
    # porque esto se serializa a JSON y ahi no hay claves enteras. `pared` sigue
    # siendo el valor por defecto de todo lo que no aparezca aqui, asi que una
    # escena vieja se lee igual y un contorno homogeneo no engorda el JSON.
    paredes: dict = field(default_factory=dict)
    # Traslacion respecto de las coordenadas del DXF, en metros. Se guarda
    # aparte y no se suma a x/y para que el original siga siendo el del CAD y
    # mover el cuerpo sea reversible y legible en el JSON.
    off_x: float = 0.0
    off_y: float = 0.0

    def arrays(self):
        return (np.asarray(self.x, float) + self.off_x,
                np.asarray(self.y, float) + self.off_y)

    # --------------------------------------------------------------- aristas
    def cerrado(self):
        x, y = self.arrays()
        return len(x) > 2 and np.hypot(x[0] - x[-1], y[0] - y[-1]) < 1e-12

    def _dueno_de_segmento(self, ang_tol=ANG_ARISTA):
        """Indice de arista de cada segmento del contorno.

        Un segmento empieza arista nueva cuando gira mas de `ang_tol` respecto
        del anterior. En un poligono cerrado la lista es ciclica: se rota para
        que la arista 0 empiece en una esquina de verdad y no en el punto por el
        que el DXF cerro el lazo, que cae en medio de un tramo recto y lo
        partiria en dos.
        """
        x, y = self.arrays()
        cerrado = self.cerrado()
        px, py = (x[:-1], y[:-1]) if cerrado else (x, y)
        dx = np.diff(px, append=px[0]) if cerrado else np.diff(px)
        dy = np.diff(py, append=py[0]) if cerrado else np.diff(py)
        n = len(dx)
        if n == 0:
            return np.zeros(0, dtype=int), cerrado
        ang = np.degrees(np.arctan2(dy, dx))
        prev = np.roll(ang, 1) if cerrado else np.concatenate(([ang[0]], ang[:-1]))
        giro = np.abs((ang - prev + 180.0) % 360.0 - 180.0)
        esq = np.flatnonzero(giro > ang_tol)
        if not len(esq):
            return np.zeros(n, dtype=int), cerrado    # circulo: una sola arista
        if not cerrado:
            esq = np.unique(np.concatenate(([0], esq)))
        dueno = np.cumsum(np.isin(np.arange(n), esq)) - 1
        if cerrado:
            # cumsum deja los segmentos anteriores a la primera esquina en -1:
            # son la cola de la ultima arista, que da la vuelta por el cierre.
            dueno[dueno < 0] = dueno.max()
        return dueno, cerrado

    def aristas(self, ang_tol=ANG_ARISTA):
        """[(xs, ys)] de cada arista, en orden. El indice es su identidad."""
        dueno, cerrado = self._dueno_de_segmento(ang_tol)
        if not len(dueno):
            return []
        x, y = self.arrays()
        px, py = (x[:-1], y[:-1]) if cerrado else (x, y)
        n = len(px)
        out = []
        for k in range(dueno.max() + 1):
            seg = np.flatnonzero(dueno == k)
            # La arista que da la vuelta por el cierre sale partida en dos
            # trozos (los ultimos segmentos y los primeros). Se rota para
            # dibujarla como la polilinea continua que es.
            corte = np.flatnonzero(np.diff(seg) > 1)
            if cerrado and len(corte):
                seg = np.roll(seg, -(int(corte[0]) + 1))
            idx = np.append(seg, (seg[-1] + 1) % n if cerrado else seg[-1] + 1)
            out.append((px[idx], py[idx]))
        return out

    def tipo_arista(self, k):
        return self.paredes.get(str(k), self.pared)

    def poner_arista(self, k, tipo):
        """Guarda solo lo que difiere de `pared`: el JSON no lista lo obvio."""
        if tipo == self.pared:
            self.paredes.pop(str(k), None)
        else:
            self.paredes[str(k)] = tipo

    def paredes_por_segmento(self):
        """Tipo de pared de cada segmento del poligono, en el orden en que el
        solver recorre los puntos. Es lo que se le pasa a el: agrupar aristas es
        cosa de la interfaz, el solver solo necesita saber, para cada tramo,
        si desliza o no."""
        dueno, _ = self._dueno_de_segmento()
        return [self.tipo_arista(int(k)) for k in dueno]

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

    # --------------------------------------------------------------- aristas
    def lado_de_arista(self, xs, ys):
        """(lado, a, b) si la arista esta pegada a un borde de la caja, o None.

        Es la distincion que decide que se le puede pedir a una arista. Pegada
        al borde no es pared inmersa: el IBM no la ve y lo que manda ahi es el
        parche del perimetro, asi que admite entrada y salida. Metida dentro del
        dominio si es pared inmersa y solo puede deslizar o no.
        """
        tol = 1e-4 * max(self.Lx, self.Ly)
        for lado, fijo, libre, largo in (
                ("left", xs, ys, 0.0), ("right", xs, ys, self.Lx),
                ("bottom", ys, xs, 0.0), ("top", ys, xs, self.Ly)):
            if np.all(np.abs(np.asarray(fijo) - largo) <= tol):
                a, b = float(np.min(libre)), float(np.max(libre))
                return lado, a, b
        return None

    def tipo_en(self, lado, s):
        """Tipo de frontera del perimetro en la coordenada `s` del lado. Manda
        el ultimo parche que la cubre, que es el orden en el que el solver los
        aplica."""
        tipo = "noslip"
        largo = self.Ly if lado in ("left", "right") else self.Lx
        for p in self.parches:
            if p.lado != lado:
                continue
            a = 0.0 if p.desde is None else p.desde
            b = largo if p.hasta is None else p.hasta
            if a - 1e-12 <= s <= b + 1e-12:
                tipo = p.tipo
        return tipo

    def poner_parche(self, lado, tipo, valor, a, b):
        """Escribe un parche y se lleva por delante los que quedan dentro del
        tramo. Sin esto, pinchar dos veces la misma boca deja la lista llena de
        parches tapados unos por otros y la tabla deja de decir lo que pasa."""
        tol = 1e-9 * max(self.Lx, self.Ly)
        largo = self.Ly if lado in ("left", "right") else self.Lx
        def dentro(p):
            u = 0.0 if p.desde is None else p.desde
            v = largo if p.hasta is None else p.hasta
            return p.lado == lado and u >= a - tol and v <= b + tol
        self.parches = [p for p in self.parches if not dentro(p)]
        self.parches.append(Parche(lado, tipo, valor, a, b))

    def arista_cercana(self, x, y, tol):
        """(i_contorno, k_arista, dist) de la arista mas cercana al punto."""
        mejor = None
        for i, c in enumerate(self.contornos):
            for k, (xs, ys) in enumerate(c.aristas()):
                d = _dist_a_polilinea(x, y, xs, ys)
                if d <= tol and (mejor is None or d < mejor[2]):
                    mejor = (i, k, d)
        return mejor

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

        # Aristas pegadas al borde: ahi no hay pared inmersa que deslice, manda
        # el parche del perimetro. Marcarlas slip y no tocar el parche es el
        # error que deja una pared que se cree deslizante y no lo es.
        for c in self.contornos:
            aristas = c.aristas()
            fuera = [k for k in map(int, c.paredes) if k >= len(aristas)]
            if fuera:
                av.append(f"El contorno '{c.nombre or '?'}' tiene condiciones "
                          f"en aristas {fuera} que ya no existen: la geometria "
                          f"ha cambiado desde que se pusieron.")
            for k, (xs, ys) in enumerate(aristas):
                if str(k) not in c.paredes:
                    continue
                borde = self.lado_de_arista(xs, ys)
                if borde is None:
                    continue
                lado, a, b = borde
                tipo = c.tipo_arista(k)
                if self.tipo_en(lado, 0.5 * (a + b)) != tipo:
                    av.append(
                        f"La arista {k} de '{c.nombre or '?'}' esta sobre el "
                        f"borde «{lado}» y ahi no hay pared inmersa: manda el "
                        f"parche del perimetro, que es "
                        f"«{self.tipo_en(lado, 0.5 * (a + b))}» y no "
                        f"«{tipo}». Pon el parche en [{a:.4g}, {b:.4g}].")

        # Parches que caen sobre pared. El solver los fuerza a WALL, asi que no
        # hacen nada; en la interfaz se ven como una tira de color y engañan.
        for p in self.parches:
            if p.tipo in ("noslip", "slip"):
                continue
            if not self.recortar_a_bocas(p):
                av.append(
                    f"El parche «{p.tipo}» de «{p.lado}» cae entero sobre "
                    f"pared: ahí no entra ni sale nada. Las bocas abiertas de "
                    f"ese lado son "
                    f"{self.tramos_abiertos(p.lado) or 'ninguna'}.")

        # El coste va con 1/dx^2 y con un DXF en milimetros leido como metros se
        # piden miles de millones de celdas sin que nada chille hasta la GPU.
        celdas = (self.Lx / max(self.dx_min, 1e-30)) * (self.Ly / max(self.dy, 1e-30))
        if celdas > 5e7:
            av.append(
                f"El dominio es de {self.Lx:.4g} x {self.Ly:.4g} y dx={self.dx_min:g}: "
                f"salen del orden de {celdas:.1e} celdas, que no caben en ninguna "
                f"GPU. ¿Están bien las unidades del DXF? Si venía en milímetros y "
                f"se ha leído como metros, la geometría es 1000 veces más grande "
                f"de lo que crees.")
        return av

    # ------------------------------------------------------------------
    def a_dict(self):
        d = asdict(self)
        d["contornos"] = [
            {**asdict(c), "x": list(map(float, c.x)), "y": list(map(float, c.y))}
            for c in self.contornos]
        d["parches"] = [asdict(p) for p in self.parches]
        return d

    def tramos_abiertos(self, lado, n=600):
        """[(a, b)] del lado que son BOCA, en coordenadas del lado.

        Con un contorno exterior que no es el rectangulo de la caja —una tobera,
        un difusor, cualquier cosa inclinada— la mayor parte del perimetro no es
        boca sino pared: entre el contorno y el borde de la caja queda solido, y
        ahi el parche no hace nada porque `_rebuild_bc_masks` lo fuerza a WALL.
        Sin esto la interfaz pinta la tira de color a lo largo del lado entero y
        no se ve que media tira es inerte.

        Se muestrea el lado por dentro (a un pelo del borde) y se mira si el
        punto cae en el contorno. Muestrear SOBRE el borde no vale: el tramo del
        contorno suele ser colineal con el y el test punto-en-poligono es
        ambiguo justo ahi.
        """
        largo = self.Ly if lado in ("left", "right") else self.Lx
        ext = self.exteriores()
        if not ext:
            return [(0.0, largo)]          # sin contorno, el lado entero es boca
        from matplotlib.path import Path
        d = 1e-3 * min(self.Lx, self.Ly)
        s = np.linspace(0.0, largo, int(n))
        if lado == "left":
            pts = np.column_stack((np.full_like(s, d), s))
        elif lado == "right":
            pts = np.column_stack((np.full_like(s, self.Lx - d), s))
        elif lado == "bottom":
            pts = np.column_stack((s, np.full_like(s, d)))
        else:
            pts = np.column_stack((s, np.full_like(s, self.Ly - d)))
        dentro = np.zeros(len(s), dtype=bool)
        for c in ext:
            x, y = c.arrays()
            dentro |= Path(np.column_stack((x, y))).contains_points(pts)
        tramos, i = [], 0
        while i < len(s):
            if not dentro[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(s) and dentro[j + 1]:
                j += 1
            tramos.append((float(s[i]), float(s[j])))
            i = j + 1
        return tramos

    def recortar_a_bocas(self, parche):
        """Trozos de un parche que caen en boca abierta. Vacio = parche inerte."""
        largo = self.Ly if parche.lado in ("left", "right") else self.Lx
        a = 0.0 if parche.desde is None else float(parche.desde)
        b = largo if parche.hasta is None else float(parche.hasta)
        if parche.tipo in ("noslip", "slip"):
            return [(a, b)]                # una pared si actua sobre la pared
        out = []
        for u, v in self.tramos_abiertos(parche.lado):
            lo, hi = max(a, u), min(b, v)
            if hi > lo:
                out.append((lo, hi))
        return out

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
            # Solo si hay excepciones: sin ellas el solver toma el camino de
            # siempre (una pared por cuerpo) y no hace ninguna busqueda de
            # segmento mas proximo.
            if c.paredes:
                cd["paredes_seg"] = c.paredes_por_segmento()
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


def _dist_a_polilinea(x, y, xs, ys):
    """Distancia de un punto a una polilinea, por segmentos."""
    ax, ay = np.asarray(xs[:-1], float), np.asarray(ys[:-1], float)
    bx, by = np.asarray(xs[1:], float), np.asarray(ys[1:], float)
    if not len(ax):
        return float(np.hypot(x - xs[0], y - ys[0]))
    ex, ey = bx - ax, by - ay
    L2 = np.maximum(ex * ex + ey * ey, 1e-30)
    t = np.clip(((x - ax) * ex + (y - ay) * ey) / L2, 0.0, 1.0)
    return float(np.min(np.hypot(x - (ax + t * ex), y - (ay + t * ey))))


def descripcion_arista(xs, ys):
    """Nombre legible de una arista: forma, inclinacion y largo."""
    largo = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
    cuerda = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
    ang = float(np.degrees(np.arctan2(ys[-1] - ys[0], xs[-1] - xs[0])) % 180.0)
    if largo > 1.02 * cuerda:
        forma = "curva"
    elif ang < 5.0 or ang > 175.0:
        forma = "horizontal"
    elif abs(ang - 90.0) < 5.0:
        forma = "vertical"
    else:
        forma = f"inclinada {ang:.0f}°"
    return f"{forma} · {largo:.4g} m"


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

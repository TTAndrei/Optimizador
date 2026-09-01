"""Importa geometria 2D de un DXF y la deja lista para el rasterizador del solver.

El solver es 2D y trabaja con poligonos cerrados en el plano, asi que DXF es el
formato natural: es 2D nativo, lo exporta cualquier CAD y `ezdxf` lo lee en
Python puro, sin binarios. STL obligaria a cortar por un plano y coser el
contorno; STEP y Parasolid exigen OpenCASCADE.

El trabajo real no es leer el fichero, son tres cosas que el CAD no garantiza:

  1. Las curvas hay que APLANAR a polilineas con un error de cuerda acotado.
     `tol_cordal` por defecto es dx/4: refinar la geometria mas que la malla no
     aporta nada, porque la rasterizacion es par/impar sobre centros de celda.
  2. Los DXF llegan como sopa de segmentos sueltos, no como lazos. Hay que
     COSERLOS por proximidad de extremos. Los que no cierran se reportan con sus
     coordenadas y NO se cierran a la fuerza: cerrar en silencio es la forma
     clasica de acabar con una geometria plausible y equivocada.
  3. Hay que decidir que lazo es el contorno EXTERIOR del dominio y cuales son
     CUERPOS dentro de el, que es un problema de anidamiento, no de lectura.

Sin dependencias de CuPy ni de la GUI: se puede probar solo.
"""
from __future__ import annotations

import math
import os

import numpy as np

# Codigos $INSUNITS de DXF -> metros. 0 = sin unidades (se asume metros).
_UNIDADES = {
    0: 1.0, 1: 0.0254, 2: 0.3048, 3: 1609.344, 4: 1e-3, 5: 1e-2, 6: 1.0,
    7: 1000.0, 8: 2.54e-8, 9: 2.54e-5, 10: 0.9144, 11: 1e-10, 12: 1e-9,
    13: 1e-6, 14: 1e-4, 15: 1e-1, 16: 10.0, 17: 1e6, 18: 1.496e11,
}

TIPOS = ("LINE", "ARC", "CIRCLE", "ELLIPSE", "LWPOLYLINE", "POLYLINE", "SPLINE")


class GeometriaAbierta(Exception):
    """Uno o mas lazos no cierran. Lleva las coordenadas de los extremos sueltos."""

    def __init__(self, extremos):
        self.extremos = extremos
        puntos = ", ".join(f"({x:.4f}, {y:.4f})" for x, y in extremos[:6])
        mas = f" y {len(extremos) - 6} mas" if len(extremos) > 6 else ""
        super().__init__(
            f"{len(extremos)} extremo(s) sin pareja: {puntos}{mas}. "
            f"Sube la tolerancia de cosido o cierra el contorno en el CAD.")


def cargar_dxf(path, capa=None, tol_cordal=0.01, tol_cosido=None,
               escala=None, estricto=True):
    """DXF -> lista de contornos cerrados.

    capa        : si se da, solo se leen entidades de esa capa (o lista de capas).
    tol_cordal  : error maximo de cuerda al aplanar arcos y splines, en unidades
                  del DXF. Ponerlo a dx_min/4.
    tol_cosido  : distancia bajo la cual dos extremos se consideran el mismo
                  punto. Por defecto 2*tol_cordal.
    escala      : factor a metros. None = deducirlo de $INSUNITS.
    estricto    : si True, un lazo abierto lanza GeometriaAbierta. Si False, se
                  devuelve igualmente con cerrado=False para que la GUI lo pinte
                  en rojo.

    Devuelve una lista de dicts con las claves x, y, rol, capa, pared, cerrado,
    area, profundidad. Coordenadas en metros, poligono cerrado (el ultimo punto
    repite el primero) y sentido antihorario.
    """
    import ezdxf
    from ezdxf import path as ezpath

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    doc = ezdxf.readfile(path)
    if escala is None:
        escala = _UNIDADES.get(int(doc.header.get("$INSUNITS", 0)), 1.0)

    capas = {capa} if isinstance(capa, str) else (set(capa) if capa else None)
    tol_cosido = tol_cosido if tol_cosido is not None else 2.0 * tol_cordal

    trozos = []   # (array Nx2, nombre de capa)
    for e in doc.modelspace():
        if e.dxftype() not in TIPOS:
            continue
        if capas is not None and e.dxf.layer not in capas:
            continue
        try:
            p = ezpath.make_path(e)
        except Exception:
            continue
        pts = np.array([(v.x, v.y) for v in p.flattening(distance=tol_cordal)])
        if len(pts) >= 2:
            trozos.append((pts, e.dxf.layer))

    if not trozos:
        raise ValueError(f"El DXF no contiene entidades utilizables "
                         f"({'/'.join(TIPOS)}) en las capas pedidas.")

    lazos = coser(trozos, tol_cosido)

    abiertos = [l for l in lazos if not l["cerrado"]]
    if abiertos and estricto:
        raise GeometriaAbierta([(l["x"][0], l["y"][0]) for l in abiertos]
                               + [(l["x"][-1], l["y"][-1]) for l in abiertos])

    for l in lazos:
        l["x"] = l["x"] * escala
        l["y"] = l["y"] * escala
        l["area"] = area_con_signo(l["x"], l["y"])

    clasificar(lazos)
    return lazos


# ----------------------------------------------------------------------
# Cosido
# ----------------------------------------------------------------------
def coser(trozos, tol):
    """Encadena trozos por proximidad de extremos hasta cerrar lazos.

    Greedy: se arranca de un trozo sin usar y se va enganchando el extremo libre
    mas cercano dentro de `tol`, invirtiendo el trozo si hace falta, hasta volver
    al punto de partida (lazo cerrado) o quedarse sin candidatos (lazo abierto,
    que se devuelve marcado y no se cierra a la fuerza).
    """
    from scipy.spatial import cKDTree

    # Un indice de extremos: 2 por trozo. La clave 2*k es el inicio, 2*k+1 el fin.
    extremos = np.array([p[i] for p, _ in trozos for i in (0, -1)])
    arbol = cKDTree(extremos)
    usados = [False] * len(trozos)
    lazos = []

    for k0 in range(len(trozos)):
        if usados[k0]:
            continue
        pts, capa = trozos[k0]
        usados[k0] = True
        cadena = [pts]
        capas = {capa}
        inicio = pts[0]
        fin = pts[-1]

        while np.hypot(*(fin - inicio)) > tol:
            cand = None
            for idx in arbol.query_ball_point(fin, tol):
                k, extremo = divmod(idx, 2)
                if usados[k]:
                    continue
                cand = (k, extremo)
                break
            if cand is None:
                break
            k, extremo = cand
            p, c = trozos[k]
            usados[k] = True
            capas.add(c)
            # Si engancho por su final, el trozo va del reves.
            p = p[::-1] if extremo == 1 else p
            cadena.append(p[1:])       # sin repetir el punto de union
            fin = p[-1]

        x = np.concatenate([c[:, 0] for c in cadena])
        y = np.concatenate([c[:, 1] for c in cadena])
        cerrado = np.hypot(x[0] - x[-1], y[0] - y[-1]) <= tol
        if cerrado and np.hypot(x[0] - x[-1], y[0] - y[-1]) > 0:
            x, y = np.append(x, x[0]), np.append(y, y[0])
        lazos.append({"x": x, "y": y, "cerrado": bool(cerrado),
                      "capa": sorted(capas)[0], "capas": sorted(capas)})
    return lazos


# ----------------------------------------------------------------------
# Anidamiento y orientacion
# ----------------------------------------------------------------------
def clasificar(lazos):
    """Decide rol y orienta cada lazo en sentido antihorario.

    Profundidad = cuantos otros lazos lo contienen. Con eso:

      - Si hay UN solo lazo de profundidad 0 y ademas contiene a otros, ese es
        el contorno exterior del dominio y los de dentro son cuerpos.
      - Si hay VARIOS lazos de profundidad 0, no hay contorno exterior: son
        todos cuerpos sumergidos en la caja (el caso del perfil de siempre).

    No es adivinable solo con geometria en todos los casos, asi que la GUI deja
    cambiarlo; esto es el valor por defecto razonable.
    """
    from matplotlib.path import Path

    cerrados = [l for l in lazos if l["cerrado"]]
    caminos = [Path(np.column_stack((l["x"], l["y"]))) for l in cerrados]
    for i, li in enumerate(cerrados):
        p = (li["x"][0], li["y"][0])
        li["profundidad"] = sum(
            1 for j, cj in enumerate(caminos) if j != i and cj.contains_point(p))

    raiz = [l for l in cerrados if l["profundidad"] == 0]
    hay_exterior = len(raiz) == 1 and len(cerrados) > 1

    for l in lazos:
        l.setdefault("profundidad", -1)
        l["rol"] = ("exterior" if (hay_exterior and l["profundidad"] == 0)
                    else "cuerpo")
        l["pared"] = "noslip"
        if l["cerrado"] and area_con_signo(l["x"], l["y"]) < 0:
            l["x"], l["y"] = l["x"][::-1].copy(), l["y"][::-1].copy()
            l["area"] = -l.get("area", 0.0)
    return lazos


def area_con_signo(x, y):
    """Positiva si el poligono va en sentido antihorario."""
    return 0.5 * float(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))


# ----------------------------------------------------------------------
def resumen(lazos):
    filas = [f"{len(lazos)} contorno(s):"]
    for i, l in enumerate(lazos):
        estado = "cerrado" if l["cerrado"] else "ABIERTO"
        filas.append(
            f"  [{i}] {l['rol']:<8} {estado:<8} capa={l['capa']:<12} "
            f"n={len(l['x']):5d}  area={l.get('area', float('nan')):+.5f}  "
            f"x[{l['x'].min():+.3f},{l['x'].max():+.3f}] "
            f"y[{l['y'].min():+.3f},{l['y'].max():+.3f}]")
    return "\n".join(filas)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit(f"uso: {sys.argv[0]} fichero.dxf [tol_cordal]")
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    print(resumen(cargar_dxf(sys.argv[1], tol_cordal=tol, estricto=False)))

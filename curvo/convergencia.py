r"""Parada por fuerzas: correr hasta que Cl y Cd dejan de moverse en el tercer decimal.

El criterio del solver (`Solver.correr(parada=...)`) mide el residuo del campo,
`max|u^{n+1}-u^n|/(u_inf dt)`. No es lo que se quiere aqui: el campo puede seguir
moviendose en la estela lejana mientras las fuerzas ya no cambian, y al reves, un
residuo pequeno no dice nada sobre lo que le queda a la fuerza por recorrer hasta
su meseta. Lo que se pide es sobre **el numero que se publica**: que Cl y Cd esten
fijados dentro de media unidad del tercer decimal.

Hay dos maneras de no estar fijado, y hacen falta las dos condiciones:

``banda``
    la media de la ventana esta mal determinada porque la senal oscila. Se mide
    por **medias de bloque**: se parte la ventana en 8 trozos y la banda es
    `1.96 s(medias) / sqrt(8)`. Ni `sigma/sqrt(N)`, que se cree muestras
    independientes que no tiene, ni `sigma/sqrt(N_ef)` con `N_ef` por
    autocorrelacion integrada, que es lo que hace el cartesiano y **se pasa de
    conservador justo en el caso limpio**: sobre un seno de amplitud 1e-2 con
    133 ciclos da 6.8e-4 cuando el error real de la media es ~1e-5, y con
    umbral 5e-4 no abre nunca. La media de un bloque de varios periodos si
    promedia la oscilacion, que es exactamente lo que se quiere saber. La
    ventana **se dobla sola** hasta que la banda baja del umbral o se agota la
    serie, sin tener que medir ningun periodo: vale igual para un ciclo limite
    y para una estela de banda ancha.

``cola``
    la media esta bien determinada pero **sigue derivando**. Aqui no basta con
    mirar la pendiente: estas series decaen como `A - B exp(-t/tau)` con
    `tau ~ 2.9` tiempos convectivos (medido sobre las cuatro validaciones), y una
    pendiente pequena multiplicada por una `tau` larga sigue siendo un cambio
    grande. Lo que se compara con el umbral es **lo que le queda por recorrer**:
    con las medias de tres bloques consecutivos `m0, m1, m2`, las diferencias
    `d0 = m1-m0` y `d1 = m2-m1` decaen geometricamente con razon `r = d1/d0`, y
    la suma de lo que falta es `d1 r/(1-r)`, con signo. Si `r >= 1` la serie no
    esta decayendo y no se para; si `r <= 0` el cambio de bloque a bloque cambia
    de signo, o esta por debajo de la banda y entonces no hay deriva que
    medir, la cola es `d1`. Sumada a la media da
    `extrapolado`, el valor de meseta: se guarda como diagnostico, pero lo que
    se publica es la media, que es lo que de verdad se ha medido.

Es la leccion de `_detect_series_convergence` del cartesiano (tres versiones,
todas falladas por medir la magnitud equivocada) aplicada a una senal distinta:
alli la que mandaba era la banda, porque habia desprendimiento; aqui a alfa = 5 y
Re = 1e5 con SA la serie es monotona y limpia y la que manda es la cola. El
umbral va en **unidades absolutas del coeficiente**, no relativo, porque "el
tercer decimal" lo es.

Uso tipico::

    parada = ParadaFuerzas(tol=5e-4)
    for k in range(pasos):
        s.paso()
        if k % 20 == 0:
            e = parada.anotar(k * dt, *s.coeficientes())
            if e["ok"]:
                break

**En estudios de malla, no usarlo**: si cada malla para en un tiempo fisico
distinto, el orden observado mide esa diferencia y no la discretizacion. Ahi se
corre presupuesto fijo e identico en las tres mallas.
"""

from __future__ import annotations

import numpy as np

__all__ = ["estado_serie", "ParadaFuerzas"]

# Media unidad del tercer decimal. Sobre Cl ~ 0.5 es el 0.1 %; sobre Cd ~ 0.019
# es el 2.6 %, muy por debajo del error de malla, asi que el Cd admite umbral
# propio (`tol_cd`) si se quiere apretar.
TOL = 5e-4


def _banda(s, bloques=8):
    """Incertidumbre de la media por medias de bloque, al 95 %."""
    m = np.array([b.mean() for b in np.array_split(s, bloques)])
    return 1.96 * float(m.std(ddof=1)) / np.sqrt(bloques)


def _media(t, s, t_a, t_b):
    """Media de `s` en `(t_a, t_b]`. `None` si el bloque tiene menos de 3 muestras."""
    w = (t > t_a) & (t <= t_b)
    return float(s[w].mean()) if w.sum() >= 3 else None


def _cola(t, s, ventana, ruido):
    """Lo que le queda a la media por recorrer, con signo, por tres bloques.

    `inf` si no hay tres bloques o si la serie no decae (`r >= 1`). Una
    diferencia de bloque a bloque por debajo de `ruido` no es deriva: no se
    puede extrapolar una tendencia que no se resuelve.
    """
    t_f = t[-1]
    m = [_media(t, s, t_f - (k + 1) * ventana, t_f - k * ventana)
         for k in (2, 1, 0)]
    if any(x is None for x in m):
        return float("inf")
    d0, d1 = m[1] - m[0], m[2] - m[1]
    if abs(d1) <= ruido or abs(d0) <= ruido:
        return d1
    r = d1 / d0
    if r <= 0.0:
        return d1
    if r >= 1.0:
        return float("inf")
    return d1 * r / (1.0 - r)


def estado_serie(t, s, tol=TOL, ventana=1.0):
    """`banda`, `cola`, `media` y `ok` de una serie, en unidades del coeficiente.

    La ventana arranca en `ventana` y se dobla hasta que la banda baja de `tol` o
    hasta agotar la serie. Ensanchar de mas no cuela un transitorio: un
    transitorio tiene cola, y la cola se evalua sobre la misma ventana.
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    fuera = {"banda": float("nan"), "cola": float("nan"),
             "media": float("nan"), "extrapolado": float("nan"),
             "ventana": float(ventana), "ok": False}
    if len(s) < 10 or not np.all(np.isfinite(s)) or not np.all(np.isfinite(t)):
        return fuera

    span = float(t[-1] - t[0])
    v = float(ventana)
    banda = media = float("nan")
    while True:
        w = t > (t[-1] - v)
        # 16 muestras es el minimo para que los 8 bloques tengan 2 cada uno.
        if w.sum() >= 16:
            sw = s[w]
            media = float(np.mean(sw))
            banda = _banda(sw)
            if banda < tol:
                break
        if v >= span:
            break
        v = min(2.0 * v, span)
    if not np.isfinite(banda):
        return fuera

    cola = _cola(t, s, v, banda)
    return {"banda": banda, "cola": cola, "media": media, "ventana": v,
            "extrapolado": media + cola if np.isfinite(cola) else float("nan"),
            "ok": bool(banda < tol and abs(cola) < tol)}


class ParadaFuerzas:
    """Acumula la historia de Cl y Cd y dice cuando dejan de cambiar.

    ==============  ===========================================================
    ``tol``         umbral absoluto sobre Cl (media unidad del tercer decimal)
    ``tol_cd``      idem para Cd; por defecto el mismo que `tol`
    ``ventana``     ventana inicial, en tiempos convectivos
    ``min_t``       antes de esto no se declara nada convergido
    ``sostenido``   comprobaciones seguidas que tienen que pasar
    ==============  ===========================================================

    `anotar` devuelve el estado de la ultima comprobacion; su clave `ok` es la
    que se mira para parar. Los dos coeficientes tienen que pasar a la vez.
    """

    def __init__(self, tol=TOL, tol_cd=None, ventana=1.0, min_t=5.0,
                 sostenido=3):
        self.tol = {"Cl": float(tol), "Cd": float(tol if tol_cd is None else tol_cd)}
        self.ventana = float(ventana)
        self.min_t = float(min_t)
        self.sostenido = int(sostenido)
        self.t = []
        self.serie = {"Cl": [], "Cd": []}
        self.seguidas = 0
        self.estado = None

    def anotar(self, t, cl, cd):
        """Mete una muestra y devuelve el estado. `estado["ok"]` = parar."""
        self.t.append(float(t))
        self.serie["Cl"].append(float(cl))
        self.serie["Cd"].append(float(cd))

        e = {n: estado_serie(self.t, self.serie[n], self.tol[n], self.ventana)
             for n in ("Cl", "Cd")}
        pasa = t >= self.min_t and all(e[n]["ok"] for n in e)
        self.seguidas = self.seguidas + 1 if pasa else 0
        self.estado = {"t": float(t), "n": len(self.t),
                       "seguidas": self.seguidas,
                       "ok": self.seguidas >= self.sostenido, **e}
        return self.estado

    def resumen(self):
        """Lo que hay que guardar en el `resultados.json` de la corrida.

        `Cl` y `Cd` son la **media de la ventana**, no el ultimo valor: si la
        serie oscila, el ultimo valor es un punto cualquiera de la oscilacion.
        """
        if self.estado is None:
            return {"convergida": False}
        e = self.estado
        return {"convergida": e["ok"], "t": e["t"], "muestras": e["n"],
                "ventana": e["Cl"]["ventana"],
                "tol": dict(self.tol),
                "Cl": e["Cl"]["media"], "Cd": e["Cd"]["media"],
                "extrapolado": {n: e[n]["extrapolado"] for n in ("Cl", "Cd")},
                "banda": {n: e[n]["banda"] for n in ("Cl", "Cd")},
                "cola": {n: e[n]["cola"] for n in ("Cl", "Cd")}}

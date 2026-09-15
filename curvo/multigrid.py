r"""Multigrid geometrico por correccion aditiva (ACM) para sistemas de 5 puntos.

El sistema es, por celda,

    aP*phi_P - aW*phi_W - aE*phi_E - aS*phi_S - aN*phi_N - aC*phi_corte = b

con todos los coeficientes de vecino positivos (upwind de 1.er orden + parte
ortogonal de la difusion: M-matriz, diagonal dominante). `aC` es el corte de
estela: la celda `(0,i)` tiene por vecino sur la `(0, nx-1-i)`, que es la misma
cara fisica vista desde el otro lado.

**Correccion aditiva, no Galerkin.** La malla gruesa se obtiene aglomerando
bloques de 2x2 celdas finas y la ecuacion gruesa es *la suma de las ecuaciones
finas del bloque*, con el ansatz de correccion constante a trozos. Las ventajas
aqui son concretas:

- no hay que rediscretizar la geometria en cada nivel (ni volver a emparejar el
  corte de estela por metricas, que es el punto que el plan marcaba como el mas
  delicado del engrosado),
- el estencil se mantiene de 5 puntos en todos los niveles,
- la conservacion es exacta por construccion: en el nivel mas grueso (una celda)
  la correccion anula la **suma** de los residuos, asi que el balance global se
  cierra a precision de maquina aunque el ciclo se pare pronto,
- funciona igual con operadores no simetricos (conveccion), que es justo el caso.

El engrosado en xi empareja **desde los dos extremos hacia dentro**. Es lo que
conserva el emparejamiento espejo del corte: si `i <-> nx-1-i` en el nivel fino,
entonces `k <-> ncx-1-k` en el grueso, exactamente. Emparejando solo desde i=0
con nx impar el espejo se desalinea una celda y el corte deja de cerrar.

Suavizador: relajacion por lineas cebra, alternando lineas eta y lineas xi. La
anisotropia de la malla C esta alineada con eta junto a la pared (a_eta/a_xi es
la relacion de aspecto, hasta ~500) pero se da la vuelta en el campo lejano, asi
que se barren las dos direcciones.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Sistema", "jerarquia", "ciclo_v", "resolver"]


def _thomas(sub, dia, sup, rhs):
    """Tridiagonales independientes a lo largo del eje 0. Arrays (n, m)."""
    n = dia.shape[0]
    cp = np.empty_like(dia)
    dp = np.empty_like(dia)
    cp[0] = sup[0] / dia[0]
    dp[0] = rhs[0] / dia[0]
    for k in range(1, n):
        m = dia[k] - sub[k] * cp[k - 1]
        cp[k] = sup[k] / m
        dp[k] = (rhs[k] - sub[k] * dp[k - 1]) / m
    x = np.empty_like(dia)
    x[-1] = dp[-1]
    for k in range(n - 2, -1, -1):
        x[k] = dp[k] - cp[k] * x[k + 1]
    return x


class Sistema:
    """Coeficientes de 5 puntos mas el acoplamiento del corte de estela.

    Formas: todo `(ny, nx)` salvo `aC`, que es `(nx,)` y solo actua en `j=0`.
    Los coeficientes de vecino valen cero donde no hay vecino.
    """

    def __init__(self, aP, aW, aE, aS, aN, b, aC=None):
        self.aP, self.aW, self.aE, self.aS, self.aN, self.b = aP, aW, aE, aS, aN, b
        self.aC = aC
        self.ny, self.nx = aP.shape

    def aplicar(self, phi):
        r = self.aP * phi
        r[:, 1:] -= self.aW[:, 1:] * phi[:, :-1]
        r[:, :-1] -= self.aE[:, :-1] * phi[:, 1:]
        r[1:, :] -= self.aS[1:, :] * phi[:-1, :]
        r[:-1, :] -= self.aN[:-1, :] * phi[1:, :]
        if self.aC is not None:
            r[0] -= self.aC * phi[0, ::-1]
        return r

    def residuo(self, phi):
        return self.b - self.aplicar(phi)

    # -- suavizado ---------------------------------------------------------
    def _retrasado_xi(self, phi):
        """Terminos que no entran en una linea eta: vecinos en xi y el corte."""
        t = np.zeros_like(phi)
        t[:, 1:] += self.aW[:, 1:] * phi[:, :-1]
        t[:, :-1] += self.aE[:, :-1] * phi[:, 1:]
        if self.aC is not None:
            t[0] += self.aC * phi[0, ::-1]
        return t

    def _retrasado_eta(self, phi):
        """Idem para una linea xi: vecinos en eta. El corte vive dentro de la
        linea j=0 pero no es tridiagonal, asi que tambien va retrasado."""
        t = np.zeros_like(phi)
        t[1:, :] += self.aS[1:, :] * phi[:-1, :]
        t[:-1, :] += self.aN[:-1, :] * phi[1:, :]
        if self.aC is not None:
            t[0] += self.aC * phi[0, ::-1]
        return t

    def _lineas_eta(self, phi, paridad):
        c = np.arange(paridad, self.nx, 2)
        if c.size == 0:
            return
        d = (self.b + self._retrasado_xi(phi))[:, c]
        phi[:, c] = _thomas(-self.aS[:, c], self.aP[:, c], -self.aN[:, c], d)

    def _lineas_xi(self, phi, paridad):
        f = np.arange(paridad, self.ny, 2)
        if f.size == 0:
            return
        d = (self.b + self._retrasado_eta(phi))[f, :].T
        phi[f, :] = _thomas(-self.aW[f, :].T, self.aP[f, :].T,
                            -self.aE[f, :].T, d).T

    def suavizar(self, phi, veces=1):
        for _ in range(veces):
            self._lineas_eta(phi, 0)
            self._lineas_eta(phi, 1)
            self._lineas_xi(phi, 0)
            self._lineas_xi(phi, 1)


# ---------------------------------------------------------------------------
# Aglomeracion
# ---------------------------------------------------------------------------
def _mapa(n, simetrico=False):
    """Indice de bloque de cada celda al emparejar de dos en dos.

    `simetrico` empareja desde los dos extremos hacia dentro, de modo que el
    espejo `i <-> n-1-i` se conserva como `k <-> nc-1-k`. El bloque suelto que
    queda con `n` impar cae en el centro, donde no hay corte.
    """
    nc = (n + 1) // 2
    i = np.arange(n)
    if simetrico:
        return np.where(2 * i < n, i // 2, nc - 1 - (n - 1 - i) // 2), nc
    return i // 2, nc


def _sumar(idx, valores, forma):
    return np.bincount(idx.ravel(), valores.ravel(),
                       minlength=forma[0] * forma[1]).reshape(forma)


def _engrosar(sis, kj, ncy, ki, ncx):
    forma = (ncy, ncx)
    idx = kj[:, None] * ncx + ki[None, :]

    mismo_e = ki[:-1] == ki[1:]                      # cara xi interna al bloque
    mismo_n = kj[:-1] == kj[1:]

    aP = _sumar(idx, sis.aP, forma)
    aP -= _sumar(idx[:, :-1], sis.aE[:, :-1] * mismo_e, forma)
    aP -= _sumar(idx[:, 1:], sis.aW[:, 1:] * mismo_e, forma)
    aP -= _sumar(idx[:-1, :], sis.aN[:-1, :] * mismo_n[:, None], forma)
    aP -= _sumar(idx[1:, :], sis.aS[1:, :] * mismo_n[:, None], forma)

    aE = _sumar(idx[:, :-1], sis.aE[:, :-1] * ~mismo_e, forma)
    aW = _sumar(idx[:, 1:], sis.aW[:, 1:] * ~mismo_e, forma)
    aN = _sumar(idx[:-1, :], sis.aN[:-1, :] * ~mismo_n[:, None], forma)
    aS = _sumar(idx[1:, :], sis.aS[1:, :] * ~mismo_n[:, None], forma)

    aC = None
    if sis.aC is not None:
        aC = np.zeros(ncx)
        propio = ki == (ncx - 1 - ki)                # bloque que es su propio espejo
        np.add.at(aC, ki[~propio], sis.aC[~propio])
        if propio.any():
            aP[0] -= np.bincount(ki[propio], sis.aC[propio], minlength=ncx)
    return Sistema(aP, aW, aE, aS, aN, np.zeros(forma), aC)


def jerarquia(sis, minimo=9):
    """Lista [(sistema, kj, ki)] de fino a grueso. Los mapas son los del nivel."""
    niveles = [(sis, None, None)]
    while sis.aP.size > minimo:
        kj, ncy = _mapa(sis.ny)
        ki, ncx = _mapa(sis.nx, simetrico=True)
        if ncy * ncx >= sis.ny * sis.nx:
            break
        sis = _engrosar(sis, kj, ncy, ki, ncx)
        niveles.append((sis, kj, ki))
    return niveles


# ---------------------------------------------------------------------------
# Ciclo
# ---------------------------------------------------------------------------
def ciclo_v(niveles, phi, nivel=0, pre=2, post=2, grueso=30):
    """Ciclo V con la correccion gruesa **escalada por el cociente de Rayleigh**.

    La aglomeracion suma las ecuaciones finas, asi que el coeficiente de una cara
    gruesa es la suma de las dos caras finas que la forman. Para la conveccion
    eso es exacto (el flujo de masa de la cara gruesa ES la suma de los finos),
    pero para la difusion no: en una malla uniforme el coeficiente difusivo de
    cara no depende de h, luego el operador grueso sale **x2** respecto al
    rediscretizado y la correccion se queda a la mitad. Medido: con escala 1 el
    factor de convergencia en difusion pura se degrada con la malla (0.70, 0.85,
    0.93, 0.96 para n = 32..256) -- el defecto conocido de la aglomeracion sin
    suavizar.

    Se corrige eligiendo `w` que minimiza el error en la norma de energia,
    `w = (r . e) / (e . A e)`, que cuesta un producto matriz-vector y sale solo:
    ~2 en difusion, ~1 en conveccion. Medido con V(2,2): difusion 0.50/0.52/0.52
    y conveccion 0.01/0.14/0.31 para n = 64/128/256. Fijar w = 2 arregla la
    difusion pero **diverge** en conveccion (factor 7.1), y minimizar la norma
    del residuo en vez de la energia es peor que no escalar (0.99).
    """
    sis = niveles[nivel][0]
    if nivel == len(niveles) - 1:
        sis.suavizar(phi, grueso)
        return phi
    sis.suavizar(phi, pre)
    sc, kj, ki = niveles[nivel + 1]
    idx = kj[:, None] * sc.nx + ki[None, :]
    r = sis.residuo(phi)
    sc.b = _sumar(idx, r, (sc.ny, sc.nx))
    e = np.zeros((sc.ny, sc.nx))
    ciclo_v(niveles, e, nivel + 1, pre, post, grueso)

    ef = e[np.ix_(kj, ki)]
    den = float((ef * sis.aplicar(ef)).sum())
    w = min(max(float((r * ef).sum()) / den, 0.5), 4.0) if abs(den) > 1e-300 else 1.0
    phi += w * ef
    sis.suavizar(phi, post)
    return phi


def resolver(sis, phi=None, tol=1e-12, ciclos=60, pre=2, post=2, niveles=None):
    """Resuelve A phi = b. Devuelve (phi, info) con el historial de residuos."""
    if phi is None:
        phi = np.zeros_like(sis.aP)
    niveles = niveles if niveles is not None else jerarquia(sis)
    escala = max(float(np.abs(sis.b).max()), 1e-300)
    hist = [float(np.abs(sis.residuo(phi)).max()) / escala]
    for _ in range(ciclos):
        ciclo_v(niveles, phi, pre=pre, post=post)
        hist.append(float(np.abs(sis.residuo(phi)).max()) / escala)
        if hist[-1] <= tol:
            break
    utiles = [b / a for a, b in zip(hist[:-1], hist[1:]) if a > 0.0]
    return phi, {"residuos": hist, "ciclos": len(hist) - 1,
                 "factor": float(np.median(utiles)) if utiles else 0.0}

r"""Conveccion-difusion conservativa e implicita sobre la malla curvilinea.

Esquema, por celda y paso de tiempo (Euler atras):

    J*(phi - phi_viejo)/dt + div(F_conv) - div(F_dif) = J*s

Todo son **flujos por cara**: lo que sale de una celda entra en la vecina por la
misma cara, asi que la cantidad transportada se conserva a nivel discreto. Es
exactamente lo que el semi-Lagrangiano con correccion de MacCormack no tiene
(backtrace + interpolacion bilineal) y el motivo del cambio de esquema.

**Correccion diferida.** La matriz que se resuelve lleva solo lo que la hace
diagonal dominante y de 5 puntos:

- conveccion: upwind de 1.er orden,
- difusion: parte ortogonal, `nu * a_cara * (phi_vecina - phi_P)`.

Lo demas va al termino fuente y se itera:

- conveccion: el salto entre el valor de cara upwind-sesgado de 2.º orden con
  limitador de Van Leer y el de 1.er orden,
- difusion: los terminos cruzados xi-eta de la metrica completa.

Asi el operador implicito es una M-matriz (multigrid contento) y la **solucion
convergida** es la de 2.º orden con metrica completa. Es lo que hacen los URANS
segregados serios (SIMPLE/PISO), y da dt grande sin perder conservacion.

Condiciones de frontera, mismo convenio que `operadores.py`: valor de Dirichlet
**en el centro de la cara**, o `None` = gradiente nulo (salida). En la malla C
la frontera sur se parte en dos: la pared del perfil y el **corte de estela**,
donde la celda `(0,i)` tiene por vecina la `(0, nx-1-i)` -- la misma cara fisica
vista desde el otro lado. El corte se trata como cara interior, no como
frontera.
"""

from __future__ import annotations

import numpy as np

from . import operadores as op
from .metrica import xp_de
from .multigrid import Sistema, jerarquia, resolver

__all__ = [
    "AIRE",
    "flujos_de_campo",
    "flujos_de_corriente",
    "corte_de_estela",
    "sistema",
    "fuente_diferida",
    "avanzar",
    "masa",
]

# Aire seco a 15 C y 101325 Pa (atmosfera estandar ISA a nivel del mar).
AIRE = {
    "rho": 1.225,        # kg/m3
    "mu": 1.7894e-5,     # Pa.s   (Sutherland a 288.15 K)
    "nu": 1.4607e-5,     # m2/s   = mu/rho
    "T": 288.15,         # K
    "p": 101325.0,       # Pa
    "a": 340.29,         # m/s, velocidad del sonido
}


# ---------------------------------------------------------------------------
# Flujos de cara
# ---------------------------------------------------------------------------
def flujos_de_campo(met, campo):
    """Flujos volumetricos por cara de un campo analitico `campo(x, y) -> (u, v)`.

    Se evalua en el **punto medio de la cara**, que es donde el vector de area
    la representa. Para un campo de divergencia nula esto da flujos discretos de
    divergencia nula hasta el orden del esquema, y exactamente cero si el campo
    es uniforme (telescopado de las areas de cara).
    """
    X, Y = met.X, met.Y
    xf = 0.5 * (X[:-1, :] + X[1:, :])
    yf = 0.5 * (Y[:-1, :] + Y[1:, :])
    u, v = campo(xf, yf)
    m_xi = u * met.Sx_xi + v * met.Sy_xi

    xf = 0.5 * (X[:, :-1] + X[:, 1:])
    yf = 0.5 * (Y[:, :-1] + Y[:, 1:])
    u, v = campo(xf, yf)
    m_eta = u * met.Sx_eta + v * met.Sy_eta
    return m_xi, m_eta


def flujos_de_corriente(psi):
    """Flujos de cara a partir de una funcion de corriente en los VERTICES.

    `m_xi[j,i] = psi[j+1,i] - psi[j,i]`, `m_eta[j,i] = -(psi[j,i+1] - psi[j,i])`:
    es el equivalente discreto de `u = rot(psi)`, y el balance de la celda
    telescopa a **cero exacto**, no a cero al orden del esquema. Es la unica
    forma de tener un campo no uniforme con divergencia discreta nula, que es lo
    que hace falta para separar el error del esquema del de los flujos.
    """
    return psi[1:, :] - psi[:-1, :], -(psi[:, 1:] - psi[:, :-1])


def _lleva(v, xp):
    """Lleva un valor de frontera al dispositivo, sin tocar escalares ni tuplas.

    `flujos_de_velocidad` pasa la frontera como la **pareja** `(u, v)`, asi que
    hay que entrar en la tupla en vez de intentar convertirla entera: `asarray`
    de una tupla de arrays de cupy no es un array, es un error.
    """
    if v is None or np.isscalar(v):
        return v
    if isinstance(v, tuple):
        return tuple(_lleva(c, xp) for c in v)
    return xp.asarray(v)


def _bc_en(bc, xp):
    """Normaliza el diccionario de fronteras y lleva los valores al dispositivo."""
    d = {k: (None if bc is None else bc.get(k)) for k in
         ("oeste", "este", "sur", "norte")}
    return {k: _lleva(v, xp) for k, v in d.items()}


def corte_de_estela(info, nx, xp=np):
    """Mascara (nx,) de las celdas de `j=0` que son corte y no pared."""
    i0, i1 = info["perfil"]
    m = np.ones(nx, dtype=bool)
    m[i0:i1 - 1] = False
    return xp.asarray(m)


# ---------------------------------------------------------------------------
# Matriz implicita
# ---------------------------------------------------------------------------
def _difusiones(met, nu, bc, corte):
    """Coeficientes difusivos por cara, ya con el tratamiento de frontera.

    Las caras de frontera con Dirichlet llevan factor 2 porque la distancia del
    centro de celda a la cara es media celda; es el mismo factor que mete
    `dif_xi_caras`. Las de gradiente nulo no difunden.
    """
    xp = met.xp
    nu_xi, nu_eta = op._nu_en_caras(met, nu)
    d_xi = nu_xi * met.a_xi + xp.zeros_like(met.a_xi)
    d_eta = nu_eta * met.a_eta + xp.zeros_like(met.a_eta)
    d_xi[:, 0] = 2.0 * d_xi[:, 0] if bc["oeste"] is not None else 0.0
    d_xi[:, -1] = 2.0 * d_xi[:, -1] if bc["este"] is not None else 0.0
    d_n = 2.0 * d_eta[-1] if bc["norte"] is not None else 0.0 * d_eta[-1]
    d_s = 2.0 * d_eta[0] if bc["sur"] is not None else 0.0 * d_eta[0]
    if corte is not None:                       # el corte es cara interior
        d_s = xp.where(corte, d_eta[0], d_s)
    d_eta[0], d_eta[-1] = d_s, d_n
    return d_xi, d_eta


def sistema(met, m_xi, m_eta, nu=0.0, dt=None, bc=None, corte=None, bdf2=False,
            sumidero=None):
    """Matriz de 5 puntos del operador implicito. Devuelve (Sistema, b_frontera).

    `sumidero` es un coeficiente positivo `c` que anade `c*J` a la diagonal, es
    decir, un termino `-c*phi` en la ecuacion tratado de forma **implicita por
    puntos**: se divide en vez de restar, asi que no puede volver negativo un
    campo positivo por grande que sea `dt`. Es lo que la destruccion de
    Spalart-Allmaras necesita para que `nu_tilde` no cambie de signo.

    `b_frontera` es la parte del termino independiente que sale de los valores
    de Dirichlet; el resto (phi viejo, fuentes, correccion diferida) lo pone
    `avanzar`. `bdf2` cambia el coeficiente temporal de `1/dt` a `1.5/dt`; el
    termino independiente que le corresponde lo pone `avanzar` al recibir el
    nivel anterior. Si se reutiliza el sistema entre pasos hay que construirlo
    con el mismo `bdf2` con el que se va a llamar a `avanzar`.

    `bc` puede ser una **lista** de fronteras, y entonces `b_frontera` sale
    apilado en un array `(k, ny, nx)`. Los coeficientes solo dependen de QUE
    caras son Dirichlet, no de su valor, asi que varios campos con la misma
    estructura de frontera -- `u` y `v` del momento -- comparten una matriz y una
    jerarquia en vez de construir dos identicas. Apilado y no en lista porque es
    la forma que el multigrid necesita para resolverlos en un solo lanzamiento.
    """
    xp = met.xp
    varios = isinstance(bc, list)
    bcs = [_bc_en(c, xp) for c in bc] if varios else [_bc_en(bc, xp)]
    # Si la estructura no coincide, la matriz no sirve para ninguno de los dos y
    # el termino de frontera saldria mal en silencio.
    assert all((c[k] is None) == (bcs[0][k] is None) for c in bcs for k in c)
    ny, nx = met.J.shape
    tipo = met.J.dtype
    d_xi, d_eta = _difusiones(met, nu, bcs[0], corte)

    aP = xp.zeros((ny, nx), dtype=tipo)
    aW = xp.zeros((ny, nx), dtype=tipo)
    aE = xp.zeros((ny, nx), dtype=tipo)
    aS = xp.zeros((ny, nx), dtype=tipo)
    aN = xp.zeros((ny, nx), dtype=tipo)
    bs = xp.zeros((len(bcs), ny, nx), dtype=tipo)

    # --- caras interiores -------------------------------------------------
    m, d = m_xi[:, 1:-1], d_xi[:, 1:-1]
    aE[:, :-1] = xp.maximum(-m, 0.0) + d
    aW[:, 1:] = xp.maximum(m, 0.0) + d
    aP[:, :-1] += xp.maximum(m, 0.0) + d
    aP[:, 1:] += xp.maximum(-m, 0.0) + d

    m, d = m_eta[1:-1, :], d_eta[1:-1, :]
    aN[:-1, :] = xp.maximum(-m, 0.0) + d
    aS[1:, :] = xp.maximum(m, 0.0) + d
    aP[:-1, :] += xp.maximum(m, 0.0) + d
    aP[1:, :] += xp.maximum(-m, 0.0) + d

    # --- fronteras --------------------------------------------------------
    # Signo: el flujo de la cara oeste/sur es positivo hacia DENTRO de la celda;
    # el de la este/norte, hacia fuera. Con gradiente nulo el valor de cara es el
    # de la celda y no difunde: queda el flujo convectivo neto en la diagonal.
    m, d = m_xi[:, 0], d_xi[:, 0]
    if bcs[0]["oeste"] is None:
        aP[:, 0] += -m
    else:
        aP[:, 0] += xp.maximum(-m, 0.0) + d
        w = xp.maximum(m, 0.0) + d
        for b, c in zip(bs, bcs):
            b[:, 0] += w * c["oeste"]

    m, d = m_xi[:, -1], d_xi[:, -1]
    if bcs[0]["este"] is None:
        aP[:, -1] += m
    else:
        aP[:, -1] += xp.maximum(m, 0.0) + d
        w = xp.maximum(-m, 0.0) + d
        for b, c in zip(bs, bcs):
            b[:, -1] += w * c["este"]

    m, d = m_eta[-1], d_eta[-1]
    if bcs[0]["norte"] is None:
        aP[-1] += m
    else:
        aP[-1] += xp.maximum(m, 0.0) + d
        w = xp.maximum(-m, 0.0) + d
        for b, c in zip(bs, bcs):
            b[-1] += w * c["norte"]

    # Sur: pared (o salida) donde no hay corte, cara interior donde si.
    m, d = m_eta[0], d_eta[0]
    if bcs[0]["sur"] is None:
        ap_s = -m
        b_s = [xp.zeros(nx, dtype=tipo) for _ in bcs]
    else:
        ap_s = xp.maximum(-m, 0.0) + d
        w = xp.maximum(m, 0.0) + d
        b_s = [w * xp.asarray(c["sur"]) * xp.ones(nx, dtype=tipo) for c in bcs]
    aC = None
    if corte is not None:
        aC = xp.where(corte, xp.maximum(m, 0.0) + d, 0.0)
        ap_s = xp.where(corte, xp.maximum(-m, 0.0) + d, ap_s)
        b_s = [xp.where(corte, 0.0, x) for x in b_s]
    aP[0] += ap_s
    for b, x in zip(bs, b_s):
        b[0] += x

    if sumidero is not None:
        aP += sumidero * met.J
    if dt is not None:
        aP += (1.5 if bdf2 else 1.0) * met.J / dt
    return Sistema(aP, aW, aE, aS, aN, xp.zeros((ny, nx), dtype=tipo), aC,
                   activo=corte), (bs if varios else bs[0])


# ---------------------------------------------------------------------------
# Correccion diferida
# ---------------------------------------------------------------------------
def _limitada(phi, m, a_izq=None, a_der=None):
    """Incremento de flujo (2.º orden TVD - upwind) en las caras internas del eje 1.

    `phi` (..., n, K) por celdas, `m` (n, K-1) flujo en las K-1 caras internas.
    Delante de la malla puede ir un eje de campos; `m` no lo lleva, porque el
    flujo de masa es el mismo para todos, y se difunde.
    Limitador de Van Leer escrito como pendiente, `2ab/(a+b)` si `ab > 0` y 0 si
    no: asi no hay que dividir por el salto aguas abajo y no hay denominador que
    proteger.

    `a_izq` y `a_der` son la pendiente `phi_U - phi_UU` a usar en la primera y
    ultima cara interna, donde la celda dos veces aguas arriba cae fuera del
    dominio. Sin ellas se cae a 1.er orden en esa capa, y eso **le quita orden a
    todo el esquema**: medido sobre la solucion manufacturada convectiva, 1.60
    en vez de 1.95. Con el valor de frontera (que esta a media celda, de ahi el
    factor 2 que pone quien llama) o con la celda espejo del corte, se recupera.
    """
    xp = xp_de(phi)
    P, E = phi[..., :-1], phi[..., 1:]
    salto = E - P
    a_pos = xp.zeros_like(salto)
    a_pos[..., 1:] = P[..., 1:] - P[..., :-1]      # phi_U - phi_UU con m > 0
    a_neg = xp.zeros_like(salto)
    a_neg[..., :-1] = E[..., :-1] - E[..., 1:]     # phi_U - phi_UU con m < 0

    hay_uu = xp.ones(salto.shape, dtype=bool)
    pos = m > 0.0
    if a_izq is None:
        hay_uu[..., 0] &= ~pos[..., 0]
    else:
        a_pos[..., 0] = a_izq
    if a_der is None:
        hay_uu[..., -1] &= pos[..., -1]
    else:
        a_neg[..., -1] = a_der

    a = xp.where(pos, a_pos, a_neg)
    sa = xp.where(pos, salto, -salto)
    suma = xp.where(a * sa > 0.0, a + sa, 1.0)
    lim = xp.where(a * sa > 0.0, 2.0 * a * sa / suma, 0.0)
    return m * 0.5 * xp.where(hay_uu, lim, 0.0)


def _pendiente_borde(phi_borde, valor, n):
    """Pendiente equivalente entre el centro de celda y el valor de cara.

    La cara esta a media celda, asi que la pendiente comparable con la de dos
    centros consecutivos es el doble.
    """
    if valor is None:
        return None
    xp = xp_de(phi_borde)
    return 2.0 * (phi_borde - xp.asarray(valor) * xp.ones(n, dtype=phi_borde.dtype))


def fuente_diferida(met, phi, m_xi, m_eta, nu=0.0, bc=None, corte=None):
    """Termino fuente de la correccion diferida, ya integrado por celda.

        -div(F_conv_2 - F_conv_upwind) + div(F_dif_cruzada)
    """
    xp = met.xp
    ny, nx = met.J.shape
    tipo = phi.dtype
    if isinstance(bc, list):
        # Un valor de Dirichlet por campo, apilado en el eje de campos: los
        # escalares salen (k,1) y los perfiles (k,n), y los dos se difunden
        # contra la cara. Es lo unico de la correccion diferida que depende del
        # campo; el resto son los mismos flujos y la misma metrica.
        #
        # Sin forzar el tipo: un valor suelto llega como float de Python, y al
        # multiplicarlo por la cara promociona a doble precision. Redondearlo
        # antes a la precision del campo cambia el termino de la capa de pared.
        bcs = [_bc_en(c, xp) for c in bc]
        bc = {k: (None if bcs[0][k] is None else xp.stack(xp.broadcast_arrays(
                  *[xp.reshape(xp.asarray(c[k]), -1) for c in bcs])))
              for k in bcs[0]}
    else:
        bc = _bc_en(bc, xp)
    cabeza = phi.shape[:-2]

    a_s = _pendiente_borde(phi[..., 0, :], bc["sur"], nx)
    if corte is not None:                       # el corte tiene celda espejo
        espejo = phi[..., 0, :] - phi[..., 0, ::-1]
        a_s = espejo if a_s is None else xp.where(corte, espejo, a_s)

    c_xi = xp.zeros(cabeza + (ny, nx + 1), dtype=tipo)
    c_xi[..., 1:-1] = _limitada(phi, m_xi[:, 1:-1],
                                _pendiente_borde(phi[..., :, 0], bc["oeste"], ny),
                                _pendiente_borde(phi[..., :, -1], bc["este"], ny))
    c_eta = xp.zeros(cabeza + (ny + 1, nx), dtype=tipo)
    tr = lambda a: a.swapaxes(-1, -2)                               # noqa: E731
    c_eta[..., 1:-1, :] = tr(_limitada(
        tr(phi), m_eta[1:-1, :].T, a_s,
        _pendiente_borde(phi[..., -1, :], bc["norte"], nx)))

    # Caras de frontera por las que SALE flujo: el upwind es la celda interior,
    # asi que el valor de cara se extrapola linealmente con la celda siguiente.
    # Sin esto el error es de 1.er orden en una franja de una celda de ancho y el
    # orden global se queda en 1.50 (medido); con ello sube a 1.95.
    m = m_xi[:, 0]
    c_xi[..., 0] = xp.where(m < 0.0,
                            m * 0.5 * (phi[..., :, 0] - phi[..., :, 1]), 0.0)
    m = m_xi[:, -1]
    c_xi[..., -1] = xp.where(m > 0.0,
                             m * 0.5 * (phi[..., :, -1] - phi[..., :, -2]), 0.0)
    m = m_eta[-1]
    c_eta[..., -1, :] = xp.where(
        m > 0.0, m * 0.5 * (phi[..., -1, :] - phi[..., -2, :]), 0.0)
    m = m_eta[0]
    c_eta[..., 0, :] = xp.where(
        m < 0.0, m * 0.5 * (phi[..., 0, :] - phi[..., 1, :]), 0.0)

    if corte is not None:
        # El corte es cara interior: la celda dos veces aguas arriba es la de
        # encima de la espejo. Las dos mitades del corte dan correcciones
        # iguales y opuestas, asi que el flujo sigue cerrando.
        entra = m > 0.0
        b0, b1 = phi[..., 0, :], phi[..., 1, :]
        e0, e1 = b0[..., ::-1], b1[..., ::-1]
        a = xp.where(entra, e0 - e1, b0 - b1)
        sa = xp.where(entra, b0 - e0, e0 - b0)
        suma = xp.where(a * sa > 0.0, a + sa, 1.0)
        lim = xp.where(a * sa > 0.0, 2.0 * a * sa / suma, 0.0)
        c_eta[..., 0, :] = xp.where(corte, m * 0.5 * lim, c_eta[..., 0, :])

    d_xi = op.dif_xi_caras(phi, bc["oeste"], bc["este"])
    d_eta = op.dif_eta_caras(phi, bc["sur"], bc["norte"])
    if corte is not None:
        d_eta[..., 0, :] = xp.where(corte, phi[..., 0, :] - phi[..., 0, ::-1],
                                    d_eta[..., 0, :])

    nu_xi, nu_eta = op._nu_en_caras(met, nu)
    x_xi = nu_xi * met.b_xi * op.a_caras_xi(d_eta)
    x_eta = nu_eta * met.b_eta * op.a_caras_eta(d_xi)

    return -op.divergencia(c_xi, c_eta) + op.divergencia(x_xi, x_eta)


# ---------------------------------------------------------------------------
# Paso de tiempo
# ---------------------------------------------------------------------------
def masa(met, phi):
    """Integral de phi sobre el dominio. Lo que la conservacion tiene que dejar fijo."""
    return float((met.J * phi).sum())


def avanzar(met, phi, m_xi, m_eta, dt=None, nu=0.0, bc=None, corte=None,
            fuente=None, correcciones=2, tol=None, ciclos=60, sis=None,
            phi_ant=None, sumidero=None):
    """Un paso implicito, o el estacionario si `dt is None`.

    Con `phi_ant` (el nivel n-1) el paso es **BDF2**, de 2.º orden en el tiempo;
    sin el, Euler atras. Merece la pena tenerlo: con CFL fijo el error de Euler
    atras es O(dt) = O(h) y **se come el 2.º orden del espacio** -- medido sobre
    un pulso advectado en diagonal, orden global 0.85 con Euler atras frente a
    1.95 con BDF2.

    **BDF2 no es monotono**, y eso no se arregla iterando: ningun metodo lineal
    multipaso de orden mayor que uno lo es (barrera de Bolley-Crouzeix), y el
    coeficiente negativo de `phi^{n-1}` es justo lo que lo rompe. Medido sobre
    la malla C con un arranque en escalon (pared a 1, campo a 0) y CFL ~1 en la
    estela: el pico llega a 1.077 y decae a 1.004 en 60 pasos; con dt cinco
    veces menor se queda en 1.013. Euler atras, en el mismo caso, da 1.000000.
    Para magnitudes que tienen que ser positivas -- la `nu_tilde` de
    Spalart-Allmaras -- conviene Euler atras, o acotar dt.

    `correcciones` es el numero de iteraciones externas de la correccion
    diferida: con 0 sale el upwind de 1.er orden puro. Tambien decide la
    acotacion: el sobreimpulso de un escalon advectado baja de -4.9e-4 con una
    iteracion a -1.5e-7 con cuatro.

    Con `bc` como **lista**, `phi`, `fuente` y `phi_ant` son `(k, ny, nx)` y los
    campos se avanzan a la vez: comparten matriz, jerarquia y **lanzamiento de
    kernel**, y solo la correccion diferida se evalua por campo, que es la unica
    parte que depende de los valores de frontera. No se acoplan entre si -- la
    aritmetica de cada uno es la misma que en serie, solo intercalada -- pero
    paran juntos, cuando todos cumplen su propio criterio de residuo.
    """
    bdf2 = phi_ant is not None
    A, b_bc = (sis if sis is not None else
               sistema(met, m_xi, m_eta, nu, dt, bc, corte, bdf2, sumidero))
    if not hasattr(A, "_niveles"):          # se reutiliza si se reutiliza `sis`
        A._niveles = jerarquia(A)
    niveles = A._niveles

    b0 = b_bc.copy()
    if dt is not None:
        b0 += met.J * (2.0 * phi - 0.5 * phi_ant if bdf2 else phi) / dt
    if fuente is not None:
        b0 += met.J * fuente

    x = phi.copy()
    info = {}
    for _ in range(correcciones + 1):
        A.b = b0 + fuente_diferida(met, x, m_xi, m_eta, nu, bc, corte)
        x, info = resolver(A, x, tol=tol, ciclos=ciclos, niveles=niveles)
    return x, info

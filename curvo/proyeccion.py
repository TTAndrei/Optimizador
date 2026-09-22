r"""Proyeccion de presion sobre la malla curvilinea.

Paso fraccionado. Dado un campo de flujos de cara provisional `m*` que no cierra
el balance, se busca la presion que lo hace solenoidal:

    div( m* - dt * grad(p).S ) = 0   =>   div( grad(p).S ) = div(m*) / dt

y se corrige `m = m* - dt * grad(p).S`.

**El operador del multigrid ES `D.G`.** No se rediscretiza el laplaciano: la
matriz sale de `conveccion.sistema` con flujo de masa nulo y `nu = 1`, es decir,
de los mismos coeficientes de cara `a_xi`, `a_eta` con los que se evalua el
gradiente de presion en la correccion. Por construccion, no por coincidencia
numerica. En el solver cartesiano no era asi -- la ruta `face_flux` usa D y G
compactos pero el suavizador iba con el estencil ancho de `d2` -- y de ahi el
46.2 % de divergencia residual en modo par-impar que documenta
`docs/OPTIMIZACION_SOLVER.md` §2.1. Con el gradiente compacto el tablero de
ajedrez no esta en el nucleo del operador: `p_{i+1} - p_i` lo ve, `p_{i+1} -
p_{i-1}` no.

Terminos cruzados. La parte ortogonal (`a`) va en la matriz de 5 puntos, que es
una M-matriz; los cruzados de la metrica completa (`b`) van al termino
independiente y se iteran, igual que en `conveccion`. La solucion convergida es
la de metrica completa.

Caras de frontera. Dirichlet -> el valor esta en el centro de cara y la distancia
al centro de celda es media celda (el factor 2 lo pone `dif_*_caras`). Neumann
-> **el flujo de la cara se anula entero, tambien el termino cruzado**; anular
solo la derivada normal deja pasar flujo por el cruzado y la proyeccion deja de
conservar masa. El corte de estela es cara interior, no frontera.

Hace falta **al menos una frontera Dirichlet**. Con Neumann puro el sistema es
singular y anclarlo en una celda mete un pico local en la presion; la malla C
tiene plano de salida, asi que el caso no se da.
"""

from __future__ import annotations

import numpy as np

from . import operadores as op
from .conveccion import _bc_en
from .metrica import xp_de
from .conveccion import sistema as _sistema_implicito
from .multigrid import jerarquia, resolver_pcg

__all__ = [
    "sistema_presion",
    "flujos_de_presion",
    "proyectar",
    "corregir_velocidad",
    "flujos_de_velocidad",
    "velocidad_de_flujos",
    "fraccion_par_impar",
]


def _bc(bc, xp=np):
    return _bc_en(bc, xp)


# ---------------------------------------------------------------------------
# Matriz
# ---------------------------------------------------------------------------
def sistema_presion(met, bc=None, corte=None):
    """Poisson de 5 puntos. Devuelve `(Sistema, b_frontera)`.

    Es literalmente el operador implicito de `conveccion` con flujo de masa cero
    y `nu = 1`: misma funcion, mismos coeficientes de cara, mismo tratamiento de
    frontera y del corte. Esa es la garantia de que el operador del multigrid
    coincide con la divergencia del gradiente compacto.
    """
    if all(_bc(bc, met.xp)[k] is None for k in ("oeste", "este", "sur", "norte")):
        raise ValueError("Poisson con Neumann puro: hace falta una cara Dirichlet")
    cero_xi = met.xp.zeros_like(met.a_xi)
    cero_eta = met.xp.zeros_like(met.a_eta)
    return _sistema_implicito(met, cero_xi, cero_eta, nu=1.0, bc=bc, corte=corte)


# ---------------------------------------------------------------------------
# Gradiente de presion en caras
# ---------------------------------------------------------------------------
def _partes(met, p, bc, corte):
    """Flujo de cara de `grad(p)`, separado en parte ortogonal y cruzada."""
    xp = met.xp
    d_xi = op.dif_xi_caras(p, bc["oeste"], bc["este"])
    d_eta = op.dif_eta_caras(p, bc["sur"], bc["norte"])
    if corte is not None:
        d_eta[0] = xp.where(corte, p[0] - p[0, ::-1], d_eta[0])

    o_xi, o_eta = met.a_xi * d_xi, met.a_eta * d_eta
    x_xi = met.b_xi * op.a_caras_xi(d_eta)
    x_eta = met.b_eta * op.a_caras_eta(d_xi)

    if bc["oeste"] is None:
        o_xi[:, 0] = x_xi[:, 0] = 0.0
    if bc["este"] is None:
        o_xi[:, -1] = x_xi[:, -1] = 0.0
    if bc["norte"] is None:
        o_eta[-1] = x_eta[-1] = 0.0
    if bc["sur"] is None:
        dentro = corte if corte is not None else xp.zeros(met.nx, bool)
        o_eta[0] = xp.where(dentro, o_eta[0], 0.0)
        x_eta[0] = xp.where(dentro, x_eta[0], 0.0)
    if corte is not None:
        # El corte es UNA cara: las dos ramas tienen que ver el mismo flujo con
        # signo opuesto. La parte ortogonal ya sale antisimetrica sola, pero el
        # termino cruzado no, y sin esto lo que sale por la rama baja no es lo
        # que entra por la alta (medido: 0.4 % del flujo de cara).
        o_eta[0] = xp.where(corte, 0.5 * (o_eta[0] - o_eta[0][::-1]), o_eta[0])
        x_eta[0] = xp.where(corte, 0.5 * (x_eta[0] - x_eta[0][::-1]), x_eta[0])
    return o_xi, o_eta, x_xi, x_eta


def flujos_de_presion(met, p, bc=None, corte=None):
    """`grad(p) . S` por cara, con metrica completa. Formas (ny,nx+1), (ny+1,nx)."""
    o_xi, o_eta, x_xi, x_eta = _partes(met, p, _bc(bc, met.xp), corte)
    return o_xi + x_xi, o_eta + x_eta


# ---------------------------------------------------------------------------
# Proyeccion
# ---------------------------------------------------------------------------
def proyectar(met, m_xi, m_eta, dt=1.0, bc=None, corte=None, correcciones=2,
              tol=None, ciclos=80, p=None, sis=None, fijos=None):
    """Hace solenoidales los flujos de cara. Devuelve `(m_xi, m_eta, p, info)`.

    `correcciones` son las iteraciones externas de los terminos cruzados; con 0
    la presion es la del laplaciano ortogonal y la divergencia residual es la de
    la oblicuidad de la malla. Convergen geometricamente, a un ritmo que fija esa
    oblicuidad: sobre el cuadrado distorsionado con amplitud 0.10 la divergencia
    relativa baja 1.0e-1, 6.4e-4, 7.6e-6, 1.0e-7 y 3.7e-13 con 0, 4, 8, 12 y 24.
    Sobre los perfiles reales la oblicuidad p99 es 0.0065 y bastan 4.

    El sistema lineal se resuelve con **PCG precondicionado por el ciclo V**, no
    con el ciclo solo: el ciclo topa en factor 0.50 (el techo de la aglomeracion
    sin suavizar) y F4 pide < 0.3. Con PCG el factor baja a 0.03-0.06 y es
    independiente de la malla.

    `fijos` es una lista de `correcciones + 1` conteos de iteraciones del PCG,
    una por correccion cruzada, y con ella no se mira el residuo (ver
    `multigrid.resolver_pcg`). En regimen asentado la segunda correccion arranca
    ya convergida y su conteo es 0, o sea no se resuelve nada.
    """
    bcn = _bc(bc, met.xp)
    A, b_bc = sis if sis is not None else sistema_presion(met, bc, corte)
    if not hasattr(A, "_niveles"):          # se reutiliza si se reutiliza `sis`
        A._niveles = jerarquia(A)
    niveles = A._niveles

    b0 = b_bc - op.divergencia(m_xi, m_eta) / dt
    p = met.xp.zeros_like(met.J) if p is None else p.copy()
    info, gastados = {}, []
    for q in range(correcciones + 1):
        _, _, x_xi, x_eta = _partes(met, p, bcn, corte)
        A.b = b0 + op.divergencia(x_xi, x_eta)
        p, info = resolver_pcg(A, p, tol=tol, ciclos=ciclos, niveles=niveles,
                               fijos=None if fijos is None else fijos[q])
        gastados.append(info["vciclos"])
    info["vciclos"] = gastados

    F_xi, F_eta = flujos_de_presion(met, p, bc, corte)
    return m_xi - dt * F_xi, m_eta - dt * F_eta, p, info


def corregir_velocidad(met, u, v, p, dt=1.0, bc=None, corte=None):
    """`u <- u - dt grad(p)` por celda, con el gradiente Green-Gauss.

    La velocidad de celda solo se usa en el termino convectivo y en las fuerzas;
    lo que transporta son los flujos de cara corregidos, que es lo que mantiene
    el desacoplo par-impar fuera.
    """
    gx, gy = op.gradiente(met, p, corte=corte, **_bc(bc, met.xp))
    return u - dt * gx, v - dt * gy


def flujos_de_velocidad(met, u, v, bc=None, corte=None):
    """Velocidad de celda -> flujos de cara. `bc[lado] = (u, v)` o `None`.

    Interpolacion lineal a la cara y producto por el vector de area. El flujo
    resultante **no** es solenoidal; lo hace serlo `proyectar`, y el que la
    correccion de presion vaya por el gradiente compacto de cara mientras la de
    celda va por Green-Gauss es lo que desacopla el tablero de ajedrez (es el
    mismo mecanismo que Rhie-Chow, sin el termino explicito).
    """
    xp = met.xp
    bc = _bc(bc, xp)

    def borde(lado, cara_u, cara_v):
        if bc[lado] is None:
            return cara_u, cara_v
        return bc[lado]

    fu = xp.empty_like(met.Sx_xi)
    fv = xp.empty_like(met.Sx_xi)
    fu[:, 1:-1] = 0.5 * (u[:, :-1] + u[:, 1:])
    fv[:, 1:-1] = 0.5 * (v[:, :-1] + v[:, 1:])
    fu[:, 0], fv[:, 0] = borde("oeste", u[:, 0], v[:, 0])
    fu[:, -1], fv[:, -1] = borde("este", u[:, -1], v[:, -1])
    m_xi = fu * met.Sx_xi + fv * met.Sy_xi

    gu = xp.empty_like(met.Sx_eta)
    gv = xp.empty_like(met.Sx_eta)
    gu[1:-1] = 0.5 * (u[:-1] + u[1:])
    gv[1:-1] = 0.5 * (v[:-1] + v[1:])
    gu[0], gv[0] = borde("sur", u[0], v[0])
    gu[-1], gv[-1] = borde("norte", u[-1], v[-1])
    if corte is not None:                   # el corte es cara interior, no pared
        gu[0] = xp.where(corte, 0.5 * (u[0] + u[0, ::-1]), gu[0])
        gv[0] = xp.where(corte, 0.5 * (v[0] + v[0, ::-1]), gv[0])
    m_eta = gu * met.Sx_eta + gv * met.Sy_eta
    return m_xi, m_eta


def velocidad_de_flujos(met, m_xi, m_eta):
    """Flujos de cara -> velocidad de celda, resolviendo el 2x2 de la celda.

    Promedia las dos caras de cada familia e invierte `u . S = m`. Es la lectura
    de la velocidad para las fuerzas y los graficos, no lo que transporta.
    """
    mx = 0.5 * (m_xi[:, :-1] + m_xi[:, 1:])
    me = 0.5 * (m_eta[:-1] + m_eta[1:])
    ax = 0.5 * (met.Sx_xi[:, :-1] + met.Sx_xi[:, 1:])
    ay = 0.5 * (met.Sy_xi[:, :-1] + met.Sy_xi[:, 1:])
    bx = 0.5 * (met.Sx_eta[:-1] + met.Sx_eta[1:])
    by = 0.5 * (met.Sy_eta[:-1] + met.Sy_eta[1:])
    det = ax * by - ay * bx
    return (mx * by - me * ay) / det, (ax * me - bx * mx) / det


def fraccion_par_impar(a):
    """Peso del modo tablero de ajedrez en un campo por celdas.

    Es el diagnostico del §2.1 del cartesiano: la componente de la divergencia
    residual sobre `(-1)^(i+j)`, normalizada. Con el gradiente compacto el modo
    no esta en el nucleo del operador y esta fraccion se queda en el ruido.
    """
    xp = xp_de(a)
    ny, nx = a.shape
    s = (-1.0) ** (xp.arange(ny)[:, None] + xp.arange(nx)[None, :])
    n = float(xp.linalg.norm(a))
    return abs(float((a * s).sum())) / np.sqrt(ny * nx) / max(n, 1e-300)

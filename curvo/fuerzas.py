r"""Fuerzas sobre el perfil, integradas sobre la linea `j=0` de la malla.

Aqui es donde la malla adaptada paga. En el solver cartesiano la superficie es
una escalera rasterizada, la longitud de arco es **sintetica**
(`ds = sqrt((n_y*vol_x)^2 + (n_x*vol_y)^2)`, `Simulador2D.py:6549`) y hay tres
estimadores de Cl que discrepan un 7 %. Aqui la superficie **es** la linea `j=0`:
`ds = |S_eta[0]|` y la normal es `S_eta[0]/|S_eta[0]|`, exactas, sin heuristica.

Convenio de signos, deducido de la metrica y no ajustado a posteriori. La celda
`(0,i)` recorre sus vertices en sentido antihorario, asi que `S_eta[0,i]`, que es
el segmento de pared girado 90 grados en ese mismo sentido, apunta **del cuerpo
hacia el fluido**: es la normal exterior del cuerpo. La traccion que el fluido
ejerce sobre el cuerpo es entonces

    t = sigma . n,   sigma = -p I + nu (grad u + grad u^T)

y la fuerza total, por unidad de densidad,

    F = - sum_i p_w,i * S_eta[0,i]  +  nu * sum_i grad(u_i) . S_eta[0,i].

El termino traspuesto se omite porque su integral sobre un contorno cerrado es
nula si la divergencia lo es, y aqui lo es a precision de maquina.

El gradiente de pared que sale de ahi es de **primer orden**: vale
`(6U/h)(1 - dy/2h)` sobre un Poiseuille exacto, o sea un error relativo de
`dy/(2 delta)` frente al espesor de capa limite. Con el paso de pared de la etapa
1 (2e-3) eso es ~2 %; con el de la etapa 2 (1.9e-4), 0.2 %. Se acepta a
proposito: usar una formula de un lado de 2.º orden con `u[0]` y `u[1]` daria
mejor numero pero **ya no seria el flujo que el esquema quita al fluido**, y
entonces el balance de cantidad de movimiento dejaria de cerrar.

**El flujo viscoso de pared es el mismo que usa el momento**, no una
reconstruccion aparte: `operadores.flujos_difusivos` con el Dirichlet de no
deslizamiento da `nu * grad(u) . S` en la cara, con el factor 2 de la media celda
y con los terminos cruzados de la metrica completa dentro. Asi la fuerza que se
integra es exactamente la que el esquema quita al fluido, y el balance cierra por
construccion en vez de por casualidad.

La presion de pared es `p[0,i]`. No es una aproximacion floja: la condicion de
frontera del Poisson en la pared es gradiente normal nulo, asi que el valor de
cara **es** el de la celda dentro de la discretizacion; extrapolar con `p[1]`
seria inventar una variacion que el operador no tiene.

Tres estimadores de Cl, que es el criterio que decide la migracion (hoy el
cartesiano da 0.6878 / 0.6508 / 0.6419, un 7 % de dispersion):

- `superficie`: la integral de arriba.
- `circulacion`: por el teorema de Stokes sobre las celdas, `Gamma = -sum rot(u)`
  hasta la linea `j=J`. Telescopa a la circulacion del contorno, asi que no hay
  que construir el lazo ni interpolar sobre el.
- `delta_cp`: `Cl = int (Cp_intrados - Cp_extrados) dx / c`, la forma de teoria
  de perfil delgado. Es la mas basta de las tres a proposito: es la que el
  cartesiano usaba y la que hay que poder comparar.
"""

from __future__ import annotations

import numpy as np

from . import operadores as op

__all__ = [
    "geometria_pared",
    "coeficiente_de_presion",
    "fuerzas",
    "circulacion",
    "estimadores_de_cl",
    "delta_cp_te",
]


_integral = getattr(np, "trapezoid", None) or np.trapz


def _host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a, dtype=float)


def _celdas_perfil(info):
    """Rango de celdas de `j=0` que son pared (el resto es corte de estela)."""
    i0, i1 = info["perfil"]
    return slice(int(i0), int(i1) - 1)


# ---------------------------------------------------------------------------
# Geometria de la pared
# ---------------------------------------------------------------------------
def geometria_pared(met, info):
    """Longitudes, normales y puntos medios de las caras de pared, en CPU.

    Devuelve un diccionario con `ds`, `nx`, `ny` (normal exterior unitaria),
    `x`, `y` (punto medio de la cara), `s` (arco acumulado desde el borde de
    salida del intrados) y `cuerda`.
    """
    c = _celdas_perfil(info)
    Sx = _host(met.Sx_eta[0])[c]
    Sy = _host(met.Sy_eta[0])[c]
    ds = np.hypot(Sx, Sy)
    X = _host(met.X[0])
    Y = _host(met.Y[0])
    x = 0.5 * (X[c.start:c.stop] + X[c.start + 1:c.stop + 1])
    y = 0.5 * (Y[c.start:c.stop] + Y[c.start + 1:c.stop + 1])
    xs = X[c.start:c.stop + 1]
    return {"ds": ds, "nx": Sx / ds, "ny": Sy / ds, "x": x, "y": y,
            "s": np.concatenate([[0.0], np.cumsum(ds)])[:-1] + 0.5 * ds,
            "cuerda": float(xs.max() - xs.min()),
            "i_le": int(np.argmin(x))}


def coeficiente_de_presion(met, p, info, u_inf):
    """`Cp = p / (0.5 u_inf^2)` sobre la pared. `p` es cinematica (p/rho)."""
    return _host(p[0])[_celdas_perfil(info)] / (0.5 * u_inf ** 2)


# ---------------------------------------------------------------------------
# Fuerzas
# ---------------------------------------------------------------------------
def fuerzas(met, u, v, p, info, nu, u_inf=1.0, alfa=0.0, bc=None):
    """Cl, Cd y Cm del perfil, separados en parte de presion y viscosa.

    `bc` son las mismas fronteras con las que se resolvio el momento; hace falta
    porque el flujo viscoso de pared se evalua con `flujos_difusivos`, que es
    literalmente el termino que el esquema le quita al fluido por esa cara.
    """
    c = _celdas_perfil(info)
    g = geometria_pared(met, info)
    bc = bc or {}
    pared = {"oeste": bc.get("oeste"), "este": bc.get("este"),
             "sur": 0.0, "norte": None}

    fx_v = _host(op.flujos_difusivos(met, u, nu, **pared)[1][0])[c]
    fy_v = _host(op.flujos_difusivos(met, v, nu, **pared)[1][0])[c]
    pw = _host(p[0])[c]
    Sx, Sy = g["nx"] * g["ds"], g["ny"] * g["ds"]

    Fp = np.array([-(pw * Sx).sum(), -(pw * Sy).sum()])
    Fv = np.array([fx_v.sum(), fy_v.sum()])

    a = np.radians(alfa)
    arrastre = np.array([np.cos(a), np.sin(a)])
    sustenta = np.array([-np.sin(a), np.cos(a)])
    q = 0.5 * u_inf ** 2 * g["cuerda"]

    # Momento respecto al cuarto de cuerda, positivo a encabritar.
    x0 = g["x"].min() + 0.25 * g["cuerda"]
    brazo_x, brazo_y = g["x"] - x0, g["y"]
    fx = -pw * Sx + fx_v
    fy = -pw * Sy + fy_v
    M = float((brazo_x * fy - brazo_y * fx).sum())

    return {"F_presion": Fp, "F_viscosa": Fv,
            "Cl": float((Fp + Fv) @ sustenta) / q,
            "Cd": float((Fp + Fv) @ arrastre) / q,
            "Cl_presion": float(Fp @ sustenta) / q,
            "Cd_presion": float(Fp @ arrastre) / q,
            "Cd_viscoso": float(Fv @ arrastre) / q,
            "Cm": M / (q * g["cuerda"]),
            "cuerda": g["cuerda"]}


# ---------------------------------------------------------------------------
# Circulacion
# ---------------------------------------------------------------------------
def _a_caras(met, a, bc, corte, lado_sur, lado_norte):
    """Interpola un campo por celdas a las caras xi y eta.

    Mismo criterio que `proyeccion.flujos_de_velocidad`: media de las dos celdas
    en el interior, valor de frontera donde lo hay y valor de la celda donde la
    frontera es de gradiente nulo. El corte de estela es cara interior y su valor
    es la media con la celda espejo.
    """
    xp = met.xp
    f_xi = xp.empty_like(met.Sx_xi)
    f_xi[:, 1:-1] = 0.5 * (a[:, :-1] + a[:, 1:])
    f_xi[:, 0] = a[:, 0] if bc.get("oeste") is None else bc["oeste"]
    f_xi[:, -1] = a[:, -1] if bc.get("este") is None else bc["este"]

    f_eta = xp.empty_like(met.Sx_eta)
    f_eta[1:-1] = 0.5 * (a[:-1] + a[1:])
    f_eta[0] = lado_sur
    f_eta[-1] = a[-1] if lado_norte is None else lado_norte
    if corte is not None:
        f_eta[0] = xp.where(corte, 0.5 * (a[0] + a[0, ::-1]), f_eta[0])
    return f_xi, f_eta


def lazo(met, info, radio):
    """Indices `(j, i_a)` del lazo que rodea al perfil a distancia `radio`.

    El lazo es el contorno del bloque de celdas `j < J`, `i_a <= i <= nx-1-i_a`.
    Que sea un lazo cerrado **alrededor del perfil** y no alrededor de todo el
    dominio es lo que importa: las caras del corte de estela que quedan dentro
    del bloque son interiores y se cancelan, y las dos lineas radiales `i = i_a` e
    `i = nx-1-i_a` arrancan del mismo punto fisico del corte, asi que juntas
    forman una sola curva que **cruza la estela una vez**. Sin eso se estaria
    midiendo la vorticidad total del dominio, que incluye el vortice de arranque
    y tiende a cero.

    `radio` se mide en cuerdas: hacia fuera, como distancia normal a la pared en
    el morro; hacia atras, como distancia al borde de salida.
    """
    X = _host(met.X)
    Y = _host(met.Y)
    g = geometria_pared(met, info)
    i0, i1 = info["perfil"]
    le = int(i0) + g["i_le"]
    d = np.hypot(X[:, le] - X[0, le], Y[:, le] - Y[0, le]) / g["cuerda"]
    J = int(np.searchsorted(d, radio))
    J = max(2, min(J, met.ny))

    x_te = X[0, int(i0)]
    estela = X[0, :int(i0)]                     # rama baja, del corte al perfil
    objetivo = x_te + radio * g["cuerda"]
    i_a = int(np.searchsorted(-estela, -objetivo))
    return J, max(1, min(i_a, int(i0) - 1))


def circulacion(met, u, v, info, bc=None, corte=None, hasta=None, desde=0,
                sur=(0.0, 0.0)):
    """`Gamma` del lazo, por Stokes sobre las celdas `j < hasta`, `desde <= i`.

    La circulacion discreta de una celda es la suma de `u_cara . arista` sobre
    sus cuatro aristas; al sumar celdas, las aristas interiores se cancelan y
    queda exactamente la circulacion del contorno del bloque. No hay que
    construir el lazo ni interpolar sobre el, que es de donde salia la
    dependencia del tamano del lazo en el cartesiano (Gamma caia de 0.65 a 0.29
    al agrandarlo).

    `sur` es la velocidad en la pared: `(0, 0)` por no deslizamiento, o `None`
    para gradiente nulo cuando `j=0` no es una pared.

    Devuelve `Gamma` con el signo aeronautico: positivo el que da sustentacion
    positiva.
    """
    bc = bc or {}
    su, sv = (u[0], v[0]) if sur is None else sur
    u_xi, u_eta = _a_caras(met, u, bc, corte, su, None)
    v_xi, v_eta = _a_caras(met, v, bc, corte, sv, None)
    # aristas: la de una cara xi es (-Sy_xi, Sx_xi); la de una cara eta,
    # (Sy_eta, -Sx_eta).
    C_xi = u_xi * (-met.Sy_xi) + v_xi * met.Sx_xi
    C_eta = u_eta * met.Sy_eta + v_eta * (-met.Sx_eta)
    circ = op.divergencia(C_xi, -C_eta)
    hasta = met.ny if hasta is None else hasta
    circ = circ[:hasta, desde:met.nx - desde]
    return -float(circ.sum())


def estimadores_de_cl(met, u, v, p, info, nu, u_inf=1.0, alfa=0.0, bc=None,
                      corte=None, radios=(0.15, 0.3, 0.6, 1.0, 1.5)):
    """Los tres estimadores de Cl del criterio de F5, en un diccionario.

    `radios` son los tamanos de lazo, en cuerdas, con los que medir `Gamma`; el
    criterio del plan pide que sea estable de 0.15c a 1.5c dentro del 5 % (hoy el
    cartesiano va de 0.65 a 0.29).
    """
    f = fuerzas(met, u, v, p, info, nu, u_inf, alfa, bc)
    cuerda = f["cuerda"]
    gamma = {}
    for r in radios:
        J, i_a = lazo(met, info, r)
        gamma[r] = circulacion(met, u, v, info, bc, corte, hasta=J, desde=i_a)
    valores = np.array(list(gamma.values()))
    medio = float(np.mean(valores))

    return {"superficie": f["Cl"],
            "circulacion": 2.0 * medio / (u_inf * cuerda),
            "delta_cp": _cl_por_delta_cp(met, p, info, u_inf, cuerda),
            "gamma": gamma,
            "gamma_dispersion": float(np.ptp(valores) / max(abs(medio), 1e-30)),
            "fuerzas": f}


def _cl_por_delta_cp(met, p, info, u_inf, cuerda):
    """`Cl = int (Cp_intrados - Cp_extrados) dx / c`, forma de perfil delgado.

    Es deliberadamente la mas basta de las tres: desprecia la componente `n_x` y
    solo vale a angulo pequeno. Se incluye porque es la que usa el solver
    cartesiano y hay que poder comparar el mismo numero.
    """
    g = geometria_pared(met, info)
    cp = coeficiente_de_presion(met, p, info, u_inf)
    le = g["i_le"]
    ramas = []
    for tramo in (slice(None, le + 1), slice(le, None)):
        x, y, c = g["x"][tramo], g["y"][tramo], cp[tramo]
        orden = np.argsort(x)
        ramas.append((x[orden], c[orden], np.sign(np.mean(y))))
    (x1, c1, s1), (x2, c2, s2) = ramas
    if s1 > 0:                                  # que la primera sea el intrados
        (x1, c1), (x2, c2) = (x2, c2), (x1, c1)
    rejilla = np.linspace(max(x1.min(), x2.min()), min(x1.max(), x2.max()), 400)
    delta = np.interp(rejilla, x1, c1) - np.interp(rejilla, x2, c2)
    return float(_integral(delta, rejilla) / cuerda)


def delta_cp_te(met, p, info, u_inf):
    """Salto de Cp entre las dos caras del borde de salida.

    La condicion de Kutta dice que tiene que ser cero. En el cartesiano vale
    -0.092, el doble de la tolerancia, y hace falta un parche explicito
    (`apply_kutta_te`) para sostenerla. Aqui el borde de salida es un punto de la
    malla y las dos caras son vecinas del corte, asi que la condicion tiene que
    salir sola o no salir.
    """
    cp = coeficiente_de_presion(met, p, info, u_inf)
    return float(cp[0] - cp[-1])

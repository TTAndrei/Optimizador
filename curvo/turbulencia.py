r"""Spalart-Allmaras sobre la malla curvilinea.

Una ecuacion de transporte para `nu_tilde`, de la que sale la viscosidad
turbulenta `nu_t = nu_tilde * f_v1`:

    D(nu_tilde)/Dt = c_b1 S~ nu_tilde
                   + (1/sigma) [ div((nu + nu_tilde) grad nu_tilde)
                                 + c_b2 |grad nu_tilde|^2 ]
                   - c_w1 f_w (nu_tilde / d)^2

Tres diferencias con la version del solver cartesiano (`Simulador2D.py:3218`),
y las tres son consecuencia de la malla, no gusto:

1. **La distancia a pared es exacta.** Se calcula como la distancia minima del
   centroide de cada celda a los segmentos del perfil -- los segmentos de verdad,
   que son los de la linea `j=0`. El cartesiano la saca de una distancia con
   signo en celdas por el minimo local `min(dx, dy)`, y en malla estirada eso se
   equivoca un 35 % (medido en conductos). Aqui no hay aproximacion: es geometria
   y se calcula una vez.

   **No vale el arco acumulado a lo largo de la linea eta**, que seria lo barato:
   en la estela la linea `j=0` es el corte, no una pared, y la distancia a la
   pared es la distancia al perfil, que esta aguas arriba.

2. **El transporte es el mismo que el del momento**: conveccion-difusion
   implicita y conservativa de `conveccion`, con `nu + nu_tilde` como coeficiente
   de cara. El termino difusivo va en su forma conservativa, no expandido en
   laplaciano mas gradiente al cuadrado; el `c_b2 |grad|^2` queda aparte, que es
   como esta escrito el modelo.

3. **Sin sub-pasos, pero por el termino fuente, no por la difusion.** El
   cartesiano subdivide el paso por estabilidad difusiva
   (`dt_visc = 0.25 dx^2 / nu_eff`). Aqui la difusion es implicita y ese limite
   no existe -- pero **no es el que manda**. El que manda es la produccion, que
   es lineal en `nu_tilde` y con signo positivo: `c_b1 S~ dt` vale 0.45 con el
   `dt` de trabajo a Re = 1e5, o sea `nu_tilde` crece un 57 % por paso y el
   modelo revienta en 25 pasos (medido: `nu_t/nu` = 4685 al paso 20, momento a
   NaN al 25).

   La salida no es sub-dividir sino **integrar el par produccion-destruccion en
   forma cerrada**. Con `S~`, `f_w` y `d` congelados en el paso, la parte fuente
   de la ecuacion es la logistica

       d(nt)/dt = a nt - b nt^2,   a = c_b1 S~,   b = c_w1 f_w / d^2

   cuya solucion exacta `nt / (e^{-a dt} + b nt (1 - e^{-a dt})/a)` es positiva,
   monotona y **acotada por el equilibrio local `a/b`** para cualquier `dt`. La
   produccion deja de poder disparar el campo y la destruccion deja de necesitar
   el tratamiento implicito por puntos. Ver `fuente_logistica`.

**Euler atras, no BDF2.** `nu_tilde` tiene que ser positiva y BDF2 no es monotono
(barrera de Bolley-Crouzeix, medido en F3: pico 1.077 con el mismo `dt`). El
momento puede ir con BDF2; esto no.
"""

from __future__ import annotations

import numpy as np

from . import conveccion as cv
from . import operadores as op

__all__ = ["CONSTANTES", "distancia_a_pared", "viscosidad_turbulenta",
           "terminos", "fuente_logistica", "avanzar_sa"]

CONSTANTES = {
    "cb1": 0.1355,
    "cb2": 0.622,
    "sigma": 2.0 / 3.0,
    "kappa": 0.41,
    "cw2": 0.3,
    "cw3": 2.0,
    "cv1": 7.1,
    "chi_max": 5.0e3,          # tope de seguridad de chi = nu_tilde/nu
    "stilde_suelo": 0.3,       # S~ >= 0.3 |omega|, el recorte estandar
    "r_max": 10.0,
}
CONSTANTES["cw1"] = (CONSTANTES["cb1"] / CONSTANTES["kappa"] ** 2
                     + (1.0 + CONSTANTES["cb2"]) / CONSTANTES["sigma"])


def distancia_a_pared(met, info, trozo=4096):
    """Distancia de cada centroide al perfil. Geometria pura, se calcula una vez.

    Minimo sobre los segmentos de la linea `j=0` que son pared. Se trocea por
    columnas de celdas para no construir la matriz entera celda x segmento.
    """
    xp = met.xp
    i0, i1 = int(info["perfil"][0]), int(info["perfil"][1])
    ax, ay = met.X[0, i0:i1 - 1], met.Y[0, i0:i1 - 1]
    bx, by = met.X[0, i0 + 1:i1], met.Y[0, i0 + 1:i1]
    ex, ey = bx - ax, by - ay
    largo2 = xp.maximum(ex * ex + ey * ey, 1e-300)

    xc = met.xc.ravel()
    yc = met.yc.ravel()
    fuera = xp.empty_like(xc)
    for k in range(0, xc.size, trozo):
        px = xc[k:k + trozo, None] - ax[None, :]
        py = yc[k:k + trozo, None] - ay[None, :]
        t = xp.clip((px * ex[None, :] + py * ey[None, :]) / largo2[None, :], 0.0, 1.0)
        dx = px - t * ex[None, :]
        dy = py - t * ey[None, :]
        fuera[k:k + trozo] = xp.sqrt((dx * dx + dy * dy).min(axis=1))
    return fuera.reshape(met.J.shape)


def viscosidad_turbulenta(nu_tilde, nu, xp=np):
    """`nu_t = nu_tilde * f_v1`, con el tope de chi."""
    c = CONSTANTES
    chi = xp.clip(nu_tilde / nu, 0.0, c["chi_max"])
    chi3 = chi ** 3
    return nu_tilde * chi3 / (chi3 + c["cv1"] ** 3)


def terminos(nt, om, d, nu, xp=np):
    """Terminos algebraicos del modelo: `(produccion, destruccion, fv1)`.

    Separados del transporte para poder comprobarlos solos. La prueba que los
    fija es el **equilibrio de la capa logaritmica**: con `nu_tilde = kappa u_t d`
    y `omega = u_t / (kappa d)`, produccion, destruccion y difusion se cancelan
    exactamente, y esa cancelacion es justo la definicion de `c_w1`. Si `S~`, `r`
    o `f_w` estan mal, no cierra.
    """
    c = CONSTANTES
    chi = xp.clip(nt / nu, 0.0, c["chi_max"])
    chi3 = chi ** 3
    fv1 = chi3 / (chi3 + c["cv1"] ** 3)
    fv2 = 1.0 - chi / (1.0 + chi * fv1)

    d2_inv = 1.0 / xp.maximum(d * d, 1e-30)
    s_tilde = om + nt * fv2 * d2_inv / c["kappa"] ** 2
    s_tilde = xp.maximum(s_tilde, c["stilde_suelo"] * om)
    s_tilde = xp.maximum(s_tilde, 1e-12)

    r = xp.minimum(nt * d2_inv / (s_tilde * c["kappa"] ** 2), c["r_max"])
    g = r + c["cw2"] * (r ** 6 - r)
    fw = g * ((1.0 + c["cw3"] ** 6) / (g ** 6 + c["cw3"] ** 6)) ** (1.0 / 6.0)

    return (c["cb1"] * s_tilde * nt,          # produccion
            c["cw1"] * fw * nt * nt * d2_inv,  # destruccion
            fv1)


def fuente_logistica(nt, om, d, nu, dt, xp=np):
    """`nu_tilde` tras integrar solo la parte fuente, en forma cerrada.

    La fuente del modelo es `a nt - b nt^2` con `a = c_b1 S~` y
    `b = c_w1 f_w / d^2`. Congelando `a` y `b` (dependen de `nt` por `f_v2` y
    `f_w`, que varian mucho mas despacio) la ecuacion es la logistica y se
    integra exacta:

        nt(dt) = nt / (e^{-a dt} + b nt g),   g = (1 - e^{-a dt}) / a

    Positiva siempre, monotona, y con techo el equilibrio local `a/b` --
    justo lo que la produccion explicita no tenia: con `a dt = 0.45` crecia un
    57 % por paso sin nada que la parase.

    `g` se escribe con el desarrollo `dt (1 - a dt/2)` cuando `a dt` es pequeno;
    la forma directa es 0/0 ahi y en float32 se nota. En el otro extremo,
    `a dt` grande hunde `e^{-a dt}` a cero y el equilibrio pasa a valer `a/b`
    exacto -- finito salvo donde `nt` es cero, porque `b` lleva `f_w(nt)`
    congelada y se anula con ella. De ahi el suelo del denominador y el techo
    `chi_max`, que es el que ya declara el modelo como red de seguridad.
    """
    c = CONSTANTES
    chi = xp.clip(nt / nu, 0.0, c["chi_max"])
    chi3 = chi ** 3
    fv1 = chi3 / (chi3 + c["cv1"] ** 3)
    fv2 = 1.0 - chi / (1.0 + chi * fv1)

    d2_inv = 1.0 / xp.maximum(d * d, 1e-30)
    s_tilde = om + nt * fv2 * d2_inv / c["kappa"] ** 2
    s_tilde = xp.maximum(s_tilde, c["stilde_suelo"] * om)
    s_tilde = xp.maximum(s_tilde, 1e-12)

    r = xp.minimum(nt * d2_inv / (s_tilde * c["kappa"] ** 2), c["r_max"])
    g6 = r + c["cw2"] * (r ** 6 - r)
    fw = g6 * ((1.0 + c["cw3"] ** 6) / (g6 ** 6 + c["cw3"] ** 6)) ** (1.0 / 6.0)

    a = c["cb1"] * s_tilde
    b = c["cw1"] * fw * d2_inv
    z = a * dt
    dec = xp.exp(-z)
    g = dt * xp.where(z > 1e-4, (1.0 - dec) / xp.maximum(z, 1e-30),
                      1.0 - 0.5 * z)
    return xp.minimum(nt / xp.maximum(dec + b * nt * g, 1e-30),
                      c["chi_max"] * nu)


def _vorticidad(met, u, v, bc_u, bc_v, corte=None):
    """|omega_z| = |dv/dx - du/dy| por celda, con el gradiente Green-Gauss."""
    dv_dx = op.gradiente(met, v, corte=corte, **bc_v)[0]
    du_dy = op.gradiente(met, u, corte=corte, **bc_u)[1]
    return met.xp.abs(dv_dx - du_dy)


def avanzar_sa(met, nu_tilde, u, v, m_xi, m_eta, d, nu, dt, bc_u, bc_v,
               bc_sa=None, corte=None, correcciones=0):
    """Un paso de `nu_tilde`. Devuelve `(nu_tilde, nu_t)`.

    `d` es la distancia a pared de `distancia_a_pared` (fija, se pasa hecha).
    `bc_sa` sigue el convenio de siempre: `sur = 0.0` en la pared (no
    deslizamiento tambien para `nu_tilde`), `norte` el valor de corriente libre y
    los planos de salida con gradiente nulo.

    `correcciones = 0` **no** es upwind de 1.er orden: el bucle de `avanzar`
    sigue evaluando la correccion diferida una vez, sobre el campo del paso
    anterior. Es Picard retrasado un paso, y marchando en el tiempo llega al
    mismo punto fijo -- medido encadenando sobre la solucion manufacturada,
    mismo error a seis decimales que con 1 o 2, orden 2.00 y 2.02.
    """
    xp = met.xp
    c = CONSTANTES
    nt = xp.maximum(nu_tilde, 0.0)
    om = _vorticidad(met, u, v, bc_u, bc_v, corte)

    # Produccion y destruccion juntas, integradas exactas como logistica. Se
    # pasan al transporte como el incremento equivalente: `dt * fuente` vale
    # `nt_fuente - nt >= -nt`, asi que el termino independiente sigue siendo
    # positivo y `nu_tilde` no puede cambiar de signo.
    fuente = (fuente_logistica(nt, om, d, nu, dt, xp) - nt) / dt

    # c_b2 |grad nu_tilde|^2 / sigma, explicito. El resto de la difusion va en la
    # matriz, en forma conservativa y con el coeficiente por cara.
    gx, gy = op.gradiente(met, nt, corte=corte, **(bc_sa or {}))
    fuente = fuente + c["cb2"] / c["sigma"] * (gx * gx + gy * gy)

    nuevo, info = cv.avanzar(met, nt, m_xi, m_eta, dt=dt,
                             nu=(nu + nt) / c["sigma"], bc=bc_sa, corte=corte,
                             fuente=fuente, correcciones=correcciones)
    nuevo = xp.maximum(nuevo, 0.0)
    return nuevo, viscosidad_turbulenta(nuevo, nu, xp)

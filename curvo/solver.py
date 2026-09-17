r"""Navier-Stokes incompresible sobre la malla C, por paso fraccionado.

Junta lo cerrado en F3 y F4:

1. **momento**, con la conveccion-difusion implicita y conservativa de
   `conveccion`, transportado por los flujos de cara del paso anterior (que son
   solenoidales exactos) y con la presion vieja como termino fuente,
2. **proyeccion** de `proyeccion`: se corrigen los flujos de cara para que vuelvan
   a cerrar el balance, y la velocidad de celda con el gradiente Green-Gauss.

Lo que transporta son **los flujos de cara**, no la velocidad de celda. La
velocidad de celda solo entra en el termino convectivo y en las fuerzas. Que la
correccion de cara use la diferencia compacta `p_{i+1} - p_i` mientras la de
celda va por Green-Gauss es el mecanismo de Rhie-Chow sin el termino explicito, y
es lo que mantiene el desacoplo par-impar fuera (medido en F4: 1e-9 frente al
46.2 % del solver cartesiano).

Fronteras. `j=0` es la pared (no deslizamiento) en el tramo del perfil y **corte
de estela** en el resto, que es cara interior y no frontera. `j=ny-1` es el arco
del campo lejano, con la corriente libre impuesta. `i=0` e `i=nx-1` son el plano
de salida, con gradiente nulo. Para la presion, pared y campo lejano son Neumann
y el plano de salida Dirichlet -- el Poisson necesita una cara Dirichlet y esa es
la unica que lo es fisicamente.

**El angulo de ataque inclina la corriente, no la geometria**: una sola malla
sirve para la polar entera, y sobre la malla simetrica el Cl a alfa = 0 sale cero
a precision de maquina en vez de a una tolerancia.

Precision. En GPU conviene float32: la 3070 Ti hace fp64 a 1/64 de fp32. Con
float32 el paso cuesta 0.12 s sobre la malla de 21 000 celdas frente a 1.87 s en
CPU float64, y la respuesta es la misma (ver `tests/test_curvo_gpu.py`).

Con `turbulento=True` se acopla Spalart-Allmaras (`turbulencia.py`): el momento
pasa a difundir con `nu + nu_t` y se transporta `nu_tilde` con el mismo esquema,
con Euler atras siempre porque tiene que ser positiva.
"""

from __future__ import annotations

import time

import numpy as np

from . import conveccion as cv
from . import operadores as op
from . import proyeccion as pr
from . import turbulencia as tu
from .metrica import Metrica

__all__ = ["Solver"]


class Solver:
    """Estado del solver y el paso de tiempo. Todo vive en `xp` (numpy o cupy).

    Parametros principales:

    ==============  ===========================================================
    ``X, Y, info``  malla y rangos, tal como los devuelve `malla.generar_c`
    ``nu``          viscosidad cinematica (`conveccion.AIRE["nu"]` por defecto)
    ``u_inf``       modulo de la corriente libre
    ``alfa``        angulo de ataque en grados; inclina la corriente
    ``dt``          paso de tiempo
    ``bdf2``        2.º orden en el tiempo. Gana orden y **pierde monotonia**
                    (barrera de Bolley-Crouzeix, medido en F3: pico 1.077)
    ``turbulento``  acopla Spalart-Allmaras
    ``cronometro``  reparte el tiempo del paso entre sus etapas (ver `tiempos`)
    ==============  ===========================================================

    `correcciones` basta que sea 1. Lo que fija es la **velocidad** con que la
    correccion diferida llega a su punto fijo, no donde esta ese punto: cada paso
    reanuda la del anterior sobre un campo que apenas se mueve, asi que marchando
    en el tiempo se acumulan miles de iteraciones. Medido sobre la solucion
    manufacturada reanudando como reanuda el solver, 1 y 2 dan el mismo error a
    seis decimales, con orden 2.00 difusivo y 2.02 convectivo. Con una sola
    llamada desde frio no es asi, y por eso el test de orden usa 8.
    """

    def __init__(self, X, Y, info, nu=None, u_inf=1.0, alfa=0.0, dt=2e-3,
                 xp=np, dtype=np.float64, bdf2=False, correcciones=1,
                 correcciones_p=2, turbulento=False, nu_tilde_inf=3.0,
                 cronometro=False):
        self.xp = xp
        self.dtype = np.dtype(dtype)
        self.met = Metrica(xp.asarray(X, dtype=dtype), xp.asarray(Y, dtype=dtype))
        self.info = info
        self.corte = cv.corte_de_estela(info, self.met.nx, xp=xp)
        self.nu = cv.AIRE["nu"] if nu is None else nu
        self.u_inf = float(u_inf)
        self.alfa = float(alfa)
        self.dt = float(dt)
        self.bdf2 = bdf2
        self.correcciones = correcciones
        self.correcciones_p = correcciones_p
        self.paso_n = 0
        self.cronometro = cronometro
        self.tiempos = {}

        a = np.radians(self.alfa)
        self.U = self.u_inf * float(np.cos(a))
        self.V = self.u_inf * float(np.sin(a))

        self.bc_u = dict(oeste=None, este=None, sur=0.0, norte=self.U)
        self.bc_v = dict(oeste=None, este=None, sur=0.0, norte=self.V)
        self.bc_p = dict(oeste=0.0, este=0.0, sur=None, norte=None)
        self.bc_vel = dict(oeste=None, este=None, sur=(0.0, 0.0),
                           norte=(self.U, self.V))
        self.sis_p = pr.sistema_presion(self.met, self.bc_p, self.corte)

        self.turbulento = turbulento
        if turbulento:
            # `nu_tilde` de corriente libre: 3 nu es el valor estandar del modelo
            # para flujo externo.
            self.nu_tilde_inf = nu_tilde_inf * self.nu
            self.bc_sa = dict(oeste=None, este=None, sur=0.0,
                              norte=self.nu_tilde_inf)
            self.d_pared = tu.distancia_a_pared(self.met, info)

        self.arrancar()

    # ------------------------------------------------------------------
    def arrancar(self):
        """Estado inicial: el flujo potencial alrededor del perfil.

        Se proyecta la corriente libre con la pared ya impermeable. Sale un campo
        de divergencia nula que cumple la condicion de no penetracion, que es
        mucho mejor punto de partida que la corriente uniforme a secas: el
        transitorio de arranque no tiene que expulsar el caudal que atravesaba el
        cuerpo.
        """
        xp, met = self.xp, self.met
        m_xi, m_eta = met.flujo_uniforme(self.dtype.type(self.U),
                                         self.dtype.type(self.V))
        m_eta = m_eta.copy()
        m_eta[0] = xp.where(self.corte, m_eta[0], 0.0)
        self.m_xi, self.m_eta, _, _ = pr.proyectar(
            met, m_xi, m_eta, bc=self.bc_p, corte=self.corte,
            correcciones=4, sis=self.sis_p)
        self.u, self.v = pr.velocidad_de_flujos(met, self.m_xi, self.m_eta)
        self.p = xp.zeros_like(met.J)
        self.u_ant = self.v_ant = None
        if self.turbulento:
            self.nu_tilde = xp.full_like(met.J, self.nu_tilde_inf)
            self.nu_t = tu.viscosidad_turbulenta(self.nu_tilde, self.nu, xp)
        else:
            self.nu_t = None

    # ------------------------------------------------------------------
    def _marca(self, nombre, t0):
        """Acumula en `tiempos` lo que ha costado una etapa. Devuelve el reloj.

        Con cupy hay que **sincronizar antes de mirar el reloj**: los lanzamientos
        son asincronos y sin la sincronizacion el reparto mide encolado, no
        trabajo, y se lo carga todo a la primera etapa que baje un dato a la CPU.
        Por eso el cronometro es opcional: la sincronizacion cuesta.
        """
        if self.xp is not np:
            self.xp.cuda.Stream.null.synchronize()
        t = time.perf_counter()
        self.tiempos[nombre] = self.tiempos.get(nombre, 0.0) + (t - t0)
        return t

    def paso(self):
        """Un paso de tiempo. Devuelve el diccionario de info del Poisson."""
        met = self.met
        reloj = time.perf_counter() if self.cronometro else None
        gx, gy = op.gradiente(met, self.p, corte=self.corte, **self.bc_p)

        bdf2 = self.bdf2 and self.u_ant is not None
        nu_ef = self.nu if self.nu_t is None else self.nu + self.nu_t
        # `u` y `v` comparten matriz y jerarquia: los coeficientes salen de los
        # flujos, de `nu_ef` y de que caras son Dirichlet, no de su valor. Van en
        # serie, no entrelazadas: `avanzar` reasigna `A.b` en cada correccion.
        A, (b_u, b_v) = cv.sistema(met, self.m_xi, self.m_eta, nu_ef, self.dt,
                                   [self.bc_u, self.bc_v], self.corte, bdf2)
        comun = dict(dt=self.dt, nu=nu_ef, corte=self.corte,
                     correcciones=self.correcciones)
        ue, _ = cv.avanzar(met, self.u, self.m_xi, self.m_eta, bc=self.bc_u,
                           fuente=-gx, phi_ant=self.u_ant if bdf2 else None,
                           sis=(A, b_u), **comun)
        ve, _ = cv.avanzar(met, self.v, self.m_xi, self.m_eta, bc=self.bc_v,
                           fuente=-gy, phi_ant=self.v_ant if bdf2 else None,
                           sis=(A, b_v), **comun)
        if reloj is not None:
            reloj = self._marca("momento", reloj)

        mx, me = pr.flujos_de_velocidad(met, ue, ve, self.bc_vel, self.corte)
        self.m_xi, self.m_eta, phi, info = pr.proyectar(
            met, mx, me, dt=self.dt, bc=self.bc_p, corte=self.corte,
            correcciones=self.correcciones_p, sis=self.sis_p)

        self.u_ant, self.v_ant = self.u, self.v
        self.u, self.v = pr.corregir_velocidad(met, ue, ve, phi, self.dt,
                                               self.bc_p, self.corte)
        self.p = self.p + phi
        if reloj is not None:
            reloj = self._marca("presion", reloj)

        if self.turbulento:
            # Euler atras siempre: `nu_tilde` tiene que ser positiva y ningun
            # metodo multipaso de orden > 1 es incondicionalmente monotono.
            self.nu_tilde, self.nu_t = tu.avanzar_sa(
                met, self.nu_tilde, self.u, self.v, self.m_xi, self.m_eta,
                self.d_pared, self.nu, self.dt, self.bc_u, self.bc_v,
                self.bc_sa, self.corte)
            if reloj is not None:
                reloj = self._marca("turbulencia", reloj)

        self.paso_n += 1
        return info

    def reparto(self):
        """`tiempos` en porcentaje y en ms por paso. Vacio si no hay cronometro."""
        total = sum(self.tiempos.values())
        return {n: {"s": t, "%": 100.0 * t / total,
                    "ms_por_paso": 1e3 * t / max(self.paso_n, 1)}
                for n, t in sorted(self.tiempos.items(), key=lambda x: -x[1])}

    # ------------------------------------------------------------------
    def correr(self, pasos, parada=None, cada=25, traza=None):
        """Avanza `pasos` pasos. Con `parada` se para si el cambio baja de ahi.

        El cambio se mide como `max|u^{n+1} - u^n| / (u_inf * dt)`, es decir, la
        derivada temporal en unidades de la corriente libre: asi el criterio no
        depende de `dt` ni de la escala de velocidad.
        """
        xp = self.xp
        historia = []
        for k in range(pasos):
            previo_u, previo_v = self.u, self.v
            self.paso()
            if (k + 1) % cada == 0 or k == pasos - 1:
                cambio = float(xp.maximum(xp.abs(self.u - previo_u),
                                          xp.abs(self.v - previo_v)).max())
                cambio /= self.u_inf * self.dt
                historia.append((self.paso_n, cambio))
                if traza is not None:
                    traza(self, cambio)
                if parada is not None and cambio < parada:
                    break
        return historia

    # ------------------------------------------------------------------
    def divergencia(self):
        """Maximo de |div(m)| relativo al flujo de cara. Debe quedarse en el ruido."""
        xp = self.xp
        d = op.divergencia(self.m_xi, self.m_eta)
        esc = (xp.abs(self.m_xi[:, 1:]) + xp.abs(self.m_xi[:, :-1])
               + xp.abs(self.m_eta[1:]) + xp.abs(self.m_eta[:-1]))
        return float(xp.max(xp.abs(d) / xp.maximum(esc, 1e-30)))

    def campos(self):
        """Los campos en CPU y en float64, para dibujar o guardar."""
        def baja(a):
            return np.asarray(a.get() if self.xp is not np else a, dtype=float)
        return {n: baja(getattr(self, n)) for n in ("u", "v", "p")}

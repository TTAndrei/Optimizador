r"""Generador de malla C monobloque adaptada al cuerpo, metricas y validacion.

Solo CPU y numpy/scipy: esto corre antes de tocar la GPU, y el validador tiene que
poder pasar sobre miles de perfiles del historial del GA en segundos.

Topologia (indice i = xi recorre la superficie, j = eta sale de la pared):

        <- <- <- <- <- <- <-   j = M-1  (campo lejano)
      ^                     ^
    ^     ___---‾‾‾---___   |  i = N-1
   |    /  extrados     \‾‾‾===== corte de estela
   |   |  LE        TE  |   ‾‾‾===== (i y N-1-i son el mismo punto fisico)
   |    \___        ___/     |  i = 0
    v      ‾‾‾---__-         v
      v                     x = x_out

`j=0` ES el perfil: no hay mascara, no hay rasterizacion, no hay ghost cells.

La malla se guarda como campos de VERTICES `X[j,i]`, `Y[j,i]`. Las metricas salen
de los segmentos rectos entre vertices, nunca de diferenciar centros de celda: es
lo unico que da conservacion geometrica exacta (ver `calidad()['div_uinf']`).
"""

from __future__ import annotations

import os

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.linalg import solve_banded

__all__ = [
    "leer_dat",
    "redistribuir_superficie",
    "curva_inicial",
    "generar_c",
    "metricas",
    "calidad",
    "dibujar",
    "plan_de_pasos",
]

# =============================================================================
#  PARAMETROS DE MALLA  --  esto es lo que se toca
# =============================================================================
#
#  Todos son valores por defecto: cualquiera se puede pasar como argumento a
#  `generar_c(...)` o por linea de ordenes sin editar este fichero. Editar aqui
#  cambia el defecto para todo el proyecto.
#
#  Para ver el efecto de un cambio:
#      .venv/bin/python -m curvo.malla --figura --perfil profiles/NACA_0012_sharp
#

# --- Resolucion normal a la pared --------------------------------------------

DN_PARED = 8.0e-5
"""TAMANO MINIMO DE CELDA alrededor del perfil, en cuerdas.

`dn` = delta ene, espaciado en direccion **normal** a la pared. En este modulo
`dn` es siempre perpendicular a la superficie y `ds` siempre a lo largo de ella
(`ds_te` es el espaciado tangencial en el borde de salida).

Es el espesor de la primera capa pegada a la pared, y el parametro que manda:
fija la resolucion de la capa limite y, con el, casi todo lo demas.

    2.0e-3  ->  y+ ~ 10 a Re=1e5   (etapa 1: misma resolucion que el cartesiano)
    1.9e-4  ->  y+ ~ 1  de mediana, pero **1.88 en el morro**
    8.0e-5  ->  y+ < 1 en TODA la pared a Re=1e5                  <- DEFECTO

Por que 8.0e-5 y no 1.9e-4: `y+` no es uniforme a lo largo del perfil, porque
`u_tau` se dispara en el pico de succion. Medido en la validacion 1 (NACA 0012,
alfa=5, Re=1e5) con 1.9e-4: mediana 0.49 pero **maximo 1.88**, con `y+ > 1` en el
2 % de arco alrededor del borde de ataque (|s - s_le| < 0.016). El maximo escala
con `dn`, asi que el paso que deja el morro por debajo de 1 es 1.9e-4 / 1.88, y
8.0e-5 lo cumple con margen: **maximo 0.80, mediana 0.33, p95 0.67**.

El recorte se paga poco -- 34 853 a 37 534 celdas, **+7.7 %** -- y la razon es
ASPECTO_MAX: en el centro de la cuerda `ds/aspecto_max` ya vale ~1.2e-4, asi que
esas columnas se quedan donde estaban y **el refinado se concentra solo donde
`ds` es pequeno, es decir en el morro y el borde de salida**, que es exactamente
donde hacia falta. Lo unico global son las 7 capas extra (92 -> 99) que necesita
el plan de referencia por arrancar mas fino.

De propina, resolucion de capa limite a media cuerda: celdas por debajo de
`y+ = 15` de 16 a 23, y en el morro de 4 a 10.

Bajarlo mete mas capas (el numero sale de DISTANCIA_LEJOS y CRECIMIENTO) y sube
la relacion de aspecto junto a la pared, que es normal y deseable. **Tiene que
ser menor que el radio del borde de ataque**: una primera celda mas gruesa que
el morro lo envuelve entero y la ortogonalidad ahi se hunde a ~54 grados.
Medido sobre un perfil con radio de morro 2.3e-3: con DN_PARED=2e-3 sale 53.9
grados, con 1e-3 sube a 62.5. Ver AJUSTAR_DN_PARED.
"""

CRECIMIENTO = 1.12
"""Razon geometrica del espaciado normal entre capas consecutivas.

Cada capa es CRECIMIENTO veces mas gruesa que la anterior. Mas bajo = transicion
mas suave y mas precision lejos de la pared, a cambio de mas capas y mas coste.
Por encima de ~1.3 la malla empieza a ser mala para el solver (el validador avisa
a 1.30 y bloquea a 2.0).
"""

DISTANCIA_LEJOS = 8.0
"""Hasta donde marcha la malla desde la pared, en cuerdas.

Fija la frontera de campo lejano. El numero de capas se deduce de este valor
junto con DN_PARED, CRECIMIENTO y DN_MAX; no se pide a mano.
"""

DN_MAX = np.inf
"""Tope absoluto del espaciado normal. `inf` = sin tope (defecto).

Sin tope, dn crece geometricamente hasta la ultima capa: medido con los valores
por defecto, de 1.78e-3 a 0.914, un factor 514 en 56 capas. **No es patologico**
-- el crecimiento es autosemejante, la celda mas gruesa mide ~1/8 de su distancia
al perfil -- pero tampoco esta acotado por diseno, solo por el hecho de que
DISTANCIA_LEJOS fija el numero de capas.

Ponerlo sirve para dos cosas: garantizar resolucion minima en el campo lejano
cuando se alarga mucho el dominio, y no dejar que un CRECIMIENTO alto genere
celdas que se salgan de la escala del problema. **Cuesta celdas**: una vez
alcanzado el tope el avance pasa a ser lineal, asi que hacen falta mas capas para
llegar a DISTANCIA_LEJOS.

Es un tope sobre el **paso prescrito**, no sobre el espaciado final: la marcha
impone ortogonalidad y volumen, no distancia, asi que el espaciado realizado
puede pasarse en torno a un 2 %.
"""

AJUSTAR_DN_PARED = 0.0
"""Recorta DN_PARED a este factor por el radio del morro. 0 = desactivado.

Con 0.5, un perfil de morro afilado usa automaticamente un paso de pared la mitad
de su radio de borde de ataque. Util cuando se mallan geometrias heterogeneas.
**Aviso medido**: sobre el historial del GA empeora (12.1 % de mallas validas
frente a 15.9 % con paso fijo), porque bajar el paso sin bajar CRECIMIENTO mete
mas capas y mas ocasiones de que la malla se pliegue.
"""

N_CAPAS_PARED = 15
"""Capas pegadas a la pared que crecen a CRECIMIENTO_PARED en vez de CRECIMIENTO.

Un bloque de capa limite: espaciado normal casi constante cerca de la pared y
crecimiento normal a partir de ahi. 0 = desactivado.

**15 con CRECIMIENTO_PARED = 1 es el defecto**, junto con DN_PARED = 8.0e-5:
medido sobre NACA 0012 a Re = 1e5, eso da `y+` del primer centro de **0.33 de
mediana, 0.67 de p95 y 0.80 de maximo**, con 8 celdas dentro de la subcapa
viscosa (`y+ < 5`) y 23 por debajo de `y+ = 15` a media cuerda. Cuesta 37 534
celdas frente a 21 065 de la etapa 1.

Subirlo por encima de 15 **no toca la pared**: con DN_PARED = 8.0e-5 el suelo de
ASPECTO_MAX ya manda sobre las primeras capas de casi todas las columnas, asi que
25 capas de bloque dan exactamente el mismo `y+` y el mismo reparto por debajo de
`y+ = 15` -- medido, 109 capas contra 99 y ni una celda mas util cerca del
cuerpo. Las 10 capas extra se van al campo intermedio.

Por que hace falta: con los valores por defecto caben 14 capas dentro de una capa
limite de 0.07c, pero `dn` ya ha crecido x5.4 al llegar a la capa 15 (de 1.97e-3
a 1.08e-2). La resolucion dentro de la capa limite no es uniforme, y el grueso
del cizallamiento esta en el primer 20 % de su espesor.

El `ds` no necesita este tratamiento: **medido, crece solo un 11 % en esas 14
capas** (la curva se alarga muy poco mientras esta pegada al cuerpo).
"""

CRECIMIENTO_PARED = 1.0
"""Razon de crecimiento dentro de la capa de pared. 1.0 = espaciado constante."""

ASPECTO_MAX = 100.0
"""Relacion de aspecto que no se deja pasar a la primera capa. `inf` = desactivado.

`dn` no es un escalar: es un perfil `dn(xi)`. El corte de estela arrastraba el
paso de pared 18 cuerdas aguas abajo, donde el espaciado tangencial ya vale 1.7,
y ahi la celda salia de 1.9e-4 x 1.73 -- **relacion de aspecto 9142**, y no por
resolver nada: en la estela lejana no hay capa limite que resolver.

La regla es `dn(xi) = max(dn_pared, ds(xi) / ASPECTO_MAX)`, asi que solo actua
donde `ds` se dispara (la estela) y **no toca el perfil**: sobre el cuerpo la
relacion de aspecto de la primera capa vale ~70 a y+=1 y ~7 a y+=10, por debajo
del tope. Cada columna resuelve entonces su propia razon de crecimiento por
biseccion para llegar a DISTANCIA_LEJOS con el mismo numero de capas, que es lo
que mantiene la malla estructurada y el campo lejano a la misma distancia.

Bajarlo mucho empieza a comerse la resolucion de la estela cercana; subirlo lo
desactiva de hecho.
"""

# --- Resolucion a lo largo de la superficie ----------------------------------

N_SUPERFICIE = 256
"""Numero de puntos que recorren el perfil, de borde de salida a borde de salida.

Fija el espaciado tangencial medio: perimetro / N_SUPERFICIE. Subirlo mejora poco
la ortogonalidad (medido: 256 -> 512 pasa de 57.3 a 59.9 grados) y cuesta celdas
en proporcion directa, asi que no es la palanca para arreglar una malla mala.
"""

RAZON_LE = 0.08
"""Espaciado en el borde de ataque, como fraccion del espaciado maximo.

0.08 = las celdas del morro son ~12 veces mas finas que las del centro de la
cuerda. Bajarlo agrupa mas puntos en el morro.
"""

RAZON_TE = 0.15
"""Lo mismo para el borde de salida."""

ANCHO_LE = 0.05
"""Anchura de la zona de agrupamiento del borde de ataque, en fraccion de arco.

Cuanto se extiende el refinado del morro a lo largo de la superficie.
"""

ANCHO_TE = 0.05
"""Lo mismo para el borde de salida."""

# --- Estela y tamano del dominio ---------------------------------------------

N_ESTELA = 64
"""Puntos a lo largo de CADA lado del corte de estela.

Los dos lados son espejo exacto uno del otro, que es lo que hace que el
emparejamiento `i <-> N-1-i` del corte sea una permutacion de indices y no una
interpolacion.
"""

X_SALIDA = 18.0
"""Posicion del plano de salida, aguas abajo del borde de ataque, en cuerdas.

Manda mas que la extension lateral: el estudio de dominio del solver cartesiano
midio que truncar la estela es el error dominante (Lx=8 da L/D 27.70 frente a
31.70 con Lx=24).
"""

CRECIMIENTO_ESTELA_MAX = 1.25
"""Tope de crecimiento del espaciado a lo largo de la estela.

La razon real se resuelve por biseccion para clavar X_SALIDA partiendo del
espaciado del borde de salida. Si N_ESTELA es demasiado bajo para llegar con
este tope, `generar_c` falla con un mensaje que lo dice.
"""

# --- Borde de salida romo ----------------------------------------------------

N_BASE_MAX = 8
"""Tope de puntos por media base del borde de salida.

El numero real sale de gap/2 dividido por el espaciado de pared. Una base mas
fina que DN_PARED no se puede resolver y se afila al punto medio, dejando
constancia en `info["te_afilado"]`.
"""

TOL_GAP = 1e-4
"""Espesor de base, relativo a la cuerda, por encima del cual el TE es romo."""

# --- Marcha hiperbolica  (avanzado: tocar solo si la malla se pliega) --------

MEZCLA_VOLUMEN = 0.30
"""Cuanto se suaviza el volumen de celda prescrito a lo largo de la superficie.

0 = el area de cada celda sigue el espaciado local exacto; 1 = sigue una version
suavizada. Subirlo alisa el campo lejano y ayuda en bordes de salida romos
(medido: 94 -> 30 celdas cruzadas), a cambio de adaptarse menos a la geometria.
"""

DISIPACION = 0.3
"""Amortiguacion implicita de la marcha, sobre el INCREMENTO de cada paso.

Es lo que sostiene la marcha en zonas concavas. Actua sobre el incremento y no
sobre la posicion: suavizar posiciones es lo que colapsa el borde de ataque.
"""

DISIPACION_CURVA = 8.0
"""Refuerzo de la amortiguacion donde la superficie es concava.

**Medido: reforzarla tambien en las esquinas convexas empeora** (con 25 aparecen
41 celdas cruzadas en un caso que con 8 sale limpio). Subirlo mucho tampoco
arregla un borde de salida romo.
"""



# ----------------------------------------------------------------------------
# Lectura de geometria
# ----------------------------------------------------------------------------
def leer_dat(ruta):
    """Lee un .dat Selig y devuelve (px, py) del contorno, sin transformar.

    No escala, no rota y NO recorta el borde de salida: el recorte de
    `parse_geometry_file` existe para garantizar 2 celdas de espesor al IBM y
    aqui no hace falta. El angulo de ataque tampoco se aplica a la geometria:
    en malla adaptada se inclina la corriente libre, asi que una sola malla
    sirve para la polar entera.
    """
    pts = []
    with open(ruta, "r", encoding="utf-8") as f:
        lineas = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    for ln in lineas[1:]:                      # primera linea = titulo
        partes = ln.replace(",", " ").split()
        if len(partes) < 2:
            continue
        try:
            pts.append((float(partes[0]), float(partes[1])))
        except ValueError:
            continue
    p = np.asarray(pts, dtype=np.float64)
    if len(p) < 20:
        raise ValueError(f"{ruta}: solo {len(p)} puntos legibles")
    return p[:, 0], p[:, 1]


# ----------------------------------------------------------------------------
# Redistribucion de la superficie
# ----------------------------------------------------------------------------
def _distribucion(h, n):
    """Muestrea n valores del parametro t en [0,1] con espaciado objetivo h(t).

    h se da sobre una rejilla fina uniforme. Integra dt/h y muestrea uniforme
    en la primitiva: siempre monotona, sin resolver ninguna ecuacion.
    """
    t = np.linspace(0.0, 1.0, len(h))
    w = np.concatenate([[0.0], np.cumsum(0.5 * (1.0 / h[1:] + 1.0 / h[:-1]) * np.diff(t))])
    w /= w[-1]
    return np.interp(np.linspace(0.0, 1.0, n), w, t)


def _atractores(t, h_max, focos, periodico=False):
    """Espaciado objetivo: h_max modulado por gaussianas en cada foco.

    focos = [(t_foco, razon, ancho), ...] con razon = h_foco / h_max.
    Producto de factores => siempre positivo, y focos solapados se combinan.

    `periodico=False` por defecto: el parametro de la superficie va de esquina a
    esquina del borde de salida, no es ciclico. Con distancia periodica los focos
    de t=0 y t=1 se solapan y el agrupamiento del TE sale al cuadrado.
    """
    h = np.full_like(t, h_max)
    for t0, razon, ancho in focos:
        d = np.abs(t - t0)
        if periodico:
            d = np.minimum(d, 1.0 - d)
        h *= 1.0 - (1.0 - razon) * np.exp(-((d / ancho) ** 2))
    return h


def _densificar(p, n=6001):
    """Spline cubico abierto por una cara, muestreado denso y su arco acumulado."""
    if len(p) < 4:
        u = np.linspace(0.0, 1.0, n)
        d = np.column_stack([np.interp(u, np.linspace(0, 1, len(p)), p[:, k]) for k in (0, 1)])
    else:
        tck, _ = splprep([p[:, 0], p[:, 1]], s=0.0, per=False, k=3)
        xs, ys = splev(np.linspace(0.0, 1.0, n), tck)
        d = np.column_stack([xs, ys])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(d, axis=0).T))])
    return d, s


def _arco_del_morro(x, s):
    """Posicion en arco del minimo de x, refinada por parabola.

    Tomar `argmin` a secas deja el morro descentrado medio intervalo del
    muestreo, y con eso el atractor del LE se descentra: medido, t_le = 0.4999
    en vez de 0.5 en un NACA 0012, que rompe la simetria de la malla en 2.5e-4 y
    con ella el test exacto de Cl = 0 a alpha = 0.
    """
    k = int(np.argmin(x))
    if k == 0 or k == len(x) - 1:
        return float(s[k])
    x0, x1, x2 = x[k - 1], x[k], x[k + 1]
    den = x0 - 2.0 * x1 + x2
    if abs(den) < 1e-300:
        return float(s[k])
    h = 0.5 * (s[k + 1] - s[k - 1])
    return float(s[k] + h * (x0 - x2) / (2.0 * den))


def redistribuir_superficie(px, py, n_sup=N_SUPERFICIE, razon_le=RAZON_LE,
                            razon_te=RAZON_TE, ancho_le=ANCHO_LE, ancho_te=ANCHO_TE,
                            dn_pared=DN_PARED, n_base_max=N_BASE_MAX, tol_gap=TOL_GAP):
    """Reparametriza el contorno por longitud de arco, agrupando en LE y TE.

    Entrada en orden Selig (TE extrados -> LE -> TE intrados). Salida en el
    orden de recorrido de la malla C, **cerrada**: el primer y el ultimo punto
    son el mismo, y son el punto del que cuelga el corte de estela.

    Dos cosas que no son opcionales:

    - **Las dos caras se splinean por separado.** Un spline periodico sobre el
      lazo entero redondea las esquinas reales: el pico del borde de salida
      afilado y la nariz en cuna a la que tiende el GA. Partiendo en LE y en las
      esquinas de la base, cada esquina queda exacta.
    - **Con borde de salida romo el corte arranca del punto medio de la base**,
      no de una esquina. El 100 % del historial del GA tiene base finita
      (`te_gap`), y colgar el corte de un solo punto mete celdas cruzadas en las
      dos uniones TE-estela. El numero de puntos de la base se deduce de
      `dn_pared`: **una base mas fina que el espaciado de pared no se puede
      resolver**, asi que por debajo de medio paso se afila al punto medio y se
      deja constancia en `info["te_afilado"]`. Afilar es una decision explicita,
      no un accidente: el desplazamiento es de gap/2, sub-celda por definicion.

    La distribucion es simetrica en el parametro, asi que un perfil simetrico da
    una malla simetrica y el test de Cl = 0 a alpha = 0 es exacto.

    Devuelve (X, Y, info) con info = {n_base, gap, i_le, ds_te}.
    """
    p = np.column_stack([np.asarray(px, float), np.asarray(py, float)])
    # El gap del borde de salida se mide ANTES de quitar el cierre duplicado:
    # un TE afilado que el fichero lista dos veces daria si no un gap igual al
    # espaciado de superficie, y se clasificaria como romo.
    cerrado = np.hypot(*(p[0] - p[-1])) < 1e-12
    if cerrado:
        p = p[:-1]

    # Contorno mojado en orden de recorrido: esquina intrados -> LE -> esquina
    # extrados. UN SOLO spline abierto: el morro queda interior, asi que sale
    # suave, y las unicas esquinas son los extremos, que son exactos por ser
    # extremos del spline. Partir tambien en el LE deja las dos mitades con
    # tangente libre e inventa un pico en una nariz redondeada.
    contorno = np.vstack([p[:1], p[::-1]]) if cerrado else p[::-1]
    esq_inf, esq_sup = contorno[0], contorno[-1]
    cuerda = max(p[:, 0].max() - p[:, 0].min(), 1e-12)
    gap = float(np.hypot(*(esq_sup - esq_inf)))
    romo = gap > tol_gap * cuerda

    denso, s = _densificar(contorno)
    L = s[-1]
    t_le = _arco_del_morro(denso[:, 0], s) / L

    h = _atractores(s / L, 1.0, [(0.0, razon_te, ancho_te),
                                 (1.0, razon_te, ancho_te),
                                 (t_le, razon_le, ancho_le)])
    s_obj = _distribucion(np.interp(np.linspace(0, 1, 4001), s / L, h), n_sup) * L
    perfil = np.column_stack([np.interp(s_obj, s, denso[:, 0]),
                              np.interp(s_obj, s, denso[:, 1])])
    ds_te = float(np.hypot(*(perfil[1] - perfil[0])))

    n_base = 0
    if romo:
        n_base = int(round(0.5 * gap / min(ds_te, dn_pared)))
        n_base = min(n_base, n_base_max)

    if n_base < 1:                              # base sub-celda: se afila
        media = 0.5 * (perfil[0] + perfil[-1])
        perfil[0] = perfil[-1] = media
        return perfil[:, 0], perfil[:, 1], dict(
            n_base=0, gap=gap, te_afilado=bool(romo), i_le=n_sup // 2, ds_te=ds_te)

    media = 0.5 * (perfil[0] + perfil[-1])
    w = np.linspace(0.0, 1.0, n_base + 1)[:-1, None]
    base_inf = media + w * (perfil[0] - media)  # medio -> esquina intrados (sin ella)
    base_sup = np.vstack([(perfil[-1] + w[1:] * (media - perfil[-1])), media])
    perfil = np.vstack([base_inf, perfil, base_sup])
    return perfil[:, 0], perfil[:, 1], dict(n_base=n_base, gap=gap, te_afilado=False,
                                            i_le=n_base + n_sup // 2, ds_te=ds_te)


def _razon_geometrica(ds0, L, n, q_max=CRECIMIENTO_ESTELA_MAX):
    """Razon q tal que ds0*(q^n - 1)/(q - 1) = L, por biseccion.

    Reescalar un estirado de razon fija para clavar L es justo lo que NO hay que
    hacer: con n=64 y q=1.25 el total se pasa por un factor 500 y el reescalado
    aplasta el primer espaciado a 3e-6, que es una celda degenerada pegada al TE.
    """
    if ds0 * n >= L:
        return 1.0                              # n de sobra: reparto casi uniforme
    total = lambda q: ds0 * n if abs(q - 1.0) < 1e-12 else ds0 * (q ** n - 1.0) / (q - 1.0)
    if total(q_max) < L:
        raise ValueError(
            f"estela: {n} puntos con crecimiento <= {q_max} no llegan a {L:.1f} "
            f"desde ds={ds0:.2e} (alcance {total(q_max):.2f}). Sube n_estela.")
    lo, hi = 1.0, q_max
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) < L:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _puntos_estela(x_te, x_out, n, ds_te, crecimiento_max=CRECIMIENTO_ESTELA_MAX):
    """Abscisas de la estela desde el TE (excluido) hasta x_out, estiradas."""
    L = x_out - x_te
    q = _razon_geometrica(ds_te, L, n, crecimiento_max)
    d = ds_te * q ** np.arange(n)
    x = x_te + np.cumsum(d)
    x *= 1.0                                    # q clava L salvo error de biseccion
    return x_te + (x - x_te) * L / (x[-1] - x_te)


def curva_inicial(px, py, n_sup=N_SUPERFICIE, n_estela=N_ESTELA, x_out=X_SALIDA,
                  razon_le=RAZON_LE, razon_te=RAZON_TE, dn_pared=DN_PARED):
    """Contorno interior completo de la malla C (j=0), como (N,2).

    Orden: corte inferior (x_out -> TE) + perfil (TE -> LE -> TE) + corte
    superior (TE -> x_out). Los dos cortes son espejo exacto uno del otro, asi
    que el emparejamiento `i <-> N-1-i` de la estela es exacto.
    """
    sx, sy, sinfo = redistribuir_superficie(px, py, n_sup, razon_le=razon_le,
                                            razon_te=razon_te, dn_pared=dn_pared)
    perfil = np.column_stack([sx, sy])          # ya cerrado: perfil[0] == perfil[-1]
    n_per = len(perfil)

    x_te, y_te = float(perfil[0, 0]), float(perfil[0, 1])
    xw = _puntos_estela(x_te, x_out, n_estela, sinfo["ds_te"])
    yw = np.full(n_estela, y_te)

    corte_inf = np.column_stack([xw[::-1], yw])
    corte_sup = np.column_stack([xw, yw])
    curva = np.vstack([corte_inf, perfil, corte_sup])

    rangos = {
        "estela_inf": (0, n_estela),
        "perfil": (n_estela, n_estela + n_per),
        "estela_sup": (n_estela + n_per, n_estela + n_per + n_estela),
        "i_le": n_estela + sinfo["i_le"],
        "gap_te": sinfo["gap"],
        "n_base": sinfo["n_base"],
        "te_afilado": sinfo["te_afilado"],
    }
    return curva, rangos


# ----------------------------------------------------------------------------
# Marcha hiperbolica (Steger-Chaussee: ortogonalidad + volumen, implicita en xi)
# ----------------------------------------------------------------------------
def _derivada_xi(r):
    """r_xi por diferencias centradas, unilaterales en los extremos."""
    d = np.empty_like(r)
    d[1:-1] = 0.5 * (r[2:] - r[:-2])
    d[0] = r[1] - r[0]
    d[-1] = r[-1] - r[-2]
    return d


def _banda_bloques(L, D, U):
    """Ensambla el sistema bloque-tridiagonal 2x2 en formato banda (kl=ku=3)."""
    N = D.shape[0]
    ab = np.zeros((7, 2 * N))
    idx = np.arange(N)
    for a in range(2):
        r = 2 * idx + a
        for b in range(2):
            m = idx >= 1
            c = 2 * (idx - 1) + b
            ab[3 + r[m] - c[m], c[m]] = L[m, a, b]

            c = 2 * idx + b
            ab[3 + r - c, c] = D[:, a, b]

            m = idx <= N - 2
            c = 2 * (idx + 1) + b
            ab[3 + r[m] - c[m], c[m]] = U[m, a, b]
    return ab


def _marchar(curva, pasos, x_out,
             mezcla_volumen=MEZCLA_VOLUMEN, disipacion=DISIPACION,
             disipacion_curva=DISIPACION_CURVA):
    """Marcha hiperbolica desde `curva` hacia el campo lejano.

    Impone, en cada nivel,
        f1 = r_xi . r_eta = 0                    (ortogonalidad)
        f2 = x_xi y_eta - x_eta y_xi - V = 0     (volumen de celda prescrito)

    Linealizacion de Newton en torno al nivel conocido:
        A r_xi + B r_eta = g,   g = A r_xi^j + B r_eta^j - f^j
    con A = [[x_eta, y_eta], [y_eta, -x_eta]], B = [[x_xi, y_xi], [-y_xi, x_xi]].
    Desarrollando, g = (r_xi.r_eta , volumen_alcanzado + V) evaluado en el nivel
    anterior. **Usar g = (0, V) a secas no vale**: el esquema queda inconsistente
    y avanza medio paso (se comprueba imponiendo r_eta = a*n, que da a = dn/2 en
    vez de a = dn).

    Se resuelve **para el incremento** D = r^{j+1} - r^j:
        D + C dxi(D) - eps dxi2(D) = B^-1 g - C r_xi^j,     C = B^-1 A
    centrado en xi => bloque-tridiagonal 2x2, resuelto en banda.

    La disipacion actua sobre el INCREMENTO, no sobre la posicion. Suavizar la
    posicion absoluta es lo que colapsa el borde de ataque: mete al perfil dentro
    de la primera capa, que es el modo de fallo del suavizado laplaciano ingenuo.

    Los extremos (i=0, i=N-1) estan sobre el plano de salida x = x_out y avanzan
    verticalmente: Dirichlet exacto sobre el incremento.
    """
    N = len(curva)
    n_capas = len(pasos) + 1
    niveles = np.empty((n_capas, N, 2))
    niveles[0] = curva

    r = curva.copy()
    r_eta = None
    ident = np.broadcast_to(np.eye(2), (N, 2, 2))
    unos = np.ones(N)
    for j in range(1, n_capas):
        dn = pasos[j - 1] * unos           # escalar o perfil dn(xi), da igual
        r_xi = _derivada_xi(r)
        xx, yy = r_xi[:, 0], r_xi[:, 1]
        s2 = np.maximum(xx * xx + yy * yy, 1e-30)
        s = np.sqrt(s2)

        if r_eta is None:                       # primer nivel: normal exacta
            r_eta = dn[:, None] * np.column_stack([-yy, xx]) / s[:, None]
        xe, ye = r_eta[:, 0], r_eta[:, 1]

        # volumen prescrito. Se mezcla hacia una version SUAVIZADA de la longitud
        # de arco local, no hacia la media global: la media esta dominada por la
        # estela y en el LE inflaria la celda por un factor ~60.
        s_v = s.copy()
        for _ in range(2 + int(6.0 * j / n_capas)):
            s_v[1:-1] = 0.5 * s_v[1:-1] + 0.25 * (s_v[:-2] + s_v[2:])
        V = dn * ((1.0 - mezcla_volumen) * s + mezcla_volumen * s_v)

        C = np.empty((N, 2, 2))                 # C = B^-1 A
        C[:, 0, 0] = (xx * xe - yy * ye) / s2
        C[:, 0, 1] = (xx * ye + yy * xe) / s2
        C[:, 1, 0] = C[:, 0, 1]
        C[:, 1, 1] = -C[:, 0, 0]

        g1 = xx * xe + yy * ye                  # ortogonalidad alcanzada
        g2 = xx * ye - xe * yy + V              # volumen alcanzado + objetivo
        binv_g = np.column_stack([xx * g1 - yy * g2, yy * g1 + xx * g2]) / s2[:, None]
        rhs = binv_g - np.einsum("nab,nb->na", C, r_xi)

        # disipacion: base, reforzada donde la curva es concava (curvatura > 0
        # con este sentido de recorrido) y donde el paso supera al espaciado local
        r_xx = np.empty_like(r)
        r_xx[1:-1] = r[2:] - 2.0 * r[1:-1] + r[:-2]
        r_xx[0] = r_xx[1]
        r_xx[-1] = r_xx[-2]
        kappa = (xx * r_xx[:, 1] - yy * r_xx[:, 0]) / np.maximum(s2 * s, 1e-30)
        # Solo la parte concava. Medido: reforzar tambien en las convexas
        # (|kappa|) empeora -- con dc=25 aparecen 41 celdas cruzadas en un caso
        # que con dc=8 salia limpio. La esquina convexa no se arregla con
        # amortiguacion.
        eps = disipacion * (1.0 + disipacion_curva * np.maximum(kappa, 0.0) * dn)
        eps = np.maximum(eps, disipacion * dn / s)

        L = (-0.5 * C - eps[:, None, None] * ident).copy()
        D = ((1.0 + 2.0 * eps)[:, None, None] * ident).copy()
        U = (0.5 * C - eps[:, None, None] * ident).copy()
        for k in (0, N - 1):                    # Dirichlet sobre el incremento
            L[k] = 0.0
            U[k] = 0.0
            D[k] = np.eye(2)
        rhs[0] = (0.0, -dn[0])
        rhs[-1] = (0.0, +dn[-1])

        delta = solve_banded((3, 3), _banda_bloques(L, D, U), rhs.ravel()).reshape(N, 2)
        nuevo = r + delta
        nuevo[0, 0] = x_out
        nuevo[-1, 0] = x_out

        r_eta = nuevo - r
        r = nuevo
        niveles[j] = r

    return niveles[:, :, 0], niveles[:, :, 1]


def plan_de_pasos(dn_pared=DN_PARED, crecimiento=CRECIMIENTO,
                  distancia=DISTANCIA_LEJOS, dn_max=DN_MAX,
                  n_capas_pared=N_CAPAS_PARED, crecimiento_pared=CRECIMIENTO_PARED,
                  tope_capas=4000):
    """Pasos normales de la marcha, de la pared al campo lejano.

    Tres tramos, cualquiera de ellos puede estar vacio:
      1. capa de pared: `n_capas_pared` pasos a `crecimiento_pared`
      2. estirado geometrico a `crecimiento`
      3. paso constante `dn_max`, si se alcanza el tope

    Se acumula en un bucle en vez de resolver la formula cerrada de cada tramo:
    son unos cientos de iteraciones y quita tres casos particulares.
    """
    pasos, dn, total = [], float(dn_pared), 0.0
    while total < distancia and len(pasos) < tope_capas:
        pasos.append(dn)
        total += dn
        q = crecimiento_pared if len(pasos) <= n_capas_pared else crecimiento
        dn = min(dn * q, dn_max)
    while len(pasos) < 8:                       # malla minima utilizable
        pasos.append(dn)
    return np.asarray(pasos)


def pasos_por_columna(ds, dn_pared=DN_PARED, crecimiento=CRECIMIENTO,
                      distancia=DISTANCIA_LEJOS, dn_max=DN_MAX,
                      n_capas_pared=N_CAPAS_PARED, crecimiento_pared=CRECIMIENTO_PARED,
                      aspecto_max=ASPECTO_MAX):
    """Pasos normales columna a columna: (n_capas-1, N). Ver ASPECTO_MAX.

        dn(xi, eta) = max( dn_ref(eta), ds(xi) / aspecto_max )

    y cada columna se reescala para que sume lo mismo que la de referencia. La
    columna arranca en su propio suelo y, en cuanto el crecimiento geometrico lo
    alcanza, sigue la distribucion de referencia: **el frente de marcha vuelve a
    ser comun en el campo lejano**, que es lo que evita el cizallamiento entre
    columnas vecinas. El reescalado es pequeno (10 % en la columna mas extrema).

    Se probo antes resolver una razon de crecimiento por columna por biseccion,
    con el mismo numero de capas. Es peor y esta medido: las columnas de estela
    avanzan mucho mas rapido desde la primera capa, el frente se descuelga y el
    campo lejano se apelotona junto al borde de salida -- la relacion de aspecto
    maxima pasaba de 9142 a 9747 en vez de bajar, y suavizar `dn0` a lo largo de
    xi lo empeoraba todavia mas (30 382 con 80 pasadas).
    """
    ref = plan_de_pasos(dn_pared, crecimiento, distancia, dn_max,
                        n_capas_pared, crecimiento_pared)
    dn0 = np.maximum(dn_pared, np.asarray(ds, float) / aspecto_max)
    pasos = np.maximum(ref[:, None], dn0[None, :])
    return pasos * (ref.sum() / pasos.sum(axis=0))[None, :]


def _radio_min_curva(c):
    """Radio de curvatura minimo de una polilinea abierta."""
    d1 = _derivada_xi(c)
    d2 = np.empty_like(c)
    d2[1:-1] = c[2:] - 2.0 * c[1:-1] + c[:-2]
    d2[0] = d2[1]
    d2[-1] = d2[-2]
    num = np.power(d1[:, 0] ** 2 + d1[:, 1] ** 2, 1.5)
    den = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    return float(np.min(np.where(den > 1e-30, num / np.maximum(den, 1e-30), np.inf)))


def generar_c(px, py, dn_pared=DN_PARED, crecimiento=CRECIMIENTO,
              distancia_lejos=DISTANCIA_LEJOS, n_sup=N_SUPERFICIE, n_estela=N_ESTELA,
              x_out=X_SALIDA, ajustar_dn_pared=AJUSTAR_DN_PARED, dn_max=DN_MAX,
              n_capas_pared=N_CAPAS_PARED, crecimiento_pared=CRECIMIENTO_PARED,
              aspecto_max=ASPECTO_MAX, **kw):
    """Perfil -> malla C. Devuelve (X, Y, info) con X, Y de forma (M, N).

    `ajustar_dn_pared = f > 0` recorta el paso de pared a `f * radio_del_morro`. El
    GA afila la nariz (radios medidos de hasta 2e-3 de cuerda) y una primera
    celda mas gruesa que el morro lo envuelve entero: la ortogonalidad en el
    punto de remanso cae a ~54 grados y no hay distribucion de superficie que lo
    arregle. Es una regla de mallado, no un defecto del generador.
    """
    razon_le = kw.pop("razon_le", RAZON_LE)
    razon_te = kw.pop("razon_te", RAZON_TE)
    curva, rangos = curva_inicial(px, py, n_sup=n_sup, n_estela=n_estela, x_out=x_out,
                                  razon_le=razon_le, razon_te=razon_te, dn_pared=dn_pared)
    if ajustar_dn_pared > 0.0:
        i0, i1 = rangos["perfil"]
        nuevo = min(dn_pared, ajustar_dn_pared * _radio_min_curva(curva[i0:i1]))
        if nuevo < dn_pared:                    # la base del TE depende del paso
            dn_pared = nuevo
            curva, rangos = curva_inicial(px, py, n_sup=n_sup, n_estela=n_estela,
                                          x_out=x_out, razon_le=razon_le,
                                          razon_te=razon_te, dn_pared=dn_pared)
    ds = np.hypot(*_derivada_xi(curva).T)
    pasos = pasos_por_columna(ds, dn_pared, crecimiento, distancia_lejos, dn_max,
                              n_capas_pared, crecimiento_pared, aspecto_max)
    X, Y = _marchar(curva, pasos, x_out, **kw)
    n_capas = len(pasos) + 1
    info = dict(rangos)
    info.update(n_capas=n_capas, dn_pared=dn_pared, crecimiento=crecimiento, x_out=x_out,
                dn_max=dn_max, n_capas_pared=n_capas_pared,
                crecimiento_pared=crecimiento_pared, aspecto_max=aspecto_max,
                forma=(n_capas, len(curva)))
    return X, Y, info


# ----------------------------------------------------------------------------
# Metricas (por vertices: conservacion geometrica exacta)
# ----------------------------------------------------------------------------
def metricas(X, Y):
    """Vectores de area de cara y volumen, desde los segmentos entre vertices.

    Cara xi de la celda (j,i-1)|(j,i): segmento vertice (j,i) -> (j+1,i).
    Cara eta de la celda (j-1,i)|(j,i): segmento vertice (j,i) -> (j,i+1).

    El telescopado sobre el cuadrilatero cerrado da div(u_inf) = 0 en maquina.
    """
    Sx_xi = Y[1:, :] - Y[:-1, :]                # (M-1, N)
    Sy_xi = -(X[1:, :] - X[:-1, :])
    Sx_eta = -(Y[:, 1:] - Y[:, :-1])            # (M, N-1)
    Sy_eta = X[:, 1:] - X[:, :-1]

    xv = np.stack([X[:-1, :-1], X[:-1, 1:], X[1:, 1:], X[1:, :-1]], axis=-1)
    yv = np.stack([Y[:-1, :-1], Y[:-1, 1:], Y[1:, 1:], Y[1:, :-1]], axis=-1)
    J = 0.5 * sum(xv[..., k] * yv[..., (k + 1) % 4] - xv[..., (k + 1) % 4] * yv[..., k]
                  for k in range(4))
    return dict(Sx_xi=Sx_xi, Sy_xi=Sy_xi, Sx_eta=Sx_eta, Sy_eta=Sy_eta, J=J)


def divergencia_uniforme(met, u0=1.0, v0=0.0):
    """Divergencia discreta de una corriente uniforme, y su escala local.

    Debe salir cero en maquina. Devuelve (div, escala) con escala = suma de los
    |flujos| de las cuatro caras: el residuo se normaliza con eso, no con el
    volumen de celda. Con celdas de pared de 1e-4 el volumen mediano se hunde e
    inflaria el cociente aunque la divergencia absoluta siga en 1e-16.
    """
    F_xi = u0 * met["Sx_xi"] + v0 * met["Sy_xi"]
    F_eta = u0 * met["Sx_eta"] + v0 * met["Sy_eta"]
    div = (F_xi[:, 1:] - F_xi[:, :-1]) + (F_eta[1:, :] - F_eta[:-1, :])
    escala = (np.abs(F_xi[:, 1:]) + np.abs(F_xi[:, :-1])
              + np.abs(F_eta[1:, :]) + np.abs(F_eta[:-1, :]))
    return div, escala


# ----------------------------------------------------------------------------
# Validacion geometrica
# ----------------------------------------------------------------------------
# Dos niveles, y la distincion importa: un bloqueante impide que el solver corra
# sobre la malla; un aviso es calidad mediocre en algun punto, que el solver
# aguanta. Mezclarlos da un veredicto binario que no dice nada util.
BLOQUEANTES = dict(
    j_negativos=0,
    ortogonalidad_pared_min=30.0,   # grados
    crecimiento_max=2.0,
    oblicuidad_p99_max=0.60,        # |g12| / sqrt(g11 g22)
    div_uinf_max=1e-12,             # conservacion geometrica
)

AVISOS = dict(
    ortogonalidad_pared_min=70.0,
    ortogonalidad_p5_min=45.0,
    crecimiento_max=1.30,
    # La relacion de aspecto alta junto a la pared es el OBJETIVO de la malla, no
    # un defecto: con y+~1 sale de 10000 sin que pase nada. El umbral solo caza
    # casos absurdos; la anisotropia la absorbe el suavizador por lineas en eta.
    aspecto_max=20000.0,
    oblicuidad_p99_max=0.30,
    dn_sobre_radio_le_max=0.5,      # el paso de pared debe caber en el morro
)


def _radio_min_pared(X, Y):
    """Radio de curvatura minimo de la linea j=0 (la pared).

    La parte de estela es recta (curvatura nula, radio infinito), asi que el
    minimo sobre toda la linea es el minimo sobre el perfil. Es el numero que
    fija cuanto se puede apretar el paso de pared: una primera celda mas gruesa
    que el radio del morro lo envuelve entero y la ortogonalidad se hunde ahi.
    """
    x, y = X[0], Y[0]
    dx = 0.5 * (np.roll(x, -1) - np.roll(x, 1))
    dy = 0.5 * (np.roll(y, -1) - np.roll(y, 1))
    ddx = np.roll(x, -1) - 2.0 * x + np.roll(x, 1)
    ddy = np.roll(y, -1) - 2.0 * y + np.roll(y, 1)
    num = np.power(dx * dx + dy * dy, 1.5)[1:-1]
    den = np.abs(dx * ddy - dy * ddx)[1:-1]
    r = np.where(den > 1e-30, num / np.maximum(den, 1e-30), np.inf)
    return float(np.min(r))


def calidad(X, Y, limites=None, perfil=None):
    """Metricas de calidad de la malla y veredicto binario.

    Sustituye a las 7 comprobaciones de `gui/vista_malla.py`, que eran
    patologias del IBM (pared mas fina que una celda, ghost cells que se
    interpenetran, ...). Estas son las que de verdad predicen si el solver va a
    converger sobre la malla.
    """
    i0, i1 = perfil["perfil"] if isinstance(perfil, dict) else (perfil or (0, X.shape[1] - 1))
    lim = dict(BLOQUEANTES)
    avi = dict(AVISOS)
    if limites:
        lim.update({k: v for k, v in limites.items() if k in lim})
        avi.update({k: v for k, v in limites.items() if k in avi})

    met = metricas(X, Y)
    J = met["J"]

    l_xi = np.hypot(X[:-1, 1:] - X[:-1, :-1], Y[:-1, 1:] - Y[:-1, :-1])
    l_eta = np.hypot(X[1:, :-1] - X[:-1, :-1], Y[1:, :-1] - Y[:-1, :-1])
    aspecto = np.maximum(l_xi, l_eta) / np.maximum(np.minimum(l_xi, l_eta), 1e-30)

    t_xi = np.stack([X[:-1, 1:] - X[:-1, :-1], Y[:-1, 1:] - Y[:-1, :-1]], -1)
    t_eta = np.stack([X[1:, :-1] - X[:-1, :-1], Y[1:, :-1] - Y[:-1, :-1]], -1)
    cos = np.abs((t_xi * t_eta).sum(-1)) / np.maximum(
        np.linalg.norm(t_xi, axis=-1) * np.linalg.norm(t_eta, axis=-1), 1e-30)
    ang = np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))

    crecimiento = l_eta[1:] / np.maximum(l_eta[:-1], 1e-30)

    # oblicuidad: cuanto pesan los terminos cruzados del laplaciano curvilineo.
    # Decide si la correccion diferida converge o hace falta stencil de 9 puntos.
    g11 = (met["Sx_xi"] ** 2 + met["Sy_xi"] ** 2)[:, :-1]
    g22 = (met["Sx_eta"] ** 2 + met["Sy_eta"] ** 2)[:-1, :]
    g12 = (met["Sx_xi"][:, :-1] * met["Sx_eta"][:-1, :]
           + met["Sy_xi"][:, :-1] * met["Sy_eta"][:-1, :])
    oblic = np.abs(g12) / np.maximum(np.sqrt(g11 * g22), 1e-30)

    radio_le = _radio_min_pared(X, Y)
    div = div_rel = 0.0
    for u0, v0 in ((1.0, 0.0), (float(np.cos(0.1)), float(np.sin(0.1)))):
        d, esc = divergencia_uniforme(met, u0, v0)
        div = max(div, float(np.abs(d).max()))
        div_rel = max(div_rel, float((np.abs(d) / np.maximum(esc, 1e-300)).max()))

    r = dict(
        n_celdas=int(J.size),
        j_min=float(J.min()),
        j_negativos=int((J <= 0.0).sum()),
        ortogonalidad_pared_min=float(ang[0].min()),
        ortogonalidad_p5=float(np.percentile(ang, 5)),
        ortogonalidad_p50=float(np.percentile(ang, 50)),
        crecimiento_max=float(crecimiento.max()) if crecimiento.size else 1.0,
        aspecto_p90=float(np.percentile(aspecto, 90)),
        aspecto_max=float(aspecto.max()),
        oblicuidad_p99=float(np.percentile(oblic, 99)),
        oblicuidad_max=float(oblic.max()),
        dn_pared_min=float(l_eta[0, i0:i1].min()),
        dn_pared_max=float(l_eta[0, i0:i1].max()),
        dn_max=float(l_eta.max()),
        ds_max=float(l_xi.max()),
        radio_le=float(radio_le),
        dn_sobre_radio_le=float(l_eta[0, i0:i1].max() / max(radio_le, 1e-30)),
        div_uinf=float(div),
        div_uinf_rel=float(div_rel),
    )

    fallos = []
    if r["j_negativos"] > lim["j_negativos"]:
        fallos.append(f"jacobiano<=0 en {r['j_negativos']} celdas")
    if r["ortogonalidad_pared_min"] < lim["ortogonalidad_pared_min"]:
        fallos.append(f"ortogonalidad en pared {r['ortogonalidad_pared_min']:.1f} deg")
    if r["crecimiento_max"] > lim["crecimiento_max"]:
        fallos.append(f"crecimiento normal {r['crecimiento_max']:.2f}")
    if r["oblicuidad_p99"] > lim["oblicuidad_p99_max"]:
        fallos.append(f"oblicuidad p99 {r['oblicuidad_p99']:.2f}")
    if r["div_uinf_rel"] > lim["div_uinf_max"]:
        fallos.append(f"div(u_inf) relativa {r['div_uinf_rel']:.1e}")

    avisos = []
    if r["ortogonalidad_pared_min"] < avi["ortogonalidad_pared_min"]:
        avisos.append(f"ortogonalidad en pared {r['ortogonalidad_pared_min']:.1f} deg")
    if r["ortogonalidad_p5"] < avi["ortogonalidad_p5_min"]:
        avisos.append(f"ortogonalidad p5 {r['ortogonalidad_p5']:.1f} deg")
    if r["crecimiento_max"] > avi["crecimiento_max"]:
        avisos.append(f"crecimiento normal {r['crecimiento_max']:.2f}")
    if r["aspecto_max"] > avi["aspecto_max"]:
        avisos.append(f"aspecto {r['aspecto_max']:.0f}")
    if r["oblicuidad_p99"] > avi["oblicuidad_p99_max"]:
        avisos.append(f"oblicuidad p99 {r['oblicuidad_p99']:.2f}")
    if r["dn_sobre_radio_le"] > avi["dn_sobre_radio_le_max"]:
        avisos.append(f"paso de pared {r['dn_sobre_radio_le']:.2f}x el radio del morro")

    r["fallos"] = fallos
    r["avisos"] = avisos
    r["valida"] = not fallos
    r["buena"] = not fallos and not avisos
    return r


# ----------------------------------------------------------------------------
# CLI: barrido de validacion sobre el historial del GA
# ----------------------------------------------------------------------------
def dibujar(X, Y, ruta, rangos=None, titulo="", paso_i=1, paso_j=1, parametros=None):
    """Guarda una figura de la malla: dominio, perfil, LE, TE y corte de estela.

    `parametros` es un dict que se escribe al pie de la figura, para que la
    imagen sea autosuficiente: quien la mira sabe con que malla se corrio sin
    abrir ningun json.

    Las celdas con jacobiano <= 0 se marcan en rojo, que es lo que hay que ver
    cuando el validador rechaza una geometria. matplotlib se importa aqui dentro
    a proposito: el barrido sobre el historial no lo necesita.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    M_, N_ = X.shape
    J = metricas(X, Y)["J"]
    malas = np.argwhere(J <= 0.0)
    cx = 0.25 * (X[:-1, :-1] + X[:-1, 1:] + X[1:, 1:] + X[1:, :-1])
    cy = 0.25 * (Y[:-1, :-1] + Y[:-1, 1:] + Y[1:, 1:] + Y[1:, :-1])

    if rangos is not None:
        i0, i1 = rangos["perfil"]
        px, py = X[0, i0:i1], Y[0, i0:i1]
    else:
        px = py = None

    x_te = float(X[0, rangos["perfil"][0]]) if rangos else 1.0
    y_te = float(Y[0, rangos["perfil"][0]]) if rangos else 0.0
    vistas = [
        ("dominio completo", (float(X.min()), float(X.max()),
                              float(Y.min()), float(Y.max())), 4, 1),
        ("perfil y estela", (-1.2, 3.5, -1.4, 1.4), 1, 1),
        ("borde de ataque", (-0.02, 0.06, -0.04, 0.04), 1, 1),
        ("borde de salida", (x_te - 0.05, x_te + 0.05, y_te - 0.04, y_te + 0.04), 1, 1),
        ("corte de estela", (x_te - 0.02, x_te + 0.6, y_te - 0.15, y_te + 0.15), 1, 1),
        ("campo lejano aguas arriba", (float(X.min()), 1.5,
                                       -0.45 * float(Y.max()), 0.45 * float(Y.max())), 2, 1),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    for ax, (tit, w, pi, pj) in zip(axes.ravel(), vistas):
        segs = [np.column_stack([X[j], Y[j]]) for j in range(0, M_, pj * paso_j)]
        segs += [np.column_stack([X[:, i], Y[:, i]]) for i in range(0, N_, pi * paso_i)]
        ax.add_collection(LineCollection(segs, colors="0.55", lw=0.35))
        if px is not None:
            ax.plot(px, py, "-", color="crimson", lw=1.4, zorder=4)
        if len(malas):
            ax.plot(cx[malas[:, 0], malas[:, 1]], cy[malas[:, 0], malas[:, 1]],
                    "o", color="red", ms=3, zorder=5)
        ax.set_xlim(w[0], w[1])
        ax.set_ylim(w[2], w[3])
        ax.set_aspect("equal")
        ax.set_title(tit, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    aviso = f"  --  {len(malas)} celdas con jacobiano <= 0 (en rojo)" if len(malas) else ""
    fig.suptitle(f"{titulo}{aviso}", fontsize=12)

    pie = dict(parametros or {})
    pie.setdefault("malla", f"{M_}x{N_} = {(M_ - 1) * (N_ - 1)} celdas")
    pie.setdefault("dominio", f"Lx={X.max() - X.min():.2f}c  Ly={Y.max() - Y.min():.2f}c  "
                              f"x in [{X.min():.2f}, {X.max():.2f}]  |y| <= {Y.max():.2f}c")
    # Ancho en caracteres de monospace de 9 pt sobre el ancho real de la figura.
    ancho = int(fig.get_size_inches()[0] * 72 / (9 * 0.60))
    filas, fila = [], ""
    for k, v in pie.items():
        t = f"{k} = {v}"
        if fila and len(fila) + len(t) + 6 > ancho:
            filas.append(fila); fila = ""
        fila = f"{fila}      {t}" if fila else t
    filas.append(fila)
    alto = 0.023 * len(filas)
    fig.text(0.5, 0.006, "\n".join(filas),
             ha="center", va="bottom", fontsize=9, family="monospace")
    fig.tight_layout(rect=[0, alto + 0.01, 1, 0.96])
    fig.savefig(ruta, dpi=135)
    plt.close(fig)
    return ruta


def _resumen(nombre, X, Y, q, ancho=28):
    est = "BUENA" if q["buena"] else ("valida" if q["valida"] else "NO   ")
    print(f"{est} {nombre:<{ancho}} celdas={q['n_celdas']:6d}  ort_pared={q['ortogonalidad_pared_min']:5.1f}  "
          f"ort_p5={q['ortogonalidad_p5']:5.1f}  crec={q['crecimiento_max']:5.3f}  "
          f"oblic99={q['oblicuidad_p99']:.3f}  dn/R_le={q['dn_sobre_radio_le']:5.2f}  "
          f"aspecto={q['aspecto_max']:6.0f}  div={q['div_uinf_rel']:.1e}")
    if q["fallos"]:
        print(f"    BLOQUEA: {'; '.join(q['fallos'])}")
    if q["avisos"]:
        print(f"    avisos:  {'; '.join(q['avisos'])}")


def _main(argv=None):
    import argparse, json, time
    from collections import Counter

    ap = argparse.ArgumentParser(description="Genera y valida mallas C.")
    ap.add_argument("--perfil", nargs="*", help="ficheros .dat a mallar")
    ap.add_argument("--validar-historial", metavar="JSONL",
                    help="barre los perfiles de aprendizaje_ML.jsonl")
    ap.add_argument("--limite", type=int, default=0, help="maximo de registros del historial")
    # Los defaults son las constantes de la cabecera del modulo: pasar la opcion
    # cambia solo esta llamada, editar la constante cambia todo el proyecto.
    ap.add_argument("--dn-pared", type=float, default=DN_PARED,
                    help=f"tamano minimo de celda junto al perfil (defecto {DN_PARED:g})")
    ap.add_argument("--crecimiento", type=float, default=CRECIMIENTO,
                    help=f"razon de crecimiento normal por capa (defecto {CRECIMIENTO})")
    ap.add_argument("--distancia-lejos", type=float, default=DISTANCIA_LEJOS,
                    help=f"alcance de la malla en cuerdas (defecto {DISTANCIA_LEJOS})")
    ap.add_argument("--n-sup", type=int, default=N_SUPERFICIE,
                    help=f"puntos sobre el perfil (defecto {N_SUPERFICIE})")
    ap.add_argument("--n-estela", type=int, default=N_ESTELA,
                    help=f"puntos por lado del corte de estela (defecto {N_ESTELA})")
    ap.add_argument("--x-salida", type=float, default=X_SALIDA,
                    help=f"plano de salida aguas abajo (defecto {X_SALIDA})")
    ap.add_argument("--razon-le", type=float, default=RAZON_LE,
                    help=f"agrupamiento en el borde de ataque (defecto {RAZON_LE})")
    ap.add_argument("--razon-te", type=float, default=RAZON_TE,
                    help=f"agrupamiento en el borde de salida (defecto {RAZON_TE})")
    ap.add_argument("--dn-max", type=float, default=DN_MAX,
                    help="tope del espaciado normal (defecto: sin tope)")
    ap.add_argument("--n-capas-pared", type=int, default=N_CAPAS_PARED,
                    help=f"capas con crecimiento reducido junto a la pared (defecto {N_CAPAS_PARED})")
    ap.add_argument("--crecimiento-pared", type=float, default=CRECIMIENTO_PARED,
                    help=f"crecimiento dentro de la capa de pared (defecto {CRECIMIENTO_PARED})")
    ap.add_argument("--aspecto-max", type=float, default=ASPECTO_MAX,
                    help=f"tope de relacion de aspecto de la primera capa (defecto {ASPECTO_MAX:g})")
    ap.add_argument("--ajustar-dn-pared", type=float, default=AJUSTAR_DN_PARED,
                    help="recorta el paso de pared a f*radio_del_morro (0 = fijo)")
    ap.add_argument("--salida-json", help="vuelca el detalle del barrido")
    ap.add_argument("--figura", metavar="DIR", nargs="?", const=".",
                    help="guarda una figura por perfil en DIR (por defecto, el directorio actual)")
    a = ap.parse_args(argv)

    kw = dict(dn_pared=a.dn_pared, crecimiento=a.crecimiento,
              distancia_lejos=a.distancia_lejos, n_sup=a.n_sup, n_estela=a.n_estela,
              x_out=a.x_salida, razon_le=a.razon_le, razon_te=a.razon_te,
              ajustar_dn_pared=a.ajustar_dn_pared, dn_max=a.dn_max,
              n_capas_pared=a.n_capas_pared, crecimiento_pared=a.crecimiento_pared,
              aspecto_max=a.aspecto_max)

    for ruta in (a.perfil or []):
        px, py = leer_dat(ruta)
        X, Y, info = generar_c(px, py, **kw)
        q = calidad(X, Y, perfil=info)
        nombre = ruta.split("/")[-1]
        _resumen(nombre, X, Y, q)
        if a.figura:
            base = nombre.rsplit(".", 1)[0]
            destino = os.path.join(a.figura, f"malla_{base}.png")
            estado = "BUENA" if q["buena"] else ("valida" if q["valida"] else "NO VALIDA")
            dibujar(X, Y, destino, rangos=info, titulo=f"{nombre} -- {estado}",
                    parametros={
                        "dn_pared": "%.2e c  (%d capas sin crecimiento, razon %.2f)"
                                    % (info["dn_pared"], info["n_capas_pared"],
                                       info["crecimiento_pared"]),
                        "crecimiento": "%.3f normal  |  dn_max %s  |  aspecto_max %g"
                                       % (info["crecimiento"], info["dn_max"],
                                          info["aspecto_max"]),
                        "superficie": "%d puntos de perfil + %d por lado de estela  |  "
                                      "x_salida %.2f c" % (a.n_sup, a.n_estela, info["x_out"]),
                        "calidad": "ort_pared %.1f deg  |  oblic_p99 %.4f  |  "
                                   "aspecto_real %.0f  |  crec_max %.3f  |  J<=0: %d"
                                   % (q["ortogonalidad_pared_min"], q["oblicuidad_p99"],
                                      q["aspecto_max"], q["crecimiento_max"],
                                      int(q["j_negativos"]))})
            print(f"    figura: {destino}")

    if a.validar_historial:
        t0 = time.time()
        filas, motivos = [], Counter()
        with open(a.validar_historial, "r", encoding="utf-8") as f:
            for n, linea in enumerate(f):
                if a.limite and n >= a.limite:
                    break
                d = json.loads(linea)
                p = np.asarray(d["perfil"], dtype=np.float64)
                meta = d.get("metadata", {})
                par = meta.get("parametrizacion", {})
                try:
                    X, Y, info = generar_c(p[:, 0], p[:, 1], **kw)
                    q = calidad(X, Y, perfil=info)
                except Exception as exc:                       # malla imposible
                    q = dict(valida=False, buena=False, avisos=[],
                             fallos=[f"excepcion: {type(exc).__name__}: {exc}"])
                for m in q["fallos"]:
                    motivos["BLOQUEA " + m.split(" ")[0]] += 1
                for m in q["avisos"]:
                    motivos["aviso " + m.split(" ")[0]] += 1
                filas.append(dict(id=d.get("id"), valida=q["valida"], buena=q["buena"],
                                  fallos=q["fallos"], avisos=q["avisos"],
                                  espesor=par.get("espesor_max"), espesor_pos=par.get("espesor_pos"),
                                  camber=par.get("camber_max"), te_gap=par.get("te_gap"),
                                  **{k: v for k, v in q.items()
                                     if k not in ("fallos", "avisos", "valida", "buena")}))
        n = len(filas)
        ok = sum(1 for r in filas if r["valida"])
        bien = sum(1 for r in filas if r.get("buena"))
        print(f"\n{n} perfiles del historial en {time.time()-t0:.1f} s")
        print(f"malla utilizable (sin bloqueantes): {ok}/{n} = {100.0*ok/max(n,1):.1f} %")
        print(f"malla buena (sin avisos):           {bien}/{n} = {100.0*bien/max(n,1):.1f} %")
        if motivos:
            print("motivos de rechazo (un perfil puede acumular varios):")
            for m, c in motivos.most_common():
                print(f"  {c:5d}  {m}")
        malos = [r for r in filas if not r["valida"] and r.get("espesor") is not None]
        if malos:
            e = np.array([r["espesor"] for r in malos])
            buenos = [r for r in filas if r["valida"] and r.get("espesor") is not None]
            eb = np.array([r["espesor"] for r in buenos]) if buenos else np.array([np.nan])
            print(f"espesor max: rechazados p50={np.median(e):.4f}  aceptados p50={np.median(eb):.4f}")
        if a.salida_json:
            with open(a.salida_json, "w", encoding="utf-8") as f:
                json.dump(filas, f)
            print("detalle en", a.salida_json)


if __name__ == "__main__":
    _main()

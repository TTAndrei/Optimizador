"""
Optimizaciones del solver de presión, aisladas del simulador.

Simulador2D.py sustituye líneas puntuales por llamadas a este módulo. Cada
función contiene DOS ramas: la rama "off" reproduce exactamente el código que
había antes, y la rama "on" aplica la optimización. Con todas las opciones
desactivadas —que es el estado por defecto— el simulador se comporta bit a bit
como antes de introducir este módulo, de forma que la infraestructura se puede
validar sin cambiar ningún resultado.

Activación:
    import opt_solver
    opt_solver.enable("warm_start")

o por entorno, para no tocar los scripts de campaña:
    OPT_SOLVER=warm_start .venv/bin/python scripts/...

Cada opción está documentada con lo que hace y con lo que puede romper.
"""
from __future__ import annotations

import os

import cupy as cp


# ----------------------------------------------------------------------
# Registro de opciones
# ----------------------------------------------------------------------
# warm_start:
#   El outer 0 de la proyección resuelve L·p = (rho/dt)·div(u*), cuya solución es
#   la presión física del paso. Entre dos pasos consecutivos ese campo cambia
#   poco, así que arrancar el multigrid desde la presión del paso anterior en vez
#   de desde cero reduce los ciclos necesarios. No cambia la ecuación resuelta ni
#   la tolerancia exigida: sólo el punto de partida del iterativo.
#   Riesgo: si la semilla está obsoleta (cambio de alpha, rollback por NaN,
#   cambio de malla) es una mala conjetura y puede costar ciclos de más. Por eso
#   la semilla se invalida explícitamente en esos casos (ver invalidate_warm).
#
# coarse_mask_majority:
#   La jerarquia multigrid engorda el solido en cada nivel: _coarsen_mask hace OR
#   sobre un bloque 3x3, asi que basta una celda solida fina para que la gruesa
#   lo sea. Con un perfil de borde de salida afilado, dos niveles de coarsening
#   convierten un TE de una celda de espesor en un cuerpo romo, y el operador
#   grueso deja de parecerse al fino. La literatura de multigrid con fronteras
#   inmersas describe justo esto: el coarsening pierde la frontera cuando hay
#   estructuras finas, y el cambio topologico entre niveles introduce modos
#   espurios que frenan la convergencia o la rompen.
#   Con esta opcion la celda gruesa es solida solo si la MAYORIA del bloque fino
#   lo es. Las estructuras finas desaparecen del nivel grueso en vez de
#   engordarlo. Desaparecer es preferible: el nivel grueso solo tiene que
#   corregir error de baja frecuencia, y un Poisson sin el cuerpo sigue siendo un
#   buen precondicionador, mientras que un cuerpo deformado no lo es.
#   No toca el nivel 0: el operador, el residuo y la tolerancia que se exigen
#   siguen siendo exactamente los de antes. Solo cambia el precondicionador.
#
# coarse_mask_none:
#   Version extrema de lo anterior: ningun solido por debajo del nivel 0. El
#   nivel grueso resuelve el Poisson del dominio entero. Sirve como cota: si esto
#   converge mejor que la jerarquia actual, queda demostrado que la mascara
#   gruesa esta restando en vez de sumar.
#
# line_smoother:
#   Sustituye el suavizador Gauss-Seidel punto a punto por relajacion por lineas
#   alternada (una pasada de lineas en Y, otra en X), resolviendo cada linea con
#   Thomas.
#   Motivo, medido sobre la malla de produccion (24x16, dx_min=0.002,
#   factor_expansion=1.1, dx_max=0.1): el 30% de las celdas tiene relacion de
#   aspecto > 8 y el 25% la tiene > 32. Lo que degrada al suavizador no es esa
#   relacion sino su cuadrado, porque el acoplamiento del laplaciano de 5 puntos
#   va como 1/h^2: la anisotropia efectiva del operador llega a 2500. Un
#   Gauss-Seidel punto a punto tiene factor de suavizado ~1 en la direccion
#   debil, o sea que en un tercio del dominio no suaviza. Sin suavizado no hay
#   multigrid: los niveles gruesos no pueden corregir un error que el suavizador
#   no ha hecho suave, y de ahi que 40 V-cycles reduzcan la divergencia solo
#   x1.75.
#   La relajacion por lineas resuelve exactamente la direccion de acoplamiento
#   fuerte, que es la receta estandar para anisotropia. Se alternan las dos
#   direcciones porque aqui la anisotropia cambia de sentido segun la zona:
#   en la estela manda Y (dx grande, dy pequeno) y sobre el perfil manda X.
#   Es un cambio de precondicionador: el operador del nivel 0, el residuo y la
#   tolerancia exigida no se tocan.
#
# line_smoother_y:
#   Solo lineas en Y, sin la pasada en X. Medido sobre la malla de produccion, un
#   barrido completo Y+X cuesta 49 veces un barrido rojo-negro, y el 69% de ese
#   coste son las lineas X: el hilo recorre una fila mientras los hilos vecinos
#   trabajan filas separadas nx en memoria, asi que ningun acceso coalesce. Las
#   lineas Y si coalescen y cuestan 15 veces el rojo-negro.
#   La zona anisotropa mas extensa es la estela (dx grande, dy pequeno), donde el
#   acoplamiento fuerte es en Y, asi que las lineas Y solas cubren la mayor parte
#   del dominio degradado a un tercio del precio. Queda sin cubrir la banda por
#   encima y debajo del perfil, donde manda X.
#
# deep_levels:
#   El V-cycle tiene el numero de niveles topado a fuego en `nlvl_use =
#   min(n_levels, 2)`, con el comentario de que con malla 20:1 los niveles
#   profundos son inconsistentes. Ese tope es una consecuencia del problema de
#   coarsening, no una ley: la mascara dilatada por OR deforma el solido y la
#   anisotropia rompe el suavizador punto a punto, y con las dos cosas rotas los
#   niveles profundos solo podian empeorar. Corregidas (coarse_mask_majority y
#   line_smoother), el tope pasa a ser lo que limita al multigrid: con 2 niveles
#   el nivel mas grueso de una malla de 1.26M celdas sigue teniendo 78k, y los
#   modos de longitud de onda larga no se corrigen en ningun sitio.
#   Requiere subir tambien mg_niveles_max, que es lo que fija cuantos niveles se
#   construyen; esta opcion solo levanta el tope de cuantos se usan.
#
# interp_float32:
#   La interpolacion bilineal calcula los pesos como `wx = x - j0`, con x float32
#   y j0 int32. La regla de promocion de NumPy y CuPy dice que float32 - int32 da
#   float64, porque un int32 no cabe exacto en un float32 y el tipo comun
#   "seguro" es el doble. A partir de ahi toda la formula bilineal —los cuatro
#   productos y la suma— se evalua en float64.
#   Medido con Nsight sobre el bucle: el 26% del tiempo de GPU son operaciones
#   float64, y sus recuentos por iteracion cuadran uno a uno con esta formula
#   (35 restas float32-int32, 72 restas de 1-w, 102 multiplicaciones, 71 sumas).
#   El coste no compra nada en el camino de las velocidades: el resultado se
#   asigna dentro de arrays float32 y el doble se trunca acto seguido.
#   En nu_tilde si cambia algo, y no para bien: `advect_sa` hace
#   `self.nu_tilde = nt`, o sea rebindea, y el campo se queda en float64 para el
#   resto de la simulacion. Con eso el modelo Spalart-Allmaras entero corre en
#   doble precision, incluidos g**6 y **(1/6), que en una GeForce van a 1/64 de
#   la velocidad de float32: 702 us por instancia, tres por iteracion.
#   Nada de esto es una decision de diseno; es una promocion accidental.
#   OJO: activarlo cambia los resultados del modelo SA, que hasta ahora corria en
#   float64 sin que nadie lo pidiera. En las velocidades no cambia nada
#   apreciable porque ya se truncaba.
#
# fast_masks:
#   Sustituye las asignaciones con mascara booleana del solver por
#   cp.copyto(dst, src, where=mascara).
#   Son semanticamente lo mismo, pero cuestan cosas muy distintas. Escribir
#   `p[mascara] = v` obliga a CuPy a convertir la mascara en indices: un prefix
#   sum sobre todo el array, un gather y un scatter. El perfil con Nsight lo
#   ensena en crudo: prepare_array_indexing, scan_naive, take, bsum_shfl y
#   scatter_update_mask suman el 23% del tiempo de GPU del bucle, con 592 scans
#   por iteracion. cp.copyto(where=) es un solo kernel elementwise.
#   Medido sobre la malla de produccion: 6.1x mas rapido en la mascara 2D y 5.9x
#   en las de borde. Y el coste no lo marcan los datos sino la maquinaria: 142 us
#   para una mascara de 713 elementos.
#   Las mascaras de borde de _aplicar_bc_mg son ademas CONSTANTES (dependen de la
#   geometria, no del campo) y se recalculaban en cada llamada; aqui se cachean
#   por nivel.
#   No cambia ningun resultado: es la misma operacion escrita de otra forma.
# Estado por defecto. Las cuatro primeras estan ACTIVAS: se validaron a t=20
# sobre la polar completa del perfil ganador (results/polar_optimizada), con el
# mismo dominio, malla, presupuesto de iteraciones y ruta de codigo que el
# estudio publicado. Las otras cuatro estan apagadas porque fallaron o empeoran;
# se conservan porque el motivo del fallo esta medido y documentado, y volver a
# probarlas sin ese contexto seria repetir el trabajo.
OPTS = {
    "fast_masks": True,            # bit a bit identico, +32%
    "interp_float32": True,        # +0.012% en Cl a t=20, +17%
    "warm_start": True,            # mejor convergido que produccion, x1.88
    "coarse_mask_majority": True,  # mejor factor multigrid, x1.14, coste cero

    "deep_levels": False,          # EMPEORA: divergencia +62%, 0.34x
    "coarse_mask_none": False,     # solo como cota experimental
    "line_smoother": False,        # converge bien pero cuesta 5.1x por ciclo
    "line_smoother_y": False,      # diverge: no suaviza la direccion X

    # Reescala la semilla de warm_start por el cambio de dt. Requiere
    # warm_start activo; sin el no hace nada. Ver init_guess.
    "warm_start_scaled": False,
    # Filtra el modo de tablero de la semilla antes de reusarla. Requiere
    # warm_start activo. Ver _filtra_tablero.
    "warm_start_filtered": False,
}

# Vuelta atras completa. Necesaria para reproducir resultados anteriores a esta
# optimizacion: con OPT_SOLVER_OFF=1 el simulador se comporta exactamente como
# antes, lo cual esta comprobado bit a bit (variante `equiv` del banco).
if os.environ.get("OPT_SOLVER_OFF", "").strip() not in ("", "0", "false", "False"):
    for _k in OPTS:
        OPTS[_k] = False


def enable(*names):
    for n in names:
        if n not in OPTS:
            raise KeyError(f"opción desconocida: {n} (válidas: {sorted(OPTS)})")
        OPTS[n] = True


def disable(*names):
    for n in names:
        if n not in OPTS:
            raise KeyError(f"opción desconocida: {n} (válidas: {sorted(OPTS)})")
        OPTS[n] = False


def reset():
    for n in OPTS:
        OPTS[n] = False


def active():
    return sorted(n for n, v in OPTS.items() if v)


def _load_env():
    """OPT_SOLVER fija la lista EXACTA de opciones activas.

    Sustituye al estado por defecto en vez de anadirse a el: ahora que las
    opciones vienen activadas, una variable que solo supiera encender no
    permitiria pedir un subconjunto. `OPT_SOLVER=warm_start` deja warm_start y
    solo warm_start; `OPT_SOLVER=` (vacia) no toca nada; para apagarlo todo
    esta OPT_SOLVER_OFF=1.
    """
    spec = os.environ.get("OPT_SOLVER", "").strip()
    if not spec:
        return
    nombres = [n.strip() for n in spec.split(",") if n.strip()]
    for n in nombres:
        if n not in OPTS:
            raise KeyError(f"opción desconocida en OPT_SOLVER: {n} "
                           f"(válidas: {sorted(OPTS)})")
    for k in OPTS:
        OPTS[k] = k in nombres


_load_env()


# ----------------------------------------------------------------------
# interp_float32
# ----------------------------------------------------------------------
def bilinear_weights(x, y, j0, i0):
    """Pesos fraccionarios de la interpolación bilineal.

    Sustituye a:
        wx = x - j0
        wy = y - i0
    """
    if not OPTS["interp_float32"]:
        return x - j0, y - i0
    # Castear el índice entero al tipo del campo evita que la resta promocione.
    return x - j0.astype(x.dtype), y - i0.astype(y.dtype)


# ----------------------------------------------------------------------
# deep_levels
# ----------------------------------------------------------------------
def mg_depth(n_levels, tope=2):
    """Niveles que usa el V-cycle. Sustituye a `min(n_levels, tope)`."""
    return int(n_levels) if OPTS["deep_levels"] else min(int(n_levels), tope)


def coarse_iters(por_defecto, nx_c, ny_c):
    """Barridos en el nivel más grueso del V-cycle.

    Sustituye a la constante `coarse_solve_iters = 20`.

    Con 2 niveles el nivel más grueso de esta malla tiene ~78 k celdas y 20
    barridos son lo que son: una corrección parcial, que es lo que el código
    pretendía ("pocas: sólo corregir modos bajos, no resolver exactamente").
    Al bajar más niveles ese nivel se hace pequeño y entonces sí conviene
    resolverlo casi exacto, porque es justo el sitio donde se eliminan los modos
    de longitud de onda larga: si se deja a medias, bajar niveles no sirve de
    nada y el experimento de profundidad daría un falso negativo por una
    constante mal escalada, no por el método.

    El tope importa. Un barrido del nivel fino cuesta ~82 us en esta malla y un
    V-cycle gasta seis, o sea ~490 us: si el nivel más grueso se lleva más que
    eso, la profundidad sale cara aunque converja mejor. Con 100 barridos, una
    jerarquía de 6 niveles (nivel más grueso ~300 celdas, limitado por
    lanzamiento) gasta ~0.2 ms por ciclo, que es asumible; una de 4 niveles
    (~20 k celdas) gasta bastante más, y eso mismo es parte de lo que se mide:
    con 4 niveles el nivel grueso todavía es grande y bajar hasta él no compensa.
    """
    if not OPTS["deep_levels"]:
        return por_defecto
    n = max(int(nx_c), int(ny_c))
    return int(min(max(por_defecto, 2 * n), 100))


# ----------------------------------------------------------------------
# warm_start
# ----------------------------------------------------------------------
_WARM_ATTR = "_opt_p_warm"
_WARM_DT_ATTR = "_opt_p_warm_dt"

# Tope del reescalado. Un dt que se desploma de golpe daria un factor enorme y la
# semilla pasaria de estar mal escalada a ser una perturbacion gigante, que es
# peor que empezar de cero. Fuera de este rango se descarta la semilla.
_RATIO_MAX = 4.0
_FILT_ATTR = "_opt_p_warm_filt"


def _filtra_tablero(mesh, p):
    """Un paso de Jacobi ponderado con omega=1/2 sobre la semilla.

    POR QUE. La malla es colocalizada, asi que el operador de presion admite el
    modo par-impar (tablero): L lo ve casi nulo y el suavizador de Gauss-Seidel
    no lo reduce. Ya estaba medido que el 46.2% de la divergencia residual esta
    en ese modo. Al reusar la presion del paso anterior, la semilla arrastra el
    tablero acumulado, y con presupuesto corto no hay ciclos para disiparlo: se
    realimenta paso a paso.

    Este filtro lo elimina de golpe. Para el modo de tablero los cuatro vecinos
    tienen signo opuesto, luego su media vale -p y el paso da

        p <- p/2 + (-p)/2 = 0            (aniquilacion exacta)

    mientras que para un modo suave la media vale ~p y lo deja como estaba:

        p <- p/2 + p/2 = p

    O sea que separa justo las dos cosas que hay que separar. Cuesta una
    plantilla de 5 puntos por paso de tiempo, frente a los seis ciclos V de la
    proyeccion: es ruido en el coste.

    La media de 4 vecinos ignora el estirado de la malla. Es deliberado: esto
    filtra, no resuelve, y el modo que se quiere matar es el de frecuencia mas
    alta en indices de celda, no en distancia fisica.
    """
    buf = getattr(mesh, _FILT_ATTR, None)
    if buf is None or buf.shape != p.shape:
        buf = cp.empty_like(p)
        setattr(mesh, _FILT_ATTR, buf)
    buf[:] = p
    buf[1:-1, 1:-1] = cp.float32(0.5) * p[1:-1, 1:-1] + cp.float32(0.125) * (
        p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:])
    p[:] = buf
    solid = getattr(mesh, "solid", None)
    if solid is not None and solid.shape == p.shape:
        cero_en_solido(p, solid)


def invalidate_warm(mesh):
    """Descarta la semilla de presión. Obligatorio tras un cambio de ángulo de
    ataque, un rollback por NaN o cualquier reconstrucción de la malla: el campo
    guardado deja de ser una conjetura razonable del siguiente."""
    for _a in (_WARM_ATTR, _WARM_DT_ATTR):
        if hasattr(mesh, _a):
            delattr(mesh, _a)


def init_guess(mesh, p_corr, outer, dt=None):
    """Inicializa p_corr antes de resolver el Poisson del outer indicado.

    Sustituye a `p_corr.fill(cp.float32(0.0))`.

    Sólo el outer 0 admite semilla: los outers siguientes resuelven un defecto
    con un RHS distinto y pequeño, para el que cero es la conjetura correcta.

    POR QUE HAY UNA VARIANTE REESCALADA. El RHS del outer 0 es (rho/dt)·div(u*)
    con u* = u^n + dt·F, o sea:

        RHS = (rho/dt)·div(u^n) + rho·div(F)

    El segundo término no depende de dt; el primero sí, y sólo desaparece si el
    paso anterior quedó perfectamente proyectado, div(u^n) = 0. Con presupuesto
    corto no lo queda: la proyección no alcanza la tolerancia y deja divergencia
    residual. Entonces el RHS —y por linealidad la solución— escala como 1/dt, y
    reusar sin más la semilla de un paso con otro dt la deja mal escalada por
    exactamente ese factor.

    Eso realimenta: residuo -> velocidad espuria -> el CFL adaptativo encoge dt
    -> el término 1/dt crece -> la semilla empeora -> más residuo. En los blowups
    observados dt caía 58x respecto al inicial antes de reventar.

    `warm_start_scaled` corrige el término dominante multiplicando la semilla por
    dt_anterior/dt_actual. Cuando dt es estable el factor vale 1 y el
    comportamiento es idéntico al de `warm_start` a secas, así que sólo actúa en
    el régimen donde el otro falla.
    """
    if not OPTS["warm_start"] or outer != 0:
        p_corr.fill(cp.float32(0.0))
        return

    warm = getattr(mesh, _WARM_ATTR, None)
    if warm is None or warm.shape != p_corr.shape:
        p_corr.fill(cp.float32(0.0))
        return

    if OPTS["warm_start_scaled"]:
        dt_prev = getattr(mesh, _WARM_DT_ATTR, None)
        if dt_prev and dt:
            r = float(dt_prev) / float(dt)
            if not (1.0 / _RATIO_MAX) <= r <= _RATIO_MAX:
                p_corr.fill(cp.float32(0.0))
                return
            p_corr[:] = warm * cp.float32(r)
        else:
            p_corr[:] = warm
    else:
        p_corr[:] = warm

    if OPTS["warm_start_filtered"]:
        _filtra_tablero(mesh, p_corr)


def store_guess(mesh, p_corr, outer, dt=None):
    """Guarda la solución del outer 0 como semilla del siguiente paso.

    Guarda también el dt con el que se obtuvo: sin él no se puede reescalar."""
    if not OPTS["warm_start"] or outer != 0:
        return
    warm = getattr(mesh, _WARM_ATTR, None)
    if warm is None or warm.shape != p_corr.shape:
        setattr(mesh, _WARM_ATTR, p_corr.copy())
    else:
        warm[:] = p_corr
    if dt:
        setattr(mesh, _WARM_DT_ATTR, float(dt))


# ----------------------------------------------------------------------
# coarse_mask_*
# ----------------------------------------------------------------------
def coarsen_solid(coarsen_or, mask, sx, sy):
    """Construye la máscara de sólidos de un nivel grueso a partir del fino.

    Sustituye a `_coarsen_mask(mask, sx, sy)` en `_init_mg_hierarchy`.

    `coarsen_or` es la función original (OR sobre el bloque 3x3), que se usa tal
    cual cuando ninguna opción está activa.
    """
    if OPTS["coarse_mask_none"]:
        ny_m, nx_m = mask.shape
        return cp.zeros(((ny_m - sy) // 2, (nx_m - sx) // 2), dtype=cp.bool_)

    if not OPTS["coarse_mask_majority"]:
        return coarsen_or(mask, sx, sy)

    # Mayoría del bloque 3x3 centrado en el vértice grueso: >=5 de 9 celdas
    # finas sólidas. Mismo muestreo que la versión OR (mismos offsets, mismo
    # recorte en los bordes) para que la geometría de los niveles no se mueva;
    # sólo cambia el criterio de decisión, de "alguna" a "la mayoría".
    ny_m, nx_m = mask.shape
    nx_c = (nx_m - sx) // 2
    ny_c = (ny_m - sy) // 2
    cuenta = cp.zeros((ny_c, nx_c), dtype=cp.int8)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            iy0, ix0 = sy + di, sx + dj
            ic_start, iy_eff = (1, iy0 + 2) if iy0 < 0 else (0, iy0)
            jc_start, ix_eff = (1, ix0 + 2) if ix0 < 0 else (0, ix0)
            if iy_eff >= ny_m or ix_eff >= nx_m:
                continue
            sub = mask[iy_eff:iy_eff + (ny_c - ic_start) * 2:2,
                       ix_eff:ix_eff + (nx_c - jc_start) * 2:2]
            h, w = sub.shape
            cuenta[ic_start:ic_start + h, jc_start:jc_start + w] += sub
    return cuenta >= 5


# ----------------------------------------------------------------------
# line_smoother
# ----------------------------------------------------------------------
# Dos kernels simetricos: cada hilo resuelve una linea completa con Thomas.
#
# Ordenacion cebra. Los hilos se lanzan por lineas alternas (primero las pares,
# luego las impares). No es un detalle de rendimiento sino de correccion: al
# resolver la linea j se leen p[.,j-1] y p[.,j+1], que son lineas vecinas. Si
# todas se actualizaran a la vez, cada hilo leeria valores a medio escribir de
# sus vecinas y el resultado dependeria del orden de ejecucion. Con cebra, las
# lineas que se actualizan en una fase nunca son adyacentes, asi que las vecinas
# estan quietas: el resultado es determinista y reproducible entre corridas.
#
# Escrituras. El barrido hacia delante de Thomas produce c' y d', que son
# cantidades intermedias, no presion. Van a dos buffers propios; p solo se
# escribe en la sustitucion hacia atras, ya con el valor definitivo. Aparcar d'
# dentro de p corrompe lo que leen las lineas vecinas.
#
# Reparto de trabajo. En lineas Y el hilo j recorre i, y los hilos consecutivos
# leen p[i*nx + j] con j consecutivo: accesos contiguos, coalescidos. En lineas X
# el hilo i recorre j y los accesos de hilos vecinos van separados nx: no
# coalescen. Se paga esa asimetria a cambio de no transponer el campo.
#
# Contorno del solido. En el kernel punto a punto, un vecino solido no aporta
# termino y su coeficiente se suma a la diagonal, que es Neumann homogeneo en la
# cara (p_solido = p_centro). Aqui se reproduce igual: coeficiente a la diagonal
# y el enlace tridiagonal correspondiente se corta a cero, para no acoplar una
# celda solida dentro de la linea. Las celdas solidas quedan p=0 y desacopladas.
#
# Las filas y columnas del borde del dominio no se tocan, igual que en el kernel
# punto a punto (hace `return` en ii<=0, ii>=ny-1, j<=0, j>=nx-1); sus valores
# los fija _aplicar_bc_mg y aqui entran como dato conocido en el termino
# independiente de la primera y ultima incognita.
_LINE_SRC = r'''
extern "C" __global__ void line_y(
    const bool*  __restrict__ solid,
    float*       __restrict__ p,
    const float* __restrict__ rhs,
    float*       __restrict__ cbuf,      // c' de Thomas
    float*       __restrict__ dbuf,      // d' de Thomas
    const float* __restrict__ d2x_W, const float* __restrict__ d2x_C,
    const float* __restrict__ d2x_E,
    const float* __restrict__ d2y_S, const float* __restrict__ d2y_C,
    const float* __restrict__ d2y_N,
    float omega, int nx, int ny, int phase)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j <= 0 || j >= nx - 1) return;
    if ((j & 1) != phase) return;
    if (ny < 3) return;

    float aW = d2x_W[j], aCx = d2x_C[j], aE = d2x_E[j];

    float c_prev = 0.0f, d_prev = 0.0f;
    for (int i = 1; i <= ny - 2; ++i) {
        int idx = i * nx + j;
        float diag, b, sub, sup;
        if (solid[idx]) {
            diag = 1.0f; b = 0.0f; sub = 0.0f; sup = 0.0f;
        } else {
            float aS = d2y_S[i], aCy = d2y_C[i], aN = d2y_N[i];
            diag = aCx + aCy;
            b = rhs[idx];
            if (solid[idx + 1]) diag += aE; else b -= aE * p[idx + 1];
            if (solid[idx - 1]) diag += aW; else b -= aW * p[idx - 1];
            sup = aN; sub = aS;
            if (solid[idx + nx]) { diag += aN; sup = 0.0f; }
            if (solid[idx - nx]) { diag += aS; sub = 0.0f; }
            if (i == ny - 2) { b -= sup * p[idx + nx]; sup = 0.0f; }
            if (i == 1)      { b -= sub * p[idx - nx]; sub = 0.0f; }
        }
        float den = diag - sub * c_prev;
        if (fabsf(den) < 1e-30f) den = 1e-30f;
        float c_i = sup / den;
        float d_i = (b - sub * d_prev) / den;
        cbuf[idx] = c_i;
        dbuf[idx] = d_i;
        c_prev = c_i; d_prev = d_i;
    }

    float next = 0.0f;
    for (int i = ny - 2; i >= 1; --i) {
        int idx = i * nx + j;
        float val = dbuf[idx] - cbuf[idx] * next;
        next = val;
        p[idx] = solid[idx] ? 0.0f : ((1.0f - omega) * p[idx] + omega * val);
    }
}

extern "C" __global__ void line_x(
    const bool*  __restrict__ solid,
    float*       __restrict__ p,
    const float* __restrict__ rhs,
    float*       __restrict__ cbuf,
    float*       __restrict__ dbuf,
    const float* __restrict__ d2x_W, const float* __restrict__ d2x_C,
    const float* __restrict__ d2x_E,
    const float* __restrict__ d2y_S, const float* __restrict__ d2y_C,
    const float* __restrict__ d2y_N,
    float omega, int nx, int ny, int phase)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i <= 0 || i >= ny - 1) return;
    if ((i & 1) != phase) return;
    if (nx < 3) return;

    float aS = d2y_S[i], aCy = d2y_C[i], aN = d2y_N[i];

    float c_prev = 0.0f, d_prev = 0.0f;
    for (int j = 1; j <= nx - 2; ++j) {
        int idx = i * nx + j;
        float diag, b, sub, sup;
        if (solid[idx]) {
            diag = 1.0f; b = 0.0f; sub = 0.0f; sup = 0.0f;
        } else {
            float aW = d2x_W[j], aCx = d2x_C[j], aE = d2x_E[j];
            diag = aCx + aCy;
            b = rhs[idx];
            if (solid[idx + nx]) diag += aN; else b -= aN * p[idx + nx];
            if (solid[idx - nx]) diag += aS; else b -= aS * p[idx - nx];
            sup = aE; sub = aW;
            if (solid[idx + 1]) { diag += aE; sup = 0.0f; }
            if (solid[idx - 1]) { diag += aW; sub = 0.0f; }
            if (j == nx - 2) { b -= sup * p[idx + 1]; sup = 0.0f; }
            if (j == 1)      { b -= sub * p[idx - 1]; sub = 0.0f; }
        }
        float den = diag - sub * c_prev;
        if (fabsf(den) < 1e-30f) den = 1e-30f;
        float c_j = sup / den;
        float d_j = (b - sub * d_prev) / den;
        cbuf[idx] = c_j;
        dbuf[idx] = d_j;
        c_prev = c_j; d_prev = d_j;
    }

    float next = 0.0f;
    for (int j = nx - 2; j >= 1; --j) {
        int idx = i * nx + j;
        float val = dbuf[idx] - cbuf[idx] * next;
        next = val;
        p[idx] = solid[idx] ? 0.0f : ((1.0f - omega) * p[idx] + omega * val);
    }
}
'''

_line_y = None
_line_x = None


def _kernels():
    global _line_y, _line_x
    if _line_y is None:
        mod = cp.RawModule(code=_LINE_SRC)
        _line_y = mod.get_function("line_y")
        _line_x = mod.get_function("line_x")
    return _line_y, _line_x


def line_smoother_active():
    return OPTS["line_smoother"] or OPTS["line_smoother_y"]


def smooth_lines(mesh, p_arr, rhs, lvl, n_sweeps, omega):
    """Suaviza con relajación por líneas alternada Y/X, cada dirección en dos
    fases cebra. Reemplaza a los dos lanzamientos rojo-negro por barrido del
    suavizador punto a punto."""
    ky, kx = _kernels()
    ny, nx = p_arr.shape
    solid = mesh._mg_solids_flat[lvl]
    d2xW, d2xC, d2xE = (mesh._mg_d2x_W[lvl], mesh._mg_d2x_C[lvl], mesh._mg_d2x_E[lvl])
    d2yS, d2yC, d2yN = (mesh._mg_d2y_S[lvl], mesh._mg_d2y_C[lvl], mesh._mg_d2y_N[lvl])

    bufs = getattr(mesh, "_opt_line_scratch", None)
    if bufs is None:
        bufs = mesh._opt_line_scratch = {}
    par = bufs.get(lvl)
    if par is None or par[0].shape != (ny * nx,):
        par = bufs[lvl] = (cp.empty(ny * nx, dtype=cp.float32),
                           cp.empty(ny * nx, dtype=cp.float32))
    cbuf, dbuf = par

    p_flat, rhs_flat = p_arr.ravel(), rhs.ravel()
    om = cp.float32(omega)
    nx_i, ny_i = cp.int32(nx), cp.int32(ny)
    fase = (cp.int32(0), cp.int32(1))
    blk = 128
    gy = ((nx + blk - 1) // blk,)
    gx = ((ny + blk - 1) // blk,)

    def base(ph):
        return (solid, p_flat, rhs_flat, cbuf, dbuf,
                d2xW, d2xC, d2xE, d2yS, d2yC, d2yN, om, nx_i, ny_i, ph)

    solo_y = OPTS["line_smoother_y"] and not OPTS["line_smoother"]
    for s in range(n_sweeps):
        if solo_y:
            orden = ((ky, gy),)
        else:
            # Alternar cual de las dos direcciones va primero evita sesgar la
            # solucion hacia la que se resuelve la ultima.
            orden = ((ky, gy), (kx, gx)) if (s % 2 == 0) else ((kx, gx), (ky, gy))
        for k, g in orden:
            for ph in fase:
                k(g, (blk,), base(ph))


# ----------------------------------------------------------------------
# fast_masks
# ----------------------------------------------------------------------
def fast_masks_active():
    return OPTS["fast_masks"]


def _bc_cache(mesh, lvl, solids, dirichlet):
    """Máscaras de borde de _aplicar_bc_mg, cacheadas por nivel.

    Dependen sólo de la geometría (sólidos y puntos de presión fija), así que son
    constantes durante toda la simulación. El código original las recalculaba en
    cada llamada, y _aplicar_bc_mg se llama varias veces por V-cycle y por nivel.
    """
    cache = getattr(mesh, "_opt_bc_cache", None)
    if cache is None:
        cache = mesh._opt_bc_cache = {}
    m = cache.get(lvl)
    if m is None:
        fl = ~solids
        m = cache[lvl] = {
            "w": fl[:, 0] & ~dirichlet[:, 0],
            "e": fl[:, -1] & ~dirichlet[:, -1],
            "s": fl[0, :] & ~dirichlet[0, :],
            "n": fl[-1, :] & ~dirichlet[-1, :],
        }
    return m


def aplicar_bc_mg(mesh, p_nivel, lvl, hay_dirichlet, valor_fijo):
    """Neumann de gradiente nulo en los bordes, respetando Dirichlet.

    Sustituye al cuerpo de `_aplicar_bc_mg`. Misma operacion, sin indexado.
    """
    ny_n, nx_n = p_nivel.shape
    if hay_dirichlet:
        dm = mesh._mg_dirichlet[lvl]
        mk = _bc_cache(mesh, lvl, mesh._mg_solids[lvl], dm)
        if nx_n >= 2:
            cp.copyto(p_nivel[:, 0], p_nivel[:, 1], where=mk["w"])
            cp.copyto(p_nivel[:, -1], p_nivel[:, -2], where=mk["e"])
        if ny_n >= 2:
            cp.copyto(p_nivel[0, :], p_nivel[1, :], where=mk["s"])
            cp.copyto(p_nivel[-1, :], p_nivel[-2, :], where=mk["n"])
        v = valor_fijo if lvl == 0 else cp.float32(0.0)
        cp.copyto(p_nivel, cp.asarray(v, dtype=p_nivel.dtype), where=dm)
    else:
        if nx_n >= 2:
            p_nivel[:, 0] = p_nivel[:, 1]
            p_nivel[:, -1] = p_nivel[:, -2]
        if ny_n >= 2:
            p_nivel[0, :] = p_nivel[1, :]
            p_nivel[-1, :] = p_nivel[-2, :]


def cero_en_solido(arr, mask):
    """Pone a cero donde la máscara es cierta. Sustituye a `arr[mask] = 0.0`."""
    if not OPTS["fast_masks"]:
        arr[mask] = cp.float32(0.0)
        return
    cp.copyto(arr, cp.zeros((), dtype=arr.dtype), where=mask)

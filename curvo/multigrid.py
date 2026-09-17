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

**numpy o cupy, indistintamente.** El modulo trabaja con el `xp` del array que
recibe. La unica pieza que no se puede escribir con operaciones de array es la
recursion de Thomas, que es secuencial: en CPU es el bucle de siempre y en GPU
un kernel con **un hilo por linea** (ver `_thomas_gpu`). En float32, que es como
corre el solver: la 3070 Ti hace fp64 a 1/64 de fp32, asi que float64 en GPU es
mas lento que la CPU.
"""

from __future__ import annotations

import numpy as np

from .metrica import xp_de

__all__ = ["Sistema", "jerarquia", "ciclo_v", "resolver", "resolver_pcg",
           "tolerancia"]


# ---------------------------------------------------------------------------
# Recursion de Thomas
# ---------------------------------------------------------------------------
_FUENTE_THOMAS = r"""
extern "C" __global__
void thomas(const {T}* sub, const {T}* dia, const {T}* sup, const {T}* rhs,
            {T}* x, {T}* cp, {T}* dp, const int n, const int m)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= m) return;

    {T} c = sup[j] / dia[j];
    {T} d = rhs[j] / dia[j];
    cp[j] = c;
    dp[j] = d;
    for (int k = 1; k < n; ++k) {
        int o = k * m + j;
        {T} den = dia[o] - sub[o] * c;
        c = sup[o] / den;
        d = (rhs[o] - sub[o] * d) / den;
        cp[o] = c;
        dp[o] = d;
    }
    x[(n - 1) * m + j] = d;
    for (int k = n - 2; k >= 0; --k) {
        int o = k * m + j;
        d = dp[o] - cp[o] * x[(k + 1) * m + j];
        x[o] = d;
    }
}
"""

_KERNELS = {}


def _kernel_thomas(dtype):
    """Kernel compilado para `dtype`, cacheado. Un hilo resuelve una linea entera.

    Los sistemas van en un array `(n, m)` contiguo: el hilo `j` recorre la
    columna `j`, y en cada paso `k` los hilos consecutivos leen posiciones
    consecutivas `k*m + j`, con lo que los accesos salen fundidos.
    """
    if dtype not in _KERNELS:
        import cupy as cp
        tipo = {np.dtype("float32"): "float", np.dtype("float64"): "double"}[dtype]
        fuente = _FUENTE_THOMAS.replace("{T}", tipo)
        _KERNELS[dtype] = cp.RawKernel(fuente, "thomas")
    return _KERNELS[dtype]


def _thomas_cpu(sub, dia, sup, rhs):
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


def _thomas_gpu(sub, dia, sup, rhs):
    import cupy as cp
    n, m = dia.shape
    sub, dia, sup, rhs = (cp.ascontiguousarray(a) for a in (sub, dia, sup, rhs))
    x = cp.empty_like(dia)
    scratch = cp.empty((2, n, m), dtype=dia.dtype)
    hilos = 128
    _kernel_thomas(dia.dtype)(
        ((m + hilos - 1) // hilos,), (hilos,),
        (sub, dia, sup, rhs, x, scratch[0], scratch[1], n, m))
    return x


def _thomas(sub, dia, sup, rhs):
    """Tridiagonales independientes a lo largo del eje 0. Arrays (n, m)."""
    if xp_de(dia) is np:
        return _thomas_cpu(sub, dia, sup, rhs)
    return _thomas_gpu(sub, dia, sup, rhs)



# ---------------------------------------------------------------------------
# Kernels fundidos del suavizador
# ---------------------------------------------------------------------------
# Una malla C de etapa 1 son 21 000 celdas: para una GPU es nada, y un ciclo V
# escrito con operaciones de array lanza ~1700 kernels diminutos. Medido: asi la
# GPU sale **mas lenta** que la CPU (0.63 s frente a 0.43 s). Lo que lo cambia es
# fundir cada barrido cebra en un solo lanzamiento -- los terminos retrasados se
# calculan dentro del hilo en vez de materializarse en memoria -- y quedarse en
# ~112 lanzamientos por ciclo.
#
# Un hilo resuelve una linea entera con la recursion de Thomas. Los coeficientes
# de la linea se leen con paso `nx` (lineas eta) o 1 (lineas xi); el rascadero de
# la recursion va indexado `k*L + t` para que los hilos consecutivos escriban
# posiciones consecutivas.
_FUENTE = r"""
extern "C" __global__
void zebra_eta({T}* phi, const {T}* aP, const {T}* aW, const {T}* aE,
               const {T}* aS, const {T}* aN, const {T}* b,
               const int* cols, {T}* cp, {T}* dp,
               const int L, const int ny, const int nx)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= L) return;
    int i = cols[t];

    {T} c = 0, d = 0;
    for (int k = 0; k < ny; ++k) {
        int o = k * nx + i;
        {T} r = b[o];
        if (i > 0)      r += aW[o] * phi[o - 1];
        if (i < nx - 1) r += aE[o] * phi[o + 1];
        {T} den = aP[o] + aS[o] * c;
        c = -aN[o] / den;
        d = (r + aS[o] * d) / den;
        cp[k * L + t] = c;
        dp[k * L + t] = d;
    }
    phi[(ny - 1) * nx + i] = d;
    for (int k = ny - 2; k >= 0; --k) {
        d = dp[k * L + t] - cp[k * L + t] * d;
        phi[k * nx + i] = d;
    }
}

extern "C" __global__
void zebra_eta_par({T}* phi, const {T}* aP, const {T}* aW, const {T}* aE,
                   const {T}* aS, const {T}* aN, const {T}* aC, const {T}* b,
                   const int* cols, {T}* cp, {T}* dp,
                   const int L, const int ny, const int nx)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= L) return;
    int ia = cols[t];
    int ib = nx - 1 - ia;
    int n = 2 * ny;

    {T} c = 0, d = 0;
    for (int k = 0; k < n; ++k) {
        int fila = (k < ny) ? (ny - 1 - k) : (k - ny);
        int i    = (k < ny) ? ia : ib;
        int o    = fila * nx + i;
        {T} sub, sup;
        if (k < ny) {                       // baja por la columna ia
            sub = -aN[o];
            sup = (k == ny - 1) ? -aC[ia] : -aS[o];
        } else {                            // sube por la columna espejo
            sub = (fila == 0) ? -aC[ib] : -aS[o];
            sup = -aN[o];
        }
        {T} r = b[o];
        if (i > 0)      r += aW[o] * phi[o - 1];
        if (i < nx - 1) r += aE[o] * phi[o + 1];
        {T} den = aP[o] - sub * c;
        c = sup / den;
        d = (r - sub * d) / den;
        cp[k * L + t] = c;
        dp[k * L + t] = d;
    }
    phi[(n - 1 - ny) * nx + ib] = d;        // ultima incognita: fila ny-1, col ib
    for (int k = n - 2; k >= 0; --k) {
        d = dp[k * L + t] - cp[k * L + t] * d;
        int fila = (k < ny) ? (ny - 1 - k) : (k - ny);
        int i    = (k < ny) ? ia : ib;
        phi[fila * nx + i] = d;
    }
}

extern "C" __global__
void zebra_xi({T}* phi, const {T}* aP, const {T}* aW, const {T}* aE,
              const {T}* aS, const {T}* aN, const {T}* aC, const {T}* b,
              const int* filas, {T}* cp, {T}* dp,
              const int L, const int ny, const int nx, const int hay_corte)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= L) return;
    int f = filas[t];

    {T} c = 0, d = 0;
    for (int k = 0; k < nx; ++k) {
        int o = f * nx + k;
        {T} r = b[o];
        if (f > 0)      r += aS[o] * phi[o - nx];
        if (f < ny - 1) r += aN[o] * phi[o + nx];
        if (hay_corte && f == 0) r += aC[k] * phi[nx - 1 - k];
        {T} den = aP[o] + aW[o] * c;
        c = -aE[o] / den;
        d = (r + aW[o] * d) / den;
        cp[k * L + t] = c;
        dp[k * L + t] = d;
    }
    phi[f * nx + nx - 1] = d;
    for (int k = nx - 2; k >= 0; --k) {
        d = dp[k * L + t] - cp[k * L + t] * d;
        phi[f * nx + k] = d;
    }
}


extern "C" __global__
void engrosar({T}* aPc, {T}* aWc, {T}* aEc, {T}* aSc, {T}* aNc, {T}* aCc,
              const {T}* aP, const {T}* aW, const {T}* aE, const {T}* aS,
              const {T}* aN, const {T}* aC,
              const int* jini, const int* jfin, const int* iini, const int* ifin,
              const int ncy, const int ncx, const int nx, const int hay_corte)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= ncy * ncx) return;
    int I = o % ncx;
    int J = o / ncx;
    int j0 = jini[J], j1 = jfin[J], i0 = iini[I], i1 = ifin[I];

    {T} sP = 0, sW = 0, sE = 0, sS = 0, sN = 0;
    for (int j = j0; j <= j1; ++j) {
        for (int i = i0; i <= i1; ++i) {
            int f = j * nx + i;
            sP += aP[f];
            if (i < i1) sP -= aE[f];        // cara interna al bloque
            if (i > i0) sP -= aW[f];
            if (j < j1) sP -= aN[f];
            if (j > j0) sP -= aS[f];
            if (i == i0) sW += aW[f];       // cara que sale del bloque
            if (i == i1) sE += aE[f];
            if (j == j0) sS += aS[f];
            if (j == j1) sN += aN[f];
        }
    }
    if (hay_corte && J == 0) {
        {T} sC = 0;
        for (int i = i0; i <= i1; ++i) sC += aC[i];
        if (I == ncx - 1 - I) {             // bloque que es su propio espejo
            sP -= sC;
            sC = 0;
        }
        aCc[I] = sC;
    }
    aPc[o] = sP; aWc[o] = sW; aEc[o] = sE; aSc[o] = sS; aNc[o] = sN;
}

extern "C" __global__
void restringir({T}* rc, const {T}* r,
                const int* jini, const int* jfin, const int* iini, const int* ifin,
                const int ncy, const int ncx, const int nx)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= ncy * ncx) return;
    int I = o % ncx;
    int J = o / ncx;
    {T} s = 0;
    for (int j = jini[J]; j <= jfin[J]; ++j)
        for (int i = iini[I]; i <= ifin[I]; ++i)
            s += r[j * nx + i];
    rc[o] = s;
}

extern "C" __global__
void prolongar({T}* phi, const {T}* e, const int* kj, const int* ki,
               const {T}* peso, const int ny, const int nx, const int ncx)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= ny * nx) return;
    phi[o] += peso[0] * e[kj[o / nx] * ncx + ki[o % nx]];
}

extern "C" __global__
void aplicar({T}* r, const {T}* phi, const {T}* aP, const {T}* aW, const {T}* aE,
             const {T}* aS, const {T}* aN, const {T}* aC, const {T}* b,
             const int ny, const int nx, const int hay_corte, const int residuo)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= ny * nx) return;
    int i = o % nx;
    int j = o / nx;
    {T} v = aP[o] * phi[o];
    if (i > 0)      v -= aW[o] * phi[o - 1];
    if (i < nx - 1) v -= aE[o] * phi[o + 1];
    if (j > 0)      v -= aS[o] * phi[o - nx];
    if (j < ny - 1) v -= aN[o] * phi[o + nx];
    if (hay_corte && j == 0) v -= aC[i] * phi[nx - 1 - i];
    r[o] = residuo ? (b[o] - v) : v;
}
"""

_MODULOS = {}


def _modulo(dtype):
    if dtype not in _MODULOS:
        import cupy as cp
        tipo = {np.dtype("float32"): "float", np.dtype("float64"): "double"}[dtype]
        _MODULOS[dtype] = cp.RawModule(code=_FUENTE.replace("{T}", tipo),
                                       options=("--use_fast_math",),
                                       name_expressions=None)
    return _MODULOS[dtype]


_HILOS = 64


def _rejilla(n):
    return ((n + _HILOS - 1) // _HILOS,), (_HILOS,)


class Sistema:
    """Coeficientes de 5 puntos mas el acoplamiento del corte de estela.

    Formas: todo `(ny, nx)` salvo `aC`, que es `(nx,)` y solo actua en `j=0`.
    Los coeficientes de vecino valen cero donde no hay vecino.
    """

    def __init__(self, aP, aW, aE, aS, aN, b, aC=None, activo=None):
        self.aP, self.aW, self.aE, self.aS, self.aN, self.b = aP, aW, aE, aS, aN, b
        self.aC = aC
        self.ny, self.nx = aP.shape
        self.xp = xp_de(aP)
        self.gpu = self.xp is not np
        self._indices(activo)
        if self.gpu:
            mod = _modulo(aP.dtype)
            self._k_eta = mod.get_function("zebra_eta")
            self._k_par = mod.get_function("zebra_eta_par")
            self._k_xi = mod.get_function("zebra_xi")
            self._k_apl = mod.get_function("aplicar")
            self._cero = self.xp.zeros(self.nx, dtype=aP.dtype)
            self._rasca = {}

    def _indices(self, activo=None):
        """Indices de los barridos cebra, precalculados.

        Se calculan una vez y se dejan en el dispositivo: construirlos en cada
        barrido con una mascara booleana obliga a sincronizar con la GPU para
        saber cuantos elementos salen.

        `activo` (las columnas de corte) es **geometria**, no valores, asi que
        viene dado desde fuera y se lleva en la CPU. Deducirlo de `aC != 0`
        obliga a bajar `aC` del dispositivo, y eso es una sincronizacion por cada
        `Sistema` construido: con la matriz del momento rehecha en cada paso y
        siete niveles, catorce sincronizaciones por paso de tiempo.
        """
        xp = self.xp
        i = np.arange(self.nx)
        espejo = self.nx - 1 - i
        if activo is not None:
            activo = np.asarray(activo.get() if xp_de(activo) is not np else activo,
                                dtype=bool)
        elif self.aC is None:
            activo = np.zeros(self.nx, bool)
        else:
            activo = (np.asarray(self.aC.get() if xp is not np else self.aC) != 0.0)
        activo = activo & (espejo != i)
        self._activo = activo
        baja = activo & (i < espejo)
        sueltas = i[~activo]
        pares = i[baja]
        tipo = np.int32 if self.gpu else np.intp
        self._col = [xp.asarray(sueltas[sueltas % 2 == k], dtype=tipo) for k in (0, 1)]
        self._par = [xp.asarray(pares[pares % 2 == k], dtype=tipo) for k in (0, 1)]
        self._par_espejo = [self.nx - 1 - c for c in self._par]
        self._fil = [xp.asarray(np.arange(k, self.ny, 2), dtype=tipo) for k in (0, 1)]

    def _scratch(self, n, L):
        clave = (n, int(L))
        if clave not in self._rasca:
            self._rasca[clave] = self.xp.empty((2, n, int(L)), dtype=self.aP.dtype)
        return self._rasca[clave]

    def aplicar(self, phi, b=None, residuo=False):
        if self.gpu:
            r = self.xp.empty_like(phi)
            n = self.ny * self.nx
            self._k_apl(*_rejilla(n), (
                r, phi, self.aP, self.aW, self.aE, self.aS, self.aN,
                self._cero if self.aC is None else self.aC,
                self._cero if b is None else b,
                np.int32(self.ny), np.int32(self.nx),
                np.int32(self.aC is not None), np.int32(residuo)))
            return r
        r = self.aP * phi
        r[:, 1:] -= self.aW[:, 1:] * phi[:, :-1]
        r[:, :-1] -= self.aE[:, :-1] * phi[:, 1:]
        r[1:, :] -= self.aS[1:, :] * phi[:-1, :]
        r[:-1, :] -= self.aN[:-1, :] * phi[1:, :]
        if self.aC is not None:
            r[0] -= self.aC * phi[0, ::-1]
        return r

    def residuo(self, phi):
        if self.gpu:
            return self.aplicar(phi, self.b, residuo=True)
        return self.b - self.aplicar(phi)

    # -- suavizado ---------------------------------------------------------
    def _retrasado_xi(self, phi):
        """Terminos que no entran en una linea eta: solo los vecinos en xi.

        El corte **si** entra en la linea eta (ver `_lineas_eta`), asi que no se
        retrasa aqui.
        """
        t = self.xp.zeros_like(phi)
        t[:, 1:] += self.aW[:, 1:] * phi[:, :-1]
        t[:, :-1] += self.aE[:, :-1] * phi[:, 1:]
        return t

    def _retrasado_eta(self, phi):
        """Idem para una linea xi: vecinos en eta. El corte vive dentro de la
        linea j=0 pero no es tridiagonal, asi que tambien va retrasado."""
        t = self.xp.zeros_like(phi)
        t[1:, :] += self.aS[1:, :] * phi[:-1, :]
        t[:-1, :] += self.aN[:-1, :] * phi[1:, :]
        if self.aC is not None:
            t[0] += self.aC * phi[0, ::-1]
        return t

    def _lineas_eta(self, phi, paridad):
        """Barrido cebra de lineas eta. Las columnas de corte van **de dos en dos**.

        En una columna de estela la linea eta no acaba en `j=0`: al otro lado de
        la cara esta la celda `(0, nx-1-i)`, que es la misma cara fisica. La
        linea correcta baja por la columna `i` desde el campo lejano, cruza el
        corte y sube por la columna espejo: una tridiagonal de longitud `2*ny`.
        Dejar `aC` retrasado atasca el multigrid -- sobre la malla C el
        acoplamiento del corte llega a la mitad de la diagonal y el factor del
        PCG pasa de 0.11 a 0.64.
        """
        xp = self.xp
        if self.gpu:
            c = self._col[paridad]
            if c.size:
                ra = self._scratch(self.ny, c.size)
                self._k_eta(*_rejilla(c.size), (
                    phi, self.aP, self.aW, self.aE, self.aS, self.aN, self.b,
                    c, ra[0], ra[1], np.int32(c.size), np.int32(self.ny),
                    np.int32(self.nx)))
            a = self._par[paridad]
            if a.size:
                ra = self._scratch(2 * self.ny, a.size)
                self._k_par(*_rejilla(a.size), (
                    phi, self.aP, self.aW, self.aE, self.aS, self.aN,
                    self.aC, self.b, a, ra[0], ra[1], np.int32(a.size),
                    np.int32(self.ny), np.int32(self.nx)))
            return

        c = self._col[paridad]
        if c.size:
            d = self.b + self._retrasado_xi(phi)
            phi[:, c] = _thomas(-self.aS[:, c], self.aP[:, c], -self.aN[:, c],
                                d[:, c])

        a = self._par[paridad]
        if a.size == 0:
            return
        # `d` se recalcula: las columnas sueltas ya se han actualizado. Una
        # suelta y una de par pueden caer adyacentes con la misma paridad -- el
        # emparejamiento espejo lo impone en la union perfil-estela -- asi que el
        # barrido no es cebra puro ahi y el orden importa. Resolver los dos
        # grupos en cascada es ademas mejor suavizador que congelar `d`.
        d = self.b + self._retrasado_xi(phi)
        b = self._par_espejo[paridad]
        inv = slice(None, None, -1)
        dia = xp.concatenate([self.aP[inv, :][:, a], self.aP[:, b]])
        sub = xp.concatenate([-self.aN[inv, :][:, a], -self.aS[:, b]])
        sup = xp.concatenate([-self.aS[inv, :][:, a], -self.aN[:, b]])
        rhs = xp.concatenate([d[inv, :][:, a], d[:, b]])
        sub[self.ny] = -self.aC[b]                  # cruce del corte, rama alta
        sup[self.ny - 1] = -self.aC[a]              # cruce del corte, rama baja
        x = _thomas(sub, dia, sup, rhs)
        phi[:, a] = x[:self.ny][inv]
        phi[:, b] = x[self.ny:]

    def _lineas_xi(self, phi, paridad):
        f = self._fil[paridad]
        if f.size == 0:
            return
        if self.gpu:
            ra = self._scratch(self.nx, f.size)
            self._k_xi(*_rejilla(f.size), (
                phi, self.aP, self.aW, self.aE, self.aS, self.aN,
                self._cero if self.aC is None else self.aC, self.b,
                f, ra[0], ra[1], np.int32(f.size), np.int32(self.ny),
                np.int32(self.nx), np.int32(self.aC is not None)))
            return
        d = (self.b + self._retrasado_eta(phi))[f, :].T
        phi[f, :] = _thomas(-self.aW[f, :].T, self.aP[f, :].T,
                            -self.aE[f, :].T, d).T

    def suavizar(self, phi, veces=1, invertido=False):
        """`invertido` recorre los cuatro barridos al reves.

        Hace falta para que el ciclo V sea un operador **simetrico**, que es lo
        que el PCG exige del precondicionador: pre-suavizado en un orden y
        post-suavizado en el contrario.
        """
        orden = [(self._lineas_eta, 0), (self._lineas_eta, 1),
                 (self._lineas_xi, 0), (self._lineas_xi, 1)]
        if invertido:
            orden.reverse()
        for _ in range(veces):
            for barrido, paridad in orden:
                barrido(phi, paridad)


# ---------------------------------------------------------------------------
# Aglomeracion
# ---------------------------------------------------------------------------
def tolerancia(dtype, tol=None):
    """Tolerancia relativa por defecto segun la precision.

    En float32 el residuo relativo no baja de ~1e-6: pedir 1e-13 no converge
    nunca y se gastan todos los ciclos. El solver corre en float32 porque la
    3070 Ti hace fp64 a 1/64 de fp32.
    """
    if tol is not None:
        return tol
    return 1e-13 if np.dtype(dtype) == np.float64 else 2e-6


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


def _bloques(mapa, nc):
    """Primer y ultimo indice fino de cada bloque grueso.

    El mapa es monotono no decreciente y cada bloque es un tramo **contiguo**,
    tanto en el engrosado normal como en el simetrico. Eso es lo que permite
    aglomerar con un kernel de recoleccion (cada hilo suma sus <=4 celdas finas)
    en vez de con `bincount`, que en cupy cuesta 0.18 ms por llamada y es el 80 %
    del tiempo de construir la jerarquia.
    """
    k = np.arange(nc)
    return (np.searchsorted(mapa, k, "left").astype(np.int32),
            (np.searchsorted(mapa, k, "right") - 1).astype(np.int32))


def _sumar(xp, idx, valores, forma):
    return xp.bincount(idx.ravel(), valores.ravel(),
                       minlength=forma[0] * forma[1]).reshape(forma).astype(
                           valores.dtype, copy=False)


def _engrosar_gpu(sis, ncy, ncx, bloq):
    xp = sis.xp
    tipo = sis.aP.dtype
    forma = (ncy, ncx)
    salida = [xp.empty(forma, dtype=tipo) for _ in range(5)]
    aCc = xp.zeros(ncx, dtype=tipo) if sis.aC is not None else xp.zeros(1, dtype=tipo)
    _modulo(tipo).get_function("engrosar")(*_rejilla(ncy * ncx), (
        *salida, aCc, sis.aP, sis.aW, sis.aE, sis.aS, sis.aN,
        sis.aC if sis.aC is not None else aCc,
        *bloq, np.int32(ncy), np.int32(ncx), np.int32(sis.nx),
        np.int32(sis.aC is not None)))
    aP, aW, aE, aS, aN = salida
    return Sistema(aP, aW, aE, aS, aN, xp.zeros(forma, dtype=tipo),
                   aCc if sis.aC is not None else None)


def _engrosar(sis, kj, ncy, ki, ncx):
    xp = sis.xp
    forma = (ncy, ncx)
    idx = xp.asarray(kj[:, None] * ncx + ki[None, :])

    mismo_e = xp.asarray(ki[:-1] == ki[1:])          # cara xi interna al bloque
    mismo_n = xp.asarray(kj[:-1] == kj[1:])

    aP = _sumar(xp, idx, sis.aP, forma)
    aP -= _sumar(xp, idx[:, :-1], sis.aE[:, :-1] * mismo_e, forma)
    aP -= _sumar(xp, idx[:, 1:], sis.aW[:, 1:] * mismo_e, forma)
    aP -= _sumar(xp, idx[:-1, :], sis.aN[:-1, :] * mismo_n[:, None], forma)
    aP -= _sumar(xp, idx[1:, :], sis.aS[1:, :] * mismo_n[:, None], forma)

    aE = _sumar(xp, idx[:, :-1], sis.aE[:, :-1] * ~mismo_e, forma)
    aW = _sumar(xp, idx[:, 1:], sis.aW[:, 1:] * ~mismo_e, forma)
    aN = _sumar(xp, idx[:-1, :], sis.aN[:-1, :] * ~mismo_n[:, None], forma)
    aS = _sumar(xp, idx[1:, :], sis.aS[1:, :] * ~mismo_n[:, None], forma)

    activo = None
    if sis._activo.any():
        activo = np.zeros(ncx, bool)
        activo[ki[sis._activo]] = True

    aC = None
    if sis.aC is not None:
        propio = ki == (ncx - 1 - ki)                # bloque que es su propio espejo
        kis = xp.asarray(ki)
        fuera = xp.asarray(~propio)
        aC = _sumar(xp, kis[fuera], sis.aC[fuera], (1, ncx)).reshape(ncx)
        if propio.any():
            dentro = xp.asarray(propio)
            aP[0] -= _sumar(xp, kis[dentro], sis.aC[dentro], (1, ncx)).reshape(ncx)
    return Sistema(aP, aW, aE, aS, aN, xp.zeros(forma, dtype=aP.dtype), aC,
                   activo=activo)


def jerarquia(sis, minimo=9):
    """Lista [(sistema, kj, ki)] de fino a grueso. Los mapas son los del nivel."""
    xp = sis.xp
    niveles = [(sis, None, None)]
    while sis.aP.size > minimo:
        kj, ncy = _mapa(sis.ny)
        ki, ncx = _mapa(sis.nx, simetrico=True)
        if ncy * ncx >= sis.ny * sis.nx:
            break
        nx_padre = sis.nx
        if sis.gpu:
            bloq = tuple(xp.asarray(a) for a in
                         (*_bloques(kj, ncy), *_bloques(ki, ncx)))
            hijo = _engrosar_gpu(sis, ncy, ncx, bloq)
            activo = np.zeros(ncx, bool)
            if sis._activo.any():
                activo[ki[sis._activo]] = True
                hijo._indices(activo)
            hijo._bloq = (bloq, nx_padre)
            sis = hijo
            niveles.append((sis, xp.asarray(kj.astype(np.int32)),
                            xp.asarray(ki.astype(np.int32))))
        else:
            sis = _engrosar(sis, kj, ncy, ki, ncx)
            sis._bloq = None
            niveles.append((sis, kj, ki))
    return niveles


# ---------------------------------------------------------------------------
# Ciclo
# ---------------------------------------------------------------------------
def ciclo_v(niveles, phi, nivel=0, pre=2, post=2, grueso=4, w=None):
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

    Con `w` dado se salta el cociente y el ciclo pasa a ser simetrico (ver
    `Sistema.suavizar`), que es lo que `resolver_pcg` necesita.

    `grueso` son los barridos del nivel final, que en la malla C es de tres
    celdas: cada barrido son cuatro lanzamientos de kernel, asi que el valor sale
    caro en latencia y no compra nada. Con 30 costaba 0.80 ms de los 4.15 de un
    ciclo V (el 19 %) para relajar tres incognitas. Medido sobre los sistemas
    reales con termino independiente aleatorio: en conveccion el historial de
    residuos es **identico** con 1, 2, 4, 8 y 30 -- el operador es tan dominante
    en diagonal que la correccion gruesa no interviene -- y en el Poisson de
    presion el factor del PCG no empeora al bajar (0.067 con 30, 0.021 con 4).
    """
    sis = niveles[nivel][0]
    xp = sis.xp
    if nivel == len(niveles) - 1:
        sis.suavizar(phi, grueso)
        return phi
    sis.suavizar(phi, pre)
    sc, kj, ki = niveles[nivel + 1]
    r = sis.residuo(phi)
    if sis.gpu:
        bloq, _ = sc._bloq
        sc.b = xp.empty((sc.ny, sc.nx), dtype=phi.dtype)
        _modulo(phi.dtype).get_function("restringir")(*_rejilla(sc.ny * sc.nx), (
            sc.b, r, *bloq, np.int32(sc.ny), np.int32(sc.nx), np.int32(sis.nx)))
    else:
        sc.b = _sumar(xp, kj[:, None] * sc.nx + ki[None, :], r, (sc.ny, sc.nx))
    e = xp.zeros((sc.ny, sc.nx), dtype=phi.dtype)
    ciclo_v(niveles, e, nivel + 1, pre, post, grueso, w)

    if sis.gpu and w is not None:
        peso = xp.asarray([w], dtype=phi.dtype)
        _modulo(phi.dtype).get_function("prolongar")(*_rejilla(sis.ny * sis.nx), (
            phi, e, kj, ki, peso, np.int32(sis.ny), np.int32(sis.nx),
            np.int32(sc.nx)))
        sis.suavizar(phi, post, invertido=True)
        return phi

    ef = e[kj[:, None], ki[None, :]]
    if w is None:
        # El peso se deja como escalar **en el dispositivo**: sacarlo a la CPU
        # con float() obliga a sincronizar dos veces por nivel y por ciclo, que
        # sobre una malla de 21 000 celdas cuesta mas que el propio ciclo.
        den = (ef * sis.aplicar(ef)).sum()
        num = (r * ef).sum()
        peso = xp.clip(num / xp.where(xp.abs(den) > 1e-30, den, 1e-30), 0.5, 4.0)
    else:
        peso = w
    phi += peso * ef
    sis.suavizar(phi, post, invertido=w is not None)
    return phi


def resolver(sis, phi=None, tol=None, ciclos=60, pre=2, post=2, niveles=None):
    """Resuelve A phi = b. Devuelve (phi, info) con el historial de residuos."""
    xp = sis.xp
    if phi is None:
        phi = xp.zeros_like(sis.aP)
    niveles = niveles if niveles is not None else jerarquia(sis)
    tol = tolerancia(sis.aP.dtype, tol)
    escala = max(float(xp.abs(sis.b).max()), 1e-300)
    hist = [float(xp.abs(sis.residuo(phi)).max()) / escala]
    cada = 1 if xp is np else 2
    for k in range(ciclos):
        ciclo_v(niveles, phi, pre=pre, post=post)
        if k % cada == cada - 1 or k == ciclos - 1:
            hist.append(float(xp.abs(sis.residuo(phi)).max()) / escala)
            if hist[-1] <= tol:
                break
    utiles = [b / a for a, b in zip(hist[:-1], hist[1:]) if a > 0.0]
    return phi, {"residuos": hist, "ciclos": len(hist) - 1,
                 "factor": float(np.median(utiles)) if utiles else 0.0}


def _precondicionar(niveles, r, pre, post, w):
    """Un ciclo V desde cero sobre el residuo `r`. Devuelve la correccion."""
    sis = niveles[0][0]
    previo = sis.b
    sis.b = r
    z = ciclo_v(niveles, sis.xp.zeros_like(r), pre=pre, post=post, w=w)
    sis.b = previo
    return z


def resolver_pcg(sis, phi=None, tol=None, ciclos=60, pre=2, post=2,
                 niveles=None, w=2.0):
    """Gradiente conjugado precondicionado con el ciclo V. Solo para A simetrica.

    El Poisson de presion lo es (`aE[i] = aW[i+1] = a_cara`, y sobre el corte
    `aC[i] = aC[nx-1-i]`), la conveccion no. Hace falta porque la aglomeracion
    sin suavizar topa en un factor de ~0.5 en difusion pura -- el coeficiente de
    cara gruesa sale x2 y el cociente de Rayleigh recupera la mitad, pero no el
    ancho de banda -- y F4 pide < 0.3. El PCG se come el resto: el ciclo V es un
    precondicionador bueno aunque no sea un solver bueno.
    """
    xp = sis.xp
    niveles = niveles if niveles is not None else jerarquia(sis)
    if phi is None:
        phi = xp.zeros_like(sis.aP)
    tol = tolerancia(sis.aP.dtype, tol)
    escala = max(float(xp.abs(sis.b).max()), 1e-300)
    r = sis.residuo(phi)
    hist = [float(xp.abs(r).max()) / escala]
    if hist[0] <= tol:
        # Sin esto, un termino independiente nulo -- un campo que ya es
        # solenoidal, que es justo el arranque del solver -- da `rz = 0` y el
        # `beta = rz_nuevo/rz` de mas abajo sale 0/0. En CPU no aparecia porque
        # se comprueba la convergencia en cada iteracion y se sale antes; en GPU
        # se comprueba cada dos.
        return phi, {"residuos": hist, "ciclos": 0, "factor": 0.0}
    z = _precondicionar(niveles, r, pre, post, w)
    d = z.copy()
    rz = (r * z).sum()
    # `alfa` y `beta` se quedan en el dispositivo; lo unico que baja a la CPU es
    # la norma del residuo, y solo cada `cada` iteraciones. Con una sincronizacion
    # por iteracion el PCG sobre la malla C tarda el doble.
    cada = 1 if xp is np else 2
    for k in range(ciclos):
        Ad = sis.aplicar(d)
        den = (d * Ad).sum()
        alfa = rz / xp.where(xp.abs(den) > 1e-30, den, 1e-30)
        phi += alfa * d
        r -= alfa * Ad
        if k % cada == cada - 1 or k == ciclos - 1:
            hist.append(float(xp.abs(r).max()) / escala)
            if hist[-1] <= tol:
                break
        z = _precondicionar(niveles, r, pre, post, w)
        rz_nuevo = (r * z).sum()
        d = z + (rz_nuevo / xp.where(xp.abs(rz) > 1e-30, rz, 1e-30)) * d
        rz = rz_nuevo
    utiles = [b / a for a, b in zip(hist[:-1], hist[1:]) if a > 0.0]
    return phi, {"residuos": hist, "ciclos": len(hist) - 1,
                 "factor": float(np.median(utiles)) if utiles else 0.0}

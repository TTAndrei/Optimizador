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


// -- reduccion ciclica paralela ---------------------------------------------
// `n` incognitas en memoria compartida con doble buffer. Los hilos con `i >= n`
// **no se pueden ir**: tienen que llegar a los `__syncthreads()` igual que los
// demas, asi que participan en las barreras y se callan en los accesos.
__device__ __forceinline__ void pcr1({T}* sa, {T}* sb, {T}* sc, {T}* sd,
                                     {T}* ta, {T}* tb, {T}* tc, {T}* td,
                                     const int n, const int i)
{
    for (int p = 1; p < n; p <<= 1) {
        if (i < n) {
            int i0 = i - p, i1 = i + p;
            {T} alfa = (i0 >= 0) ? -sa[i] / sb[i0] : ({T})0;
            {T} beta = (i1 < n) ? -sc[i] / sb[i1] : ({T})0;
            {T} nb = sb[i], nd = sd[i], na = 0, nc = 0;
            if (i0 >= 0) { nb += alfa * sc[i0]; nd += alfa * sd[i0]; na = alfa * sa[i0]; }
            if (i1 < n)  { nb += beta * sa[i1]; nd += beta * sd[i1]; nc = beta * sc[i1]; }
            ta[i] = na; tb[i] = nb; tc[i] = nc; td[i] = nd;
        }
        __syncthreads();
        if (i < n) { sa[i] = ta[i]; sb[i] = tb[i]; sc[i] = tc[i]; sd[i] = td[i]; }
        __syncthreads();
    }
}

// Igual, con dos terminos independientes que comparten matriz: los factores
// `alfa` y `beta` salen de los coeficientes, asi que se calculan una vez.
__device__ __forceinline__ void pcr2({T}* sa, {T}* sb, {T}* sc,
                                     {T}* sd0, {T}* sd1,
                                     {T}* ta, {T}* tb, {T}* tc,
                                     {T}* td0, {T}* td1,
                                     const int n, const int i)
{
    for (int p = 1; p < n; p <<= 1) {
        if (i < n) {
            int i0 = i - p, i1 = i + p;
            {T} alfa = (i0 >= 0) ? -sa[i] / sb[i0] : ({T})0;
            {T} beta = (i1 < n) ? -sc[i] / sb[i1] : ({T})0;
            {T} nb = sb[i], nd0 = sd0[i], nd1 = sd1[i], na = 0, nc = 0;
            if (i0 >= 0) {
                nb += alfa * sc[i0]; na = alfa * sa[i0];
                nd0 += alfa * sd0[i0]; nd1 += alfa * sd1[i0];
            }
            if (i1 < n) {
                nb += beta * sa[i1]; nc = beta * sc[i1];
                nd0 += beta * sd0[i1]; nd1 += beta * sd1[i1];
            }
            ta[i] = na; tb[i] = nb; tc[i] = nc; td0[i] = nd0; td1[i] = nd1;
        }
        __syncthreads();
        if (i < n) {
            sa[i] = ta[i]; sb[i] = tb[i]; sc[i] = tc[i];
            sd0[i] = td0[i]; sd1[i] = td1[i];
        }
        __syncthreads();
    }
}


// -- lineas xi por reduccion ciclica paralela -------------------------------
// Un hilo por linea deja el nivel fino con 49 hilos trabajando de los 6144
// nucleos, cada uno arrastrando una recursion de 384 pasos en serie: medido con
// nsys, 143 us por barrido cuando un recorrido completo de la malla (el producto
// matriz-vector `aplicar`) cuesta 1.45 us. Aqui va **un bloque por linea** y un
// hilo por incognita, con reduccion ciclica paralela: log2(n) pasos en los que
// todos los hilos trabajan.
//
// PCR es estable sin pivoteo en matrices con diagonal dominante, que es lo que
// la construccion garantiza (M-matriz: upwind de 1.er orden mas la parte
// ortogonal de la difusion). Resuelve **la misma** tridiagonal que Thomas: lo
// unico que cambia es el orden de las operaciones, y con el float32 del solver
// eso son diferencias de redondeo.
//
// Los coeficientes van en memoria compartida con doble buffer (8 arrays de
// `n` flotantes = 12 KB con n = 384, de los 48 KB por bloque).
extern "C" __global__
void zebra_xi_pcr({T}* phi, const {T}* aP, const {T}* aW, const {T}* aE,
                  const {T}* aS, const {T}* aN, const {T}* aC, const {T}* b,
                  const int* filas, const int ny, const int nx,
                  const int hay_corte)
{
    extern __shared__ char bruto[];
    {T}* sa = ({T}*)bruto;
    {T}* sb = sa + nx;  {T}* sc = sb + nx;  {T}* sd = sc + nx;
    {T}* ta = sd + nx;
    {T}* tb = ta + nx;  {T}* tc = tb + nx;  {T}* td = tc + nx;

    int i = threadIdx.x;
    int f = filas[blockIdx.x];
    if (i < nx) {
        int o = f * nx + i;
        // El termino retrasado se lee ANTES de tocar `phi`, igual que en la
        // version de Thomas: en la fila del corte la celda espejo esta en esta
        // misma linea.
        {T} r = b[o];
        if (f > 0)      r += aS[o] * phi[o - nx];
        if (f < ny - 1) r += aN[o] * phi[o + nx];
        if (hay_corte && f == 0) r += aC[i] * phi[nx - 1 - i];
        sa[i] = -aW[o];  sb[i] = aP[o];  sc[i] = -aE[o];  sd[i] = r;
    }
    __syncthreads();
    pcr1(sa, sb, sc, sd, ta, tb, tc, td, nx, i);
    if (i < nx) phi[f * nx + i] = sd[i] / sb[i];
}


// Lineas eta: la linea es la columna `i`, con paso `nx` entre incognitas.
extern "C" __global__
void zebra_eta_pcr({T}* phi, const {T}* aP, const {T}* aW, const {T}* aE,
                   const {T}* aS, const {T}* aN, const {T}* b,
                   const int* cols, const int ny, const int nx)
{
    extern __shared__ char bruto[];
    {T}* sa = ({T}*)bruto;
    {T}* sb = sa + ny;  {T}* sc = sb + ny;  {T}* sd = sc + ny;
    {T}* ta = sd + ny;
    {T}* tb = ta + ny;  {T}* tc = tb + ny;  {T}* td = tc + ny;

    int k = threadIdx.x;
    int i = cols[blockIdx.x];
    if (k < ny) {
        int o = k * nx + i;
        {T} r = b[o];
        if (i > 0)      r += aW[o] * phi[o - 1];
        if (i < nx - 1) r += aE[o] * phi[o + 1];
        sa[k] = -aS[o];  sb[k] = aP[o];  sc[k] = -aN[o];  sd[k] = r;
    }
    __syncthreads();
    pcr1(sa, sb, sc, sd, ta, tb, tc, td, ny, k);
    if (k < ny) phi[k * nx + i] = sd[k] / sb[k];
}


// Lineas eta emparejadas por el corte de estela: baja por la columna `ia` y
// sube por su espejo `ib`, una tridiagonal de longitud `2*ny`. Mismo recorrido
// que `zebra_eta_par`.
extern "C" __global__
void zebra_eta_par_pcr({T}* phi, const {T}* aP, const {T}* aW, const {T}* aE,
                       const {T}* aS, const {T}* aN, const {T}* aC,
                       const {T}* b, const int* cols,
                       const int ny, const int nx)
{
    extern __shared__ char bruto[];
    int n = 2 * ny;
    {T}* sa = ({T}*)bruto;
    {T}* sb = sa + n;  {T}* sc = sb + n;  {T}* sd = sc + n;
    {T}* ta = sd + n;
    {T}* tb = ta + n;  {T}* tc = tb + n;  {T}* td = tc + n;

    int k = threadIdx.x;
    int ia = cols[blockIdx.x];
    int ib = nx - 1 - ia;
    int fila = 0, i = 0, o = 0;
    if (k < n) {
        fila = (k < ny) ? (ny - 1 - k) : (k - ny);
        i    = (k < ny) ? ia : ib;
        o    = fila * nx + i;
        {T} sub, sup;
        if (k < ny) {
            sub = -aN[o];
            sup = (k == ny - 1) ? -aC[ia] : -aS[o];
        } else {
            sub = (fila == 0) ? -aC[ib] : -aS[o];
            sup = -aN[o];
        }
        {T} r = b[o];
        if (i > 0)      r += aW[o] * phi[o - 1];
        if (i < nx - 1) r += aE[o] * phi[o + 1];
        sa[k] = sub;  sb[k] = aP[o];  sc[k] = sup;  sd[k] = r;
    }
    __syncthreads();
    pcr1(sa, sb, sc, sd, ta, tb, tc, td, n, k);
    if (k < n) phi[o] = sd[k] / sb[k];
}


// Los dos gemelos de dos campos, para el momento.
extern "C" __global__
void zebra_eta_pcr2({T}* phi, const {T}* aP, const {T}* aW, const {T}* aE,
                    const {T}* aS, const {T}* aN, const {T}* b,
                    const int* cols, const int ny, const int nx)
{
    extern __shared__ char bruto[];
    {T}* sa = ({T}*)bruto;
    {T}* sb = sa + ny;   {T}* sc = sb + ny;
    {T}* sd0 = sc + ny;  {T}* sd1 = sd0 + ny;
    {T}* ta = sd1 + ny;
    {T}* tb = ta + ny;   {T}* tc = tb + ny;
    {T}* td0 = tc + ny;  {T}* td1 = td0 + ny;

    int k = threadIdx.x;
    int i = cols[blockIdx.x];
    int N = ny * nx;
    if (k < ny) {
        int o = k * nx + i;
        {T} r0 = b[o], r1 = b[N + o];
        if (i > 0)      { r0 += aW[o] * phi[o - 1];      r1 += aW[o] * phi[N + o - 1]; }
        if (i < nx - 1) { r0 += aE[o] * phi[o + 1];      r1 += aE[o] * phi[N + o + 1]; }
        sa[k] = -aS[o];  sb[k] = aP[o];  sc[k] = -aN[o];
        sd0[k] = r0;     sd1[k] = r1;
    }
    __syncthreads();
    pcr2(sa, sb, sc, sd0, sd1, ta, tb, tc, td0, td1, ny, k);
    if (k < ny) {
        int o = k * nx + i;
        phi[o] = sd0[k] / sb[k];
        phi[N + o] = sd1[k] / sb[k];
    }
}

extern "C" __global__
void zebra_eta_par_pcr2({T}* phi, const {T}* aP, const {T}* aW, const {T}* aE,
                        const {T}* aS, const {T}* aN, const {T}* aC,
                        const {T}* b, const int* cols,
                        const int ny, const int nx)
{
    extern __shared__ char bruto[];
    int n = 2 * ny;
    {T}* sa = ({T}*)bruto;
    {T}* sb = sa + n;    {T}* sc = sb + n;
    {T}* sd0 = sc + n;   {T}* sd1 = sd0 + n;
    {T}* ta = sd1 + n;
    {T}* tb = ta + n;    {T}* tc = tb + n;
    {T}* td0 = tc + n;   {T}* td1 = td0 + n;

    int k = threadIdx.x;
    int ia = cols[blockIdx.x];
    int ib = nx - 1 - ia;
    int N = ny * nx;
    int fila = 0, i = 0, o = 0;
    if (k < n) {
        fila = (k < ny) ? (ny - 1 - k) : (k - ny);
        i    = (k < ny) ? ia : ib;
        o    = fila * nx + i;
        {T} sub, sup;
        if (k < ny) {
            sub = -aN[o];
            sup = (k == ny - 1) ? -aC[ia] : -aS[o];
        } else {
            sub = (fila == 0) ? -aC[ib] : -aS[o];
            sup = -aN[o];
        }
        {T} r0 = b[o], r1 = b[N + o];
        if (i > 0)      { r0 += aW[o] * phi[o - 1];  r1 += aW[o] * phi[N + o - 1]; }
        if (i < nx - 1) { r0 += aE[o] * phi[o + 1];  r1 += aE[o] * phi[N + o + 1]; }
        sa[k] = sub;  sb[k] = aP[o];  sc[k] = sup;  sd0[k] = r0;  sd1[k] = r1;
    }
    __syncthreads();
    pcr2(sa, sb, sc, sd0, sd1, ta, tb, tc, td0, td1, n, k);
    if (k < n) {
        phi[o] = sd0[k] / sb[k];
        phi[N + o] = sd1[k] / sb[k];
    }
}


// -- gemelos de dos campos --------------------------------------------------
// `phi` y `b` son (2, ny, nx) contiguos; los coeficientes se comparten y se leen
// una sola vez. Los coeficientes de la recursion (`cp`) no dependen del termino
// independiente, asi que tambien se calculan una vez: solo `dp` va por campo.
//
// Los dos campos van en escalares `d0`, `d1`, no en un `d[nc]` recorrido por un
// bucle: con el limite en tiempo de ejecucion el array no cabe en registros, se
// va a memoria local y el kernel sale **mas lento que las dos llamadas en serie**
// (medido: 0.73x). Con escalares, 36 registros por hilo, cero memoria local, y
// 1.76x. Por eso son kernels gemelos y no una generalizacion a `nc` campos.

extern "C" __global__
void zebra_eta2({T}* phi, const {T}* aP, const {T}* aW, const {T}* aE,
                const {T}* aS, const {T}* aN, const {T}* b,
                const int* cols, {T}* cp, {T}* dp,
                const int L, const int ny, const int nx)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= L) return;
    int i = cols[t];
    int N = ny * nx;

    {T} c = 0, d0 = 0, d1 = 0;
    for (int k = 0; k < ny; ++k) {
        int o = k * nx + i;
        {T} aSo = aS[o], aWo = aW[o], aEo = aE[o];
        {T} inv = ({T})1 / (aP[o] + aSo * c);
        {T} r0 = b[o], r1 = b[N + o];
        if (i > 0)      { r0 += aWo * phi[o - 1];  r1 += aWo * phi[N + o - 1]; }
        if (i < nx - 1) { r0 += aEo * phi[o + 1];  r1 += aEo * phi[N + o + 1]; }
        d0 = (r0 + aSo * d0) * inv;
        d1 = (r1 + aSo * d1) * inv;
        c = -aN[o] * inv;
        int q = (k * L + t) * 2;
        cp[k * L + t] = c;
        dp[q] = d0; dp[q + 1] = d1;
    }
    phi[(ny - 1) * nx + i] = d0;
    phi[N + (ny - 1) * nx + i] = d1;
    for (int k = ny - 2; k >= 0; --k) {
        {T} cc = cp[k * L + t];
        int q = (k * L + t) * 2;
        d0 = dp[q] - cc * d0;
        d1 = dp[q + 1] - cc * d1;
        phi[k * nx + i] = d0;
        phi[N + k * nx + i] = d1;
    }
}

extern "C" __global__
void zebra_eta_par2({T}* phi, const {T}* aP, const {T}* aW, const {T}* aE,
                    const {T}* aS, const {T}* aN, const {T}* aC, const {T}* b,
                    const int* cols, {T}* cp, {T}* dp,
                    const int L, const int ny, const int nx)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= L) return;
    int ia = cols[t];
    int ib = nx - 1 - ia;
    int n = 2 * ny;
    int N = ny * nx;

    {T} c = 0, d0 = 0, d1 = 0;
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
        {T} aWo = aW[o], aEo = aE[o];
        {T} inv = ({T})1 / (aP[o] - sub * c);
        {T} r0 = b[o], r1 = b[N + o];
        if (i > 0)      { r0 += aWo * phi[o - 1];  r1 += aWo * phi[N + o - 1]; }
        if (i < nx - 1) { r0 += aEo * phi[o + 1];  r1 += aEo * phi[N + o + 1]; }
        d0 = (r0 - sub * d0) * inv;
        d1 = (r1 - sub * d1) * inv;
        c = sup * inv;
        int q = (k * L + t) * 2;
        cp[k * L + t] = c;
        dp[q] = d0; dp[q + 1] = d1;
    }
    phi[(n - 1 - ny) * nx + ib] = d0;       // ultima incognita: fila ny-1, col ib
    phi[N + (n - 1 - ny) * nx + ib] = d1;
    for (int k = n - 2; k >= 0; --k) {
        {T} cc = cp[k * L + t];
        int q = (k * L + t) * 2;
        d0 = dp[q] - cc * d0;
        d1 = dp[q + 1] - cc * d1;
        int fila = (k < ny) ? (ny - 1 - k) : (k - ny);
        int i    = (k < ny) ? ia : ib;
        phi[fila * nx + i] = d0;
        phi[N + fila * nx + i] = d1;
    }
}

extern "C" __global__
void zebra_xi2({T}* phi, const {T}* aP, const {T}* aW, const {T}* aE,
               const {T}* aS, const {T}* aN, const {T}* aC, const {T}* b,
               const int* filas, {T}* cp, {T}* dp,
               const int L, const int ny, const int nx, const int hay_corte)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= L) return;
    int f = filas[t];
    int N = ny * nx;

    {T} c = 0, d0 = 0, d1 = 0;
    for (int k = 0; k < nx; ++k) {
        int o = f * nx + k;
        {T} aWo = aW[o], aSo = aS[o], aNo = aN[o];
        {T} inv = ({T})1 / (aP[o] + aWo * c);
        {T} r0 = b[o], r1 = b[N + o];
        if (f > 0)      { r0 += aSo * phi[o - nx];  r1 += aSo * phi[N + o - nx]; }
        if (f < ny - 1) { r0 += aNo * phi[o + nx];  r1 += aNo * phi[N + o + nx]; }
        if (hay_corte && f == 0) {
            {T} aCk = aC[k];
            r0 += aCk * phi[nx - 1 - k];
            r1 += aCk * phi[N + nx - 1 - k];
        }
        d0 = (r0 + aWo * d0) * inv;
        d1 = (r1 + aWo * d1) * inv;
        c = -aE[o] * inv;
        int q = (k * L + t) * 2;
        cp[k * L + t] = c;
        dp[q] = d0; dp[q + 1] = d1;
    }
    phi[f * nx + nx - 1] = d0;
    phi[N + f * nx + nx - 1] = d1;
    for (int k = nx - 2; k >= 0; --k) {
        {T} cc = cp[k * L + t];
        int q = (k * L + t) * 2;
        d0 = dp[q] - cc * d0;
        d1 = dp[q + 1] - cc * d1;
        phi[f * nx + k] = d0;
        phi[N + f * nx + k] = d1;
    }
}

extern "C" __global__
void aplicar2({T}* r, const {T}* phi, const {T}* aP, const {T}* aW, const {T}* aE,
              const {T}* aS, const {T}* aN, const {T}* aC, const {T}* b,
              const int ny, const int nx, const int hay_corte, const int residuo)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int N = ny * nx;
    if (o >= N) return;
    int i = o % nx;
    int j = o / nx;
    {T} aPo = aP[o];
    {T} v0 = aPo * phi[o], v1 = aPo * phi[N + o];
    if (i > 0)      { {T} a = aW[o]; v0 -= a * phi[o - 1];  v1 -= a * phi[N + o - 1]; }
    if (i < nx - 1) { {T} a = aE[o]; v0 -= a * phi[o + 1];  v1 -= a * phi[N + o + 1]; }
    if (j > 0)      { {T} a = aS[o]; v0 -= a * phi[o - nx]; v1 -= a * phi[N + o - nx]; }
    if (j < ny - 1) { {T} a = aN[o]; v0 -= a * phi[o + nx]; v1 -= a * phi[N + o + nx]; }
    if (hay_corte && j == 0) {
        {T} a = aC[i];
        v0 -= a * phi[nx - 1 - i];
        v1 -= a * phi[N + nx - 1 - i];
    }
    r[o]     = residuo ? (b[o] - v0)     : v0;
    r[N + o] = residuo ? (b[N + o] - v1) : v1;
}

extern "C" __global__
void restringir2({T}* rc, const {T}* r,
                 const int* jini, const int* jfin, const int* iini, const int* ifin,
                 const int ncy, const int ncx, const int ny, const int nx)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int Nc = ncy * ncx;
    if (o >= Nc) return;
    int I = o % ncx;
    int J = o / ncx;
    int N = ny * nx;
    {T} s0 = 0, s1 = 0;
    for (int j = jini[J]; j <= jfin[J]; ++j)
        for (int i = iini[I]; i <= ifin[I]; ++i) {
            s0 += r[j * nx + i];
            s1 += r[N + j * nx + i];
        }
    rc[o] = s0;
    rc[Nc + o] = s1;
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

# Longitud minima de linea a partir de la cual compensa la reduccion ciclica
# paralela (un bloque por linea) en vez de Thomas (un hilo por linea). Por
# debajo, la linea es corta, hay muchas y el problema deja de ser de paralelismo.
PCR_MINIMO = 64


def _rejilla(n):
    return ((n + _HILOS - 1) // _HILOS,), (_HILOS,)


def _pcr(n):
    """Si la linea de `n` incognitas va por reduccion ciclica paralela."""
    return PCR_MINIMO <= n <= 1024


def _bloque(n):
    """Hilos por bloque para una linea de `n`: warps enteros, `n` redondeado."""
    return int((n + 31) // 32 * 32)


class Sistema:
    """Coeficientes de 5 puntos mas el acoplamiento del corte de estela.

    Formas: todo `(ny, nx)` salvo `aC`, que es `(nx,)` y solo actua en `j=0`.
    Los coeficientes de vecino valen cero donde no hay vecino.
    """

    def __init__(self, aP, aW, aE, aS, aN, b, aC=None, activo=None, xi=True):
        self.aP, self.aW, self.aE, self.aS, self.aN, self.b = aP, aW, aE, aS, aN, b
        self.aC = aC
        self.xi = xi
        self.ny, self.nx = aP.shape
        self.xp = xp_de(aP)
        self.gpu = self.xp is not np
        self._indices(activo)
        if self.gpu:
            mod = _modulo(aP.dtype)
            self._k_eta = mod.get_function("zebra_eta")
            self._k_par = mod.get_function("zebra_eta_par")
            self._k_xi = mod.get_function("zebra_xi")
            self._k_xi_pcr = mod.get_function("zebra_xi_pcr")
            self._k_eta_pcr = mod.get_function("zebra_eta_pcr")
            self._k_par_pcr = mod.get_function("zebra_eta_par_pcr")
            self._k_eta_pcr2 = mod.get_function("zebra_eta_pcr2")
            self._k_par_pcr2 = mod.get_function("zebra_eta_par_pcr2")
            self._k_apl = mod.get_function("aplicar")
            self._k_eta2 = mod.get_function("zebra_eta2")
            self._k_par2 = mod.get_function("zebra_eta_par2")
            self._k_xi2 = mod.get_function("zebra_xi2")
            self._k_apl2 = mod.get_function("aplicar2")
            self._cero = self.xp.zeros(self.nx, dtype=aP.dtype)
            self._rasca = {}
            # Argumentos constantes del lanzamiento, ya convertidos. Construir
            # los `np.int32` y calcular bloques y memoria compartida en cada
            # barrido es trabajo de CPU en el camino caliente, y con ~2400
            # lanzamientos por paso de tiempo el cuello es justo ese.
            self._i_ny = np.int32(self.ny)
            self._i_nx = np.int32(self.nx)
            self._i_corte = np.int32(self.aC is not None)
            self._aC_o_cero = self._cero if self.aC is None else self.aC
            octetos = aP.dtype.itemsize
            self._pcr_eta = (_pcr(self.ny), (_bloque(self.ny),),
                             8 * self.ny * octetos, 10 * self.ny * octetos)
            n2 = 2 * self.ny
            self._pcr_par = (_pcr(n2), (_bloque(n2),),
                             8 * n2 * octetos, 10 * n2 * octetos)
            self._pcr_xi = (_pcr(self.nx), (_bloque(self.nx),),
                            8 * self.nx * octetos, 10 * self.nx * octetos)

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
        # Rejillas de "un bloque por linea" (columnas sueltas, pares del corte y
        # filas), una por paridad.
        self._rej = [((self._col[k].size,), (self._par[k].size,),
                      (self._fil[k].size,)) for k in (0, 1)]

    def _scratch(self, n, L, nc=1):
        """`(cp, dp)` de la recursion. `cp` es comun a los campos, `dp` no.

        Los coeficientes `c` de Thomas salen solo de la matriz, asi que con dos
        campos se calculan una vez. `dp` va intercalado por campo (`n, L, nc`):
        las dos escrituras de una celda caen juntas.
        """
        clave = (n, int(L), nc)
        if clave not in self._rasca:
            tipo = self.aP.dtype
            self._rasca[clave] = (self.xp.empty((n, int(L)), dtype=tipo),
                                  self.xp.empty((n, int(L), nc), dtype=tipo))
        return self._rasca[clave]

    def aplicar(self, phi, b=None, residuo=False):
        if self.gpu:
            r = self.xp.empty_like(phi)
            n = self.ny * self.nx
            kern = self._k_apl2 if phi.ndim == 3 else self._k_apl
            kern(*_rejilla(n), (
                r, phi, self.aP, self.aW, self.aE, self.aS, self.aN,
                self._cero if self.aC is None else self.aC,
                self._cero if b is None else b,
                np.int32(self.ny), np.int32(self.nx),
                np.int32(self.aC is not None), np.int32(residuo)))
            return r
        if phi.ndim == 3:                   # en CPU los campos van en bucle
            return self.xp.stack([self.aplicar(x) for x in phi])
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
            nc = phi.shape[0] if phi.ndim == 3 else 1
            c = self._col[paridad]
            if c.size:
                usa_pcr, bloque, sh1, sh2 = self._pcr_eta
                if usa_pcr:
                    kern = self._k_eta_pcr2 if nc == 2 else self._k_eta_pcr
                    kern(self._rej[paridad][0], bloque, (
                        phi, self.aP, self.aW, self.aE, self.aS, self.aN,
                        self.b, c, self._i_ny, self._i_nx),
                        shared_mem=sh2 if nc == 2 else sh1)
                else:
                    cp_, dp = self._scratch(self.ny, c.size, nc)
                    (self._k_eta2 if nc == 2 else self._k_eta)(*_rejilla(c.size), (
                        phi, self.aP, self.aW, self.aE, self.aS, self.aN, self.b,
                        c, cp_, dp, np.int32(c.size), self._i_ny, self._i_nx))
            a = self._par[paridad]
            if a.size:
                usa_pcr, bloque, sh1, sh2 = self._pcr_par
                if usa_pcr:
                    kern = self._k_par_pcr2 if nc == 2 else self._k_par_pcr
                    kern(self._rej[paridad][1], bloque, (
                        phi, self.aP, self.aW, self.aE, self.aS, self.aN,
                        self.aC, self.b, a, self._i_ny, self._i_nx),
                        shared_mem=sh2 if nc == 2 else sh1)
                else:
                    cp_, dp = self._scratch(2 * self.ny, a.size, nc)
                    (self._k_par2 if nc == 2 else self._k_par)(*_rejilla(a.size), (
                        phi, self.aP, self.aW, self.aE, self.aS, self.aN,
                        self.aC, self.b, a, cp_, dp, np.int32(a.size),
                        self._i_ny, self._i_nx))
            return

        if phi.ndim == 3:                   # en CPU los campos van en bucle
            b_todo = self.b
            for q, x in enumerate(phi):
                self.b = b_todo[q]
                self._lineas_eta(x, paridad)
            self.b = b_todo
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
            nc = phi.shape[0] if phi.ndim == 3 else 1
            usa_pcr, bloque, sh1, _ = self._pcr_xi
            if nc == 1 and usa_pcr:
                # Un bloque por linea, un hilo por incognita. La memoria
                # compartida son los ocho arrays del doble buffer de la PCR.
                self._k_xi_pcr(self._rej[paridad][2], bloque, (
                    phi, self.aP, self.aW, self.aE, self.aS, self.aN,
                    self._aC_o_cero, self.b, f, self._i_ny, self._i_nx,
                    self._i_corte), shared_mem=sh1)
                return
            cp_, dp = self._scratch(self.nx, f.size, nc)
            (self._k_xi2 if nc == 2 else self._k_xi)(*_rejilla(f.size), (
                phi, self.aP, self.aW, self.aE, self.aS, self.aN,
                self._aC_o_cero, self.b, f, cp_, dp, np.int32(f.size),
                self._i_ny, self._i_nx, self._i_corte))
            return
        if phi.ndim == 3:
            b_todo = self.b
            for q, x in enumerate(phi):
                self.b = b_todo[q]
                self._lineas_xi(x, paridad)
            self.b = b_todo
            return
        d = (self.b + self._retrasado_eta(phi))[f, :].T
        phi[f, :] = _thomas(-self.aW[f, :].T, self.aP[f, :].T,
                            -self.aE[f, :].T, d).T

    def suavizar(self, phi, veces=1, invertido=False):
        """`invertido` recorre los barridos al reves.

        Hace falta para que el ciclo V sea un operador **simetrico**, que es lo
        que el PCG exige del precondicionador: pre-suavizado en un orden y
        post-suavizado en el contrario. Con `xi = False` son dos barridos en vez
        de cuatro, y la simetria se conserva igual.

        **`xi = False` solo vale para la conveccion.** Medido sobre las matrices
        reales del caso v3 con termino independiente aleatorio: con lineas eta
        solas y tres barridos, el momento deja el residuo en 3.8e-5 tras dos
        ciclos frente a 1.2e-4 del ADI, y cuesta 2.90 ms por ciclo en vez de
        4.57 -- mejor y 1.58 veces mas barato, porque una linea xi son 384
        incognitas en serie por hilo y solo hay 49 lineas por paridad. En el
        Poisson de presion es al reves y por goleada: el factor del PCG pasa de
        **0.031 a 0.93** (la puerta de F4 es 0.3), que es lo que cabe esperar de
        un operador isotropo en el campo lejano.
        """
        orden = [(self._lineas_eta, 0), (self._lineas_eta, 1)]
        if self.xi:
            orden += [(self._lineas_xi, 0), (self._lineas_xi, 1)]
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
    # `activo` vacio a proposito: quien sabe que columnas gruesas tienen corte es
    # `jerarquia`, que lo deduce del mapa y llama a `_indices` acto seguido. Sin
    # esto, el constructor lo deduciria de `aC`, y eso es **bajar `aC` a la CPU**:
    # medido con nsys, 14 sincronizaciones bloqueantes por paso de tiempo (una
    # por nivel y matriz) de las 16 que quedaban.
    return Sistema(aP, aW, aE, aS, aN, xp.zeros(forma, dtype=tipo),
                   aCc if sis.aC is not None else None,
                   activo=np.zeros(ncx, bool), xi=sis.xi)


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
                   activo=activo, xi=sis.xi)


# La jerarquia se para cuando el nivel baja de `1/RAZON_GRUESA` de las celdas
# finas, no en un numero absoluto de celdas: lo que fija el factor de
# convergencia es la **razon** entre el nivel mas grueso y el fino, asi que un
# numero fijo daria factores distintos segun la malla.
#
# Recortar niveles es tentador desde que el suavizador va por reduccion ciclica:
# cada nivel cuesta una tanda de lanzamientos y el cuello es ese. Pero se paga
# en escalabilidad. Medido sobre el cuadrado distorsionado de los tests (factor
# del ciclo V de conveccion / factor del PCG de presion):
#
#     razon      n=64           n=128          n=256
#     1/16       0.477 / 0.263  0.636 / 0.499  1.033 / 0.721
#     1/64       0.474 / 0.096  0.494 / 0.310  0.641 / 0.589
#     1/256      0.474 / 0.055  0.494 / 0.099  0.499 / 0.248
#     1/1024     0.474 / 0.045  0.494 / 0.073  0.499 / 0.117
#     completa   0.474 / 0.045  0.494 / 0.045  0.499 / 0.066
#
# Las puertas son 0.65 para el ciclo V (`test_el_multigrid_no_se_degrada_al_refinar`)
# y 0.3 para el PCG (`test_el_pcg_no_se_degrada_al_refinar`). Con 1/64 las dos se
# salen a n=256; con 1/256 pasan pero el PCG se queda a un 17 % de la puerta y
# empeora claramente al refinar, que es justo la propiedad que el multigrid
# aporta. 1/1024 se queda a un tercio de la puerta y sobre la malla C de
# produccion son 7 niveles en vez de 8: 56.4 -> 54.1 ms/paso. La parada
# siguiente, 1/256, daria 46.6 ms y es una decision de riesgo, no de codigo.
RAZON_GRUESA = 1024


def jerarquia(sis, minimo=None):
    """Lista [(sistema, kj, ki)] de fino a grueso. Los mapas son los del nivel.

    `minimo` es el tamano por debajo del cual se deja de engrosar; por defecto,
    `celdas_finas / RAZON_GRUESA`.
    """
    if minimo is None:
        minimo = max(9, sis.aP.size // RAZON_GRUESA)
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
    nc = phi.shape[0] if phi.ndim == 3 else 1
    if sis.gpu and nc == 2:
        bloq, _ = sc._bloq
        sc.b = xp.empty((2, sc.ny, sc.nx), dtype=phi.dtype)
        _modulo(phi.dtype).get_function("restringir2")(*_rejilla(sc.ny * sc.nx), (
            sc.b, r, *bloq, np.int32(sc.ny), np.int32(sc.nx), np.int32(sis.ny),
            np.int32(sis.nx)))
    elif sis.gpu:
        bloq, _ = sc._bloq
        sc.b = xp.empty((sc.ny, sc.nx), dtype=phi.dtype)
        _modulo(phi.dtype).get_function("restringir")(*_rejilla(sc.ny * sc.nx), (
            sc.b, r, *bloq, np.int32(sc.ny), np.int32(sc.nx), np.int32(sis.nx)))
    elif nc == 2:
        idx = kj[:, None] * sc.nx + ki[None, :]
        sc.b = xp.stack([_sumar(xp, idx, x, (sc.ny, sc.nx)) for x in r])
    else:
        sc.b = _sumar(xp, kj[:, None] * sc.nx + ki[None, :], r, (sc.ny, sc.nx))
    e = xp.zeros(phi.shape[:-2] + (sc.ny, sc.nx), dtype=phi.dtype)
    ciclo_v(niveles, e, nivel + 1, pre, post, grueso, w)

    if sis.gpu and w is not None:
        peso = xp.asarray([w], dtype=phi.dtype)
        _modulo(phi.dtype).get_function("prolongar")(*_rejilla(sis.ny * sis.nx), (
            phi, e, kj, ki, peso, np.int32(sis.ny), np.int32(sis.nx),
            np.int32(sc.nx)))
        sis.suavizar(phi, post, invertido=True)
        return phi

    ef = e[..., kj[:, None], ki[None, :]]
    if w is None:
        # El peso se deja como escalar **en el dispositivo**: sacarlo a la CPU
        # con float() obliga a sincronizar dos veces por nivel y por ciclo, que
        # sobre una malla de 21 000 celdas cuesta mas que el propio ciclo. Con
        # dos campos es un peso por campo: el cociente de Rayleigh minimiza el
        # error de **cada** correccion, y compartirlo los estropearia a los dos.
        ejes = (-2, -1)
        den = (ef * sis.aplicar(ef)).sum(axis=ejes)
        num = (r * ef).sum(axis=ejes)
        peso = xp.clip(num / xp.where(xp.abs(den) > 1e-30, den, 1e-30), 0.5, 4.0)
        if nc > 1:
            peso = peso[:, None, None]
    else:
        peso = w
    phi += peso * ef
    sis.suavizar(phi, post, invertido=w is not None)
    return phi


def resolver(sis, phi=None, tol=None, ciclos=60, pre=2, post=2, niveles=None,
             fijos=None):
    """Resuelve A phi = b. Devuelve (phi, info) con el historial de residuos.

    Con `fijos` se hacen **esos ciclos V y no se mira el residuo**. Comprobarlo
    obliga a bajar un escalar del dispositivo, y cada bajada drena la tuberia:
    medido con nsys, 14 sincronizaciones por paso de tiempo y 16.6 ms de espera
    en la API de memcpy. En regimen asentado el criterio no decide nada -- el
    momento siempre gasta 2 ciclos y `nu_tilde` 2 --, asi que quien decide es
    `Solver`, que recalibra el numero cada 50 pasos con un paso comprobado.

    `info["vciclos"]` es el numero de ciclos V gastados. No es `info["ciclos"]`:
    ese cuenta **comprobaciones**, y en GPU cada una cubre dos ciclos.
    """
    xp = sis.xp
    if phi is None:
        phi = xp.zeros_like(sis.aP)
    niveles = niveles if niveles is not None else jerarquia(sis)
    tol = tolerancia(sis.aP.dtype, tol)
    if fijos is not None:
        for _ in range(fijos):
            ciclo_v(niveles, phi, pre=pre, post=post)
        return phi, {"residuos": [], "ciclos": fijos, "vciclos": fijos,
                     "factor": 0.0}
    # Con varios campos cada uno se normaliza por **su** termino independiente y
    # se para cuando todos cumplen su propio criterio. Normalizar los dos por el
    # maximo comun relajaria al pequeño: a alfa = 5 grados el termino de `v` es
    # un orden menor que el de `u` y se daria por convergido antes de tiempo.
    ejes = (-2, -1)
    escala = xp.maximum(xp.abs(sis.b).max(axis=ejes), 1e-30)
    def resid():
        return float((xp.abs(sis.residuo(phi)).max(axis=ejes) / escala).max())
    hist = [resid()]
    cada = 1 if xp is np else 2
    k = -1
    for k in range(ciclos):
        ciclo_v(niveles, phi, pre=pre, post=post)
        if k % cada == cada - 1 or k == ciclos - 1:
            hist.append(resid())
            if hist[-1] <= tol:
                break
    utiles = [b / a for a, b in zip(hist[:-1], hist[1:]) if a > 0.0]
    return phi, {"residuos": hist, "ciclos": len(hist) - 1, "vciclos": k + 1,
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
                 niveles=None, w=2.0, fijos=None):
    """Gradiente conjugado precondicionado con el ciclo V. Solo para A simetrica.

    El Poisson de presion lo es (`aE[i] = aW[i+1] = a_cara`, y sobre el corte
    `aC[i] = aC[nx-1-i]`), la conveccion no. Hace falta porque la aglomeracion
    sin suavizar topa en un factor de ~0.5 en difusion pura -- el coeficiente de
    cara gruesa sale x2 y el cociente de Rayleigh recupera la mitad, pero no el
    ancho de banda -- y F4 pide < 0.3. El PCG se come el resto: el ciclo V es un
    precondicionador bueno aunque no sea un solver bueno.

    `fijos` son iteraciones a cuenta fija, sin mirar el residuo: ver `resolver`.
    Con `fijos = 0` no se hace nada, que es lo que pasa en la segunda correccion
    cruzada de cada paso -- el incremento de presion ya viene convergido.
    """
    xp = sis.xp
    niveles = niveles if niveles is not None else jerarquia(sis)
    if phi is None:
        phi = xp.zeros_like(sis.aP)
    tol = tolerancia(sis.aP.dtype, tol)
    if fijos == 0:
        return phi, {"residuos": [], "ciclos": 0, "vciclos": 0, "factor": 0.0}
    comprobar = fijos is None
    tope = ciclos if comprobar else fijos
    hist = []
    if comprobar:
        escala = max(float(xp.abs(sis.b).max()), 1e-300)
    r = sis.residuo(phi)
    if comprobar:
        hist.append(float(xp.abs(r).max()) / escala)
    if comprobar and hist[0] <= tol:
        # Sin esto, un termino independiente nulo -- un campo que ya es
        # solenoidal, que es justo el arranque del solver -- da `rz = 0` y el
        # `beta = rz_nuevo/rz` de mas abajo sale 0/0. En CPU no aparecia porque
        # se comprueba la convergencia en cada iteracion y se sale antes; en GPU
        # se comprueba cada dos.
        return phi, {"residuos": hist, "ciclos": 0, "vciclos": 0, "factor": 0.0}
    z = _precondicionar(niveles, r, pre, post, w)
    d = z.copy()
    rz = (r * z).sum()
    # `alfa` y `beta` se quedan en el dispositivo; lo unico que baja a la CPU es
    # la norma del residuo, y solo cada `cada` iteraciones. Con una sincronizacion
    # por iteracion el PCG sobre la malla C tarda el doble.
    cada = 1 if xp is np else 2
    v = 1                                       # ciclos V gastados, el de `z`
    for k in range(tope):
        Ad = sis.aplicar(d)
        den = (d * Ad).sum()
        alfa = rz / xp.where(xp.abs(den) > 1e-30, den, 1e-30)
        phi += alfa * d
        r -= alfa * Ad
        if comprobar and (k % cada == cada - 1 or k == tope - 1):
            hist.append(float(xp.abs(r).max()) / escala)
            if hist[-1] <= tol:
                break
        if not comprobar and k == tope - 1:
            break                               # la ultima direccion no se usa
        z = _precondicionar(niveles, r, pre, post, w)
        rz_nuevo = (r * z).sum()
        d = z + (rz_nuevo / xp.where(xp.abs(rz) > 1e-30, rz, 1e-30)) * d
        rz = rz_nuevo
        v += 1
    utiles = [b / a for a, b in zip(hist[:-1], hist[1:]) if a > 0.0]
    return phi, {"residuos": hist, "ciclos": len(hist) - 1 if comprobar else tope,
                 "vciclos": v,
                 "factor": float(np.median(utiles)) if utiles else 0.0}

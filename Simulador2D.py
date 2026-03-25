import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cupy as cp
from cupyx.scipy.ndimage import distance_transform_edt
import os
import json
from tqdm import tqdm
import signal
import sys
from datetime import datetime
from multiprocessing import shared_memory
import struct


# ============================================================
# GENERACIÓN DE MALLA CON DENSIDAD VARIABLE (stretching 1D)
# ============================================================

def generar_malla_estirada(L, x_centro, dx_min, factor_expansion=1.05,
                           ancho_zona_fina=None, dx_max=None):
    """
    Genera un vector 1D de posiciones de nodo de 0 a L con
    espaciado variable:  fino (dx_min) alrededor de x_centro,
    expandiéndose geométricamente hacia los extremos del dominio.

    Parámetros
    ----------
    L              : float  – longitud total del dominio.
    x_centro       : float  – centro de la zona de refinamiento.
    dx_min         : float  – espaciado mínimo (en la zona fina).
    factor_expansion : float  – factor geométrico de crecimiento (>1).
    ancho_zona_fina  : float | None – ancho de la banda con dx_min uniforme.
                       Si None se usa 0.1*L.
    dx_max         : float | None – espaciado máximo (caps la expansión).
                       Si None se usa 20*dx_min.

    Retorna
    -------
    X_1d : np.ndarray (float64) – posiciones ordenadas, X_1d[0]=0, X_1d[-1]=L.
    """
    if ancho_zona_fina is None:
        ancho_zona_fina = 0.1 * L
    if dx_max is None:
        dx_max = 20.0 * dx_min

    # Límites de la zona fina (clamp al dominio)
    x_fino_min = max(0.0, x_centro - ancho_zona_fina / 2.0)
    x_fino_max = min(L,   x_centro + ancho_zona_fina / 2.0)

    # --- Zona fina: nodos uniformes con dx_min ---
    n_fino = max(1, int(round((x_fino_max - x_fino_min) / dx_min)))
    x_fino = np.linspace(x_fino_min, x_fino_max, n_fino + 1)

    # --- Expansión izquierda (de x_fino_min hacia 0) ---
    x_left = []
    x = x_fino_min
    dx = dx_min
    while x > 0.0:
        dx = min(dx * factor_expansion, dx_max)
        x = x - dx
        if x <= 0.0:
            x = 0.0
        x_left.append(x)
    x_left = np.array(x_left[::-1])  # ascendente

    # --- Expansión derecha (de x_fino_max hacia L) ---
    x_right = []
    x = x_fino_max
    dx = dx_min
    while x < L:
        dx = min(dx * factor_expansion, dx_max)
        x = x + dx
        if x >= L:
            x = L
        x_right.append(x)
    x_right = np.array(x_right)

    # --- Combinar y eliminar duplicados ---
    X_1d = np.concatenate([x_left, x_fino, x_right])
    X_1d = np.unique(X_1d)          # ordena + elimina duplicados

    # Asegurar extremos exactos
    if X_1d[0] != 0.0:
        X_1d = np.concatenate([[0.0], X_1d])
    if X_1d[-1] != L:
        X_1d = np.concatenate([X_1d, [L]])

    return X_1d.astype(np.float64)


def calcular_metricas_1d(pos_1d_gpu):
    """
    Dada una secuencia monótona de posiciones (CuPy float32, longitud N),
    devuelve las distancias a vecinos y los coeficientes de diferencias
    finitas (primera y segunda derivada) en stencil de 3 puntos para
    malla no‑uniforme.

    Retorna dict con arrays CuPy float32 de longitud N:
        dx_e   – distancia al vecino Este  (padded en borde derecho)
        dx_w   – distancia al vecino Oeste (padded en borde izquierdo)
        d1_W, d1_C, d1_E  – coeficientes ∂/∂x
        d2_W, d2_C, d2_E  – coeficientes ∂²/∂x²
    """
    N = len(pos_1d_gpu)
    dx_e = cp.zeros(N, dtype=cp.float32)
    dx_w = cp.zeros(N, dtype=cp.float32)

    diffs = cp.diff(pos_1d_gpu)          # (N-1,)
    dx_e[:-1] = diffs;  dx_e[-1] = diffs[-1]   # padding borde
    dx_w[1:]  = diffs;  dx_w[0]  = diffs[0]

    he = dx_e
    hw = dx_w
    hsum = he + hw
    eps = cp.float32(1e-30)   # evitar /0 en bordes

    # ---- Primera derivada centrada (no-uniforme) ----
    # u'_j ≈ a_W u_{j-1} + a_C u_j + a_E u_{j+1}
    d1_W = -he / (hw * hsum + eps)
    d1_C = (he - hw) / (he * hw + eps)
    d1_E =  hw / (he * hsum + eps)

    # ---- Segunda derivada (no-uniforme) ----
    # u''_j ≈ b_W u_{j-1} + b_C u_j + b_E u_{j+1}
    d2_W =  cp.float32(2.0) / (hw * hsum + eps)
    d2_C = -cp.float32(2.0) / (he * hw + eps)
    d2_E =  cp.float32(2.0) / (he * hsum + eps)

    return {
        'dx_e': dx_e, 'dx_w': dx_w,
        'd1_W': d1_W, 'd1_C': d1_C, 'd1_E': d1_E,
        'd2_W': d2_W, 'd2_C': d2_C, 'd2_E': d2_E,
    }


class Mesh:
    '''Constructor y basicos'''
    def __init__(self, Lx, Ly, p0, v0x, v0y, dx, dy,
                 usar_wale=True,
                 X_1d=None, Y_1d=None):
        """
        Parámetros
        ----------
        Lx, Ly       : dimensiones del dominio (m).
        p0            : presión inicial.
        v0x, v0y      : velocidad inicial.
        dx, dy        : espaciado de referencia (uniforme si X_1d/Y_1d=None;
                         usado para CFL/estabilidad si se pasan X_1d/Y_1d).
        X_1d, Y_1d    : arrays numpy (float64) con posiciones de nodos en x/y.
                         Si se proporcionan, la malla es de densidad variable.
                         Si None, se genera malla uniforme desde dx/dy.
        """
        self.Lx = Lx
        self.Ly = Ly
        self.usar_wale = usar_wale

        # ================================================================
        # MALLA DE POSICIONES  (nueva infraestructura de malla variable)
        # ================================================================
        if X_1d is not None:
            self.X_1d = cp.asarray(X_1d, dtype=cp.float32)
            self.nx = len(self.X_1d)
            self.malla_variable = True
        else:
            self.nx = int(Lx / dx)
            self.X_1d = cp.arange(self.nx, dtype=cp.float32) * cp.float32(dx)
            self.malla_variable = False

        if Y_1d is not None:
            self.Y_1d = cp.asarray(Y_1d, dtype=cp.float32)
            self.ny = len(self.Y_1d)
        else:
            self.ny = int(Ly / dy)
            self.Y_1d = cp.arange(self.ny, dtype=cp.float32) * cp.float32(dy)

        # dx, dy escalar = espaciado MÍNIMO (para CFL, estabilidad viscosa)
        if self.malla_variable:
            self.dx = float(cp.min(cp.diff(self.X_1d)))
            self.dy = float(cp.min(cp.diff(self.Y_1d)))
        else:
            self.dx = dx
            self.dy = dy

        # ---- Métricas 1D pre‑computadas --------------------------------
        met_x = calcular_metricas_1d(self.X_1d)
        met_y = calcular_metricas_1d(self.Y_1d)

        # Distancias a vecinos (1D, se broadcastean a 2D)
        self.dx_e = met_x['dx_e']   # (nx,)
        self.dx_w = met_x['dx_w']
        self.dy_n = met_y['dx_e']   # "e" = dirección + (norte)
        self.dy_s = met_y['dx_w']   # "w" = dirección - (sur)

        # Coeficientes primera derivada ∂/∂x  (1D, broadcastean a 2D)
        self.d1x_W = met_x['d1_W']
        self.d1x_C = met_x['d1_C']
        self.d1x_E = met_x['d1_E']
        # ∂/∂y
        self.d1y_S = met_y['d1_W']  # "S" ↔ -dirección
        self.d1y_C = met_y['d1_C']
        self.d1y_N = met_y['d1_E']  # "N" ↔ +dirección

        # Coeficientes segunda derivada ∂²/∂x²
        self.d2x_W = met_x['d2_W']
        self.d2x_C = met_x['d2_C']
        self.d2x_E = met_x['d2_E']
        # ∂²/∂y²
        self.d2y_S = met_y['d2_W']
        self.d2y_C = met_y['d2_C']
        self.d2y_N = met_y['d2_E']

        # Volúmenes de celda:  vol_x[j] = (he+hw)/2 ,  vol_y[i] = (hn+hs)/2
        # Sirven para simetrizar el Laplaciano (M = diag(V)·L es simétrico → CG)
        self.vol_x = (met_x['dx_e'] + met_x['dx_w']) * cp.float32(0.5)   # (nx,)
        self.vol_y = (met_y['dx_e'] + met_y['dx_w']) * cp.float32(0.5)   # (ny,)
        self._vol_2d_flat = (self.vol_y[:, cp.newaxis] * self.vol_x[cp.newaxis, :]).ravel()  # (ny*nx,)

        # Mallas 2D de posiciones físicas (para rasterización, visualización, IBM)
        self.XX, self.YY = cp.meshgrid(self.X_1d, self.Y_1d, indexing='xy')

        # --- Campos principales en GPU ---
        self.u = cp.zeros((self.ny, self.nx), dtype=cp.float32)
        self.v = cp.zeros((self.ny, self.nx), dtype=cp.float32)
        self.p = cp.full((self.ny, self.nx), p0, dtype=cp.float32)

        self.dvector = cp.zeros(1, dtype=cp.float32)
        self.lvector = cp.zeros(1, dtype=cp.float32)
        self.cdvector = cp.zeros(1, dtype=cp.float32)
        self.clvector = cp.zeros(1, dtype=cp.float32)
        self.divvector = cp.zeros(1, dtype=cp.float32)
        self.mg_cycles_vector = cp.zeros(1, dtype=cp.int32)
        self.guardado = 1

        # Historial/estadística de Cp a lo largo de la simulación (promedio por punto de cuerda)
        self.cp_bins = 200
        # Separar intradós y extradós para promediar por separado
        self.cp_profile_sum_ex = cp.zeros(self.cp_bins, dtype=cp.float32)
        self.cp_profile_count_ex = cp.zeros(self.cp_bins, dtype=cp.float32)
        self.cp_profile_sum_in = cp.zeros(self.cp_bins, dtype=cp.float32)
        self.cp_profile_count_in = cp.zeros(self.cp_bins, dtype=cp.float32)
        self.cp_profile_x = cp.linspace(0.0, 1.0, self.cp_bins, dtype=cp.float32)

        self.u[:] = v0x
        self.v[:] = v0y
        self.p[:] = p0
        
        # Velocidad de referencia (para clamp en ghost-cell IBM)
        self._vel_ref = float(max(np.sqrt(v0x**2 + v0y**2), 1.0))

        # Precomputar mallas de índices para backtracing
        j = cp.arange(self.nx, dtype=cp.float32)
        i = cp.arange(self.ny, dtype=cp.float32)
        self.JJ, self.II = cp.meshgrid(j, i, indexing='xy')

        # --- Condiciones de frontera ---
        self.boundaries = {
            "left": None, "right": None, "top": None, "bottom": None
        }

        # --- Sólido (máscara booleana) ---
        self.solid = cp.zeros((self.ny, self.nx), dtype=cp.bool_)

        # --- Refuerzo de impermeabilidad (interfaz sólido-fluido) ---
        # Se precomputan normales n=(nx,ny) a partir de un campo de distancia firmado.
        # En cada paso se anula la componente normal de (u,v) en la 1ª capa de fluido
        # adyacente al sólido.
        self._normales_validas = False
        self._nx_hat = None
        self._ny_hat = None
        self._mask_interfaz = None
        
        # --- Ghost-Cell IBM ---
        self._ghost_cell_ready = False
        self._ghost_mask = None
        
        # Ángulo de ataque del perfil (para transformación Drag/Lift)
        self.alpha_deg = 0.0
        self.alpha_geometry = 0.0  # Ángulo con que se cargó la geometría en la malla

        # --- Presión fija (Dirichlet) ---
        # Se separa en:
        # - user: puntos fijados manualmente
        # - bc: fronteras con presión fijada (p.ej. outflow con p0)
        # y se combina en fixed_pressure_mask (máscara efectiva usada por los solvers).
        self.fixed_pressure_mask_user = cp.zeros((self.ny, self.nx), dtype=cp.bool_)
        self.fixed_pressure_mask_bc = cp.zeros((self.ny, self.nx), dtype=cp.bool_)
        self.fixed_pressure_mask = cp.zeros((self.ny, self.nx), dtype=cp.bool_)
        # valor por defecto (inicialmente p0)
        self.fixed_pressure_value = cp.float32(p0)

        # CUDA kernel para hacer spreading por punto Lagrangiano (usa atomicAdd)
        kernel_src = r'''
        extern "C" __device__ float peskin_delta_f(float r) {
            float ar = fabsf(r);
            if (ar <= 1.0f) {
                return (3.0f - 2.0f*ar + sqrtf(1.0f + 4.0f*ar - 4.0f*ar*ar)) * 0.125f;
            } else if (ar <= 2.0f) {
                return (5.0f - 2.0f*ar - sqrtf(-7.0f + 12.0f*ar - 4.0f*ar*ar)) * 0.125f;
            } else {
                return 0.0f;
            }
        }

        extern "C" __global__ void spread(
            const float* __restrict__ lag_x,
            const float* __restrict__ lag_y,
            const float* __restrict__ fx,
            const float* __restrict__ fy,
            float* F_x,
            float* F_y,
            int N,
            int nx,
            int ny,
            float dx,
            float dy)
        {
            int k = blockDim.x * blockIdx.x + threadIdx.x;
            if (k >= N) return;

            float xk = lag_x[k];
            float yk = lag_y[k];

            int j0 = (int)floorf(xk);
            int i0 = (int)floorf(yk);

            for (int di = -1; di <= 2; ++di) {
                int i = i0 + di;
                if (i < 0 || i >= ny) continue;
                for (int dj = -1; dj <= 2; ++dj) {
                    int j = j0 + dj;
                    if (j < 0 || j >= nx) continue;

                    // Usar delta en unidades de índice (consistente con xk,yk)
                    float rx = ((float)j) - xk;
                    float ry = ((float)i) - yk;
                    float wx = peskin_delta_f(rx);
                    float wy = peskin_delta_f(ry);
                    float w = wx * wy;

                    int idx = i * nx + j;
                    atomicAdd(&F_x[idx], fx[k] * w);
                    atomicAdd(&F_y[idx], fy[k] * w);
                }
            }
        }
        '''
        self._spread_kernel = cp.RawKernel(kernel_src, 'spread')

        # ============================================================
        # Kernels ENMASCARADOS (sólidos) para Poisson/proyección
        # - Trata fronteras sólido-fluido con Neumann homogéneo (flujo normal cero)
        # - Evita que el operador use vecinos sólidos (clave para que la proyección
        #   realmente reduzca div(u) cerca del perfil)
        # ============================================================
        self._jacobi_update_masked = cp.ElementwiseKernel(
            in_params='raw bool solid, raw float32 p, raw float32 rhs, raw float32 d2x_W, raw float32 d2x_C, raw float32 d2x_E, raw float32 d2y_S, raw float32 d2y_C, raw float32 d2y_N, int32 nx, int32 ny',
            out_params='float32 out',
            operation=r'''
            int idx = i;
            int j = idx % nx;
            int ii = idx / nx;
            if (ii<=0 || ii>=ny-1 || j<=0 || j>=nx-1 || solid[idx]) {
                out = p[idx];
            } else {
                int E = ii*nx + (j+1);
                int W = ii*nx + (j-1);
                int N = (ii+1)*nx + j;
                int S = (ii-1)*nx + j;
                float aW = d2x_W[j], aC_x = d2x_C[j], aE = d2x_E[j];
                float aS = d2y_S[ii], aC_y = d2y_C[ii], aN = d2y_N[ii];
                float sum_off = 0.0f;
                float diag = aC_x + aC_y;
                if (!solid[E]) { sum_off += aE * p[E]; } else { diag += aE; }
                if (!solid[W]) { sum_off += aW * p[W]; } else { diag += aW; }
                if (!solid[N]) { sum_off += aN * p[N]; } else { diag += aN; }
                if (!solid[S]) { sum_off += aS * p[S]; } else { diag += aS; }
                if (fabsf(diag) > 1e-30f) {
                    out = (rhs[idx] - sum_off) / diag;
                } else {
                    out = p[idx];
                }
            }
            ''',
            name='jacobi_update_masked'
        )

        self._laplacian_kernel_masked = cp.ElementwiseKernel(
            in_params='raw bool solid, raw float32 x, raw float32 d2x_W, raw float32 d2x_C, raw float32 d2x_E, raw float32 d2y_S, raw float32 d2y_C, raw float32 d2y_N, int32 nx, int32 ny',
            out_params='float32 Ax',
            operation=r'''
            int idx = i;
            int j = idx % nx;
            int ii = idx / nx;
            if (ii<=0 || ii>=ny-1 || j<=0 || j>=nx-1 || solid[idx]) {
                Ax = 0.0f;
            } else {
                int E = ii*nx + (j+1);
                int W = ii*nx + (j-1);
                int N = (ii+1)*nx + j;
                int S = (ii-1)*nx + j;
                float aW = d2x_W[j], aC_x = d2x_C[j], aE = d2x_E[j];
                float aS = d2y_S[ii], aC_y = d2y_C[ii], aN = d2y_N[ii];
                float xc = x[idx];
                float sum_off = 0.0f;
                float diag = aC_x + aC_y;
                if (!solid[E]) { sum_off += aE * x[E]; } else { diag += aE; }
                if (!solid[W]) { sum_off += aW * x[W]; } else { diag += aW; }
                if (!solid[N]) { sum_off += aN * x[N]; } else { diag += aN; }
                if (!solid[S]) { sum_off += aS * x[S]; } else { diag += aS; }
                Ax = sum_off + diag * xc;
            }
            ''',
            name='laplacian_masked'
        )

        self._precond_jacobi_kernel_masked = cp.ElementwiseKernel(
            in_params='raw bool solid, raw float32 r, raw float32 d2x_W, raw float32 d2x_C, raw float32 d2x_E, raw float32 d2y_S, raw float32 d2y_C, raw float32 d2y_N, int32 nx, int32 ny',
            out_params='float32 z',
            operation=r'''
            int idx = i;
            int j = idx % nx;
            int ii = idx / nx;
            if (ii<=0 || ii>=ny-1 || j<=0 || j>=nx-1 || solid[idx]) {
                z = 0.0f;
            } else {
                int E = ii*nx + (j+1);
                int W = ii*nx + (j-1);
                int N = (ii+1)*nx + j;
                int S = (ii-1)*nx + j;
                float aW = d2x_W[j], aC_x = d2x_C[j], aE = d2x_E[j];
                float aS = d2y_S[ii], aC_y = d2y_C[ii], aN = d2y_N[ii];
                float diag = aC_x + aC_y;
                if (solid[E]) diag += aE;
                if (solid[W]) diag += aW;
                if (solid[N]) diag += aN;
                if (solid[S]) diag += aS;
                if (fabsf(diag) > 1e-30f) {
                    // M^{-1} r = r / diag(A)  (diag < 0 para Laplaciano)
                    z = r[idx] / diag;
                } else {
                    z = 0.0f;
                }
            }
            ''',
            name='precond_jacobi_masked'
        )

        # ============================================================
        # Kernel Red-Black Gauss-Seidel SOR (in-place, ~2x más rápido que Jacobi)
        # ============================================================
        self._rb_gs_sor_kernel = cp.RawKernel(r'''
        extern "C" __global__ void rb_gs_sor(
            const bool* __restrict__ solid,
            float* __restrict__ p,
            const float* __restrict__ rhs,
            const float* __restrict__ d2x_W,
            const float* __restrict__ d2x_C,
            const float* __restrict__ d2x_E,
            const float* __restrict__ d2y_S,
            const float* __restrict__ d2y_C,
            const float* __restrict__ d2y_N,
            float omega,
            int nx, int ny, int phase
        ) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= nx * ny) return;
            int j = idx % nx;
            int ii = idx / nx;
            if (((ii + j) & 1) != phase) return;
            if (ii <= 0 || ii >= ny-1 || j <= 0 || j >= nx-1 || solid[idx]) return;

            int E = ii*nx + (j+1);
            int W = ii*nx + (j-1);
            int N = (ii+1)*nx + j;
            int S = (ii-1)*nx + j;
            float aW = d2x_W[j], aC_x = d2x_C[j], aE = d2x_E[j];
            float aS = d2y_S[ii], aC_y = d2y_C[ii], aN = d2y_N[ii];
            float sum_off = 0.0f;
            float diag = aC_x + aC_y;
            if (!solid[E]) { sum_off += aE * p[E]; } else { diag += aE; }
            if (!solid[W]) { sum_off += aW * p[W]; } else { diag += aW; }
            if (!solid[N]) { sum_off += aN * p[N]; } else { diag += aN; }
            if (!solid[S]) { sum_off += aS * p[S]; } else { diag += aS; }
            if (fabsf(diag) > 1e-30f) {
                float p_gs = (rhs[idx] - sum_off) / diag;
                p[idx] = (1.0f - omega) * p[idx] + omega * p_gs;
            }
        }
        ''', 'rb_gs_sor')

        # ============================================================
        # Kernel fusionado: Restricción 2D (fine → coarse, full-weighting)
        # Reemplaza ~8 operaciones CuPy (slicing + aritmética) con 1 lanzamiento
        # ============================================================
        self._restrict_kernel = cp.RawKernel(r'''
        extern "C" __global__ void restrict_2d(
            const float* __restrict__ fine,
            float* __restrict__ coarse,
            int nx_f, int ny_f, int nx_c, int ny_c
        ) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= nx_c * ny_c) return;
            int jc = idx % nx_c;
            int ic = idx / nx_c;
            int jf = jc * 2;
            int if_ = ic * 2;
            if (jf + 1 >= nx_f || if_ + 1 >= ny_f) {
                coarse[idx] = 0.0f;
                return;
            }
            int base = if_ * nx_f + jf;
            coarse[idx] = 0.25f * (fine[base] + fine[base + 1]
                                 + fine[base + nx_f] + fine[base + nx_f + 1]);
        }
        ''', 'restrict_2d')

        # ============================================================
        # Kernel fusionado: Prolongación + suma (coarse → fine, bilineal, += )
        # Reemplaza ~15 operaciones CuPy (slicing condicional) con 1 lanzamiento
        # ============================================================
        self._prolongate_add_kernel = cp.RawKernel(r'''
        extern "C" __global__ void prolongate_add_2d(
            const float* __restrict__ coarse,
            float* __restrict__ fine,
            const float* __restrict__ wx_arr,
            const float* __restrict__ wy_arr,
            int nx_c, int ny_c, int nx_f, int ny_f
        ) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= nx_f * ny_f) return;
            int jf = idx % nx_f;
            int if_ = idx / nx_f;
            int jc = jf >> 1;
            int ic = if_ >> 1;
            // Clamp
            if (jc >= nx_c) jc = nx_c - 1;
            if (ic >= ny_c) ic = ny_c - 1;
            int jc1 = jc + 1 < nx_c ? jc + 1 : jc;
            int ic1 = ic + 1 < ny_c ? ic + 1 : ic;
            // Pesos basados en posicion real (pre-computados)
            float wx = wx_arr[jf];
            float wy = wy_arr[if_];
            // Si es impar y no hay vecino superior, skip
            if (wx > 0.0f && jc1 == jc) return;
            if (wy > 0.0f && ic1 == ic) return;
            float f00 = coarse[ic * nx_c + jc];
            float f10 = coarse[ic * nx_c + jc1];
            float f01 = coarse[ic1 * nx_c + jc];
            float f11 = coarse[ic1 * nx_c + jc1];
            fine[idx] += (1.0f-wx)*(1.0f-wy)*f00 + wx*(1.0f-wy)*f10
                       + (1.0f-wx)*wy*f01 + wx*wy*f11;
        }
        ''', 'prolongate_add_2d')

        # ============================================================
        # Kernel fusionado: Divergencia enmascarada
        # Reemplaza ~20 operaciones CuPy (boolean masks + fancy indexing) con 1
        # ============================================================
        self._divergence_kernel = cp.RawKernel(r'''
        extern "C" __global__ void divergence_masked(
            const float* __restrict__ u, const float* __restrict__ v,
            const bool* __restrict__ solid,
            float* __restrict__ div,
            const float* __restrict__ d1x_W,
            const float* __restrict__ d1x_C,
            const float* __restrict__ d1x_E,
            const float* __restrict__ d1y_S,
            const float* __restrict__ d1y_C,
            const float* __restrict__ d1y_N,
            int nx, int ny
        ) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= nx * ny) return;
            int j = idx % nx;
            int i = idx / nx;
            if (i <= 0 || i >= ny-1 || j <= 0 || j >= nx-1 || solid[idx]) {
                div[idx] = 0.0f;
                return;
            }
            int E = i*nx + (j+1);
            int W = i*nx + (j-1);
            int N = (i+1)*nx + j;
            int S = (i-1)*nx + j;
            bool sE = solid[E], sW = solid[W], sN = solid[N], sS = solid[S];
            // Solo computar componente du/dx si NINGUN vecino en x es solido.
            // Si alguno es solid (ghost cell), el valor IBM no satisface
            // continuidad discreta -> genera divergencia espuria.
            float du_dx = (!sE && !sW) ? (d1x_W[j]*u[W] + d1x_C[j]*u[idx] + d1x_E[j]*u[E]) : 0.0f;
            float dv_dy = (!sN && !sS) ? (d1y_S[i]*v[S] + d1y_C[i]*v[idx] + d1y_N[i]*v[N]) : 0.0f;
            div[idx] = du_dx + dv_dy;
        }
        ''', 'divergence_masked')

        # ============================================================
        # Kernel fusionado: Ghost-Cell IBM (interpolación bilineal + asignación)
        # Reemplaza ~30 operaciones CuPy con 1 lanzamiento
        # ============================================================
        self._ghost_cell_kernel = cp.RawKernel(r'''
        extern "C" __global__ void ghost_cell_bc_kernel(
            float* __restrict__ u, float* __restrict__ v,
            const long long* __restrict__ gi, const long long* __restrict__ gj,
            const float* __restrict__ img_x, const float* __restrict__ img_y,
            int n_ghost, int nx, int ny
        ) {
            int g = blockDim.x * blockIdx.x + threadIdx.x;
            if (g >= n_ghost) return;
            int row = (int)gi[g];
            int col = (int)gj[g];
            float x = img_x[g];
            float y = img_y[g];
            // Clamp
            if (x < 0.0f) x = 0.0f;
            if (x > (float)(nx-1) - 0.001f) x = (float)(nx-1) - 0.001f;
            if (y < 0.0f) y = 0.0f;
            if (y > (float)(ny-1) - 0.001f) y = (float)(ny-1) - 0.001f;
            int j0 = (int)floorf(x);
            int i0 = (int)floorf(y);
            int j1 = j0 + 1 < nx ? j0 + 1 : nx - 1;
            int i1 = i0 + 1 < ny ? i0 + 1 : ny - 1;
            float wx = x - (float)j0;
            float wy = y - (float)i0;
            float w00 = (1.0f-wx)*(1.0f-wy);
            float w10 = wx*(1.0f-wy);
            float w01 = (1.0f-wx)*wy;
            float w11 = wx*wy;
            int idx00 = i0*nx + j0, idx10 = i0*nx + j1;
            int idx01 = i1*nx + j0, idx11 = i1*nx + j1;
            float u_img = w00*u[idx00] + w10*u[idx10] + w01*u[idx01] + w11*u[idx11];
            float v_img = w00*v[idx00] + w10*v[idx10] + w01*v[idx01] + w11*v[idx11];
            u[row*nx + col] = -u_img;
            v[row*nx + col] = -v_img;
        }
        ''', 'ghost_cell_bc_kernel')

        # Kernel: Poner a cero u,v en sólido interior
        self._zero_solid_kernel = cp.RawKernel(r'''
        extern "C" __global__ void zero_solid(
            float* __restrict__ u, float* __restrict__ v,
            const bool* __restrict__ mask, int n
        ) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= n || !mask[idx]) return;
            u[idx] = 0.0f;
            v[idx] = 0.0f;
        }
        ''', 'zero_solid')

        # Kernel: Corrección de velocidad tras proyección (fused grad + subtract)
        self._velocity_correction_kernel = cp.RawKernel(r'''
        extern "C" __global__ void velocity_correction(
            float* __restrict__ u, float* __restrict__ v,
            const float* __restrict__ p,
            const bool* __restrict__ solid,
            float coef,
            const float* __restrict__ d1x_W,
            const float* __restrict__ d1x_C,
            const float* __restrict__ d1x_E,
            const float* __restrict__ d1y_S,
            const float* __restrict__ d1y_C,
            const float* __restrict__ d1y_N,
            int nx, int ny
        ) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= nx * ny) return;
            int j = idx % nx;
            int i = idx / nx;
            if (i <= 0 || i >= ny-1 || j <= 0 || j >= nx-1 || solid[idx]) return;
            int E = i*nx + (j+1);
            int W = i*nx + (j-1);
            int N = (i+1)*nx + j;
            int S = (i-1)*nx + j;
            bool sE = solid[E], sW = solid[W], sN = solid[N], sS = solid[S];
            // Gradiente de presion: consistente con divergencia.
            // Si el kernel de divergencia pone du/dx=0 (porque algun vecino x
            // es solido), aqui tambien ponemos dp/dx=0 para esa celda.
            // Idem para dp/dy. Esto evita correcciones espurias en celdas
            // adyacentes al cuerpo.
            float pW = sW ? p[idx] : p[W];
            float pE = sE ? p[idx] : p[E];
            float pS = sS ? p[idx] : p[S];
            float pN = sN ? p[idx] : p[N];
            float dp_dx = (!sE && !sW) ? (d1x_W[j]*pW + d1x_C[j]*p[idx] + d1x_E[j]*pE) : 0.0f;
            float dp_dy = (!sN && !sS) ? (d1y_S[i]*pS + d1y_C[i]*p[idx] + d1y_N[i]*pN) : 0.0f;
            u[idx] -= coef * dp_dx;
            v[idx] -= coef * dp_dy;
        }
        ''', 'velocity_correction')

        # ============================================================
        # Kernel: Gradiente adjunto D^T·p  (transpuesto del operador divergencia)
        # Garantiza que D(D^T p) = (DD^T)p sea simétrico → CG converge.
        # (D^T p)_u[i,j] = d1x_W[j+1]*p[E] + d1x_C[j]*p[C] + d1x_E[j-1]*p[W]
        # (D^T p)_v[i,j] = d1y_S[i+1]*p[N] + d1y_C[i]*p[C] + d1y_N[i-1]*p[S]
        # ============================================================
        self._adjoint_gradient_kernel = cp.RawKernel(r'''
        extern "C" __global__ void adjoint_gradient(
            const float* __restrict__ p,
            const bool* __restrict__ solid,
            float* __restrict__ grad_u,
            float* __restrict__ grad_v,
            const float* __restrict__ d1x_W,
            const float* __restrict__ d1x_C,
            const float* __restrict__ d1x_E,
            const float* __restrict__ d1y_S,
            const float* __restrict__ d1y_C,
            const float* __restrict__ d1y_N,
            int nx, int ny
        ) {
            /*  Calcula D^T · p  (transpuesta exacta del operador divergencia).
             *
             *  Para cada celda (i,j), suma contribuciones de las celdas de
             *  divergencia INTERIORES (no-borde, no-sólido) que referencian
             *  u[i,j] o v[i,j] en su estencil.
             *
             *  Esto garantiza que  <q, D*DT*p> = <DT*q, DT*p>  para todo q,p
             *  (simetria exacta de DD^T).
             */
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= nx * ny) return;
            int j = idx % nx;
            int i = idx / nx;

            if (solid[idx]) {
                grad_u[idx] = 0.0f;
                grad_v[idx] = 0.0f;
                return;
            }

            float gu = 0.0f, gv = 0.0f;

            /* ---- u-component of D^T·p ----
             * u[i,j] aparece en  du/dx  de  div[i, J]  cuando J = j+1, j, j-1.
             * Condiciones por cada celda divergencia (i,J):
             *   1) Interior:  1<=i<=ny-2  &&  1<=J<=nx-2
             *   2) No sólida: !solid[i,J]
             *   3) Ambos vecinos x no sólidos: !solid[i,J-1] && !solid[i,J+1]
             */
            if (i >= 1 && i <= ny-2) {
                // div[i, j+1] usa u[i,j] con coef d1x_W[j+1]
                if (j+1 >= 1 && j+1 <= nx-2) {
                    int dv = i*nx + (j+1);
                    if (!solid[dv] && !solid[dv-1] && !solid[dv+1])
                        gu += d1x_W[j+1] * p[dv];
                }
                // div[i, j] usa u[i,j] con coef d1x_C[j]
                if (j >= 1 && j <= nx-2) {
                    if (!solid[idx] && !solid[idx-1] && !solid[idx+1])
                        gu += d1x_C[j] * p[idx];
                }
                // div[i, j-1] usa u[i,j] con coef d1x_E[j-1]
                if (j-1 >= 1 && j-1 <= nx-2) {
                    int dv = i*nx + (j-1);
                    if (!solid[dv] && !solid[dv-1] && !solid[dv+1])
                        gu += d1x_E[j-1] * p[dv];
                }
            }

            /* ---- v-component of D^T·p ----
             * v[i,j] aparece en  dv/dy  de  div[I, j]  cuando I = i+1, i, i-1.
             */
            if (j >= 1 && j <= nx-2) {
                // div[i+1, j] usa v[i,j] con coef d1y_S[i+1]
                if (i+1 >= 1 && i+1 <= ny-2) {
                    int dv = (i+1)*nx + j;
                    if (!solid[dv] && !solid[dv-nx] && !solid[dv+nx])
                        gv += d1y_S[i+1] * p[dv];
                }
                // div[i, j] usa v[i,j] con coef d1y_C[i]
                if (i >= 1 && i <= ny-2) {
                    if (!solid[idx] && !solid[idx-nx] && !solid[idx+nx])
                        gv += d1y_C[i] * p[idx];
                }
                // div[i-1, j] usa v[i,j] con coef d1y_N[i-1]
                if (i-1 >= 1 && i-1 <= ny-2) {
                    int dv = (i-1)*nx + j;
                    if (!solid[dv] && !solid[dv-nx] && !solid[dv+nx])
                        gv += d1y_N[i-1] * p[dv];
                }
            }

            grad_u[idx] = gu;
            grad_v[idx] = gv;
        }
        ''', 'adjoint_gradient')

        # Flag de jerarquía multigrid (se inicializa en _init_mg_hierarchy)
        self._mg_initialized = False

    def _init_mg_hierarchy(self, niveles_max=4):
        """Pre-computa jerarquía multigrid: máscaras, h2, buffers por nivel."""
        ny, nx = self.p.shape
        self._mg_niveles = max(0, min(niveles_max, int(np.log2(min(ny, nx))) - 2))

        def _coarsen_mask(mask):
            ny_m, nx_m = mask.shape
            ny_c, nx_c = ny_m // 2, nx_m // 2
            m = mask[:ny_c*2, :nx_c*2]
            return m[0::2, 0::2] | m[1::2, 0::2] | m[0::2, 1::2] | m[1::2, 1::2]

        # Jerarquía de máscaras de sólidos (nivel 0 = fino)
        self._mg_solids = [self.solid]
        for lvl in range(self._mg_niveles):
            self._mg_solids.append(_coarsen_mask(self._mg_solids[-1]))

        # Jerarquía Dirichlet
        self._mg_hay_dirichlet = bool(cp.any(self.fixed_pressure_mask))
        self._mg_dirichlet = [self.fixed_pressure_mask]
        if self._mg_hay_dirichlet:
            self._mg_fixed_mask_flat = self.fixed_pressure_mask.ravel()
            for lvl in range(self._mg_niveles):
                self._mg_dirichlet.append(_coarsen_mask(self._mg_dirichlet[-1]))
        else:
            self._mg_fixed_mask_flat = None
            for lvl in range(self._mg_niveles):
                s = self._mg_solids[lvl + 1].shape
                self._mg_dirichlet.append(cp.zeros(s, dtype=cp.bool_))

        # Per-level stencil coefficients (d2) for non-uniform Laplacian
        self._mg_d2x_W = []
        self._mg_d2x_C = []
        self._mg_d2x_E = []
        self._mg_d2y_S = []
        self._mg_d2y_C = []
        self._mg_d2y_N = []
        X_1d_lvl = self.X_1d
        Y_1d_lvl = self.Y_1d
        for lvl in range(self._mg_niveles + 1):
            met_x = calcular_metricas_1d(X_1d_lvl)
            met_y = calcular_metricas_1d(Y_1d_lvl)
            self._mg_d2x_W.append(met_x['d2_W'])
            self._mg_d2x_C.append(met_x['d2_C'])
            self._mg_d2x_E.append(met_x['d2_E'])
            self._mg_d2y_S.append(met_y['d2_W'])
            self._mg_d2y_C.append(met_y['d2_C'])
            self._mg_d2y_N.append(met_y['d2_E'])
            if lvl < self._mg_niveles:
                # Coarsen positions: take every-other node
                nx_c = len(X_1d_lvl) // 2
                ny_c = len(Y_1d_lvl) // 2
                X_1d_lvl = X_1d_lvl[:nx_c*2:2]
                Y_1d_lvl = Y_1d_lvl[:ny_c*2:2]

        # Pre-computar pesos de prolongación por nivel (malla no-uniforme)
        # wx[jf]=0 si jf par, wx[jf]=(x_f[jf]-x_f[jf-1])/(x_f[jf+1]-x_f[jf-1]) si impar
        self._mg_prolong_wx = []
        self._mg_prolong_wy = []
        X_1d_lvl = self.X_1d
        Y_1d_lvl = self.Y_1d
        for lvl in range(self._mg_niveles + 1):
            nx_f = len(X_1d_lvl)
            ny_f = len(Y_1d_lvl)
            wx = cp.zeros(nx_f, dtype=cp.float32)
            wy = cp.zeros(ny_f, dtype=cp.float32)
            # Pesos x: para índices impares
            for jf in range(1, nx_f, 2):
                if jf + 1 < nx_f:
                    denom = float(X_1d_lvl[jf + 1] - X_1d_lvl[jf - 1])
                    if abs(denom) > 1e-30:
                        wx[jf] = float(X_1d_lvl[jf] - X_1d_lvl[jf - 1]) / denom
                    else:
                        wx[jf] = 0.5
                else:
                    wx[jf] = 0.5
            # Pesos y: para índices impares
            for if_ in range(1, ny_f, 2):
                if if_ + 1 < ny_f:
                    denom = float(Y_1d_lvl[if_ + 1] - Y_1d_lvl[if_ - 1])
                    if abs(denom) > 1e-30:
                        wy[if_] = float(Y_1d_lvl[if_] - Y_1d_lvl[if_ - 1]) / denom
                    else:
                        wy[if_] = 0.5
                else:
                    wy[if_] = 0.5
            self._mg_prolong_wx.append(wx)
            self._mg_prolong_wy.append(wy)
            if lvl < self._mg_niveles:
                nx_c = len(X_1d_lvl) // 2
                ny_c = len(Y_1d_lvl) // 2
                X_1d_lvl = X_1d_lvl[:nx_c*2:2]
                Y_1d_lvl = Y_1d_lvl[:ny_c*2:2]

        # Buffers pre-alocados por nivel
        self._mg_bufs = []
        for lvl in range(self._mg_niveles + 1):
            s = self._mg_solids[lvl].shape
            self._mg_bufs.append({
                'error': cp.zeros(s, dtype=cp.float32),
                'res':   cp.zeros(s, dtype=cp.float32),
                'rhs':   cp.zeros(s, dtype=cp.float32),
                'prolonged': cp.zeros(self._mg_solids[max(0, lvl-1)].shape, dtype=cp.float32) if lvl > 0 else None,
            })

        # Flat views
        self._mg_solids_flat = [s.ravel() for s in self._mg_solids]

        # Pre-computar dimensiones de kernel por nivel (evitar recomputar cada V-cycle)
        self._mg_kdims = []
        for lvl in range(self._mg_niveles + 1):
            s = self._mg_solids[lvl].shape
            total = s[0] * s[1]
            self._mg_kdims.append({
                'total': total,
                'grid': (total + 255) // 256,
                'nx_i': cp.int32(s[1]),
                'ny_i': cp.int32(s[0]),
                'ny': s[0],
                'nx': s[1],
            })

        # Máscaras float para corrección de velocidad (evitar cp.where)
        free = ~self.solid
        free_c = free[1:-1, 1:-1]
        self._mg_mask_x_f32 = (free_c & free[1:-1, 2:] & free[1:-1, :-2]).astype(cp.float32)
        self._mg_mask_y_f32 = (free_c & free[2:, 1:-1] & free[:-2, 1:-1]).astype(cp.float32)

        # Máscara de fluido (para normas)
        self._mg_free = free
        self._mg_free_flat = free.ravel()
        self._mg_n_free = float(cp.sum(free))

        self._mg_niveles_max_param = niveles_max
        self._mg_initialized = True

    def _calcular_mask_interfaz(self):
        """Devuelve máscara booleana (fluido) de celdas adyacentes 4-vecinas a sólido."""
        solid = self.solid
        ny, nx = solid.shape
        if ny < 3 or nx < 3:
            return cp.zeros_like(solid, dtype=cp.bool_)

        # Vecinos del sólido (sin wrap): construir con slicing para evitar cp.roll
        vecino_solido = cp.zeros_like(solid, dtype=cp.bool_)
        # Norte (i+1), Sur (i-1), Este (j+1), Oeste (j-1)
        vecino_solido[1:-1, 1:-1] = (
            solid[1:-1, 2:] | solid[1:-1, :-2] | solid[2:, 1:-1] | solid[:-2, 1:-1]
        )

        return (~solid) & vecino_solido

    def _precomputar_normales_impermeabilidad(self, eps=1e-12):
        """Precomputar normales unitarias en interfaz sólido-fluido usando distancia (GPU)."""
        solid = self.solid
        ny, nx = solid.shape
        if ny < 3 or nx < 3:
            self._normales_validas = False
            self._nx_hat = None
            self._ny_hat = None
            self._mask_interfaz = None
            return

        # Campo de distancia firmado: phi>0 en fluido, phi<0 en sólido
        # - dist_fuera: distancia del fluido al sólido
        # - dist_dentro: distancia del sólido al fluido
        # Nota: distance_transform_edt calcula distancia al cero para elementos no-cero.
        fluido_int = (~solid).astype(cp.int8)
        solido_int = solid.astype(cp.int8)
        dist_fuera = distance_transform_edt(fluido_int)
        dist_dentro = distance_transform_edt(solido_int)
        phi = dist_fuera - dist_dentro

        # Gradiente de phi (diferencias centradas); normal apunta hacia +phi (hacia el fluido)
        inv_2dx = cp.float32(0.5 / float(self.dx))
        inv_2dy = cp.float32(0.5 / float(self.dy))
        dphi_dx = cp.zeros_like(phi, dtype=cp.float32)
        dphi_dy = cp.zeros_like(phi, dtype=cp.float32)
        dphi_dx[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]).astype(cp.float32) * inv_2dx
        dphi_dy[1:-1, :] = (phi[2:, :] - phi[:-2, :]).astype(cp.float32) * inv_2dy

        norma = cp.sqrt(dphi_dx * dphi_dx + dphi_dy * dphi_dy + cp.float32(eps)).astype(cp.float32)
        nx_hat = dphi_dx / norma
        ny_hat = dphi_dy / norma

        mask_interfaz = self._calcular_mask_interfaz()
        # Solo interesan normales fiables en la interfaz
        nx_hat = cp.where(mask_interfaz, nx_hat, cp.float32(0.0))
        ny_hat = cp.where(mask_interfaz, ny_hat, cp.float32(0.0))

        self._nx_hat = nx_hat.astype(cp.float32, copy=False)
        self._ny_hat = ny_hat.astype(cp.float32, copy=False)
        self._mask_interfaz = mask_interfaz
        self._normales_validas = True

    def reforzar_impermeabilidad(self):
        """Anula componente normal de velocidad en la capa de fluido adyacente al sólido."""
        if not getattr(self, '_normales_validas', False) or self._nx_hat is None:
            self._precomputar_normales_impermeabilidad()
        if not getattr(self, '_normales_validas', False) or self._mask_interfaz is None:
            return

        mask = self._mask_interfaz
        if not cp.any(mask):
            return

        nx_hat = self._nx_hat
        ny_hat = self._ny_hat

        # Proyección tangencial: u <- u - (u·n) n
        un = (self.u * nx_hat + self.v * ny_hat).astype(cp.float32)
        self.u = cp.where(mask, self.u - un * nx_hat, self.u)
        self.v = cp.where(mask, self.v - un * ny_hat, self.v)

        # Seguridad: interior sólido siempre a cero (ghost cells conservan su valor)
        if self._ghost_cell_ready:
            self.u[self._solid_interior] = 0.0
            self.v[self._solid_interior] = 0.0
        else:
            self.u[self.solid] = 0.0
            self.v[self.solid] = 0.0

    def _precomputar_ghost_cell(self):
        """
        Precomputa los datos necesarios para el método Ghost-Cell IBM.
        
        Identifica celdas ghost (sólidas adyacentes a fluido) y calcula 
        los puntos imagen simétricos en el fluido para cada una.
        Se llama una vez después de cargar la geometría.
        """
        solid = self.solid
        ny, nx = solid.shape
        dx = self.dx; dy = self.dy
        
        # Ghost cells: celdas SÓLIDAS con al menos un vecino FLUIDO (4-vecinos)
        fluid = ~solid
        vecino_fluido = cp.zeros_like(solid, dtype=cp.bool_)
        vecino_fluido[1:-1, 1:-1] = (
            fluid[1:-1, 2:] | fluid[1:-1, :-2] | fluid[2:, 1:-1] | fluid[:-2, 1:-1]
        )
        ghost_mask = solid & vecino_fluido
        
        # Excluir bordes del dominio
        ghost_mask[0, :] = False; ghost_mask[-1, :] = False
        ghost_mask[:, 0] = False; ghost_mask[:, -1] = False
        
        # Obtener distancia firmada y normales
        sd, nx_hat, ny_hat = self._signed_distance_and_normals()
        
        # Indices de ghost cells
        ghost_indices = cp.where(ghost_mask)
        i_ghost = ghost_indices[0].astype(cp.float32)  # filas (y)
        j_ghost = ghost_indices[1].astype(cp.float32)  # columnas (x)
        n_ghost = len(i_ghost)
        
        if n_ghost == 0:
            self._ghost_cell_ready = False
            self._ghost_mask = ghost_mask
            self._ghost_direct_zero = cp.zeros(0, dtype=cp.bool_)
            return
        
        # Distancia de cada ghost cell a la pared (en celdas, siempre negativa dentro del sólido)
        # sd < 0 dentro del sólido, sd > 0 fuera
        d_ghost = sd[ghost_indices[0], ghost_indices[1]]  # negativo
        
        # Normal en cada ghost cell (apunta hacia el fluido)
        nx_g = nx_hat[ghost_indices[0], ghost_indices[1]]
        ny_g = ny_hat[ghost_indices[0], ghost_indices[1]]
        
        # Punto imagen: reflejar ghost a través de la pared hacia el fluido
        # x_imagen = x_ghost + 2*|d|*n  (en unidades de celda, d ya está en celdas)
        # |d| = -d_ghost (porque d_ghost < 0 dentro del sólido)
        # Mínimo desplazamiento: 1.5 celdas para que el punto imagen esté en fluido puro
        d_abs = cp.maximum(-d_ghost, cp.float32(0.5))  # al menos 0.5 celdas
        desplazamiento = cp.float32(2.0) * d_abs
        # Mínimo: mover al menos 1.5 celdas para alcanzar fluido
        desplazamiento = cp.maximum(desplazamiento, cp.float32(1.5))
        
        # Coordenadas imagen en índices (j=x, i=y)
        # nx_g está en coord. físicas: dx*normal_x, pero sd ya está en celdas
        j_image = j_ghost + desplazamiento * nx_g  # x-direction
        i_image = i_ghost + desplazamiento * ny_g  # y-direction
        
        # Clamp para que queden dentro del dominio
        j_image = cp.clip(j_image, cp.float32(1.0), cp.float32(nx - 2))
        i_image = cp.clip(i_image, cp.float32(1.0), cp.float32(ny - 2))

        # ── Validar que los 4 nodos de la interpolación bilineal estén en fluido ──
        # Causa de inestabilidad con dx_min fino: en esquinas del perfil escalonado
        # (staircase), la normal apunta en diagonal y el punto imagen puede caer
        # sobre otra ghost cell B, generando u_ghost_A = +u_imagen_B > 0 (fuente de
        # aceleración en vez de freno). Se extiende el desplazamiento 1 celda a la vez
        # hasta que todos los nodos bilineales sean fluido.
        n_corregidos = 0
        for _iter in range(25):
            j0_v = cp.clip(cp.floor(j_image).astype(cp.int32), 0, nx - 2)
            i0_v = cp.clip(cp.floor(i_image).astype(cp.int32), 0, ny - 2)
            # Los 4 nodos bilineales que usará _bilinear_interpolate
            bad = (solid[i0_v,     j0_v    ] |
                   solid[i0_v,     j0_v + 1] |
                   solid[i0_v + 1, j0_v    ] |
                   solid[i0_v + 1, j0_v + 1])
            if not cp.any(bad):
                break
            if _iter == 0:
                n_corregidos = int(cp.sum(bad))
            # Empujar 1 celda más lejos en dirección normal para los problemáticos
            desplazamiento = cp.where(bad, desplazamiento + cp.float32(1.0), desplazamiento)
            j_image = j_ghost + desplazamiento * nx_g
            i_image = i_ghost + desplazamiento * ny_g
            j_image = cp.clip(j_image, cp.float32(1.0), cp.float32(nx - 2))
            i_image = cp.clip(i_image, cp.float32(1.0), cp.float32(ny - 2))

        # Para los pocos casos donde incluso 25 iteraciones no bastan (ángulos muy
        # oblicuos en aristas de sólido): marcar esos ghost cells como "directos"
        # (se fijarán a 0 en apply_ghost_cell_bc en vez de usar el punto imagen).
        j0_v = cp.clip(cp.floor(j_image).astype(cp.int32), 0, nx - 2)
        i0_v = cp.clip(cp.floor(i_image).astype(cp.int32), 0, ny - 2)
        still_bad = (solid[i0_v,     j0_v    ] |
                     solid[i0_v,     j0_v + 1] |
                     solid[i0_v + 1, j0_v    ] |
                     solid[i0_v + 1, j0_v + 1])
        n_still_bad = int(cp.sum(still_bad))
        if n_corregidos > 0 or n_still_bad > 0:
            print(f"  [Ghost-cell] {n_corregidos} image points corregidos | "
                  f"{n_still_bad} irreparables (forzados a 0) | total ghost={n_ghost}")
        # Guardar máscara de ghost cells "directos" (u_ghost = 0 para estos)
        self._ghost_direct_zero = still_bad  # shape: (n_ghost,)

        self._ghost_mask = ghost_mask
        self._ghost_i = ghost_indices[0]
        self._ghost_j = ghost_indices[1]
        self._image_j_idx = j_image  # para _bilinear_interpolate (x = j)
        self._image_i_idx = i_image  # para _bilinear_interpolate (y = i)
        self._n_ghost = n_ghost
        self._ghost_cell_ready = True
        
        # Máscara de sólido interior (celdas sólidas que NO son ghost)
        self._solid_interior = solid & (~ghost_mask)
    
    def apply_ghost_cell_bc(self):
        """
        Aplica condición IBM Ghost-Cell: impone u=0 en la posición exacta de la pared.
        
        Para cada celda ghost (sólida adyacente a fluido):
            u_ghost = -u_imagen
        donde u_imagen se interpola bilinealmente desde el campo fluido
        en el punto simétrico al ghost respecto a la pared real.
        """
        if not self._ghost_cell_ready:
            self.u[self.solid] = 0.0
            self.v[self.solid] = 0.0
            return
        
        # Interpolar velocidad en puntos imagen desde el campo actual
        u_image = self._bilinear_interpolate(self.u, self._image_j_idx, self._image_i_idx)
        v_image = self._bilinear_interpolate(self.v, self._image_j_idx, self._image_i_idx)
        
        # Sanitizar NaN (punto imagen caía en zona problemática)
        u_image = cp.where(cp.isnan(u_image), cp.float32(0.0), u_image)
        v_image = cp.where(cp.isnan(v_image), cp.float32(0.0), v_image)
        
        # Limitar magnitud con velocidad de referencia fija (no depende del campo actual)
        clamp_max = cp.float32(5.0 * self._vel_ref)
        u_image = cp.clip(u_image, -clamp_max, clamp_max)
        v_image = cp.clip(v_image, -clamp_max, clamp_max)
        
        # Ghost cell = negativo de imagen (para que en la pared: promedio = 0)
        self.u[self._ghost_i, self._ghost_j] = -u_image
        self.v[self._ghost_i, self._ghost_j] = -v_image

        # Ghost cells "directos" (imagen en sólido incluso tras 25 iters):
        # imponer velocidad cero directamente (no-slip de primer orden, estable)
        if hasattr(self, '_ghost_direct_zero') and cp.any(self._ghost_direct_zero):
            gi = self._ghost_i[self._ghost_direct_zero]
            gj = self._ghost_j[self._ghost_direct_zero]
            self.u[gi, gj] = cp.float32(0.0)
            self.v[gi, gj] = cp.float32(0.0)
        
        # Sólido interior (lejos de la interfaz): mantener a cero
        self.u[self._solid_interior] = 0.0
        self.v[self._solid_interior] = 0.0

    def info(self):
        print(f"Mesh: {self.nx} x {self.ny}  ({self.nx * self.ny:,} nodos)")
        dx_min = float(cp.min(cp.diff(self.X_1d)))
        dx_max = float(cp.max(cp.diff(self.X_1d)))
        dy_min = float(cp.min(cp.diff(self.Y_1d)))
        dy_max = float(cp.max(cp.diff(self.Y_1d)))
        print(f"Malla variable: {self.malla_variable}")
        print(f"dx: min={dx_min:.6f}  max={dx_max:.6f}  ratio={dx_max/dx_min:.2f}")
        print(f"dy: min={dy_min:.6f}  max={dy_max:.6f}  ratio={dy_max/dy_min:.2f}")
        print(f"dx (CFL)={self.dx:.6f}, dy (CFL)={self.dy:.6f}")
        print(f"Dominio: [{float(self.X_1d[0]):.3f}, {float(self.X_1d[-1]):.3f}] x "
              f"[{float(self.Y_1d[0]):.3f}, {float(self.Y_1d[-1]):.3f}]")
        print(f"Campos GPU shape: ({self.ny}, {self.nx})")

    def set_fixed_pressure_points(self, points, p_value=None):
        """
        points: lista de tuplas (i,j) en índices de celda (fila, columna).
        p_value: valor de presión a fijar (si None usa self.fixed_pressure_value).
        Marca las celdas dadas en la máscara y asigna el valor p en `self.p`.
        """
        if p_value is not None:
            self.fixed_pressure_value = cp.float32(p_value)

        if points is None or len(points) == 0:
            return

        # convertir a arrays de índices
        idxs = np.array(points, dtype=np.int32)
        ii = idxs[:, 0]
        jj = idxs[:, 1]
        ii_cp = cp.asarray(ii, dtype=cp.int32)
        jj_cp = cp.asarray(jj, dtype=cp.int32)
        try:
            self.fixed_pressure_mask_user[ii_cp, jj_cp] = True
            self.p[ii_cp, jj_cp] = self.fixed_pressure_value
        except Exception:
            # protección si índices fuera de rango
            for (i, j) in points:
                if 0 <= i < self.ny and 0 <= j < self.nx:
                    self.fixed_pressure_mask_user[i, j] = True
                    self.p[i, j] = self.fixed_pressure_value

        # Recalcular máscara efectiva
        self.fixed_pressure_mask[:] = self.fixed_pressure_mask_user | self.fixed_pressure_mask_bc

    def _get_freestream_angle_rad(self):
        """Obtiene el ángulo del flujo libre desde las condiciones inflow."""
        for side, bc in self.boundaries.items():
            if bc is not None and bc[0] == "inflow":
                vx, vy = bc[1]
                return float(np.arctan2(vy, vx))
        return 0.0

    def cambiar_angulo_ataque(self, nuevo_alpha_deg, U_inf):
        """
        Cambia el ángulo de ataque modificando la dirección del flujo.
        Asume que la geometría está cargada a 0° (plan_polar).
        Establece v0x = U·cos(α), v0y = U·sin(α).

        Parámetros:
            nuevo_alpha_deg: ángulo de ataque deseado en grados
            U_inf: módulo de la velocidad del flujo libre
        
        Retorna:
            (v0x_new, v0y_new): nuevas componentes de velocidad libre
        """
        alpha_rad = np.deg2rad(nuevo_alpha_deg)
        v0x_new = float(U_inf * np.cos(alpha_rad))
        v0y_new = float(U_inf * np.sin(alpha_rad))

        # Actualizar alpha interno
        self.alpha_deg = nuevo_alpha_deg

        # Actualizar todas las fronteras inflow
        for side, bc in self.boundaries.items():
            if bc is not None and bc[0] == "inflow":
                self.boundaries[side] = ("inflow", (v0x_new, v0y_new))

        # Aplicar inmediatamente
        self.apply_boundaries(after_projection=False)

        return v0x_new, v0y_new

    def set_boundary(self, side, bc_type, value=None):
        """side: left/right/top/bottom ; bc_type: noslip/inflow/outflow/slip"""
        assert side in self.boundaries
        assert bc_type in ("inflow", "outflow", "noslip", "slip")
        self.boundaries[side] = (bc_type, value)

        # Si se fija presión en una frontera (outflow con value), convertirlo en Dirichlet real
        # para que project_cg lo respete durante las iteraciones.
        # Nota: un único valor escalar para toda la frontera.
        if side == "left":
            p_slice = (slice(None), 0)
        elif side == "right":
            p_slice = (slice(None), -1)
        elif side == "bottom":
            p_slice = (0, slice(None))
        elif side == "top":
            p_slice = (-1, slice(None))

        if bc_type == "outflow" and value is not None:
            self.fixed_pressure_value = cp.float32(value)
            self.fixed_pressure_mask_bc[p_slice] = True
            self.p[p_slice] = self.fixed_pressure_value
        else:
            # Si no hay presión fijada en esta frontera, limpiar solo la parte BC
            self.fixed_pressure_mask_bc[p_slice] = False

        # Recalcular máscara efectiva
        self.fixed_pressure_mask[:] = self.fixed_pressure_mask_user | self.fixed_pressure_mask_bc

    def apply_boundaries(self, after_projection=False):
        """
        Aplica condiciones de frontera.
        
        Parámetros:
            after_projection: Si True, no sobrescribir outflow (ya tiene grad(p) aplicado)
        """
        for side, bc in self.boundaries.items():
            if bc is None:
                continue

            bc_type, value = bc

            # Indexación correcta: filas (i, y) y columnas (j, x)
            if side == "left":
                u_slice = (slice(None), 0)   # columna j=0
                v_slice = (slice(None), 0)
            elif side == "right":
                u_slice = (slice(None), -1)  # columna j=-1
                v_slice = (slice(None), -1)
            elif side == "bottom":
                u_slice = (0, slice(None))   # fila i=0
                v_slice = (0, slice(None))
            elif side == "top":
                u_slice = (-1, slice(None))  # fila i=-1
                v_slice = (-1, slice(None))

            # -----------------------------
            # Condiciones físicas genéricas
            # -----------------------------
            if bc_type == "noslip":
                self.u[u_slice] = 0
                self.v[v_slice] = 0

            elif bc_type == "inflow":
                # ⭐ SIEMPRE aplicar inflow (incluso después de proyección)
                # value = (u_in, v_in)
                self.u[u_slice] = value[0]
                self.v[v_slice] = value[1]

            elif bc_type == "outflow":
                # Presión Dirichlet en la salida (p = value, típicamente 0).
                # Esto estabiliza la proyección incompresible.
                if value is not None:
                    self.p[u_slice] = value

                # Neumann de velocidad SIEMPRE (antes y después de proyección).
                # El kernel de corrección no actúa en j=nx-1, así que la celda
                # frontera no se actualiza por proyección; es necesario copiar
                # para mantener consistencia con el interior en todo momento.
                if side == "left":
                    self.u[:, 0] = self.u[:, 1]
                    self.v[:, 0] = self.v[:, 1]
                elif side == "right":
                    self.u[:, -1] = self.u[:, -2]
                    self.v[:, -1] = self.v[:, -2]
                elif side == "bottom":
                    self.u[0, :] = self.u[1, :]
                    self.v[0, :] = self.v[1, :]
                elif side == "top":
                    self.u[-1, :] = self.u[-2, :]
                    self.v[-1, :] = self.v[-2, :]

            elif bc_type == "slip":
                # No penetración → velocidad normal = 0
                if side in ["top", "bottom"]:
                    self.v[v_slice] = 0  # normal (y)
                else:
                    self.u[u_slice] = 0  # normal (x)

                # Copiar tangencial del interior
                if side == "left":
                    self.v[:, 0] = self.v[:, 1]
                elif side == "right":
                    self.v[:, -1] = self.v[:, -2]
                elif side == "bottom":
                    self.u[0, :] = self.u[1, :]
                elif side == "top":
                    self.u[-1, :] = self.u[-2, :]

    def _aplicar_bc_presion_neumann(self):
        """Aplica dp/dn=0 en los bordes donde NO haya Dirichlet.

        Importante: la proyección usa grad(p) con stencils que tocan bordes.
        Si p se queda constante/incorrecta en bordes, se reinyecta divergencia.
        """
        p = self.p
        fixed = self.fixed_pressure_mask

        # Left: p[:,0] = p[:,1]
        if p.shape[1] >= 2:
            mask = ~fixed[:, 0]
            p[mask, 0] = p[mask, 1]

            # Right: si no es Dirichlet, Neumann
            mask = ~fixed[:, -1]
            p[mask, -1] = p[mask, -2]

        # Bottom / Top
        if p.shape[0] >= 2:
            mask = ~fixed[0, :]
            p[0, mask] = p[1, mask]

            mask = ~fixed[-1, :]
            p[-1, mask] = p[-2, mask]

        # Reimponer Dirichlet por seguridad
        try:
            if cp.any(fixed):
                p[fixed] = self.fixed_pressure_value
        except Exception:
            pass

    def _bilinear_interpolate(self, field, x_idx, y_idx):
        """
        field: cp.array (ny, nx)
        x_idx, y_idx: coordenadas en *índices* (float) — j (x) y i (y), mismos shape
        Devuelve interpolación bilineal en cada punto (mismo shape).

        Comentarios/equación:
        - Ecuación bilineal que se evalúa:
            f(x,y) = (1-w_x)(1-w_y) f00 + w_x(1-w_y) f10 + (1-w_x)w_y f01 + w_x w_y f11
        - Mapeo en código:
            * `j0,i0` = índices inferiores → celdas f00,f10,f01,f11
            * `wx,wy` = pesos fraccionales w_x,w_y
            * `f00 = field[i0,j0]` etc. y retorno directo de la fórmula.
        """
        ny, nx = field.shape

        # clamp en los bordes (evitar overflow)
        x = cp.clip(x_idx, 0.0, nx - 1.000001)
        y = cp.clip(y_idx, 0.0, ny - 1.000001)

        j0 = cp.floor(x).astype(cp.int32)
        i0 = cp.floor(y).astype(cp.int32)
        j1 = cp.minimum(j0 + 1, nx - 1)
        i1 = cp.minimum(i0 + 1, ny - 1)

        wx = x - j0
        wy = y - i0

        f00 = field[i0, j0]
        f10 = field[i0, j1]
        f01 = field[i1, j0]
        f11 = field[i1, j1]

        # bilinear
        return (1-wx)*(1-wy)*f00 + wx*(1-wy)*f10 + (1-wx)*wy*f01 + wx*wy*f11

    def advect_backtrace(self, dt):
        """
        Calcula la posición anterior de cada partícula en *índices* (j,i),
        listos para la interpolación bilineal.

        Para malla variable, la conversión físico→índice usa searchsorted.
        Para malla uniforme, usa la conversión directa u/dx (más rápida).
        """
        if self.malla_variable:
            # ---- Malla variable: backtrace en coordenadas físicas ----
            dt_f = cp.float32(dt)
            x_dep = self.XX - self.u * dt_f   # (ny, nx) posición física de partida
            y_dep = self.YY - self.v * dt_f

            # Clamp al dominio
            x_dep = cp.clip(x_dep, self.X_1d[0], self.X_1d[-1])
            y_dep = cp.clip(y_dep, self.Y_1d[0], self.Y_1d[-1])

            # Convertir coordenada física → índice fraccionario
            x_flat = x_dep.ravel()
            y_flat = y_dep.ravel()

            kx = cp.searchsorted(self.X_1d, x_flat, side='right') - 1
            ky = cp.searchsorted(self.Y_1d, y_flat, side='right') - 1

            kx = cp.clip(kx, 0, self.nx - 2)
            ky = cp.clip(ky, 0, self.ny - 2)

            # Fracción local dentro de la celda
            dx_loc = self.X_1d[kx + 1] - self.X_1d[kx]
            dy_loc = self.Y_1d[ky + 1] - self.Y_1d[ky]
            dx_loc = cp.maximum(dx_loc, cp.float32(1e-30))
            dy_loc = cp.maximum(dy_loc, cp.float32(1e-30))

            self.x_prev_idx = (kx.astype(cp.float32) + (x_flat - self.X_1d[kx]) / dx_loc).reshape(x_dep.shape)
            self.y_prev_idx = (ky.astype(cp.float32) + (y_flat - self.Y_1d[ky]) / dy_loc).reshape(y_dep.shape)
        else:
            # ---- Malla uniforme: conversión directa (rápida) ----
            JJ = self.JJ
            II = self.II
            u_idx = (self.u / self.dx).astype(cp.float32)
            v_idx = (self.v / self.dy).astype(cp.float32)
            self.x_prev_idx = JJ - u_idx * dt
            self.y_prev_idx = II - v_idx * dt

    '''Velocidades y presiones'''
    
    def advect_velocities(self, dt):
        """Advección semi-Lagrangiana; steady=True activa under-relaxation para régimen estacionario.

        Ecuación continua (transporte por características):
            u^{n+1}(x) = u^n(x_prev),    where x_prev = x - u*dt
        Mapeo en código:
            - `advect_backtrace` calcula x_prev
            - `_bilinear_interpolate(..., x_prev, y_prev)` evalúa u^n en x_prev
            - se imponen `u=0` dentro de sólidos
        """
        # 1️⃣ Backtrace
        self.advect_backtrace(dt)

        # 2️⃣ Interpolación bilineal
        new_u = self._bilinear_interpolate(self.u, self.x_prev_idx, self.y_prev_idx)
        new_v = self._bilinear_interpolate(self.v, self.x_prev_idx, self.y_prev_idx)

        new_u[self.solid] = 0.0
        new_v[self.solid] = 0.0

        self.u = new_u
        self.v = new_v
        # Aplicar Ghost-Cell IBM (superficie suave)
        self.apply_ghost_cell_bc()
        # Refuerzo de impermeabilidad cerca del sólido (evita "jets" en la interfaz)
        self.reforzar_impermeabilidad()

    def diffuse_velocity(self, nu, dt, usar_wale=True):
        """
        Difusión viscosa de u,v con modelo WALE de viscosidad turbulenta
        y viscosidad de estela (surrogate 3D).
        Versión transitoria explícita (Forward Euler).
        
        Parámetros:
            nu: viscosidad molecular (cinemática)
            dt: paso temporal
            usar_wale: si True, usa nu_eff = nu + nu_t (WALE); si False, solo nu molecular
        
        Ecuación resuelta:
            ∂u/∂t = ∇·(ν_eff ∇u)
            donde ν_eff = ν + ν_t(x,y)
        """
        nu_f = float(nu)
        if nu_f <= 0.0 and not usar_wale:
            return

        dx = float(self.dx)
        dy = float(self.dy)
        dx_min = min(dx, dy)
        
        # ⭐ Calcular viscosidad efectiva
        if usar_wale:
            nu_t = self.compute_wale_viscosity()
            nu_eff = cp.float32(nu_f) + nu_t
        else:
            nu_eff = cp.full_like(self.u, nu_f, dtype=cp.float32)
        
        # Determinar viscosidad máxima para sub-stepping
        nu_max = float(cp.max(nu_eff))
        
        if nu_max <= 0.0:
            return
        
        # Para estabilidad explícita: dt_sub <= C * min(dx,dy)^2 / nu_max
        C_visc = 0.25
        dt_visc = C_visc * (dx_min * dx_min) / nu_max

        # Sub-stepping para respetar dt_visc
        if dt > dt_visc and dt_visc > 0.0 and dt_visc != float('inf'):
            n_sub = int(np.ceil(dt / dt_visc))
            n_sub = max(n_sub, 1)
            dt_sub = dt / n_sub
        else:
            n_sub = 1
            dt_sub = dt

        dt_sub_cp = cp.float32(dt_sub)
        # Coeficientes de stencil para derivadas (soporta malla variable)
        # x-dir: (nx-2,), broadcastea por columnas
        _d1x_W = self.d1x_W[1:-1];  _d1x_C = self.d1x_C[1:-1];  _d1x_E = self.d1x_E[1:-1]
        _d2x_W = self.d2x_W[1:-1];  _d2x_C = self.d2x_C[1:-1];  _d2x_E = self.d2x_E[1:-1]
        # y-dir: (ny-2, 1), broadcastea por filas
        _d1y_S = self.d1y_S[1:-1, cp.newaxis];  _d1y_C = self.d1y_C[1:-1, cp.newaxis];  _d1y_N = self.d1y_N[1:-1, cp.newaxis]
        _d2y_S = self.d2y_S[1:-1, cp.newaxis];  _d2y_C = self.d2y_C[1:-1, cp.newaxis];  _d2y_N = self.d2y_N[1:-1, cp.newaxis]

        # Prealocar temporales
        u_new = self.u.astype(cp.float32, copy=True)
        v_new = self.v.astype(cp.float32, copy=True)

        for _ in range(n_sub):
            u = u_new
            v = v_new
            
            # ⭐ Con viscosidad variable: ∇·(ν_eff ∇u) ≠ ν_eff ∇²u
            # Forma correcta: ∂(ν ∂u/∂x)/∂x + ∂(ν ∂u/∂y)/∂y
            viscosidad_variable = usar_wale
            if viscosidad_variable:
                # Calcular gradientes de u,v
                du_dx = cp.zeros_like(u, dtype=cp.float32)
                du_dy = cp.zeros_like(u, dtype=cp.float32)
                dv_dx = cp.zeros_like(v, dtype=cp.float32)
                dv_dy = cp.zeros_like(v, dtype=cp.float32)
                
                du_dx[:, 1:-1] = _d1x_W * u[:, :-2] + _d1x_C * u[:, 1:-1] + _d1x_E * u[:, 2:]
                du_dy[1:-1, :] = _d1y_S * u[:-2, :] + _d1y_C * u[1:-1, :] + _d1y_N * u[2:, :]
                dv_dx[:, 1:-1] = _d1x_W * v[:, :-2] + _d1x_C * v[:, 1:-1] + _d1x_E * v[:, 2:]
                dv_dy[1:-1, :] = _d1y_S * v[:-2, :] + _d1y_C * v[1:-1, :] + _d1y_N * v[2:, :]
                
                # Calcular gradientes de nu_eff
                dnu_dx = cp.zeros_like(nu_eff, dtype=cp.float32)
                dnu_dy = cp.zeros_like(nu_eff, dtype=cp.float32)
                dnu_dx[:, 1:-1] = _d1x_W * nu_eff[:, :-2] + _d1x_C * nu_eff[:, 1:-1] + _d1x_E * nu_eff[:, 2:]
                dnu_dy[1:-1, :] = _d1y_S * nu_eff[:-2, :] + _d1y_C * nu_eff[1:-1, :] + _d1y_N * nu_eff[2:, :]
                
                # Laplaciano de u,v (stencil no-uniforme)
                lap_u = cp.zeros_like(u, dtype=cp.float32)
                lap_v = cp.zeros_like(v, dtype=cp.float32)
                lap_u[1:-1, 1:-1] = (
                    _d2x_W * u[1:-1, :-2] + _d2x_C * u[1:-1, 1:-1] + _d2x_E * u[1:-1, 2:]
                    + _d2y_S * u[:-2, 1:-1] + _d2y_C * u[1:-1, 1:-1] + _d2y_N * u[2:, 1:-1]
                )
                lap_v[1:-1, 1:-1] = (
                    _d2x_W * v[1:-1, :-2] + _d2x_C * v[1:-1, 1:-1] + _d2x_E * v[1:-1, 2:]
                    + _d2y_S * v[:-2, 1:-1] + _d2y_C * v[1:-1, 1:-1] + _d2y_N * v[2:, 1:-1]
                )
                
                # Término completo: ∇·(ν_eff ∇u) = ν_eff ∇²u + ∇ν_eff · ∇u
                div_visc_u = nu_eff * lap_u + dnu_dx * du_dx + dnu_dy * du_dy
                div_visc_v = nu_eff * lap_v + dnu_dx * dv_dx + dnu_dy * dv_dy
                
                u_new = u + dt_sub_cp * div_visc_u
                v_new = v + dt_sub_cp * div_visc_v
            else:
                # Viscosidad constante: caso simple (stencil no-uniforme)
                lap_u = cp.zeros_like(u, dtype=cp.float32)
                lap_v = cp.zeros_like(v, dtype=cp.float32)
                lap_u[1:-1, 1:-1] = (
                    _d2x_W * u[1:-1, :-2] + _d2x_C * u[1:-1, 1:-1] + _d2x_E * u[1:-1, 2:]
                    + _d2y_S * u[:-2, 1:-1] + _d2y_C * u[1:-1, 1:-1] + _d2y_N * u[2:, 1:-1]
                )
                lap_v[1:-1, 1:-1] = (
                    _d2x_W * v[1:-1, :-2] + _d2x_C * v[1:-1, 1:-1] + _d2x_E * v[1:-1, 2:]
                    + _d2y_S * v[:-2, 1:-1] + _d2y_C * v[1:-1, 1:-1] + _d2y_N * v[2:, 1:-1]
                )
                u_new = u + nu_eff[0,0] * dt_sub_cp * lap_u
                v_new = v + nu_eff[0,0] * dt_sub_cp * lap_v

            # Enmascarar sólido
            u_new[self.solid] = 0.0
            v_new[self.solid] = 0.0

        self.u = u_new
        self.v = v_new
        # Aplicar Ghost-Cell IBM (superficie suave)
        self.apply_ghost_cell_bc()
        # Refuerzo de impermeabilidad cerca del sólido
        self.reforzar_impermeabilidad()
        self.apply_boundaries(after_projection=False)

    def apply_pressure_gradient(self, rho, dt):
        """
        Aplica -(dt/rho)*grad(p^n) al campo de velocidad (paso predictor
        del método incremental de corrección de presión).
        Reutiliza el kernel de corrección de velocidad.
        """
        if float(cp.max(cp.abs(self.p))) < 1e-30:
            return  # No hay presión acumulada aún
        ny, nx = self.p.shape
        total = nx * ny
        block = 256
        grid = (total + block - 1) // block
        coef = cp.float32(float(dt) / float(rho))
        self._velocity_correction_kernel(
            (grid,), (block,),
            (self.u.ravel(), self.v.ravel(), self.p.ravel(), self.solid.ravel(),
             coef,
             self.d1x_W, self.d1x_C, self.d1x_E,
             self.d1y_S, self.d1y_C, self.d1y_N,
             cp.int32(nx), cp.int32(ny)))

    def _compute_divergence_field(self, out=None):
        """
        Calcula el campo completo de divergencia del^2 u = du/dx + dv/dy.
        Usa stencil adaptativo cerca de sólidos para evitar gradientes explosivos.
        
        Parámetros:
            out: buffer opcional (ny,nx) float32 para evitar nuevas alocaciones.

        Retorna:
            cp.array: Campo de divergencia (ny, nx)
        """
        # Diferencias centrales estándar:
        # ∂u/∂x ≈ (u_{i,j+1} - u_{i,j-1}) / (2 Δx)
        # ∂v/∂y ≈ (v_{i+1,j} - v_{i-1,j}) / (2 Δy)
        return self._compute_divergence_field_uv(self.u, self.v, out=out)

    def _compute_divergence_field_uv(self, u, v, out=None):
        """
        Divergencia ∇·(u,v) usando kernel CUDA fusionado.
        Reemplaza ~20 operaciones CuPy (boolean masks + fancy indexing) con 1 lanzamiento.
        """
        if out is None:
            div = cp.empty_like(self.p, dtype=cp.float32)
        else:
            div = out
        ny, nx = div.shape
        total = nx * ny
        block = 256
        grid = (total + block - 1) // block
        self._divergence_kernel(
            (grid,), (block,),
            (u.ravel(), v.ravel(), self.solid.ravel(), div.ravel(),
             self.d1x_W, self.d1x_C, self.d1x_E,
             self.d1y_S, self.d1y_C, self.d1y_N,
             cp.int32(nx), cp.int32(ny))
        )
        return div

    def compute_wale_viscosity(self, Cw=0.325, eps=1e-16):
        """
        Calcula la viscosidad sub-malla ν_t según el modelo WALE (Nicoud & Ducros).
        Devuelve un array `nu_t` con la misma forma que `self.u`.
        Todo cálculo en GPU (cupy, float32).
        """
        dx = self.dx; dy = self.dy
        u = self.u.astype(cp.float32, copy=False)
        v = self.v.astype(cp.float32, copy=False)

        ny, nx = u.shape
        # Gradientes g_ij = ∂_j u_i
        g11 = cp.zeros_like(u, dtype=cp.float32)  # du/dx
        g12 = cp.zeros_like(u, dtype=cp.float32)  # du/dy
        g21 = cp.zeros_like(u, dtype=cp.float32)  # dv/dx
        g22 = cp.zeros_like(u, dtype=cp.float32)  # dv/dy

        # Coeficientes de stencil para primera derivada (soporta malla variable)
        _d1x_W = self.d1x_W[1:-1];  _d1x_C = self.d1x_C[1:-1];  _d1x_E = self.d1x_E[1:-1]
        _d1y_S = self.d1y_S[1:-1, cp.newaxis];  _d1y_C = self.d1y_C[1:-1, cp.newaxis];  _d1y_N = self.d1y_N[1:-1, cp.newaxis]

        # centrales (interior) — stencil no-uniforme
        g11[:, 1:-1] = _d1x_W * u[:, :-2] + _d1x_C * u[:, 1:-1] + _d1x_E * u[:, 2:]
        g12[1:-1, :] = _d1y_S * u[:-2, :] + _d1y_C * u[1:-1, :] + _d1y_N * u[2:, :]
        g21[:, 1:-1] = _d1x_W * v[:, :-2] + _d1x_C * v[:, 1:-1] + _d1x_E * v[:, 2:]
        g22[1:-1, :] = _d1y_S * v[:-2, :] + _d1y_C * v[1:-1, :] + _d1y_N * v[2:, :]

        # Partes simétricas S_ij = 0.5*(g_ij + g_ji)
        S11 = g11
        S12 = 0.5 * (g12 + g21)
        S22 = g22

        # Invariante Q_S = S_ij S_ij (2D: S11^2 + 2 S12^2 + S22^2)
        Qs = S11 * S11 + 2.0 * S12 * S12 + S22 * S22

        # g² = g·g (producto matricial del tensor de gradientes de velocidad)
        g2_11 = g11 * g11 + g12 * g21
        g2_12 = g11 * g12 + g12 * g22
        g2_21 = g21 * g11 + g22 * g21
        g2_22 = g21 * g12 + g22 * g22

        # Parte simétrica de g²: Sym(g²) = (g² + (g²)^T) / 2
        G2_11 = g2_11                       # diagonal: ya simétrica
        G2_12 = 0.5 * (g2_12 + g2_21)      # off-diagonal: simetrizar
        G2_22 = g2_22                       # diagonal: ya simétrica

        trace_G2 = G2_11 + G2_22

        # S^d: parte deviatórica (sin traza) de Sym(g²)
        # En 2D se divide por 2 (dimensión), no por 3
        Sd11 = G2_11 - (trace_G2 * 0.5)
        Sd12 = G2_12
        Sd22 = G2_22 - (trace_G2 * 0.5)

        # Invariante Q_Sd = S^d_ij S^d_ij
        Qsd = Sd11 * Sd11 + 2.0 * Sd12 * Sd12 + Sd22 * Sd22

        # Evitar ceros exactos
        Qs_pos = Qs + cp.float32(eps)
        Qsd_pos = Qsd + cp.float32(eps)

        num = cp.power(Qsd_pos, 1.5)
        den = cp.power(Qs_pos, 2.5) + cp.power(Qsd_pos, 1.25) + cp.float32(eps)

        # Delta local: filtro de sub-malla basado en tamaño de celda local
        dx_local = (self.dx_e + self.dx_w) * cp.float32(0.5)  # (nx,)
        dy_local = (self.dy_n + self.dy_s) * cp.float32(0.5)  # (ny,)
        Delta2 = dx_local[cp.newaxis, :] * dy_local[:, cp.newaxis]  # (ny, nx) area de celda
        Delta = cp.sqrt(Delta2)
        coef = (cp.float32(Cw) ** 2) * Delta2

        nu_t = coef * (num / den)

        # No-negatividad y enmascarar sólidos
        nu_t = cp.where(cp.isfinite(nu_t), nu_t, cp.float32(0.0))
        nu_t = cp.maximum(nu_t, cp.float32(0.0))
        
        # ⭐ LIMITAR valores extremos de nu_t que pueden desestabilizar la simulación
        # En zonas con gradientes muy altos (cerca de paredes), WALE puede dar valores excesivos
        # Limitar nu_t a un múltiplo razonable de la escala física: nu_t_max ~ U*L / Re_crítico
        # Para Reynolds típicos (10^3 - 10^6), limitar nu_t a ~100× el tamaño de celda × velocidad
        nu_t_max_fisica = 100.0 * Delta  # límite físico basado en escala de malla
        nu_t = cp.minimum(nu_t, nu_t_max_fisica)
        
        try:
            nu_t[self.solid] = cp.float32(0.0)
        except Exception:
            pass

        return nu_t.astype(cp.float32, copy=False)

    def _anchor_pressure(self):
        """
        Elimina el grado de libertad constante de la presión en la malla
        restando la media de `p` sobre las celdas de fluido.
        Esto no impone BCs, solo fija la referencia numérica para evitar
        la singularidad del sistema de Poisson en subdominios sin Dirichlet.
        """
        try:
            # Excluir celdas fijadas cuando se calcula la referencia
            fixed = getattr(self, 'fixed_pressure_mask', None)
            if fixed is None:
                free = ~self.solid
            else:
                free = (~self.solid) & (~fixed)

            if cp.any(free):
                p_ref = cp.mean(self.p[free])
                # restar p_ref solo si es distinto de cero para evitar trabajo innecesario
                if abs(float(p_ref)) > 0.0:
                    self.p = self.p - p_ref
            else:
                # caso extremo: intentar hallar alguna celda de fluido no fijada
                try:
                    candidates = (~self.solid).ravel()
                    if fixed is not None:
                        candidates = candidates & (~fixed.ravel())
                    idx = int(cp.argmax(candidates).item())
                    i = idx // self.nx
                    j = idx % self.nx
                    p_ref = float(self.p[i, j])
                except Exception:
                    p_ref = float(self.p[1, 1])

                if abs(p_ref) > 0.0:
                    self.p = self.p - p_ref
        except Exception:
            pass

    def project_cg(self, rho_sim, dt, tol_div=1e-1, tol_residual=1e-6,
                   max_iter=500, min_iter=5, check_every=50,
                   verbose=False, print_every=100,
                   max_outer=5, modo_adaptativo=True):
        """
        Proyección incompresible con CG simétrico + defect-correction.

        Operador CG: A = -V*L  (SPD, rápida convergencia).
        Compatible con malla variable (coefs d2x/d2y/d1x/d1y por nodo).

        Sigue el mismo esquema de robustez que project_multigrid:
        - Safety: si corrección amplifica div, revert + reduce omega + retry
        - Mantenimiento: cuando div < tol, trabajo ligero
        - Global revert: si todo empeoró, restaurar estado inicial
        - Iteraciones CG adaptativas según div/tol_div
        """
        rho_f = cp.float32(rho_sim)
        dt_f = cp.float32(dt)
        ny, nx = self.p.shape
        coef_f = dt_f / rho_f

        free = ~self.solid
        solid_flat = self.solid.ravel()
        free_flat = free.ravel()
        vol_flat = self._vol_2d_flat
        block_sz = 256
        total_cells = ny * nx
        grid_k = (total_cells + block_sz - 1) // block_sz
        nx_i32 = cp.int32(nx)
        ny_i32 = cp.int32(ny)

        try:
            hay_dirichlet = bool(cp.any(self.fixed_pressure_mask))
        except Exception:
            hay_dirichlet = False

        # Ghost-Cell IBM
        self.apply_ghost_cell_bc()

        # Divergencia ANTES
        div_before = self._compute_divergence_field()
        n_free = float(cp.sum(free))
        div_mean_before = float(cp.sum(cp.abs(div_before[free]))) / n_free
        if verbose:
            print(f"[CG] Divergencia inicial: {div_mean_before:.6e}")

        # --- Determinar modo y número de outers ---
        if div_mean_before < tol_div:
            n_outer = 2
            maintenance_mode = True
        else:
            maintenance_mode = False
            if modo_adaptativo:
                if div_mean_before < tol_div * 3.0:
                    n_outer = max(3, max_outer // 2)
                else:
                    n_outer = max_outer
            else:
                n_outer = max_outer

        # ============================================================
        # Operador SPD:  A = -V * L   (V = volumen, L = Laplaciano d2)
        # ============================================================
        def _aplicar_A(vec_flat):
            Lx = self._laplacian_kernel_masked(
                solid_flat, vec_flat,
                self.d2x_W, self.d2x_C, self.d2x_E,
                self.d2y_S, self.d2y_C, self.d2y_N,
                nx_i32, ny_i32, size=vec_flat.size)
            return -(Lx * vol_flat)

        def _aplicar_Minv(res_flat):
            r_unscaled = res_flat / vol_flat
            z_L = self._precond_jacobi_kernel_masked(
                solid_flat, r_unscaled,
                self.d2x_W, self.d2x_C, self.d2x_E,
                self.d2y_S, self.d2y_C, self.d2y_N,
                nx_i32, ny_i32, size=r_unscaled.size)
            return -z_L

        # Kernel corrección velocidad
        vc_kernel = self._velocity_correction_kernel

        # =========================================================
        # DEFECT-CORRECTION
        # =========================================================
        p_acumulada = cp.zeros((ny, nx), dtype=cp.float32)
        total_iters = 0
        converged = False
        div_mean_current = div_mean_before

        # Guardar estado inicial para protección contra NaN
        u_initial = self.u.copy()
        v_initial = self.v.copy()

        for outer in range(n_outer):
            # Re-computar divergencia
            div_field = self._compute_divergence_field()
            div_mean_current = float(cp.sum(cp.abs(div_field[free]))) / n_free

            # --- Check convergencia ---
            if div_mean_current < tol_div:
                if maintenance_mode and outer >= 1:
                    converged = True
                    if verbose:
                        print(f"[CG] Mantenimiento OK outer {outer}: div={div_mean_current:.6e}")
                    break
                elif not maintenance_mode and outer > 0:
                    converged = True
                    if verbose:
                        print(f"[CG] Convergencia div outer {outer}: div={div_mean_current:.6e}")
                    break

            # --- Iteraciones CG adaptativas ---
            if maintenance_mode:
                max_inner = max(100, max_iter // 4)
            elif modo_adaptativo:
                if div_mean_current < tol_div * 2.0:
                    max_inner = max(100, max_iter // 2)
                elif div_mean_current < tol_div * 5.0:
                    max_inner = max(200, (max_iter * 2) // 3)
                else:
                    max_inner = max_iter
            else:
                max_inner = max_iter

            # RHS: rhs = (rho/dt) * div(u)
            rhs = cp.zeros((ny, nx), dtype=cp.float32)
            rhs[1:-1, 1:-1] = (rho_f / dt_f) * div_field[1:-1, 1:-1]
            rhs_flat = rhs.ravel()

            # Compatibilidad Neumann (volume-weighted)
            if not hay_dirichlet:
                vrs = cp.sum(rhs_flat[free_flat] * vol_flat[free_flat])
                vt = cp.sum(vol_flat[free_flat])
                rhs_flat[free_flat] -= vrs / vt

            # RHS para operador SPD: b = -V * rhs
            rhs_M = -(rhs_flat * vol_flat)

            # CG init - siempre desde ceros (cada outer resuelve corrección incremental)
            x = cp.zeros(total_cells, dtype=cp.float32)

            if hay_dirichlet:
                x[self.fixed_pressure_mask.ravel()] = self.fixed_pressure_value

            Ax = _aplicar_A(x)
            r = rhs_M - Ax
            if hay_dirichlet:
                r[self.fixed_pressure_mask.ravel()] = 0.0
            z = _aplicar_Minv(r)
            p_dir = z.copy()
            rz = float(cp.dot(r, z))

            # CG inner loop
            recompute_every = 50
            for it_cg in range(max_inner):
                Ap = _aplicar_A(p_dir)
                pAp = float(cp.dot(p_dir, Ap))
                if abs(pAp) < 1e-30:
                    break
                alpha_f = rz / pAp
                alpha = cp.float32(alpha_f)
                x += alpha * p_dir
                r -= alpha * Ap

                # Recomputo periódico del residuo real (evita drift float32)
                if (it_cg + 1) % recompute_every == 0:
                    r = rhs_M - _aplicar_A(x)

                if hay_dirichlet:
                    r[self.fixed_pressure_mask.ravel()] = 0.0
                elif (it_cg + 1) % recompute_every == 0:
                    # Proyectar fuera del null-space (constante) para Neumann
                    mean_x = float(cp.sum(x[free_flat] * vol_flat[free_flat]) /
                                   cp.sum(vol_flat[free_flat]))
                    x[free_flat] -= cp.float32(mean_x)
                    r = rhs_M - _aplicar_A(x)

                if it_cg >= min_iter and it_cg % check_every == 0:
                    res_norm = float(cp.sqrt(cp.dot(r, r)))
                    if verbose and it_cg % print_every == 0:
                        print(f"  [CG outer={outer+1}] it={it_cg+1:5d}  res={res_norm:.6e}")
                    if res_norm < tol_residual:
                        break

                z = _aplicar_Minv(r)
                rz_new = float(cp.dot(r, z))
                if abs(rz) < 1e-30:
                    break
                beta = rz_new / rz
                rz = rz_new
                p_dir = z + cp.float32(beta) * p_dir

            total_iters += it_cg + 1
            p_corr = x.reshape(ny, nx)
            if hay_dirichlet:
                p_corr[self.fixed_pressure_mask] = self.fixed_pressure_value

            # ============================================================
            # Corrección velocidad
            # ============================================================

            coef_eff = cp.float32(float(coef_f))
            vc_kernel(
                (grid_k,), (block_sz,),
                (self.u.ravel(), self.v.ravel(), p_corr.ravel(), solid_flat,
                 coef_eff,
                 self.d1x_W, self.d1x_C, self.d1x_E,
                 self.d1y_S, self.d1y_C, self.d1y_N,
                 nx_i32, ny_i32))

            self.apply_ghost_cell_bc()
            self.reforzar_impermeabilidad()

            # Protección contra NaN
            has_nan = bool(cp.isnan(self.u).any() or cp.isnan(self.v).any())
            if has_nan:
                self.u[:] = u_initial
                self.v[:] = v_initial
                p_acumulada[:] = 0.0
                if verbose:
                    print(f"[CG] NaN detectado en outer {outer+1}, revertido")
                break

            p_acumulada += p_corr

            if verbose:
                div_check = self._compute_divergence_field()
                div_after_corr = float(cp.sum(cp.abs(div_check[free]))) / n_free
                print(f"[CG] Outer {outer+1}/{n_outer} ({it_cg+1} iters): div={div_after_corr:.6e}")

            # Actualizar estado guardado para protección NaN
            u_initial[:] = self.u
            v_initial[:] = self.v

        # =========================================================
        # FINALIZACIÓN
        # =========================================================
        if float(cp.max(cp.abs(p_acumulada))) > 0.0:
            self.p = p_acumulada
        self._aplicar_bc_presion_neumann()
        if hay_dirichlet:
            self.p[self.fixed_pressure_mask] = self.fixed_pressure_value

        self.apply_ghost_cell_bc()
        self.reforzar_impermeabilidad()
        self.apply_boundaries(after_projection=True)

        if not hay_dirichlet:
            try:
                self._anchor_pressure()
            except Exception:
                pass

        div_after = self._compute_divergence_field()
        div_mean_after = float(cp.sum(cp.abs(div_after[free]))) / n_free

        if verbose:
            print(f"[CG] div_before={div_mean_before:.6e}  div_after={div_mean_after:.6e}  iters={total_iters}")

        return {
            'iterations': total_iters,
            'cycles': total_iters,
            'outers': outer + 1,
            'converged': converged or (div_mean_after < tol_div),
            'div_before': div_mean_before,
            'div_after': div_mean_after,
            'n_outer_used': n_outer,
        }

    def project_multigrid(self, rho_sim, dt, tol_div=1e-2,
                          max_outer=8, cycles_per_outer=5,
                          niveles_max=8, pre_suavizado=3, post_suavizado=3,
                          omega=1.15, verbose=False,
                          modo_adaptativo=True,
                          guard_residual_every_outer=True,
                          adaptive_outer0_cycles=False,
                          apply_ibm_each_outer=True,
                          rollback_on_nan=True,
                          compute_div_after=True):
        """
        Proyección incompresible con defect-correction iterativo + multigrid.
        Smoother: Red-Black Gauss-Seidel SOR (CUDA kernel in-place).

        Estrategia: múltiples iteraciones externas, cada una re-computa la
        divergencia residual y resuelve un nuevo Poisson con V-cycles multigrid.
        La corrección se aplica siempre (sin line-search): la divergencia se mide
        antes del post-procesado IBM (que reintroduce divergencia en la interfaz
        sólido-fluido y haría rechazar correcciones válidas).

        Ciclos adaptativos: una vez div < tol_div, reduce ciclos de cálculo
        a un mínimo de mantenimiento para maximizar velocidad.

        Parámetros:
            tol_div: tolerancia media sobre |div(u)|
            max_outer: iteraciones externas de defect-correction
            cycles_per_outer: V-cycles por iteración externa (máximo)
            niveles_max: niveles de coarsening (2×2)
            pre_suavizado/post_suavizado: iteraciones GS-SOR por nivel
            omega: factor sobre-relajación (1.0-1.5, típico 1.1-1.2)
            modo_adaptativo: ajustar outers y ciclos según div
            guard_residual_every_outer: valida estabilidad V-cycle en cada outer.
                                      Si False, solo valida en outer=0 (más rápido).
            adaptive_outer0_cycles: si True, aplica adaptación de cycles también
                                    en outer=0 (más rápido en pasos cercanos a tol).
            apply_ibm_each_outer: aplica Ghost-Cell e impermeabilidad tras cada outer.
                                  Si False, se aplica solo al final (más rápido).
            rollback_on_nan: guarda/restaura estado por outer para recuperación NaN.
                             Si False, elimina copias extra por outer.
            compute_div_after: calcula div_after exacto al final para reporte.
                               Si False, usa estimación del último outer.
        """
        # ================================================================
        # Constantes y pre-cómputos
        # ================================================================
        rho_f = cp.float32(rho_sim)
        dt_f = cp.float32(dt)
        ny, nx = self.p.shape
        coef_f = dt_f / rho_f          # dt/rho para corrección de velocidad

        block_sz = 256
        total_cells = ny * nx
        grid_k = (total_cells + block_sz - 1) // block_sz
        nx_i32 = cp.int32(nx)
        ny_i32 = cp.int32(ny)

        # --- Inicializar jerarquía multigrid si hace falta ---
        if not self._mg_initialized:
            self._init_mg_hierarchy(niveles_max)

        n_levels = self._mg_niveles   # niveles de coarsening disponibles

        # --- Máscaras ---
        free = self._mg_free
        free_flat = self._mg_free_flat
        n_free = self._mg_n_free

        try:
            hay_dirichlet = bool(cp.any(self.fixed_pressure_mask))
        except Exception:
            hay_dirichlet = False

        solid_flat = self._mg_solids_flat[0]

        # Iteraciones en el nivel más grueso (resolver cuasi-exactamente)
        coarsest_iters = max(50, 8 * max(
            self._mg_kdims[n_levels]['nx'],
            self._mg_kdims[n_levels]['ny']))

        # ================================================================
        # Funciones locales
        # ================================================================
        rb_kernel = self._rb_gs_sor_kernel
        restrict_k = self._restrict_kernel
        prolong_k = self._prolongate_add_kernel
        vc_kernel = self._velocity_correction_kernel

        omega_f32 = cp.float32(omega)
        p0_i32    = cp.int32(0)
        p1_i32    = cp.int32(1)

        omega_coarse_f32 = cp.float32(1.0)   # pure Gauss-Seidel para niveles gruesos

        def _smooth(p_arr, rhs, lvl, n_sweeps):
            kd      = self._mg_kdims[lvl]
            sol     = self._mg_solids_flat[lvl]
            nx_l    = kd['nx_i']
            ny_l    = kd['ny_i']
            grid_l  = (kd['grid'],)
            p_flat   = p_arr.ravel()
            rhs_flat = rhs.ravel()
            d2xW = self._mg_d2x_W[lvl]; d2xC = self._mg_d2x_C[lvl]; d2xE = self._mg_d2x_E[lvl]
            d2yS = self._mg_d2y_S[lvl]; d2yC = self._mg_d2y_C[lvl]; d2yN = self._mg_d2y_N[lvl]
            # Usar omega solo en nivel fino; GS puro (omega=1) en niveles gruesos
            om = omega_f32 if lvl == 0 else omega_coarse_f32
            for _ in range(n_sweeps):
                rb_kernel(grid_l, (256,),
                          (sol, p_flat, rhs_flat,
                           d2xW, d2xC, d2xE, d2yS, d2yC, d2yN,
                           om, nx_l, ny_l, p0_i32))
                rb_kernel(grid_l, (256,),
                          (sol, p_flat, rhs_flat,
                           d2xW, d2xC, d2xE, d2yS, d2yC, d2yN,
                           om, nx_l, ny_l, p1_i32))

        def _aplicar_bc_mg(p_nivel, lvl):
            """Fuerza Neumann cero-gradiente en bordes (copiando celda vecina),
               respetando Dirichlet donde aplica."""
            ny_n, nx_n = p_nivel.shape
            if hay_dirichlet:
                dm = self._mg_dirichlet[lvl]
                fl = ~self._mg_solids[lvl]
                if nx_n >= 2:
                    mk = fl[:, 0] & ~dm[:, 0]
                    p_nivel[mk, 0] = p_nivel[mk, 1]
                    mk = fl[:, -1] & ~dm[:, -1]
                    p_nivel[mk, -1] = p_nivel[mk, -2]
                if ny_n >= 2:
                    mk = fl[0, :] & ~dm[0, :]
                    p_nivel[0, mk] = p_nivel[1, mk]
                    mk = fl[-1, :] & ~dm[-1, :]
                    p_nivel[-1, mk] = p_nivel[-2, mk]
                if lvl == 0:
                    p_nivel[dm] = self.fixed_pressure_value
                else:
                    p_nivel[dm] = cp.float32(0.0)
            else:
                if nx_n >= 2:
                    p_nivel[:, 0]  = p_nivel[:, 1]
                    p_nivel[:, -1] = p_nivel[:, -2]
                if ny_n >= 2:
                    p_nivel[0, :]  = p_nivel[1, :]
                    p_nivel[-1, :] = p_nivel[-2, :]

        def _residual_norm(p, rhs_r, lvl=0):
            """Calcula ||rhs - L*p|| en el nivel dado."""
            kd = self._mg_kdims[lvl]
            Lp = self._laplacian_kernel_masked(
                self._mg_solids_flat[lvl], p.ravel(),
                self._mg_d2x_W[lvl], self._mg_d2x_C[lvl], self._mg_d2x_E[lvl],
                self._mg_d2y_S[lvl], self._mg_d2y_C[lvl], self._mg_d2y_N[lvl],
                kd['nx_i'], kd['ny_i'], size=kd['total'])
            return float(cp.linalg.norm(rhs_r - Lp))

        # Limitar niveles efectivos de MG: con malla 20:1 no-uniforme, niveles
        # profundos son inconsistentes (L_c ≠ R·L_f·P). Usando máx 2 niveles
        # la relación geométrica local es ~4x y la aproximación es suficiente.
        nlvl_use = min(n_levels, 2)
        # Iteraciones en nivel 'coarsest' del V-cycle (puede ser nivel 1 o 2)
        coarse_solve_iters = 20   # pocas: sólo corregir modos bajos, no resolver exactamente

        def _v_cycle(p_arr, rhs, _dbg=False):
            """V-cycle iterativo: resuelve L*p = rhs usando nlvl_use niveles."""

            if _dbg:
                r_antes = _residual_norm(p_arr, rhs.ravel())
                print(f"  [DBG] ||r|| inicial={r_antes:.4e}  ||rhs||={float(cp.linalg.norm(rhs)):.4e}")

            # Arrays por nivel (sólo hasta nlvl_use)
            p_lv   = [None] * (nlvl_use + 1)
            rhs_lv = [None] * (nlvl_use + 1)
            p_lv[0]   = p_arr
            rhs_lv[0] = rhs

            # ---- Descenso ----
            for lvl in range(nlvl_use):
                _smooth(p_lv[lvl], rhs_lv[lvl], lvl, pre_suavizado)
                _aplicar_bc_mg(p_lv[lvl], lvl)

                if _dbg and lvl == 0:
                    print(f"  [DBG] ||r|| tras presmooth lvl0={_residual_norm(p_lv[0], rhs.ravel()):.4e}")

                # Residuo: r = rhs - L*p
                kd  = self._mg_kdims[lvl]
                sol = self._mg_solids_flat[lvl]
                Lp  = self._laplacian_kernel_masked(
                    sol, p_lv[lvl].ravel(),
                    self._mg_d2x_W[lvl], self._mg_d2x_C[lvl], self._mg_d2x_E[lvl],
                    self._mg_d2y_S[lvl], self._mg_d2y_C[lvl], self._mg_d2y_N[lvl],
                    kd['nx_i'], kd['ny_i'], size=kd['total'])
                res_buf = self._mg_bufs[lvl]['res']
                cp.subtract(rhs_lv[lvl].ravel(), Lp, out=res_buf.ravel())

                # Restricción residuo → RHS del nivel grueso
                kd_c = self._mg_kdims[lvl + 1]
                if kd_c['ny'] < 2 or kd_c['nx'] < 2:
                    for lu in range(lvl, -1, -1):
                        _smooth(p_lv[lu], rhs_lv[lu], lu, post_suavizado)
                        _aplicar_bc_mg(p_lv[lu], lu)
                    return

                rhs_c = self._mg_bufs[lvl + 1]['rhs']
                restrict_k(
                    (kd_c['grid'],), (256,),
                    (res_buf.ravel(), rhs_c.ravel(),
                     cp.int32(kd['nx']), cp.int32(kd['ny']),
                     cp.int32(kd_c['nx']), cp.int32(kd_c['ny'])))
                rhs_lv[lvl + 1] = rhs_c

                e_c = self._mg_bufs[lvl + 1]['error']
                e_c[:] = 0.0
                p_lv[lvl + 1] = e_c

            # ---- Nivel más grueso del V-cycle (= nlvl_use) ----
            _smooth(p_lv[nlvl_use], rhs_lv[nlvl_use], nlvl_use, coarse_solve_iters)
            _aplicar_bc_mg(p_lv[nlvl_use], nlvl_use)

            # ---- Ascenso ----
            for lvl in range(nlvl_use - 1, -1, -1):
                kd_c = self._mg_kdims[lvl + 1]
                kd_f = self._mg_kdims[lvl]
                wx   = self._mg_prolong_wx[lvl]
                wy   = self._mg_prolong_wy[lvl]
                prolong_k(
                    (kd_f['grid'],), (256,),
                    (p_lv[lvl + 1].ravel(), p_lv[lvl].ravel(), wx, wy,
                     cp.int32(kd_c['nx']), cp.int32(kd_c['ny']),
                     cp.int32(kd_f['nx']), cp.int32(kd_f['ny'])))
                p_lv[lvl][self._mg_solids[lvl]] = cp.float32(0.0)
                _smooth(p_lv[lvl], rhs_lv[lvl], lvl, post_suavizado)
                _aplicar_bc_mg(p_lv[lvl], lvl)

                if _dbg and lvl == 0:
                    print(f"  [DBG] ||r|| tras postsmooth lvl0={_residual_norm(p_lv[0], rhs.ravel()):.4e}")

            if _dbg:
                print(f"  [DBG] ||r|| tras V-cycle COMPLETO={_residual_norm(p_arr, rhs.ravel()):.4e}")

        # ================================================================
        # Preparar campo (Ghost-Cell IBM)
        # ================================================================
        self.apply_ghost_cell_bc()

        # ================================================================
        # Divergencia ANTES
        # ================================================================
        div_work = cp.empty((ny, nx), dtype=cp.float32)
        div_abs_work = cp.empty((ny, nx), dtype=cp.float32)
        inv_n_free = 1.0 / max(float(n_free), 1.0)
        div_before_field = self._compute_divergence_field(out=div_work)
        cp.abs(div_before_field, out=div_abs_work)
        # Nota: div=0 en sólidos/bordes por kernel; media normalizada por n_free
        # reproduce la métrica anterior sin fancy indexing.
        div_mean_before   = float(cp.sum(div_abs_work)) * inv_n_free
        div_max_before    = float(cp.max(div_abs_work))             # L∞ (independiente del dominio)
        # Métrica efectiva: máxima entre media y valor típico local (percentil grueso).
        # Esto evita que un dominio grande diluya la media y engañe al modo adaptativo.
        div_eff_before    = max(div_mean_before, div_max_before * 0.01)

        if verbose:
            print(f"[MG] Divergencia inicial: media={div_mean_before:.4e}  max={div_max_before:.4e}")

        # ================================================================
        # Modo adaptativo: número de outers y ciclos
        # ================================================================
        if modo_adaptativo:
            if div_eff_before < tol_div * 0.8:
                n_outer = 1
                n_cycles = 1
                maintenance_mode = True
            elif div_eff_before < tol_div:
                n_outer = 2
                n_cycles = max(2, cycles_per_outer // 2)
                maintenance_mode = True
            elif div_eff_before < tol_div * 3.0:
                n_outer = max(3, max_outer // 2)
                n_cycles = cycles_per_outer
                maintenance_mode = False
            else:
                n_outer = max_outer
                n_cycles = cycles_per_outer
                maintenance_mode = False
        else:
            n_outer = max_outer
            n_cycles = cycles_per_outer
            maintenance_mode = False

        # ================================================================
        # Bucle externo de defect-correction
        # ================================================================
        p_last = cp.zeros((ny, nx), dtype=cp.float32)   # presión del último outer
        rhs = cp.zeros((ny, nx), dtype=cp.float32)
        p_corr = cp.zeros((ny, nx), dtype=cp.float32)
        rhs_flat_ref = rhs.ravel()
        coef_eff = cp.float32(float(coef_f))
        # Evitar alocaciones por outer en la columna de salida
        u_out_save = cp.empty((ny,), dtype=cp.float32)
        v_out_save = cp.empty((ny,), dtype=cp.float32)
        total_cycles = 0
        converged = False
        div_mean_current = div_mean_before
        div_max_current = div_max_before
        div_eff_current = div_eff_before

        # Estado inicial para recuperación ante NaN
        if rollback_on_nan:
            u_initial = self.u.copy()
            v_initial = self.v.copy()
        else:
            u_initial = None
            v_initial = None

        for outer in range(n_outer):
            # --- Divergencia del outer actual ---
            if outer == 0:
                # Reutilizar la divergencia ya calculada antes del bucle.
                div_field = div_before_field
                div_mean_current = div_mean_before
                div_max_current = div_max_before
                div_eff_current = div_eff_before
            else:
                div_field = self._compute_divergence_field(out=div_work)
                cp.abs(div_field, out=div_abs_work)
                div_mean_current = float(cp.sum(div_abs_work)) * inv_n_free
                div_max_current  = float(cp.max(div_abs_work))
                div_eff_current  = max(div_mean_current, div_max_current * 0.01)

            # --- Criterio de convergencia (sólo divergencia) ---
            # NOTA: solo romper si ya se aplicó al menos 1 corrección (outer > 0).
            # Si rompemos en outer=0, NO se aplica nada aunque la media esté diluida
            # por un dominio grande con poca divergencia local pero significativa.
            if div_eff_current < tol_div and outer > 0:
                converged = True
                if verbose:
                    print(f"[MG] Convergido en outer {outer}: div_media={div_mean_current:.4e}  div_max={div_max_current:.4e}")
                break

            # --- Ciclos adaptativos intra-outer ---
            if modo_adaptativo and (outer > 0 or adaptive_outer0_cycles):
                ratio = div_mean_current / max(tol_div, 1e-30)
                if ratio < 1.0:
                    cycles_this = 1
                elif ratio < 2.0:
                    cycles_this = max(2, n_cycles // 2)
                else:
                    cycles_this = n_cycles
            else:
                cycles_this = n_cycles

            # --- RHS: (rho/dt) * div(u*) ---
            rhs.fill(cp.float32(0.0))
            rhs[1:-1, 1:-1] = (rho_f / dt_f) * div_field[1:-1, 1:-1]

            # Columna de salida (j=nx-2): la BC Neumann u[nx-1]=u[nx-2] hace que
            # la divergencia allí sea una diferencia unilateral que el solver de
            # presión no puede compensar sin oscilar (polo amplificante).
            # Solución: vaciar el término fuente en esa columna y no aplicar
            # corrección de velocidad → la proyección no "toca" la salida.
            rhs[:, -2] = cp.float32(0.0)

            # Compatibilidad Neumann: media ponderada por volumen = 0
            if not hay_dirichlet:
                vol_flat = self._vol_2d_flat
                vrs = cp.sum(rhs_flat_ref[free_flat] * vol_flat[free_flat])
                vt = cp.sum(vol_flat[free_flat])
                if float(vt) > 0:
                    rhs_flat_ref[free_flat] -= vrs / vt

            # --- Resolver Poisson: L * p_corr = rhs ---
            p_corr.fill(cp.float32(0.0))
            if hay_dirichlet:
                p_corr[self.fixed_pressure_mask] = self.fixed_pressure_value

            # Residuo para detectar inestabilidad del V-cycle
            do_guard_check = guard_residual_every_outer or (outer == 0)
            if do_guard_check:
                r_antes = _residual_norm(p_corr, rhs_flat_ref)

            for cyc_i in range(cycles_this):
                _v_cycle(p_corr, rhs, _dbg=(verbose and outer == 0 and cyc_i < 2))
            total_cycles += cycles_this

            # Seguridad: si el V-cycle amplificó el residuo de Poisson,
            # descartamos y usamos GS solo en nivel fino con suficientes sweeps
            if do_guard_check:
                r_despues = _residual_norm(p_corr, rhs_flat_ref)
                if r_despues > r_antes * 1.5:
                    if verbose:
                        print(f"  [MG] V-cycle diverge ({r_antes:.3e}→{r_despues:.3e}): usando GS fino")
                    p_corr[:] = 0.0
                    if hay_dirichlet:
                        p_corr[self.fixed_pressure_mask] = self.fixed_pressure_value
                    # 300 sweeps: factor reducción ~0.98^300 ≈ 0.002 (99.8%)
                    gs_sweeps = max(300, cycles_this * (pre_suavizado + post_suavizado + 50))
                    _smooth(p_corr, rhs, 0, gs_sweeps)
                    _aplicar_bc_mg(p_corr, 0)

            if hay_dirichlet:
                p_corr[self.fixed_pressure_mask] = self.fixed_pressure_value

            # Neumann: eliminar modo constante de la corrección
            if not hay_dirichlet:
                p_free = p_corr[free]
                if p_free.size > 0:
                    p_corr -= cp.mean(p_free)

            # --- Aplicar corrección: u -= (dt/rho) * grad(p_corr) ---
            # IMPORTANTE: no se usa line-search. La divergencia se mide ANTES del
            # post-procesado IBM (que reintroduce divergencia en la interfaz y haría
            # rechazar correcciones válidas). La corrección se aplica siempre.
            # Guardar columna de salida: no aplicamos corrección allí (ver rhs[:,-2]=0)
            u_out_save[:] = self.u[:, -2]
            v_out_save[:] = self.v[:, -2]
            vc_kernel(
                (grid_k,), (block_sz,),
                (self.u.ravel(), self.v.ravel(), p_corr.ravel(), solid_flat,
                 coef_eff,
                 self.d1x_W, self.d1x_C, self.d1x_E,
                 self.d1y_S, self.d1y_C, self.d1y_N,
                 nx_i32, ny_i32))
            # Restaurar columna de salida (la proyección no actúa allí)
            self.u[:, -2] = u_out_save
            self.v[:, -2] = v_out_save

            if apply_ibm_each_outer:
                self.apply_ghost_cell_bc()
                self.reforzar_impermeabilidad()

            # Protección NaN: revertir al estado inicial y salir
            if bool(cp.isnan(self.u).any() or cp.isnan(self.v).any()):
                if rollback_on_nan and u_initial is not None:
                    self.u[:] = u_initial
                    self.v[:] = v_initial
                if verbose:
                    print(f"[MG] NaN outer {outer+1}, revertido")
                break

            p_last[:] = p_corr

            if verbose:
                div_tmp = self._compute_divergence_field(out=div_work)
                cp.abs(div_tmp, out=div_abs_work)
                div_log = float(cp.sum(div_abs_work)) * inv_n_free
                print(f"[MG] Outer {outer+1}/{n_outer} ({cycles_this} V-cyc): "
                      f"div={div_log:.6e}")

            # Guardar estado para la siguiente iteración
            if rollback_on_nan and u_initial is not None:
                u_initial[:] = self.u
                v_initial[:] = self.v

        # ================================================================
        # Finalización
        # ================================================================
        if float(cp.max(cp.abs(p_last))) > 0.0:
            self.p = p_last

        self._aplicar_bc_presion_neumann()
        if hay_dirichlet:
            self.p[self.fixed_pressure_mask] = self.fixed_pressure_value

        self.apply_ghost_cell_bc()
        self.reforzar_impermeabilidad()
        self.apply_boundaries(after_projection=True)

        if not hay_dirichlet:
            try:
                self._anchor_pressure()
            except Exception:
                pass

        if compute_div_after:
            div_after_field = self._compute_divergence_field(out=div_work)
            cp.abs(div_after_field, out=div_abs_work)
            div_mean_after = float(cp.sum(div_abs_work)) * inv_n_free
            div_max_after  = float(cp.max(div_abs_work))
            div_eff_after  = max(div_mean_after, div_max_after * 0.01)
        else:
            # Estimación rápida con la última divergencia evaluada en el bucle.
            div_mean_after = div_mean_current
            div_max_after = div_max_current
            div_eff_after = div_eff_current

        if verbose:
            print(f"[MG] div_media: {div_mean_before:.4e}→{div_mean_after:.4e}  "
                  f"div_max: {div_max_before:.4e}→{div_max_after:.4e}  cycles={total_cycles}")

        return {
            'iterations': total_cycles,
            'cycles': total_cycles,
            'outers': outer + 1 if n_outer > 0 else 0,
            'converged': converged or (div_eff_after < tol_div),
            'div_before': div_mean_before,
            'div_after': div_mean_after,
            'n_outer_used': n_outer,
        }

    '''Sólidos'''

    def _xy_grids(self):
        """Mallas 2D de posiciones físicas (usa las almacenadas)."""
        return self.XX, self.YY

    def add_solid_circle(self, cx, cy, radius):
        """
        Añade un sólido circular centrado en (cx,cy) con radio 'radius' (unidades físicas).
        """
        XX, YY = self._xy_grids()
        mask = ( (XX - cx)**2 + (YY - cy)**2 ) <= (radius*radius)
        self.solid = cp.logical_or(self.solid, mask)
        # Anular velocidades dentro del sólido
        self.u[self.solid] = 0.0
        self.v[self.solid] = 0.0
        # Recalcular normales para impermeabilidad
        self._precomputar_normales_impermeabilidad()
        self._precomputar_ghost_cell()
        self._mg_initialized = False  # Invalidar jerarquía multigrid

    def add_solid_rectangle(self, x_min, y_min, x_max, y_max):
        """
        Añade un sólido rectangular delimitado por [x_min,x_max]×[y_min,y_max] (unidades físicas).
        """
        XX, YY = self._xy_grids()
        mask = (XX >= x_min) & (XX <= x_max) & (YY >= y_min) & (YY <= y_max)
        self.solid = cp.logical_or(self.solid, mask)
        self.u[self.solid] = 0.0
        self.v[self.solid] = 0.0
        # Recalcular normales para impermeabilidad
        self._precomputar_normales_impermeabilidad()
        self._precomputar_ghost_cell()
        self._mg_initialized = False  # Invalidar jerarquía multigrid

    def load_solids_from_file(self, filepath,
                              chord=1.0,
                              x_offset=0.0,
                              y_offset=0.0,
                              alpha_deg=0.0,
                              fill=True,
                              plot=True,
                              min_te_height=None):
        """
        Carga un airfoil desde archivo de texto y lo rasteriza como sólido.
        Formato esperado (ejemplo NACA):
            Primera línea: título (se ignora)
            Resto: pares x y (separados por espacios o tabs)
        Parámetros:
            chord      : escala de la cuerda (multiplica x normalizado)
            x_offset,y_offset : traslado del perfil en coordenadas físicas
            alpha_deg  : ángulo de ataque en grados (rotación antihoraria)
            fill       : si True, rellena el interior (sólido completo)
            plot       : si True, dibuja el perfil sobre la malla
        Nota:
            - El archivo debe listar puntos desde el borde de salida recorriendo
              superficie superior hasta el borde de ataque y volver por inferior
              al borde de salida (orden típico NACA). Se cierra automáticamente.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

        # Leer todas las líneas y filtrar vacías / comentarios
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

        if len(lines) < 2:
            raise ValueError("Archivo no contiene suficientes puntos.")

        # Ignorar primera línea (título), parsear el resto
        coord_lines = lines[1:]
        pts = []
        for ln in coord_lines:
            parts = ln.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                pts.append((x, y))
            except ValueError:
                continue

        if len(pts) < 1:
            raise ValueError("No se pudieron leer puntos suficientes del perfil.")

        pts = np.array(pts, dtype=np.float32)  # en CPU

        # Normalizar suponiendo x ya en [0,1]; escalado por cuerda
        x_raw = pts[:, 0] * chord
        y_raw = pts[:, 1] * chord  # mantiene proporciones (espesor relativo)

        # --- Recorte del trailing edge para garantizar espesor mínimo ---
        # Se asegura que el borde de escape tenga al menos min_te_height
        # de espesor (por defecto 2*dx) para que ocupe al menos 2 celdas.
        if min_te_height is None:
            min_te_height = 2.0 * float(self.dx)

        # Encontrar el leading edge (mínimo x)
        le_idx = int(np.argmin(x_raw))

        # Superficie superior: índices [0, le_idx], x decrece de ~chord a ~0
        upper_x = x_raw[:le_idx + 1]
        upper_y = y_raw[:le_idx + 1]
        # Superficie inferior: índices [le_idx, end], x crece de ~0 a ~chord
        lower_x = x_raw[le_idx:]
        lower_y = y_raw[le_idx:]

        # Espesor actual del trailing edge
        te_thickness = abs(float(upper_y[0]) - float(lower_y[-1]))

        if te_thickness < min_te_height and len(upper_x) > 2 and len(lower_x) > 2:
            # Para interpolar, necesitamos x monótonamente creciente
            upper_x_inc = upper_x[::-1].copy()
            upper_y_inc = upper_y[::-1].copy()

            # Muestrear x desde el TE hacia el LE para encontrar dónde el
            # espesor alcanza min_te_height
            x_te = min(float(upper_x_inc[-1]), float(lower_x[-1]))
            x_le = max(float(upper_x_inc[0]),  float(lower_x[0]))
            x_sample = np.linspace(x_te, x_le, 2000)

            y_up_s = np.interp(x_sample, upper_x_inc, upper_y_inc)
            y_lo_s = np.interp(x_sample, lower_x, lower_y)
            thickness = y_up_s - y_lo_s

            # Primer x (desde TE hacia LE) donde espesor >= min_te_height
            valid = np.where(thickness >= min_te_height)[0]

            if len(valid) > 0:
                x_cut = float(x_sample[valid[0]])
                y_cut_upper = float(np.interp(x_cut, upper_x_inc, upper_y_inc))
                y_cut_lower = float(np.interp(x_cut, lower_x, lower_y))

                # Descartar puntos más allá de x_cut (zona delgada del TE)
                eps = 1e-8
                mask_up = upper_x < (x_cut - eps)   # upper va de alto a bajo x
                mask_lo = lower_x < (x_cut - eps)   # lower va de bajo a alto x

                trimmed_upper_x = upper_x[mask_up]
                trimmed_upper_y = upper_y[mask_up]
                trimmed_lower_x = lower_x[mask_lo]
                trimmed_lower_y = lower_y[mask_lo]

                # Reconstruir perfil: TE_upper → interior superior → LE →
                #                     interior inferior → TE_lower
                x_raw = np.concatenate([[x_cut], trimmed_upper_x,
                                        trimmed_lower_x, [x_cut]])
                y_raw = np.concatenate([[y_cut_upper], trimmed_upper_y,
                                        trimmed_lower_y, [y_cut_lower]])

                new_chord_eff = x_cut
                print(f"  [TE trim] Trailing edge recortado: x_cut={x_cut:.6f}, "
                      f"espesor TE={y_cut_upper - y_cut_lower:.6f} "
                      f"(mín requerido: {min_te_height:.6f}, "
                      f"cuerda efectiva: {new_chord_eff:.6f})")
            else:
                print(f"  [TE trim] AVISO: El perfil es demasiado delgado para "
                      f"alcanzar espesor TE={min_te_height:.6f}. "
                      f"Se mantiene el perfil original.")

        # Centro para rotación: usar cuarto de cuerda (convención aero)
        cx_rot = 0.25 * chord
        cy_rot = 0.0

        # Rotar por ángulo de ataque (convención: α>0 eleva el perfil)
        alpha = -np.deg2rad(alpha_deg)  # invertir signo para que α>0 rote antihorario en el dominio usado
        ca = np.cos(alpha); sa = np.sin(alpha)
        x_shift = x_raw - cx_rot
        y_shift = y_raw - cy_rot
        x_rot = x_shift * ca - y_shift * sa + cx_rot
        y_rot = x_shift * sa + y_shift * ca + cy_rot

        # Trasladar
        x_final = x_rot + x_offset
        y_final = y_rot + y_offset

        # Asegurar cierre del polígono
        if not (np.isclose(x_final[0], x_final[-1]) and np.isclose(y_final[0], y_final[-1])):
            x_final = np.concatenate([x_final, x_final[0:1]])
            y_final = np.concatenate([y_final, y_final[0:1]])

        # Rasterizar sobre la malla
        from matplotlib.path import Path
        poly = Path(np.column_stack((x_final, y_final)))

        # Coordenadas de centros de celdas (físicas, compatibles con malla variable)
        if self.malla_variable:
            x_centers = cp.asnumpy(self.X_1d)
            y_centers = cp.asnumpy(self.Y_1d)
        else:
            x_centers = (np.arange(self.nx, dtype=np.float32) + 0.5) * self.dx
            y_centers = (np.arange(self.ny, dtype=np.float32) + 0.5) * self.dy
        XXc, YYc = np.meshgrid(x_centers, y_centers, indexing='xy')
        pts_grid = np.column_stack((XXc.ravel(), YYc.ravel()))

        inside = poly.contains_points(pts_grid)
        inside = inside.reshape(self.ny, self.nx)

        # Borde (celdas tocando polígono) usando distancia (opcional simple: dilate)
        # Aquí tratamos todo el interior como sólido si fill=True
        solid_mask = inside if fill else np.zeros_like(inside, dtype=bool)

        # Convertir a CuPy y actualizar
        solid_cp = cp.asarray(solid_mask)
        self.solid = cp.logical_or(self.solid, solid_cp)

        # Anular velocidades en el sólido
        self.u[self.solid] = 0.0
        self.v[self.solid] = 0.0

        # Recalcular normales para impermeabilidad (perfil estático)
        self._precomputar_normales_impermeabilidad()
        self._precomputar_ghost_cell()
        self._mg_initialized = False  # Invalidar jerarquía multigrid

        # Reaplicar fronteras
        self.apply_boundaries()
        
        # Almacenar ángulo de ataque y ángulo de geometría
        self.alpha_deg = alpha_deg
        self.alpha_geometry = alpha_deg

        if plot:
            plt.figure(figsize=(6, 3))
            plt.plot(x_final, y_final, 'k-', linewidth=1.0, label="Perfil")
            plt.fill(x_final, y_final, alpha=0.2, color='tab:blue')
            plt.xlabel("x"); plt.ylabel("y")
            plt.axis("equal")
            plt.title(f"Airfoil cargado (α={alpha_deg}°)")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.show()

        return {
            "n_points": int(len(pts)),
            "chord": chord,
            "alpha_deg": alpha_deg,
            "bbox": (float(x_final.min()), float(y_final.min()),
                     float(x_final.max()), float(y_final.max()))
        }

    '''Visualización'''

    def visualize_velocity(self, cmap="rainbow", normalize=False, title="Velocidad |u|", show=True, save_path=None, return_fig=False):
        """
        Visualiza la magnitud de la velocidad en colores (sin flechas).
        - cmap: colormap de matplotlib.
        - normalize: si True, normaliza a [0,1] por el máximo actual.
        - show: si True, muestra la figura.
        - save_path: si se indica, guarda la imagen en esa ruta.
        - return_fig: si True, retorna la figura en lugar de mostrarla.
        """
        # Magnitud de la velocidad
        speed = cp.sqrt(self.u**2 + self.v**2)

        # Traer a CPU para matplotlib
        speed_np = cp.asnumpy(speed)

        # Coordenadas físicas para pcolormesh (malla variable)
        X_np = cp.asnumpy(self.XX)
        Y_np = cp.asnumpy(self.YY)

        ar = self.Ly / self.Lx
        fig = plt.figure(figsize=(12, 12 * ar))
        ax = plt.gca()
        im = ax.pcolormesh(X_np, Y_np, speed_np, cmap=cmap, shading='auto')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, self.Lx)
        ax.set_ylim(0, self.Ly)
        im.set_clim(0.0, speed_np.max())
        plt.colorbar(im, label="|u|")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)

        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches="tight")
        
        if return_fig:
            return fig
        elif show:
            plt.show()
        else:
            plt.close()

    def visualize_surface_traction(self, mu, rho=1.0, max_arrows=100, arrow_scale=0.02, 
                                   show_annotations=True, annotation_step=5, 
                                   show=True, save_path=None):
        """
        Visualiza campos de tracción en la superficie usando compute_surface_forces_definitive.
        Usa solo viscosidad molecular en pared (correcto físicamente con WALE).
        """
        data = self.compute_surface_forces_definitive(mu, rho=rho, return_per_face=True, return_cp=True)
        Xb = cp.asnumpy(data["Xb"])
        Yb = cp.asnumpy(data["Yb"])
        Tx_p = cp.asnumpy(data["Tx_p_face"])
        Ty_p = cp.asnumpy(data["Ty_p_face"])
        Tx_v = cp.asnumpy(data["Tx_v_face"])
        Ty_v = cp.asnumpy(data["Ty_v_face"])
        
        # Calcular fuerza total
        Tx_total = Tx_p + Tx_v
        Ty_total = Ty_p + Ty_v
        force_mag = np.sqrt(Tx_total**2 + Ty_total**2)
        
        # Submuestreo uniforme para limitar número de flechas
        n_points = len(Xb)
        if n_points > max_arrows:
            step = n_points // max_arrows
            indices = np.arange(0, n_points, step)
        else:
            indices = np.arange(n_points)
        
        # Aplicar submuestreo
        Xb_sub = Xb[indices]
        Yb_sub = Yb[indices]
        Tx_p_sub = Tx_p[indices]
        Ty_p_sub = Ty_p[indices]
        Tx_v_sub = Tx_v[indices]
        Ty_v_sub = Ty_v[indices]
        Tx_total_sub = Tx_total[indices]
        Ty_total_sub = Ty_total[indices]
        force_mag_sub = force_mag[indices]
        
        # Calcular escala automática para evitar flechas enormes
        max_force = np.max(force_mag_sub) if len(force_mag_sub) > 0 else 1.0
        auto_scale = min(self.Lx, self.Ly) * arrow_scale / max_force if max_force > 0 else arrow_scale
        
        ar = self.Ly / self.Lx
        fig, ax = plt.subplots(figsize=(16, 16 * ar))
        
        # Fondo: magnitud de velocidad para referencia
        speed = cp.sqrt(self.u**2 + self.v**2)
        ax.imshow(cp.asnumpy(speed), origin="lower",
                  extent=[0, self.Lx, 0, self.Ly], cmap="Greys", alpha=0.3)
        
        # Quiver presión (rojo)
        Q1 = ax.quiver(Xb_sub, Yb_sub, Tx_p_sub * auto_scale, Ty_p_sub * auto_scale, 
                       color="red", angles="xy", scale_units="xy", scale=1,
                       width=0.003, label="Tracción presión", alpha=0.7)
        
        # Quiver viscosa (azul)
        Q2 = ax.quiver(Xb_sub, Yb_sub, Tx_v_sub * auto_scale, Ty_v_sub * auto_scale, 
                       color="blue", angles="xy", scale_units="xy", scale=1,
                       width=0.003, label="Tracción viscosa", alpha=0.7)
        
        # Quiver total (negro, más grueso)
        Q3 = ax.quiver(Xb_sub, Yb_sub, Tx_total_sub * auto_scale, Ty_total_sub * auto_scale, 
                       color="black", angles="xy", scale_units="xy", scale=1,
                       width=0.005, label="Tracción total", alpha=0.9)
        
        # Anotaciones con valores de fuerza
        if show_annotations and len(force_mag_sub) > 0:
            for i in range(0, len(Xb_sub), annotation_step):
                ax.annotate(f'{force_mag_sub[i]:.3f}', 
                           xy=(Xb_sub[i], Yb_sub[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='darkgreen',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))
        
        # Información en el título
        avg_force = np.mean(force_mag_sub)
        max_force_val = np.max(force_mag_sub)
        ax.set_title(f"Tracción en la superficie (μ={mu:.4f})\n"
                    f"Fuerza promedio: {avg_force:.4f}, Máxima: {max_force_val:.4f}\n"
                    f"Mostrando {len(Xb_sub)} de {n_points} puntos", 
                    fontsize=12, fontweight='bold')
        
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

    def plot_forces_over_time(self, show=True, return_fig=False):
            """
            Grafica Cd y Cl con selector interactivo de rango para calcular la media.
            Usa matplotlib.widgets.SpanSelector para elegir la horquilla visualmente.
            """
            from matplotlib.widgets import SpanSelector

            cd = cp.asnumpy(self.cdvector).flatten()
            cl = cp.asnumpy(self.clvector).flatten()
            x_iter = np.arange(len(cd)) * self.guardado

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
            fig.subplots_adjust(hspace=0.35)

            # --- CD ---
            ax1.plot(x_iter, cd, linewidth=1.2, color='blue', alpha=0.7)
            ax1.set_ylabel("CD", fontsize=11)
            ax1.set_title("Coeficiente de Drag — arrastra para seleccionar rango", fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            hline_cd = ax1.axhline(y=np.mean(cd), color='darkblue', linestyle='--', linewidth=2)
            texto_cd = ax1.text(0.02, 0.92, '', transform=ax1.transAxes, fontsize=11,
                                fontweight='bold', color='darkblue',
                                bbox=dict(facecolor='white', alpha=0.8, edgecolor='darkblue'))
            shade_cd = [None]

            # --- CL ---
            ax2.plot(x_iter, cl, linewidth=1.2, color='green', alpha=0.7)
            ax2.set_xlabel(f"Iteración (guardado cada {self.guardado})", fontsize=11)
            ax2.set_ylabel("CL", fontsize=11)
            ax2.set_title("Coeficiente de Lift — arrastra para seleccionar rango", fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            hline_cl = ax2.axhline(y=np.mean(cl), color='darkgreen', linestyle='--', linewidth=2)
            texto_cl = ax2.text(0.02, 0.92, '', transform=ax2.transAxes, fontsize=11,
                                fontweight='bold', color='darkgreen',
                                bbox=dict(facecolor='white', alpha=0.8, edgecolor='darkgreen'))
            shade_cl = [None]

            def _actualizar(ax, datos, x, hline, texto, shade, xmin, xmax):
                mask = (x >= xmin) & (x <= xmax)
                if not np.any(mask):
                    return
                vals = datos[mask]
                media = float(np.mean(vals))
                std = float(np.std(vals))
                hline.set_ydata([media, media])
                n_sel = int(np.sum(mask))
                texto.set_text(f'Media: {media:.6f}  (±{std:.6f})  [{n_sel} pts]')
                if shade[0] is not None:
                    shade[0].remove()
                shade[0] = ax.axvspan(xmin, xmax, alpha=0.15, color=hline.get_color())
                fig.canvas.draw_idle()

            def on_select_cd(xmin, xmax):
                _actualizar(ax1, cd, x_iter, hline_cd, texto_cd, shade_cd, xmin, xmax)

            def on_select_cl(xmin, xmax):
                _actualizar(ax2, cl, x_iter, hline_cl, texto_cl, shade_cl, xmin, xmax)

            span_cd = SpanSelector(ax1, on_select_cd, 'horizontal',
                                   useblit=True, props=dict(alpha=0.3, facecolor='blue'),
                                   interactive=True, drag_from_anywhere=True)
            span_cl = SpanSelector(ax2, on_select_cl, 'horizontal',
                                   useblit=True, props=dict(alpha=0.3, facecolor='green'),
                                   interactive=True, drag_from_anywhere=True)

            # Calcular media inicial (todo el rango)
            _actualizar(ax1, cd, x_iter, hline_cd, texto_cd, shade_cd, x_iter[0], x_iter[-1])
            _actualizar(ax2, cl, x_iter, hline_cl, texto_cl, shade_cl, x_iter[0], x_iter[-1])

            # Guardar refs para que no se recolecten por GC
            fig._span_refs = [span_cd, span_cl]

            if return_fig:
                return fig
            elif show:
                plt.show()
            else:
                plt.close()


    def plot_convergence_history(self, show=True, return_fig=False):
        """
        Grafica la evolución de la media acumulada de Cd y Cl para mostrar convergencia.
        Útil para verificar si la simulación ha alcanzado un estado estacionario.
        
        Parámetros:
            show: si True, muestra la figura
            return_fig: si True, retorna la figura en lugar de mostrarla
        """
        # Convertir a NumPy
        cd = cp.asnumpy(self.cdvector).flatten()
        cl = cp.asnumpy(self.clvector).flatten()
        
        # Excluir el primer 10% para el cálculo de medias
        inicio_calculo = int(len(cd) * 0.1)
        cd_calculo = cd[inicio_calculo:]
        cl_calculo = cl[inicio_calculo:]
        
        # Calcular medias acumuladas solo desde el 10%
        cd_mean_running = np.cumsum(cd_calculo) / np.arange(1, len(cd_calculo) + 1)
        cl_mean_running = np.cumsum(cl_calculo) / np.arange(1, len(cl_calculo) + 1)
        
        # Ejes x en iteraciones reales
        x_full = np.arange(len(cd)) * self.guardado
        x_mean = np.arange(inicio_calculo, len(cd)) * self.guardado

        # Detectar máximos y mínimos simples (vecinos inmediatos) en la porción usada
        def detect_peaks_and_troughs(signal):
            if len(signal) < 3:
                return np.array([], dtype=int), np.array([], dtype=int)
            # Relativos a la porción (0..N-1)
            rel_max = np.where((signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:]))[0] + 1
            rel_min = np.where((signal[1:-1] < signal[:-2]) & (signal[1:-1] < signal[2:]))[0] + 1
            return rel_max.astype(int), rel_min.astype(int)

        cd_rel_max, cd_rel_min = detect_peaks_and_troughs(cd_calculo)
        cl_rel_max, cl_rel_min = detect_peaks_and_troughs(cl_calculo)

        # Convertir a índices globales (respecto a cd/cl completos) y luego a iteraciones reales
        cd_max_idx = (cd_rel_max + inicio_calculo) * self.guardado
        cd_min_idx = (cd_rel_min + inicio_calculo) * self.guardado
        cl_max_idx = (cl_rel_max + inicio_calculo) * self.guardado
        cl_min_idx = (cl_rel_min + inicio_calculo) * self.guardado

        # Ajustar línea de tendencia (ajuste lineal) sobre máximos y mínimos si hay suficientes puntos
        def linear_trend(x_pts, y_pts, x_eval):
            if len(x_pts) >= 2:
                coeffs = np.polyfit(x_pts, y_pts, 1)
                return np.polyval(coeffs, x_eval), coeffs
            else:
                return None, None

        # Usar índices en la escala de iteraciones reales
        cd_max_idx_real = cd_rel_max + inicio_calculo
        cd_min_idx_real = cd_rel_min + inicio_calculo
        cl_max_idx_real = cl_rel_max + inicio_calculo
        cl_min_idx_real = cl_rel_min + inicio_calculo
        
        cd_trend_max, cd_coef_max = linear_trend(cd_max_idx, cd[cd_max_idx_real] if len(cd_max_idx_real)>0 else np.array([]), x_mean)
        cd_trend_min, cd_coef_min = linear_trend(cd_min_idx, cd[cd_min_idx_real] if len(cd_min_idx_real)>0 else np.array([]), x_mean)
        cl_trend_max, cl_coef_max = linear_trend(cl_max_idx, cl[cl_max_idx_real] if len(cl_max_idx_real)>0 else np.array([]), x_mean)
        cl_trend_min, cl_coef_min = linear_trend(cl_min_idx, cl[cl_min_idx_real] if len(cl_min_idx_real)>0 else np.array([]), x_mean)

        # Crear figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Subplot 1: Cd
        ax1.plot(x_full, cd, 'b-', alpha=0.25, linewidth=0.8, label='Cd instantáneo')
        ax1.plot(x_mean, cd_mean_running, 'b-', linewidth=2, label='Cd promedio acumulado (desde 10%)')
        ax1.axhline(y=cd_mean_running[-1], color='r', linestyle='--', linewidth=1.2, label=f'Media final: {cd_mean_running[-1]:.6f}')
        ax1.axvline(x=inicio_calculo * self.guardado, color='orange', linestyle=':', linewidth=1, alpha=0.7, label='Inicio cálculo (10%)')

        # Graficar máximos y mínimos detectados
        if len(cd_max_idx) > 0:
            ax1.scatter(cd_max_idx, cd[cd_max_idx_real], color='red', s=20, zorder=5, label='Máximos (Cd)')
        if len(cd_min_idx) > 0:
            ax1.scatter(cd_min_idx, cd[cd_min_idx_real], color='cyan', s=20, zorder=5, label='Mínimos (Cd)')

        # Graficar líneas de tendencia (si hay ajuste)
        if cd_trend_max is not None:
            ax1.plot(x_mean, cd_trend_max, '--', color='red', linewidth=1.5, alpha=0.9, label='Tend. máximos (Cd)')
        if cd_trend_min is not None:
            ax1.plot(x_mean, cd_trend_min, '--', color='cyan', linewidth=1.5, alpha=0.9, label='Tend. mínimos (Cd)')

        ax1.set_xlabel(f'Iteración (guardado cada {self.guardado})', fontsize=11)
        ax1.set_ylabel('Coeficiente de Arrastre (Cd)', fontsize=11)
        ax1.set_title('Convergencia del Coeficiente de Arrastre', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Cl
        ax2.plot(x_full, cl, 'g-', alpha=0.25, linewidth=0.8, label='Cl instantáneo')
        ax2.plot(x_mean, cl_mean_running, 'g-', linewidth=2, label='Cl promedio acumulado (desde 10%)')
        ax2.axhline(y=cl_mean_running[-1], color='r', linestyle='--', linewidth=1.2, label=f'Media final: {cl_mean_running[-1]:.6f}')
        ax2.axvline(x=inicio_calculo * self.guardado, color='orange', linestyle=':', linewidth=1, alpha=0.7, label='Inicio cálculo (10%)')

        if len(cl_max_idx) > 0:
            ax2.scatter(cl_max_idx, cl[cl_max_idx_real], color='darkgreen', s=20, zorder=5, label='Máximos (Cl)')
        if len(cl_min_idx) > 0:
            ax2.scatter(cl_min_idx, cl[cl_min_idx_real], color='lime', s=20, zorder=5, label='Mínimos (Cl)')

        if cl_trend_max is not None:
            ax2.plot(x_mean, cl_trend_max, '--', color='darkgreen', linewidth=1.5, alpha=0.9, label='Tend. máximos (Cl)')
        if cl_trend_min is not None:
            ax2.plot(x_mean, cl_trend_min, '--', color='lime', linewidth=1.5, alpha=0.9, label='Tend. mínimos (Cl)')

        ax2.set_xlabel(f'Iteración (guardado cada {self.guardado})', fontsize=11)
        ax2.set_ylabel('Coeficiente de Sustentación (Cl)', fontsize=11)
        ax2.set_title('Convergencia del Coeficiente de Sustentación', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if return_fig:
            return fig
        elif show:
            plt.show()
        else:
            plt.close()

    def compute_divergence_mean(self):
        """
        Calcula la divergencia media absoluta del campo de velocidad.
        div(u) = du/dx + dv/dy
        
        Para fluidos incompresibles, div(u) DEBE ser 0 en todo momento
        (tanto en régimen transitorio como estacionario). La proyección de
        Poisson debería garantizar esto. Valores no nulos indican:
        - Error numérico acumulado
        - Tolerancia insuficiente en solver de Poisson
        - Problemas en condiciones de frontera o interpolaciones
        
        Retorna:
            float: Valor medio absoluto de la divergencia en el dominio
        """
        # Usar el mismo cálculo enmascarado que usa la proyección (evita stencils a través del sólido)
        div = self._compute_divergence_field()
        free = ~self.solid
        return float(cp.mean(cp.abs(div[free])))

    def plot_multigrid_cycles_history(self, show=True, return_fig=False):
        """
        Grafica el número de ciclos multigrid usados en cada paso temporal.
        Útil para diagnosticar eficiencia de convergencia y ajustar parámetros.
        
        Parámetros:
            show: Si True, muestra la figura al final (llama plt.show()).
            return_fig: Si True, retorna (fig, ax) para posterior personalización.
        
        Retorna:
            Si return_fig=True: (fig, (ax1, ax2)). De lo contrario, None.
        """
        try:
            # Mover datos a CPU
            cycles = self.mg_cycles_vector.get()
            
            # Filtrar valores válidos (>0)
            valid_mask = cycles > 0
            if not valid_mask.any():
                print("⚠️ No hay datos de ciclos multigrid para plotear")
                return None
            
            cycles = cycles[valid_mask]
            iterations = np.arange(len(cycles))
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Subplot 1: Serie temporal completa
            ax1.plot(iterations, cycles, 'b-', linewidth=0.8, alpha=0.7)
            ax1.set_xlabel('Iteración', fontsize=11)
            ax1.set_ylabel('Ciclos Multigrid', fontsize=11)
            ax1.set_title('Ciclos Multigrid por Paso Temporal', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Estadísticas
            mean_cycles = np.mean(cycles)
            median_cycles = np.median(cycles)
            max_cycles = np.max(cycles)
            min_cycles = np.min(cycles)
            
            ax1.axhline(mean_cycles, color='r', linestyle='--', linewidth=1.5, label=f'Media: {mean_cycles:.1f}')
            ax1.axhline(median_cycles, color='orange', linestyle='--', linewidth=1.5, label=f'Mediana: {median_cycles:.1f}')
            ax1.legend(loc='upper right', fontsize=10)
            
            # Subplot 2: Histograma
            ax2.hist(cycles, bins=min(30, max_cycles-min_cycles+1), color='steelblue', alpha=0.7, edgecolor='black')
            ax2.axvline(mean_cycles, color='r', linestyle='--', linewidth=2, label=f'Media: {mean_cycles:.1f}')
            ax2.axvline(median_cycles, color='orange', linestyle='--', linewidth=2, label=f'Mediana: {median_cycles:.1f}')
            ax2.set_xlabel('Ciclos Multigrid', fontsize=11)
            ax2.set_ylabel('Frecuencia', fontsize=11)
            ax2.set_title('Distribución de Ciclos', fontsize=13, fontweight='bold')
            ax2.legend(loc='upper right', fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Info adicional
            textstr = f'Min: {min_cycles}\nMax: {max_cycles}\nStd: {np.std(cycles):.1f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax2.text(0.98, 0.97, textstr, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right', bbox=props)
            
            plt.tight_layout()
            
            if show:
                plt.show()
            
            if return_fig:
                return fig, (ax1, ax2)
            
        except Exception as e:
            print(f"Error al plotear historial de ciclos multigrid: {e}")
            return None

    def plot_divergence_history(self, show=True, return_fig=False):
        """
        Grafica la evolución de la divergencia máxima a lo largo del tiempo.
        
        Para fluidos incompresibles, div(u) = 0 es una condición exacta que debe
        cumplirse en todo momento. Una divergencia cercana a la precisión de máquina
        indica que el solver de Poisson está funcionando correctamente.
        
        Valores típicamente aceptables: < 1e-6
        
        Parámetros:
            show: si True, muestra la figura
            return_fig: si True, retorna la figura en lugar de mostrarla
        """
        # Convertir a NumPy
        div = cp.asnumpy(self.divvector).flatten()
        
        # Crear eje X en iteraciones reales
        x_iter = np.arange(len(div)) * self.guardado
        
        # Crear figura
        fig = plt.figure(figsize=(10, 6))
        plt.semilogy(x_iter, div, linewidth=2, color='red', label='max|div(u)|')
        plt.xlabel(f"Iteración (guardado cada {self.guardado})", fontsize=11)
        plt.ylabel("Divergencia máxima |div(u)|", fontsize=11)
        plt.title("Evolución de la Divergencia (Incompresibilidad)", fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3, which='both')
        plt.legend(fontsize=10)
        
        # Añadir línea de referencia
        if len(div) > 0:
            mean_div = np.mean(div)
            plt.axhline(y=mean_div, color='orange', linestyle='--', 
                       linewidth=1.5, label=f'Promedio: {mean_div:.2e}')
            plt.legend(fontsize=10)
        
        plt.tight_layout()
        
        if return_fig:
            return fig
        elif show:
            plt.show()
        else:
            plt.close()

    def save_frame(self, out_dir, iteration, kind="velocity", scale=1.0):
        """
        Guarda una imagen del estado actual en out_dir con nombre frame_XXXX.png.
        kind: "velocity" (mapa |u|) o "traction" (quiver presión/viscosa en frontera).
        scale: factor de escala para flechas de tracción.
        """
        # Crear carpeta si no existe
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"frame_{iteration:06d}.png")

        # Guardar backend actual y cambiar a Agg temporalmente (sin GUI, más seguro)
        backend_original = matplotlib.get_backend()
        if backend_original != 'agg':
            matplotlib.use('Agg', force=True)
        
        fig = None
        try:
            if kind == "velocity":
                # Magnitud de la velocidad
                speed = cp.sqrt(self.u**2 + self.v**2)
                speed_np = cp.asnumpy(speed)
                X_np = cp.asnumpy(self.XX)
                Y_np = cp.asnumpy(self.YY)
                ar = self.Ly / self.Lx
                fig = plt.figure(figsize=(10, 10 * ar))
                ax = plt.gca()
                im = ax.pcolormesh(X_np, Y_np, speed_np, cmap='rainbow', shading='auto')
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlim(0, self.Lx)
                ax.set_ylim(0, self.Ly)
                plt.colorbar(im, label="|u|")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.title(f"Velocidad |u| (iter {iteration})")
                plt.savefig(fname, dpi=600, bbox_inches="tight")
            elif kind == "traction":
                # Necesita mu; si no lo tienes global, ajusta el valor al llamar
                # Usamos mu=1 por defecto; puedes pasar el real vía una variante si lo prefieres
                mu = 1.0
                data = self.compute_surface_traction_field(mu)
                Xb = cp.asnumpy(data["Xb"]); Yb = cp.asnumpy(data["Yb"])
                Tx_p = cp.asnumpy(data["Tx_p"]); Ty_p = cp.asnumpy(data["Ty_p"])
                Tx_v = cp.asnumpy(data["Tx_v"]); Ty_v = cp.asnumpy(data["Ty_v"])
                fig = plt.figure(figsize=(6,6))
                speed = cp.sqrt(self.u**2 + self.v**2)
                X_np = cp.asnumpy(self.XX)
                Y_np = cp.asnumpy(self.YY)
                plt.pcolormesh(X_np, Y_np, cp.asnumpy(speed), cmap='Greys', alpha=0.3, shading='auto')
                plt.quiver(Xb, Yb, scale*Tx_p, scale*Ty_p, color="red", angles="xy", scale_units="xy", scale=1,
                           label="Tracción presión")
                plt.quiver(Xb, Yb, scale*Tx_v, scale*Ty_v, color="blue", angles="xy", scale_units="xy", scale=1,
                           label="Tracción viscosa")
                plt.legend()
                plt.xlabel("x"); plt.ylabel("y")
                plt.title(f"Tracción en superficie (iter {iteration})")
                plt.savefig(fname, dpi=600, bbox_inches="tight")
            else:
                raise ValueError("kind debe ser 'velocity' o 'traction'")
        finally:
            # Siempre cerrar la figura, incluso si hay error
            if fig is not None:
                plt.close(fig)
            else:
                plt.close()
            
            # Restaurar backend original
            if backend_original != 'agg':
                matplotlib.use(backend_original, force=True)

    '''Fuerzas'''

    def _solid_boundary_mask(self, use_diagonals=True):
        """
        Devuelve máscara de celdas de fluido adyacentes al sólido (frontera del sólido).
        """
        solid = self.solid
        fluid = ~solid

        # Vecinos ortogonales
        nb = cp.zeros_like(solid, dtype=cp.bool_)
        nb |= cp.roll(solid, 1, axis=0)   # S
        nb |= cp.roll(solid, -1, axis=0)  # N
        nb |= cp.roll(solid, 1, axis=1)   # W
        nb |= cp.roll(solid, -1, axis=1)  # E

        if use_diagonals:
            nb |= cp.roll(cp.roll(solid, 1, axis=0), 1, axis=1)    # SW
            nb |= cp.roll(cp.roll(solid, 1, axis=0), -1, axis=1)   # SE
            nb |= cp.roll(cp.roll(solid, -1, axis=0), 1, axis=1)   # NW
            nb |= cp.roll(cp.roll(solid, -1, axis=0), -1, axis=1)  # NE

        boundary = fluid & nb

        # Excluir bordes del dominio para evitar artefactos
        boundary[0, :] = False
        boundary[-1, :] = False
        boundary[:, 0] = False
        boundary[:, -1] = False

        return boundary

    def _signed_distance_and_normals(self):
        """
        Calcula distancia signed del sólido y normales unitarias hacia el fluido.
        """
        # distance_transform_edt espera booleanos en CPU o GPU (cupyx soporta GPU)
        d_out = distance_transform_edt(~self.solid)  # distancia al fluido
        d_in  = distance_transform_edt(self.solid)   # distancia al sólido
        sd = d_out - d_in  # positivo fuera del sólido

        # Gradiente central en 2D (dx=dy)
        dx = self.dx; dy = self.dy
        # Usamos diferencias centradas internas; clamp en bordes
        gx = cp.zeros_like(sd, dtype=cp.float32)
        gy = cp.zeros_like(sd, dtype=cp.float32)
        gx[:, 1:-1] = (sd[:, 2:] - sd[:, :-2]) / (2.0 * dx)
        gy[1:-1, :] = (sd[2:, :] - sd[:-2, :]) / (2.0 * dy)

        # Normal hacia el fluido (gradiente de sd ya apunta del sólido al fluido)
        mag = cp.sqrt(gx*gx + gy*gy) + cp.float32(1e-12)
        nx = gx / mag
        ny = gy / mag

        return sd, nx, ny

    # NOTA: compute_surface_forces, compute_surface_forces2,
    #       compute_surface_forces_layers, compute_drag_lift_layers
    #       ELIMINADOS — usar compute_surface_forces_definitive y compute_drag_lift

    def compute_drag_lift(self, mu, rho=1.0, n_extrap_layers=5):
        """
        Calcula Drag y Lift en el sistema aerodinámico (relativo al flujo libre).
        Hace la transformación de coordenadas desde fuerzas en ejes del cuerpo (Fx, Fy)
        a ejes aerodinámicos usando el ángulo de ataque.
        
        Drag: paralelo al flujo libre
        Lift: perpendicular al flujo libre
        """
        res = self.compute_surface_forces_definitive(mu, rho=rho, n_extrap_layers=n_extrap_layers)
        
        # Descomponer en ejes viento usando ángulo real del flujo libre
        alpha_rad = self._get_freestream_angle_rad()
        cos_a = np.cos(alpha_rad)
        sin_a = np.sin(alpha_rad)
        
        # Fuerzas totales
        Fx = res["Fx"]
        Fy = res["Fy"]
        Drag = Fx * cos_a + Fy * sin_a
        Lift = -Fx * sin_a + Fy * cos_a
        
        # Componentes de presión
        Fx_p = res["Fx_p"]
        Fy_p = res["Fy_p"]
        Drag_p = Fx_p * cos_a + Fy_p * sin_a
        Lift_p = -Fx_p * sin_a + Fy_p * cos_a
        
        # Componentes viscosas
        Fx_v = res["Fx_v"]
        Fy_v = res["Fy_v"]
        Drag_v = Fx_v * cos_a + Fy_v * sin_a
        Lift_v = -Fx_v * sin_a + Fy_v * cos_a
        
        return {
            "Drag": Drag, "Lift": Lift,
            "Drag_p": Drag_p, "Lift_p": Lift_p,
            "Drag_v": Drag_v, "Lift_v": Lift_v,
            "Fx": Fx, "Fy": Fy
        }
    def compute_surface_forces_definitive(self, mu, rho=1.0, return_per_face=False, return_cp=True, n_extrap_layers=5):
        """
        Versión definitiva de la integral de esfuerzos sobre el perfil.
        - Usa extrapolación de presión y viscosidad desde múltiples capas para capturar gradientes lejanos en flujos turbulentos.
        - Integra solo sobre la primera capa (frontera) para evitar inflación artificial.
        - n_extrap_layers: número de capas para extrapolación (default 5, posiciones 0.5, 1.5, ..., 4.5)

        Parámetros:
            mu: viscosidad dinámica
            rho: densidad
            return_per_face: si True devuelve arrays por-cara
            return_cp: si True calcula Cp
            n_extrap_layers: capas para extrapolación (2-10 recomendado)

        Retorna diccionario con Fx,Fy, etc.
        """
        # 1) frontera y normales
        boundary = self._solid_boundary_mask(use_diagonals=True)
        _, nx_all, ny_all = self._signed_distance_and_normals()

        # 2) elementos y corrección para diagonales
        solid = self.solid
        dx = self.dx; dy = self.dy
        dx_f = cp.float32(dx)
        # diag mask: detecta celdas frontera que tocan el sólido por esquina
        diag_mask = (
            (cp.roll(cp.roll(solid, 1, axis=0), 1, axis=1)) |
            (cp.roll(cp.roll(solid, 1, axis=0), -1, axis=1)) |
            (cp.roll(cp.roll(solid, -1, axis=0), 1, axis=1)) |
            (cp.roll(cp.roll(solid, -1, axis=0), -1, axis=1))
        ) & boundary
        ds = cp.where(diag_mask, cp.float32(dx * np.sqrt(2.0)), dx_f)

        # 3) construir coordenadas indices para muestreo en caras
        JJ, II = self.JJ, self.II
        j_face = JJ + 0.5 * nx_all
        i_face = II + 0.5 * ny_all

        # 4) interpolar en múltiples capas para extrapolación
        positions = cp.array([0.5 + i * 1.0 for i in range(n_extrap_layers)], dtype=cp.float32)
        p_layers = []
        for pos in positions:
            j_face_k = JJ + pos * nx_all
            i_face_k = II + pos * ny_all
            p_k = self._bilinear_interpolate(self.p, j_face_k, i_face_k)
            p_layers.append(p_k)

        # Extrapolación de p_wall usando gradiente de las capas extremas (simple y robusto)
        p1 = p_layers[0]
        pN = p_layers[-1]
        dist = positions[-1] - positions[0]
        grad_p = (pN - p1) / dist
        p_wall = p1 - grad_p * 0.5  # extrapolar a pos=0

        # mu_eff en la cara: molecular + turbulenta (si WALE activo)
        if self.usar_wale:
            nu_t_field = self.compute_wale_viscosity()
            nu_t_face = self._bilinear_interpolate(nu_t_field, j_face, i_face)
            mu_eff_face = cp.float32(mu) + cp.float32(rho) * nu_t_face
        else:
            mu_eff_face = cp.float32(mu)

        # 5) gradientes en centros y muestreo en caras (usando capa 1)
        u = self.u; v = self.v
        inv_dx = cp.float32(1.0 / dx); inv_dy = cp.float32(1.0 / dy)
        du_dx = cp.zeros_like(u, dtype=cp.float32); du_dy = cp.zeros_like(u, dtype=cp.float32)
        dv_dx = cp.zeros_like(v, dtype=cp.float32); dv_dy = cp.zeros_like(v, dtype=cp.float32)
        du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) * (0.5 * inv_dx)
        du_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) * (0.5 * inv_dy)
        dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) * (0.5 * inv_dx)
        dv_dy[1:-1, :] = (v[2:, :] - v[:-2, :]) * (0.5 * inv_dy)

        # muestrear gradientes en caras
        du_dx_f = self._bilinear_interpolate(du_dx, j_face, i_face)
        du_dy_f = self._bilinear_interpolate(du_dy, j_face, i_face)
        dv_dx_f = self._bilinear_interpolate(dv_dx, j_face, i_face)
        dv_dy_f = self._bilinear_interpolate(dv_dy, j_face, i_face)

        # 6) componentes de S en las caras
        nx_face = nx_all; ny_face = ny_all
        S11 = du_dx_f
        S12 = 0.5 * (du_dy_f + dv_dx_f)
        S22 = dv_dy_f

        # tracción viscosa con mu_eff (signo negativo: n apunta hacia fluido, queremos fuerza sobre sólido)
        Tx_v = mu_eff_face * (2.0 * S11 * nx_face + 2.0 * S12 * ny_face)
        Ty_v = mu_eff_face * (2.0 * S12 * nx_face + 2.0 * S22 * ny_face)

        # 7) enmascarar e integrar solo sobre boundary (primera capa)
        w = boundary.astype(cp.float32)
        # ⭐ CORRECCIÓN: Fuerza de presión sobre sólido = -∫ p·n dS
        # Ambas componentes deben tener signo negativo (n apunta hacia fluido)
        Tx_p = -p_wall * nx_face
        Ty_p = -p_wall * ny_face
        Fx_p = cp.sum(Tx_p * w * ds)
        Fy_p = cp.sum(Ty_p * w * ds)
        Fx_v = cp.sum(Tx_v * w * ds)
        Fy_v = cp.sum(Ty_v * w * ds)

        Fx = Fx_p + Fx_v
        Fy = Fy_p + Fy_v

        result = {
            "Fx": float(Fx), "Fy": float(Fy),
            "Fx_p": float(Fx_p), "Fy_p": float(Fy_p),
            "Fx_v": float(Fx_v), "Fy_v": float(Fy_v),
        }

        # 8) Calcular Cp si solicitado (usando p_wall extrapolado)
        if return_cp:
            band = max(1, int(min(self.nx, self.ny) * 0.05))
            mask_edges = cp.zeros_like(self.p, dtype=cp.bool_)
            mask_edges[:band, :] = True
            mask_edges[-band:, :] = True
            mask_edges[:, :band] = True
            mask_edges[:, -band:] = True
            try:
                p_ref = cp.mean(self.p[mask_edges])
                U_ref = cp.sqrt(cp.mean((self.u[mask_edges]**2 + self.v[mask_edges]**2)))
            except Exception:
                p_ref = cp.mean(self.p)
                U_ref = cp.sqrt(cp.mean(self.u**2 + self.v**2))
            eps = cp.float32(1e-12)
            denom = 0.5 * cp.float32(rho) * (U_ref**2) + eps
            cp_face = (p_wall - p_ref) / denom

            x_face = j_face * cp.float32(dx)
            y_face = i_face * cp.float32(dy)

            Xb = x_face[boundary]; Yb = y_face[boundary]
            Cp_b = cp_face[boundary]
            extr_mask = (ny_face > 0) & boundary
            intro_mask = (ny_face <= 0) & boundary
            X_ex = x_face[extr_mask]; Y_ex = y_face[extr_mask]; Cp_ex = cp_face[extr_mask]
            X_in = x_face[intro_mask]; Y_in = y_face[intro_mask]; Cp_in = cp_face[intro_mask]

            result.update({
                "Cp_face": Cp_b,
                "Xb": Xb, "Yb": Yb,
                "Cp_extrados": Cp_ex, "X_extrados": X_ex, "Y_extrados": Y_ex,
                "Cp_intrados": Cp_in,  "X_intrados": X_in,  "Y_intrados": Y_in,
            })

            if return_per_face:
                result.update({
                    "Tx_p_face": Tx_p[boundary], "Ty_p_face": Ty_p[boundary],
                    "Tx_v_face": Tx_v[boundary], "Ty_v_face": Ty_v[boundary],
                })

        return result

    def update_cp_profile(self, mu=1.0, rho=1.0):
        """
        Muestrea Cp instantáneo en las caras frontera y actualiza la suma y contador
        para el promedio temporal por punto de cuerda (bins en `self.cp_bins`).
        """
        data = self.compute_surface_forces_definitive(mu=mu, rho=rho, return_per_face=True, return_cp=True)
        # obtener referencia de cuerda usando todas las caras (si existen)
        Xb = data.get('Xb', None)
        if Xb is None or Xb.size == 0:
            return
        x_all = Xb
        x_min = float(cp.min(x_all))
        x_max = float(cp.max(x_all))
        chord = x_max - x_min if x_max > x_min else 1.0

        nbins = int(self.cp_bins)

        # Extrados
        X_ex = data.get('X_extrados', None)
        Cp_ex = data.get('Cp_extrados', None)
        if X_ex is not None and Cp_ex is not None and X_ex.size > 0:
            x_norm_ex = (X_ex - x_min) / chord
            idx_ex = cp.floor(x_norm_ex * cp.float32(nbins - 1)).astype(cp.int32)
            idx_ex = cp.clip(idx_ex, 0, nbins - 1)
            sums_ex = cp.bincount(idx_ex, weights=Cp_ex, minlength=nbins).astype(cp.float32)
            counts_ex = cp.bincount(idx_ex, minlength=nbins).astype(cp.float32)
            self.cp_profile_sum_ex += sums_ex
            self.cp_profile_count_ex += counts_ex

        # Intrados
        X_in = data.get('X_intrados', None)
        Cp_in = data.get('Cp_intrados', None)
        if X_in is not None and Cp_in is not None and X_in.size > 0:
            x_norm_in = (X_in - x_min) / chord
            idx_in = cp.floor(x_norm_in * cp.float32(nbins - 1)).astype(cp.int32)
            idx_in = cp.clip(idx_in, 0, nbins - 1)
            sums_in = cp.bincount(idx_in, weights=Cp_in, minlength=nbins).astype(cp.float32)
            counts_in = cp.bincount(idx_in, minlength=nbins).astype(cp.float32)
            self.cp_profile_sum_in += sums_in
            self.cp_profile_count_in += counts_in

    def get_cp_profile_mean(self):
        """
        Retorna (x_array, cp_mean_extrados, cp_mean_intrados) como numpy arrays.
        Si no hay conteos devuelve arrays vacíos.
        """
        import numpy as np
        counts_ex = self.cp_profile_count_ex
        counts_in = self.cp_profile_count_in
        total_counts = int(cp.sum(counts_ex + counts_in).item())
        if total_counts == 0:
            return np.array([]), np.array([]), np.array([])

        mean_ex = cp.zeros_like(self.cp_profile_sum_ex)
        mean_in = cp.zeros_like(self.cp_profile_sum_in)
        mask_ex = counts_ex > 0
        mask_in = counts_in > 0
        mean_ex[mask_ex] = self.cp_profile_sum_ex[mask_ex] / (counts_ex[mask_ex] + 1e-12)
        mean_in[mask_in] = self.cp_profile_sum_in[mask_in] / (counts_in[mask_in] + 1e-12)
        x = cp.asnumpy(self.cp_profile_x)
        return x, cp.asnumpy(mean_ex), cp.asnumpy(mean_in)

    def plot_cp_vs_chord(self, mu=1.0, rho=1.0, normalize=True, show=True, save_path=None, return_data=False):
        """
        Grafica Cp a lo largo de la cuerda del perfil usando datos de la máscara del sólido.
        - Llama a `compute_surface_forces_definitive` para obtener Cp en caras.
        - Normaliza por la cuerda si `normalize=True` (x/c en eje horizontal).
        - Devuelve (opcional) los arrays numpy para intradós y extradós.
        """
        # Si hay perfil promedio acumulado (extrados o intrados), usarlo. Si no, calcular instantáneo.
        has_history = bool((cp.sum(self.cp_profile_count_ex) + cp.sum(self.cp_profile_count_in)).item() > 0)

        import numpy as _np
        import matplotlib.pyplot as _plt

        fig = _plt.figure(figsize=(8, 4))
        ax = fig.gca()

        if has_history:
            # Recuperar promedios separados por cara (extrados / intrados)
            x_plot, mean_ex, mean_in = self.get_cp_profile_mean()
            if x_plot.size == 0:
                ax.text(0.5, 0.5, 'No hay datos promedio de Cp', ha='center')
            else:
                if normalize:
                    xlabel = 'x/c'
                    x_plot_plot = x_plot
                else:
                    xlabel = 'x (m)'
                    x_plot_plot = x_plot * float(self.Lx)
                ax.plot(x_plot_plot, mean_ex, '-r', linewidth=2, label='Cp promedio extrados')
                ax.plot(x_plot_plot, mean_in, '-b', linewidth=2, label='Cp promedio intrados')
                ax.set_xlabel(xlabel)
                ax.set_ylabel('Cp')
                ax.set_title('Cp promedio sobre la cuerda')
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.invert_yaxis()

        else:
            # calcular instantáneo usando la función definitiva
            data = self.compute_surface_forces_definitive(mu=mu, rho=rho, return_per_face=True, return_cp=True)
            Xb = data.get('Xb', cp.array([], dtype=cp.float32))
            Cp_b = data.get('Cp_face', cp.array([], dtype=cp.float32))
            if Xb.size > 0:
                x_phys = cp.asnumpy(Xb)
                cp_vals = cp.asnumpy(Cp_b)
                # normalizar por cuerda
                x_min = float(x_phys.min()); x_max = float(x_phys.max()); chord = x_max - x_min if x_max>x_min else 1.0
                if normalize:
                    x_plot = (x_phys - x_min) / chord
                    xlabel = 'x/c'
                else:
                    x_plot = x_phys - x_min
                    xlabel = 'x (m)'
                idx = _np.argsort(x_plot)
                ax.plot(x_plot[idx], cp_vals[idx], '-o')
                ax.set_xlabel(xlabel); ax.set_ylabel('Cp'); ax.set_title('Cp instantáneo sobre la cuerda')
                ax.grid(True, alpha=0.3); ax.invert_yaxis()
            else:
                ax.text(0.5,0.5,'No hay datos de cara para Cp', ha='center')

        if save_path:
            _plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            _plt.show()
        else:
            _plt.close()

        if return_data:
            if has_history:
                x, mean_ex, mean_in = self.get_cp_profile_mean()
                return {'x': x, 'cp_mean_extrados': mean_ex, 'cp_mean_intrados': mean_in, 'fig': fig}
            else:
                return {'x': None, 'cp_mean_extrados': None, 'cp_mean_intrados': None, 'fig': fig}


def generar_graficos_y_outputs(mesh_gruesa: 'Mesh', iteraciones: int, guardado: int, it_actual: int, 
                                tiempo_fisico_acumulado: float, rho: float, U_inf: float, chord: float, nu: float, mu: float,
                                graficos: bool = True, verbose: bool = True):
    """
    Genera todos los gráficos y reportes de la simulación con los datos actuales.
    Puede ser llamada durante la simulación o al final.
    
    Parámetros:
        mesh_gruesa: malla principal con datos de simulación
        iteraciones: número total de iteraciones programadas
        guardado: frecuencia de guardado
        it_actual: iteración actual
        tiempo_fisico_acumulado: tiempo físico transcurrido
        rho, U_inf, chord, nu, mu: parámetros físicos
        graficos: si True, genera visualizaciones
        verbose: si True, imprime reportes detallados
    """
    # Calcular índice válido para promedios (hasta donde hay datos)
    idx_valido = min((it_actual // guardado) + 1, len(mesh_gruesa.cdvector))
    
    if idx_valido <= 1:
        return
    
    # Excluir 10% inicial para medias (o al menos 1 elemento)
    inicio_calculo = max(1, int(idx_valido * 0.1))
    
    cd_medio = cp.mean(mesh_gruesa.cdvector[inicio_calculo:idx_valido]).get()
    cl_medio = cp.mean(mesh_gruesa.clvector[inicio_calculo:idx_valido]).get()
    cd_final = mesh_gruesa.cdvector[idx_valido-1].get()
    cl_final = mesh_gruesa.clvector[idx_valido-1].get()
    Re = (U_inf * chord) / nu
    
    # Calcular fuerzas detalladas finales/actuales
    forces_final = mesh_gruesa.compute_drag_lift(mu, rho=rho, n_extrap_layers=5)
    
    # Fuerzas en unidades físicas
    Fx_total = forces_final['Drag']
    Fy_total = forces_final['Lift']
    Fx_presion = forces_final['Drag_p']
    Fy_presion = forces_final['Lift_p']
    Fx_friccion = forces_final['Drag_v']
    Fy_friccion = forces_final['Lift_v']
    
    # Coeficientes por componentes
    Cd_presion = 2 * Fx_presion / (rho * U_inf**2 * chord)
    Cd_friccion = 2 * Fx_friccion / (rho * U_inf**2 * chord)
    Cl_presion = 2 * Fy_presion / (rho * U_inf**2 * chord)
    Cl_friccion = 2 * Fy_friccion / (rho * U_inf**2 * chord)
    
    ld_medio = cl_medio / cd_medio if abs(cd_medio) > 1e-12 else 0.0
    print(f"\n--- Resultados (Re={Re:.0f}, t={tiempo_fisico_acumulado:.4f}s) ---")
    print(f"  Cl medio = {cl_medio:.6f}   Cd medio = {cd_medio:.6f}   L/D = {ld_medio:.2f}")
    print(f"{'─'*52}")
    print(f"  Descomposición de fuerzas (valor instantáneo final):")
    print(f"  {'':20s}  {'Drag (X)':>12s}  {'Lift (Y)':>12s}")
    print(f"  {'─'*50}")
    print(f"  {'TOTAL':20s}  {Fx_total:>12.6f}  {Fy_total:>12.6f}  N/m")
    print(f"  {'  Presión':20s}  {Fx_presion:>12.6f}  {Fy_presion:>12.6f}  N/m")
    print(f"  {'  Viscosa (fricción)':20s}  {Fx_friccion:>12.6f}  {Fy_friccion:>12.6f}  N/m")
    print(f"  {'─'*50}")
    print(f"  {'Cd total':20s}  {2*Fx_total/(rho*U_inf**2*chord):>12.6f}")
    print(f"  {'  Cd presión':20s}  {Cd_presion:>12.6f}  ({100*Cd_presion/(2*Fx_total/(rho*U_inf**2*chord)+1e-30):.1f}%)")
    print(f"  {'  Cd fricción':20s}  {Cd_friccion:>12.6f}  ({100*Cd_friccion/(2*Fx_total/(rho*U_inf**2*chord)+1e-30):.1f}%)")
    print(f"  {'Cl total':20s}  {2*Fy_total/(rho*U_inf**2*chord):>12.6f}")
    print(f"  {'  Cl presión':20s}  {Cl_presion:>12.6f}  ({100*Cl_presion/(2*Fy_total/(rho*U_inf**2*chord)+1e-30):.1f}%)")
    print(f"  {'  Cl fricción':20s}  {Cl_friccion:>12.6f}  ({100*Cl_friccion/(2*Fy_total/(rho*U_inf**2*chord)+1e-30):.1f}%)")
    print(f"{'─'*52}")
    # retener variables para posible uso posterior
    
    if graficos:    
        mesh_gruesa.visualize_velocity()
        mesh_gruesa.plot_forces_over_time()
        mesh_gruesa.visualize_surface_traction(mu)
        mesh_gruesa.plot_convergence_history()
        mesh_gruesa.plot_cp_vs_chord(mu, rho, normalize=True, show=True)
        mesh_gruesa.plot_divergence_history()
        mesh_gruesa.plot_multigrid_cycles_history()



        mesh_gruesa.plot_multigrid_cycles_history()


def _generar_graficos_polar(polar_data, filepath, carpeta_salida="."):
    """
    Genera 3 gráficos de la polar: CL vs alpha, CD vs alpha, eficiencia (CL/CD) vs alpha.
    Los guarda como PNG con nombre: {perfil}_{fecha}_{hora}_polar_{tipo}.png
    """
    if len(polar_data) < 2:
        print("⚠ Polar con menos de 2 puntos, no se generan gráficos polares.")
        return

    nombre_perfil = os.path.basename(filepath).replace(" ", "_")
    fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefijo = os.path.join(carpeta_salida, f"{nombre_perfil}_{fecha}_polar")

    alphas = np.array([p['alpha'] for p in polar_data])
    cds = np.array([p['Cd'] for p in polar_data])
    cls = np.array([p['Cl'] for p in polar_data])
    eficiencia = np.where(np.abs(cds) > 1e-12, cls / cds, np.nan)

    # --- CL vs Alpha ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, cls, 'o-', color='#1f77b4', linewidth=2, markersize=6)
    ax.set_xlabel('Ángulo de ataque α [°]', fontsize=12)
    ax.set_ylabel('CL', fontsize=12)
    ax.set_title(f'Polar CL vs α — {nombre_perfil}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    fig.tight_layout()
    ruta_cl = f"{prefijo}_CL.png"
    fig.savefig(ruta_cl, dpi=150)
    plt.close(fig)

    # --- CD vs Alpha ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, cds, 's-', color='#d62728', linewidth=2, markersize=6)
    ax.set_xlabel('Ángulo de ataque α [°]', fontsize=12)
    ax.set_ylabel('CD', fontsize=12)
    ax.set_title(f'Polar CD vs α — {nombre_perfil}', fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    ruta_cd = f"{prefijo}_CD.png"
    fig.savefig(ruta_cd, dpi=150)
    plt.close(fig)

    # --- Eficiencia CL/CD vs Alpha ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, eficiencia, 'D-', color='#2ca02c', linewidth=2, markersize=6)
    ax.set_xlabel('Ángulo de ataque α [°]', fontsize=12)
    ax.set_ylabel('CL / CD', fontsize=12)
    ax.set_title(f'Eficiencia aerodinámica CL/CD vs α — {nombre_perfil}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    fig.tight_layout()
    ruta_ef = f"{prefijo}_eficiencia.png"
    fig.savefig(ruta_ef, dpi=150)
    plt.close(fig)

    print(f"📊 Gráficos polares guardados:")
    print(f"   CL vs α:        {ruta_cl}")
    print(f"   CD vs α:        {ruta_cd}")
    print(f"   CL/CD vs α:     {ruta_ef}")


def _guardar_punto_polar(polar_data, mesh, mu, rho, U_inf, chord,
                         alpha, iter_inicio, iter_fin, guardado, descarte=0.3):
    """
    Calcula Cd/Cl medio sobre el rango [iter_inicio, iter_fin), descartando
    la fracción 'descarte' inicial (transitorio tras cambio de alpha).
    Usa los valores ya almacenados en cdvector/clvector.
    """
    idx_ini = iter_inicio // guardado
    idx_fin = iter_fin // guardado
    n_total = idx_fin - idx_ini
    
    if n_total > 2:
        n_descartar = max(1, int(n_total * descarte))
        idx_media_ini = idx_ini + n_descartar
        idx_media_fin = idx_fin
        
        cd_arr = cp.asnumpy(mesh.cdvector[idx_media_ini:idx_media_fin])
        cl_arr = cp.asnumpy(mesh.clvector[idx_media_ini:idx_media_fin])
        
        # Filtrar ceros (iteraciones no alcanzadas)
        mask = (cd_arr != 0) | (cl_arr != 0)
        if np.any(mask):
            cd_arr = cd_arr[mask]
            cl_arr = cl_arr[mask]
        
        cd_mean = float(np.mean(cd_arr)) if len(cd_arr) > 0 else 0.0
        cl_mean = float(np.mean(cl_arr)) if len(cl_arr) > 0 else 0.0
        cd_std = float(np.std(cd_arr)) if len(cd_arr) > 0 else 0.0
        cl_std = float(np.std(cl_arr)) if len(cl_arr) > 0 else 0.0
    else:
        # Muy pocas muestras: usar valor instantáneo
        forces = mesh.compute_drag_lift(mu, rho=rho, n_extrap_layers=5)
        cd_mean = 2 * forces['Drag'] / (rho * U_inf**2 * chord)
        cl_mean = 2 * forces['Lift'] / (rho * U_inf**2 * chord)
        cd_std = 0.0
        cl_std = 0.0

    # Componentes instantáneas (último valor)
    forces_now = mesh.compute_drag_lift(mu, rho=rho, n_extrap_layers=5)
    q = rho * U_inf**2 * chord

    polar_data.append({
        'alpha': float(alpha),
        'Cd': cd_mean,
        'Cl': cl_mean,
        'Cd_std': cd_std,
        'Cl_std': cl_std,
        'Cd_p': 2 * forces_now['Drag_p'] / q,
        'Cd_v': 2 * forces_now['Drag_v'] / q,
        'Cl_p': 2 * forces_now['Lift_p'] / q,
        'Cl_v': 2 * forces_now['Lift_v'] / q,
        'iter_inicio': int(iter_inicio),
        'iter_fin': int(iter_fin),
        'n_muestras': int(len(cd_arr)) if n_total > 2 else 1,
        'descarte': float(descarte)
    })


def main(
    # Parámetros temporales
    CFL=0.8,
    
    # Geometría del perfil
    alpha_deg=5,
    chord=1.0,
    filepath="NACA_0012",
    
    # Tamaño del dominio
    Lx=7,
    Ly=6,
    
    # Resolución de malla
    dx_min=0.01,       # Espaciado mínimo (zona fina alrededor del perfil)
    dy_min=None,       # Si None, se usa dx_min
    factor_expansion=1.05,  # Factor geométrico de crecimiento
    ancho_zona_fina_x=None,  # Ancho de zona fina en X (None → 0.1*Lx)
    ancho_zona_fina_y=None,  # Ancho de zona fina en Y (None → 0.1*Ly)
    dx_max=None,       # Espaciado máximo (None → 20*dx_min)
    dy_max=None,       # Espaciado máximo Y (None → 20*dy_min)
    usar_wale=False,   # Si True, activa modelo de turbulencia WALE
    
    # Posición del perfil
    cx=2,
    cy=None,  # Si None, se centra verticalmente
    
    # Condiciones iniciales
    p0=0,      # Pa
    v0x=5,     # m/s
    v0y=0.0,   # m/s
    
    # Propiedades del fluido
    rho=1.225,   # kg/m^3
    nu=1.5e-5,   # m^2/s (viscosidad cinemática)
    divergencia=1e-1,
    
    # Condiciones de frontera
    boundary_left=("inflow", None),
    boundary_top=("slip", None),
    boundary_bottom=("slip", None),
    # outflow con value=0.0: fija p=0 como Dirichlet en la salida.
    # Esto estabiliza la proyección: evita el modo amplificante Neumann+Neumann
    # que produce picos u = 2*u[n-2] - u[n-3] en la frontera de salida.
    boundary_right=("outflow", 0.0),
    
    # Parámetros de simulación
    guardado=50,
    iteraciones=2000,

    # Tuning de proyección multigrid (por defecto conserva comportamiento actual)
    mg_max_outer=8,
    mg_cycles_per_outer=5,
    mg_pre_suavizado=3,
    mg_post_suavizado=3,
    mg_guard_residual_every_outer=True,
    mg_adaptive_outer0_cycles=False,
    mg_apply_ibm_each_outer=True,
    mg_rollback_on_nan=True,
    mg_compute_div_after=True,
    mg_modo_rapido=False,
    mg_modo_turbo=False,
    
    # Opciones de visualización y guardado
    save_frames=False,
    frames_dir_grueso=None,
    graficos=False,
    
    # Control de convergencia
    stop_on_convergence=True,
    
    # Plan polar automático
    plan_polar=None,
    polar_descarte=0.3,
    
    # Vista en tiempo real
    live_view=False,
    
    # Visualización de malla
    mostrar_malla=False,

    # Diagnóstico detallado de spikes (costoso; usar solo al depurar)
    debug_spikes=False
):
    # Procesar valores por defecto
    if dy_min is None:
        dy_min = dx_min
    if cy is None:
        cy = Ly / 2.0
    
    # Calcular viscosidad dinámica
    mu = rho * nu

    # Perfil rápido opcional de MG (sin tocar defaults del solver base)
    if mg_modo_rapido:
        mg_max_outer = min(mg_max_outer, 5)
        mg_cycles_per_outer = min(mg_cycles_per_outer, 4)
        mg_pre_suavizado = min(mg_pre_suavizado, 2)
        mg_post_suavizado = min(mg_post_suavizado, 2)
        mg_guard_residual_every_outer = False
        mg_adaptive_outer0_cycles = True
        mg_apply_ibm_each_outer = False
        mg_rollback_on_nan = False
        mg_compute_div_after = False
        print("[MG-fast] activo: max_outer<=5, cycles<=4, pre/post<=2, guard outer0, adaptive outer0 ON, IBM por outer OFF, rollback OFF")

    # Perfil turbo opcional: más agresivo (prioriza rendimiento)
    if mg_modo_turbo:
        mg_max_outer = min(mg_max_outer, 4)
        mg_cycles_per_outer = min(mg_cycles_per_outer, 3)
        mg_pre_suavizado = min(mg_pre_suavizado, 1)
        mg_post_suavizado = min(mg_post_suavizado, 1)
        mg_guard_residual_every_outer = False
        mg_adaptive_outer0_cycles = True
        mg_apply_ibm_each_outer = False
        mg_rollback_on_nan = False
        mg_compute_div_after = False
        print("[MG-turbo] activo: max_outer<=4, cycles<=3, pre/post<=1, guard outer0, IBM por outer OFF, rollback OFF")

    # ============================================================
    # GENERAR MALLA VARIABLE (stretching 1D)
    # ============================================================
    X_1d = generar_malla_estirada(
        L=Lx, x_centro=cx + chord * 0.5,
        dx_min=dx_min, factor_expansion=factor_expansion,
        ancho_zona_fina=ancho_zona_fina_x, dx_max=dx_max
    )
    Y_1d = generar_malla_estirada(
        L=Ly, x_centro=cy,
        dx_min=dy_min if dy_min else dx_min,
        factor_expansion=factor_expansion,
        ancho_zona_fina=ancho_zona_fina_y, dx_max=dy_max
    )
    
    print(f"Malla variable: {len(X_1d)} x {len(Y_1d)} nodos")
    print(f"  X: dx_min={np.min(np.diff(X_1d)):.6f}, dx_max={np.max(np.diff(X_1d)):.6f}")
    print(f"  Y: dy_min={np.min(np.diff(Y_1d)):.6f}, dy_max={np.max(np.diff(Y_1d)):.6f}")

    # Crear malla con densidad variable
    mesh_gruesa = Mesh(Lx, Ly, p0, v0x, v0y, dx_min, dy_min if dy_min else dx_min,
                       usar_wale=usar_wale, X_1d=X_1d, Y_1d=Y_1d)

    # Ángulo de geometría: si hay plan_polar, cargar a 0° (el flujo se rota)
    alpha_geom = 0.0 if (plan_polar is not None and len(plan_polar) > 0) else alpha_deg
    
    # Espesor mínimo del TE
    min_te = 2.0 * dx_min
    
    # Añadir sólido a malla
    mesh_gruesa.load_solids_from_file(
        filepath=filepath,
        chord=chord,
        x_offset=cx, y_offset=cy,
        alpha_deg=alpha_geom, fill=True, plot=False,
        min_te_height=min_te
    )

    # Aplicar condiciones de frontera a malla gruesa
    boundary_type_left, boundary_val_left = boundary_left
    boundary_type_top, boundary_val_top = boundary_top
    boundary_type_bottom, boundary_val_bottom = boundary_bottom
    boundary_type_right, boundary_val_right = boundary_right
    
    # Usar valores por defecto si no se especifican
    if boundary_val_left is None:
        boundary_val_left = (v0x, v0y)
    # Derecha: outflow con presión fija p0 por defecto (Dirichlet), ahora soportado por CG
    if boundary_val_right is None:
        boundary_val_right = p0
    
    mesh_gruesa.set_boundary("left", boundary_type_left, value=boundary_val_left)
    mesh_gruesa.set_boundary("top", boundary_type_top, value=boundary_val_top)
    mesh_gruesa.set_boundary("bottom", boundary_type_bottom, value=boundary_val_bottom)
    mesh_gruesa.set_boundary("right", boundary_type_right, value=boundary_val_right)
    
    # Parámetros físicos
    CFL = CFL
    dt = CFL * min(dx_min, dy_min if dy_min else dx_min) / np.sqrt(v0x**2 + v0y**2)
    print(f"dt = {dt:.6f} s  (CFL={CFL})")  

    guardado = guardado
    iteraciones = iteraciones
    if iteraciones % guardado != 0:
        iteraciones += guardado - (iteraciones % guardado)
    # Inicializar vectores de resultados para ambas mallas
    mesh_gruesa.guardado = guardado
    mesh_gruesa.cdvector = cp.zeros(iteraciones // guardado, dtype=cp.float32)
    mesh_gruesa.clvector = cp.zeros(iteraciones // guardado, dtype=cp.float32)
    mesh_gruesa.divvector = cp.zeros(iteraciones // guardado, dtype=cp.float32)
    mesh_gruesa.clcdvector = cp.zeros(iteraciones // guardado, dtype=cp.float32)
    mesh_gruesa.mg_cycles_vector = cp.zeros(iteraciones, dtype=cp.int32)
    
    # Estadísticas
    print(f"Malla: {mesh_gruesa.nx} x {mesh_gruesa.ny} ({mesh_gruesa.nx * mesh_gruesa.ny:,} celdas)")

    # ============================================================
    # VISUALIZACIÓN DE DENSIDAD DE MALLA (opcional, auto-cierre 5s)
    # ============================================================
    if mostrar_malla:
        _fig_m, _axes_m = plt.subplots(1, 2, figsize=(14, 5))
        # Panel 1: mapa de tamaño de celda (área = dx*dy)
        _dx_1d = np.diff(X_1d)
        _dy_1d = np.diff(Y_1d)
        _cell_area = np.outer(_dy_1d, _dx_1d)  # (ny-1, nx-1)
        _xc = 0.5 * (X_1d[:-1] + X_1d[1:])
        _yc = 0.5 * (Y_1d[:-1] + Y_1d[1:])
        _XC, _YC = np.meshgrid(_xc, _yc)
        _pcm = _axes_m[0].pcolormesh(_XC, _YC, _cell_area, shading='auto', cmap='viridis_r')
        _fig_m.colorbar(_pcm, ax=_axes_m[0], label='Área celda (m²)')
        _axes_m[0].set_title('Densidad de malla (área de celda)')
        _axes_m[0].set_xlabel('X (m)')
        _axes_m[0].set_ylabel('Y (m)')
        _axes_m[0].set_aspect('equal')
        # Dibujar sólido encima
        _solid_np = cp.asnumpy(mesh_gruesa.solid)
        _XX_np = cp.asnumpy(mesh_gruesa.XX)
        _YY_np = cp.asnumpy(mesh_gruesa.YY)
        _axes_m[0].contour(_XX_np, _YY_np, _solid_np.astype(float), levels=[0.5], colors='r', linewidths=1.5)
        # Panel 2: grilla de líneas (cada N líneas para no saturar)
        _max_lines = 80
        _step_x = max(1, len(X_1d) // _max_lines)
        _step_y = max(1, len(Y_1d) // _max_lines)
        for _xi in X_1d[::_step_x]:
            _axes_m[1].axvline(_xi, color='steelblue', linewidth=0.3, alpha=0.7)
        for _yi in Y_1d[::_step_y]:
            _axes_m[1].axhline(_yi, color='steelblue', linewidth=0.3, alpha=0.7)
        _axes_m[1].contour(_XX_np, _YY_np, _solid_np.astype(float), levels=[0.5], colors='r', linewidths=1.5)
        _axes_m[1].set_title(f'Líneas de malla (cada {_step_x}/{_step_y})')
        _axes_m[1].set_xlabel('X (m)')
        _axes_m[1].set_ylabel('Y (m)')
        _axes_m[1].set_aspect('equal')
        _fig_m.suptitle(f'Malla {mesh_gruesa.nx}x{mesh_gruesa.ny} — dx_min={dx_min}, factor={factor_expansion}', fontsize=12)
        _fig_m.tight_layout()
        plt.show(block=False)
        plt.pause(10.0)
        plt.close(_fig_m)

    # ============================================================
    # INSTRUMENTACIÓN DE TIMING
    # ============================================================
    timing_stats = {
        'recalculo_dt': 0.0,
        'adveccion': 0.0,
        'difusion': 0.0,
        'proyeccion': 0.0,
        'guardado': 0.0,
        'otros': 0.0,
        'total_por_paso': []
    }
    
    # ⭐ Variable para acumular tiempo físico simulado
    tiempo_fisico_acumulado = 0.0
    
    # ============================================================
    # SISTEMA DE CONTROL INTERACTIVO: TRIGGERS Y SEÑALES
    # ============================================================
    # Variables de control
    interrupcion_solicitada = False
    plot_solicitado = False
    
    def signal_handler(sig, frame):
        """Manejador para Ctrl+C: finaliza limpiamente con todos los outputs"""
        nonlocal interrupcion_solicitada
        print("\n\n" + "="*70)
        print("⚠️  INTERRUPCIÓN DETECTADA (Ctrl+C)")
        print("="*70)
        print("Finalizando simulación limpiamente...")
        print("Se generarán todos los outputs y gráficos con los datos actuales.")
        print("="*70)
        interrupcion_solicitada = True
    
    # Instalar manejador de señal Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Archivos trigger para control durante ejecución
    trigger_plot = "PLOT_NOW.trigger"
    trigger_stop = "STOP_SIMULATION.trigger"
    trigger_alpha = "CHANGE_ALPHA.trigger"
    
    # Velocidad libre (módulo constante)
    U_inf = np.sqrt(v0x**2 + v0y**2)
    alpha_actual = alpha_deg  # Ángulo de ataque actual
    
    # Registro polar: almacena Cd/Cl convergido para cada alpha
    polar_data = []  # Lista de dicts: {alpha, Cd, Cl, Cd_p, Cd_v, Cl_p, Cl_v, iter_inicio, iter_fin}
    iter_inicio_alpha = 0  # Iteración donde empezó el alpha actual
    archivo_polar = "polar_results.json"
    
    # ============================================================
    # PLAN POLAR AUTOMÁTICO
    # ============================================================
    # Construir diccionario {iteracion_absoluta: nuevo_alpha} a partir de plan_polar
    cambios_alpha_programados = {}
    if plan_polar is not None and len(plan_polar) > 0:
        # Recalcular iteraciones totales según el plan
        total_plan = sum(n for _, n in plan_polar)
        if total_plan > iteraciones:
            iteraciones = total_plan
            # Redimensionar vectores
            mesh_gruesa.cdvector = cp.zeros(iteraciones // guardado + 1, dtype=cp.float32)
            mesh_gruesa.clvector = cp.zeros(iteraciones // guardado + 1, dtype=cp.float32)
            mesh_gruesa.divvector = cp.zeros(iteraciones // guardado + 1, dtype=cp.float32)
            mesh_gruesa.clcdvector = cp.zeros(iteraciones // guardado + 1, dtype=cp.float32)
            mesh_gruesa.mg_cycles_vector = cp.zeros(iteraciones, dtype=cp.int32)
        
        # Geometría cargada a 0°: SIEMPRE programar el primer alpha
        # (el flujo arranca horizontal, hay que rotarlo al primer alpha del plan)
        acum = 0
        for idx_plan, (alpha_plan, n_iter_plan) in enumerate(plan_polar):
            cambios_alpha_programados[acum] = alpha_plan
            acum += n_iter_plan
        
        print("\n" + "="*70)
        print("📅 PLAN POLAR AUTOMÁTICO:")
        print("="*70)
        iter_acum = 0
        for alpha_p, n_p in plan_polar:
            print(f"   α = {alpha_p:+6.2f}°  |  iter {iter_acum:>6d} → {iter_acum+n_p:>6d}  ({n_p} iters, descarte {polar_descarte*100:.0f}%)")
            iter_acum += n_p
        print(f"   Total: {iteraciones} iteraciones")
        print("="*70 + "\n")
    
    # Limpiar triggers antiguos si existen
    for trigger_file in [trigger_plot, trigger_stop, trigger_alpha]:
        if os.path.exists(trigger_file):
            try:
                os.remove(trigger_file)
            except:
                pass
    
    print("\n" + "="*70)
    print("🎮 CONTROLES INTERACTIVOS ACTIVOS:")
    print("="*70)
    print(f"  • Presiona Ctrl+C para finalizar limpiamente con outputs completos")
    print(f"  • Crea '{trigger_plot}' para generar gráficos sin detener")
    print(f"  • Crea '{trigger_stop}' para detener limpiamente")
    print(f"  • Crea '{trigger_alpha}' con el nuevo ángulo (ej: '8.0') para cambiar alpha")
    print("="*70 + "\n")
    
    # ============================================================
    # CRITERIO DE CONVERGENCIA A ESTADO ESTACIONARIO
    # ============================================================
    # Configuración
    check_convergence_every = 50  # Chequear cada N iteraciones
    min_iters_before_check = 200  # Mínimo de iteraciones antes de chequear
    tol_u = 1e-2  # Tolerancia relativa para velocidad u
    tol_v = 1e-1  # Tolerancia relativa para velocidad v
    tol_p = 1e-2  # Tolerancia relativa para presión
    
    # Variables para almacenar campos previos
    u_prev = None
    v_prev = None
    p_prev = None
    converged_to_steady = False
    
    # ============================================================
    # VISTA EN TIEMPO REAL (memoria compartida)
    # ============================================================
    shm_meta = None
    shm_data = None
    if live_view:
        try:
            ny_g, nx_g = mesh_gruesa.ny, mesh_gruesa.nx
            n_cells = ny_g * nx_g
            # Layout: speed(float32) + solid(uint8) + vorticity(float32)
            data_size = n_cells * 4 + n_cells + n_cells * 4
            meta_size = 40  # ny(4)+nx(4)+iter(4)+cd(4)+cl(4)+alpha(4)+timestamp(8)+v0x(4)+v0y(4)
            
            # Limpiar bloques previos si existen
            for name in ["sim2d_meta", "sim2d_live"]:
                try:
                    old = shared_memory.SharedMemory(name=name, create=False)
                    old.close()
                    old.unlink()
                except:
                    pass
            
            shm_meta = shared_memory.SharedMemory(name="sim2d_meta", create=True, size=meta_size)
            shm_data = shared_memory.SharedMemory(name="sim2d_live", create=True, size=data_size)
            
            # Escribir dimensiones iniciales
            struct.pack_into('ii', shm_meta.buf, 0, ny_g, nx_g)
            print(f"\n\U0001f4fa Live view activado: ejecuta 'python viewer_live.py' en otra terminal")
            print(f"   Memoria compartida: {(data_size + meta_size) / 1024:.1f} KB")
        except Exception as e:
            print(f"\n\u26a0 No se pudo activar live view: {e}")
            live_view = False
            shm_meta = None
            shm_data = None
    
    # Velocidad máxima permitida (red de seguridad contra blowup)
    U_ref = float(np.sqrt(v0x**2 + v0y**2))
    vel_clamp_max = 50.0 * max(U_ref, 1.0)

    # ── Diagnóstico de blowup (opcional; coste alto en mallas grandes) ──────
    if debug_spikes:
        # Umbral a partir del cual se activa el diagnóstico detallado (3× libre)
        _DIAG_THRESHOLD = 3.0 * max(U_ref, 1.0)
        _diag_triggered = False   # evitar inundación de prints

        def _diag_check(etapa):
            """Imprime info detallada si la velocidad supera _DIAG_THRESHOLD."""
            nonlocal _diag_triggered
            speed = cp.sqrt(mesh_gruesa.u**2 + mesh_gruesa.v**2)
            fluid_mask = ~mesh_gruesa.solid
            if not cp.any(fluid_mask):
                return
            speed_fluid = speed[fluid_mask]
            vel_max = float(cp.max(speed_fluid))
            if vel_max <= _DIAG_THRESHOLD:
                return

            # Localizar celda con velocidad máxima (en toda la malla, no solo fluido)
            idx_max = int(cp.argmax(speed))
            j_max = idx_max % mesh_gruesa.nx
            i_max = idx_max // mesh_gruesa.nx
            x_phys = float(mesh_gruesa.X_1d[j_max])
            y_phys = float(mesh_gruesa.Y_1d[i_max])
            es_solido   = bool(mesh_gruesa.solid[i_max, j_max])
            es_ghost    = (mesh_gruesa._ghost_cell_ready and
                           bool(mesh_gruesa._ghost_mask[i_max, j_max]))
            u_val = float(mesh_gruesa.u[i_max, j_max])
            v_val = float(mesh_gruesa.v[i_max, j_max])

            # Vecinas de la celda máxima
            ny_m, nx_m = mesh_gruesa.solid.shape
            vecinas = ""
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ii, jj = i_max+di, j_max+dj
                if 0 <= ii < ny_m and 0 <= jj < nx_m:
                    s = "S" if mesh_gruesa.solid[ii,jj] else "F"
                    g = "G" if (mesh_gruesa._ghost_cell_ready and mesh_gruesa._ghost_mask[ii,jj]) else ""
                    vecinas += f"({di:+d},{dj:+d}):{s}{g}={mesh_gruesa.u[ii,jj]:.2f} "

            tipo = "SÓLIDO/GHOST" if es_solido else "FLUIDO"
            if es_ghost:
                tipo = "GHOST"
            print(f"\n{'─'*65}")
            print(f"⚠️  SPIKE detectado tras [{etapa}]  vel_max={vel_max:.2f} m/s  (umbral={_DIAG_THRESHOLD:.1f})")
            print(f"   Celda ({i_max},{j_max})  físico=({x_phys:.4f},{y_phys:.4f})  tipo={tipo}")
            print(f"   u={u_val:.4f}  v={v_val:.4f}")
            print(f"   Vecinas: {vecinas}")
            if not _diag_triggered:
                # Primera vez: imprimir estadísticas globales de ghost cells
                if mesh_gruesa._ghost_cell_ready:
                    print(f"   Ghost cells totales: {mesh_gruesa._n_ghost}")
                    if hasattr(mesh_gruesa, '_ghost_direct_zero'):
                        n_dz = int(cp.sum(mesh_gruesa._ghost_direct_zero))
                        print(f"   Ghost 'direct-zero': {n_dz}")
                _diag_triggered = True
            print(f"{'─'*65}")
    else:
        # Sin coste extra por iteración cuando no se está depurando.
        def _diag_check(_etapa):
            return
    
    # ============================================================
    # BUCLE PRINCIPAL: MALLA SIMPLE (SOLO GRUESA)
    # ============================================================
    for it in tqdm(range(iteraciones)):
        t_paso_inicio = time.time()
        
        # Recalcular dt cada iteración (CFL adaptativo + restricción viscosa)
        t0 = time.time()
        try:
            # Calcular velocidad máxima SOLO en el fluido (excluir sólidos)
            speed_g = cp.sqrt(mesh_gruesa.u * mesh_gruesa.u + mesh_gruesa.v * mesh_gruesa.v)
            fluid_mask = ~mesh_gruesa.solid
            
            if cp.any(fluid_mask):
                Umax_cp = cp.max(speed_g[fluid_mask])
                Umax = float(Umax_cp) if float(Umax_cp) > 1e-12 else float(np.sqrt(v0x**2 + v0y**2))
            else:
                Umax = float(np.sqrt(v0x**2 + v0y**2))
            
            # dt advectivo (CFL)
            dt_adv = float(CFL * min(mesh_gruesa.dx, mesh_gruesa.dy) / max(Umax, 1e-12))
            
            # dt viscoso (estabilidad difusiva)
            if mesh_gruesa.usar_wale:
                nu_t_g = mesh_gruesa.compute_wale_viscosity()
                
                nu_t_max_permitido = 100.0 * nu
                nu_t_g = cp.minimum(nu_t_g, cp.float32(nu_t_max_permitido))
                
                nu_eff_max_cp = cp.max(cp.float32(nu) + nu_t_g)
                nu_eff_max = float(nu_eff_max_cp) if float(nu_eff_max_cp) > 0 else float(nu)
            else:
                nu_eff_max = float(nu)
            
            # ⚠️ Advertencia si nu_eff es anormalmente alto
            if nu_eff_max > 10.0 * nu and it % (guardado * 10) == 0:
                ratio_nu = nu_eff_max / nu
                print(f"\n⚠️  [Iter {it}] nu_efectiva muy alta: {nu_eff_max:.2e} ({ratio_nu:.1f}× nu molecular)")
                print(f"    Esto puede reducir dt significativamente")
            
            C_visc = 0.25
            if nu_eff_max > 1e-12:
                dt_visc = float(C_visc * (min(mesh_gruesa.dx, mesh_gruesa.dy)**2) / nu_eff_max)
            else:
                dt_visc = float('inf')
            
            # Usar el más restrictivo, pero limitar crecimiento al doble del dt nominal
            dt_use = min(dt_adv, dt_visc, dt * 2.0)
            
            # Seguridad: evitar dt muy pequeños que pueden estancar la simulación
            if dt_use < 1e-8:
                dt_use = dt
                
        except Exception as e:
            # En caso de error, usar dt nominal
            dt_use = dt
            
        timing_stats['recalculo_dt'] += time.time() - t0
        
        # Advección
        t0 = time.time()
        mesh_gruesa.advect_velocities(dt_use)
        mesh_gruesa.apply_boundaries(after_projection=False)
        _diag_check("ADVECCIÓN")
        timing_stats['adveccion'] += time.time() - t0
        
        # Difusión
        t0 = time.time()
        mesh_gruesa.diffuse_velocity(nu, dt_use, usar_wale=mesh_gruesa.usar_wale)
        mesh_gruesa.apply_boundaries(after_projection=False)
        _diag_check("DIFUSIÓN")
        timing_stats['difusion'] += time.time() - t0
        
        # Proyección
        t0 = time.time()
        mg_info = mesh_gruesa.project_multigrid(
            rho, dt_use,
            tol_div=divergencia,
            max_outer=mg_max_outer,
            cycles_per_outer=mg_cycles_per_outer,
            pre_suavizado=mg_pre_suavizado,
            post_suavizado=mg_post_suavizado,
            guard_residual_every_outer=mg_guard_residual_every_outer,
            adaptive_outer0_cycles=mg_adaptive_outer0_cycles,
            apply_ibm_each_outer=mg_apply_ibm_each_outer,
            rollback_on_nan=mg_rollback_on_nan,
            compute_div_after=mg_compute_div_after,
            verbose=False
        )
        #mg_info = mesh_gruesa.project_cg(rho,dt_use,tol_div=divergencia, verbose=False)
        
        
        # Almacenar ciclos usados
        mesh_gruesa.mg_cycles_vector[it] = mg_info['cycles']
        mesh_gruesa.apply_boundaries(after_projection=True)
        _diag_check("PROYECCIÓN")
        timing_stats['proyeccion'] += time.time() - t0
        
        # ============================================================
        # CLAMP DE VELOCIDADES Y PRESIÓN (red de seguridad contra blowup)
        # ============================================================
        mesh_gruesa.u = cp.clip(mesh_gruesa.u, -vel_clamp_max, vel_clamp_max)
        mesh_gruesa.v = cp.clip(mesh_gruesa.v, -vel_clamp_max, vel_clamp_max)
        # Presión: limitar a un rango razonable basado en presión dinámica
        p_clamp_max = cp.float32(100.0 * rho * U_ref * U_ref + 1000.0)
        mesh_gruesa.p = cp.clip(mesh_gruesa.p, -p_clamp_max, p_clamp_max)
        
        # ============================================================
        # DETECCIÓN DE INESTABILIDAD (NaN o blowup de velocidad)
        # ============================================================
        if it % 10 == 0:
            tiene_nan = (bool(cp.isnan(mesh_gruesa.u).any()) or
                         bool(cp.isnan(mesh_gruesa.v).any()) or
                         bool(cp.isnan(mesh_gruesa.p).any()))
            
            # Detectar blowup: velocidad máxima > umbral razonable
            vel_max_actual = float(cp.max(cp.abs(mesh_gruesa.u)).item())
            vel_max_v = float(cp.max(cp.abs(mesh_gruesa.v)).item())
            vel_max_actual = max(vel_max_actual, vel_max_v)
            blowup = vel_max_actual > vel_clamp_max * 0.9  # Cerca del clamp = inestable
            
            if tiene_nan or blowup:
                motivo = "NaN detectado" if tiene_nan else f"Blowup de velocidad ({vel_max_actual:.1f} m/s)"
                print(f"\n{'='*70}")
                print(f"❌ {motivo} en iteración {it} — simulación inestable")
                print(f"{'='*70}")
                print(f"   |u|_max={vel_max_actual:.2f}, umbral={vel_clamp_max:.2f}")
                print(f"   dt_use={dt_use:.2e}")
                if live_view and shm_data is not None:
                    try:
                        shm_data.close(); shm_data.unlink()
                        shm_meta.close(); shm_meta.unlink()
                    except Exception:
                        pass
                raise RuntimeError(f"Simulación abortada: {motivo} en iteración {it}")
        
        # ⭐ ACUMULAR TIEMPO FÍSICO después de completar el paso temporal
        tiempo_fisico_acumulado += dt_use
        
        # ============================================================
        # CHEQUEO DE TRIGGERS DE CONTROL INTERACTIVO
        # ============================================================
        # Chequear trigger de plot (cada cierto número de iteraciones para no saturar I/O)
        if it % 10 == 0:  # Verificar cada 10 iteraciones
            if os.path.exists(trigger_plot):
                print("\n" + "="*70)
                print(f"🎨 TRIGGER DETECTADO: Generando gráficos (iter {it})")
                print("="*70)
                try:
                    generar_graficos_y_outputs(
                        mesh_gruesa, iteraciones, guardado, it,
                        tiempo_fisico_acumulado, rho, U_inf, chord, nu, mu,
                        graficos=True, verbose=True
                    )
                    # Eliminar archivo trigger
                    os.remove(trigger_plot)
                    print(f"✓ Gráficos generados. Continuando simulación...\n")
                except Exception as e:
                    print(f"⚠ Error al generar gráficos: {e}")
                    try:
                        os.remove(trigger_plot)
                    except:
                        pass
            
            # Chequear trigger de stop
            if os.path.exists(trigger_stop):
                print("\n" + "="*70)
                print(f"🛑 TRIGGER DE STOP DETECTADO (iter {it})")
                print("="*70)
                print("Deteniendo simulación limpiamente...")
                interrupcion_solicitada = True
                try:
                    os.remove(trigger_stop)
                except:
                    pass
            
            # Chequear trigger de cambio de ángulo de ataque
            if os.path.exists(trigger_alpha):
                try:
                    with open(trigger_alpha, 'r') as f:
                        contenido = f.read().strip()
                    nuevo_alpha = float(contenido)
                    print("\n" + "="*70)
                    print(f"✈ CAMBIO DE ÁNGULO DE ATAQUE (iter {it})")
                    print(f"   α: {alpha_actual:.2f}° → {nuevo_alpha:.2f}°")
                    print("="*70)
                    
                    # Guardar resultado del alpha que termina (media temporal)
                    _guardar_punto_polar(polar_data, mesh_gruesa, mu, rho, U_inf, chord,
                                         alpha_actual, iter_inicio_alpha, it, guardado, polar_descarte)
                    try:
                        with open(archivo_polar, 'w') as fp:
                            json.dump(polar_data, fp, indent=2)
                        print(f"   Polar guardada: {len(polar_data)} puntos en {archivo_polar}")
                    except Exception:
                        pass
                    
                    v0x, v0y = mesh_gruesa.cambiar_angulo_ataque(nuevo_alpha, U_inf)
                    alpha_actual = nuevo_alpha
                    iter_inicio_alpha = it
                    
                    print(f"   v0x={v0x:.4f}, v0y={v0y:.4f} m/s")
                    print(f"   Continuando simulación...\n")
                    os.remove(trigger_alpha)
                except Exception as e:
                    print(f"⚠ Error al cambiar alpha: {e}")
                    try:
                        os.remove(trigger_alpha)
                    except:
                        pass
        
        # ============================================================
        # CAMBIO PROGRAMADO DE ALPHA (plan_polar)
        # ============================================================
        if it in cambios_alpha_programados:
            nuevo_alpha = cambios_alpha_programados[it]
            if abs(nuevo_alpha - alpha_actual) > 0.001 or it == 0:
                # Guardar punto polar del alpha que termina (salvo primera iteración)
                if it > 0:
                    _guardar_punto_polar(polar_data, mesh_gruesa, mu, rho, U_inf, chord,
                                         alpha_actual, iter_inicio_alpha, it, guardado, polar_descarte)
                    try:
                        with open(archivo_polar, 'w') as fp:
                            json.dump(polar_data, fp, indent=2)
                    except Exception:
                        pass
                
                print(f"\n✈ [PLAN POLAR] α: {alpha_actual:.2f}° → {nuevo_alpha:.2f}° (iter {it})")
                v0x, v0y = mesh_gruesa.cambiar_angulo_ataque(nuevo_alpha, U_inf)
                alpha_actual = nuevo_alpha
                iter_inicio_alpha = it
        
        # Si hay interrupción solicitada, salir del bucle
        if interrupcion_solicitada:
            print(f"\n⚠️  Simulación interrumpida en iteración {it}/{iteraciones}")
            print(f"   Tiempo físico simulado: {tiempo_fisico_acumulado:.4f} s")
            # Ajustar iteraciones efectivas para reportes
            iteraciones_efectivas = it
            break
        
        # Guardar resultados
        t0 = time.time()
        if it % guardado == 0:
            forces = mesh_gruesa.compute_drag_lift(mu, rho=rho, n_extrap_layers=5)
            cd_val = 2 * forces['Drag'] / (rho * U_inf**2 * chord)
            cl_val = 2 * forces['Lift'] / (rho * U_inf**2 * chord)
            
            # Chequear NaN en coeficientes aerodinámicos
            if np.isnan(cd_val) or np.isnan(cl_val):
                print(f"\n{'='*70}")
                print(f"❌ NaN DETECTADO en Cd/Cl en iteración {it} — simulación inestable")
                print(f"{'='*70}")
                print(f"   Cd={cd_val}, Cl={cl_val}")
                if live_view and shm_data is not None:
                    try:
                        shm_data.close(); shm_data.unlink()
                        shm_meta.close(); shm_meta.unlink()
                    except Exception:
                        pass
                raise RuntimeError(f"Simulación abortada: NaN en Cd/Cl en iteración {it}")
            
            mesh_gruesa.cdvector[it // guardado] = cd_val
            mesh_gruesa.clvector[it // guardado] = cl_val
            try:
                ratio = cl_val / cd_val if abs(cd_val) >= 1e-12 else np.nan
            except Exception:
                ratio = np.nan
            mesh_gruesa.clcdvector[it // guardado] = ratio
            mesh_gruesa.divvector[it // guardado] = mesh_gruesa.compute_divergence_mean()
            mesh_gruesa.update_cp_profile(mu, rho)
            
            # Publicar en memoria compartida (coste: ~1ms)
            if live_view and shm_data is not None:
                try:
                    speed_np = cp.asnumpy(cp.sqrt(mesh_gruesa.u**2 + mesh_gruesa.v**2))
                    solid_np = cp.asnumpy(mesh_gruesa.solid).astype(np.uint8)
                    # Vorticidad
                    dvdx = cp.zeros_like(mesh_gruesa.u)
                    dudy = cp.zeros_like(mesh_gruesa.u)
                    dvdx[:, 1:-1] = (mesh_gruesa.v[:, 2:] - mesh_gruesa.v[:, :-2]) / (2*mesh_gruesa.dx)
                    dudy[1:-1, :] = (mesh_gruesa.u[2:, :] - mesh_gruesa.u[:-2, :]) / (2*mesh_gruesa.dy)
                    vort_np = cp.asnumpy(dvdx - dudy).astype(np.float32)
                    
                    n_cells = mesh_gruesa.ny * mesh_gruesa.nx
                    off_s = 0
                    off_solid = n_cells * 4
                    off_vort = off_solid + n_cells
                    
                    shm_data.buf[off_s:off_s + n_cells*4] = speed_np.tobytes()
                    shm_data.buf[off_solid:off_solid + n_cells] = solid_np.tobytes()
                    shm_data.buf[off_vort:off_vort + n_cells*4] = vort_np.tobytes()
                    
                    # Metadata
                    struct.pack_into('ii', shm_meta.buf, 0, mesh_gruesa.ny, mesh_gruesa.nx)
                    struct.pack_into('i', shm_meta.buf, 8, it)
                    struct.pack_into('f', shm_meta.buf, 12, float(cd_val))
                    struct.pack_into('f', shm_meta.buf, 16, float(cl_val))
                    struct.pack_into('f', shm_meta.buf, 20, float(alpha_actual))
                    struct.pack_into('d', shm_meta.buf, 24, time.time())
                    struct.pack_into('f', shm_meta.buf, 32, float(v0x))
                    struct.pack_into('f', shm_meta.buf, 36, float(v0y))
                except Exception:
                    pass  # No interrumpir simulación por error de viewer
            
            if save_frames and frames_dir_grueso:
                mesh_gruesa.save_frame(frames_dir_grueso, it, kind="velocity")
                # Liberar memoria de figuras matplotlib cada cierto número de frames
                if it % (guardado * 10) == 0:
                    plt.close('all')
        
        if it % guardado == 0:
            timing_stats['guardado'] += time.time() - t0
        
        # ============================================================
        # CHEQUEO DE CONVERGENCIA A ESTADO ESTACIONARIO
        # ============================================================
        if it >= min_iters_before_check and it % check_convergence_every == 0:
            # Calcular cambios relativos (norma L2)
            if u_prev is not None:
                # Máscara de fluido (excluir sólidos del cálculo)
                fluid_mask = ~mesh_gruesa.solid
                
                # Cambio en u
                du = mesh_gruesa.u - u_prev
                norm_du = float(cp.sqrt(cp.mean(du[fluid_mask]**2)))
                norm_u = float(cp.sqrt(cp.mean(mesh_gruesa.u[fluid_mask]**2)))
                change_u = norm_du / (norm_u + 1e-12)  # Cambio relativo
                
                # Cambio en v
                dv = mesh_gruesa.v - v_prev
                norm_dv = float(cp.sqrt(cp.mean(dv[fluid_mask]**2)))
                norm_v = float(cp.sqrt(cp.mean(mesh_gruesa.v[fluid_mask]**2)))
                change_v = norm_dv / (norm_v + 1e-12)
                
                # Cambio en p
                dp = mesh_gruesa.p - p_prev
                norm_dp = float(cp.sqrt(cp.mean(dp[fluid_mask]**2)))
                norm_p = float(cp.sqrt(cp.mean(mesh_gruesa.p[fluid_mask]**2)))
                change_p = norm_dp / (norm_p + 1e-12)
                
                # Verificar convergencia
                if change_u < tol_u and change_v < tol_v and change_p < tol_p:
                    # Solo imprimir el mensaje la primera vez que se detecta convergencia
                    if not converged_to_steady:
                        print(f"\\n{'='*70}")
                        print(f"✓ ESTADO ESTACIONARIO ALCANZADO en iteración {it}")
                        print(f"{'='*70}")
                        print(f"  Cambio relativo u: {change_u:.2e} < {tol_u:.2e}")
                        print(f"  Cambio relativo v: {change_v:.2e} < {tol_v:.2e}")
                        print(f"  Cambio relativo p: {change_p:.2e} < {tol_p:.2e}")
                        print(f"  Tiempo simulado: {tiempo_fisico_acumulado:.4f} s")
                        if not stop_on_convergence:
                            print(f"  ⚠ Continuando hasta completar iteraciones (stop_on_convergence=False)")
                        print(f"{'='*70}\\n")
                        converged_to_steady = True
                        
                        # Ajustar vectores para que tengan el tamaño correcto
                        # Rellenar el resto con el último valor válido
                        if stop_on_convergence and it // guardado < len(mesh_gruesa.cdvector) - 1:
                            last_idx = it // guardado
                            mesh_gruesa.cdvector[last_idx+1:] = mesh_gruesa.cdvector[last_idx]
                            mesh_gruesa.clvector[last_idx+1:] = mesh_gruesa.clvector[last_idx]
                            mesh_gruesa.divvector[last_idx+1:] = mesh_gruesa.divvector[last_idx]
                            mesh_gruesa.clcdvector[last_idx+1:] = mesh_gruesa.clcdvector[last_idx]
                    
                    # Salir del bucle solo si stop_on_convergence está activado
                    if stop_on_convergence:
                        break
                
                # Imprimir progreso ocasionalmente
                elif it % (check_convergence_every * 10) == 0:
                    print(f"\\n[Iter {it}] Convergencia a steady: u={change_u:.2e}, v={change_v:.2e}, p={change_p:.2e}")
            
            # Almacenar campos actuales como referencia para próximo chequeo
            u_prev = mesh_gruesa.u.copy()
            v_prev = mesh_gruesa.v.copy()
            p_prev = mesh_gruesa.p.copy()
        
        t_paso_total = time.time() - t_paso_inicio
        timing_stats['total_por_paso'].append(t_paso_total)
    
    # ============================================================
    # FIN DEL BUCLE PRINCIPAL
    # ============================================================
    # Si se interrumpió, ajustar datos para reflejar solo lo simulado
    if interrupcion_solicitada:
        # Truncar vectores de resultados al tamaño real
        idx_final = min((it // guardado) + 1, len(mesh_gruesa.cdvector))
        mesh_gruesa.cdvector = mesh_gruesa.cdvector[:idx_final]
        mesh_gruesa.clvector = mesh_gruesa.clvector[:idx_final]
        mesh_gruesa.divvector = mesh_gruesa.divvector[:idx_final]
        mesh_gruesa.clcdvector = mesh_gruesa.clcdvector[:idx_final]
        mesh_gruesa.mg_cycles_vector = mesh_gruesa.mg_cycles_vector[:it+1]
        iteraciones = it  # Actualizar para reportes
    

    # ============================================================
    # REPORTE DE CONVERGENCIA A ESTADO ESTACIONARIO
    # ============================================================
    if interrupcion_solicitada:
        print("\n" + "="*70)
        print("⚠️  SIMULACIÓN INTERRUMPIDA POR USUARIO")
        print("="*70)
        print(f"  Se ejecutaron {it} de {iteraciones} iteraciones programadas.")
        print(f"  Tiempo físico simulado: {tiempo_fisico_acumulado:.4f} s")
        print(f"  Generando outputs con datos disponibles...")
        print("="*70)
    elif converged_to_steady:
        print("\n" + "="*70)
        print("✓ SIMULACIÓN CONVERGIÓ A ESTADO ESTACIONARIO")
        print("="*70)
        print(f"  La simulación alcanzó convergencia antes de completar")
        print(f"  todas las iteraciones programadas.")
        print(f"  Los resultados representan un estado estacionario válido.")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("✓ SIMULACIÓN COMPLETADA")
        print("="*70)
        print(f"  Se ejecutaron todas las {iteraciones} iteraciones programadas.")
        if not stop_on_convergence:
            print(f"  (Nota: chequeo de convergencia steady desactivado)")
        else:
            print(f"  Si buscas estado estacionario, considera aumentar iteraciones")
            print(f"  o relajar tolerancias (tol_u, tol_v, tol_p).")
        print("="*70)
    
    # ============================================================
    # REPORTE DE TIMING
    # ============================================================
    print("\n" + "="*70)
    print("ESTADÍSTICAS DE RENDIMIENTO")
    print("="*70)
    
    num_pasos = len(timing_stats['total_por_paso'])
    if num_pasos > 0:
        tiempo_medio_paso = sum(timing_stats['total_por_paso']) / num_pasos
        print(f"\nTiempo promedio por paso dt: {tiempo_medio_paso:.4f} s")
        print(f"Pasos simulados: {num_pasos}")
        print(f"Tiempo total simulación: {sum(timing_stats['total_por_paso']):.2f} s\n")
        
        print("Desglose por componente (tiempo total | % del total):")
        print("-" * 70)
        componentes = [
            ('Recálculo dt', 'recalculo_dt'),
            ('Advección', 'adveccion'),
            ('Difusión', 'difusion'),
            ('Proyección', 'proyeccion'),
            ('Guardado/fuerzas', 'guardado')
        ]
        
        tiempo_total = sum(timing_stats['total_por_paso'])
        for nombre, clave in componentes:
            t = timing_stats[clave]
            pct = 100 * t / tiempo_total if tiempo_total > 0 else 0
            print(f"  {nombre:.<30} {t:>8.2f} s  ({pct:>5.1f}%)")
        
        # Calcular tiempo no contabilizado
        tiempo_contabilizado = sum(timing_stats[k] for _, k in componentes)
        timing_stats['otros'] = tiempo_total - tiempo_contabilizado
        if timing_stats['otros'] > 0.01:
            pct_otros = 100 * timing_stats['otros'] / tiempo_total
            print(f"  {'Otros (overhead)':.<30} {timing_stats['otros']:>8.2f} s  ({pct_otros:>5.1f}%)")
        
        print("-" * 70)
        
        # Identificar cuellos de botella
        print("\n🔍 ANÁLISIS:")
        max_componente = max(componentes, key=lambda x: timing_stats[x[1]])
        max_nombre, max_clave = max_componente
        max_tiempo = timing_stats[max_clave]
        max_pct = 100 * max_tiempo / tiempo_total
        
        print(f"   Componente más costoso: {max_nombre}")
        print(f"   Consume: {max_tiempo:.2f} s ({max_pct:.1f}% del tiempo total)")
        print(f"   Tiempo promedio por paso: {max_tiempo/num_pasos:.4f} s")

        # Estadística de ciclos MG realmente usados (útil para tuning)
        try:
            cycles_np = cp.asnumpy(mesh_gruesa.mg_cycles_vector[:num_pasos]).astype(np.float32)
            cycles_np = cycles_np[cycles_np > 0]
            if cycles_np.size > 0:
                c_mean = float(np.mean(cycles_np))
                c_p95  = float(np.percentile(cycles_np, 95))
                c_max  = float(np.max(cycles_np))
                print(f"   Ciclos MG por paso: media={c_mean:.2f}  p95={c_p95:.2f}  max={c_max:.0f}")
        except Exception:
            pass
    
    print("="*70 + "\n")
    
    # Guardar punto polar del último alpha simulado (media temporal)
    _guardar_punto_polar(polar_data, mesh_gruesa, mu, rho, U_inf, chord,
                         alpha_actual, iter_inicio_alpha, iteraciones, guardado, polar_descarte)
    try:
        with open(archivo_polar, 'w') as fp:
            json.dump(polar_data, fp, indent=2)
        print(f"\n✅ Polar completa guardada: {len(polar_data)} puntos en {archivo_polar}")
        # Resumen tabla
        print(f"\n{'alpha':>8s} {'Cd':>10s} {'Cl':>10s} {'Cl/Cd':>10s}")
        print("-"*42)
        for p in polar_data:
            clcd = p['Cl']/p['Cd'] if abs(p['Cd']) > 1e-12 else float('nan')
            print(f"{p['alpha']:>8.2f} {p['Cd']:>10.6f} {p['Cl']:>10.6f} {clcd:>10.3f}")
    except Exception as e:
        print(f"\n⚠ Error guardando polar: {e}")

    # Generar gráficos polares si hay plan polar con suficientes puntos
    if plan_polar is not None and len(polar_data) >= 2:
        try:
            _generar_graficos_polar(polar_data, filepath)
        except Exception as e:
            print(f"⚠ Error generando gráficos polares: {e}")

    # ============================================================
    # GENERACIÓN DE GRÁFICOS Y REPORTES FINALES
    # ============================================================
    generar_graficos_y_outputs(
        mesh_gruesa, iteraciones, guardado, iteraciones,
        tiempo_fisico_acumulado, rho, U_inf, chord, nu, mu,
        graficos=graficos, verbose=True
    )
    
    # Liberar memoria compartida del live view
    if shm_data is not None:
        try:
            shm_data.close()
            shm_data.unlink()
        except:
            pass
    if shm_meta is not None:
        try:
            shm_meta.close()
            shm_meta.unlink()
        except:
            pass

    return mesh_gruesa

if __name__ == "__main__":
    mesh = main(    
        Lx=12,
        Ly=8,  
        cx=2,
        CFL=0.5,
        alpha_deg=0,
        polar_descarte=0.3,
        iteraciones=10000,
        divergencia=1e-1,
        v0x=1,
        v0y=0,
        rho=1.0,
        nu=1/100000,
        filepath="NACA_0012",
        chord=1.0,
        dx_min=0.001,
        ancho_zona_fina_x=1.2,
        ancho_zona_fina_y=1,
        factor_expansion=1.1,
        graficos=True,
        save_frames=False, 
        frames_dir_grueso="",
        usar_wale=False,
        stop_on_convergence=False,
        live_view=True,
        mostrar_malla=True,
        mg_modo_rapido=True,
    )
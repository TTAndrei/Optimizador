import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cupy as cp
from cupyx.scipy.ndimage import distance_transform_edt
import os
import json
from tqdm import tqdm
import mesh_multires as mmr
import signal
import sys
from datetime import datetime
from multiprocessing import shared_memory
import struct

class Mesh:
    '''Constructor y basicos'''
    def __init__(self, Lx,Ly, p0, v0x, v0y, dx, dy, usar_wale=True, usar_viscosidad_estela=False, C_estela=0.05):

        self.Lx=Lx
        self.Ly=Ly

        self.dx = dx
        self.dy = dy
        self.usar_wale = usar_wale  # Flag para activar/desactivar modelo WALE
        self.usar_viscosidad_estela = usar_viscosidad_estela  # Viscosidad extra en estela (surrogate 3D)
        self.C_estela = C_estela  # Coeficiente de viscosidad de estela
        self._wake_mask = None  # Se precomputa al llamar a compute_wake_viscosity


        self.nx = int(self.Lx/self.dx)
        self.ny = int(self.Ly/self.dy)

        # --- Campos principales en GPU ---
        self.u = cp.zeros((self.ny, self.nx), dtype=cp.float32)
        self.v = cp.zeros((self.ny, self.nx), dtype=cp.float32)
        self.p = cp.full((self.ny, self.nx), p0, dtype=cp.float32)

        self.dvector = cp.zeros(1, dtype=cp.float32)
        self.lvector = cp.zeros(1, dtype=cp.float32)
        self.cdvector = cp.zeros(1, dtype=cp.float32)
        self.clvector = cp.zeros(1, dtype=cp.float32)
        self.divvector = cp.zeros(1, dtype=cp.float32)
        self.mg_cycles_vector = cp.zeros(1, dtype=cp.int32)  # Ciclos multigrid por paso
        self.guardado = 1  # Frecuencia de guardado (se actualiza en main)

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

        self._jacobi_update = cp.ElementwiseKernel(
            in_params='float32 p_ref, raw float32 p, raw float32 rhs, float32 h2, int32 nx, int32 ny',
            out_params='float32 out',
            operation=r'''
            int idx = i;
            int j = idx % nx;
            int ii = idx / nx;
            // bordes: copiar valor actual
            if (ii<=0 || ii>=ny-1 || j<=0 || j>=nx-1) {
                out = p[idx];
            } else {
                int E = ii*nx + (j+1);
                int W = ii*nx + (j-1);
                int N = (ii+1)*nx + j;
                int S = (ii-1)*nx + j;
                // Jacobi: 0.25*(sum vecinos - h^2 * rhs)
                out = 0.25f * (p[E] + p[W] + p[N] + p[S] - h2 * rhs[idx]);
            }
            ''',
            name='jacobi_update'
        )

        # Kernel CG: Aplicar Laplaciano discreto (A*x)
        # A*x = (x_E + x_W + x_N + x_S - 4*x_C) / h²
        self._laplacian_kernel = cp.ElementwiseKernel(
            in_params='raw float32 x, float32 h2, int32 nx, int32 ny',
            out_params='float32 Ax',
            operation=r'''
            int idx = i;
            int j = idx % nx;
            int ii = idx / nx;
            // Bordes: Neumann homogéneo (Ax = 0 en bordes)
            if (ii <= 0 || ii >= ny-1 || j <= 0 || j >= nx-1) {
                Ax = 0.0f;
            } else {
                int E = ii*nx + (j+1);
                int W = ii*nx + (j-1);
                int N = (ii+1)*nx + j;
                int S = (ii-1)*nx + j;
                Ax = (x[E] + x[W] + x[N] + x[S] - 4.0f * x[idx]) / h2;
            }
            ''',
            name='laplacian_cg'
        )
        
        # Kernel CG: Precondicionador Jacobi (M^-1 * r)
        # Para Laplaciano: diag(A) = -4/h^2, entonces M^-1 = -h^2/4
        self._precond_jacobi_kernel = cp.ElementwiseKernel(
            in_params='raw float32 r, float32 h2, int32 nx, int32 ny',
            out_params='float32 z',
            operation=r'''
            int idx = i;
            int j = idx % nx;
            int ii = idx / nx;
            // Bordes: no precondicionar
            if (ii <= 0 || ii >= ny-1 || j <= 0 || j >= nx-1) {
                z = 0.0f;
            } else {
                // M^-1 = -h^2/4 (inverso de la diagonal del Laplaciano)
                z = -0.25f * h2 * r[idx];
            }
            ''',
            name='precond_jacobi'
        )

        # ============================================================
        # Kernels ENMASCARADOS (sólidos) para Poisson/proyección
        # - Trata fronteras sólido-fluido con Neumann homogéneo (flujo normal cero)
        # - Evita que el operador use vecinos sólidos (clave para que la proyección
        #   realmente reduzca div(u) cerca del perfil)
        # ============================================================
        self._jacobi_update_masked = cp.ElementwiseKernel(
            in_params='raw bool solid, raw float32 p, raw float32 rhs, float32 h2, int32 nx, int32 ny',
            out_params='float32 out',
            operation=r'''
            int idx = i;
            int j = idx % nx;
            int ii = idx / nx;
            // Bordes o sólido: no actualizar
            if (ii<=0 || ii>=ny-1 || j<=0 || j>=nx-1 || solid[idx]) {
                out = p[idx];
            } else {
                int E = ii*nx + (j+1);
                int W = ii*nx + (j-1);
                int N = (ii+1)*nx + j;
                int S = (ii-1)*nx + j;
                float sum_nb = 0.0f;
                int n_nb = 0;
                if (!solid[E]) { sum_nb += p[E]; n_nb++; }
                if (!solid[W]) { sum_nb += p[W]; n_nb++; }
                if (!solid[N]) { sum_nb += p[N]; n_nb++; }
                if (!solid[S]) { sum_nb += p[S]; n_nb++; }
                if (n_nb > 0) {
                    // Jacobi con vecindario variable (Neumann en sólido)
                    out = (sum_nb - h2 * rhs[idx]) / (float)n_nb;
                } else {
                    // Celda aislada (raro): mantener valor
                    out = p[idx];
                }
            }
            ''',
            name='jacobi_update_masked'
        )

        self._laplacian_kernel_masked = cp.ElementwiseKernel(
            in_params='raw bool solid, raw float32 x, float32 h2, int32 nx, int32 ny',
            out_params='float32 Ax',
            operation=r'''
            int idx = i;
            int j = idx % nx;
            int ii = idx / nx;
            // Bordes o sólido: Ax=0 (no se resuelve en sólido)
            if (ii<=0 || ii>=ny-1 || j<=0 || j>=nx-1 || solid[idx]) {
                Ax = 0.0f;
            } else {
                int E = ii*nx + (j+1);
                int W = ii*nx + (j-1);
                int N = (ii+1)*nx + j;
                int S = (ii-1)*nx + j;
                float xc = x[idx];
                float sum_nb = 0.0f;
                int n_nb = 0;
                if (!solid[E]) { sum_nb += x[E]; n_nb++; }
                if (!solid[W]) { sum_nb += x[W]; n_nb++; }
                if (!solid[N]) { sum_nb += x[N]; n_nb++; }
                if (!solid[S]) { sum_nb += x[S]; n_nb++; }
                // Laplaciano con Neumann en fronteras sólido: (sum_nb - n_nb*xc)/h^2
                Ax = (sum_nb - (float)n_nb * xc) / h2;
            }
            ''',
            name='laplacian_masked'
        )

        self._precond_jacobi_kernel_masked = cp.ElementwiseKernel(
            in_params='raw bool solid, raw float32 r, float32 h2, int32 nx, int32 ny',
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
                int n_nb = 0;
                if (!solid[E]) n_nb++;
                if (!solid[W]) n_nb++;
                if (!solid[N]) n_nb++;
                if (!solid[S]) n_nb++;
                if (n_nb > 0) {
                    // diag(A) = -n_nb/h^2  =>  M^-1 = -h^2/n_nb
                    z = -(h2 / (float)n_nb) * r[idx];
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
            float h2, float omega,
            int nx, int ny, int phase
        ) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= nx * ny) return;
            int j = idx % nx;
            int ii = idx / nx;
            // Solo actualizar celdas del color correcto (checkerboard)
            if (((ii + j) & 1) != phase) return;
            if (ii <= 0 || ii >= ny-1 || j <= 0 || j >= nx-1 || solid[idx]) return;

            int E = ii*nx + (j+1);
            int W = ii*nx + (j-1);
            int N = (ii+1)*nx + j;
            int S = (ii-1)*nx + j;
            float sum_nb = 0.0f;
            int n_nb = 0;
            if (!solid[E]) { sum_nb += p[E]; n_nb++; }
            if (!solid[W]) { sum_nb += p[W]; n_nb++; }
            if (!solid[N]) { sum_nb += p[N]; n_nb++; }
            if (!solid[S]) { sum_nb += p[S]; n_nb++; }
            if (n_nb > 0) {
                float p_gs = (sum_nb - h2 * rhs[idx]) / (float)n_nb;
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
            // Pesos: 0.0 para par, 0.5 para impar
            float wx = (jf & 1) ? 0.5f : 0.0f;
            float wy = (if_ & 1) ? 0.5f : 0.0f;
            // Si es impar y no hay vecino superior, skip (coincide con original)
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
            float inv_2dx, float inv_2dy, int nx, int ny
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
            float du_dx = (!solid[E] && !solid[W]) ? (u[E] - u[W]) * inv_2dx : 0.0f;
            float dv_dy = (!solid[N] && !solid[S]) ? (v[N] - v[S]) * inv_2dy : 0.0f;
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
            float coef, float inv_2dx, float inv_2dy,
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
            if (!solid[E] && !solid[W])
                u[idx] -= coef * (p[E] - p[W]) * inv_2dx;
            if (!solid[N] && !solid[S])
                v[idx] -= coef * (p[N] - p[S]) * inv_2dy;
        }
        ''', 'velocity_correction')

        # Flag de jerarquía multigrid (se inicializa en _init_mg_hierarchy)
        self._mg_initialized = False

    def _init_mg_hierarchy(self, niveles_max=4):
        """Pre-computa jerarquía multigrid: máscaras, h2, buffers por nivel."""
        ny, nx = self.p.shape
        self._mg_niveles = max(1, min(niveles_max, int(np.log2(min(ny, nx))) - 2))

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

        # h2 pre-computado por nivel
        h2_base = cp.float32(self.dx * self.dx)
        self._mg_h2 = [cp.float32(h2_base * (4.0 ** lvl)) for lvl in range(self._mg_niveles + 1)]

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
        
        # Verificar que los puntos imagen están en fluido, si no, ajustar
        # (interpolación bilineal podría necesitar celdas vecinas en fluido)
        
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
        
        # Sólido interior (lejos de la interfaz): mantener a cero
        self.u[self._solid_interior] = 0.0
        self.v[self._solid_interior] = 0.0

    def info(self):
        print(f"Mesh: {self.nx} x {self.ny}")
        print(f"Nodos: {self.nx * self.ny}")
        print(f"dx={self.dx}, dy={self.dy}")
        print("Campos en GPU: u, v, p")
        print("u shape:", self.u.shape)
        print("v shape:", self.v.shape)
        print("p shape:", self.p.shape)

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

        # Invalidar máscara de estela (se recalculará en el siguiente paso)
        self._wake_mask = None

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
                # Presión fija en la frontera: p = p0 (pasar p0 como 'value' en set_boundary)
                if value is not None:
                    self.p[u_slice] = value
                
                # ⚠️ IMPORTANTE: NO sobrescribir velocidades después de proyección
                # La proyección ya aplicó grad(p) correctamente en los bordes
                if not after_projection:
                    # Solo aplicar Neumann antes de proyección
                    # grad(u) = grad(v) = 0 → copiar interior correcto
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
        """
        # usar JJ/II precomputadas
        JJ = self.JJ
        II = self.II

        # convertir velocidades a índices/tiempo
        # Ecuación característica (backtrace): x_prev = x - u * dt
        # En unidades de índice: u_idx = u / dx, v_idx = v / dy
        u_idx = (self.u / self.dx).astype(cp.float32)
        v_idx = (self.v / self.dy).astype(cp.float32)

        # backtrace en índices (elementwise)
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
            donde ν_eff = ν + ν_t(x,y) + ν_wake(x,y)
        """
        nu_f = float(nu)
        if nu_f <= 0.0 and not usar_wale and not self.usar_viscosidad_estela:
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
        
        # ⭐ Añadir viscosidad de estela (surrogate 3D)
        if self.usar_viscosidad_estela:
            nu_wake = self.compute_wake_viscosity()
            nu_eff = nu_eff + nu_wake
        
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
        inv_dx2 = cp.float32(1.0 / (dx * dx))
        inv_dy2 = cp.float32(1.0 / (dy * dy))
        inv_2dx = cp.float32(0.5 / dx)
        inv_2dy = cp.float32(0.5 / dy)

        # Prealocar temporales
        u_new = self.u.astype(cp.float32, copy=True)
        v_new = self.v.astype(cp.float32, copy=True)

        for _ in range(n_sub):
            u = u_new
            v = v_new
            
            # ⭐ Con viscosidad variable: ∇·(ν_eff ∇u) ≠ ν_eff ∇²u
            # Forma correcta: ∂(ν ∂u/∂x)/∂x + ∂(ν ∂u/∂y)/∂y
            viscosidad_variable = usar_wale or self.usar_viscosidad_estela
            if viscosidad_variable:
                # Calcular gradientes de u,v
                du_dx = cp.zeros_like(u, dtype=cp.float32)
                du_dy = cp.zeros_like(u, dtype=cp.float32)
                dv_dx = cp.zeros_like(v, dtype=cp.float32)
                dv_dy = cp.zeros_like(v, dtype=cp.float32)
                
                du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) * inv_2dx
                du_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) * inv_2dy
                dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) * inv_2dx
                dv_dy[1:-1, :] = (v[2:, :] - v[:-2, :]) * inv_2dy
                
                # Calcular gradientes de nu_eff
                dnu_dx = cp.zeros_like(nu_eff, dtype=cp.float32)
                dnu_dy = cp.zeros_like(nu_eff, dtype=cp.float32)
                dnu_dx[:, 1:-1] = (nu_eff[:, 2:] - nu_eff[:, :-2]) * inv_2dx
                dnu_dy[1:-1, :] = (nu_eff[2:, :] - nu_eff[:-2, :]) * inv_2dy
                
                # Laplaciano de u,v (parte nu·∇²u)
                lap_u = cp.zeros_like(u, dtype=cp.float32)
                lap_v = cp.zeros_like(v, dtype=cp.float32)
                lap_u[1:-1, 1:-1] = (
                    (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) * inv_dx2
                    + (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) * inv_dy2
                )
                lap_v[1:-1, 1:-1] = (
                    (v[1:-1, 2:] - 2.0 * v[1:-1, 1:-1] + v[1:-1, :-2]) * inv_dx2
                    + (v[2:, 1:-1] - 2.0 * v[1:-1, 1:-1] + v[:-2, 1:-1]) * inv_dy2
                )
                
                # Término completo: ∇·(ν_eff ∇u) = ν_eff ∇²u + ∇ν_eff · ∇u
                div_visc_u = nu_eff * lap_u + dnu_dx * du_dx + dnu_dy * du_dy
                div_visc_v = nu_eff * lap_v + dnu_dx * dv_dx + dnu_dy * dv_dy
                
                u_new = u + dt_sub_cp * div_visc_u
                v_new = v + dt_sub_cp * div_visc_v
            else:
                # Viscosidad constante: caso simple
                lap_u = cp.zeros_like(u, dtype=cp.float32)
                lap_v = cp.zeros_like(v, dtype=cp.float32)
                lap_u[1:-1, 1:-1] = (
                    (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) * inv_dx2
                    + (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) * inv_dy2
                )
                lap_v[1:-1, 1:-1] = (
                    (v[1:-1, 2:] - 2.0 * v[1:-1, 1:-1] + v[1:-1, :-2]) * inv_dx2
                    + (v[2:, 1:-1] - 2.0 * v[1:-1, 1:-1] + v[:-2, 1:-1]) * inv_dy2
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

    def project2_adaptive(self, rho_sim, dt, tol_div=1e-5, tol_poisson=1e-5, 
                          max_iter=500, min_iter=5, check_every=20, omega=0.8,
                          print_every=100, verbose=False):
   
        dx = self.dx
        dy = self.dy
        assert abs(dx - dy) < 1e-6, "Esta version asume dx~dy."
        
        h2 = cp.float32(dx * dx)
        rho_f = cp.float32(rho_sim)
        dt_f = cp.float32(dt)
        ny, nx = self.p.shape
        
        # Mascara de celdas libres
        free = ~self.solid
        
        # ⭐ Aplicar Ghost-Cell IBM antes de calcular divergencia
        self.apply_ghost_cell_bc()
        
        # ============================================================
        # PRE-CHECK: Medir divergencia ANTES de proyectar
        # Estrategia adaptativa SIMPLIFICADA (para usar como pre-suavizador):
        # - Si div_before < tol_div: ejecutar max_iter/2 iteraciones (mantenimiento)
        # - Si div_before >= tol_div: ejecutar max_iter iteraciones completas
        # ============================================================
        div_before = self._compute_divergence_field()
        div_mean_before = float(cp.mean(cp.abs(div_before[free])))
        
        if verbose:
            print(f"[Proyección Jacobi] Divergencia inicial: {div_mean_before:.6e}")
        
        # Ajustar iteraciones según estado actual
        if div_mean_before < tol_div:
            # Divergencia aceptable: modo mantenimiento (mitad de iteraciones)
            adaptive_max_iter = max(min_iter, max_iter // 2)
            if verbose:
                print(f"[Proyección Jacobi] Modo mantenimiento: {adaptive_max_iter} iters")
        else:
            # Divergencia alta: usar max_iter completo
            adaptive_max_iter = max_iter
            if verbose:
                print(f"[Proyección Jacobi] Modo completo: {adaptive_max_iter} iters")
        
        # ============================================================
        # Setup Poisson: del^2 p = (rho/dt) * div(u*)
        # ============================================================
        rhs = cp.zeros((ny, nx), dtype=cp.float32)
        rhs[1:-1, 1:-1] = (rho_f / dt_f) * div_before[1:-1, 1:-1]

        # Compatibilidad Neumann (si no hay Dirichlet de presión): sum(rhs)=0 en fluido
        try:
            hay_dirichlet = bool(cp.any(self.fixed_pressure_mask))
        except Exception:
            hay_dirichlet = False
        if not hay_dirichlet:
            mean_rhs = cp.mean(rhs[free])
            rhs[free] = rhs[free] - mean_rhs
        
        # Warm start
        p = self.p.astype(cp.float32, copy=True)
        p_flat = p.ravel()
        rhs_flat = rhs.ravel().astype(cp.float32, copy=False)
        
        # ============================================================
        # Iteraciones Jacobi con chequeo de divergencia
        # ============================================================
        converged = False
        div_mean_after = div_mean_before  # Inicializar para el retorno
        
        for it in range(adaptive_max_iter):
            # Update Jacobi
            out = self._jacobi_update_masked(
                self.solid.ravel(), p_flat, rhs_flat, h2,
                cp.int32(nx), cp.int32(ny), size=p_flat.size
            )

            # Enforce fixed-pressure Dirichlet puntos (si hay máscara)
            try:
                if cp.any(self.fixed_pressure_mask):
                    mask_flat = self.fixed_pressure_mask.ravel()
                    out[mask_flat] = self.fixed_pressure_value
            except Exception:
                pass

            if omega != 1.0:
                p_flat = (1.0 - omega) * p_flat + omega * out
            else:
                p_flat = out

            # Reimponer valor fijo tras mezcla (por seguridad)
            try:
                if cp.any(self.fixed_pressure_mask):
                    p_flat[self.fixed_pressure_mask.ravel()] = self.fixed_pressure_value
            except Exception:
                pass
            
            # Chequear convergencia cada check_every (pero minimo min_iter)
            if it >= min_iter and it % check_every == 0:
                # Actualizar presion temporal
                self.p = p_flat.reshape(ny, nx)
                self._aplicar_bc_presion_neumann()
                
                # Corregir velocidad temporalmente para medir div(u)
                u_temp = self.u.copy()
                v_temp = self.v.copy()
                
                coef = dt_f / rho_f
                # grad(p) enmascarado cerca de sólidos (evita stencils que crucen sólido)
                free_c = free[1:-1, 1:-1]
                mask_x = free_c & free[1:-1, 2:] & free[1:-1, :-2]
                mask_y = free_c & free[2:, 1:-1] & free[:-2, 1:-1]
                dpdx = (self.p[1:-1, 2:] - self.p[1:-1, :-2]) / (2.0 * dx)
                dpdy = (self.p[2:, 1:-1] - self.p[:-2, 1:-1]) / (2.0 * dy)
                u_temp[1:-1, 1:-1] -= coef * cp.where(mask_x, dpdx, 0.0)
                v_temp[1:-1, 1:-1] -= coef * cp.where(mask_y, dpdy, 0.0)
                
                # Calcular divergencia del campo corregido (consistente con sólidos)
                div_after = self._compute_divergence_field_uv(u_temp, v_temp)
                div_mean_after = float(cp.mean(cp.abs(div_after[free])))
                
                # CRITERIO PRINCIPAL: si div(u) es suficientemente pequeño, SALIR
                if div_mean_after < tol_div:
                    converged = True
                    if verbose:
                        print(f"[Proyección] ✓ Convergencia alcanzada en iter {it+1}: div={div_mean_after:.6e}")
                    break
                
                # Criterio secundario: residuo Poisson (opcional)
                # Residuo Poisson con operador enmascarado (si hay sólido, evita stencil)
                lap_flat = self._laplacian_kernel_masked(
                    self.solid.ravel(), p_flat, h2, cp.int32(nx), cp.int32(ny), size=p_flat.size
                )
                lap = lap_flat.reshape(ny, nx)[1:-1, 1:-1]
                res_field = lap - rhs[1:-1, 1:-1]
                res = cp.max(cp.abs(res_field[free[1:-1, 1:-1]]))
                
                if res < tol_poisson:
                    converged = True
                    if verbose:
                        print(f"[Proyección] ✓ Convergencia Poisson en iter {it+1}: res={float(res):.6e}")
                    break
            
            # Imprimir progreso cada print_every iteraciones
            if verbose and (it + 1) % print_every == 0:
                # Si acabamos de calcular div, mostrarla; si no, indicar que se calcula cada check_every
                if it >= min_iter and (it % check_every == 0):
                    print(f"  Iter {it+1:6d}: div={div_mean_after:.6e}")
                else:
                    print(f"  Iter {it+1:6d}: (div se calcula cada {check_every} iters)")
        
        # Advertencia si no convergió
        if not converged and verbose:
            print(f"[Proyección] ⚠ ADVERTENCIA: Alcanzado max_iter={adaptive_max_iter} sin convergencia completa")
            print(f"              div_final={div_mean_after:.6e} (objetivo: {tol_div:.6e})")
        
        # ============================================================
        # Aplicar correccion final
        # ============================================================
        self.p = p_flat.reshape(ny, nx)
        self._aplicar_bc_presion_neumann()
        # Reimponer presion fija en celdas marcadas
        try:
            if cp.any(self.fixed_pressure_mask):
                self.p[self.fixed_pressure_mask] = self.fixed_pressure_value
        except Exception:
            pass
        
        coef = dt_f / rho_f
        free_c = free[1:-1, 1:-1]
        mask_x = free_c & free[1:-1, 2:] & free[1:-1, :-2]
        mask_y = free_c & free[2:, 1:-1] & free[:-2, 1:-1]
        dpdx = (self.p[1:-1, 2:] - self.p[1:-1, :-2]) / (2.0 * dx)
        dpdy = (self.p[2:, 1:-1] - self.p[:-2, 1:-1]) / (2.0 * dy)
        self.u[1:-1, 1:-1] -= coef * cp.where(mask_x, dpdx, 0.0)
        self.v[1:-1, 1:-1] -= coef * cp.where(mask_y, dpdy, 0.0)
        
        self.apply_ghost_cell_bc()
        self.reforzar_impermeabilidad()
        # ⭐ Usar after_projection=True para no sobrescribir outflow con Neumann
        self.apply_boundaries(after_projection=True)
        
        # Asegurar referencia de presión antes de salir (malla fina sin Dirichlet)
        try:
            self._anchor_pressure()
        except Exception:
            pass

        # Opcional: retornar info de diagnostico
        return {
            'iterations': it + 1,
            'converged': converged,
            'div_before': div_mean_before,
            'div_after': div_mean_after,
            'adaptive_max': adaptive_max_iter
        }
    
    def _compute_divergence_field(self):
        """
        Calcula el campo completo de divergencia del^2 u = du/dx + dv/dy.
        Usa stencil adaptativo cerca de sólidos para evitar gradientes explosivos.
        
        Retorna:
            cp.array: Campo de divergencia (ny, nx)
        """
        # Diferencias centrales estándar:
        # ∂u/∂x ≈ (u_{i,j+1} - u_{i,j-1}) / (2 Δx)
        # ∂v/∂y ≈ (v_{i+1,j} - v_{i-1,j}) / (2 Δy)
        return self._compute_divergence_field_uv(self.u, self.v)

    def _compute_divergence_field_uv(self, u, v):
        """
        Divergencia ∇·(u,v) usando kernel CUDA fusionado.
        Reemplaza ~20 operaciones CuPy (boolean masks + fancy indexing) con 1 lanzamiento.
        """
        div = cp.zeros_like(self.p, dtype=cp.float32)
        ny, nx = div.shape
        total = nx * ny
        block = 256
        grid = (total + block - 1) // block
        self._divergence_kernel(
            (grid,), (block,),
            (u.ravel(), v.ravel(), self.solid.ravel(), div.ravel(),
             cp.float32(1.0 / (2.0 * self.dx)),
             cp.float32(1.0 / (2.0 * self.dy)),
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

        inv2dx = cp.float32(0.5 / dx)
        inv2dy = cp.float32(0.5 / dy)

        # centrales (interior)
        g11[:, 1:-1] = (u[:, 2:] - u[:, :-2]) * inv2dx
        g12[1:-1, :] = (u[2:, :] - u[:-2, :]) * inv2dy
        g21[:, 1:-1] = (v[:, 2:] - v[:, :-2]) * inv2dx
        g22[1:-1, :] = (v[2:, :] - v[:-2, :]) * inv2dy

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

        # Mantener cálculos en GPU: representar dx,dy como scalars de CuPy
        dx_cp = cp.asarray(dx, dtype=cp.float32)
        dy_cp = cp.asarray(dy, dtype=cp.float32)
        Delta = cp.sqrt(dx_cp * dy_cp)
        coef = (cp.float32(Cw) * Delta) ** 2

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

    def compute_wake_viscosity(self):
        """
        Calcula viscosidad turbulenta extra en la estela del perfil.
        Simula la disipación 3D spanwise ausente en simulaciones 2D.
        
        nu_wake = C_estela * Delta^2 * |omega| * mascara_estela
        
        La máscara de estela se construye detectando la región aguas abajo
        del sólido, en dirección del flujo libre.
        """
        dx = self.dx; dy = self.dy
        u = self.u; v = self.v
        ny, nx = u.shape
        
        # --- Vorticidad: omega = dv/dx - du/dy ---
        inv2dx = cp.float32(0.5 / dx)
        inv2dy = cp.float32(0.5 / dy)
        
        dvdx = cp.zeros_like(u, dtype=cp.float32)
        dudy = cp.zeros_like(u, dtype=cp.float32)
        dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) * inv2dx
        dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) * inv2dy
        omega = cp.abs(dvdx - dudy)
        
        # --- Construir máscara de estela ---
        if self._wake_mask is None or not hasattr(self, '_wake_alpha_cache'):
            self._recompute_wake_mask()
        
        # --- nu_wake = C * Delta^2 * |omega| * mascara ---
        Delta2 = cp.float32(dx * dy)
        C = cp.float32(self.C_estela)
        
        nu_wake = C * Delta2 * omega * self._wake_mask
        
        # No-negatividad
        nu_wake = cp.maximum(nu_wake, cp.float32(0.0))
        nu_wake[self.solid] = cp.float32(0.0)
        
        return nu_wake.astype(cp.float32, copy=False)
    
    def _recompute_wake_mask(self):
        """
        Precomputa la máscara de estela basada en la posición del sólido
        y la dirección del flujo libre.
        La estela se define como la región aguas abajo del borde de salida
        del perfil, con un ensanchamiento progresivo.
        """
        ny, nx = self.u.shape
        solid_np = cp.asnumpy(self.solid)
        
        # Encontrar bounding box del sólido
        solid_rows, solid_cols = np.where(solid_np)
        if len(solid_rows) == 0:
            self._wake_mask = cp.zeros((ny, nx), dtype=cp.float32)
            self._wake_alpha_cache = getattr(self, 'alpha_deg', 0.0)
            return
        
        # Borde de salida: columna máxima del sólido (en dirección x)
        j_te = int(np.max(solid_cols))  # trailing edge column
        i_min_solid = int(np.min(solid_rows))
        i_max_solid = int(np.max(solid_rows))
        i_center = (i_min_solid + i_max_solid) // 2
        espesor_solid = i_max_solid - i_min_solid + 1
        
        # Ángulo del flujo (para orientar la estela)
        alpha_rad = self._get_freestream_angle_rad()
        cos_a = np.cos(alpha_rad)
        sin_a = np.sin(alpha_rad)
        
        # Crear coordenadas de malla
        jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))
        
        # Vector desde el trailing edge a cada punto
        dj = jj - j_te  # dirección x (columnas)
        di = ii - i_center  # dirección y (filas, invertida)
        
        # Distancia aguas abajo (proyección en dirección del flujo)
        # En la malla: x crece con j, y crece hacia abajo con i
        dist_downstream = dj * cos_a - di * sin_a  # positivo = aguas abajo
        
        # Distancia perpendicular al flujo
        dist_perp = np.abs(dj * sin_a + di * cos_a)
        
        # La estela empieza justo después del sólido y se ensancha
        # Ensanchamiento: el ancho de la estela crece como sqrt(x)
        wake_width = espesor_solid * 0.5 + np.sqrt(np.maximum(dist_downstream, 0.0)) * 0.5
        
        # Máscara: aguas abajo y dentro del cono de estela
        mascara = (dist_downstream > 0) & (dist_perp < wake_width) & (~solid_np)
        
        # Intensidad decreciente con la distancia (más fuerte cerca del perfil)
        longitud_estela = nx - j_te  # longitud total disponible
        intensidad = np.where(
            mascara,
            np.exp(-0.5 * dist_downstream / max(longitud_estela * 0.5, 1.0)),
            0.0
        ).astype(np.float32)
        
        self._wake_mask = cp.asarray(intensidad, dtype=cp.float32)
        self._wake_alpha_cache = getattr(self, 'alpha_deg', 0.0)

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
                   max_iter=2000, min_iter=5, check_every=5, 
                   verbose=False, print_every=50,
                   usar_operador_spd=True,
                   modo_adaptativo=False,
                   detectar_estancamiento=False):
        """
        Proyección incompresible usando Gradiente Conjugado (CG) precondicionado.
        
        CG converge 10-100× más rápido que Jacobi para sistemas elípticos.
        Usa precondicionador Jacobi (diagonal) para mejorar el número de condición.
        
        Parámetros adicionales:
            verbose: activar impresiones de monitoreo
            print_every: cada cuantas iteraciones imprimir (si verbose=True)
        """
        dx = self.dx
        dy = self.dy
        assert abs(dx - dy) < 1e-6, "Esta version asume dx ~ dy."
        
        h2 = cp.float32(dx * dx)
        rho_f = cp.float32(rho_sim)
        dt_f = cp.float32(dt)
        ny, nx = self.p.shape
        
        # Mascara de fluido
        free = ~self.solid
        
        # ⭐ Aplicar Ghost-Cell IBM antes de calcular divergencia
        self.apply_ghost_cell_bc()
        
        # ============================================================
        # PRE-CHECK: Medir divergencia ANTES
        # ============================================================
        div_before = self._compute_divergence_field()
        div_mean_before = float(cp.mean(cp.abs(div_before[free])))
        
        if verbose:
            print(f"[CG] Divergencia inicial: {div_mean_before:.6e}")
        
        # Iteraciones: por defecto NO usar modo adaptativo para evitar deriva (sub-resolución)
        if modo_adaptativo:
            if div_mean_before < tol_div * 0.1:
                adaptive_max_iter = 10
            elif div_mean_before < tol_div:
                adaptive_max_iter = 100
            else:
                adaptive_max_iter = max_iter
        else:
            adaptive_max_iter = max_iter
        
        # ============================================================
        # PASO 1: Construir RHS = (rho/dt) * div(u*)
        # ============================================================
        # Comentarios:
        # - Se resuelve A x = b con A la discretización del Laplaciano (5-point)
        # - Precondicionador Jacobi implementado en `_precond_jacobi_kernel` (M^-1 r)
        # - `_laplacian_kernel` aplica A * x con la forma:
        #     (x_E + x_W + x_N + x_S - 4 x_C) / h^2
        rhs = cp.zeros((ny, nx), dtype=cp.float32)
        rhs[1:-1, 1:-1] = (rho_f / dt_f) * div_before[1:-1, 1:-1]

        # Compatibilidad Neumann (si no hay Dirichlet de presión): sum(rhs)=0 en fluido
        try:
            hay_dirichlet = bool(cp.any(self.fixed_pressure_mask))
        except Exception:
            hay_dirichlet = False
        if not hay_dirichlet:
            mean_rhs = cp.mean(rhs[free])
            rhs[free] = rhs[free] - mean_rhs
        
        # Aplanar arrays para kernels
        rhs_flat = rhs.ravel().astype(cp.float32, copy=False)

        # ============================================================
        # CG requiere operador SPD. Nuestro kernel devuelve Laplaciano (definido negativo).
        # Para garantizar SPD usamos A = -L y b = -rhs.
        # Esto mejora estabilidad y evita comportamientos raros a largo plazo.
        # ============================================================
        if usar_operador_spd:
            rhs_flat = -rhs_flat

        def _aplicar_A(vec_flat):
            out_flat = self._laplacian_kernel_masked(
                self.solid.ravel(), vec_flat, h2, cp.int32(nx), cp.int32(ny), size=vec_flat.size
            )
            return -out_flat if usar_operador_spd else out_flat

        def _aplicar_Minv(res_flat):
            out_flat = self._precond_jacobi_kernel_masked(
                self.solid.ravel(), res_flat, h2, cp.int32(nx), cp.int32(ny), size=res_flat.size
            )
            # El kernel implementa -(h^2/n_nb)*r. Para A=-L, Minv debe ser +(h^2/n_nb)*r.
            return -out_flat if usar_operador_spd else out_flat
        
        # ============================================================
        # PASO 2: Inicializar CG
        # ============================================================
        # x0 = presion actual (warm start)
        x = self.p.ravel().astype(cp.float32, copy=True)
        # Forzar valores fijos en x0 (si hay celdas Dirichlet puntuales)
        try:
            if cp.any(self.fixed_pressure_mask):
                mask_flat = self.fixed_pressure_mask.ravel()
                x[mask_flat] = self.fixed_pressure_value
        except Exception:
            pass
        
        # r0 = b - Ax0
        Ax = cp.empty_like(x)
        Ax = _aplicar_A(x)
        r = rhs_flat - Ax
        # Anular residuo en nodos con presión fijada (no deben moverse)
        try:
            if cp.any(self.fixed_pressure_mask):
                r[self.fixed_pressure_mask.ravel()] = 0.0
        except Exception:
            pass
        
        # z0 = M^-1 r0 (precondicionador Jacobi)
        z = cp.empty_like(r)
        z = _aplicar_Minv(r)
        
        # p0 = z0
        p = z.copy()
        
        # r^T z inicial
        rz = cp.dot(r, z)
        
        # ============================================================
        # PASO 3: Iteraciones CG con chequeo adaptativo + detección de estancamiento
        # ============================================================
        converged = False
        div_mean_after = div_mean_before  # Inicializar
        div_history = []
        stall_window = 50
        stall_tol_abs = 1e-8
        stall_tol_rel = 1e-8
        
        # ⭐ Variables para detectar oscilación cerca de la tolerancia
        oscilacion_detectada = False
        ventana_oscilacion = 10  # Últimas N mediciones para detectar oscilación
        factor_tolerancia_oscilacion = 2.0  # Considerar "cerca" si div < factor × tol_div
        
        for it in range(adaptive_max_iter):
            # Ap = A * p
            Ap = cp.empty_like(p)
            Ap = _aplicar_A(p)
            
            # alpha = (r^T z) / (p^T Ap)
            pAp = cp.dot(p, Ap)
            if cp.abs(pAp) < 1e-30:
                if verbose:
                    print(f"[CG] Terminación temprana: pAp ~ 0 en iter {it+1}")
                break  # Evitar division por cero
            alpha = rz / pAp
            
            # x = x + alpha*p
            x = x + alpha * p
            # Reimponer valores fijos en x durante iteraciones
            try:
                if cp.any(self.fixed_pressure_mask):
                    x[self.fixed_pressure_mask.ravel()] = self.fixed_pressure_value
            except Exception:
                pass
            
            # r = r - alpha*Ap
            r = r - alpha * Ap
            # Asegurar residuo cero en nodos fijados
            try:
                if cp.any(self.fixed_pressure_mask):
                    r[self.fixed_pressure_mask.ravel()] = 0.0
            except Exception:
                pass
            
            # Determinar si calcular divergencia e imprimir
            debe_chequear = (it >= min_iter and it % check_every == 0)
            debe_imprimir = (verbose and (it + 1) % print_every == 0)
            
            # Calcular divergencia si toca chequear O si toca imprimir
            if debe_chequear or debe_imprimir:
                # Actualizar presion temporal
                self.p = x.reshape(ny, nx)
                
                # Corregir velocidad temporalmente
                u_temp = self.u.copy()
                v_temp = self.v.copy()
                
                coef = dt_f / rho_f
                free_c = free[1:-1, 1:-1]
                mask_x = free_c & free[1:-1, 2:] & free[1:-1, :-2]
                mask_y = free_c & free[2:, 1:-1] & free[:-2, 1:-1]
                dpdx = (self.p[1:-1, 2:] - self.p[1:-1, :-2]) / (2.0 * dx)
                dpdy = (self.p[2:, 1:-1] - self.p[:-2, 1:-1]) / (2.0 * dy)
                u_temp[1:-1, 1:-1] -= coef * cp.where(mask_x, dpdx, 0.0)
                v_temp[1:-1, 1:-1] -= coef * cp.where(mask_y, dpdy, 0.0)
                
                # ⭐ CRÍTICO: Anular velocidad en sólido ANTES de calcular divergencia
                u_temp[self.solid] = 0.0
                v_temp[self.solid] = 0.0
                
                # Calcular div(u) del campo corregido (consistente con sólidos)
                div_after = self._compute_divergence_field_uv(u_temp, v_temp)
                div_mean_after = float(cp.mean(cp.abs(div_after[free])))
                
                # Guardar en historial para detectar estancamiento
                div_history.append(div_mean_after)
                
                # Detección de estancamiento (opcional)
                if detectar_estancamiento and len(div_history) > stall_window:
                    div_recent = div_history[-stall_window:]
                    div_min_recent = min(div_recent)
                    div_max_recent = max(div_recent)
                    div_range = div_max_recent - div_min_recent
                    
                    # Cambio relativo: (max-min)/min
                    if div_min_recent > 0:
                        cambio_relativo = div_range / div_min_recent
                    else:
                        cambio_relativo = div_range
                    
                    # Estancamiento si no cambia al 4º decimal (abs o relativo)
                    if (div_range < stall_tol_abs) or (cambio_relativo < stall_tol_rel):
                        if verbose:
                            print(f"[CG] ⚠ ESTANCAMIENTO detectado en iter {it+1}")
                            print(f"     Divergencia estancada en {div_mean_after:.6e}")
                            print(f"     Rango abs últimas {stall_window} iters: {div_range:.6e} < {stall_tol_abs:.6e}")
                            print(f"     Cambio relativo últimas {stall_window} iters: {cambio_relativo:.6e} < {stall_tol_rel:.6e}")
                            print(f"     → Continuando con siguiente paso dt")
                        converged = False
                        break
                
                # Imprimir progreso si corresponde
                if debe_imprimir:
                    print(f"  Iter {it+1:5d}: div={div_mean_after:.6e}")
                
                # ⭐ DETECCIÓN DE OSCILACIÓN cerca de la tolerancia
                # Si estamos cerca de la tolerancia, verificar si hay oscilación
                if debe_chequear and len(div_history) >= ventana_oscilacion:
                    # Tomar últimas mediciones
                    div_reciente = div_history[-ventana_oscilacion:]
                    div_promedio = sum(div_reciente) / len(div_reciente)
                    
                    # Verificar si estamos cerca de la tolerancia
                    if div_promedio < factor_tolerancia_oscilacion * tol_div:
                        # Detectar oscilación: si sube y baja repetidamente
                        oscilaciones = 0
                        for i in range(1, len(div_reciente)):
                            if (div_reciente[i] > div_reciente[i-1] and i > 1 and div_reciente[i-1] < div_reciente[i-2]) or \
                               (div_reciente[i] < div_reciente[i-1] and i > 1 and div_reciente[i-1] > div_reciente[i-2]):
                                oscilaciones += 1
                        
                        # Si hay más de 3 oscilaciones en la ventana, considerarlo oscilante
                        if oscilaciones >= 3:
                            oscilacion_detectada = True
                            if verbose and not converged:
                                print(f"[CG] 🔄 Oscilación detectada cerca de tolerancia en iter {it+1}")
                                print(f"     div_promedio={div_promedio:.6e}, oscilaciones={oscilaciones}")
                                print(f"     → Continuando hasta max_iter para reducir más la divergencia")
                
                # Chequear convergencia si corresponde
                if debe_chequear and div_mean_after < tol_div:
                    # ⭐ Si hay oscilación, NO salir todavía (continuar hasta max_iter)
                    if not oscilacion_detectada:
                        converged = True
                        if verbose:
                            print(f"[CG] ✓ Convergencia alcanzada en iter {it+1}: div={div_mean_after:.6e}")
                        break
                    else:
                        # Oscilando: marcar como convergido pero seguir iterando
                        converged = True
                        if verbose and it == min_iter + check_every:
                            print(f"[CG] ⚙️  Convergencia con oscilación: continuando hasta max_iter")

            
            # z = M^-1 r (precondicionador)
            z = _aplicar_Minv(r)
            
            # beta = (r_new^T z_new) / (r_old^T z_old)
            rz_new = cp.dot(r, z)
            beta = rz_new / rz
            rz = rz_new
            
            # p = z + beta*p
            p = z + beta * p
        
        # Mensaje final según el estado
        if not converged and verbose:
            print(f"[CG] No convergió en {adaptive_max_iter} iters: div={div_mean_after:.6e}")
        elif converged and oscilacion_detectada and verbose:
            print(f"[CG] ✓ Completó {adaptive_max_iter} iters por oscilación: div_final={div_mean_after:.6e}")
        
        # ============================================================
        # PASO 4: Escribir presion y corregir velocidad
        # ============================================================
        self.p = x.reshape(ny, nx)
        # Reimponer presion fija en celdas marcadas (CG)
        try:
            if cp.any(self.fixed_pressure_mask):
                self.p[self.fixed_pressure_mask] = self.fixed_pressure_value
        except Exception:
            pass
        
        # u <- u - (dt/rho) grad(p)
        coef = dt_f / rho_f
        free_c = free[1:-1, 1:-1]
        mask_x = free_c & free[1:-1, 2:] & free[1:-1, :-2]
        mask_y = free_c & free[2:, 1:-1] & free[:-2, 1:-1]
        dpdx = (self.p[1:-1, 2:] - self.p[1:-1, :-2]) / (2.0 * dx)
        dpdy = (self.p[2:, 1:-1] - self.p[:-2, 1:-1]) / (2.0 * dy)
        self.u[1:-1, 1:-1] -= coef * cp.where(mask_x, dpdx, 0.0)
        self.v[1:-1, 1:-1] -= coef * cp.where(mask_y, dpdy, 0.0)
        
        # Aplicar Ghost-Cell IBM y reforzar impermeabilidad
        self.apply_ghost_cell_bc()
        self.reforzar_impermeabilidad()
        # ⭐ Usar after_projection=True para no sobrescribir outflow con Neumann
        self.apply_boundaries(after_projection=True)
        
        # Asegurar referencia de presión antes de salir (malla fina sin Dirichlet)
        try:
            self._anchor_pressure()
        except Exception:
            pass

        # Retornar info de diagnostico
        return {
            'iterations': it + 1,
            'converged': converged,
            'div_before': div_mean_before,
            'div_after': div_mean_after,
            'adaptive_max': adaptive_max_iter
        }

    def project_multigrid(self, rho_sim, dt, tol_div=1e-2,
                          max_outer=20, cycles_per_outer=5,
                          niveles_max=4, pre_suavizado=2, post_suavizado=2,
                          omega=1.15, verbose=False,
                          modo_adaptativo=True):
        """
        Proyección incompresible con defect-correction iterativo + multigrid.
        Smoother: Red-Black Gauss-Seidel SOR (CUDA kernel in-place).

        Estrategia: múltiples iteraciones externas, cada una re-computa la
        divergencia residual y resuelve un nuevo Poisson.  Esto elimina la
        inconsistencia entre el Laplaciano compacto del solver y el gradiente
        central de la corrección de velocidad, logrando ~10× mejor reducción
        de divergencia para el mismo número total de V-cycles.

        Parámetros:
            tol_div: tolerancia media sobre |div(u)|
            max_outer: iteraciones externas de defect-correction
            cycles_per_outer: V-cycles por iteración externa
            niveles_max: niveles de coarsening (2×2)
            pre_suavizado/post_suavizado: iteraciones GS-SOR por nivel
            omega: factor sobre-relajación (1.0-1.5, típico 1.1-1.2)
            modo_adaptativo: ajustar outers según div inicial
        """
        dx = self.dx
        dy = self.dy
        assert abs(dx - dy) < 1e-6, "Esta version asume dx ~ dy."

        rho_f = cp.float32(rho_sim)
        dt_f  = cp.float32(dt)
        ny, nx = self.p.shape

        # Inicializar jerarquía si falta o cambió niveles_max
        if not getattr(self, '_mg_initialized', False) or self._mg_niveles_max_param != niveles_max:
            self._init_mg_hierarchy(niveles_max)

        # --- Referencias cacheadas ---
        nivel_max    = self._mg_niveles
        solids       = self._mg_solids
        solids_flat  = self._mg_solids_flat
        h2_levels    = self._mg_h2
        bufs         = self._mg_bufs
        dir_masks    = self._mg_dirichlet
        hay_dir      = self._mg_hay_dirichlet
        fixed_flat   = self._mg_fixed_mask_flat
        fixed_val    = self.fixed_pressure_value
        free         = self._mg_free
        n_free       = self._mg_n_free
        mask_x_f     = self._mg_mask_x_f32
        mask_y_f     = self._mg_mask_y_f32
        kernel_gs    = self._rb_gs_sor_kernel
        omega_f32    = cp.float32(omega)
        block_sz     = 256

        # Aplicar Ghost-Cell IBM antes de proyección
        self.apply_ghost_cell_bc()

        # Constantes
        coef     = dt_f / rho_f
        inv_2dx  = cp.float32(1.0 / (2.0 * dx))
        inv_2dy  = cp.float32(1.0 / (2.0 * dy))

        # Divergencia inicial
        div_before = self._compute_divergence_field()
        div_mean_before = float(cp.sum(cp.abs(div_before[free]))) / n_free

        if verbose:
            print(f"[MG] Divergencia inicial: {div_mean_before:.6e}")

        # Adaptar outers
        if modo_adaptativo:
            if div_mean_before < tol_div * 0.5:
                n_outer = max(1, max_outer // 8)
            elif div_mean_before < tol_div:
                n_outer = max(2, max_outer // 2)
            else:
                n_outer = max_outer
        else:
            n_outer = max_outer

        # =========================================================
        # FUNCIONES AUXILIARES (closures optimizadas con kdims cacheados)
        # =========================================================
        kdims      = self._mg_kdims
        restrict_k = self._restrict_kernel
        prolong_k  = self._prolongate_add_kernel
        vc_kernel  = self._velocity_correction_kernel
        p0_i32     = cp.int32(0)
        p1_i32     = cp.int32(1)

        def _suavizar(p_lvl, rhs_lvl, nivel, n_iter):
            kd = kdims[nivel]
            p_flat = p_lvl.ravel()
            rhs_flat = rhs_lvl.ravel()
            sf = solids_flat[nivel]
            h2_l = h2_levels[nivel]
            g = (kd['grid'],)
            b = (block_sz,)
            nx_i = kd['nx_i']
            ny_i = kd['ny_i']
            for _ in range(int(n_iter)):
                kernel_gs(g, b, (sf, p_flat, rhs_flat, h2_l, omega_f32, nx_i, ny_i, p0_i32))
                kernel_gs(g, b, (sf, p_flat, rhs_flat, h2_l, omega_f32, nx_i, ny_i, p1_i32))
            if hay_dir:
                dm = dir_masks[nivel]
                if nivel == 0:
                    p_flat[fixed_flat] = fixed_val
                else:
                    p_lvl[dm] = 0.0

        def _aplicar_bc(p_nivel, nivel):
            ny_n, nx_n = p_nivel.shape
            if not hay_dir:
                # Neumann puro: copia simple (sin indexing complejo)
                if nx_n >= 2:
                    p_nivel[:, 0] = p_nivel[:, 1]
                    p_nivel[:, -1] = p_nivel[:, -2]
                if ny_n >= 2:
                    p_nivel[0, :] = p_nivel[1, :]
                    p_nivel[-1, :] = p_nivel[-2, :]
            else:
                solid_n = solids[nivel]
                fl = ~solid_n
                dm = dir_masks[nivel]
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
                if nivel == 0:
                    p_nivel[dm] = fixed_val
                else:
                    p_nivel[dm] = 0.0

        # =========================================================
        # V-CYCLE con kernels fusionados (restrict/prolongate)
        # Ahorra ~2200 lanzamientos de kernel por proyección
        # =========================================================
        def _v_cycle(p_in, rhs_in):
            p_by_level = [None] * (nivel_max + 1)
            rhs_by_level = [None] * (nivel_max + 1)
            p_by_level[0] = p_in
            rhs_by_level[0] = rhs_in

            # --- Descenso ---
            for lvl in range(nivel_max):
                kd = kdims[lvl]
                _suavizar(p_by_level[lvl], rhs_by_level[lvl], lvl, pre_suavizado)
                # Residuo: res = rhs - Lap(p)
                res_buf = bufs[lvl]['res']
                lap = self._laplacian_kernel_masked(
                    solids_flat[lvl], p_by_level[lvl].ravel(), h2_levels[lvl],
                    kd['nx_i'], kd['ny_i'], size=kd['total'])
                cp.subtract(rhs_by_level[lvl].ravel(), lap, out=res_buf.ravel())
                # Restricción con kernel fusionado (1 lanzamiento en vez de ~8)
                kd_c = kdims[lvl + 1]
                if kd_c['ny'] < 2 or kd_c['nx'] < 2:
                    for lvl_up in range(lvl, -1, -1):
                        _suavizar(p_by_level[lvl_up], rhs_by_level[lvl_up], lvl_up, post_suavizado)
                        _aplicar_bc(p_by_level[lvl_up], lvl_up)
                    return p_by_level[0]
                rhs_coarse = bufs[lvl + 1]['rhs']
                restrict_k(
                    (kd_c['grid'],), (block_sz,),
                    (res_buf.ravel(), rhs_coarse.ravel(),
                     kd['nx_i'], kd['ny_i'], kd_c['nx_i'], kd_c['ny_i']))
                rhs_by_level[lvl + 1] = rhs_coarse
                e_c = bufs[lvl + 1]['error']
                e_c[:] = 0.0
                p_by_level[lvl + 1] = e_c

            # --- Nivel más grueso: suavizado extra ---
            _suavizar(p_by_level[nivel_max], rhs_by_level[nivel_max],
                      nivel_max, pre_suavizado + 10)
            _aplicar_bc(p_by_level[nivel_max], nivel_max)

            # --- Ascenso con prolongación fusionada (add directo, 1 vs ~15 launches) ---
            for lvl in range(nivel_max - 1, -1, -1):
                kd_c = kdims[lvl + 1]
                kd_f = kdims[lvl]
                prolong_k(
                    (kd_f['grid'],), (block_sz,),
                    (p_by_level[lvl + 1].ravel(), p_by_level[lvl].ravel(),
                     kd_c['nx_i'], kd_c['ny_i'], kd_f['nx_i'], kd_f['ny_i']))
                _suavizar(p_by_level[lvl], rhs_by_level[lvl], lvl, post_suavizado)
                _aplicar_bc(p_by_level[lvl], lvl)

            return p_by_level[0]

        # =========================================================
        # DEFECT-CORRECTION: iteraciones externas optimizadas
        # - Corrección de velocidad fusionada (1 kernel vs ~8 ops)
        # - Ghost-Cell IBM + impermeabilidad después de cada corrección
        # - RHS reutiliza buffer pre-alocado (evita cp.zeros cada outer)
        # - Neumann mean subtraction sin GPU→CPU sync
        # =========================================================
        p_acumulada = cp.zeros((ny, nx), dtype=cp.float32)
        total_cycles = 0
        converged = False
        div_mean_current = div_mean_before

        total_cells = ny * nx
        grid_vc = (total_cells + block_sz - 1) // block_sz
        nx_i32 = cp.int32(nx)
        ny_i32 = cp.int32(ny)
        solid_flat = self.solid.ravel()

        for outer in range(n_outer):
            div_field = self._compute_divergence_field()
            div_mean_current = float(cp.sum(cp.abs(div_field[free]))) / n_free

            if outer > 0 and div_mean_current < tol_div:
                converged = True
                if verbose:
                    print(f"[MG] Convergencia div en outer {outer}: div={div_mean_current:.6e}")
                break

            # RHS desde divergencia (reutilizar buffer pre-alocado)
            rhs = bufs[0]['rhs']
            rhs[:] = 0.0
            rhs[1:-1, 1:-1] = (rho_f / dt_f) * div_field[1:-1, 1:-1]
            if not hay_dir:
                mean_rhs = cp.sum(rhs[free]) / cp.float32(n_free)
                rhs[free] -= mean_rhs

            if outer == 0:
                p_corr = self.p.copy()
            else:
                p_corr = cp.zeros((ny, nx), dtype=cp.float32)
            for cyc in range(cycles_per_outer):
                p_corr = _v_cycle(p_corr, rhs)
                if hay_dir:
                    p_corr[self.fixed_pressure_mask] = self.fixed_pressure_value
            total_cycles += cycles_per_outer

            _aplicar_bc(p_corr, 0)
            p_acumulada += p_corr

            # Corrección velocidad fusionada (1 kernel: grad(p) + sustracción)
            vc_kernel(
                (grid_vc,), (block_sz,),
                (self.u.ravel(), self.v.ravel(), p_corr.ravel(), solid_flat,
                 coef, inv_2dx, inv_2dy, nx_i32, ny_i32))

            # Ghost-Cell IBM + impermeabilidad después de cada corrección
            self.apply_ghost_cell_bc()
            self.reforzar_impermeabilidad()

            if verbose:
                div_after_outer = self._compute_divergence_field()
                d = float(cp.sum(cp.abs(div_after_outer[free]))) / n_free
                print(f"[MG] Outer {outer+1}/{n_outer} ({cycles_per_outer}Vc): div={d:.6e}")

        # =========================================================
        # FINALIZACIÓN
        # =========================================================
        self.p = p_acumulada
        self._aplicar_bc_presion_neumann()
        if hay_dir:
            self.p[self.fixed_pressure_mask] = self.fixed_pressure_value

        # Ghost-Cell IBM final + refuerzo
        self.apply_ghost_cell_bc()
        self.reforzar_impermeabilidad()
        self.apply_boundaries(after_projection=True)

        if not hay_dir:
            try:
                self._anchor_pressure()
            except Exception:
                pass

        div_after = self._compute_divergence_field()
        div_mean_after = float(cp.sum(cp.abs(div_after[free]))) / n_free

        return {
            'cycles': total_cycles,
            'outers': outer + 1,
            'converged': converged or (div_mean_after < tol_div),
            'div_before': div_mean_before,
            'div_after': div_mean_after,
            'n_outer_used': n_outer
        }

    '''Sólidos'''

    def _xy_grids(self):
        # Mallas físicas para rasterizar sólidos
        x = cp.arange(self.nx, dtype=cp.float32) * self.dx
        y = cp.arange(self.ny, dtype=cp.float32) * self.dy
        XX, YY = cp.meshgrid(x, y, indexing='xy')
        return XX, YY

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
                              plot=True):
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

        # Coordenadas de centros de celdas
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

        # Tamaño de figura proporcional al dominio para mantener proporciones
        ar = self.Ly / self.Lx
        fig = plt.figure(figsize=(12, 12 * ar))
        im = plt.imshow(
            speed_np,
            origin="lower",
            interpolation="none",
            cmap=cmap,
            extent=[0, self.Lx, 0, self.Ly]
        )
        # Asegurar que no se deforme: mismo escalado en x e y
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
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
                ar = self.Ly / self.Lx
                fig = plt.figure(figsize=(10, 10 * ar))
                im = plt.imshow(
                    speed_np,
                    origin="lower",
                    interpolation="none",
                    cmap="rainbow",
                    extent=[0, self.Lx, 0, self.Ly]
                )
                # Asegurar que no se deforme: mismo escalado en x e y
                ax = plt.gca()
                ax.set_aspect('equal', adjustable='box')
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
                plt.imshow(cp.asnumpy(speed), origin="lower",
                           extent=[0, self.Lx, 0, self.Ly], cmap="Greys", alpha=0.3)
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

    def compute_surface_forces(self, mu, separate_components=True):
        """
        Integral de esfuerzos en la superficie del sólido (2D, steady):
        - T = -p n + mu (∇u + ∇u^T) · n
        - F_total = sum_boundary T ds
        Devuelve:
          - dict con Fx, Fy, Fx_p, Fy_p, Fx_v, Fy_v si separate_components=True
          - si False, solo Fx, Fy
        """
        # 1) Frontera del sólido
        boundary = self._solid_boundary_mask(use_diagonals=True)

        # 2) Normales (desde sólido hacia fluido)
        _, nx, ny = self._signed_distance_and_normals()

        # 3) Longitud elemental
        ds = cp.float32(min(self.dx, self.dy))

        # 4) Campos y gradientes en GPU
        p = self.p
        u = self.u
        v = self.v
        dx = self.dx; dy = self.dy
        inv_dx = cp.float32(1.0 / dx)
        inv_dy = cp.float32(1.0 / dy)

        # Diferencias centradas
        du_dx = cp.zeros_like(u, dtype=cp.float32)
        du_dy = cp.zeros_like(u, dtype=cp.float32)
        dv_dx = cp.zeros_like(v, dtype=cp.float32)
        dv_dy = cp.zeros_like(v, dtype=cp.float32)

        du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) * (0.5 * inv_dx)
        du_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) * (0.5 * inv_dy)
        dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) * (0.5 * inv_dx)
        dv_dy[1:-1, :] = (v[2:, :] - v[:-2, :]) * (0.5 * inv_dy)

        # 5) Tracción de presión: -p n
        Tx_p = -p * nx
        Ty_p = -p * ny

        # 6) Parte viscosa: -mu (∇u + ∇u^T) · n
        # (signo negativo porque n apunta hacia el fluido y queremos fuerza sobre el sólido)
        # Suma simétrica: (∂u/∂y + ∂v/∂x)
        sym_xy = du_dy + dv_dx

        # Componentes
        Tx_v = -mu * (2.0 * du_dx * nx + sym_xy * ny)
        Ty_v = -mu * (sym_xy * nx + 2.0 * dv_dy * ny)

        # 7) Integración sobre frontera
        w = boundary.astype(cp.float32)  # peso 1 en frontera, 0 fuera

        Fx_p = cp.sum(Tx_p * w) * ds
        Fy_p = cp.sum(Ty_p * w) * ds
        Fx_v = cp.sum(Tx_v * w) * ds
        Fy_v = cp.sum(Ty_v * w) * ds

        Fx = Fx_p + Fx_v
        Fy = Fy_p + Fy_v

        if separate_components:
            return {
                "Fx": float(Fx), "Fy": float(Fy),
                "Fx_p": float(Fx_p), "Fy_p": float(Fy_p),
                "Fx_v": float(Fx_v), "Fy_v": float(Fy_v)
            }
        else:
            return {"Fx": float(Fx), "Fy": float(Fy)}

    def compute_surface_forces2(self, mu, separate_components=True):
        """
        Integral de esfuerzos en dos capas alrededor del sólido:
        - Capa 1: celdas de fluido adyacentes al sólido (frontera) muestreadas en 0.5 pasos sobre la normal.
        - Capa 2: anillo exterior (una celda más hacia el fluido) muestreado en 1.5 pasos sobre la normal.
        Se suman las fuerzas de ambas capas.
        """
        # 1) Frontera del sólido (capa 1)
        boundary1 = self._solid_boundary_mask(use_diagonals=True)

        # 2) Normales (desde sólido hacia fluido)
        sd, nx_all, ny_all = self._signed_distance_and_normals()
        # Normal solo donde hay capa válida
        nx1 = nx_all * boundary1
        ny1 = ny_all * boundary1

        # 3) Construir capa 2: dilatación de boundary1 hacia el fluido, excluyendo sólido y capa1
        solid = self.solid
        fluid = ~solid
        dil = (
            cp.roll(boundary1, 1, axis=0) |
            cp.roll(boundary1, -1, axis=0) |
            cp.roll(boundary1, 1, axis=1) |
            cp.roll(boundary1, -1, axis=1)
        )
        # incluir diagonales también para coherencia
        dil |= cp.roll(cp.roll(boundary1, 1, axis=0), 1, axis=1)
        dil |= cp.roll(cp.roll(boundary1, 1, axis=0), -1, axis=1)
        dil |= cp.roll(cp.roll(boundary1, -1, axis=0), 1, axis=1)
        dil |= cp.roll(cp.roll(boundary1, -1, axis=0), -1, axis=1)
        boundary2 = fluid & dil & (~boundary1)

        # Excluir bordes del dominio
        for b in (boundary1, boundary2):
            b[0, :] = False; b[-1, :] = False; b[:, 0] = False; b[:, -1] = False

        # Normales para capa 2 (del mismo campo, enmascaradas)
        nx2 = nx_all * boundary2
        ny2 = ny_all * boundary2

        dx = self.dx; dy = self.dy
        assert abs(dx - dy) < 1e-6, "Esta integración asume dx≈dy"

        # Elemento de arco con corrección para diagonales
        diag_mask1 = (
            cp.roll(cp.roll(solid, 1, axis=0), 1, axis=1) |
            cp.roll(cp.roll(solid, 1, axis=0), -1, axis=1) |
            cp.roll(cp.roll(solid, -1, axis=0), 1, axis=1) |
            cp.roll(cp.roll(solid, -1, axis=0), -1, axis=1)
        ) & boundary1
        diag_mask2 = (
            cp.roll(cp.roll(boundary1, 1, axis=0), 1, axis=1) |
            cp.roll(cp.roll(boundary1, 1, axis=0), -1, axis=1) |
            cp.roll(cp.roll(boundary1, -1, axis=0), 1, axis=1) |
            cp.roll(cp.roll(boundary1, -1, axis=0), -1, axis=1)
        ) & boundary2

        dx_f = cp.float32(dx)
        sqrt2_dx = cp.float32(dx * np.sqrt(2.0))
        ds1 = cp.where(diag_mask1, sqrt2_dx, dx_f)
        ds2 = cp.where(diag_mask2, sqrt2_dx, dx_f)

        # 4) Muestreo en centros de cara desplazados a lo largo de n (en índices)
        JJ, II = self.JJ, self.II

        # Capa 1: 0.5 pasos (en índices; dx=dy)
        j_face1 = JJ + 0.5 * nx1
        i_face1 = II + 0.5 * ny1

        # Capa 2: 1.5 pasos
        j_face2 = JJ + 1.5 * nx2
        i_face2 = II + 1.5 * ny2

        # 5) Interpolar presión y velocidades en cada capa
        p1 = self._bilinear_interpolate(self.p, j_face1, i_face1)
        u1 = self._bilinear_interpolate(self.u, j_face1, i_face1)
        v1 = self._bilinear_interpolate(self.v, j_face1, i_face1)

        p2 = self._bilinear_interpolate(self.p, j_face2, i_face2)
        u2 = self._bilinear_interpolate(self.u, j_face2, i_face2)
        v2 = self._bilinear_interpolate(self.v, j_face2, i_face2)

        inv_dx = cp.float32(1.0 / dx)
        inv_dy = cp.float32(1.0 / dy)

        # Gradientes centrados por capa (para esfuerzo viscoso)
        def grads(u_f, v_f):
            du_dx = cp.zeros_like(u_f, dtype=cp.float32)
            du_dy = cp.zeros_like(u_f, dtype=cp.float32)
            dv_dx = cp.zeros_like(v_f, dtype=cp.float32)
            dv_dy = cp.zeros_like(v_f, dtype=cp.float32)
            du_dx[:, 1:-1] = (u_f[:, 2:] - u_f[:, :-2]) * (0.5 * inv_dx)
            du_dy[1:-1, :] = (u_f[2:, :] - u_f[:-2, :]) * (0.5 * inv_dy)
            dv_dx[:, 1:-1] = (v_f[:, 2:] - v_f[:, :-2]) * (0.5 * inv_dx)
            dv_dy[1:-1, :] = (v_f[2:, :] - v_f[:-2, :]) * (0.5 * inv_dy)
            return du_dx, du_dy, dv_dx, dv_dy

        du_dx1, du_dy1, dv_dx1, dv_dy1 = grads(u1, v1)
        du_dx2, du_dy2, dv_dx2, dv_dy2 = grads(u2, v2)

        # 6) Tracciones por capa
        # Presión
        Tx_p1 = -p1 * nx1; Ty_p1 = -p1 * ny1
        Tx_p2 = -p2 * nx2; Ty_p2 = -p2 * ny2
        # Viscosa (signo negativo: n apunta hacia fluido, queremos fuerza sobre sólido)
        sym1 = du_dy1 + dv_dx1
        sym2 = du_dy2 + dv_dx2
        Tx_v1 = -mu * (2.0 * du_dx1 * nx1 + sym1 * ny1)
        Ty_v1 = -mu * (sym1 * nx1 + 2.0 * dv_dy1 * ny1)
        Tx_v2 = -mu * (2.0 * du_dx2 * nx2 + sym2 * ny2)
        Ty_v2 = -mu * (sym2 * nx2 + 2.0 * dv_dy2 * ny2)

        # 7) Integración por capa y suma
        w1 = boundary1.astype(cp.float32)
        w2 = boundary2.astype(cp.float32)

        Fx_p = cp.sum(Tx_p1 * w1 * ds1) + cp.sum(Tx_p2 * w2 * ds2)
        Fy_p = cp.sum(Ty_p1 * w1 * ds1) + cp.sum(Ty_p2 * w2 * ds2)
        Fx_v = cp.sum(Tx_v1 * w1 * ds1) + cp.sum(Tx_v2 * w2 * ds2)
        Fy_v = cp.sum(Ty_v1 * w1 * ds1) + cp.sum(Ty_v2 * w2 * ds2)

        Fx = Fx_p + Fx_v
        Fy = Fy_p + Fy_v

        if separate_components:
            return {
                "Fx": float(Fx), "Fy": float(Fy),
                "Fx_p": float(Fx_p), "Fy_p": float(Fy_p),
                "Fx_v": float(Fx_v), "Fy_v": float(Fy_v)
            }
        else:
            return {"Fx": float(Fx), "Fy": float(Fy)}

    def compute_surface_forces_layers(self, mu, n_layers=2, step_base=0.5, step_inc=1.0, use_diagonals=True, separate_components=True):
        """
        Integra fuerzas en N capas alrededor del sólido.
        - n_layers: número de capas (>=1)
        - step_base: desplazamiento (en celdas) de la primera capa desde la frontera (ej. 0.5)
        - step_inc: incremento adicional por capa (ej. 1.0 → capa k usa step_base + (k-1)*step_inc)
        - use_diagonals: al construir anillos, usa vecinos diagonales también
        - Devuelve suma de componentes en todas las capas.
        """
        assert n_layers >= 1, "n_layers debe ser >= 1"
        dx = self.dx; dy = self.dy
        assert abs(dx - dy) < 1e-6, "Esta integración asume dx≈dy"

        # Frontera de sólido como capa base (capa 1 sin desplazamiento inicial)
        boundary = self._solid_boundary_mask(use_diagonals=use_diagonals)

        # Normales desde SDF (del sólido hacia el fluido)
        _, nx_all, ny_all = self._signed_distance_and_normals()
        JJ, II = self.JJ, self.II

        # Construir capas sucesivas dilatando hacia el fluido
        solid = self.solid
        fluid = ~solid

        def dilate(mask):
            m = (
                cp.roll(mask, 1, axis=0) |
                cp.roll(mask, -1, axis=0) |
                cp.roll(mask, 1, axis=1) |
                cp.roll(mask, -1, axis=1)
            )
            if use_diagonals:
                m |= cp.roll(cp.roll(mask, 1, axis=0), 1, axis=1)
                m |= cp.roll(cp.roll(mask, 1, axis=0), -1, axis=1)
                m |= cp.roll(cp.roll(mask, -1, axis=0), 1, axis=1)
                m |= cp.roll(cp.roll(mask, -1, axis=0), -1, axis=1)
            return m

        # Lista de máscaras de capas (todas en fluido, sin solaparse)
        layers = []
        used = cp.zeros_like(fluid, dtype=cp.bool_)
        curr = boundary.copy()

        for k in range(n_layers):
            # Capa k: fluido adyacente a la previa, excluyendo sólido y capas ya usadas
            if k == 0:
                mk = curr & fluid & (~used)
            else:
                curr = dilate(curr)
                mk = curr & fluid & (~used)
            # Excluir bordes de dominio
            mk[0, :] = False; mk[-1, :] = False; mk[:, 0] = False; mk[:, -1] = False
            layers.append(mk)
            used |= mk

        # Elemento de arco por capa: ajusta diagonales
        def arc_length_mask(base_mask, ref_mask):
            diag = (
                cp.roll(cp.roll(ref_mask, 1, axis=0), 1, axis=1) |
                cp.roll(cp.roll(ref_mask, 1, axis=0), -1, axis=1) |
                cp.roll(cp.roll(ref_mask, -1, axis=0), 1, axis=1) |
                cp.roll(cp.roll(ref_mask, -1, axis=0), -1, axis=1)
            ) & base_mask
            dx_f = cp.float32(dx)
            ds = cp.where(diag, cp.float32(dx * np.sqrt(2.0)), dx_f)
            return ds

        inv_dx = cp.float32(1.0 / dx)
        inv_dy = cp.float32(1.0 / dy)

        Fx_p = cp.float32(0.0); Fy_p = cp.float32(0.0)
        Fx_v = cp.float32(0.0); Fy_v = cp.float32(0.0)

        for k, mk in enumerate(layers, start=1):
            # Normales en la capa
            nx_k = nx_all * mk
            ny_k = ny_all * mk

            # Desplazamiento (en índices) de la cara hacia el fluido
            step_k = cp.float32(step_base + (k - 1) * step_inc)
            j_face = JJ + step_k * nx_k
            i_face = II + step_k * ny_k

            # Interpolar p, u, v en la cara desplazada
            p_k = self._bilinear_interpolate(self.p, j_face, i_face)
            u_k = self._bilinear_interpolate(self.u, j_face, i_face)
            v_k = self._bilinear_interpolate(self.v, j_face, i_face)

            # Gradientes centrados locales para viscoso
            du_dx = cp.zeros_like(u_k, dtype=cp.float32)
            du_dy = cp.zeros_like(u_k, dtype=cp.float32)
            dv_dx = cp.zeros_like(v_k, dtype=cp.float32)
            dv_dy = cp.zeros_like(v_k, dtype=cp.float32)
            du_dx[:, 1:-1] = (u_k[:, 2:] - u_k[:, :-2]) * (0.5 * inv_dx)
            du_dy[1:-1, :] = (u_k[2:, :] - u_k[:-2, :]) * (0.5 * inv_dy)
            dv_dx[:, 1:-1] = (v_k[:, 2:] - v_k[:, :-2]) * (0.5 * inv_dx)
            dv_dy[1:-1, :] = (v_k[2:, :] - v_k[:-2, :]) * (0.5 * inv_dy)

            sym_xy = du_dy + dv_dx

            # Tracciones
            Tx_p = -p_k * nx_k
            Ty_p = -p_k * ny_k
            Tx_v = -mu * (2.0 * du_dx * nx_k + sym_xy * ny_k)
            Ty_v = -mu * (sym_xy * nx_k + 2.0 * dv_dy * ny_k)

            # Longitud elemental de arco (corrige diagonales usando la referencia de la capa previa)
            ref_mask = layers[k-2] if k > 1 else self.solid  # para la capa 1 referenciamos el sólido
            ds_k = arc_length_mask(mk, ref_mask)

            w = mk.astype(cp.float32)

            Fx_p += cp.sum(Tx_p * w * ds_k)
            Fy_p += cp.sum(Ty_p * w * ds_k)
            Fx_v += cp.sum(Tx_v * w * ds_k)
            Fy_v += cp.sum(Ty_v * w * ds_k)

        Fx = Fx_p + Fx_v
        Fy = Fy_p + Fy_v

        if separate_components:
            return {
                "Fx": float(Fx), "Fy": float(Fy),
                "Fx_p": float(Fx_p), "Fy_p": float(Fy_p),
                "Fx_v": float(Fx_v), "Fy_v": float(Fy_v)
            }
        else:
            return {"Fx": float(Fx), "Fy": float(Fy)}


    def compute_drag_lift_layers(self, mu, n_layers=2, step_base=0.5, step_inc=1.0):
        """
        Calcula Drag y Lift usando múltiples capas de integración.
        Transforma de coordenadas del cuerpo a sistema aerodinámico.
        """
        res = self.compute_surface_forces_layers(mu, n_layers=n_layers, step_base=step_base, step_inc=step_inc, separate_components=True)
        
        # Descomponer en ejes viento usando ángulo real del flujo libre
        alpha_rad = self._get_freestream_angle_rad()
        cos_a = np.cos(alpha_rad)
        sin_a = np.sin(alpha_rad)
        
        Drag = res["Fx"] * cos_a + res["Fy"] * sin_a
        Lift = -res["Fx"] * sin_a + res["Fy"] * cos_a
        Drag_p = res["Fx_p"] * cos_a + res["Fy_p"] * sin_a
        Lift_p = -res["Fx_p"] * sin_a + res["Fy_p"] * cos_a
        Drag_v = res["Fx_v"] * cos_a + res["Fy_v"] * sin_a
        Lift_v = -res["Fx_v"] * sin_a + res["Fy_v"] * cos_a
        
        return {
            "Drag": Drag, "Lift": Lift,
            "Drag_p": Drag_p, "Lift_p": Lift_p,
            "Drag_v": Drag_v, "Lift_v": Lift_v
        }

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


    def save_state(self, filepath):
        """
        Guarda el estado completo de la malla para poder reanudar la simulación.
        
        Parámetros:
            filepath: Ruta donde guardar el archivo (se añadirá extensión .npz)
        """
        if not filepath.endswith('.npz'):
            filepath += '.npz'
        
        # Convertir arrays de GPU a CPU
        state = {
            # Campos de flujo
            'u': cp.asnumpy(self.u),
            'v': cp.asnumpy(self.v),
            'p': cp.asnumpy(self.p),
            
            # Vectores de históricos
            'dvector': cp.asnumpy(self.dvector),
            'lvector': cp.asnumpy(self.lvector),
            'cdvector': cp.asnumpy(self.cdvector),
            'clcdvector': cp.asnumpy(self.clcdvector),
            'clvector': cp.asnumpy(self.clvector),
            'divvector': cp.asnumpy(self.divvector),
            
            # Máscara de sólido
            'solid': cp.asnumpy(self.solid),
            
            # Parámetros de la malla
            'Lx': self.Lx,
            'Ly': self.Ly,
            'dx': self.dx,
            'dy': self.dy,
            'nx': self.nx,
            'ny': self.ny,
            
            # Ángulo de geometría
            'alpha_geometry': getattr(self, 'alpha_geometry', 0.0),
        }
        
        np.savez_compressed(filepath, **state)
        print(f"✓ Estado guardado en: {filepath}")
    
    def load_state(self, filepath, allow_geometry_change=False):
        """
        Carga el estado completo de la malla desde un archivo.
        
        Parámetros:
            filepath: Ruta del archivo a cargar
            allow_geometry_change: Si True, permite cargar campos de flujo incluso si 
                                   la geometría del sólido es diferente. Útil para usar
                                   un estado convergido como base para nuevas geometrías.
        """
        if not filepath.endswith('.npz'):
            filepath += '.npz'

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
        
        data = np.load(filepath)
        
        # Verificar compatibilidad de dimensiones
        if data['nx'] != self.nx or data['ny'] != self.ny:
            raise ValueError(f"Dimensiones incompatibles: archivo tiene {data['nx']}x{data['ny']}, "
                           f"pero la malla actual es {self.nx}x{self.ny}")
        
        # Restaurar campos de flujo
        u_cargado = cp.array(data['u'], dtype=cp.float32)
        v_cargado = cp.array(data['v'], dtype=cp.float32)
        p_cargado = cp.array(data['p'], dtype=cp.float32)
        
        # Restaurar vectores de históricos
        self.dvector = cp.array(data['dvector'], dtype=cp.float32)
        self.lvector = cp.array(data['lvector'], dtype=cp.float32)
        self.cdvector = cp.array(data['cdvector'], dtype=cp.float32)
        # Restaurar también el vector Cl/Cd si existe
        if 'clcdvector' in data:
            self.clcdvector = cp.array(data['clcdvector'], dtype=cp.float32)
        else:
            self.clcdvector = cp.zeros_like(self.cdvector)
        self.clvector = cp.array(data['clvector'], dtype=cp.float32)
        self.divvector = cp.array(data['divvector'], dtype=cp.float32)
        
        # Restaurar ángulo de geometría si existe
        if 'alpha_geometry' in data:
            self.alpha_geometry = float(data['alpha_geometry'])
        
        if allow_geometry_change:
            # Modo compatible con cambio de geometría
            solid_cargado = cp.array(data['solid'], dtype=cp.bool_)
            
            # Comparar geometrías
            cambios_geometria = cp.sum(self.solid != solid_cargado)
            porcentaje_cambio = 100 * float(cambios_geometria) / (self.nx * self.ny)
            
            print(f"⚠ Cambio de geometría detectado: {cambios_geometria} celdas ({porcentaje_cambio:.2f}%)")
            
            # Aplicar campos de flujo solo donde NO hay sólido en la nueva geometría
            # En zonas con nuevo sólido, mantener las condiciones iniciales (ya establecidas)
            self.u = cp.where(~self.solid, u_cargado, self.u)
            self.v = cp.where(~self.solid, v_cargado, self.v)
            self.p = cp.where(~self.solid, p_cargado, self.p)
            
            # En zonas donde había sólido pero ya no, usar interpolación de vecinos
            zona_nuevo_fluido = (~self.solid) & solid_cargado
            if cp.any(zona_nuevo_fluido):
                n_nuevas = int(cp.sum(zona_nuevo_fluido))
                print(f"  → Inicializando {n_nuevas} celdas nuevas de fluido por interpolación")
                
                # Interpolación simple: promedio de vecinos válidos
                for field, field_cargado in [(self.u, u_cargado), (self.v, v_cargado), (self.p, p_cargado)]:
                    # Crear copia con valores interpolados
                    field_interp = field.copy()
                    
                    # Promedio de vecinos (arriba, abajo, izq, der)
                    vecinos = (
                        cp.roll(field_cargado, 1, axis=0) +
                        cp.roll(field_cargado, -1, axis=0) +
                        cp.roll(field_cargado, 1, axis=1) +
                        cp.roll(field_cargado, -1, axis=1)
                    ) / 4.0
                    
                    # Aplicar solo en zona_nuevo_fluido
                    if field is self.u:
                        self.u = cp.where(zona_nuevo_fluido, vecinos, self.u)
                    elif field is self.v:
                        self.v = cp.where(zona_nuevo_fluido, vecinos, self.v)
                    else:
                        self.p = cp.where(zona_nuevo_fluido, vecinos, self.p)
            
            print(f"✓ Estado adaptado a nueva geometría")
        else:
            # Modo estricto: restaurar tal cual (incluyendo sólido)
            self.u = u_cargado
            self.v = v_cargado
            self.p = p_cargado
            self.solid = cp.array(data['solid'], dtype=cp.bool_)
        
        print(f"✓ Estado cargado desde: {filepath}")
        print(f"  Históricos: {len(self.cdvector)} iteraciones guardadas")


def save_complete_checkpoint(filepath, mesh_fina, mesh_gruesa, metadata):
    """
    Guarda un checkpoint completo con ambas mallas y metadata en un solo archivo.
    
    Parámetros:
        filepath: Ruta del archivo .npz a crear
        mesh_fina: Instancia de Mesh con malla fina
        mesh_gruesa: Instancia de Mesh con malla gruesa
        metadata: Diccionario con información adicional (iteración, tiempo, etc.)
    """
    if not filepath.endswith('.npz'):
        filepath += '.npz'
    
    # Preparar datos de malla fina (prefijo 'fina_')
    state = {
        # Malla fina
        'fina_u': cp.asnumpy(mesh_fina.u),
        'fina_v': cp.asnumpy(mesh_fina.v),
        'fina_p': cp.asnumpy(mesh_fina.p),
        'fina_dvector': cp.asnumpy(mesh_fina.dvector),
        'fina_lvector': cp.asnumpy(mesh_fina.lvector),
        'fina_cdvector': cp.asnumpy(mesh_fina.cdvector),
        'fina_clcdvector': cp.asnumpy(mesh_fina.clcdvector),
        'fina_clvector': cp.asnumpy(mesh_fina.clvector),
        'fina_divvector': cp.asnumpy(mesh_fina.divvector),
        'fina_solid': cp.asnumpy(mesh_fina.solid),
        'fina_Lx': mesh_fina.Lx,
        'fina_Ly': mesh_fina.Ly,
        'fina_dx': mesh_fina.dx,
        'fina_dy': mesh_fina.dy,
        'fina_nx': mesh_fina.nx,
        'fina_ny': mesh_fina.ny,
        
        # Malla gruesa
        'gruesa_u': cp.asnumpy(mesh_gruesa.u),
        'gruesa_v': cp.asnumpy(mesh_gruesa.v),
        'gruesa_p': cp.asnumpy(mesh_gruesa.p),
        'gruesa_dvector': cp.asnumpy(mesh_gruesa.dvector),
        'gruesa_lvector': cp.asnumpy(mesh_gruesa.lvector),
        'gruesa_cdvector': cp.asnumpy(mesh_gruesa.cdvector),
        'gruesa_clcdvector': cp.asnumpy(getattr(mesh_gruesa, 'clcdvector', cp.zeros_like(mesh_gruesa.cdvector))),
        'gruesa_clvector': cp.asnumpy(mesh_gruesa.clvector),
        'gruesa_divvector': cp.asnumpy(mesh_gruesa.divvector),
        'gruesa_solid': cp.asnumpy(mesh_gruesa.solid),
        'gruesa_Lx': mesh_gruesa.Lx,
        'gruesa_Ly': mesh_gruesa.Ly,
        'gruesa_dx': mesh_gruesa.dx,
        'gruesa_dy': mesh_gruesa.dy,
        'gruesa_nx': mesh_gruesa.nx,
        'gruesa_ny': mesh_gruesa.ny,
    }
    
    # Agregar metadata (valores escalares y dict serializado)
    # Guardar también parámetros críticos para reanudación consistente
    metadata_expandida = dict(metadata) if metadata is not None else {}
    metadata_expandida.setdefault('dx_fino', float(mesh_fina.dx))
    metadata_expandida.setdefault('dy_fino', float(mesh_fina.dy))
    metadata_expandida.setdefault('dx_grueso', float(mesh_gruesa.dx))
    metadata_expandida.setdefault('dy_grueso', float(mesh_gruesa.dy))
    metadata_expandida.setdefault('Lx', float(mesh_gruesa.Lx))
    metadata_expandida.setdefault('Ly', float(mesh_gruesa.Ly))

    # Serializar metadata completa para conservar estructuras (tuplas/dicts)
    state['meta_json'] = json.dumps(metadata_expandida, ensure_ascii=False)

    # Guardar también claves escalares como meta_* (útil para lectura rápida)
    for key, value in metadata_expandida.items():
        if isinstance(value, (int, float, np.integer, np.floating, bool)):
            state[f'meta_{key}'] = value
    
    np.savez_compressed(filepath, **state)
    print(f"✓ Checkpoint completo guardado en: {filepath}")


def load_complete_checkpoint(filepath, mesh_fina, mesh_gruesa, allow_geometry_change=False):
    """
    Carga un checkpoint completo desde un solo archivo.
    
    Parámetros:
        filepath: Ruta del archivo .npz a cargar
        mesh_fina: Instancia de Mesh donde cargar malla fina
        mesh_gruesa: Instancia de Mesh donde cargar malla gruesa
        allow_geometry_change: Permitir cambios de geometría
    
    Retorna:
        dict: Metadata del checkpoint (iteración, tiempo, etc.)
    """
    if not filepath.endswith('.npz'):
        filepath += '.npz'

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
    
    data = np.load(filepath)
    
    # Verificar dimensiones de malla fina
    if data['fina_nx'] != mesh_fina.nx or data['fina_ny'] != mesh_fina.ny:
        raise ValueError(f"Dimensiones de malla fina incompatibles: archivo tiene {data['fina_nx']}x{data['fina_ny']}, "
                       f"pero la malla actual es {mesh_fina.nx}x{mesh_fina.ny}")
    
    # Verificar dimensiones de malla gruesa
    if data['gruesa_nx'] != mesh_gruesa.nx or data['gruesa_ny'] != mesh_gruesa.ny:
        raise ValueError(f"Dimensiones de malla gruesa incompatibles: archivo tiene {data['gruesa_nx']}x{data['gruesa_ny']}, "
                       f"pero la malla actual es {mesh_gruesa.nx}x{mesh_gruesa.ny}")

    # Verificar parámetros de malla (dx/dy/Lx/Ly) para evitar inconsistencias silenciosas
    dx_fino_ckpt = float(data['fina_dx'])
    dy_fino_ckpt = float(data['fina_dy'])
    dx_grueso_ckpt = float(data['gruesa_dx'])
    dy_grueso_ckpt = float(data['gruesa_dy'])

    if abs(float(mesh_fina.dx) - dx_fino_ckpt) > 1e-12 or abs(float(mesh_fina.dy) - dy_fino_ckpt) > 1e-12:
        raise ValueError(
            f"dx/dy de malla fina incompatibles: checkpoint tiene dx={dx_fino_ckpt}, dy={dy_fino_ckpt}, "
            f"pero la malla actual tiene dx={mesh_fina.dx}, dy={mesh_fina.dy}. "
            f"Solución: crea la simulación con esos dx/dy o activa usar_parametros_checkpoint=True."
        )

    if abs(float(mesh_gruesa.dx) - dx_grueso_ckpt) > 1e-12 or abs(float(mesh_gruesa.dy) - dy_grueso_ckpt) > 1e-12:
        raise ValueError(
            f"dx/dy de malla gruesa incompatibles: checkpoint tiene dx={dx_grueso_ckpt}, dy={dy_grueso_ckpt}, "
            f"pero la malla actual tiene dx={mesh_gruesa.dx}, dy={mesh_gruesa.dy}. "
            f"Solución: crea la simulación con esos dx/dy o activa usar_parametros_checkpoint=True."
        )
    
    # Cargar malla fina
    # Nota: np.load devuelve escalares como numpy.ndarray 0-D; convertir a float evita errores con CuPy
    mesh_fina.Lx = float(data['fina_Lx'])
    mesh_fina.Ly = float(data['fina_Ly'])
    mesh_fina.dx = float(data['fina_dx'])
    mesh_fina.dy = float(data['fina_dy'])

    u_fina = cp.array(data['fina_u'], dtype=cp.float32)
    v_fina = cp.array(data['fina_v'], dtype=cp.float32)
    p_fina = cp.array(data['fina_p'], dtype=cp.float32)
    
    mesh_fina.dvector = cp.array(data['fina_dvector'], dtype=cp.float32)
    mesh_fina.lvector = cp.array(data['fina_lvector'], dtype=cp.float32)
    mesh_fina.cdvector = cp.array(data['fina_cdvector'], dtype=cp.float32)
    # Restaurar también cl/cd ratio si está en el checkpoint
    if 'fina_clcdvector' in data:
        mesh_fina.clcdvector = cp.array(data['fina_clcdvector'], dtype=cp.float32)
    else:
        mesh_fina.clcdvector = cp.zeros_like(mesh_fina.cdvector)
    mesh_fina.clvector = cp.array(data['fina_clvector'], dtype=cp.float32)
    mesh_fina.divvector = cp.array(data['fina_divvector'], dtype=cp.float32)

    # Cargar malla gruesa
    mesh_gruesa.Lx = float(data['gruesa_Lx'])
    mesh_gruesa.Ly = float(data['gruesa_Ly'])
    mesh_gruesa.dx = float(data['gruesa_dx'])
    mesh_gruesa.dy = float(data['gruesa_dy'])

    u_gruesa = cp.array(data['gruesa_u'], dtype=cp.float32)
    v_gruesa = cp.array(data['gruesa_v'], dtype=cp.float32)
    p_gruesa = cp.array(data['gruesa_p'], dtype=cp.float32)
    
    mesh_gruesa.dvector = cp.array(data['gruesa_dvector'], dtype=cp.float32)
    mesh_gruesa.lvector = cp.array(data['gruesa_lvector'], dtype=cp.float32)
    mesh_gruesa.cdvector = cp.array(data['gruesa_cdvector'], dtype=cp.float32)
    if 'gruesa_clcdvector' in data:
        mesh_gruesa.clcdvector = cp.array(data['gruesa_clcdvector'], dtype=cp.float32)
    else:
        mesh_gruesa.clcdvector = cp.zeros_like(mesh_gruesa.cdvector)
    mesh_gruesa.clvector = cp.array(data['gruesa_clvector'], dtype=cp.float32)
    mesh_gruesa.divvector = cp.array(data['gruesa_divvector'], dtype=cp.float32)
    
    # Manejar geometría
    if allow_geometry_change:
        # Malla fina
        solid_fina_cargado = cp.array(data['fina_solid'], dtype=cp.bool_)
        cambios_fina = cp.sum(mesh_fina.solid != solid_fina_cargado)
        porcentaje_fina = 100 * float(cambios_fina) / (mesh_fina.nx * mesh_fina.ny)
        
        if cambios_fina > 0:
            print(f"⚠ Cambio de geometría en malla fina: {cambios_fina} celdas ({porcentaje_fina:.2f}%)")
            mesh_fina.u = cp.where(~mesh_fina.solid, u_fina, mesh_fina.u)
            mesh_fina.v = cp.where(~mesh_fina.solid, v_fina, mesh_fina.v)
            mesh_fina.p = cp.where(~mesh_fina.solid, p_fina, mesh_fina.p)
        else:
            mesh_fina.u = u_fina
            mesh_fina.v = v_fina
            mesh_fina.p = p_fina
        
        # Malla gruesa
        solid_gruesa_cargado = cp.array(data['gruesa_solid'], dtype=cp.bool_)
        cambios_gruesa = cp.sum(mesh_gruesa.solid != solid_gruesa_cargado)
        porcentaje_gruesa = 100 * float(cambios_gruesa) / (mesh_gruesa.nx * mesh_gruesa.ny)
        
        if cambios_gruesa > 0:
            print(f"⚠ Cambio de geometría en malla gruesa: {cambios_gruesa} celdas ({porcentaje_gruesa:.2f}%)")
            mesh_gruesa.u = cp.where(~mesh_gruesa.solid, u_gruesa, mesh_gruesa.u)
            mesh_gruesa.v = cp.where(~mesh_gruesa.solid, v_gruesa, mesh_gruesa.v)
            mesh_gruesa.p = cp.where(~mesh_gruesa.solid, p_gruesa, mesh_gruesa.p)
        else:
            mesh_gruesa.u = u_gruesa
            mesh_gruesa.v = v_gruesa
            mesh_gruesa.p = p_gruesa
    else:
        # Modo estricto
        mesh_fina.u = u_fina
        mesh_fina.v = v_fina
        mesh_fina.p = p_fina
        mesh_fina.solid = cp.array(data['fina_solid'], dtype=cp.bool_)
        
        mesh_gruesa.u = u_gruesa
        mesh_gruesa.v = v_gruesa
        mesh_gruesa.p = p_gruesa
        mesh_gruesa.solid = cp.array(data['gruesa_solid'], dtype=cp.bool_)
    
    # Extraer metadata
    metadata = {}

    # JSON completo (si existe)
    if 'meta_json' in data.files:
        try:
            meta_texto = data['meta_json'].item() if hasattr(data['meta_json'], 'item') else str(data['meta_json'])
            metadata.update(json.loads(meta_texto))
        except Exception:
            pass

    # Claves meta_* (sobrescriben si aplica)
    for key in data.files:
        if key.startswith('meta_'):
            try:
                metadata[key[5:]] = data[key].item() if hasattr(data[key], 'ndim') and data[key].ndim == 0 else data[key]
            except Exception:
                metadata[key[5:]] = data[key]
    
    print(f"✓ Checkpoint completo cargado desde: {filepath}")
    print(f"  Históricos: {len(mesh_fina.cdvector)} iteraciones guardadas")
    
    return metadata


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
    T=0.2,
    
    # Geometría del perfil
    alpha_deg=5,
    chord=1.0,
    filepath="NACA_0012",
    
    # Tamaño del dominio
    Lx=7,
    Ly=6,
    
    # Resolución de mallas
    dx_grueso=0.01,   
    dx_fino=1,
    dy_grueso=None,  # Si None, se usa dx_grueso
    dy_fino=None,    # Si None, se usa dx_fino
    usar_wale=False,  # Si True, activa modelo de turbulencia WALE
    usar_viscosidad_estela=False,  # Si True, añade viscosidad extra en la estela (surrogate 3D)
    C_estela=0.05,  # Coeficiente de viscosidad de estela (0.01-0.1 típico)
    # Posición del perfil
    cx=2,
    cy=None,  # Si None, se centra verticalmente
    
    # Zona de refinamiento
    ancho_ref_factor=3,  # Factor multiplicador del chord
    alto_ref_factor=1,   # Factor multiplicador del chord
    offset_refinado=0.5,   # Offset del centro refinado respecto al leading edge (en fracciones de chord)
    
    # Condiciones iniciales
    p0=0,  # Pa
    v0x=5,  # m/s
    v0y=0.0,  # m/s
    
    # Propiedades del fluido
    rho=1.225,#1.225,  # kg/m^3 (aire a nivel del mar)
    nu=1.5e-5,#1.5e-5,  # m^2/s (viscosidad cinemática del aire)
    divergencia=1e-1,  # Relajado: nested mesh no puede alcanzar divergencia muy baja
    # Condiciones de frontera
    boundary_left=("inflow", None),  # (tipo, valor) - si valor es None, usa (v0x, v0y)
    boundary_top=("slip", None),
    boundary_bottom=("slip", None),
    boundary_right=("outflow", None),  # si valor es None, usa p0 (presión fija en salida)
    
    # Parámetros numéricos
    tol_poisson=1,
    max_iter_poisson=500,
    guardado=50,  # guardar cada N iteraciones
    iteraciones=2000,
    # Parámetros de cálculo de fuerzas
    n_layers=1,
    step_base=0.0,
    step_inc=0.1,
    
    # Opciones de visualización y guardado
    save_frames=False,
    frames_dir_fino=None,
    frames_dir_grueso=None,
    visualize_mesh=False,
    show_weights=True,
    graficos=False,
    
    # Opciones de checkpoint (guardar/reanudar)
    checkpoint_file=None,  # Si se especifica, intenta cargar desde este archivo
    save_checkpoint_every=None,  # Guardar checkpoint cada N iteraciones (None = no guardar)
    checkpoint_dir="checkpoints",  # Directorio donde guardar checkpoints
    allow_geometry_change=False,  # Permitir cargar checkpoint con geometría diferente
    usar_parametros_checkpoint=True,  # Si True, al cargar se fuerzan dx/dy/Lx/Ly/dt/guardado del checkpoint
    
    # Control de convergencia
    stop_on_convergence=True,  # Si True, detiene la simulación al alcanzar estado estacionario
    
    # Plan polar automático: lista de (alpha_grados, n_iteraciones)
    # Ej: [(0,3000),(2,2000),(4,2000),(6,2000),(8,2000)]
    # Si None, usa iteraciones normales sin cambio automático de alpha
    plan_polar=None,
    # Fracción del inicio de cada tramo a descartar para la media (transitorio)
    polar_descarte=0.3,
    
    # Vista en tiempo real (proceso externo)
    live_view=False  # Si True, publica datos en memoria compartida para viewer_live.py
):
    # Procesar valores por defecto
    if dy_fino is None:
        dy_fino = dx_fino
    if dy_grueso is None:
        dy_grueso = dx_grueso
    if cy is None:
        cy = Ly / 2.0
    
    ancho_ref = chord * ancho_ref_factor
    alto_ref = chord * alto_ref_factor
    
    # Calcular viscosidad dinámica
    mu = rho * nu

    # ============================================================
    # PRE-LOAD (opcional): si hay checkpoint, usar sus parámetros para evitar dx/dy inconsistentes
    # ============================================================
    checkpoint_meta = None
    if checkpoint_file and os.path.exists(checkpoint_file) and usar_parametros_checkpoint:
        try:
            _ckpt = np.load(checkpoint_file)
            dx_fino = float(_ckpt['fina_dx'])
            dy_fino = float(_ckpt['fina_dy'])
            dx_grueso = float(_ckpt['gruesa_dx'])
            dy_grueso = float(_ckpt['gruesa_dy'])
            Lx = float(_ckpt['gruesa_Lx'])
            Ly = float(_ckpt['gruesa_Ly'])
            if 'meta_json' in _ckpt.files:
                try:
                    meta_texto = _ckpt['meta_json'].item() if hasattr(_ckpt['meta_json'], 'item') else str(_ckpt['meta_json'])
                    checkpoint_meta = json.loads(meta_texto)
                except Exception:
                    checkpoint_meta = None
            print("\n" + "="*70)
            print("CHECKPOINT: usando parámetros guardados")
            print("="*70)
            print(f"  dx_fino={dx_fino}, dy_fino={dy_fino}")
            print(f"  dx_grueso={dx_grueso}, dy_grueso={dy_grueso}")
            print(f"  Lx={Lx}, Ly={Ly}")
        except Exception as e:
            print(f"⚠ No se pudieron leer parámetros del checkpoint: {e}")
            checkpoint_meta = None

    # Crear geometría de mallas
    geometria = mmr.MallaMultiRes(
        Lx=Lx, Ly=Ly,
        dx_grueso=dx_grueso, dx_fino=dx_fino,
        centro_refinado=(cx + chord * offset_refinado, cy),
        ancho_refinado=ancho_ref, alto_refinado=alto_ref, tipo_zona="rectangular"
    )
    
    # Crear mallas física (gruesa y fina)
    mesh_gruesa = Mesh(Lx, Ly, p0, v0x, v0y, dx_grueso, dy_grueso, usar_wale,
                       usar_viscosidad_estela=usar_viscosidad_estela, C_estela=C_estela)

    limites = geometria.get_limites_fino()
    mesh_fina = Mesh(limites['Lx'], limites['Ly'], p0, v0x, v0y, dx_fino, dy_fino)

    
    # Ángulo de geometría: si hay plan_polar, cargar a 0° (el flujo se rota)
    # Si no hay plan_polar, cargar al alpha_deg solicitado (flujo horizontal)
    alpha_geom = 0.0 if (plan_polar is not None and len(plan_polar) > 0) else alpha_deg
    
    # Añadir sólido a malla gruesa
    mesh_gruesa.load_solids_from_file(
        filepath=filepath,
        chord=chord,
        x_offset=cx, y_offset=cy,
        alpha_deg=alpha_geom, fill=True, plot=False
    )
    
    #mesh_gruesa.add_solid_circle(cx,cy,0.5*chord)  # Para probar sólido circular (descomentar)

    # Añadir sólido a malla fina (coordenadas trasladadas)
    cx_fino, cy_fino = geometria.offset_solido_para_fino(cx, cy)
    mesh_fina.load_solids_from_file(
        filepath=filepath,
        chord=chord,
        x_offset=cx_fino, y_offset=cy_fino,
        alpha_deg=alpha_geom, fill=True, plot=False
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

    # La malla fina NO debe tener boundaries configuradas porque es embebida
    # Recibe todos sus valores de borde por interpolación de la malla gruesa
    # Si se configuran boundaries, apply_boundaries() sobreescribe los valores interpolados
    # mesh_fina.set_boundary("top", boundary_type_top, value=boundary_val_top)
    # mesh_fina.set_boundary("bottom", boundary_type_bottom, value=boundary_val_bottom)
    
    # Visualizar mallas si se solicita
    if visualize_mesh:
        geometria.visualizar_mallado(mostrar_pesos=show_weights)
    
    # Parámetros físicos (ya calculados arriba)
    # rho, nu, mu ya están definidos
    
    tol_poisson = tol_poisson
    CFL = CFL
    dt = CFL * min(dx_grueso,dy_grueso) / np.sqrt(v0x**2 + v0y**2)
    print(f"dt = {dt:.6f} s  (CFL={CFL})")  
    # Si el checkpoint trae dt/guardado, podemos forzar consistencia (para reanudar sin saltos)
    if checkpoint_meta is not None and usar_parametros_checkpoint:
        if 'guardado' in checkpoint_meta:
            guardado_ckpt = int(checkpoint_meta['guardado'])
            if guardado_ckpt != guardado:
                print(f"⚠ guardado cambiado por checkpoint: {guardado} → {guardado_ckpt}")
                guardado = guardado_ckpt
        if 'dt' in checkpoint_meta:
            dt_ckpt = float(checkpoint_meta['dt'])
            if abs(dt_ckpt - dt) > 1e-12:
                print(f"⚠ dt cambiado por checkpoint: {dt} → {dt_ckpt}")
                dt = dt_ckpt

    guardado = guardado  # guardar cada N iteraciones
    iteraciones = iteraciones
    if iteraciones % guardado != 0:
        iteraciones += guardado - (iteraciones % guardado)
    # Inicializar vectores de resultados para ambas mallas
    mesh_gruesa.guardado = guardado
    mesh_gruesa.cdvector = cp.zeros(iteraciones // guardado, dtype=cp.float32)
    mesh_gruesa.clvector = cp.zeros(iteraciones // guardado, dtype=cp.float32)
    mesh_gruesa.divvector = cp.zeros(iteraciones // guardado, dtype=cp.float32)
    mesh_gruesa.clcdvector = cp.zeros(iteraciones // guardado, dtype=cp.float32)
    mesh_gruesa.mg_cycles_vector = cp.zeros(iteraciones, dtype=cp.int32)  # Cada iteración
    
    mesh_fina.guardado = guardado
    mesh_fina.cdvector = cp.zeros(iteraciones // guardado, dtype=cp.float32)
    mesh_fina.clvector = cp.zeros(iteraciones // guardado, dtype=cp.float32)
    mesh_fina.divvector = cp.zeros(iteraciones // guardado, dtype=cp.float32)
    mesh_fina.clcdvector = cp.zeros(iteraciones // guardado, dtype=cp.float32)
    mesh_fina.mg_cycles_vector = cp.zeros(iteraciones, dtype=cp.int32)

    # Estadísticas
    total_gruesa = mesh_gruesa.nx * mesh_gruesa.ny
    activas_gruesa = int(cp.sum(geometria.mascara_gruesa_activa))
    total_fina = mesh_fina.nx * mesh_fina.ny
    borde_fino = int(cp.sum(geometria.mascara_fina_borde))
    celdas_uniforme = int(Lx * Ly / (dx_fino ** 2))
    celdas_multires = total_gruesa + total_fina
    print(f"Malla gruesa: {mesh_gruesa.nx} x {mesh_gruesa.ny}  ({activas_gruesa:,} celdas activas)")

    # ============================================================
    # CHECKPOINT: Cargar estado previo si existe
    # ============================================================
    iteracion_inicial = 0
    tiempo_simulado_previo = 0.0
    
    if checkpoint_file and os.path.exists(checkpoint_file):
        print("\n" + "="*70)
        print("CARGANDO CHECKPOINT")
        print("="*70)
        try:
            # Cargar checkpoint completo (ambas mallas + metadata en un solo archivo)
            metadata = load_complete_checkpoint(
                checkpoint_file,
                mesh_fina,
                mesh_gruesa,
                allow_geometry_change=allow_geometry_change
            )
            
            # Extraer información de la metadata
            if 'iteracion' in metadata:
                iteracion_inicial = int(metadata['iteracion'])
            if 'tiempo_simulado' in metadata:
                tiempo_simulado_previo = float(metadata['tiempo_simulado'])
            
            print(f"  Reanudando desde iteración: {iteracion_inicial}")
            print(f"  Tiempo simulado previo: {tiempo_simulado_previo:.4f} s")
            
            # Mostrar info de geometría si hubo cambios
            if allow_geometry_change and 'alpha_deg' in metadata:
                alpha_prev = float(metadata['alpha_deg'])
                if abs(alpha_prev - alpha_deg) > 0.01:
                    print(f"  ⚠ Cambio de ángulo: {alpha_prev:.1f}° → {alpha_deg:.1f}°")
            
            print("✓ Checkpoint cargado exitosamente")
        except Exception as e:
            print(f"⚠ Error al cargar checkpoint: {e}")
            print("  Iniciando simulación desde cero")
            iteracion_inicial = 0
            tiempo_simulado_previo = 0.0
    
    # Crear directorio de checkpoints si se requiere guardar
    if save_checkpoint_every:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"\n✓ Checkpoints se guardarán cada {save_checkpoint_every} iteraciones en: {checkpoint_dir}")

################################################################################################################################################################################################################################################################################################################################################################################################################################

    # ============================================================
    # INSTRUMENTACIÓN DE TIMING (medición de rendimiento)
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
    
    # ⭐ Variable para acumular tiempo físico simulado (transitorio)
    tiempo_fisico_acumulado = tiempo_simulado_previo
    
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
    iter_inicio_alpha = iteracion_inicial  # Iteración donde empezó el alpha actual
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
        acum = iteracion_inicial
        for idx_plan, (alpha_plan, n_iter_plan) in enumerate(plan_polar):
            cambios_alpha_programados[acum] = alpha_plan
            acum += n_iter_plan
        
        print("\n" + "="*70)
        print("📅 PLAN POLAR AUTOMÁTICO:")
        print("="*70)
        iter_acum = iteracion_inicial
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
    vel_clamp_max = 10.0 * max(U_ref, 1.0)
    
    # ============================================================
    # BUCLE PRINCIPAL: MALLA SIMPLE (SOLO GRUESA)
    # ============================================================
    for it in tqdm(range(iteracion_inicial, iteraciones)):
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
            dt_adv = float(CFL * min(dx_grueso, dy_grueso) / max(Umax, 1e-12))
            
            # dt viscoso (estabilidad difusiva)
            if mesh_gruesa.usar_wale:
                nu_t_g = mesh_gruesa.compute_wale_viscosity()
                
                # ⭐ LIMITAR nu_t para evitar valores excesivos que reduzcan dt demasiado
                nu_t_max_permitido = 100.0 * nu
                nu_t_g = cp.minimum(nu_t_g, cp.float32(nu_t_max_permitido))
                
                nu_eff_max_cp = cp.max(cp.float32(nu) + nu_t_g)
                nu_eff_max = float(nu_eff_max_cp) if float(nu_eff_max_cp) > 0 else float(nu)
            else:
                nu_eff_max = float(nu)
            
            # Incluir viscosidad de estela en cálculo de dt
            if mesh_gruesa.usar_viscosidad_estela:
                nu_wake_g = mesh_gruesa.compute_wake_viscosity()
                nu_eff_con_estela = cp.float32(nu_eff_max) + cp.max(nu_wake_g)
                nu_eff_max = float(nu_eff_con_estela)
            
            # ⚠️ Advertencia si nu_eff es anormalmente alto
            if nu_eff_max > 10.0 * nu and it % (guardado * 10) == 0:
                ratio_nu = nu_eff_max / nu
                print(f"\n⚠️  [Iter {it}] nu_efectiva muy alta: {nu_eff_max:.2e} ({ratio_nu:.1f}× nu molecular)")
                print(f"    Esto puede reducir dt significativamente")
            
            C_visc = 0.25
            if nu_eff_max > 1e-12:
                dt_visc = float(C_visc * (min(dx_grueso, dy_grueso)**2) / nu_eff_max)
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
        
        #dt_use = dt  # Usar dt fijo en este bucle simple
        # Advección
        t0 = time.time()
        mesh_gruesa.advect_velocities(dt_use)
        mesh_gruesa.apply_boundaries(after_projection=False)
        timing_stats['adveccion'] += time.time() - t0
        
        # Difusión
        t0 = time.time()
        mesh_gruesa.diffuse_velocity(nu, dt_use, usar_wale=mesh_gruesa.usar_wale)
        mesh_gruesa.apply_boundaries(after_projection=False)
        timing_stats['difusion'] += time.time() - t0
        
        # Proyección
        t0 = time.time()
        
        #mesh_gruesa.project2_adaptive(rho,dt_use,tol_div=divergencia,max_iter=200,verbose=False)
        '''
        mesh_gruesa.project_cg(
            rho, dt_use,
            tol_div=divergencia,
            max_iter=3000,
            verbose=False,
            usar_operador_spd=True,
            modo_adaptativo=True,
            detectar_estancamiento=True
        )
        '''
        mg_info = mesh_gruesa.project_multigrid(rho,dt_use,tol_div=divergencia, verbose=False)
        # Almacenar ciclos usados
        mesh_gruesa.mg_cycles_vector[it] = mg_info['cycles']
        # ⭐ DESPUÉS de proyección: NO sobrescribir outflow
        mesh_gruesa.apply_boundaries(after_projection=True)
        timing_stats['proyeccion'] += time.time() - t0
        
        # ============================================================
        # CLAMP DE VELOCIDADES (red de seguridad contra blowup)
        # ============================================================
        mesh_gruesa.u = cp.clip(mesh_gruesa.u, -vel_clamp_max, vel_clamp_max)
        mesh_gruesa.v = cp.clip(mesh_gruesa.v, -vel_clamp_max, vel_clamp_max)
        
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
            if abs(nuevo_alpha - alpha_actual) > 0.001 or it == iteracion_inicial:
                # Guardar punto polar del alpha que termina (salvo primera iteración)
                if it > iteracion_inicial:
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
        
        # Checkpoint periódico
        if save_checkpoint_every and (it % save_checkpoint_every == 0) and (it > iteracion_inicial):
            checkpoint_name = f"checkpoint_iter_{it:06d}.npz"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            
            metadata = {
                'iteracion': it,
                'tiempo_simulado': tiempo_fisico_acumulado,  # ⭐ Usar tiempo acumulado real
                'CFL': CFL,
                'alpha_deg': alpha_actual,
                'chord': chord,
                'Reynolds': (U_inf * chord) / nu,
                'dt': dt,
                'dt_ultimo': float(dt_use),  # ⭐ Guardar último dt usado
                'guardado': guardado
            }
            save_complete_checkpoint(checkpoint_path, mesh_gruesa, None, metadata)
        
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
    
    '''
    # ============================================================
    # BUCLE ALTERNATIVO: NESTED MESH (GRUESA + FINA)
    # ============================================================
    for it in tqdm(range(iteracion_inicial, iteraciones)):
        t_paso_inicio = time.time()
        
        # Recalcular dt cada iteración: usar el mínimo entre advectivo y difusivo
        t0 = time.time()
        try:
            # Velocidad máxima (fina y gruesa)
            speed_f = cp.sqrt(mesh_fina.u * mesh_fina.u + mesh_fina.v * mesh_fina.v)
            speed_g = cp.sqrt(mesh_gruesa.u * mesh_gruesa.u + mesh_gruesa.v * mesh_gruesa.v)
            Umax_cp = cp.maximum(cp.max(speed_f), cp.max(speed_g))
            Umax = float(Umax_cp) if float(Umax_cp) > 1e-12 else float(U_inf)

            # dt advectivo
            dt_adv = float(CFL * min(dx_fino, dy_fino) / max(Umax, 1e-12))

            # dt difusivo: calcular nu_t en malla fina (con WALE si está activo) y usar nu_eff max
            if mesh_fina.usar_wale:
                nu_t_f = mesh_fina.compute_wale_viscosity()
                nu_cp = cp.float32(nu)
                nu_eff_max_cp = cp.max(nu_cp + nu_t_f)
                nu_eff_max = float(nu_eff_max_cp) if float(nu_eff_max_cp) > 0 else 0.0
            else:
                nu_eff_max = float(nu)

            C_visc = 0.25
            if nu_eff_max > 0:
                dt_visc = float(C_visc * (min(dx_fino, dy_fino)**2) / nu_eff_max)
            else:
                dt_visc = float('inf')

            dt_use = min(dt_adv, dt_visc)
        except Exception:
            # En caso de fallo, conservar dt previamente calculado
            dt_use = dt
        timing_stats['recalculo_dt'] += time.time() - t0

        # ============================================================
        # PASO A: Resolver COMPLETAMENTE malla gruesa (3 pasos)
        # ============================================================
        # A.1) Advección gruesa
        t0 = time.time()
        u_prev = mesh_gruesa.u.copy()
        v_prev = mesh_gruesa.v.copy()
        
        mesh_gruesa.advect_velocities(dt_use)
        
        # Restaurar zona refinada (no evolucionada en malla gruesa directamente)
        mesh_gruesa.u = cp.where(geometria.mascara_gruesa_activa, 
                                 mesh_gruesa.u, u_prev)
        mesh_gruesa.v = cp.where(geometria.mascara_gruesa_activa,
                                 mesh_gruesa.v, v_prev)
        timing_stats['adveccion_gruesa'] += time.time() - t0
        
        # A.2) Difusión gruesa
        t0 = time.time()
        mesh_gruesa.diffuse_velocity(nu, dt_use, usar_wale=mesh_gruesa.usar_wale)
        timing_stats['difusion_gruesa'] += time.time() - t0

        # ============================================================
        # PASO B: Transferir BC actualizadas (gruesa → fina)
        # ============================================================
        t0 = time.time()
        j_g, i_g = geometria.coordenadas_finas_a_indices_gruesos()
        
        # Interpolar TODOS los campos (u,v,p) para mantener coherencia física
        u_interp = mesh_gruesa._bilinear_interpolate(mesh_gruesa.u, j_g, i_g)
        v_interp = mesh_gruesa._bilinear_interpolate(mesh_gruesa.v, j_g, i_g)
        p_interp = mesh_gruesa._bilinear_interpolate(mesh_gruesa.p, j_g, i_g)

        # Aplicar en el borde de malla fina
        # NOTA: Esto viola incompresibilidad pero es necesario para BC físicas
        mesh_fina.u = cp.where(geometria.mascara_fina_borde, u_interp, mesh_fina.u)
        mesh_fina.v = cp.where(geometria.mascara_fina_borde, v_interp, mesh_fina.v)
        mesh_fina.p = cp.where(geometria.mascara_fina_borde, p_interp, mesh_fina.p)
        timing_stats['transferencia_borde'] += time.time() - t0

        # ============================================================
        # PASO C: Resolver COMPLETAMENTE malla fina (3 pasos)
        # ============================================================
        # C.1) Advección fina
        t0 = time.time()
        mesh_fina.advect_velocities(dt_use)
        timing_stats['adveccion_fina'] += time.time() - t0
        
        # C.2) Difusión fina
        t0 = time.time()
        mesh_fina.diffuse_velocity(nu, dt_use, usar_wale=mesh_fina.usar_wale)
        timing_stats['difusion_fina'] += time.time() - t0
        
        # C.3) Proyección final (con tolerancia relajada, aceptando BC inconsistentes)
        t0 = time.time()
        mesh_fina.project_cg(rho, dt_use, tol_div=divergencia, max_iter=100, verbose=False)
        timing_stats['proyeccion_fina'] += time.time() - t0
        


        # ============================================================
        # PASO D: Retroalimentar fina → gruesa (en zona de solapamiento)
        # ============================================================
        t0 = time.time()
        # Coordenadas de centros de celdas gruesas en sistema fino
        x_g = (cp.arange(mesh_gruesa.nx) + 0.5) * dx_grueso
        y_g = (cp.arange(mesh_gruesa.ny) + 0.5) * dy_grueso
        XX_g, YY_g = cp.meshgrid(x_g, y_g, indexing='xy')
        
        # Convertir a índices en malla fina
        j_f = (XX_g - geometria.x_min_fino) / dx_fino - 0.5
        i_f = (YY_g - geometria.y_min_fino) / dy_fino - 0.5
        
        # Interpolar campos finos a posiciones gruesas
        u_fina_a_gruesa = mesh_fina._bilinear_interpolate(mesh_fina.u, j_f, i_f)
        v_fina_a_gruesa = mesh_fina._bilinear_interpolate(mesh_fina.v, j_f, i_f)
        p_fina_a_gruesa = mesh_fina._bilinear_interpolate(mesh_fina.p, j_f, i_f)
        
        # Blend en zona de solapamiento
        mesh_gruesa.u = cp.where(geometria.peso_fino > 0,
                                 geometria.peso_grueso * mesh_gruesa.u + geometria.peso_fino * u_fina_a_gruesa,
                                 mesh_gruesa.u)
        mesh_gruesa.v = cp.where(geometria.peso_fino > 0,
                                 geometria.peso_grueso * mesh_gruesa.v + geometria.peso_fino * v_fina_a_gruesa,
                                 mesh_gruesa.v)
        mesh_gruesa.p = cp.where(geometria.peso_fino > 0,
                                 geometria.peso_grueso * mesh_gruesa.p + geometria.peso_fino * p_fina_a_gruesa,
                                 mesh_gruesa.p)
        timing_stats['retroalimentacion'] += time.time() - t0


        # A.3) Proyección gruesa
        t0 = time.time()
        mesh_gruesa.project2_adaptive(rho, dt_use, tol_div=divergencia, max_iter=500, verbose=False)
        timing_stats['proyeccion_gruesa'] += time.time() - t0


        # Guardar fuerzas (calculadas en malla fina)
        t0 = time.time()
        if it % guardado == 0:
            forces = mesh_fina.compute_drag_lift(mu, rho=rho, n_extrap_layers=5)     
            cd_val = 2 * forces['Drag'] / (rho * U_inf**2 * chord)
            cl_val = 2 * forces['Lift'] / (rho * U_inf**2 * chord)
            mesh_fina.cdvector[it // guardado] = cd_val
            mesh_fina.clvector[it // guardado] = cl_val
            # Guardar ratio Cl/Cd de forma segura
            try:
                if abs(cd_val) < 1e-12:
                    ratio = np.nan
                else:
                    ratio = cl_val / cd_val
            except Exception:
                ratio = np.nan
            mesh_fina.clcdvector[it // guardado] = ratio
            mesh_fina.divvector[it // guardado] = mesh_fina.compute_divergence_mean()
            mesh_fina.update_cp_profile(mu, rho)

            # Guardar frames si se solicita
            if save_frames:
                if frames_dir_fino:
                    mesh_fina.save_frame(frames_dir_fino, it, kind="velocity")
                if frames_dir_grueso:
                    mesh_gruesa.save_frame(frames_dir_grueso, it, kind="velocity")
                # Liberar memoria de figuras matplotlib cada cierto número de frames
                if it % (guardado * 10) == 0:
                    plt.close('all')
        
        # Guardar checkpoint periódicamente
        if save_checkpoint_every and (it % save_checkpoint_every == 0) and (it > iteracion_inicial):
            checkpoint_name = f"checkpoint_iter_{it:06d}.npz"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            
            # Metadata para reanudación consistente
            metadata = {
                'iteracion': it,
                'tiempo_simulado': tiempo_simulado_previo + (it - iteracion_inicial) * dt,
                'CFL': CFL,
                'alpha_deg': alpha_actual,
                'chord': chord,
                'Reynolds': (U_inf * chord) / nu,
                'dt': dt,
                'guardado': guardado,
                'dx_fino': float(dx_fino),
                'dy_fino': float(dy_fino),
                'dx_grueso': float(dx_grueso),
                'dy_grueso': float(dy_grueso),
                'Lx': float(Lx),
                'Ly': float(Ly),
                'v0x': float(v0x),
                'v0y': float(v0y),
                'p0': float(p0),
                'rho': float(rho),
                'nu': float(nu),
                'tol_poisson': float(tol_poisson),
                'max_iter_poisson': int(max_iter_poisson),
                'ancho_ref_factor': float(ancho_ref_factor),
                'alto_ref_factor': float(alto_ref_factor),
                'offset_refinado': float(offset_refinado),
                'cx': float(cx),
                'cy': float(cy),
                'filepath': str(filepath),
                'boundary_left': boundary_left,
                'boundary_top': boundary_top,
                'boundary_bottom': boundary_bottom,
                'boundary_right': boundary_right,
                'n_layers': int(n_layers),
                'step_base': float(step_base),
                'step_inc': float(step_inc)
            }            
            # Guardar checkpoint completo (un solo archivo)
            save_complete_checkpoint(checkpoint_path, mesh_fina, mesh_gruesa, metadata)
        
        # Registrar tiempo total del paso y tiempo de guardado
        if it % guardado == 0:
            timing_stats['guardado'] += time.time() - t0
        
        t_paso_total = time.time() - t_paso_inicio
        timing_stats['total_por_paso'].append(t_paso_total)
   '''

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

    return mesh_fina, mesh_gruesa, geometria
    # Parámetros de simulación 

if __name__ == "__main__":
    mesh_finat,mesh_gruesat,geo=main(    
    Lx=6,
    Ly=6 ,  
    T=0.5,
    cx=2,
    CFL=0.8,
    alpha_deg=6,
    polar_descarte=0.3,
    iteraciones=4000,
    divergencia=1e-1,
    v0x=5,
    v0y=0,
    filepath="AG24",
    chord=1.0,
    dx_grueso=0.002,    
    graficos=True,
    save_frames=False, 
    frames_dir_grueso="",
    usar_wale=False,
    usar_viscosidad_estela=False,
    C_estela=0.05,
    stop_on_convergence=False,
    live_view=True

)
'''    
plan_polar=[
           # α=0°  durante 3000 iteraciones
        (2,  3000),   # α=2°  durante 2000 iteraciones
        (4,  3000),   # α=4°  durante 2000 iteraciones
        (6,  3000),   # α=6°  durante 2000 iteraciones
           # α=8°  durante 2000 iteraciones

    ],
'''
# OPCIÓN 1: Descomentar para ejecutar con checkpoint
# mesh_finat,mesh_gruesat,geo=main(checkpoint_file="checkpoints_test/checkpoint_iter_001000.npz",graficos=True)

# OPCIÓN 2: Descomentar para ejecutar sin checkpoint (simulación nueva)
# mesh_finat,mesh_gruesat,geo=main(CFL=0.8, T=0.2, alpha_deg=5, graficos=True)
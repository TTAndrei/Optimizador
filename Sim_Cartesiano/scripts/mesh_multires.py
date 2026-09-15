import cupy as cp
import numpy as np

class MallaMultiRes:
    """
    Genera y gestiona dos mallas acopladas:
    - Malla gruesa (global): cubre todo el dominio
    - Malla fina (local): región rectangular alrededor del sólido
    
    NO contiene lógica de simulación, solo geometría y máscaras.
    """
    
    def __init__(self, Lx, Ly, dx_grueso, dx_fino, centro_refinado, ancho_refinado, alto_refinado, tipo_zona="rectangular"):
        """
        Parámetros:
            Lx, Ly: dimensiones del dominio (m)
            dx_grueso: resolución de malla gruesa (m)
            dx_fino: resolución de malla fina (m)
            centro_refinado: (cx, cy) centro de región refinada (m)
            ancho_refinado: ancho de región refinada (m)
            alto_refinado: alto de región refinada (m)
            tipo_zona: "rectangular" o "circular" - forma de la zona refinada
        """
        self.Lx = Lx
        self.Ly = Ly
        self.dx_grueso = dx_grueso
        self.dx_fino = dx_fino
        self.centro_ref = centro_refinado
        self.ancho_ref = ancho_refinado
        self.alto_ref = alto_refinado
        self.tipo_zona = tipo_zona
        
        assert tipo_zona in ["rectangular", "circular"], "tipo_zona debe ser 'rectangular' o 'circular'"
        
        # Dimensiones malla gruesa
        self.nx_grueso = int(Lx / dx_grueso)
        self.ny_grueso = int(Ly / dx_grueso)
        
        # Límites de malla fina (rectángulo de refinamiento)
        cx, cy = centro_refinado
        self.x_min_fino = max(0, cx - ancho_refinado / 2)
        self.x_max_fino = min(Lx, cx + ancho_refinado / 2)
        self.y_min_fino = max(0, cy - alto_refinado / 2)
        self.y_max_fino = min(Ly, cy + alto_refinado / 2)
        
        Lx_fino = self.x_max_fino - self.x_min_fino
        Ly_fino = self.y_max_fino - self.y_min_fino
        
        # Dimensiones malla fina
        self.nx_fino = int(Lx_fino / dx_fino)
        self.ny_fino = int(Ly_fino / dx_fino)
        
        # Zona de solapamiento (aumentada para transición más suave)
        self.n_solapa = 8  # Aumentado de 4 a 8 celdas gruesas
        self.ancho_solapa = self.n_solapa * dx_grueso
        
        # Generar máscaras y pesos
        self._generar_mascaras()

        self._compilar_kernel_retroalimentacion()
        
        print(f"Malla gruesa: {self.nx_grueso}x{self.ny_grueso} (dx={dx_grueso:.4f}m)")
        print(f"Malla fina: {self.nx_fino}x{self.ny_fino} (dx={dx_fino:.4f}m)")
        print(f"Factor refinamiento: {dx_grueso/dx_fino:.1f}x")
        print(f"Tipo de zona refinada: {tipo_zona}")
    
    def _generar_mascaras(self):
        """Genera máscaras y pesos para acoplamiento rectangular o circular."""
        # Coordenadas físicas centros de celda
        x_g = (cp.arange(self.nx_grueso) + 0.5) * self.dx_grueso
        y_g = (cp.arange(self.ny_grueso) + 0.5) * self.dx_grueso
        XX_g, YY_g = cp.meshgrid(x_g, y_g, indexing='xy')
        
        x_f = self.x_min_fino + (cp.arange(self.nx_fino) + 0.5) * self.dx_fino
        y_f = self.y_min_fino + (cp.arange(self.ny_fino) + 0.5) * self.dx_fino
        XX_f, YY_f = cp.meshgrid(x_f, y_f, indexing='xy')
        
        cx, cy = self.centro_ref
        
        if self.tipo_zona == "rectangular":
            # ========== ZONA RECTANGULAR ==========
            # Límites de región refinada
            x_min_ref = cx - self.ancho_ref / 2
            x_max_ref = cx + self.ancho_ref / 2
            y_min_ref = cy - self.alto_ref / 2
            y_max_ref = cy + self.alto_ref / 2
            
            # Máscara gruesa: celdas FUERA del rectángulo refinado
            margen = self.dx_grueso
            dentro_rect_x = (XX_g >= x_min_ref - margen) & (XX_g <= x_max_ref + margen)
            dentro_rect_y = (YY_g >= y_min_ref - margen) & (YY_g <= y_max_ref + margen)
            self.mascara_gruesa_activa = ~(dentro_rect_x & dentro_rect_y)
            
            # Máscara fina: borde (reciben datos de gruesa)
            margen_borde = 3 * self.dx_fino
            en_borde_x = (XX_f < self.x_min_fino + margen_borde) | (XX_f > self.x_max_fino - margen_borde)
            en_borde_y = (YY_f < self.y_min_fino + margen_borde) | (YY_f > self.y_max_fino - margen_borde)
            self.mascara_fina_borde = en_borde_x | en_borde_y
            
            # Pesos para blend (transición suave con coseno en los bordes del rectángulo)
            self.peso_fino = cp.zeros_like(XX_g, dtype=cp.float32)
            
            # Distancias a los bordes del rectángulo (negativo=dentro, positivo=fuera)
            dist_x = cp.maximum(x_min_ref - XX_g, XX_g - x_max_ref)
            dist_y = cp.maximum(y_min_ref - YY_g, YY_g - y_max_ref)
            dist_borde = cp.maximum(dist_x, dist_y)
            
            # Zona de solapamiento
            borde_interior = -self.ancho_solapa
            borde_exterior = 0.0
            
            # Peso fino = 1 dentro, 0 fuera, coseno en solapamiento
            self.peso_fino[dist_borde < borde_interior] = 1.0
            en_solapa = (dist_borde >= borde_interior) & (dist_borde <= borde_exterior)
            t = (dist_borde[en_solapa] - borde_interior) / self.ancho_solapa
            self.peso_fino[en_solapa] = 0.5 * (1 + cp.cos(cp.pi * t))
            
        else:  # tipo_zona == "circular"
            # ========== ZONA CIRCULAR ==========
            # Radio de la región refinada (promedio de ancho y alto)
            radio_ref = (self.ancho_ref + self.alto_ref) / 4.0  # dividir por 4 para convertir diámetros a radio
            
            # Distancias al centro
            dist_centro_g = cp.sqrt((XX_g - cx)**2 + (YY_g - cy)**2)
            dist_centro_f = cp.sqrt((XX_f - cx)**2 + (YY_f - cy)**2)
            
            # Máscara gruesa: celdas FUERA del círculo refinado
            margen = self.dx_grueso
            self.mascara_gruesa_activa = dist_centro_g > (radio_ref + margen)
            
            # Máscara fina: borde circular (reciben datos de gruesa)
            margen_borde = 3 * self.dx_fino
            radio_fina = cp.sqrt((self.x_max_fino - self.x_min_fino)**2 + 
                                 (self.y_max_fino - self.y_min_fino)**2) / 2.0
            self.mascara_fina_borde = dist_centro_f > (radio_fina - margen_borde)
            
            # Pesos para blend (transición suave con coseno radial)
            self.peso_fino = cp.zeros_like(XX_g, dtype=cp.float32)
            
            # Distancia al borde del círculo (negativo=dentro, positivo=fuera)
            dist_borde = dist_centro_g - radio_ref
            
            # Zona de solapamiento
            borde_interior = -self.ancho_solapa
            borde_exterior = 0.0
            
            # Peso fino = 1 dentro, 0 fuera, coseno en solapamiento
            self.peso_fino[dist_borde < borde_interior] = 1.0
            en_solapa = (dist_borde >= borde_interior) & (dist_borde <= borde_exterior)
            t = (dist_borde[en_solapa] - borde_interior) / self.ancho_solapa
            self.peso_fino[en_solapa] = 0.5 * (1 + cp.cos(cp.pi * t))
        
        self.peso_grueso = 1.0 - self.peso_fino
        
        # Guardar coordenadas
        self.XX_g = XX_g
        self.YY_g = YY_g
        self.XX_f = XX_f
        self.YY_f = YY_f
    
    def indices_fino_a_grueso(self, j_f, i_f):
        """
        Convierte índices de malla fina a índices de malla gruesa.
        
        Retorna: (j_grueso, i_grueso) en floats para interpolación
        """
        x_f = self.x_min_fino + (j_f + 0.5) * self.dx_fino
        y_f = self.y_min_fino + (i_f + 0.5) * self.dx_fino
        
        j_g = (x_f - 0.5 * self.dx_grueso) / self.dx_grueso
        i_g = (y_f - 0.5 * self.dx_grueso) / self.dx_grueso
        
        return j_g, i_g
    
    def coordenadas_finas_a_indices_gruesos(self):
        """
        Retorna arrays 2D con índices gruesos para cada punto fino.
        Útil para interpolación bilineal gruesa → fina.
        """
        j_g = (self.XX_f - 0.5 * self.dx_grueso) / self.dx_grueso
        i_g = (self.YY_f - 0.5 * self.dx_grueso) / self.dx_grueso
        
        return j_g, i_g
    
    def get_limites_fino(self):
        """Retorna límites físicos de malla fina."""
        return {
            'x_min': self.x_min_fino,
            'x_max': self.x_max_fino,
            'y_min': self.y_min_fino,
            'y_max': self.y_max_fino,
            'Lx': self.x_max_fino - self.x_min_fino,
            'Ly': self.y_max_fino - self.y_min_fino
        }
    
    def offset_solido_para_fino(self, x_offset, y_offset):

        """
        Ajusta coordenadas de sólido de sistema grueso a sistema fino.
        
        Retorna: (x_offset_fino, y_offset_fino)
        """
        return x_offset - self.x_min_fino, y_offset - self.y_min_fino

    def visualizar_mallado(self, mostrar_pesos=False, guardar=None):
        """
        Visualiza la configuración de mallas y máscaras.
        
        Parámetros:
            mostrar_pesos: si True, muestra los pesos de blend
            guardar: ruta para guardar la figura (None = solo mostrar)
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        fig, axes = plt.subplots(1, 2 if not mostrar_pesos else 3, figsize=(15 if not mostrar_pesos else 20, 6))
        
        # Convertir a numpy para visualización
        mascara_gruesa_np = cp.asnumpy(self.mascara_gruesa_activa)
        mascara_fina_np = cp.asnumpy(self.mascara_fina_borde)
        
        # Panel 1: Malla gruesa con región refinada marcada
        ax1 = axes[0]
        im1 = ax1.imshow(mascara_gruesa_np, origin="lower", cmap="RdYlGn", alpha=0.6,
                        extent=[0, self.Lx, 0, self.Ly], interpolation='nearest')
        ax1.set_title(f"Malla Gruesa (dx={self.dx_grueso:.4f}m)\nVerde=Activa, Rojo=Refinada")
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")
        
        cx, cy = self.centro_ref
        
        if self.tipo_zona == "rectangular":
            # Marcar región refinada (rectángulo principal)
            rect_refinado = Rectangle(
                (cx - self.ancho_ref/2, cy - self.alto_ref/2),
                self.ancho_ref, self.alto_ref,
                fill=False, edgecolor='blue', linewidth=2, linestyle='--', label='Región refinada')
            ax1.add_patch(rect_refinado)
            
            # Marcar zona de solapamiento
            rect_solapa = Rectangle(
                (cx - self.ancho_ref/2 + self.ancho_solapa, cy - self.alto_ref/2 + self.ancho_solapa),
                self.ancho_ref - 2*self.ancho_solapa, self.alto_ref - 2*self.ancho_solapa,
                fill=False, edgecolor='orange', linewidth=1.5, linestyle=':', label='Zona solapamiento')
            ax1.add_patch(rect_solapa)
        else:  # circular
            from matplotlib.patches import Circle
            radio_ref = (self.ancho_ref + self.alto_ref) / 4.0
            
            # Marcar región refinada (círculo principal)
            circ_refinado = Circle(
                (cx, cy), radio_ref,
                fill=False, edgecolor='blue', linewidth=2, linestyle='--', label='Región refinada')
            ax1.add_patch(circ_refinado)
            
            # Marcar zona de solapamiento
            circ_solapa = Circle(
                (cx, cy), radio_ref - self.ancho_solapa,
                fill=False, edgecolor='orange', linewidth=1.5, linestyle=':', label='Zona solapamiento')
            ax1.add_patch(circ_solapa)
        
        # Marcar dominio fino
        rect_fino = Rectangle((self.x_min_fino, self.y_min_fino),
                        self.x_max_fino - self.x_min_fino,
                        self.y_max_fino - self.y_min_fino,
                        fill=False, edgecolor='red', linewidth=2, label='Dominio fino')
        ax1.add_patch(rect_fino)
        
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Malla fina con bordes marcados
        ax2 = axes[1]
        im2 = ax2.imshow(~mascara_fina_np, origin="lower", cmap="RdYlGn", alpha=0.6,
                        extent=[self.x_min_fino, self.x_max_fino,
                               self.y_min_fino, self.y_max_fino],
                        interpolation='nearest')
        ax2.set_title(f"Malla Fina (dx={self.dx_fino:.4f}m)\nVerde=Interior, Rojo=Borde")
        ax2.set_xlabel("x (m)")
        ax2.set_ylabel("y (m)")
        ax2.grid(True, alpha=0.3)
        
        # Panel 3 (opcional): Pesos de blend
        if mostrar_pesos:
            ax3 = axes[2]
            peso_fino_np = cp.asnumpy(self.peso_fino)
            im3 = ax3.imshow(peso_fino_np, origin="lower", cmap="coolwarm",
                           extent=[0, self.Lx, 0, self.Ly], interpolation='bilinear')
            ax3.set_title("Pesos de Blend\nAzul=Gruesa, Rojo=Fina")
            ax3.set_xlabel("x (m)")
            ax3.set_ylabel("y (m)")
            plt.colorbar(im3, ax=ax3, label="Peso malla fina")
            
            # Marcar transición según el tipo
            if self.tipo_zona == "rectangular":
                rect_ext = Rectangle(
                    (cx - self.ancho_ref/2, cy - self.alto_ref/2),
                    self.ancho_ref, self.alto_ref,
                    fill=False, edgecolor='white', linewidth=2, linestyle='--')
                rect_int = Rectangle(
                    (cx - self.ancho_ref/2 + self.ancho_solapa, cy - self.alto_ref/2 + self.ancho_solapa),
                    self.ancho_ref - 2*self.ancho_solapa, self.alto_ref - 2*self.ancho_solapa,
                    fill=False, edgecolor='white', linewidth=2, linestyle='--')
                ax3.add_patch(rect_ext)
                ax3.add_patch(rect_int)
            else:  # circular
                from matplotlib.patches import Circle
                radio_ref = (self.ancho_ref + self.alto_ref) / 4.0
                circ_ext = Circle((cx, cy), radio_ref,
                                 fill=False, edgecolor='white', linewidth=2, linestyle='--')
                circ_int = Circle((cx, cy), radio_ref - self.ancho_solapa,
                                 fill=False, edgecolor='white', linewidth=2, linestyle='--')
                ax3.add_patch(circ_ext)
                ax3.add_patch(circ_int)
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if guardar:
            plt.savefig(guardar, dpi=150, bbox_inches='tight')
            print(f"Figura guardada en: {guardar}")
        
        plt.show()

    def _compilar_kernel_retroalimentacion(self):
        """Kernel GPU para promediar malla fina → gruesa con filtrado de sólido y clamping."""
        kernel_src = r'''
        extern "C" __global__
        void retroalimentar_fina_gruesa(
            const float* __restrict__ u_fina,
            const float* __restrict__ v_fina,
            const float* __restrict__ p_fina,
            const unsigned char* __restrict__ solid_fina, // 0=fluido, 1=sólido
            float* __restrict__ u_gruesa,
            float* __restrict__ v_gruesa,
            float* __restrict__ p_gruesa,
            const float* __restrict__ peso_grueso,
            const float* __restrict__ peso_fino,
            int nx_grueso, int ny_grueso,
            int nx_fino, int ny_fino,
            float dx_grueso, float dx_fino,
            float x_min_fino, float y_min_fino,
            int ratio)
        {
            int j_g = blockIdx.x * blockDim.x + threadIdx.x;
            int i_g = blockIdx.y * blockDim.y + threadIdx.y;
            if (i_g >= ny_grueso || j_g >= nx_grueso) return;

            int idx_g = i_g * nx_grueso + j_g;

            // Solo procesar celdas con peso_fino > 0 (solapamiento)
            float w_f = peso_fino[idx_g];
            if (w_f <= 0.0f) return;

            // Coordenadas físicas centro celda gruesa
            float x_g = (j_g + 0.5f) * dx_grueso;
            float y_g = (i_g + 0.5f) * dx_grueso;

            // Verificar si está dentro del rectángulo fino
            float x_max_fino = x_min_fino + nx_fino * dx_fino;
            float y_max_fino = y_min_fino + ny_fino * dx_fino;
            if (x_g < x_min_fino || x_g >= x_max_fino ||
                y_g < y_min_fino || y_g >= y_max_fino) return;

            // Índice central en malla fina (clamping)
            int j_f = (int)((x_g - x_min_fino) / dx_fino);
            int i_f = (int)((y_g - y_min_fino) / dx_fino);
            if (j_f < 0) j_f = 0;
            if (i_f < 0) i_f = 0;
            if (j_f >= nx_fino) j_f = nx_fino - 1;
            if (i_f >= ny_fino) i_f = ny_fino - 1;

            // Promediar ventana ratio×ratio filtrando sólido
            float suma_u = 0.0f;
            float suma_v = 0.0f;
            float suma_p = 0.0f;
            int cuenta = 0;

            int half_ratio = max(1, ratio / 2);
            for (int di = -half_ratio; di <= half_ratio; di++) {
                for (int dj = -half_ratio; dj <= half_ratio; dj++) {
                    int i_local = i_f + di;
                    int j_local = j_f + dj;

                    if (i_local < 0 || i_local >= ny_fino ||
                        j_local < 0 || j_local >= nx_fino) continue;

                    int idx_f = i_local * nx_fino + j_local;

                    // Ignorar sólido fino
                    if (solid_fina[idx_f] != 0) continue;

                    float uf = u_fina[idx_f];
                    float vf = v_fina[idx_f];
                    float pf = p_fina[idx_f];

                    // Filtrar NaN/Inf
                    if (!isfinite(uf) || !isfinite(vf) || !isfinite(pf)) continue;

                    suma_u += uf;
                    suma_v += vf;
                    suma_p += pf;
                    cuenta++;
                }
            }

            if (cuenta == 0) return; // sin contribución válida

            float u_prom = suma_u / (float)cuenta;
            float v_prom = suma_v / (float)cuenta;
            float p_prom = suma_p / (float)cuenta;

            // Blend con pesos
            float w_g = peso_grueso[idx_g];
            u_gruesa[idx_g] = w_g * u_gruesa[idx_g] + w_f * u_prom;
            v_gruesa[idx_g] = w_g * v_gruesa[idx_g] + w_f * v_prom;
            p_gruesa[idx_g] = w_g * p_gruesa[idx_g] + w_f * p_prom;
        }
        '''
        self._kernel_retroalimentar = cp.RawKernel(kernel_src, 'retroalimentar_fina_gruesa')

    def retroalimentar_a_gruesa_gpu(self, mesh_gruesa, mesh_fina):
        """
        Retroalimenta solución de malla fina → gruesa usando kernel GPU.
        """
        # Validar ratio entero
        ratio = int(round(self.dx_grueso / self.dx_fino))
        if ratio < 1:
            ratio = 1

        # Configuración del kernel
        block = (16, 16)
        grid = (
            (self.nx_grueso + block[0] - 1) // block[0],
            (self.ny_grueso + block[1] - 1) // block[1]
        )

        # Aplanados y tipos
        u_f = mesh_fina.u.ravel().astype(cp.float32, copy=False)
        v_f = mesh_fina.v.ravel().astype(cp.float32, copy=False)
        p_f = mesh_fina.p.ravel().astype(cp.float32, copy=False)
        # Máscara sólido fina: asumir mesh_fina.solid tipo bool → uint8
        solid_f = mesh_fina.solid.ravel().astype(cp.uint8, copy=False)

        u_g = mesh_gruesa.u.ravel().astype(cp.float32, copy=False)
        v_g = mesh_gruesa.v.ravel().astype(cp.float32, copy=False)
        p_g = mesh_gruesa.p.ravel().astype(cp.float32, copy=False)

        wg = self.peso_grueso.ravel().astype(cp.float32, copy=False)
        wf = self.peso_fino.ravel().astype(cp.float32, copy=False)

        # Lanzar kernel
        self._kernel_retroalimentar(
            grid, block,
            (
                u_f, v_f, p_f,
                solid_f,
                u_g, v_g, p_g,
                wg, wf,
                cp.int32(self.nx_grueso), cp.int32(self.ny_grueso),
                cp.int32(self.nx_fino),   cp.int32(self.ny_fino),
                cp.float32(self.dx_grueso), cp.float32(self.dx_fino),
                cp.float32(self.x_min_fino), cp.float32(self.y_min_fino),
                cp.int32(ratio)
            )
        )


if __name__ == "__main__":
    """
    Ejemplo de uso: genera mallas para simulación alrededor de NACA 0012
    """
    # Parámetros del dominio
    Lx, Ly = 20.0, 8.0  # metros
    
    # Resoluciones
    dx_grueso = 0.01    
    dx_fino = 0.0025   
    
    # Región refinada rectangular (centrada en el perfil)
    cx, cy = 2.5, Ly/2   # centro del dominio
    chord = 1.0         # cuerda del perfil
    ancho_refinado = 3.0 * chord  # 3 cuerdas de ancho
    alto_refinado = 1.5 * chord   # 1.5 cuerdas de alto
    
    # Crear sistema de mallas
    print("=" * 60)
    print("Generando sistema de mallas multi-resolución")
    print("=" * 60)
    
    malla = MallaMultiRes(
        Lx=Lx,
        Ly=Ly,
        dx_grueso=dx_grueso,
        dx_fino=dx_fino,
        centro_refinado=(cx, cy),
        ancho_refinado=ancho_refinado,
        alto_refinado=alto_refinado
    )
    
    print("\n" + "=" * 60)
    print("Información de máscaras:")
    print("=" * 60)
    
    # Estadísticas
    total_gruesa = malla.nx_grueso * malla.ny_grueso
    activas_gruesa = int(cp.sum(malla.mascara_gruesa_activa))
    print(f"Celdas gruesas activas: {activas_gruesa}/{total_gruesa} ({100*activas_gruesa/total_gruesa:.1f}%)")
    
    total_fina = malla.nx_fino * malla.ny_fino
    borde_fino = int(cp.sum(malla.mascara_fina_borde))
    print(f"Celdas finas en borde: {borde_fino}/{total_fina} ({100*borde_fino/total_fina:.1f}%)")
    
    print(f"\nAhorro de memoria vs malla uniforme fina:")
    celdas_uniforme = int(Lx * Ly / (dx_fino ** 2))
    celdas_multires = total_gruesa + total_fina
    ahorro = 100 * (1 - celdas_multires / celdas_uniforme)
    print(f"  Uniforme fina: {celdas_uniforme:,} celdas")
    print(f"  Multi-res: {celdas_multires:,} celdas")
    print(f"  Ahorro: {ahorro:.1f}%")
    
    # Visualizar
    print("\n" + "=" * 60)
    print("Generando visualización...")
    print("=" * 60)
    malla.visualizar_mallado(mostrar_pesos=True, guardar="mallado_multires.png")
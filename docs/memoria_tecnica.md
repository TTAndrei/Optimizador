# Memoria Técnica: Simulador CFD 2D con Optimización Genética de Perfiles Aerodinámicos

---

## 1. Introducción

### 1.1 Motivación

El diseño de perfiles aerodinámicos eficientes es un problema central en ingeniería aeronáutica, eólica y automovilística. Los ensayos experimentales en túnel de viento tienen un coste elevado en tiempo e infraestructura, y las simulaciones tridimensionales de alta fidelidad (RANS o LES 3D) requieren cientos de horas de CPU en clústeres de computación. Existe una necesidad clara de herramientas rápidas y suficientemente precisas para la exploración de geometrías en etapas de diseño preliminar, donde el número de configuraciones a evaluar puede ser del orden de miles.

El avance reciente de las GPUs de consumo, junto con bibliotecas como CuPy que permiten ejecutar código CUDA desde Python, ha abierto la posibilidad de implementar solvers de fluidos 2D con un rendimiento de varios pasos temporales por segundo en hardware accesible.

### 1.2 Objetivos del trabajo

- Desarrollar un simulador CFD 2D transitorio completo: malla variable, bucle temporal, condiciones de frontera, sólidos embebidos.
- Implementar Large Eddy Simulation (LES) con el modelo de viscosidad sub-malla WALE para capturar efectos turbulentos.
- Acelerar todo el cálculo mediante GPU utilizando CuPy y kernels CUDA personalizados, minimizando transferencias CPU-GPU.
- Calcular fuerzas aerodinámicas (sustentación y resistencia) sobre perfiles mediante integración de esfuerzos en superficie.
- Optimizar geometrías de perfiles aerodinámicos mediante un algoritmo genético que usa el simulador CFD como función de evaluación.

### 1.3 Alcance y limitaciones

- **Flujo 2D incompresible**: las ecuaciones de Navier-Stokes se resuelven en el plano (x, y). No se modelan efectos tridimensionales.
- **Régimen transitorio**: el bucle temporal avanza paso a paso; no se asume estado estacionario.
- **Aproximación LES**: las escalas sub-malla se modelan con viscosidad turbulenta (modelo WALE). Las escalas resueltas dependen de la resolución de malla.
- **Optimización centrada en perfiles aerodinámicos**: el GA opera sobre coordenadas de perfiles en formato Selig (.dat). La función de fitness es L/D obtenido de una simulación CFD completa.

### 1.4 Estructura del documento

El capítulo 2 presenta la base teórica: ecuaciones de gobierno, modelo turbulento y aerodinámica de perfiles. El capítulo 3 describe la implementación completa del simulador y el optimizador. El capítulo 4 resume los resultados obtenidos. Los capítulos 5 y 6 discuten limitaciones y conclusiones.

---

## 2. Teoría

### 2.1 Ecuaciones de Navier-Stokes

Las ecuaciones que gobiernan el flujo incompresible viscoso son la conservación de cantidad de movimiento y la condición de masa:

**Conservación de cantidad de movimiento:**

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu_{eff} \nabla^2 \mathbf{u}$$

**Conservación de masa (incompresibilidad):**

$$\nabla \cdot \mathbf{u} = 0$$

donde **u** = (u, v) es el campo de velocidades, p la presión, ρ la densidad y ν_eff la viscosidad cinemática efectiva (molecular + turbulenta).

Cada término tiene un significado físico concreto:
- **Advección** (u·∇)u: transporte de momento por el propio flujo. Responsable de la convección de estructuras turbulentas aguas abajo.
- **Difusión viscosa** ν∇²u: redistribución del momento por fuerzas de fricción molecular y turbulenta. Tiende a homogeneizar gradientes de velocidad.
- **Gradiente de presión** -(1/ρ)∇p: fuerza por unidad de masa debida a diferencias de presión. Acelera el fluido desde zonas de alta a baja presión y, bajo la restricción de incompresibilidad, actúa como multiplicador de Lagrange que proyecta el campo de velocidades sobre el espacio solenoidal.

### 2.2 Restricción de incompresibilidad

La condición ∇·u = 0 expresa la conservación exacta de masa en fluidos incompresibles: el volumen de fluido es constante. En la discretización numérica, esta condición no se satisface automáticamente tras el paso de advección y difusión, por lo que se requiere un paso de proyección (Chorin) que resuelve una ecuación de Poisson para la presión y corrige las velocidades.

La presión no tiene dinámica propia en flujo incompresible: actúa como multiplicador de Lagrange que ajusta el campo de velocidades para que satisfaga la divergencia cero en cada paso temporal.

### 2.3 Turbulencia

La turbulencia es un régimen de flujo caracterizado por fluctuaciones de velocidad en un amplio espectro de escalas espaciales y temporales, con transferencia de energía desde las grandes escalas (energéticas) hasta las pequeñas (disipativas). En 2D el comportamiento difiere del caso 3D: la cascada de energía puede ser inversa y no existe el mecanismo de estiramiento de vórtices.

Los enfoques numéricos para turbulencia se clasifican según qué escalas se resuelven directamente:
- **DNS** (Direct Numerical Simulation): resuelve todas las escalas, coste prohibitivo para Re moderado-alto.
- **RANS** (Reynolds-Averaged N-S): promedia en tiempo, modela toda la turbulencia. Barato pero inexacto en flujos separados.
- **LES** (Large Eddy Simulation): resuelve las escalas grandes, modela solo las sub-malla (SGS). Equilibrio entre coste y fidelidad.

Se elige LES porque captura correctamente la dinámica de las grandes estructuras coherentes (vórtices de Kármán, burbujas de separación) que determinan el comportamiento aerodinámico. Las escalas sub-malla, más universales y menos dependientes de la geometría, se modelan con el modelo WALE.

### 2.4 Modelo WALE

El modelo WALE (Wall-Adapting Local Eddy-viscosity, Nicoud & Ducros 1999) calcula la viscosidad turbulenta sub-malla ν_t a partir del tensor de gradientes de velocidad g_ij = ∂u_i/∂x_j.

**Tensor de tasas de deformación:**

$$S_{ij} = \frac{1}{2}(g_{ij} + g_{ji})$$

**Tensor de gradientes al cuadrado y su parte simétrica desviatorica:**

$$g^2_{ij} = g_{ik}g_{kj}, \quad S^d_{ij} = \frac{1}{2}(g^2_{ij} + g^2_{ji}) - \frac{\delta_{ij}}{2}tr(g^2)$$

**Viscosidad turbulenta:**

$$\nu_t = C_w^2 \Delta^2 \frac{(S^d_{ij}S^d_{ij})^{3/2}}{(S_{ij}S_{ij})^{5/2} + (S^d_{ij}S^d_{ij})^{5/4}}$$

donde Δ es el tamaño local de celda (raíz cuadrada del área) y C_w es el coeficiente WALE (por defecto 0.325 en 3D, en la implementación se usa 0.15 para simulaciones 2D, ajustado empíricamente).

Las ventajas frente al modelo de Smagorinsky son:
- **Comportamiento correcto en paredes**: ν_t → 0 en la pared sin necesidad de funciones de amortiguamiento artificiales.
- **Desaparición natural en flujo laminar**: cuando S^d_ij → 0, ν_t = 0. Esto es fundamental para perfiles a bajo Reynolds donde el flujo es laminar sobre la mayor parte de la superficie.

En la implementación (`compute_wale_viscosity`, línea 1884), todos los cálculos se realizan en GPU con CuPy float32. El filtro local Δ = sqrt(dx_local · dy_local) usa los espaciados reales de la malla variable. Se aplica un cap de seguridad ν_t ≤ 100·Δ para evitar inestabilidades en zonas de malla gruesa.

### 2.5 Aerodinámica de perfiles

Un perfil aerodinámico genera fuerzas sobre el fluido por la combinación de la distribución de presión y los esfuerzos viscosos en su superficie. Las fuerzas se descomponen en:
- **Sustentación (Lift L)**: componente perpendicular a la dirección del flujo libre U∞.
- **Resistencia (Drag D)**: componente paralela a U∞.

Los coeficientes adimensionales son:

$$C_L = \frac{2L}{\rho U_\infty^2 c}, \quad C_D = \frac{2D}{\rho U_\infty^2 c}$$

donde c es la longitud de cuerda. La eficiencia aerodinámica es el cociente L/D = C_L/C_D.

La distribución de presión en la superficie se caracteriza mediante el coeficiente de presión:

$$C_p = \frac{p - p_\infty}{0.5\rho U_\infty^2}$$

El extradós (superficie superior) presenta Cp < 0 (succión) mientras el intradós (superficie inferior) presenta Cp > 0, generando la diferencia de presión que produce sustentación.

El ángulo de ataque α es el ángulo entre la cuerda del perfil y el vector de velocidad libre. Para perfiles simétricos como el NACA 0012, C_L ≈ 0 a α = 0° y crece aproximadamente de forma lineal con α hasta el ángulo de pérdida.

### 2.6 Métodos numéricos en CFD

La discretización espacial se realiza mediante diferencias finitas en malla estructurada cartesiana. Las derivadas parciales de primer y segundo orden se aproximan con stencils de 3 puntos centrados (no uniformes por la malla variable):

$$\frac{\partial u}{\partial x}\bigg|_j \approx a_W u_{j-1} + a_C u_j + a_E u_{j+1}$$

$$\frac{\partial^2 u}{\partial x^2}\bigg|_j \approx b_W u_{j-1} + b_C u_j + b_E u_{j+1}$$

Los coeficientes a_W, a_C, a_E (primera derivada) y b_W, b_C, b_E (segunda derivada) se obtienen de la interpolación de Lagrange para malla no uniforme y se precomputan en float64 para preservar exactamente la simetría cuando la malla es simétrica (`calcular_metricas_1d`, línea 105).

**Estabilidad numérica**: la condición CFL (Courant-Friedrichs-Lewy) limita el paso temporal:

$$\Delta t_{adv} \leq \text{CFL} \cdot \frac{\Delta x_{min}}{U_{max}}$$

La estabilidad del esquema explícito de difusión requiere adicionalmente:

$$\Delta t_{visc} \leq C_{visc} \cdot \frac{\Delta x_{min}^2}{\nu_{eff,max}}$$

con C_visc = 0.25. El paso temporal efectivo es el mínimo de ambas restricciones.

Los errores numéricos principales son la difusión numérica (asociada a la advección semi-Lagrangiana de primer orden) y los errores de truncamiento de los stencils. La malla variable mitiga la difusión numérica al concentrar resolución donde los gradientes son altos.

---

## 3. Método

### 3.1 Arquitectura general del simulador

El simulador está implementado en `Simulador2D.py` como una clase `Mesh` y una función `main()`. La organización es la siguiente:

1. **Función `main()`** (línea 4662): punto de entrada que recibe todos los parámetros, genera la malla, carga la geometría del perfil, aplica condiciones de frontera e inicia el bucle temporal.
2. **Clase `Mesh`**: encapsula el dominio computacional, los campos (u, v, p) en GPU, las métricas de la malla, los kernels CUDA compilados y todos los métodos del solver.
3. **Bucle principal** (línea 5249): itera sobre `iteraciones` pasos temporales. En cada paso: recálculo de dt adaptativo → advección → difusión → proyección → almacenamiento de fuerzas cada `guardado` iteraciones.
4. **Postprocesado**: cálculo de fuerzas, visualización, guardado de historial.

### 3.2 Método de splitting de Chorin

El método de splitting de proyección (Chorin 1968) separa el problema en tres subproblemas independientes por paso temporal:

1. **Advección**: u\* = A(u^n), transporte sin gradiente de presión.
2. **Difusión**: u\*\* = D(u\*), adición de viscosidad sin gradiente de presión.
3. **Proyección**: resolver ∇²φ = (ρ/dt)∇·u\*\*, luego u^{n+1} = u\*\* - (dt/ρ)∇φ, con p^{n+1} = φ.

Las ventajas computacionales son: cada subproblema es más fácil de resolver que el sistema acoplado completo, y cada paso admite métodos optimizados (semi-Lagrangiana para advección, explícito para difusión, multigrid para proyección).

En el código, la secuencia por iteración es (líneas 5306-5356):

```python
mesh_gruesa.advect_velocities(dt_use)
mesh_gruesa.apply_boundaries(after_projection=False)
mesh_gruesa.diffuse_velocity(nu, dt_use, usar_wale=mesh_gruesa.usar_wale)
mesh_gruesa.apply_boundaries(after_projection=False)
mg_info = mesh_gruesa.project_multigrid(rho, dt_use, ...)
mesh_gruesa.apply_boundaries(after_projection=True)
```

### 3.3 Discretización espacial y temporal

**Malla de densidad variable**: la función `generar_malla_estirada()` (línea 21) genera vectores 1D de posiciones de nodo con espaciado mínimo dx_min alrededor del perfil y expansión geométrica con factor `factor_expansion` hacia los bordes. La zona fina cubre un ancho configurable (`ancho_zona_fina_x`, `ancho_zona_fina_y`). El espaciado máximo se limita por `dx_max = ratio_max_malla × dx_min`.

Cuando el dominio es simétrico respecto a su punto central (e.g., cy = Ly/2), la mitad derecha se refleja exactamente sobre la izquierda (línea 94-100) para garantizar que los pesos del stencil sean simétricos hasta precisión de máquina float64.

Los campos de velocidad y presión se almacenan en la GPU como arrays CuPy float32 de forma (ny, nx). Las métricas 1D precomputadas (dx_e, dx_w, d1x_W/C/E, d2x_W/C/E, etc.) se calculan en float64 en CPU y se transfieren a float32 en GPU, conservando la simetría exacta en los pesos.

**Paso temporal adaptativo**: en cada iteración se recalcula dt como el mínimo de dt_adv y dt_visc, limitado a 2× el dt nominal para evitar oscilaciones. Si dt < 1e-8 se usa el dt nominal como fallback de seguridad.

### 3.4 Advección semi-Lagrangiana

La advección semi-Lagrangiana (método de backtracing) resuelve:

$$\mathbf{u}^*(\mathbf{x}) = \mathbf{u}^n(\mathbf{x} - \mathbf{u}^n(\mathbf{x}) \cdot \Delta t)$$

Para cada celda (i,j), se calcula la posición de origen: x_dep = x_{ij} - u_{ij}·dt. Se interpolan las velocidades en x_dep mediante interpolación bilineal sobre la malla.

La interpolación usa índices reales (posición en la malla de nodos) calculados con `searchsorted` sobre las posiciones 1D de la malla variable. Los índices se clampean al dominio para implementar la condición de frontera naturalmente.

Las ventajas son la estabilidad incondicional (no hay restricción CFL para la advección) y la buena paralelización en GPU (cada celda es independiente). La limitación principal es la difusión numérica de primer orden (el backtracing introduce interpolación bilineal que actúa como filtrado espacial).

### 3.5 Difusión viscosa

La difusión se resuelve explícitamente mediante Forward Euler con sub-stepping cuando la restricción viscosa lo requiere. Para cada sub-paso de tamaño dt_sub:

$$u^{n+1} = u^n + \nu_{eff} \cdot \Delta t_{sub} \cdot \nabla^2 u^n$$

La viscosidad efectiva es:

$$\nu_{eff} = \nu_{molecular} + \nu_t(x,y)$$

Si WALE está activo, ν_t se calcula antes del paso de difusión (`compute_wale_viscosity`) y se usa como campo espacialmente variable. El Laplaciano se evalúa con el stencil de diferencias finitas no uniforme precomputado. Se aplican las máscaras de sólido (velocidad = 0 en interior sólido) y las condiciones Ghost-Cell IBM tras cada sub-paso.

En el código (`diffuse_velocity`, línea ~1790), el número de sub-pasos se determina como `n_sub = max(1, ceil(dt / dt_visc))`.

### 3.6 Proyección de presión

La ecuación de Poisson para la corrección de presión es:

$$\nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^{**}$$

La solución de esta ecuación y la corrección de velocidades garantizan ∇·u^{n+1} ≈ 0.

**Defect-correction iterativo**: el solver `project_multigrid()` (línea 2300) implementa un bucle externo (defect-correction) de hasta `mg_max_outer` iteraciones. En cada outer se recalcula la divergencia actual, se resuelve el sistema de Poisson incremental y se corrige la velocidad. El criterio de convergencia es:

$$\frac{1}{N_{fluido}} \sum_{i,j \notin \Omega_{solid}} |\nabla \cdot u^n|_{ij} < \text{tol\_div\_rel} \cdot \frac{U_\infty}{L_x}$$

La tolerancia relativa se normaliza por U∞/chord para que sea adimensional e independiente de la escala del problema.

**V-cycle multigrid geométrico**: cada iteración del defect-correction aplica `mg_cycles_per_outer` V-cycles. El V-cycle implementado es:

1. **Pre-suavizado** (`mg_pre_suavizado` sweeps): Red-Black Gauss-Seidel SOR en el nivel fino, alternando el color inicial par/impar en cada sweep para reducir sesgos.
2. **Restricción**: el residuo del nivel fino (r = RHS - L·p) se transfiere al nivel grueso mediante el kernel `restrict_2d` con stencil full-weighting 1-2-1⊗1-2-1 / 16 (operador de restricción simétrico, par de Galerkin con la prolongación bilineal).
3. **Resolución en nivel grueso**: 20 sweeps RB-SOR.
4. **Prolongación**: la corrección del nivel grueso se interpola al nivel fino mediante el kernel `prolongate_add_2d` con bilineal vertex-centered. Los pesos wx/wy están precomputados en float64 para preservar simetría.
5. **Post-suavizado** (`mg_post_suavizado` sweeps): RB-SOR en el nivel fino.

El kernel RB-SOR (`rb_gs_sor_kernel`, línea 506) selecciona celdas de la misma paridad (i+j)%2 == phase en cada lanzamiento, permitiendo actualización in-place sin conflictos de escritura. El factor de sobre-relajación ω se usa en el nivel fino (típicamente 1.5-1.8); en niveles gruesos se usa ω = 1 (Gauss-Seidel puro).

**Coarsening de máscara vertex-centered** (`_coarsen_mask`, línea 904): la máscara de sólidos en la malla gruesa se construye con un stencil 3×3 OR centrado en el vértice del nivel fino correspondiente a cada celda gruesa. Esto preserva la simetría cuando los offsets sx, sy alinean los vértices gruesos con el eje de simetría del dominio (detectados automáticamente por `_detect_sym_offset`).

**Modos de configuración del solver** (línea 4780-4836):
- `mg_modo_turbo`: max_outer=4, cycles=5, pre/post=1, sin guard de residuo, sin IBM por outer. (~4.9 it/s)
- `mg_modo_turbo_hd` (T2_L1): igual que turbo + niveles=1 + div_tol=0.05. (~2.1 it/s)
- `mg_modo_turbo_ultra` (T2_L2): igual que turbo_hd + niveles=2. (~1.4 it/s)

**Protección de residuo del V-cycle** (`guard_residual_every_outer`): si el V-cycle amplifica el residuo de Poisson (||r_después|| > 1.5 × ||r_antes||), se descarta la corrección del V-cycle y se aplican 300 sweeps de GS en el nivel fino como fallback seguro.

### 3.7 Tratamiento de sólidos

**Carga del perfil** (`load_solids_from_file`, línea ~2860): el perfil en formato Selig (.dat) se lee, escala por `chord`, se rota por α alrededor del cuarto de cuerda (convención aeronáutica) y se traslada a la posición (cx, cy). La rasterización sobre la malla usa `matplotlib.path.Path.contains_points` con radio -1e-9 para clasificar las celdas cuyos centros caen dentro del polígono del perfil. Todo el proceso se realiza en float64 para preservar simetría exacta en el caso α = 0°.

El trailing edge demasiado delgado (< 2·dx_min) se recorta automáticamente para garantizar al menos 2 celdas de espesor y estabilidad del IBM.

**Ghost-Cell IBM** (`_precomputar_ghost_cell`, línea 1178): las celdas sólidas adyacentes a fluido (ghost cells) se identifican. Para cada ghost cell (g_row, g_col), se calcula el punto imagen simétrico respecto a la pared del perfil, ubicado en el fluido. En cada iteración, el kernel `ghost_cell_bc_kernel` (línea 669) interpola bilinealmente la velocidad en el punto imagen y asigna u_ghost = -u_imagen, v_ghost = -v_imagen. Esto impone condición de no deslizamiento (no-slip) en la superficie del perfil con precisión de interpolación bilineal.

**Refuerzo de impermeabilidad** (`reforzar_impermeabilidad`, línea 1151): en la capa de fluido adyacente al sólido, se calcula la componente normal de velocidad respecto a la normal de la pared (obtenida del campo de distancia firmada mediante `distance_transform_edt`) y se anula: u ← u - (u·n̂)n̂. Esto garantiza la condición de no penetración incluso si el Ghost-Cell IBM introduce pequeños errores de interpolación.

### 3.8 Cálculo de fuerzas aerodinámicas

El método `compute_surface_forces_definitive()` (línea 3917) integra los esfuerzos en la superficie del perfil:

**Presión en pared**: se interpola la presión en n_extrap_layers capas (separadas 1 celda cada una) desde la superficie hacia el fluido, y se extrapola a la pared (x = 0) mediante regresión lineal por mínimos cuadrados sobre todas las capas. Esto es más robusto que la interpolación directa de un solo punto cuando hay gradientes de presión fuertes en el TE.

**Corrección de presión de fondo** (`correct_pressure_offset`): se ajusta un plano afín p_bg = a + b·x + c·y a los valores de presión en las celdas del borde del dominio (far-field), y se resta de p_wall. Adicionalmente, se elimina el offset residual promedio ponderado sobre el contorno. Esto elimina el sesgo espurio de lift que aparece cuando la presión tiene un offset global no nulo y el contorno discreto no cierra exactamente.

**Esfuerzo viscoso**: se calculan los gradientes del tensor de deformación (S_ij = (∂u_i/∂x_j + ∂u_j/∂x_i)/2) en la superficie mediante stencils no uniformes, y se samplea la viscosidad efectiva (molecular + turbulenta WALE) en la cara. La tracción viscosa es T_v = μ_eff · 2·S·n.

**Integración**: las fuerzas se obtienen por integración sobre la frontera del sólido:

$$F_x = \int_\partial\Omega (-p\,n_x + T_x^v)\,dS, \quad F_y = \int_\partial\Omega (-p\,n_y + T_y^v)\,dS$$

donde dS es el elemento de arco local calculado con las métricas de la malla variable.

**Transformación a Drag/Lift**: las fuerzas en el sistema de coordenadas de la malla (Fx, Fy) se transforman al sistema aerodinámico mediante la rotación por α (línea 3900-3915).

**Coeficientes adimensionales** (líneas 4507-4510):

$$C_d = \frac{2\,D}{\rho\,U_\infty^2\,c}, \quad C_l = \frac{2\,L}{\rho\,U_\infty^2\,c}$$

Los coeficientes se guardan en `cdvector` y `clvector` (arrays CuPy de longitud `iteraciones // guardado`) cada `guardado` iteraciones.

### 3.9 Implementación en GPU

Toda la computación intensiva se realiza en GPU mediante CuPy. Se distinguen tres tipos de kernels:

**ElementwiseKernel**: aplicado a operaciones sobre todos los nodos (e.g., `_jacobi_update_masked`, `_laplacian_kernel_masked`, `_precond_jacobi_kernel_masked`). Cada hilo procesa un nodo; la indexación (i, j) se obtiene de la posición lineal `idx = i * nx + j`. Los vecinos sólidos se tratan con condición de Neumann homogéneo (se acumula el coeficiente al diagonal).

**RawKernel (CUDA C)**: para operaciones que requieren control más fino. Los kernels implementados son:
- `rb_gs_sor` (línea 506): Red-Black SOR in-place. Filtra por paridad `(i+j)%2 == phase`, actualiza ω·p_GS + (1-ω)·p_old.
- `restrict_2d` (línea 553): restricción full-weighting. Stencil 1-2-1⊗1-2-1/16 con clamp en bordes.
- `prolongate_add_2d` (línea 590): prolongación bilineal += sobre el nivel fino. Usa wx/wy precomputados para interpolación física correcta en malla no uniforme.
- `divergence_masked` (línea 630): divergencia con stencil adaptativo: si algún vecino es sólido, no contribuye esa dirección (evita gradientes explosivos en la interfaz).
- `ghost_cell_bc_kernel` (línea 669): asignación Ghost-Cell, u_ghost = -u_imagen con interpolación bilineal.
- `velocity_correction_kernel` (línea 719): u -= (dt/ρ)·∇p. Usa la misma lógica de vecinos sólidos que la divergencia, garantizando consistencia discreta D·D^T.
- `adjoint_gradient_kernel` (línea 766): gradiente adjunto exacto D^T·p, transpuesto del operador divergencia. Garantiza que el operador DD^T sea simétrico definido positivo.
- `zero_solid_kernel` (línea 707): pone u = v = 0 en celdas sólidas interiores.
- `spread` (kernel IBM de Peskin, línea 359): spreading de fuerzas Lagrangianas a la malla Euleriana con función delta de Peskin (soporte ±2 celdas), usando `atomicAdd` para correcta acumulación paralela.

**Paralelización**: todos los kernels se lanzan con bloques de 256 hilos. La dimensión del grid es `ceil(total_cells / 256)`. Las operaciones de reducción (sum, max, norm) usan las primitivas CuPy que internamente emplean reducción paralela eficiente en GPU.

**Transferencias CPU-GPU**: se minimizan transfiriendo solo los valores escalares (Cl, Cd, div_mean, div_max) tras cada `guardado` iteraciones. Las métricas de malla (d2x, d2y, wx, wy, etc.) se transfieren una sola vez durante la inicialización.

**Warmup JIT**: antes del bucle principal se fuerza la compilación JIT de todos los kernels CuPy (línea 5229-5243). Esto evita que la primera iteración tarde 30-90 segundos en compilar.

### 3.10 Optimización geométrica

#### 3.10.1 Algoritmo genético

El optimizador genético (`RunGA.py`) implementa el ciclo evolutivo clásico sobre una población de perfiles aerodinámicos. La configuración base usa:
- `poblacion_tamano = 12` individuos por generación
- `elites = 4` individuos preservados intactos
- `torneo_tamano = 4` para selección de padres

**Ciclo por generación:**
1. Evaluar fitness de todos los individuos de la generación actual (simulación CFD).
2. Seleccionar los `elites` mejores (elitismo).
3. Para el resto de la población: selección por torneo (4 candidatos aleatorios, gana el mejor), cruce (average de coordenadas Y de dos padres), mutación paramétrica.
4. Aplicar filtro IA si está activo.
5. Registrar resultados y avanzar generación.

#### 3.10.2 Parametrización del perfil

Las coordenadas del perfil se representan como puntos (x_i, y_i) en formato Selig. La mutación opera en el espacio de camber c(x) y espesor t(x), que son funciones más físicamente significativas que las coordenadas directas:

$$c(x) = \frac{y_{upper}(x) + y_{lower}(x)}{2}, \quad t(x) = y_{upper}(x) - y_{lower}(x)$$

La perturbación se aplica mediante funciones suaves controladas por nodos de control: una perturbación en el nodo k se interpola a todos los puntos del perfil con un kernel suave. Parámetros:
- `camber_mut_std = 0.0035`: desviación estándar de la perturbación de camber.
- `espesor_mut_std = 0.0045`: desviación estándar de la perturbación de espesor.
- `n_control_mutacion = 7`: nodos de control para las perturbaciones.

**Restricciones geométricas preservadas:**
- Leading edge: posición (x, y) fija; el radio mínimo se controla por `le_radio_factor_min = 0.40` relativo al perfil base.
- Trailing edge: espesor mínimo absoluto de `te_espesor_min_absoluto = 3e-4` y relativo `[0.65, 3.50]` × base.
- Espesor mínimo global: `espesor_min_global = 2e-4` en toda la cuerda (excepto LE).
- Protección de zona LE: mutación atenuada en x ∈ [0, 0.06·chord].

Si la geometría generada viola alguna restricción, se reintenta hasta `max_intentos_geometria = 120` veces (con fallback a mutación tipo "legacy" con bumps gaussianos).

#### 3.10.3 Evaluación CFD como fitness

Cada individuo se evalúa mediante `Simulador2D.main()` con los parámetros configurados en `CONFIG['sim_extra_params']`. La simulación corre `simulacion_iteraciones = 2000` pasos con la malla variable y el modo multigrid turbo activo.

El fitness es la eficiencia aerodinámica media L/D calculada sobre el último 70% de las iteraciones (descartando el transitorio). Si `multi_angulo = True`, se evalúa a (α - δ, α, α + δ) y se combina el fitness según el modo configurado (mean, min, o weighted).

#### 3.10.4 Filtro predictivo mediante IA

La clase `OraculoAerodinamico` (`RunGA.py`, línea 243) implementa un RandomForestRegressor de scikit-learn (100 árboles, profundidad máxima 15) que predice el fitness a partir de las coordenadas Y del perfil (features compactas).

**Flujo de trabajo:**
1. Si hay al menos 20 evaluaciones previas, el modelo está entrenado y puede predecir.
2. Para cada nuevo individuo antes de la simulación, se predice el fitness.
3. Si la predicción < `umbral_calidad × max_fitness_histórico` (umbral dinámico 0.70), el individuo se descarta sin simular.
4. Todos los individuos simulados (aprobados o no por el filtro) se registran en el DataLogger para enriquecer el entrenamiento futuro.

La memoria del modelo persiste en disco (`cerebro_aerodinamico.pkl`) entre ejecuciones del GA mediante pickle, acumulando experiencia de múltiples sesiones de optimización.

---

## 4. Resultados

### 4.1 Validación del solver

La divergencia ∇·u se monitoriza en cada paso y se normaliza por U∞/chord para obtener una métrica adimensional. Se registra tanto la media (L1) como el máximo (L∞) en `divvector` y `divvector_max`.

Las zonas de referencia establecidas en el visualizador (`plot_divergence_history`, línea 3663) son:
- **< 0.01**: bien resuelto (solver converge suficientemente).
- **0.01 – 0.10**: aceptable para estudios de diseño.
- **> 0.10**: solver insuficiente para ese paso temporal.

El criterio de tolerancia por defecto (`divergencia = 0.10`) corresponde al umbral aceptable. Los modos turbo-hd y turbo-ultra usan `divergencia = 0.05` para mayor precisión.

La estabilidad temporal se garantiza mediante el clamp de velocidades a 50·U_ref y de presión a 100·ρ·U_ref² + 1000, con detección de NaN y rollback automático cuando el V-cycle diverge (ratio ||r_after||/||r_before|| > 1.5).

### 4.2 Resultados fluidodinámicos

Los campos de velocidad y presión se visualizan mediante `visualize_velocity()` (pcolormesh |u|) y `visualize_velocity_vectors()` (quiver con submuestreo). La vorticidad se puede calcular como ω_z = ∂v/∂x - ∂u/∂y usando los stencils d1x y d1y.

Para Re ~ 100.000 (ν = 1e-5, U = 1, chord = 1), el flujo sobre el NACA 0012 a α = 10° muestra la formación de la estela de von Kármán con desprendimiento periódico de vórtices, capturada por el solver LES.

### 4.3 Resultados aerodinámicos

Los coeficientes Cl y Cd se calculan cada `guardado` iteraciones y se almacenan en cdvector/clvector. El método `plot_forces_over_time()` muestra la evolución temporal con medias acumuladas, detección de picos y ajuste lineal de tendencia sobre máximos y mínimos.

Para el NACA 0012 a Re = 100.000 y α = 10° con `mg_modo_turbo=True`, los valores típicos obtenidos son Cl ≈ 0.41 y Cd en el rango 0.02-0.05, con razón L/D del orden de 10-15. Los valores dependen de la convergencia de la divergencia.

La distribución de Cp se calcula mediante `compute_surface_forces_definitive()` con `return_cp=True` y se visualiza con `plot_cp_vs_chord()`, separando extradós e intradós.

### 4.4 Rendimiento computacional

El rendimiento medido en benchmarks (`benchmark_mg_params.py`) sobre RTX 3070 Ti:
- **Turbo (L0)**: ~4.9 it/s para malla típica ~600×400 nodos con dx_min=0.001.
- **Turbo-HD (L1, 1 nivel MG)**: ~2.1 it/s (mayor calidad de divergencia).
- **Turbo-Ultra (L2, 2 niveles MG)**: ~1.4 it/s (mejor convergencia en cada paso).

El cuello de botella es el ancho de banda de memoria GPU, no la capacidad de cómputo. Los kernels multigrid y de advección acceden a arrays 2D en patrones de acceso no coalesced (vecinos en y implican saltos de nx elementos), lo que limita el rendimiento respecto al pico teórico.

El paso de difusión es el más rápido (vectorizado, sin condiciones). La proyección multigrid consume típicamente 60-80% del tiempo total de iteración debido a los múltiples lanzamientos de kernel por V-cycle.

El impacto del modelo WALE es moderado (~10-15% de overhead) ya que el cálculo de ν_t requiere 4 gradientes de velocidad y varias operaciones tensoriales, pero es un único paso por iteración.

### 4.5 Resultados de optimización

El GA parte de un perfil base y evoluciona hacia geometrías con mayor L/D. El fitness aumenta generación a generación con algunos retrocesos por la variabilidad estocástica de las simulaciones transitorias. Los cambios geométricos observados típicamente incluyen aumento de camber (mayor curvatura) y ajuste del TE para mejorar la separación del flujo.

El filtro IA reduce el coste computacional al descartar individuos con predicción de L/D baja sin simularlos. Tras varias sesiones acumulando datos, el RandomForest aprende la relación geometría→fitness con suficiente fidelidad para actuar como surrogate efectivo.

---

## 5. Discusión

### 5.1 Limitaciones físicas

La aproximación bidimensional excluye fenómenos tridimensionales relevantes: estiramiento de vórtices, inestabilidades de Kelvin-Helmholtz en 3D, estructuras coherentes del boundary layer 3D. En geometrías de ala finita, los efectos de borde (vórtices de punta de ala) son completamente ignorados.

La hipótesis de incompresibilidad limita la aplicabilidad a Mach < 0.3. Para velocidades más altas, los efectos compresibles modifican significativamente la distribución de Cp y la ubicación del choque.

El modelo WALE, como todos los modelos LES, requiere que las escalas energéticas estén resueltas. En la capa límite y en la estela cercana al TE, donde los gradientes son máximos, la resolución de malla puede ser insuficiente para una simulación LES rigurosa.

### 5.2 Limitaciones numéricas

La advección semi-Lagrangiana introduce difusión numérica de primer orden proporcional a U·Δx. Esta difusión efectiva añade una contribución a la viscosidad que puede ser comparable o mayor que ν_molecular para mallas gruesas, contaminando el campo de ν_t del modelo WALE.

El esquema explícito de difusión impone la restricción C_visc·Δx²/ν_eff, que se vuelve restrictiva cuando ν_t es grande (flujo turbulento intenso) o la malla es fina. Esto se mitiga con sub-stepping automático pero aumenta el coste.

La sensibilidad a la resolución de malla es significativa: los coeficientes Cl y Cd pueden variar 5-20% entre dx_min = 0.002 y dx_min = 0.0005 para el mismo α y Re. El estudio de convergencia de malla es necesario antes de usar los resultados para diseño.

### 5.3 Coste computacional

El balance entre precisión y velocidad está dominado por el solver de proyección. Aumentar `mg_cycles_per_outer` o `mg_max_outer` reduce la divergencia residual pero disminuye las iteraciones por segundo. Los modos turbo ofrecen el mejor compromiso para exploración de diseños (alta velocidad, divergencia aceptable). Los modos HD y Ultra son preferibles para validación final.

El coste de LES (modelo WALE) es bajo (~10-15%) comparado con la proyección. La advección es el segundo componente más costoso. El cálculo de fuerzas (extrapolación de presión en 5 capas, gradientes de velocidad) es significativo pero se ejecuta solo cada `guardado` iteraciones.

El beneficio real de la GPU frente a CPU es de 20-50× para mallas de 200k-500k celdas, dependiendo de la GPU y los patrones de acceso a memoria. El cuello de botella es el ancho de banda (no la capacidad aritmética), por lo que GPUs con mayor bandwidth (A100, RTX 4090) escalan mejor.

### 5.4 Calidad de la optimización

El GA es sensible a los parámetros evolutivos, especialmente al balance entre exploración (amplitud de mutación) y explotación (elitismo). Con `camber_mut_std = 0.0035` y `espesor_mut_std = 0.0045` (determinados por barrido de parámetros), la convergencia es razonable en 5-10 generaciones.

El riesgo de convergencia prematura es real con poblaciones pequeñas (12 individuos). El elitismo fuerte (4/12 = 33%) puede reducir la diversidad genética y atrapar el GA en óptimos locales.

El filtro IA introuce sesgo si el modelo RF no está bien entrenado: puede descartar incorrectamente buenas geometrías si son muy diferentes del dataset de entrenamiento. El umbral dinámico (0.70 × max histórico) mitiga este riesgo al ser conservador.

### 5.5 Posibles mejoras

- **Esquemas de mayor orden**: advección de segundo orden (McCormack, Lax-Wendroff) o Runge-Kutta temporal reducirían la difusión numérica significativamente.
- **Métodos menos difusivos**: advección QUICK o ENO/WENO proporcionarían mayor precisión sin aumentar el dominio de dependencia.
- **Malla adaptativa dinámica (AMR)**: refinar automáticamente en zonas de gradiente elevado (TE, capa límite) y desrefinar en el far-field reduciría el número de celdas para igual precisión.
- **Extensión a simulaciones 3D**: el núcleo GPU es reutilizable; la extensión principal sería la gestión de memoria (arrays 3D) y la parametrización 3D del perfil (ala finita).
- **Modelos turbulentos más avanzados**: Dynamic Smagorinsky (ajusta C_s localmente), o modelos σ (mejor comportamiento en flujo rotacional puro).
- **Acoplamiento directo CFD + ML**: reemplazar completamente el solver CFD por una red neuronal de sustitución entrenada sobre datos CFD, con coste de inferencia órdenes de magnitud menor.
- **Optimización multiobjetivo**: optimizar simultáneamente Cl, Cd/Cl, y robustez ante cambios de α mediante algoritmos NSGA-II o MOEA/D.
- **Redes neuronales sustitutas**: graph neural networks o physics-informed neural networks entrenadas sobre el dataset acumulado por el DataLogger para reemplazar el RandomForest por un surrogate más preciso.

### 5.6 Aplicabilidad práctica

El simulador es una herramienta funcional para:
- **Diseño preliminar de perfiles**: explorar rápidamente el espacio de diseño antes de pasar a herramientas de alta fidelidad (RANS 3D, experimentos).
- **Herramienta educativa**: visualización interactiva de fenómenos aerodinámicos (capa límite, estela, Cp) con comprensión directa del código numérico.
- **Base para entornos de optimización más avanzados**: el acoplamiento GA + CFD es extensible a multifidelidad (CFD grueso para cribar, CFD fino para validar) o a algoritmos bayesianos.
- **Generación de datos de entrenamiento para ML**: el DataLogger acumula (geometría, condiciones, Cl/Cd) en formato JSONL, disponible para entrenar modelos de predicción aerodinámica.

---

## 6. Conclusiones

### 6.1 Resumen del trabajo realizado

Se ha desarrollado un simulador CFD 2D transitorio completo que incluye: generación de malla variable con refinamiento adaptativo alrededor del perfil, advección semi-Lagrangiana, difusión viscosa con sub-stepping, solver de proyección multigrid geométrico con defect-correction, modelo turbulento WALE, Immersed Boundary Method Ghost-Cell, cálculo de fuerzas aerodinámicas por integración de esfuerzos con corrección de presión de fondo, y un sistema de visualización interactivo con memoria compartida GPU-CPU.

El optimizador genético acopla el simulador con un algoritmo evolutivo que opera en el espacio de camber y espesor, con restricciones geométricas físicamente motivadas y un filtro predictivo Random Forest que reduce el número de simulaciones CFD necesarias.

### 6.2 Principales aportaciones

- **Solver transitorio acelerado por GPU**: implementación completa en CuPy con kernels CUDA personalizados. Rendimiento de ~4.9 it/s en malla ~240k celdas sobre RTX 3070 Ti.
- **Implementación multigrid y defect-correction**: V-cycles con smoother RB-SOR, restricción full-weighting, prolongación bilineal vertex-centered. Coarsening de máscara IBM con preservación de simetría. Protección automática contra divergencia del V-cycle.
- **Integración CFD + optimización genética**: mutación paramétrica (camber/espesor) con restricciones geométricas físicas, elitismo, selección por torneo. Evaluación multi-ángulo opcional.
- **Uso de IA para reducción de coste**: filtro Random Forest persistente que aprende geometría→fitness y descarta individuos poco prometedores sin simularlos, con umbral dinámico adaptativo.

### 6.3 Trabajo futuro

- Extensión del solver a 3D, aprovechando la arquitectura GPU existente.
- Mejora del modelo físico con esquemas de advección de mayor orden (ENO/WENO) y modelos turbulentos más avanzados (dynamic WALE, σ-model).
- Refinamiento adaptativo de malla (AMR) guiado por estimadores de error locales.
- Integración con modelos ML avanzados (graph neural networks, physics-informed NN) entrenados sobre el dataset acumulado por el DataLogger.
- Validación experimental comparando Cl/Cd simulados con datos de túnel de viento para perfiles estándar (NACA 0012, NACA 4412) a distintos Reynolds.

---

*Código fuente: `Simulador2D.py` (solver CFD + kernels GPU), `RunGA.py` (optimizador genético + filtro IA). Todos los resultados referenciados son reproducibles ejecutando `main()` con los parámetros indicados.*

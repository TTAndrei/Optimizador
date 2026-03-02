"""
RunGA.py - Optimizador Genético de Perfiles Aerodinámicos
=========================================================

Optimiza la geometría de perfiles aerodinámicos usando:
  - Algoritmo genético con mutación por bumps gaussianos suaves
  - Simulación CFD 2D (Simulador2D.py) como evaluador de fitness
  - Evaluación multi-ángulo opcional (alpha ± delta)
  - Filtro IA (Random Forest) para descartar diseños malos sin simularlos
  - Logging completo en JSONL para entrenamiento de modelos ML futuros
  - Parada segura con Ctrl+C (guarda progreso)

Restricciones geométricas:
  - Leading edge: posición fija (x,y no cambian)
  - Trailing edge: coordenada Y libre (permite mayor curvatura/camber)
  - Puntos interiores: solo Y se modifica (X preservado)

Uso:
    python RunGA.py

Controles:
    Ctrl+C  ->  Parada segura (termina generación actual y guarda todo)
"""

import numpy as np
import random
import os
import copy
import signal
import sys
import json
import pickle
import time
import gc
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import cupy as cp
from sklearn.ensemble import RandomForestRegressor

import Simulador2D

# ==========================================
# 1. CONFIGURACIÓN DEL SISTEMA
# ==========================================
CONFIG = {
    # --- Archivos ---
    'archivo_base': 'AG24',                             # Perfil base (formato Selig)
    'archivo_memoria_ia': 'cerebro_aerodinamico.pkl',   # Memoria persistente del filtro IA
    'archivo_datos_ml': 'aprendizaje_ML.jsonl',         # Datos acumulados para ML futuro
    'directorio_resultados': 'resultados_ga',           # Carpeta de salida

    # --- Parámetros Evolutivos ---
    'poblacion_tamano': 10,        # Individuos por generación
    'generaciones': 5,            # Máximo de generaciones (parable con Ctrl+C)
    'elites': 4,                   # Individuos preservados intactos por elitismo
    'torneo_tamano': 4,            # Tamaño del torneo de selección de padres

    # --- Mutación (Bumps Gaussianos) ---
    #   Cada bump es una campana de Gauss centrada en un punto aleatorio
    #   que desplaza Y de los puntos cercanos. Produce deformaciones suaves.
    'prob_mutacion': 0.85,              # Probabilidad de mutar un hijo
    'n_bumps_min': 1,                   # Mínimo de bumps por mutación
    'n_bumps_max': 4,                   # Máximo de bumps por mutación
    'sigma_bump_min': 0.03,             # Anchura mín del bump (fracción de nº puntos)
    'sigma_bump_max': 0.15,             # Anchura máx del bump
    'amplitud_mutacion': 0.008,         # Desviación estándar del desplazamiento en Y
    'prob_mutar_trailing_edge': 0.3,    # Probabilidad de mover el trailing edge
    'amplitud_trailing_edge': 0.002,    # Desviación estándar del TE en Y

    # --- Simulación CFD ---
    'simulacion_iteraciones': 3000,     # Iteraciones por simulación CFD
    'v0x': 5.0,                         # Velocidad del flujo libre (m/s)
    'alpha_deg': 5.0,                   # Ángulo de ataque base (grados)
    'chord': 1.0,                       # Longitud de cuerda (m)
    'dx_fino': 0.001,                   # Resolución malla fina (m)
    'dx_grueso': 0.004,                  # Resolución malla gruesa (m)
    'CFL': 0.8,                         # Número de Courant
    'rho': 1.225,                       # Densidad del aire (kg/m³)
    'nu': 1.5e-5,                       # Viscosidad cinemática (m²/s)

    # --- Multi-ángulo (opcional) ---
    #   Si activo, cada perfil se simula a alpha-delta, alpha, alpha+delta.
    #   Evalúa robustez del perfil a variaciones del ángulo de ataque.
    #   Poner multi_angulo=False para optimizar solo al ángulo base.
    'multi_angulo': False,               # True = evaluar alpha ± delta_angulo
    'delta_angulo': 1.0,                # ± grados alrededor de alpha_deg
    'fitness_modo': 'mean',             # 'mean' | 'min' | 'weighted'
    'peso_angulo_base': 2.0,            # Peso extra para ángulo base (modo 'weighted')

    # --- IA (Filtro predictivo) ---
    'usar_ia': True,                    # Activar filtro IA (Random Forest)
    'umbral_calidad': 0.7,              # Umbral dinámico (fracción del top histórico)

    # --- Suavizado ---
    'suavizado_iteraciones': 2,         # Pasadas de suavizado laplaciano post-mutación

    # --- Parámetros extra del simulador (opcionales) ---
    #   Dict con parámetros adicionales para Simulador2D.main()
    #   Ej: {'Lx': 12, 'Ly': 8, 'usar_wale': True}
    'sim_extra_params': {'divergencia': 1e-1},
}


# ==========================================
# 2. CONTROL DE PARADA SEGURA
# ==========================================
PARADA_SOLICITADA = False


def manejar_parada(signum, frame):
    """Captura Ctrl+C para parada segura"""
    global PARADA_SOLICITADA
    print("\n\n" + "!" * 60)
    print(">>> PARADA SOLICITADA (Ctrl+C)")
    print(">>> Terminando evaluación actual y guardando progreso...")
    print(">>> NO cierre la ventana a la fuerza.")
    print("!" * 60 + "\n")
    PARADA_SOLICITADA = True


# ==========================================
# 3. DATA LOGGER PARA ML
# ==========================================
class DataLoggerML:
    """
    Registra cada evaluación CFD exitosa en formato JSONL (JSON Lines).

    Cada línea del archivo es un JSON independiente con:
      - Geometría completa del perfil (puntos x,y)
      - Condiciones de vuelo (velocidad, ángulo, Reynolds, etc.)
      - Resultados aerodinámicos (Cl, Cd, L/D por cada ángulo evaluado)
      - Metadata (generación, fitness, timestamp)

    Este archivo alimentará modelos ML que, dados parámetros deseados
    (velocidad, ángulo, Cl/Cd objetivo), recomienden una geometría de perfil.

    El formato JSONL permite:
      - Append eficiente (no hay que reescribir todo el archivo)
      - Lectura línea a línea (memoria eficiente para datasets grandes)
      - Acumulación entre múltiples ejecuciones del GA
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.count = 0

        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                self.count = sum(1 for line in f if line.strip())
            print(f" [ML] Datos existentes: {self.count} registros en '{filepath}'")
        else:
            print(f" [ML] Nuevo archivo de datos: {filepath}")

    def registrar(self, perfil_puntos, condiciones, resultados_por_angulo, metadata=None):
        """
        Añade un registro de evaluación al archivo JSONL.

        Args:
            perfil_puntos: np.array (N, 2) con coordenadas del perfil
            condiciones: dict con v0x, alpha_base, Re, chord, etc.
            resultados_por_angulo: dict {"5.0": {"cl": ..., "cd": ..., "ld": ...}}
            metadata: dict opcional con generación, individuo, fitness, etc.
        """
        registro = {
            'timestamp': datetime.now().isoformat(),
            'id': self.count,
            'perfil': (perfil_puntos.tolist()
                       if isinstance(perfil_puntos, np.ndarray)
                       else perfil_puntos),
            'n_puntos': len(perfil_puntos),
            'condiciones': condiciones,
            'resultados': resultados_por_angulo,
            'metadata': metadata or {}
        }

        try:
            with open(self.filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(registro, ensure_ascii=False) + '\n')
            self.count += 1
        except Exception as e:
            print(f" [ML] Error registrando datos: {e}")

    def cargar_todos(self):
        """Carga todos los registros como lista de dicts (para entrenamiento ML)"""
        registros = []
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r', encoding='utf-8') as f:
                for linea in f:
                    linea = linea.strip()
                    if linea:
                        try:
                            registros.append(json.loads(linea))
                        except json.JSONDecodeError:
                            continue
        return registros


# ==========================================
# 4. ORÁCULO AERODINÁMICO (FILTRO IA)
# ==========================================
class OraculoAerodinamico:
    """
    Filtro inteligente que predice el fitness de un perfil antes de simularlo.

    Usa un Random Forest entrenado con coordenadas Y del perfil como features
    y el fitness (L/D) como target. Descarta diseños que predice como malos,
    ahorrando tiempo de simulación CFD.

    La memoria persiste entre ejecuciones (pickle), permitiendo que la IA
    mejore progresivamente con cada corrida del GA.
    """

    def __init__(self, archivo_memoria):
        self.archivo = archivo_memoria
        self.modelo = RandomForestRegressor(
            n_estimators=100, max_depth=15,
            n_jobs=-1, random_state=42
        )
        self.entrenado = False
        self.datos_X = []   # Coordenadas Y del perfil (features compactas)
        self.datos_y = []   # Fitness L/D (target)
        self.conteo_descartes = 0
        self.cargar_memoria()

    def cargar_memoria(self):
        """Carga experiencia previa desde disco"""
        if os.path.exists(self.archivo):
            try:
                print(f" [IA] Cargando memoria desde '{self.archivo}'...")
                with open(self.archivo, 'rb') as f:
                    memoria = pickle.load(f)
                    self.datos_X = memoria.get('X', [])
                    self.datos_y = memoria.get('y', [])

                print(f" [IA] Memoria recuperada: {len(self.datos_X)} experiencias")
                if len(self.datos_X) >= 20:
                    self.modelo.fit(self.datos_X, self.datos_y)
                    self.entrenado = True
                    print(" [IA] Modelo re-entrenado y listo.")
            except Exception as e:
                print(f" [IA] Error cargando memoria (reiniciando): {e}")
                self.datos_X = []
                self.datos_y = []
        else:
            print(" [IA] Sin memoria previa. Aprendizaje desde cero.")

    def guardar_memoria(self):
        """Persiste experiencia acumulada a disco"""
        try:
            with open(self.archivo, 'wb') as f:
                pickle.dump({'X': self.datos_X, 'y': self.datos_y}, f)
        except Exception as e:
            print(f" [IA] Error guardando memoria: {e}")

    def aprender(self, poblacion_evaluada):
        """Incorpora resultados de una generación para mejorar predicciones"""
        nuevos = [
            (ind.genes[:, 1].tolist(), ind.fitness)
            for ind in poblacion_evaluada if ind.fitness > 0
        ]

        if not nuevos:
            return

        for x, y in nuevos:
            self.datos_X.append(x)
            self.datos_y.append(y)

        if len(self.datos_X) >= 20:
            try:
                self.modelo.fit(self.datos_X, self.datos_y)
                self.entrenado = True
            except Exception as e:
                print(f" [IA] Error entrenando modelo: {e}")

        self.guardar_memoria()

    def predecir_si_vale_la_pena(self, genes):
        """
        Retorna True si el diseño merece ser simulado.
        Compara la predicción con un umbral dinámico basado en la mitad
        superior del historial de fitness.
        """
        if not self.entrenado or not CONFIG['usar_ia']:
            return True

        try:
            y_coords = genes[:, 1].tolist()
            prediccion = self.modelo.predict([y_coords])[0]

            if self.datos_y:
                n_top = max(1, len(self.datos_y) // 2)
                top_historico = np.mean(sorted(self.datos_y, reverse=True)[:n_top])
                limite = top_historico * CONFIG['umbral_calidad']

                if prediccion >= limite:
                    return True
                else:
                    self.conteo_descartes += 1
                    return False
        except Exception:
            return True  # En caso de error, simular por seguridad

        return True


# ==========================================
# 5. GESTIÓN DE PERFILES AERODINÁMICOS
# ==========================================
def cargar_perfil(filepath):
    """
    Carga un perfil aerodinámico en formato Selig.

    Formato esperado:
        Línea 1: Nombre/header del perfil
        Líneas 2+: coordenadas  x  y
        Orden: TE superior -> LE -> TE inferior

    Returns:
        (puntos, header, le_idx, te_indices)
        - puntos: np.array (N, 2)
        - header: str (primera línea del archivo)
        - le_idx: int (índice del leading edge = min x)
        - te_indices: list[int] (índices del trailing edge)
    """
    puntos = []
    header = ""

    if not os.path.exists(filepath):
        print(f" ERROR: Archivo no encontrado: {filepath}")
        return np.array([]), "", -1, []

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if not lines:
            return np.array([]), "", -1, []
        header = lines[0]
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    puntos.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue

    puntos = np.array(puntos)

    if len(puntos) < 3:
        print(" ERROR: Perfil tiene menos de 3 puntos.")
        return puntos, header, -1, []

    # Leading edge = punto con menor coordenada X
    le_idx = int(np.argmin(puntos[:, 0]))

    # Trailing edge = primer y último punto (cerca de x ≈ 1.0 en formato Selig)
    te_indices = []
    if puntos[0, 0] > 0.8:
        te_indices.append(0)
    if puntos[-1, 0] > 0.8:
        te_indices.append(len(puntos) - 1)

    print(f" Perfil cargado: {os.path.basename(filepath)} ({len(puntos)} puntos)")
    print(f"   Leading edge:  idx={le_idx} "
          f"({puntos[le_idx, 0]:.6f}, {puntos[le_idx, 1]:.6f}) [FIJO]")
    for ti in te_indices:
        print(f"   Trailing edge: idx={ti} "
              f"({puntos[ti, 0]:.6f}, {puntos[ti, 1]:.6f}) [Y libre]")

    return puntos, header, le_idx, te_indices


def guardar_perfil(filepath, puntos, header):
    """Guarda perfil en formato Selig estándar"""
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(header)
        for p in puntos:
            f.write(f"  {p[0]:.6f}  {p[1]:.6f}\n")


def suavizar_perfil(puntos, le_idx, n_pasadas=2):
    """
    Suavizado laplaciano de coordenadas Y.
    Preserva leading edge, primer punto y último punto.
    Filtro: Y_i = (Y_{i-1} + 2*Y_i + Y_{i+1}) / 4
    """
    resultado = puntos.copy()
    n = len(puntos)

    for _ in range(n_pasadas):
        y_nuevo = resultado[:, 1].copy()
        for i in range(1, n - 1):
            if i == le_idx:
                continue
            y_nuevo[i] = (resultado[i - 1, 1]
                          + 2 * resultado[i, 1]
                          + resultado[i + 1, 1]) / 4.0
        resultado[:, 1] = y_nuevo

    return resultado


# ==========================================
# 6. INDIVIDUO
# ==========================================
class Individuo:
    """Representa un perfil aerodinámico candidato en la población"""

    def __init__(self, genes, header):
        self.genes = genes.copy()
        self.header = header
        self.fitness = 0.0
        self.resultados = {}  # {"5.0": {"cl": float, "cd": float, "ld": float}}

    def __repr__(self):
        return f"Ind(fitness={self.fitness:.4f})"


# ==========================================
# 7. OPERADORES GENÉTICOS
# ==========================================
def mutar_perfil(genes, le_idx, te_indices):
    """
    Mutación suave por superposición de bumps gaussianos.

    Cada bump es una campana de Gauss centrada en un punto aleatorio del perfil
    que desplaza las coordenadas Y de los puntos cercanos. Esto produce
    deformaciones suaves y físicamente plausibles.

    Restricciones:
      - Leading edge (le_idx): completamente fijo (x,y no cambian)
      - Trailing edge (te_indices): Y puede variar opcionalmente
      - Todos los demás puntos: solo Y se modifica

    Args:
        genes: np.array (N, 2) con coordenadas del perfil
        le_idx: índice del leading edge (protegido)
        te_indices: lista de índices del trailing edge
    """
    resultado = genes.copy()
    n = len(genes)

    n_bumps = random.randint(CONFIG['n_bumps_min'], CONFIG['n_bumps_max'])

    for _ in range(n_bumps):
        # Centro del bump: índice aleatorio, evitando el leading edge
        centro = random.randint(0, n - 1)
        intentos = 0
        while centro == le_idx and intentos < 20:
            centro = random.randint(0, n - 1)
            intentos += 1

        if centro == le_idx:
            continue

        # Anchura del bump (sigma) en espacio de índices
        sigma = random.uniform(
            CONFIG['sigma_bump_min'] * n,
            CONFIG['sigma_bump_max'] * n
        )
        sigma = max(sigma, 1.0)

        # Amplitud (distribución normal para diversidad)
        amp = random.gauss(0, CONFIG['amplitud_mutacion'])

        # Calcular pesos gaussianos
        indices = np.arange(n, dtype=float)
        pesos = np.exp(-0.5 * ((indices - centro) / sigma) ** 2)

        # Proteger leading edge
        pesos[le_idx] = 0.0

        # Aplicar desplazamiento en Y
        resultado[:, 1] += amp * pesos

    # Mutación opcional del trailing edge en Y
    if random.random() < CONFIG['prob_mutar_trailing_edge']:
        for te_idx in te_indices:
            delta_y = random.gauss(0, CONFIG['amplitud_trailing_edge'])
            resultado[te_idx, 1] += delta_y

    return resultado


def cruce(padre1, padre2, le_idx):
    """
    Cruce BLX (Blend Crossover) punto a punto.

    Para cada punto, la coordenada Y del hijo es una mezcla aleatoria
    entre los valores de ambos padres. Permite explorar el espacio
    entre ambos padres de forma continua y diversa.

    El leading edge se hereda del padre con mejor fitness.
    Las coordenadas X no se modifican.
    """
    genes_hijo = padre1.genes.copy()
    n = len(genes_hijo)

    # Peso aleatorio por punto para mezclar Y de ambos padres
    alpha = np.random.uniform(0.15, 0.85, size=n)

    # Mezclar solo coordenadas Y
    genes_hijo[:, 1] = (alpha * padre1.genes[:, 1]
                        + (1 - alpha) * padre2.genes[:, 1])

    # Preservar leading edge del padre con mejor fitness
    mejor_padre = padre1 if padre1.fitness >= padre2.fitness else padre2
    genes_hijo[le_idx, :] = mejor_padre.genes[le_idx, :]

    return genes_hijo


# ==========================================
# 8. EVALUACIÓN CFD
# ==========================================
def simular_perfil(filepath_temp, alpha_deg, config):
    """
    Ejecuta una simulación CFD 2D para un perfil a un ángulo dado.

    Llama a Simulador2D.main() y retorna Cl, Cd promedio descartando
    el 10% transitorio inicial.

    Returns:
        dict {"cl": float, "cd": float, "ld": float} o None si falla
    """
    mesh_fina = None
    mesh_gruesa = None

    try:
        # Construir parámetros del simulador
        sim_params = {
            'filepath': filepath_temp,
            'iteraciones': config['simulacion_iteraciones'],
            'v0x': config['v0x'],
            'CFL': config['CFL'],
            'dx_fino': config['dx_fino'],
            'dx_grueso': config['dx_grueso'],
            'alpha_deg': alpha_deg,
            'chord': config['chord'],
            'graficos': False,
        }

        # Parámetros extra opcionales del usuario
        sim_params.update(config.get('sim_extra_params', {}))

        mesh_fina, mesh_gruesa, _ = Simulador2D.main(**sim_params)

        # Cl/Cd se almacenan solo en mesh_gruesa (bucle principal)
        if mesh_gruesa.cdvector is None or len(mesh_gruesa.cdvector) == 0:
            return None

        # Promediar descartando 10% inicial (transitorio)
        n_datos = len(mesh_gruesa.cdvector)
        inicio = max(1, int(n_datos * 0.1))

        cd_val = float(cp.mean(mesh_gruesa.cdvector[inicio:]).get())
        cl_val = float(cp.mean(mesh_gruesa.clvector[inicio:]).get())

        # Validar: descartar NaN/Inf
        if (np.isnan(cd_val) or np.isnan(cl_val)
                or np.isinf(cd_val) or np.isinf(cl_val)):
            print(f"   [!] Resultado NaN/Inf @ alpha={alpha_deg}°")
            return None

        ld = cl_val / cd_val if abs(cd_val) > 1e-6 else 0.0

        return {'cl': round(cl_val, 6), 'cd': round(cd_val, 6), 'ld': round(ld, 4)}

    except Exception as e:
        print(f"   [!] Error simulación @ alpha={alpha_deg}°: {e}")
        return None

    finally:
        # Liberar memoria GPU agresivamente
        del mesh_fina, mesh_gruesa
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()


def calcular_fitness(resultados, config):
    """
    Calcula fitness combinado a partir de resultados multi-ángulo.

    Modos:
      'mean':     Promedio de L/D de todos los ángulos (balance)
      'min':      L/D del peor ángulo (conservador/robusto)
      'weighted': Media ponderada con más peso en el ángulo base
    """
    if not resultados:
        return 0.0

    lds = {ang: res['ld'] for ang, res in resultados.items() if res['ld'] > 0}

    if not lds:
        return 0.0

    modo = config.get('fitness_modo', 'mean')

    if modo == 'min':
        return min(lds.values())

    elif modo == 'weighted':
        alpha_base_str = f"{config['alpha_deg']:.1f}"
        peso_base = config.get('peso_angulo_base', 2.0)

        total_peso = 0.0
        total_ld = 0.0
        for ang_str, ld in lds.items():
            peso = peso_base if ang_str == alpha_base_str else 1.0
            total_ld += ld * peso
            total_peso += peso

        return total_ld / total_peso if total_peso > 0 else 0.0

    else:  # 'mean'
        return float(np.mean(list(lds.values())))


def evaluar_poblacion(poblacion, gen, angulos, config, oraculo, logger, condiciones):
    """
    Evalúa todos los individuos no evaluados de la población.

    Para cada individuo:
      1. Filtro IA (si entrenado): predice si vale la pena simularlo
      2. Simulación CFD a cada ángulo configurado
      3. Cálculo de fitness combinado
      4. Registro de datos para ML
    """
    n_total_pendiente = sum(1 for ind in poblacion if ind.fitness == 0)
    n_angulos = len(angulos)
    ang_str = ', '.join(f"{a:.1f}°" for a in angulos)

    print(f"\n--- Evaluando Gen {gen + 1}: {n_total_pendiente} individuos x "
          f"{n_angulos} ángulo{'s' if n_angulos > 1 else ''} ({ang_str}) ---")

    n_evaluados = 0
    n_descartados_ia = 0

    for i, ind in enumerate(poblacion):
        if PARADA_SOLICITADA:
            break

        if ind.fitness != 0:
            continue  # Ya evaluado (elite de generación anterior)

        # Filtro IA
        if oraculo.entrenado and config['usar_ia']:
            if not oraculo.predecir_si_vale_la_pena(ind.genes):
                ind.fitness = 0.0  # Queda sin evaluar
                n_descartados_ia += 1
                continue

        # Crear archivo temporal del perfil
        nombre_temp = f"_temp_gen{gen}_ind{i}.dat"
        guardar_perfil(nombre_temp, ind.genes, ind.header)

        resultados = {}
        try:
            for alpha in angulos:
                if PARADA_SOLICITADA:
                    break

                print(f" [{n_evaluados + 1}/{n_total_pendiente}] "
                      f"Ind {i + 1}/{len(poblacion)} @ alpha={alpha:.1f}°",
                      end="  ")

                res = simular_perfil(nombre_temp, alpha, config)

                if res is not None:
                    resultados[f"{alpha:.1f}"] = res
                    print(f"-> Cl={res['cl']:.4f}  "
                          f"Cd={res['cd']:.5f}  L/D={res['ld']:.2f}")
                else:
                    print("-> FALLO")

        finally:
            if os.path.exists(nombre_temp):
                os.remove(nombre_temp)

        ind.resultados = resultados
        ind.fitness = calcular_fitness(resultados, config)
        n_evaluados += 1

        # Registrar para ML (solo evaluaciones exitosas)
        if ind.fitness > 0 and resultados:
            logger.registrar(
                perfil_puntos=ind.genes,
                condiciones=condiciones,
                resultados_por_angulo=resultados,
                metadata={
                    'generacion': gen,
                    'individuo': i,
                    'fitness': ind.fitness,
                    'perfil_base': config['archivo_base'],
                    'multi_angulo': config['multi_angulo'],
                }
            )

    print(f"\n Evaluación: {n_evaluados} simulados, "
          f"{n_descartados_ia} descartados por IA")

    return n_evaluados, n_descartados_ia


# ==========================================
# 9. VISUALIZACIÓN
# ==========================================
def guardar_top_perfiles(poblacion, coords_base, gen, directorio_base, n_top=2):
    """
    Guarda imágenes de los n_top mejores perfiles de la generación.
    Cada generación crea una subcarpeta: directorio_base/gen_XXX/
    """
    carpeta_gen = os.path.join(directorio_base, f"gen_{gen + 1:03d}")
    os.makedirs(carpeta_gen, exist_ok=True)

    mejores = sorted(
        [ind for ind in poblacion if ind.fitness > 0],
        key=lambda x: x.fitness, reverse=True
    )[:n_top]

    for rank, ind in enumerate(mejores):
        fig, ax = plt.subplots(figsize=(10, 3.5))

        # Perfil base (referencia)
        ax.plot(coords_base[:, 0], coords_base[:, 1],
                'k--', linewidth=0.8, alpha=0.4, label='Base')

        # Perfil optimizado
        ax.plot(ind.genes[:, 0], ind.genes[:, 1],
                'b-', linewidth=1.5, label=f'Top {rank + 1}')
        ax.fill(ind.genes[:, 0], ind.genes[:, 1],
                alpha=0.08, color='steelblue')

        # Info en título
        titulo = f"Gen {gen + 1} — Top {rank + 1}  |  L/D = {ind.fitness:.3f}"
        if ind.resultados:
            partes = []
            for ang in sorted(ind.resultados.keys(), key=float):
                r = ind.resultados[ang]
                partes.append(f"{ang}°: Cl={r['cl']:.4f} Cd={r['cd']:.5f}")
            titulo += "  |  " + "  /  ".join(partes)
        ax.set_title(titulo, fontsize=9)

        ax.set_aspect('equal')
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        nombre = os.path.join(carpeta_gen, f"top_{rank + 1}_LD{ind.fitness:.2f}.png")
        plt.savefig(nombre, dpi=120, bbox_inches='tight')
        plt.close(fig)

    if mejores:
        print(f"   Imágenes top {len(mejores)} guardadas en: {carpeta_gen}")


def visualizar_comparativa(original, optimizado, fitness, resultados, save_path=None):
    """Gráfico comparativo: perfil original vs optimizado + eficiencia por ángulo"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Perfiles superpuestos
    ax1 = axes[0]
    ax1.plot(original[:, 0], original[:, 1], '--k', alpha=0.5,
             linewidth=1.5, label='Original')
    ax1.plot(optimizado[:, 0], optimizado[:, 1], 'r-', linewidth=2.0,
             label=f'Optimizado (L/D={fitness:.2f})')

    # Marcar zona de mayor diferencia
    if len(original) == len(optimizado):
        diff_y = optimizado[:, 1] - original[:, 1]
        max_diff_idx = np.argmax(np.abs(diff_y))
        if abs(diff_y[max_diff_idx]) > 1e-5:
            ax1.annotate(
                f'Max delta_y={diff_y[max_diff_idx]:+.4f}',
                xy=(optimizado[max_diff_idx, 0], optimizado[max_diff_idx, 1]),
                fontsize=8, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=0.8)
            )

    ax1.set_title("Comparativa de Perfiles")
    ax1.set_xlabel("x/c")
    ax1.set_ylabel("y/c")
    ax1.legend(fontsize=9)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Eficiencia por ángulo
    ax2 = axes[1]
    if resultados:
        angulos_sorted = sorted(resultados.keys(), key=float)
        lds = [resultados[a]['ld'] for a in angulos_sorted]
        cls = [resultados[a]['cl'] for a in angulos_sorted]
        cds = [resultados[a]['cd'] for a in angulos_sorted]

        ax2.bar([f"{a}°" for a in angulos_sorted], lds,
                color='steelblue', alpha=0.8, edgecolor='navy')

        ax2.set_ylabel("L/D (Eficiencia)")
        ax2.set_xlabel("Ángulo de ataque")
        ax2.set_title("Eficiencia Aerodinámica por Ángulo")
        ax2.grid(True, alpha=0.3, axis='y')

        for j, (a, ld) in enumerate(zip(angulos_sorted, lds)):
            ax2.annotate(
                f"Cl={cls[j]:.3f}\nCd={cds[j]:.5f}",
                (j, ld), textcoords="offset points",
                xytext=(0, 8), ha='center', fontsize=8
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Gráfico guardado: {save_path}")

    try:
        plt.show()
    except Exception:
        pass

    plt.close(fig)


def plot_convergencia(historial, save_path=None):
    """Gráfico de convergencia del fitness a lo largo de las generaciones"""
    if not historial:
        return

    gens = [h['gen'] + 1 for h in historial]
    mejores = [h['mejor'] for h in historial]
    medias = [h['media'] for h in historial]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, mejores, 'r-o', linewidth=2, markersize=5, label='Mejor L/D')
    ax.plot(gens, medias, 'b--s', linewidth=1, markersize=3,
            alpha=0.7, label='Media L/D')
    ax.fill_between(gens, medias, mejores, alpha=0.1, color='red')

    ax.set_xlabel("Generación")
    ax.set_ylabel("L/D")
    ax.set_title("Convergencia del Algoritmo Genético")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    try:
        plt.show()
    except Exception:
        pass

    plt.close(fig)


# ==========================================
# 10. PERSISTENCIA DEL ESTADO DEL GA
# ==========================================
def guardar_estado_ga(filepath, poblacion, mejor_global, gen, historial):
    """
    Guarda el estado completo del GA en JSON para inspección y reanudación.
    """
    estado = {
        'timestamp': datetime.now().isoformat(),
        'generacion_actual': gen,
        'config': {k: v for k, v in CONFIG.items()
                   if isinstance(v, (int, float, str, bool, list, dict))},
        'historial': historial,
        'mejor_global': {
            'genes': mejor_global.genes.tolist() if mejor_global else None,
            'header': mejor_global.header if mejor_global else None,
            'fitness': mejor_global.fitness if mejor_global else 0,
            'resultados': mejor_global.resultados if mejor_global else {}
        },
        'poblacion_resumen': [
            {
                'fitness': ind.fitness,
                'resultados': ind.resultados,
            }
            for ind in poblacion
        ],
    }

    try:
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(estado, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f" Error guardando estado GA: {e}")


# ==========================================
# 11. FUNCIÓN PRINCIPAL
# ==========================================
def main():
    global PARADA_SOLICITADA
    signal.signal(signal.SIGINT, manejar_parada)

    print("\n" + "=" * 60)
    print(" OPTIMIZADOR GENÉTICO DE PERFILES AERODINÁMICOS")
    print("=" * 60)
    print(f" Fecha:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Perfil base:  {CONFIG['archivo_base']}")
    print(f" Velocidad:    {CONFIG['v0x']} m/s")
    print(f" Alpha base:   {CONFIG['alpha_deg']}°")
    print(f" Población:    {CONFIG['poblacion_tamano']} individuos")
    print(f" Generaciones: {CONFIG['generaciones']} (máximo)")

    # Crear directorio de resultados
    os.makedirs(CONFIG['directorio_resultados'], exist_ok=True)

    # =====================
    # PASO 1: Cargar perfil base
    # =====================
    coords_base, header_base, le_idx, te_indices = cargar_perfil(
        CONFIG['archivo_base']
    )
    if len(coords_base) == 0:
        print("\nERROR CRÍTICO: No se encontró el archivo de perfil base.")
        return

    # =====================
    # PASO 2: Determinar ángulos de simulación
    # =====================
    alpha_base = CONFIG['alpha_deg']
    if CONFIG['multi_angulo']:
        delta = CONFIG['delta_angulo']
        angulos = [
            round(alpha_base - delta, 1),
            round(alpha_base, 1),
            round(alpha_base + delta, 1)
        ]
        print(f"\n Multi-ángulo ACTIVADO: {angulos}°")
        print(f" Modo fitness: {CONFIG['fitness_modo']}")
    else:
        angulos = [round(alpha_base, 1)]
        print(f"\n Ángulo único: {alpha_base}°")

    sims_por_gen = CONFIG['poblacion_tamano'] * len(angulos)
    print(f" Simulaciones por generación: ~{sims_por_gen}")
    print(f" Iteraciones CFD por simulación: {CONFIG['simulacion_iteraciones']}")

    # =====================
    # PASO 3: Inicializar sistemas auxiliares
    # =====================
    oraculo = OraculoAerodinamico(CONFIG['archivo_memoria_ia'])
    logger = DataLoggerML(CONFIG['archivo_datos_ml'])

    # Condiciones de vuelo (se registran con cada evaluación para ML)
    Re = CONFIG['v0x'] * CONFIG['chord'] / CONFIG['nu']
    condiciones = {
        'v0x': CONFIG['v0x'],
        'alpha_base': alpha_base,
        'angulos_evaluados': angulos,
        'chord': CONFIG['chord'],
        'Re': round(Re, 1),
        'rho': CONFIG['rho'],
        'nu': CONFIG['nu'],
        'dx_fino': CONFIG['dx_fino'],
        'dx_grueso': CONFIG['dx_grueso'],
        'iteraciones_cfd': CONFIG['simulacion_iteraciones'],
        'fitness_modo': CONFIG['fitness_modo'],
    }

    # =====================
    # PASO 4: Crear población inicial
    # =====================
    print(f"\n Generando población inicial "
          f"({CONFIG['poblacion_tamano']} individuos)...")
    poblacion = []
    for _ in range(CONFIG['poblacion_tamano']):
        genes = mutar_perfil(coords_base.copy(), le_idx, te_indices)
        genes = suavizar_perfil(genes, le_idx, CONFIG['suavizado_iteraciones'])
        poblacion.append(Individuo(genes, header_base))

    mejor_global = None
    historial_fitness = []
    gen_actual = -1

    # =====================
    # PASO 5: BUCLE EVOLUTIVO
    # =====================
    print(f"\n{'=' * 60}")
    print(f" INICIANDO EVOLUCIÓN")
    print(f"{'=' * 60}")

    t_total_inicio = time.time()

    for gen in range(CONFIG['generaciones']):
        gen_actual = gen

        if PARADA_SOLICITADA:
            print("\n Parada solicitada. Saliendo del bucle evolutivo...")
            break

        t_gen_inicio = time.time()

        print(f"\n{'=' * 55}")
        print(f" GENERACIÓN {gen + 1} / {CONFIG['generaciones']}")
        print(f"{'=' * 55}")

        # --- A. EVALUACIÓN CFD ---
        n_eval, n_desc = evaluar_poblacion(
            poblacion, gen, angulos, CONFIG, oraculo, logger, condiciones
        )

        # --- B. APRENDIZAJE IA ---
        oraculo.aprender(poblacion)

        # --- C. ESTADÍSTICAS ---
        poblacion.sort(key=lambda x: x.fitness, reverse=True)
        mejor_gen = poblacion[0]
        fitness_validos = [ind.fitness for ind in poblacion if ind.fitness > 0]

        t_gen = time.time() - t_gen_inicio

        print(f"\n{'─' * 50}")
        print(f" GEN {gen + 1} COMPLETADA  ({t_gen:.0f}s)")
        print(f"{'─' * 50}")
        print(f"   Mejor L/D:     {mejor_gen.fitness:.4f}")

        if fitness_validos:
            print(f"   Media L/D:     {np.mean(fitness_validos):.4f}")
            print(f"   Peor  L/D:     {min(fitness_validos):.4f}")
            print(f"   Evaluados:     {len(fitness_validos)}/{len(poblacion)}")

        if n_desc > 0:
            print(f"   IA descartes:  {n_desc}")
        oraculo.conteo_descartes = 0

        # Detalle por ángulo del mejor individuo
        if mejor_gen.resultados:
            for ang in sorted(mejor_gen.resultados.keys(), key=float):
                res = mejor_gen.resultados[ang]
                print(f"   @{ang}°: Cl={res['cl']:.4f}  "
                      f"Cd={res['cd']:.5f}  L/D={res['ld']:.2f}")

        historial_fitness.append({
            'gen': gen,
            'mejor': mejor_gen.fitness,
            'media': (float(np.mean(fitness_validos))
                      if fitness_validos else 0.0),
            'peor': (float(min(fitness_validos))
                     if fitness_validos else 0.0),
            'evaluados': len(fitness_validos),
            'descartados_ia': n_desc,
            'tiempo_seg': round(t_gen, 1),
        })

        # --- Guardar imágenes de los 2 mejores perfiles ---
        try:
            guardar_top_perfiles(
                poblacion, coords_base, gen,
                os.path.join(CONFIG['directorio_resultados'], 'perfiles_top')
            )
        except Exception as e:
            print(f"   [!] Error guardando imágenes top: {e}")

        # --- Nuevo récord global ---
        if mejor_global is None or mejor_gen.fitness > mejor_global.fitness:
            mejor_global = copy.deepcopy(mejor_gen)
            print(f"\n   >>> NUEVO RÉCORD GLOBAL: "
                  f"L/D = {mejor_global.fitness:.4f} <<<")
            guardar_perfil(
                os.path.join(CONFIG['directorio_resultados'],
                             "OPTIMO_PARCIAL.dat"),
                mejor_global.genes, mejor_global.header
            )

        # --- Guardar estado del GA ---
        guardar_estado_ga(
            os.path.join(CONFIG['directorio_resultados'], "estado_ga.json"),
            poblacion, mejor_global, gen, historial_fitness
        )

        # --- D. REPRODUCCIÓN ---
        nueva_poblacion = []

        # Elitismo: preservar los mejores intactos
        for elite in poblacion[:CONFIG['elites']]:
            ind_elite = copy.deepcopy(elite)
            nueva_poblacion.append(ind_elite)

        # Seleccionar pool de candidatos para reproducción
        candidatos = [ind for ind in poblacion if ind.fitness > 0]
        if len(candidatos) < 2:
            candidatos = poblacion  # Fallback

        intentos_fallidos = 0
        MAX_INTENTOS = 80

        while len(nueva_poblacion) < CONFIG['poblacion_tamano']:
            if PARADA_SOLICITADA:
                break

            # Selección por torneo
            tam_torneo = min(CONFIG['torneo_tamano'], len(candidatos))
            padre1 = max(
                random.sample(candidatos, tam_torneo),
                key=lambda x: x.fitness
            )
            padre2 = max(
                random.sample(candidatos, tam_torneo),
                key=lambda x: x.fitness
            )

            # Cruce
            genes_hijo = cruce(padre1, padre2, le_idx)

            # Mutación
            if random.random() < CONFIG['prob_mutacion']:
                genes_hijo = mutar_perfil(genes_hijo, le_idx, te_indices)

            # Suavizado post-mutación
            genes_hijo = suavizar_perfil(
                genes_hijo, le_idx, CONFIG['suavizado_iteraciones']
            )

            hijo = Individuo(genes_hijo, header_base)

            # Filtro IA pre-evaluación
            if (oraculo.entrenado and CONFIG['usar_ia']
                    and intentos_fallidos < MAX_INTENTOS):
                if not oraculo.predecir_si_vale_la_pena(hijo.genes):
                    intentos_fallidos += 1
                    continue

            nueva_poblacion.append(hijo)
            intentos_fallidos = 0

        poblacion = nueva_poblacion

    # ==========================================
    # FINALIZACIÓN
    # ==========================================
    t_total = time.time() - t_total_inicio

    print("\n" + "=" * 60)
    print(" OPTIMIZACIÓN FINALIZADA")
    print("=" * 60)
    print(f" Generaciones completadas: {gen_actual + 1}")
    print(f" Tiempo total: {t_total / 60:.1f} minutos")
    print(f" Datos ML recopilados: {logger.count} registros "
          f"en '{CONFIG['archivo_datos_ml']}'")

    if mejor_global:
        print(f"\n Mejor L/D global: {mejor_global.fitness:.4f}")

        if mejor_global.resultados:
            for ang in sorted(mejor_global.resultados.keys(), key=float):
                res = mejor_global.resultados[ang]
                print(f"   @{ang}°: Cl={res['cl']:.4f}  "
                      f"Cd={res['cd']:.5f}  L/D={res['ld']:.2f}")

        # Guardar perfil final
        nombre_final = os.path.join(
            CONFIG['directorio_resultados'], "OPTIMO_FINAL.dat"
        )
        guardar_perfil(nombre_final, mejor_global.genes, mejor_global.header)
        print(f"\n Perfil óptimo guardado en: {nombre_final}")

        # Guardar historial de fitness
        historial_path = os.path.join(
            CONFIG['directorio_resultados'], "historial_fitness.json"
        )
        with open(historial_path, 'w', encoding='utf-8') as f:
            json.dump(historial_fitness, f, indent=2, ensure_ascii=False)
        print(f" Historial guardado en: {historial_path}")

        # Guardar estado final del GA
        guardar_estado_ga(
            os.path.join(CONFIG['directorio_resultados'],
                         "estado_ga_final.json"),
            poblacion, mejor_global, gen_actual, historial_fitness
        )

        # Generar gráficos finales
        try:
            visualizar_comparativa(
                coords_base, mejor_global.genes, mejor_global.fitness,
                mejor_global.resultados,
                save_path=os.path.join(
                    CONFIG['directorio_resultados'],
                    "comparativa_perfiles.png"
                )
            )
            plot_convergencia(
                historial_fitness,
                save_path=os.path.join(
                    CONFIG['directorio_resultados'], "convergencia.png"
                )
            )
        except Exception as e:
            print(f" Error generando gráficos: {e}")
    else:
        print("\n No se generaron individuos válidos en ninguna generación.")

    # Listar archivos generados
    print(f"\n Archivos en '{CONFIG['directorio_resultados']}':")
    if os.path.exists(CONFIG['directorio_resultados']):
        for f_name in sorted(os.listdir(CONFIG['directorio_resultados'])):
            fpath = os.path.join(CONFIG['directorio_resultados'], f_name)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"   - {f_name} ({size_kb:.1f} KB)")

    print()


if __name__ == "__main__":
    main()

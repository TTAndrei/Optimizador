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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cupy as cp
from sklearn.ensemble import RandomForestRegressor

import Simulador2D

# Parámetros óptimos determinados para generación de hijos (barrido extendido).
PARAMETROS_OPTIMOS_HIJOS = {
    'modo_mutacion': 'parametrica',
    'prob_mutacion': 0.85,
    'camber_mut_std': 0.0035,
    'espesor_mut_std': 0.0045,
    'te_camber_shift_std': 0.0015,
    'te_espesor_shift_std': 0.0012,
}

# ==========================================
# 1. CONFIGURACIÓN DEL SISTEMA
# ==========================================
CONFIG = {
    # --- Archivos ---
    'archivo_base': 'profiles/s1014.dat',                    # Perfil base (formato Selig)
    'archivo_memoria_ia': 'cerebro_aerodinamico.pkl',   # Memoria persistente del filtro IA
    'archivo_datos_ml': 'aprendizaje_ML.jsonl',         # Datos acumulados para ML futuro
    'directorio_resultados': 'auto',   # 'auto' -> results/{perfil}_Re{Re}_a{alpha}/

    # --- Parámetros Evolutivos ---
    'poblacion_tamano': 10,        # Individuos por generación
    'generaciones': 30,            # Máximo de generaciones (parable con Ctrl+C)
    'elites': 4,                   # Individuos preservados intactos por elitismo
    'torneo_tamano': 4,            # Tamaño del torneo de selección de padres

    # --- Mutación (Bumps Gaussianos) ---
    #   Cada bump es una campana de Gauss centrada en un punto aleatorio
    #   que desplaza Y de los puntos cercanos. Produce deformaciones suaves.
    'prob_mutacion': PARAMETROS_OPTIMOS_HIJOS['prob_mutacion'],
                                         # Probabilidad óptima de mutar un hijo
    'n_bumps_min': 1,                   # Mínimo de bumps por mutación
    'n_bumps_max': 4,                   # Máximo de bumps por mutación
    'sigma_bump_min': 0.03,             # Anchura mín del bump (fracción de nº puntos)
    'sigma_bump_max': 0.15,             # Anchura máx del bump
    'amplitud_mutacion': 0.008,         # Desviación estándar del desplazamiento en Y
    'prob_mutar_trailing_edge': 0.3,    # Probabilidad de mover el trailing edge
    'amplitud_trailing_edge': 0.002,    # Desviación estándar del TE en Y

    # --- Mutación Paramétrica (Camber/Espesor) ---
    #   Muta funciones suaves de camber c(x) y espesor t(x), luego reconstruye
    #   extrados/intrados con restricciones geométricas físicas.
    'modo_mutacion': PARAMETROS_OPTIMOS_HIJOS['modo_mutacion'],
                                         # 'parametrica' | 'legacy' (óptimo: parametrica)
    'usar_legacy_fallback': True,        # Si falla la vía paramétrica, probar legacy
    'max_intentos_geometria': 120,       # Reintentos para generar un hijo válido
    'n_control_mutacion': 7,             # Nodos de control para perturbaciones suaves
    'camber_mut_std': PARAMETROS_OPTIMOS_HIJOS['camber_mut_std'],
                                         # Intensidad óptima de mutación del camber
    'espesor_mut_std': PARAMETROS_OPTIMOS_HIJOS['espesor_mut_std'],
                                         # Intensidad óptima de mutación del espesor
    'te_camber_shift_std': PARAMETROS_OPTIMOS_HIJOS['te_camber_shift_std'],
                                         # Desplazamiento óptimo de camber concentrado en TE
    'te_espesor_shift_std': PARAMETROS_OPTIMOS_HIJOS['te_espesor_shift_std'],
                                         # Ajuste óptimo extra del espesor en TE

    # S1: mutación adaptativa (exploración temprana -> refinamiento tardío).
    'sigma_adaptativa': False,           # True = escala las *_mut_std por generación
    'sigma_factor_inicial': 3.0,         # Factor de sigma en gen 0
    'sigma_factor_final': 1.0,           # Factor de sigma en la última generación
    'le_proteccion_x': 0.06,             # Zona [0, x] con mutación atenuada en LE
    'espesor_min_global': 2e-4,          # Espesor mínimo global (excepto LE)
    'te_espesor_min_absoluto': 3e-4,     # Cota inferior absoluta para espesor TE
    'te_espesor_rel_min': 0.65,          # Cota inferior relativa al espesor TE base
    'te_espesor_rel_max': 3.50,          # Cota superior relativa al espesor TE base
    'te_validacion_tol': 1e-9,           # Tolerancia numérica para validación de TE
    'le_radio_factor_min': 0.40,         # Radio LE mínimo relativo al perfil base
    'le_radio_min_absoluto': 2e-4,       # Radio LE mínimo absoluto
    'le_puntos_preservar': 5,            # Puntos por lado del LE a preservar parcialmente
    'reportar_diagnostico_geometria': True,

    # --- Simulación CFD (config de referencia validada: consistent+sa+maccormack) ---
    'simulacion_iteraciones': 8000,     # Iteraciones por simulación CFD (lo fija el orquestador)
    'v0x': 1,                         # Velocidad del flujo libre (m/s)
    'alpha_deg': 4.0,                   # Ángulo de ataque base (grados)
    'chord': 1.0,                       # Longitud de cuerda (m)
    'dx_min': 0.002,                    # Espaciado mínimo malla variable (m)
    'CFL': 0.5,                         # Número de Courant
    'rho': 1.0,                       # Densidad del aire (kg/m³)
    'nu': 1e-5,                       # Viscosidad cinemática (m²/s) -> Re=1e5

    # Config de referencia del solver (fiable). Ver RESUMEN.md.
    'turb_model': 'sa',                 # Spalart-Allmaras (elimina LSB/colapso)
    'wall_treatment': 'consistent',     # divergencia face_flux + grad one-sided + reforzar OFF
    'advection_scheme': 'maccormack',   # corrige difusión numérica del SL bilineal
    'min_te_height_factor': 1.0,        # perfiles con TE afilado
    'wake_refinement_mode': 'long_fine_x',
    'mg_niveles_max': 2,
    'mg_max_outer': 8,
    'divergencia': 0.02,                # tolerancia de la proyección
    'stop_on_clcd_convergence': True,   # parada temprana cuando Cl/Cd se estacionan

    # --- Multi-ángulo (opcional) ---
    #   Si activo, cada perfil se simula a alpha-delta, alpha, alpha+delta.
    #   Evalúa robustez del perfil a variaciones del ángulo de ataque.
    #   Poner multi_angulo=False para optimizar solo al ángulo base.
    'multi_angulo': False,               # True = evaluar alpha ± delta_angulo
    'delta_angulo': 1.0,                # ± grados alrededor de alpha_deg
    'fitness_modo': 'mean',             # 'mean' | 'min' | 'weighted'
    'peso_angulo_base': 2.0,            # Peso extra para ángulo base (modo 'weighted')

    # --- IA (Filtro predictivo) ---
    'usar_ia': False,                    # Activar filtro IA (Random Forest)
    'umbral_calidad': 0.7,              # Umbral dinámico (fracción del top histórico)

    # --- Suavizado ---
    'suavizado_iteraciones': 2,         # Pasadas de suavizado laplaciano post-mutación

    # --- Semillas múltiples (opcional; habilita población mixta / Arm B) ---
    #   Si es una lista de rutas .dat, la población inicial se siembra en
    #   round-robin sobre TODAS las semillas (permite convergencia inter-semilla).
    #   Si es None, se usa 'archivo_base'.
    'archivos_base': None,

    # --- Límite de tiempo (deadline). None = sin límite. ---
    'tiempo_limite_s': None,

    # --- Parámetros extra del simulador (mesh/dominio de la config de referencia) ---
    'sim_extra_params': {'Lx': 8, 'Ly': 5, 'cx': 2,
                         'ancho_zona_fina_x': 1.5, 'ancho_zona_fina_y': 1.0,
                         'factor_expansion': 1.1},

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

    # Mostrar linaje si existe companion .meta.json
    for meta_candidate in (filepath + '.meta.json',
                           os.path.splitext(filepath)[0] + '.meta.json'):
        if os.path.exists(meta_candidate):
            try:
                with open(meta_candidate, 'r', encoding='utf-8') as _f:
                    _meta = json.load(_f)
                print(f"   Linaje:    {_meta.get('perfil_padre', '?')} "
                      f"-> {os.path.basename(filepath)}  "
                      f"(opt. gen {_meta.get('generacion_optimizacion', '?')})")
                _cond = _meta.get('condiciones', {})
                if _cond:
                    print(f"   Cond. previas: Re={_cond.get('Re','?')}  "
                          f"alpha={_cond.get('alpha_deg','?')}deg  "
                          f"v0x={_cond.get('v0x','?')} m/s")
                _prev = _meta.get('resultados', {})
                for _ang, _r in sorted(_prev.items(), key=lambda x: float(x[0])):
                    print(f"   @{_ang}deg: Cl={_r.get('cl','?')}  "
                          f"Cd={_r.get('cd','?')}  L/D={_r.get('ld','?')}")
            except Exception:
                pass
            break

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


def _meta_ancestro(base_path):
    """Lee el .meta.json compañero para extraer (ancestro_original, generacion_optimizacion+1)."""
    for candidate in (
        base_path + '.meta.json',
        os.path.splitext(base_path)[0] + '.meta.json',
    ):
        if os.path.exists(candidate):
            try:
                with open(candidate, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                return (
                    meta.get('ancestro_original',
                             os.path.splitext(os.path.basename(base_path))[0]),
                    int(meta.get('generacion_optimizacion', 0)) + 1,
                )
            except Exception:
                break
    return os.path.splitext(os.path.basename(base_path))[0], 1


def generar_directorio_resultados(config):
    """
    Genera la ruta del directorio de resultados bajo results/.

    Formato: results/{ancestro}[_g{N}]_Re{Re:.0f}_a{alpha:.1f}
    Se llama al inicio de main() cuando directorio_resultados == 'auto'.
    """
    ancestro, gen_opt = _meta_ancestro(config['archivo_base'])
    re_val = config['v0x'] * config['chord'] / config['nu']
    gen_tag = f'_g{gen_opt}' if gen_opt > 1 else ''
    nombre = f"{ancestro}{gen_tag}_Re{round(re_val)}_a{config['alpha_deg']:.1f}"
    return os.path.join('results', nombre)


def generar_stem_optimizado(config, fitness):
    """
    Genera el stem (nombre sin extensión) para el perfil optimizado.

    Formato:  {ancestro}[_g{N}]_Re{Re:.0f}_a{alpha:.1f}_LD{ld:.2f}
    Ejemplos: AG24_Re100000_a4.0_LD15.23
              AG24_g2_Re100000_a4.0_LD16.01  (segunda corrida de optimización)

    Returns: (stem: str, lineage: dict)
    """
    ancestro, gen_opt = _meta_ancestro(config['archivo_base'])
    padre_name = os.path.splitext(os.path.basename(config['archivo_base']))[0]
    re_val = config['v0x'] * config['chord'] / config['nu']
    gen_tag = f'_g{gen_opt}' if gen_opt > 1 else ''
    stem = (f"{ancestro}{gen_tag}"
            f"_Re{round(re_val)}"
            f"_a{config['alpha_deg']:.1f}"
            f"_LD{fitness:.2f}")
    lineage = {
        'ancestro_original': ancestro,
        'perfil_padre': padre_name,
        'generacion_optimizacion': gen_opt,
    }
    return stem, lineage


def guardar_perfil_con_metadata(directorio, stem, genes, config, fitness,
                                 resultados, lineage, historial=None):
    """
    Guarda el perfil .dat con nombre descriptivo + un .meta.json compañero.

    El .meta.json permite relanzar el GA desde este perfil y reconstruir
    la cadena de linaje completa. Para continuar la optimización basta con
    apuntar CONFIG['archivo_base'] al .dat generado.

    Returns: ruta absoluta del .dat guardado.
    """
    os.makedirs(directorio, exist_ok=True)
    re_val = config['v0x'] * config['chord'] / config['nu']

    header = (
        f"{stem}  "
        f"[padre:{lineage['perfil_padre']}  "
        f"Re:{round(re_val)}  alpha:{config['alpha_deg']}deg  "
        f"L/D:{fitness:.4f}]\n"
    )
    filepath_dat = os.path.join(directorio, stem + '.dat')
    guardar_perfil(filepath_dat, genes, header)

    meta = {
        'nombre': stem,
        'timestamp': datetime.now().isoformat(),
        'ancestro_original': lineage['ancestro_original'],
        'perfil_padre': lineage['perfil_padre'],
        'generacion_optimizacion': lineage['generacion_optimizacion'],
        'condiciones': {
            'v0x': config['v0x'],
            'alpha_deg': config['alpha_deg'],
            'chord': config['chord'],
            'Re': round(re_val, 1),
            'rho': config['rho'],
            'nu': config['nu'],
            'dx_min': config['dx_min'],
            'iteraciones_cfd': config['simulacion_iteraciones'],
            'CFL': config['CFL'],
        },
        'resultados': resultados,
        'fitness': round(fitness, 6),
        'historial_fitness': historial or [],
        'config_ga': {k: v for k, v in config.items()
                      if isinstance(v, (int, float, str, bool, list, dict))},
    }
    try:
        with open(os.path.join(directorio, stem + '.meta.json'),
                  'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f'   [!] Error guardando metadata: {e}')

    return filepath_dat


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


def _smoothstep(z):
    """Interpolación suave en [0, 1] para crear envolventes sin quiebres."""
    z = np.clip(z, 0.0, 1.0)
    return z * z * (3.0 - 2.0 * z)


def _asegurar_x_estrictamente_creciente(x):
    """Evita problemas numéricos en interpolación si hay empates de X."""
    x_out = np.asarray(x, dtype=float).copy()
    if len(x_out) < 2:
        return x_out

    eps = 1e-10
    for i in range(1, len(x_out)):
        if x_out[i] <= x_out[i - 1]:
            x_out[i] = x_out[i - 1] + eps
    return x_out


def _suavizar_vector_1d(vec, n_pasadas=2):
    """Suavizado laplaciano 1D liviano para funciones c(x) y t(x)."""
    out = np.asarray(vec, dtype=float).copy()
    if len(out) < 3:
        return out

    for _ in range(max(1, int(n_pasadas))):
        nuevo = out.copy()
        nuevo[1:-1] = (out[:-2] + 2.0 * out[1:-1] + out[2:]) / 4.0
        out = nuevo
    return out


def descomponer_camber_espesor(puntos, le_idx, n_muestras=None):
    """
    Descompone un perfil (orden Selig) en camber c(x) y espesor t(x).
    """
    n = len(puntos)
    if n < 4 or le_idx <= 0 or le_idx >= n - 1:
        return None, None, None

    upper = puntos[:le_idx + 1]      # TE superior -> LE
    lower = puntos[le_idx:]          # LE -> TE inferior

    upper_x_inc = _asegurar_x_estrictamente_creciente(upper[::-1, 0])
    upper_y_inc = np.asarray(upper[::-1, 1], dtype=float)
    lower_x_inc = _asegurar_x_estrictamente_creciente(lower[:, 0])
    lower_y_inc = np.asarray(lower[:, 1], dtype=float)

    if n_muestras is None:
        n_muestras = max(len(upper_x_inc), len(lower_x_inc))

    x_ini = max(float(upper_x_inc[0]), float(lower_x_inc[0]))
    x_fin = min(float(upper_x_inc[-1]), float(lower_x_inc[-1]))
    if x_fin <= x_ini + 1e-12:
        return None, None, None

    x_common = np.linspace(x_ini, x_fin, int(n_muestras))
    y_up = np.interp(x_common, upper_x_inc, upper_y_inc)
    y_lo = np.interp(x_common, lower_x_inc, lower_y_inc)

    camber = 0.5 * (y_up + y_lo)
    espesor = np.maximum(y_up - y_lo, 0.0)
    espesor[0] = 0.0  # LE cerrado

    return x_common, camber, espesor


def resamplear_a_grid(seed_coords, seed_le_idx, ref_coords, ref_le_idx):
    """
    Reproyecta un perfil `seed` sobre la rejilla X del perfil `ref` (mismo nº de
    puntos, mismo le_idx y mismas X que ref), interpolando su Y. Esto hace que
    semillas con distinto nº de puntos sean compatibles con cruce/mutación en una
    población mixta. Devuelve genes (N,2) con las X de ref y las Y de la semilla.
    """
    ref = np.asarray(ref_coords, dtype=float)
    seed = np.asarray(seed_coords, dtype=float)

    s_up = seed[:seed_le_idx + 1]      # TE sup -> LE
    s_lo = seed[seed_le_idx:]          # LE -> TE inf
    s_up_x = _asegurar_x_estrictamente_creciente(s_up[::-1, 0])
    s_up_y = np.asarray(s_up[::-1, 1], dtype=float)
    s_lo_x = _asegurar_x_estrictamente_creciente(s_lo[:, 0])
    s_lo_y = np.asarray(s_lo[:, 1], dtype=float)

    out = ref.copy()
    # Parte superior de ref: TE sup -> LE (X decreciente)
    out[:ref_le_idx + 1, 1] = np.interp(ref[:ref_le_idx + 1, 0], s_up_x, s_up_y)
    # Parte inferior de ref: LE -> TE inf (X creciente)
    out[ref_le_idx:, 1] = np.interp(ref[ref_le_idx:, 0], s_lo_x, s_lo_y)
    return out


def estimar_radio_le(puntos, le_idx):
    """Estima radio local de LE usando el circuncírculo de 3 puntos."""
    if le_idx <= 0 or le_idx >= len(puntos) - 1:
        return np.inf

    p0 = np.asarray(puntos[le_idx - 1], dtype=float)
    p1 = np.asarray(puntos[le_idx], dtype=float)
    p2 = np.asarray(puntos[le_idx + 1], dtype=float)

    a = np.linalg.norm(p1 - p0)
    b = np.linalg.norm(p2 - p1)
    c = np.linalg.norm(p2 - p0)
    v1 = p1 - p0
    v2 = p2 - p0
    area2 = abs(v1[0] * v2[1] - v1[1] * v2[0])  # 2 * area en 2D

    if area2 < 1e-12:
        return np.inf

    return float((a * b * c) / (2.0 * area2))


def extraer_parametrizacion(puntos, le_idx):
    """
    Extrae parámetros de diseño compactos de un perfil para el dataset de IA.

    Returns dict con: camber_max, camber_pos, espesor_max, espesor_pos,
                      le_radius, te_gap. None en cada campo si falla.
    """
    resultado = {
        'camber_max': None, 'camber_pos': None,
        'espesor_max': None, 'espesor_pos': None,
        'le_radius': None, 'te_gap': None,
    }

    try:
        x_common, camber, espesor = descomponer_camber_espesor(puntos, le_idx)
        if x_common is not None and len(camber) > 0:
            idx_c = int(np.argmax(np.abs(camber)))
            idx_e = int(np.argmax(espesor))
            resultado['camber_max'] = round(float(camber[idx_c]), 6)
            resultado['camber_pos'] = round(float(x_common[idx_c]), 6)
            resultado['espesor_max'] = round(float(espesor[idx_e]), 6)
            resultado['espesor_pos'] = round(float(x_common[idx_e]), 6)

        r_le = estimar_radio_le(puntos, le_idx)
        resultado['le_radius'] = round(float(r_le), 6) if np.isfinite(r_le) else None

        resultado['te_gap'] = round(abs(float(puntos[0, 1] - puntos[-1, 1])), 6)
    except Exception:
        pass

    return resultado


def construir_restricciones_geometricas(perfil_base, le_idx, config):
    """Deriva umbrales geométricos automáticos a partir del perfil base."""
    te_base = abs(float(perfil_base[0, 1] - perfil_base[-1, 1]))
    radio_le_base = estimar_radio_le(perfil_base, le_idx)
    if not np.isfinite(radio_le_base):
        radio_le_base = 0.001

    te_min = max(
        float(config.get('te_espesor_min_absoluto', 0.0)),
        te_base * float(config.get('te_espesor_rel_min', 1.0))
    )
    te_max = max(
        te_min * 1.25,
        te_base * float(config.get('te_espesor_rel_max', 1.0))
    )
    radio_le_min = max(
        float(config.get('le_radio_min_absoluto', 0.0)),
        radio_le_base * float(config.get('le_radio_factor_min', 1.0))
    )

    n_muestras = max(le_idx + 1, len(perfil_base) - le_idx)

    return {
        'te_base': te_base,
        'te_gap_min': float(te_min),
        'te_gap_max': float(te_max),
        'le_radio_base': float(radio_le_base),
        'le_radius_min': float(radio_le_min),
        'n_muestras': int(n_muestras),
    }


def _perturbacion_suave(x_common, std, n_control, le_proteccion_x):
    """Genera perturbación suave con nodos de control e interpolación lineal."""
    x_common = np.asarray(x_common, dtype=float)
    if len(x_common) < 2 or std <= 0:
        return np.zeros_like(x_common)

    n_ctrl = max(4, int(n_control))
    ctrl_x = np.linspace(x_common[0], x_common[-1], n_ctrl)
    ctrl_y = np.random.normal(0.0, std, size=n_ctrl)
    ctrl_y[0] = 0.0
    if n_ctrl > 1:
        ctrl_y[1] *= 0.35

    perturb = np.interp(x_common, ctrl_x, ctrl_y)

    x_norm = (x_common - x_common[0]) / max(1e-12, x_common[-1] - x_common[0])
    le_zone = max(0.0, min(float(le_proteccion_x), 0.40))
    if le_zone > 0:
        escala = _smoothstep((x_norm - le_zone) / max(1e-12, 1.0 - le_zone))
        perturb *= escala

    return perturb


def proyectar_perfil_parametrico(genes, le_idx, restricciones, config, perturbar=True):
    """
    Proyecta un perfil al espacio camber/espesor y aplica restricciones duras.
    Si perturbar=True, además aplica mutaciones suaves en ese espacio.
    """
    x_common, camber, espesor = descomponer_camber_espesor(
        genes, le_idx, n_muestras=restricciones['n_muestras']
    )
    if x_common is None:
        return genes.copy()

    camber_ref = camber.copy()
    espesor_ref = espesor.copy()
    x_norm = (x_common - x_common[0]) / max(1e-12, x_common[-1] - x_common[0])

    if perturbar:
        camber += _perturbacion_suave(
            x_common,
            std=float(config.get('camber_mut_std', 0.0)),
            n_control=config.get('n_control_mutacion', 6),
            le_proteccion_x=config.get('le_proteccion_x', 0.05)
        )
        espesor += _perturbacion_suave(
            x_common,
            std=float(config.get('espesor_mut_std', 0.0)),
            n_control=config.get('n_control_mutacion', 6),
            le_proteccion_x=config.get('le_proteccion_x', 0.05)
        )

        # Permite mover TE fuera de y=0 mediante una variación de camber hacia TE.
        te_shift = random.gauss(0.0, float(config.get('te_camber_shift_std', 0.0)))
        camber += te_shift * (x_norm ** 2)

    # Forzar espesor de TE dentro de rango físico configurable.
    te_obj = float(espesor[-1])
    if perturbar:
        te_obj += random.gauss(0.0, float(config.get('te_espesor_shift_std', 0.0)))
    te_obj = float(np.clip(te_obj, restricciones['te_gap_min'], restricciones['te_gap_max']))

    w_te = _smoothstep((x_norm - 0.82) / 0.18)
    espesor += (te_obj - espesor[-1]) * w_te

    # Mantener una zona de nariz suave anclada al perfil de referencia local.
    le_zone = max(1e-6, float(config.get('le_proteccion_x', 0.05)))
    w_le = 1.0 - _smoothstep(x_norm / le_zone)
    camber = camber * (1.0 - w_le) + camber_ref * w_le
    espesor = espesor * (1.0 - w_le) + espesor_ref * w_le

    camber = _suavizar_vector_1d(camber, n_pasadas=2)
    espesor = _suavizar_vector_1d(espesor, n_pasadas=2)

    # Espesor mínimo global (excepto LE), para evitar cruces extrados/intrados.
    esp_min = float(config.get('espesor_min_global', 0.0))
    piso = esp_min * _smoothstep(x_norm / le_zone)
    piso[0] = 0.0
    espesor = np.maximum(espesor, piso)
    espesor[0] = 0.0
    espesor[-1] = te_obj

    y_up_common = camber + 0.5 * espesor
    y_lo_common = camber - 0.5 * espesor

    upper = genes[:le_idx + 1]   # TE -> LE
    lower = genes[le_idx:]       # LE -> TE
    upper_x_inc = _asegurar_x_estrictamente_creciente(upper[::-1, 0])
    lower_x_inc = _asegurar_x_estrictamente_creciente(lower[:, 0])

    y_up_inc = np.interp(upper_x_inc, x_common, y_up_common)
    y_lo_inc = np.interp(lower_x_inc, x_common, y_lo_common)

    y_le = 0.5 * (y_up_inc[0] + y_lo_inc[0])
    y_up_inc[0] = y_le
    y_lo_inc[0] = y_le

    resultado = genes.copy()
    resultado[:le_idx + 1, 1] = y_up_inc[::-1]
    resultado[le_idx:, 1] = y_lo_inc
    resultado[le_idx, 1] = y_le

    # Preserva suavidad/circularidad local del LE mezclando con geometría original.
    n_preservar = max(1, int(config.get('le_puntos_preservar', 5)))
    i_ini = max(0, le_idx - n_preservar)
    i_fin = min(len(resultado) - 1, le_idx + n_preservar)
    denom = float(max(1, n_preservar))
    for i in range(i_ini, i_fin + 1):
        dist = abs(i - le_idx) / denom
        # Peso 1.0 en LE, decrece suavemente hacia el borde de la ventana.
        w = (1.0 - min(1.0, dist)) ** 2
        resultado[i, 1] = (1.0 - w) * resultado[i, 1] + w * genes[i, 1]

    # El punto LE queda exactamente fijo.
    resultado[le_idx, :] = genes[le_idx, :]

    # TE recto/romo con x fijo y camber libre.
    x_te = float(np.clip(resultado[0, 0], x_common[0], x_common[-1]))
    camber_te = float(np.interp(x_te, x_common, camber))
    resultado[0, 1] = camber_te + 0.5 * te_obj
    resultado[-1, 1] = camber_te - 0.5 * te_obj

    return resultado


def mutar_perfil_parametrico(genes, le_idx, te_indices, restricciones, config):
    """Mutación principal en espacio camber/espesor con restricciones físicas."""
    _ = te_indices  # se conserva firma para compatibilidad con el pipeline actual
    return proyectar_perfil_parametrico(
        genes, le_idx, restricciones, config, perturbar=True
    )


def validar_geometria_perfil(puntos, le_idx, restricciones, config):
    """
    Valida integridad geométrica del perfil antes de IA/CFD.
    """
    if puntos is None or len(puntos) < 4:
        return False, {'motivo': 'perfil_corto'}

    if not np.all(np.isfinite(puntos)):
        return False, {'motivo': 'nan_inf'}

    if le_idx <= 0 or le_idx >= len(puntos) - 1:
        return False, {'motivo': 'le_idx_invalido'}

    upper_x = puntos[:le_idx + 1, 0]
    lower_x = puntos[le_idx:, 0]
    if np.any(np.diff(upper_x) > 1e-8):
        return False, {'motivo': 'upper_no_monotona'}
    if np.any(np.diff(lower_x) < -1e-8):
        return False, {'motivo': 'lower_no_monotona'}

    te_gap = abs(float(puntos[0, 1] - puntos[-1, 1]))
    te_tol = float(config.get('te_validacion_tol', 1e-9))
    if (te_gap < (restricciones['te_gap_min'] - te_tol)
            or te_gap > (restricciones['te_gap_max'] + te_tol)):
        return False, {'motivo': 'te_fuera_rango', 'te_gap': te_gap}

    x_common, _, espesor = descomponer_camber_espesor(
        puntos, le_idx, n_muestras=restricciones['n_muestras']
    )
    if x_common is None:
        return False, {'motivo': 'descomposicion_fallida'}

    x_norm = (x_common - x_common[0]) / max(1e-12, x_common[-1] - x_common[0])
    mask = x_norm > 0.03
    if np.any(mask):
        espesor_min = float(np.min(espesor[mask]))
    else:
        espesor_min = float(np.min(espesor[1:])) if len(espesor) > 1 else 0.0

    if espesor_min <= max(1e-8, float(config.get('espesor_min_global', 0.0)) * 0.8):
        return False, {'motivo': 'cruce_superficies', 'espesor_min': espesor_min}

    radio_le = estimar_radio_le(puntos, le_idx)
    if (not np.isfinite(radio_le)
            or radio_le < float(restricciones['le_radius_min'])):
        return False, {'motivo': 'le_agudo', 'radio_le': float(radio_le)}

    return True, {
        'te_gap': te_gap,
        'espesor_min': espesor_min,
        'radio_le': float(radio_le),
    }


def reparar_leading_edge_agudo(genes_candidato, genes_referencia, le_idx,
                               restricciones, config):
    """
    Intenta reparar un LE agudo mezclando localmente con una referencia suave.
    """
    candidato = genes_candidato.copy()
    ref = genes_referencia
    n = len(candidato)
    n_preservar = max(1, int(config.get('le_puntos_preservar', 5)))
    i_ini = max(0, le_idx - n_preservar)
    i_fin = min(n - 1, le_idx + n_preservar)
    denom = float(max(1, n_preservar))

    # Escala de mezcla creciente para forzar radios LE más suaves.
    for alpha in (0.20, 0.35, 0.50, 0.65, 0.80):
        reparado = candidato.copy()
        for i in range(i_ini, i_fin + 1):
            dist = abs(i - le_idx) / denom
            w = alpha * ((1.0 - min(1.0, dist)) ** 2)
            reparado[i, 1] = (1.0 - w) * reparado[i, 1] + w * ref[i, 1]

        reparado[le_idx, :] = ref[le_idx, :]
        ok, diag = validar_geometria_perfil(
            reparado, le_idx, restricciones, config
        )
        if ok:
            return reparado, True, diag

        candidato = reparado

    ok_final, diag_final = validar_geometria_perfil(
        candidato, le_idx, restricciones, config
    )
    return candidato, ok_final, diag_final


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


def mutar_y_validar_hijo(genes_hijo, le_idx, te_indices, restricciones, config,
                         aplicar_mutacion=True):
    """
    Aplica mutación según el modo configurado y valida restricciones geométricas.

    Returns:
        (genes_resultado, valido, modo_usado, diagnostico)
    """
    modo = str(config.get('modo_mutacion', 'legacy')).lower()

    if modo == 'parametrica':
        genes_param = proyectar_perfil_parametrico(
            genes_hijo, le_idx, restricciones, config,
            perturbar=aplicar_mutacion
        )
        valido, diag = validar_geometria_perfil(
            genes_param, le_idx, restricciones, config
        )
        if valido:
            return genes_param, True, 'parametrica', diag

        if isinstance(diag, dict) and diag.get('motivo') == 'le_agudo':
            genes_le_rep, ok_le_rep, diag_le_rep = reparar_leading_edge_agudo(
                genes_param, genes_hijo, le_idx, restricciones, config
            )
            if ok_le_rep:
                return genes_le_rep, True, 'le_reparada', diag_le_rep
            genes_param = genes_le_rep
            diag = diag_le_rep

        # Intento de reparación: reproyección sin perturbaciones adicionales.
        genes_reparado = proyectar_perfil_parametrico(
            genes_param, le_idx, restricciones, config,
            perturbar=False
        )
        valido_rep, diag_rep = validar_geometria_perfil(
            genes_reparado, le_idx, restricciones, config
        )
        if valido_rep:
            return genes_reparado, True, 'parametrica_reparada', diag_rep

        if config.get('usar_legacy_fallback', False):
            genes_legacy = genes_hijo.copy()
            if aplicar_mutacion:
                genes_legacy = mutar_perfil(genes_legacy, le_idx, te_indices)
            genes_legacy = suavizar_perfil(
                genes_legacy, le_idx,
                config.get('suavizado_iteraciones', 2)
            )

            # Proyección final sin perturbar para reimponer constraints duros.
            genes_legacy = proyectar_perfil_parametrico(
                genes_legacy, le_idx, restricciones, config,
                perturbar=False
            )

            valido_fb, diag_fb = validar_geometria_perfil(
                genes_legacy, le_idx, restricciones, config
            )
            return genes_legacy, valido_fb, 'legacy_fallback', diag_fb

        return genes_param, False, 'parametrica_invalida', diag

    # Modo legacy puro
    genes_legacy = genes_hijo.copy()
    if aplicar_mutacion:
        genes_legacy = mutar_perfil(genes_legacy, le_idx, te_indices)
    genes_legacy = suavizar_perfil(
        genes_legacy, le_idx,
        config.get('suavizado_iteraciones', 2)
    )

    # Si hay restricciones definidas, se valida igualmente antes de CFD.
    if restricciones is not None:
        genes_legacy = proyectar_perfil_parametrico(
            genes_legacy, le_idx, restricciones, config,
            perturbar=False
        )
        valido, diag = validar_geometria_perfil(
            genes_legacy, le_idx, restricciones, config
        )
        return genes_legacy, valido, 'legacy', diag

    return genes_legacy, True, 'legacy', {}


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
    mesh_gruesa = None

    try:
        # Construir parámetros del simulador (config de referencia validada)
        # 'consistent' fuerza internamente one_sided/face_flux/reforzar OFF.
        sim_params = {
            'filepath': filepath_temp,
            'iteraciones': config['simulacion_iteraciones'],
            'v0x': config['v0x'],
            'v0y': 0.0,
            'CFL': config['CFL'],
            'dx_min': config['dx_min'],
            'alpha_deg': alpha_deg,
            'chord': config['chord'],
            'rho': config.get('rho', 1.0),
            'nu': config.get('nu', 1e-5),
            'turb_model': config.get('turb_model', 'sa'),
            'wall_treatment': config.get('wall_treatment', 'consistent'),
            'advection_scheme': config.get('advection_scheme', 'maccormack'),
            'min_te_height_factor': config.get('min_te_height_factor', 1.0),
            'wake_refinement_mode': config.get('wake_refinement_mode', 'long_fine_x'),
            'mg_niveles_max': config.get('mg_niveles_max', 2),
            'mg_max_outer': config.get('mg_max_outer', 8),
            'divergencia': config.get('divergencia', 0.02),
            'stop_on_clcd_convergence': config.get('stop_on_clcd_convergence', True),
            'graficos': False,
            'live_view': False,
            'mostrar_malla': False,
        }

        # Parámetros extra opcionales del usuario (mesh/dominio)
        sim_params.update(config.get('sim_extra_params', {}))

        mesh_gruesa = Simulador2D.main(**sim_params)

        # Cl/Cd se almacenan solo en mesh_gruesa (bucle principal)
        if mesh_gruesa.cdvector is None or len(mesh_gruesa.cdvector) == 0:
            return None

        # Promediar sobre la ventana del último 20% (estacionario)
        n_datos = len(mesh_gruesa.cdvector)
        w = max(1, n_datos // 5)

        cd_arr = mesh_gruesa.cdvector[-w:]
        cl_arr = mesh_gruesa.clvector[-w:]

        cd_val = float(cp.mean(cd_arr).get())
        cl_val = float(cp.mean(cl_arr).get())
        cd_std = float(cp.std(cd_arr).get())
        cl_std = float(cp.std(cl_arr).get())

        # Validar: descartar NaN/Inf
        if (np.isnan(cd_val) or np.isnan(cl_val)
                or np.isinf(cd_val) or np.isinf(cl_val)):
            print(f"   [!] Resultado NaN/Inf @ alpha={alpha_deg}°")
            return None

        cl_force = cl_val
        try:
            rho = float(sim_params.get('rho', 1.225))
            nu = float(sim_params.get('nu', 1.5e-5))
            audit = mesh_gruesa.extract_surface_force_audit(
                mu=rho * nu,
                rho=rho,
                chord=float(sim_params.get('chord', 1.0)),
                n_extrap_layers=5,
            )
            val = float(audit.get("summary", {}).get("Cl_from_Cp_force_consistent", cl_val))
            if np.isfinite(val):
                cl_force = val
        except Exception:
            pass

        ld = cl_force / cd_val if abs(cd_val) > 1e-6 else 0.0

        return {
            'cl': round(cl_force, 6), 'cd': round(cd_val, 6), 'ld': round(ld, 4),
            'cl_raw': round(cl_val, 6),
            'cl_std': round(cl_std, 6), 'cd_std': round(cd_std, 6),
            'n_samples': int(len(cd_arr)),
        }

    except Exception as e:
        print(f"   [!] Error simulación @ alpha={alpha_deg}°: {e}")
        return None

    finally:
        # Liberar memoria GPU agresivamente
        del mesh_gruesa
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


def evaluar_poblacion(poblacion, gen, angulos, config, oraculo, logger, condiciones,
                     le_idx=None, restricciones=None):
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
    n_descartados_geom = 0

    for i, ind in enumerate(poblacion):
        if PARADA_SOLICITADA:
            break

        if ind.fitness != 0:
            continue  # Ya evaluado (elite de generación anterior)

        # Filtro geométrico duro (previo a IA y CFD)
        if le_idx is not None and restricciones is not None:
            valido_geom, _ = validar_geometria_perfil(
                ind.genes, le_idx, restricciones, config
            )
            if not valido_geom:
                ind.fitness = 0.0
                n_descartados_geom += 1
                continue

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
            # Parametrización geométrica compacta para IA
            _param = extraer_parametrizacion(ind.genes, le_idx) if le_idx is not None else {}

            # Stats de convergencia agregadas sobre todos los ángulos evaluados
            _cl_stds = [r['cl_std'] for r in resultados.values() if 'cl_std' in r]
            _cd_stds = [r['cd_std'] for r in resultados.values() if 'cd_std' in r]
            _ns = [r['n_samples'] for r in resultados.values() if 'n_samples' in r]
            _conv = {
                'cl_std_mean': round(float(np.mean(_cl_stds)), 6) if _cl_stds else None,
                'cd_std_mean': round(float(np.mean(_cd_stds)), 6) if _cd_stds else None,
                'n_samples': int(np.mean(_ns)) if _ns else None,
            }

            # Resultados limpios (sin campos de convergencia inline)
            resultados_limpios = {
                ang: {k: v for k, v in r.items() if k in ('cl', 'cd', 'ld')}
                for ang, r in resultados.items()
            }

            logger.registrar(
                perfil_puntos=ind.genes,
                condiciones=condiciones,
                resultados_por_angulo=resultados_limpios,
                metadata={
                    'generacion': gen,
                    'individuo': i,
                    'fitness': ind.fitness,
                    'perfil_base': config['archivo_base'],
                    'multi_angulo': config['multi_angulo'],
                    'parametrizacion': _param,
                    'convergencia': _conv,
                }
            )

    print(f"\n Evaluación: {n_evaluados} simulados, "
        f"{n_descartados_ia} descartados por IA, "
        f"{n_descartados_geom} descartados por geometría")

    return n_evaluados, n_descartados_ia, n_descartados_geom


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
def main(config=None):
    """Ejecuta el GA. `config` (dict opcional) sobreescribe CONFIG para esta corrida
    (permite lanzar N corridas distintas en un mismo proceso). Devuelve un resumen."""
    global PARADA_SOLICITADA
    PARADA_SOLICITADA = False
    if config:
        CONFIG.update(config)
    try:
        signal.signal(signal.SIGINT, manejar_parada)
    except ValueError:
        pass  # signal solo funciona en el hilo principal

    print("\n" + "=" * 60)
    print(" OPTIMIZADOR GENÉTICO DE PERFILES AERODINÁMICOS")
    print("=" * 60)
    print(f" Fecha:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Perfil base:  {CONFIG['archivo_base']}")
    print(f" Velocidad:    {CONFIG['v0x']} m/s")
    print(f" Alpha base:   {CONFIG['alpha_deg']}°")
    print(f" Población:    {CONFIG['poblacion_tamano']} individuos")
    print(f" Generaciones: {CONFIG['generaciones']} (máximo)")

    # Resolver directorio automático antes de usarlo
    if CONFIG.get('directorio_resultados') == 'auto':
        CONFIG['directorio_resultados'] = generar_directorio_resultados(CONFIG)
    print(f" Directorio:   {CONFIG['directorio_resultados']}")

    # Crear directorio de resultados
    os.makedirs(CONFIG['directorio_resultados'], exist_ok=True)

    # =====================
    # PASO 1: Cargar perfil(es) base
    # =====================
    #   Si CONFIG['archivos_base'] es una lista, se usa la 1ª como referencia de
    #   rejilla y el resto se reproyecta sobre ella -> población mixta (Arm B).
    _semillas_paths = CONFIG.get('archivos_base')
    _ref_path = (_semillas_paths[0] if _semillas_paths else CONFIG['archivo_base'])

    coords_base, header_base, le_idx, te_indices = cargar_perfil(_ref_path)
    if len(coords_base) == 0:
        print("\nERROR CRÍTICO: No se encontró el archivo de perfil base.")
        return

    restricciones_geom = construir_restricciones_geometricas(
        coords_base, le_idx, CONFIG
    )

    # Semillas adicionales reproyectadas a la rejilla de referencia
    semillas_genes = [coords_base.copy()]
    if _semillas_paths:
        print(f"\n Población mixta: {len(_semillas_paths)} semillas -> rejilla de "
              f"{os.path.basename(_ref_path)} ({len(coords_base)} puntos)")
        for _sp in _semillas_paths[1:]:
            _sc, _sh, _sle, _ste = cargar_perfil(_sp)
            if len(_sc) == 0:
                print(f"   [!] Semilla no cargada: {_sp}")
                continue
            try:
                semillas_genes.append(
                    resamplear_a_grid(_sc, _sle, coords_base, le_idx)
                )
            except Exception as e:
                print(f"   [!] Fallo reproyectando {_sp}: {e}")
    print("\n Restricciones geométricas activas:")
    print(f"   TE espesor base: {restricciones_geom['te_base']:.6f}")
    print(f"   TE espesor rango: [{restricciones_geom['te_gap_min']:.6f}, "
          f"{restricciones_geom['te_gap_max']:.6f}]")
    print(f"   LE radio base: {restricciones_geom['le_radio_base']:.6f}")
    print(f"   LE radio mínimo: {restricciones_geom['le_radius_min']:.6f}")

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
        'dx_min': CONFIG['dx_min'],
        'iteraciones_cfd': CONFIG['simulacion_iteraciones'],
        'fitness_modo': CONFIG['fitness_modo'],
    }

    # =====================
    # PASO 2b: Evaluar perfil base (referencia para comparativas)
    # =====================
    print("\n Evaluando perfil base (referencia)...")
    _resultados_base = {}
    _nombre_base_tmp = "_temp_base_eval.dat"
    guardar_perfil(_nombre_base_tmp, coords_base, header_base)
    try:
        for _alpha in angulos:
            _res = simular_perfil(_nombre_base_tmp, _alpha, CONFIG)
            if _res is not None:
                _resultados_base[f"{_alpha:.1f}"] = _res
                print(f"   @{_alpha:.1f}°: Cl={_res['cl']:.4f}  "
                      f"Cd={_res['cd']:.5f}  L/D={_res['ld']:.2f}")
            else:
                print(f"   @{_alpha:.1f}°: FALLO")
    finally:
        if os.path.exists(_nombre_base_tmp):
            os.remove(_nombre_base_tmp)

    fitness_base = calcular_fitness(_resultados_base, CONFIG)
    if fitness_base > 0:
        print(f" Fitness BASE: L/D = {fitness_base:.4f}")
    else:
        print(" AVISO: No se pudo evaluar el perfil base. Comparativa desactivada.")
        fitness_base = None

    # =====================
    # PASO 4: Crear población inicial
    # =====================
    print(f"\n Generando población inicial "
          f"({CONFIG['poblacion_tamano']} individuos)...")
    poblacion = []
    descartes_geom_ini = 0
    fallback_legacy_ini = 0
    reparaciones_param_ini = 0
    intentos_ini = 0
    max_intentos_ini = max(
        CONFIG['poblacion_tamano'] * 4,
        CONFIG.get('max_intentos_geometria', 120)
    )

    # Población mixta: sembrar cada forma-semilla (proyectada válida, sin mutar)
    # para que la generación 0 contenga literalmente cada perfil base.
    if len(semillas_genes) > 1:
        for _sg in semillas_genes:
            if len(poblacion) >= CONFIG['poblacion_tamano']:
                break
            try:
                _sg_valido = proyectar_perfil_parametrico(
                    _sg.copy(), le_idx, restricciones_geom, CONFIG,
                    perturbar=False
                )
                poblacion.append(Individuo(_sg_valido, header_base))
            except Exception:
                pass

    while (len(poblacion) < CONFIG['poblacion_tamano']
           and intentos_ini < max_intentos_ini):
        # Round-robin sobre las semillas (una sola en modo clásico)
        genes_semilla = semillas_genes[intentos_ini % len(semillas_genes)].copy()
        genes_nuevo, valido, modo_usado, _ = mutar_y_validar_hijo(
            genes_semilla,
            le_idx,
            te_indices,
            restricciones_geom,
            CONFIG,
            aplicar_mutacion=True
        )
        intentos_ini += 1

        if not valido:
            descartes_geom_ini += 1
            continue

        if modo_usado == 'legacy_fallback':
            fallback_legacy_ini += 1
        elif modo_usado in ('parametrica_reparada', 'le_reparada'):
            reparaciones_param_ini += 1

        poblacion.append(Individuo(genes_nuevo, header_base))

    # En caso extremo, completar con perfil base proyectado para no abortar la corrida.
    while len(poblacion) < CONFIG['poblacion_tamano']:
        genes_base_valido = proyectar_perfil_parametrico(
            coords_base.copy(), le_idx, restricciones_geom, CONFIG,
            perturbar=False
        )
        poblacion.append(Individuo(genes_base_valido, header_base))

    if CONFIG.get('reportar_diagnostico_geometria', True):
        print(f"   Iniciales válidos: {len(poblacion)}")
        print(f"   Iniciales descartados por geometría: {descartes_geom_ini}")
        if reparaciones_param_ini > 0:
            print(f"   Iniciales reparados (paramétrica): {reparaciones_param_ini}")
        if fallback_legacy_ini > 0:
            print(f"   Iniciales usando fallback legacy: {fallback_legacy_ini}")

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
    _tiempo_limite = CONFIG.get('tiempo_limite_s')

    # S1: sigma de mutación adaptativa. Factor grande al inicio (exploración,
    # escape de cuenca) que decae a 1.0 (refinamiento). Ataca el "no convergen"
    # por hill-climb local con mutación diminuta (camber_mut_std=0.0035).
    _sigma_adaptativa = CONFIG.get('sigma_adaptativa', False)
    _sigma_keys = ('camber_mut_std', 'espesor_mut_std',
                   'te_camber_shift_std', 'te_espesor_shift_std')
    _sigma_base = {k: float(CONFIG[k]) for k in _sigma_keys}
    _sigma_f0 = float(CONFIG.get('sigma_factor_inicial', 3.0))
    _sigma_f1 = float(CONFIG.get('sigma_factor_final', 1.0))

    for gen in range(CONFIG['generaciones']):
        gen_actual = gen

        if _sigma_adaptativa:
            _g = CONFIG['generaciones']
            _frac = gen / max(1, _g - 1)
            _factor = _sigma_f0 + (_sigma_f1 - _sigma_f0) * _frac
            for _k in _sigma_keys:
                CONFIG[_k] = _sigma_base[_k] * _factor
            print(f"   [S1] sigma_factor={_factor:.2f} "
                  f"(camber_std={CONFIG['camber_mut_std']:.4f})")

        if PARADA_SOLICITADA:
            print("\n Parada solicitada. Saliendo del bucle evolutivo...")
            break

        if _tiempo_limite is not None and (time.time() - t_total_inicio) > _tiempo_limite:
            print(f"\n Deadline alcanzado ({_tiempo_limite:.0f}s). Cerrando tras "
                  f"{gen} generaciones completas...")
            break

        t_gen_inicio = time.time()

        print(f"\n{'=' * 55}")
        print(f" GENERACIÓN {gen + 1} / {CONFIG['generaciones']}")
        print(f"{'=' * 55}")

        # --- A. EVALUACIÓN CFD ---
        n_eval, n_desc, n_desc_geom_eval = evaluar_poblacion(
            poblacion, gen, angulos, CONFIG, oraculo, logger, condiciones,
            le_idx=le_idx,
            restricciones=restricciones_geom
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
        if n_desc_geom_eval > 0:
            print(f"   Geometría descartes (pre-CFD): {n_desc_geom_eval}")
        oraculo.conteo_descartes = 0

        # Comparativa con perfil base
        _delta_base = None
        if fitness_base is not None:
            _delta_base = mejor_gen.fitness - fitness_base
            _pct_base = (_delta_base / fitness_base) * 100
            _signo = "+" if _delta_base >= 0 else ""
            _tag = "MEJOR" if _delta_base >= 0 else "PEOR ⚠"
            print(f"   vs BASE:       {_signo}{_delta_base:.4f} "
                  f"({_signo}{_pct_base:.1f}%) [{_tag}]")

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
            'descartados_geom_eval': n_desc_geom_eval,
            'tiempo_seg': round(t_gen, 1),
            'vs_base': round(_delta_base, 6) if _delta_base is not None else None,
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
            # Checkpoint de seguridad (se sobreescribe en cada récord)
            guardar_perfil(
                os.path.join(CONFIG['directorio_resultados'],
                             "OPTIMO_PARCIAL.dat"),
                mejor_global.genes, mejor_global.header
            )
            # Copia nombrada en records/ — una por cada nuevo récord
            _stem_rec, _lin_rec = generar_stem_optimizado(
                CONFIG, mejor_global.fitness
            )
            guardar_perfil_con_metadata(
                os.path.join(CONFIG['directorio_resultados'], 'records'),
                _stem_rec, mejor_global.genes, CONFIG,
                mejor_global.fitness, mejor_global.resultados, _lin_rec,
                historial=historial_fitness,
            )
            print(f"   Record guardado: records/{_stem_rec}.dat")

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
        MAX_INTENTOS = max(80, int(CONFIG.get('max_intentos_geometria', 120)))
        descartes_geom_repro = 0
        fallback_legacy_repro = 0
        reparaciones_param_repro = 0

        while len(nueva_poblacion) < CONFIG['poblacion_tamano']:
            if PARADA_SOLICITADA:
                break

            if intentos_fallidos >= MAX_INTENTOS:
                # Evita bloqueo si el espacio de búsqueda queda demasiado restringido.
                clon = copy.deepcopy(poblacion[0])
                clon.fitness = 0.0
                clon.resultados = {}
                nueva_poblacion.append(clon)
                intentos_fallidos = 0
                continue

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

            aplicar_mutacion = random.random() < CONFIG['prob_mutacion']
            genes_hijo, valido_geom, modo_usado, _ = mutar_y_validar_hijo(
                genes_hijo,
                le_idx,
                te_indices,
                restricciones_geom,
                CONFIG,
                aplicar_mutacion=aplicar_mutacion
            )

            if not valido_geom:
                descartes_geom_repro += 1
                intentos_fallidos += 1
                continue

            if modo_usado == 'legacy_fallback':
                fallback_legacy_repro += 1
            elif modo_usado in ('parametrica_reparada', 'le_reparada'):
                reparaciones_param_repro += 1

            hijo = Individuo(genes_hijo, header_base)

            # Filtro IA pre-evaluación
            if (oraculo.entrenado and CONFIG['usar_ia']
                    and intentos_fallidos < MAX_INTENTOS):
                if not oraculo.predecir_si_vale_la_pena(hijo.genes):
                    intentos_fallidos += 1
                    continue

            nueva_poblacion.append(hijo)
            intentos_fallidos = 0

        if CONFIG.get('reportar_diagnostico_geometria', True):
            if descartes_geom_repro > 0:
                print(f"   Reproducción - descartes geométricos: {descartes_geom_repro}")
            if reparaciones_param_repro > 0:
                print(f"   Reproducción - reparados (paramétrica): {reparaciones_param_repro}")
            if fallback_legacy_repro > 0:
                print(f"   Reproducción - fallback legacy: {fallback_legacy_repro}")

        if historial_fitness:
            historial_fitness[-1]['descartados_geom_repro'] = descartes_geom_repro
            historial_fitness[-1]['reparados_param_repro'] = reparaciones_param_repro
            historial_fitness[-1]['fallback_legacy_repro'] = fallback_legacy_repro

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

        # Comparativa final vs perfil base
        if fitness_base is not None:
            _delta_fin = mejor_global.fitness - fitness_base
            _pct_fin = (_delta_fin / fitness_base) * 100
            if _delta_fin < 0:
                print(f"\n{'!' * 60}")
                print(f"  ADVERTENCIA: EL ÓPTIMO ES PEOR QUE EL PERFIL INICIAL")
                print(f"{'!' * 60}")
                print(f"  Fitness BASE:   {fitness_base:.4f}")
                print(f"  Fitness ÓPTIMO: {mejor_global.fitness:.4f}")
                print(f"  Diferencia:     {_delta_fin:.4f} ({_pct_fin:.1f}%)")
                print(f"  → La evolución no mejoró el perfil de entrada.")
                print(f"{'!' * 60}")
            else:
                print(f"\n  Mejora sobre BASE: +{_delta_fin:.4f} (+{_pct_fin:.1f}%)")

        # Guardar perfil final con nombre descriptivo + metadata
        stem_final, lineage_final = generar_stem_optimizado(
            CONFIG, mejor_global.fitness
        )
        nombre_final = guardar_perfil_con_metadata(
            CONFIG['directorio_resultados'],
            stem_final, mejor_global.genes, CONFIG,
            mejor_global.fitness, mejor_global.resultados, lineage_final,
            historial=historial_fitness,
        )
        print(f"\n Perfil optimo guardado en: {nombre_final}")
        print(f"\n {'=' * 58}")
        print(f"  Para continuar la optimizacion desde este perfil:")
        print(f"  Cambia en CONFIG:")
        print(f"    'archivo_base': '{nombre_final}'")
        print(f"  El linaje se preservara automaticamente.")
        print(f" {'=' * 58}")

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

    return {
        'directorio': CONFIG['directorio_resultados'],
        'mejor_fitness': (mejor_global.fitness if mejor_global else None),
        'mejor_resultados': (mejor_global.resultados if mejor_global else None),
        'fitness_base': fitness_base,
        'generaciones_completadas': gen_actual + 1,
        'tiempo_seg': round(t_total, 1),
        'historial': historial_fitness,
    }


if __name__ == "__main__":
    main()

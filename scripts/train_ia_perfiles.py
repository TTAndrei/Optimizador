"""
train_ia_perfiles.py
=====================
Entrena dos modelos de IA a partir de dataset_ia.csv:

  Modelo A — Surrogate forward
    Input:  [camber_max, camber_pos, espesor_max, espesor_pos, le_radius, te_gap,
             v0x, alpha, Re]
    Output: [Cl, Cd]
    Uso: validar calidad del dataset, exploración rápida sin CFD

  Modelo B — Predictor inverso
    Input:  [v0x, alpha, Re, Cl_target, Cd_target]
    Output: [camber_max, camber_pos, espesor_max, espesor_pos, le_radius, te_gap]
    Uso: dado un punto de operación y coeficientes deseados → parámetros geométricos

Arquitectura: MLP sklearn (MLPRegressor) — sin dependencias externas adicionales.
Si PyTorch está disponible, se usa para el modelo inverso (mejor manejo de
problemas one-to-many con dropout).

Guardado:
    data/modelo_surrogate.pkl   — surrogate forward
    data/modelo_inverso.pkl     — predictor inverso
    data/scaler_surrogate.pkl   — scalers del surrogate
    data/scaler_inverso.pkl     — scalers del inverso

Uso:
    python scripts/train_ia_perfiles.py
    python scripts/train_ia_perfiles.py --input data/dataset_ia.csv
    python scripts/train_ia_perfiles.py --solo-surrogate
    python scripts/train_ia_perfiles.py --solo-inverso
    python scripts/train_ia_perfiles.py --evaluar  (carga modelos existentes y evalúa)
"""

import argparse
import json
import os
import pickle
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')

# ==========================================
# FEATURES Y TARGETS
# ==========================================
GEOM_FEATURES = ['camber_max', 'camber_pos', 'espesor_max', 'espesor_pos', 'le_radius', 'te_gap']
COND_FEATURES = ['v0x', 'alpha', 'Re']

# Surrogate: (geom + cond) → (Cl, Cd)
SURROGATE_X_COLS = GEOM_FEATURES + COND_FEATURES
SURROGATE_Y_COLS = ['cl', 'cd']

# Inverso: (cond + targets) → geom
INVERSO_X_COLS = COND_FEATURES + ['cl', 'cd']
INVERSO_Y_COLS = GEOM_FEATURES

INPUT_DEFAULT = 'data/dataset_ia.csv'
MODEL_DIR = 'data'


# ==========================================
# CARGA DE DATOS
# ==========================================
def cargar_dataset(path):
    import csv
    filas = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filas.append(row)

    if not filas:
        raise ValueError(f"Dataset vacío: {path}")

    todas_cols = list(filas[0].keys())
    cols_necesarias = set(SURROGATE_X_COLS + SURROGATE_Y_COLS)
    faltantes = cols_necesarias - set(todas_cols)
    if faltantes:
        raise ValueError(f"Columnas faltantes en CSV: {faltantes}")

    data = {}
    for col in todas_cols:
        vals = []
        for f in filas:
            v = f.get(col, '')
            try:
                vals.append(float(v) if v not in ('', 'None', 'nan', 'null') else np.nan)
            except ValueError:
                vals.append(np.nan)
        data[col] = np.array(vals)

    return data, len(filas)


def preparar_matrices(data, x_cols, y_cols):
    """Filtra NaN y construye matrices X, y."""
    idx_validos = np.ones(len(data[x_cols[0]]), dtype=bool)
    for col in x_cols + y_cols:
        if col in data:
            idx_validos &= np.isfinite(data[col])

    X = np.column_stack([data[col][idx_validos] for col in x_cols])
    y = np.column_stack([data[col][idx_validos] for col in y_cols])
    if y.shape[1] == 1:
        y = y.ravel()

    return X, y, int(np.sum(idx_validos))


# ==========================================
# ENTRENAMIENTO SKLEARN
# ==========================================
def entrenar_mlp_sklearn(X_train, y_train, X_test, y_test, nombre, y_cols):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error

    scaler_X = StandardScaler()
    X_tr_s = scaler_X.fit_transform(X_train)
    X_te_s = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_tr_s = scaler_y.fit_transform(y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train)

    modelo = MLPRegressor(
        hidden_layer_sizes=(128, 128, 64),
        activation='relu',
        solver='adam',
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=42,
        verbose=False,
    )

    modelo.fit(X_tr_s, y_tr_s)

    y_pred_s = modelo.predict(X_te_s)
    y_pred = scaler_y.inverse_transform(
        y_pred_s.reshape(-1, 1) if y_pred_s.ndim == 1 else y_pred_s
    )
    y_te = y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test

    print(f"\n  [{nombre}] Test set ({len(y_te)} muestras):")
    for i, col in enumerate(y_cols):
        r2 = r2_score(y_te[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_te[:, i], y_pred[:, i])
        print(f"    {col:15s}: R²={r2:.4f}  MAE={mae:.5f}")

    return modelo, scaler_X, scaler_y


# ==========================================
# GUARDAR / CARGAR MODELOS
# ==========================================
def guardar_modelo(modelo, scaler_X, scaler_y, nombre_base, meta=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    obj = {
        'modelo': modelo,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'meta': meta or {},
    }
    path = os.path.join(MODEL_DIR, f'{nombre_base}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"  Guardado: {path}")
    return path


def cargar_modelo(nombre_base):
    path = os.path.join(MODEL_DIR, f'{nombre_base}.pkl')
    if not os.path.exists(path):
        return None, None, None, None
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj['modelo'], obj['scaler_X'], obj['scaler_y'], obj.get('meta', {})


# ==========================================
# PREDICCIÓN
# ==========================================
def predecir_surrogate(v0x, alpha, Re, camber_max, camber_pos, espesor_max,
                        espesor_pos, le_radius, te_gap):
    """Predice (Cl, Cd) dado geometría + condiciones."""
    modelo, scaler_X, scaler_y, meta = cargar_modelo('modelo_surrogate')
    if modelo is None:
        raise FileNotFoundError("modelo_surrogate.pkl no encontrado. Entrena primero.")
    x = np.array([[camber_max, camber_pos, espesor_max, espesor_pos,
                   le_radius, te_gap, v0x, alpha, Re]])
    x_s = scaler_X.transform(x)
    y_s = modelo.predict(x_s)
    y = scaler_y.inverse_transform(y_s.reshape(1, -1))
    cl, cd = float(y[0, 0]), float(y[0, 1])
    ld = cl / cd if abs(cd) > 1e-6 else 0.0
    return {'cl': round(cl, 6), 'cd': round(cd, 6), 'ld': round(ld, 4)}


def predecir_inverso(v0x, alpha, Re, cl_target, cd_target):
    """Predice parámetros geométricos dado condiciones + coeficientes target."""
    modelo, scaler_X, scaler_y, meta = cargar_modelo('modelo_inverso')
    if modelo is None:
        raise FileNotFoundError("modelo_inverso.pkl no encontrado. Entrena primero.")
    x = np.array([[v0x, alpha, Re, cl_target, cd_target]])
    x_s = scaler_X.transform(x)
    y_s = modelo.predict(x_s)
    y = scaler_y.inverse_transform(y_s.reshape(1, -1))
    resultado = {col: float(y[0, i]) for i, col in enumerate(INVERSO_Y_COLS)}
    return resultado


# ==========================================
# EVALUACIÓN DE MODELOS EXISTENTES
# ==========================================
def evaluar_modelos(data):
    print("\n EVALUANDO modelos existentes...")

    for nombre, x_cols, y_cols in [
        ('modelo_surrogate', SURROGATE_X_COLS, SURROGATE_Y_COLS),
        ('modelo_inverso', INVERSO_X_COLS, INVERSO_Y_COLS),
    ]:
        modelo, scaler_X, scaler_y, meta = cargar_modelo(nombre)
        if modelo is None:
            print(f"  [{nombre}] NO encontrado.")
            continue

        from sklearn.metrics import r2_score, mean_absolute_error
        X, y, n = preparar_matrices(data, x_cols, y_cols)
        if n < 5:
            print(f"  [{nombre}] Insuficientes datos ({n}).")
            continue

        X_s = scaler_X.transform(X)
        y_pred_s = modelo.predict(X_s)
        y_pred = scaler_y.inverse_transform(
            y_pred_s.reshape(-1, 1) if y_pred_s.ndim == 1 else y_pred_s
        )
        y = y.reshape(-1, 1) if y.ndim == 1 else y

        print(f"\n  [{nombre}]  n={n}  entrenado: {meta.get('fecha', '?')}")
        for i, col in enumerate(y_cols):
            r2 = r2_score(y[:, i], y_pred[:, i])
            mae = mean_absolute_error(y[:, i], y_pred[:, i])
            print(f"    {col:15s}: R²={r2:.4f}  MAE={mae:.5f}")


# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Entrena surrogate + modelo inverso de perfiles')
    parser.add_argument('--input', default=INPUT_DEFAULT)
    parser.add_argument('--solo-surrogate', action='store_true')
    parser.add_argument('--solo-inverso', action='store_true')
    parser.add_argument('--evaluar', action='store_true', help='Solo evaluar modelos existentes')
    parser.add_argument('--test-frac', type=float, default=0.2, help='Fracción test (default 0.20)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} no encontrado. Ejecuta build_dataset.py primero.")
        return

    print(f"\n Cargando dataset: {args.input}")
    data, n_total = cargar_dataset(args.input)
    print(f" {n_total} registros en dataset")

    if args.evaluar:
        evaluar_modelos(data)
        return

    from sklearn.model_selection import train_test_split

    fecha = __import__('datetime').datetime.now().isoformat()

    # ── SURROGATE FORWARD ─────────────────────────────────────
    if not args.solo_inverso:
        print("\n" + "=" * 55)
        print(" MODELO A — SURROGATE FORWARD  (geom+cond → Cl,Cd)")
        print("=" * 55)

        X, y, n = preparar_matrices(data, SURROGATE_X_COLS, SURROGATE_Y_COLS)
        print(f" Muestras válidas: {n}")

        if n < 20:
            print(" ADVERTENCIA: menos de 20 muestras. Surrogate poco fiable.")
            print(" Ejecuta el barrido de condiciones para generar más datos.")
        else:
            idx = np.arange(n)
            tr_idx, te_idx = train_test_split(idx, test_size=args.test_frac, random_state=42)
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            modelo, scaler_X, scaler_y = entrenar_mlp_sklearn(
                X_tr, y_tr, X_te, y_te, 'Surrogate', SURROGATE_Y_COLS
            )
            guardar_modelo(modelo, scaler_X, scaler_y, 'modelo_surrogate', meta={
                'fecha': fecha,
                'n_train': len(tr_idx),
                'n_test': len(te_idx),
                'x_cols': SURROGATE_X_COLS,
                'y_cols': SURROGATE_Y_COLS,
            })

    # ── PREDICTOR INVERSO ─────────────────────────────────────
    if not args.solo_surrogate:
        print("\n" + "=" * 55)
        print(" MODELO B — PREDICTOR INVERSO  (cond+Cl,Cd → geom)")
        print("=" * 55)
        print(" NOTA: el espacio inverso es one-to-many.")
        print("       El modelo aprende la tendencia media del dataset.")

        X, y, n = preparar_matrices(data, INVERSO_X_COLS, INVERSO_Y_COLS)
        print(f" Muestras válidas: {n}")

        if n < 20:
            print(" ADVERTENCIA: menos de 20 muestras. Modelo inverso no entrenable.")
            print(" Ejecuta el barrido de condiciones para generar más datos.")
        else:
            idx = np.arange(n)
            tr_idx, te_idx = train_test_split(idx, test_size=args.test_frac, random_state=42)
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            modelo, scaler_X, scaler_y = entrenar_mlp_sklearn(
                X_tr, y_tr, X_te, y_te, 'Inverso', INVERSO_Y_COLS
            )
            guardar_modelo(modelo, scaler_X, scaler_y, 'modelo_inverso', meta={
                'fecha': fecha,
                'n_train': len(tr_idx),
                'n_test': len(te_idx),
                'x_cols': INVERSO_X_COLS,
                'y_cols': INVERSO_Y_COLS,
            })

    print("\n Entrenamiento completado.")


if __name__ == '__main__':
    main()

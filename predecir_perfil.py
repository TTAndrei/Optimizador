"""
predecir_perfil.py
===================
CLI principal: dado un punto de operación y coeficientes objetivo,
predice y genera el perfil aerodinámico más adecuado.

Flujo:
  1. Carga modelo_inverso.pkl  (cond + Cl_target + Cd_target → parámetros geom)
  2. Predice parámetros geométricos
  3. Reconstruye geometría como array de puntos Selig
  4. Guarda perfil_predicho.dat (y .png si --plot)
  5. Verifica con surrogate forward (muestra Cl/Cd estimados por IA)

Uso:
    python predecir_perfil.py --v 1 --alpha 4 --Re 100000 --cl 0.6 --cd 0.05
    python predecir_perfil.py --v 5 --alpha 8 --Re 500000 --cl 0.9 --cd 0.03 --output mi_perfil.dat
    python predecir_perfil.py --v 1 --alpha 4 --Re 100000 --cl 0.6 --cd 0.05 --plot
    python predecir_perfil.py --info   (muestra info de los modelos disponibles)

Notas:
  - Requiere haber ejecutado scripts/train_ia_perfiles.py primero.
  - La calidad de la predicción depende del tamaño y diversidad del dataset.
  - Con 84 registros a condición única: predicciones limitadas fuera del rango entrenado.
  - Con ≥200 registros de script_barrido_conditions.py: predicciones útiles.
"""

import argparse
import os
import sys

import numpy as np

# Añadir scripts/ al path para imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))


def cargar_modelo(nombre_base):
    import pickle
    path = os.path.join('data', f'{nombre_base}.pkl')
    if not os.path.exists(path):
        return None, None, None, None
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj['modelo'], obj['scaler_X'], obj['scaler_y'], obj.get('meta', {})


def predecir_geom(v0x, alpha, Re, cl_target, cd_target):
    modelo, scaler_X, scaler_y, meta = cargar_modelo('modelo_inverso')
    if modelo is None:
        raise FileNotFoundError(
            "data/modelo_inverso.pkl no encontrado.\n"
            "Ejecuta: python scripts/train_ia_perfiles.py"
        )
    x = np.array([[v0x, alpha, Re, cl_target, cd_target]])
    x_s = scaler_X.transform(x)
    y_s = modelo.predict(x_s)
    y = scaler_y.inverse_transform(y_s.reshape(1, -1))
    cols = meta.get('y_cols', ['camber_max', 'camber_pos', 'espesor_max',
                                'espesor_pos', 'le_radius', 'te_gap'])
    return {col: float(y[0, i]) for i, col in enumerate(cols)}, meta


def verificar_surrogate(geom, v0x, alpha, Re):
    modelo, scaler_X, scaler_y, meta = cargar_modelo('modelo_surrogate')
    if modelo is None:
        return None
    try:
        x = np.array([[
            geom['camber_max'], geom['camber_pos'],
            geom['espesor_max'], geom['espesor_pos'],
            geom['le_radius'], geom['te_gap'],
            v0x, alpha, Re
        ]])
        x_s = scaler_X.transform(x)
        y_s = modelo.predict(x_s)
        y = scaler_y.inverse_transform(y_s.reshape(1, -1))
        cl, cd = float(y[0, 0]), float(y[0, 1])
        ld = cl / cd if abs(cd) > 1e-6 else 0.0
        return {'cl': cl, 'cd': cd, 'ld': ld}
    except Exception:
        return None


def mostrar_info():
    print("\n MODELOS DISPONIBLES")
    print("=" * 50)
    for nombre in ('modelo_surrogate', 'modelo_inverso'):
        _, _, _, meta = cargar_modelo(nombre)
        path = os.path.join('data', f'{nombre}.pkl')
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"\n {nombre}.pkl  ({size_kb:.1f} KB)")
            if meta:
                print(f"   Entrenado: {meta.get('fecha', '?')}")
                print(f"   Train:     {meta.get('n_train', '?')} muestras")
                print(f"   Test:      {meta.get('n_test', '?')} muestras")
                print(f"   X cols:    {meta.get('x_cols', '?')}")
                print(f"   Y cols:    {meta.get('y_cols', '?')}")
        else:
            print(f"\n {nombre}.pkl  [NO ENCONTRADO]")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Predice perfil aerodinámico óptimo dado condiciones de vuelo y coeficientes objetivo'
    )
    parser.add_argument('--v', type=float, help='Velocidad de vuelo (m/s)')
    parser.add_argument('--alpha', type=float, help='Ángulo de ataque (grados)')
    parser.add_argument('--Re', type=float, help='Número de Reynolds')
    parser.add_argument('--cl', type=float, help='Coeficiente de sustentación objetivo')
    parser.add_argument('--cd', type=float, help='Coeficiente de arrastre objetivo')
    parser.add_argument('--output', default='perfil_predicho.dat', help='Archivo de salida .dat')
    parser.add_argument('--plot', action='store_true', help='Generar imagen del perfil')
    parser.add_argument('--info', action='store_true', help='Mostrar info de modelos disponibles')
    args = parser.parse_args()

    if args.info:
        mostrar_info()
        return

    # Validar argumentos requeridos
    required = [('--v', args.v), ('--alpha', args.alpha),
                ('--Re', args.Re), ('--cl', args.cl), ('--cd', args.cd)]
    missing = [name for name, val in required if val is None]
    if missing:
        print(f"ERROR: argumentos faltantes: {', '.join(missing)}")
        parser.print_help()
        return

    v0x = args.v
    alpha = args.alpha
    Re = args.Re
    cl_target = args.cl
    cd_target = args.cd

    print("\n" + "=" * 55)
    print(" PREDICTOR DE PERFILES AERODINÁMICOS")
    print("=" * 55)
    print(f" Velocidad:  {v0x} m/s")
    print(f" Alpha:      {alpha}°")
    print(f" Reynolds:   {Re:.0f}")
    print(f" Cl target:  {cl_target}")
    print(f" Cd target:  {cd_target}")
    if abs(cd_target) > 1e-9:
        print(f" L/D target: {cl_target/cd_target:.2f}")
    print("=" * 55)

    # 1. Predecir parámetros geométricos
    print("\n Prediciendo parámetros geométricos...")
    try:
        geom, meta_inv = predecir_geom(v0x, alpha, Re, cl_target, cd_target)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return

    print("\n Parámetros predichos:")
    print(f"   camber_max:  {geom['camber_max']:.5f}  (pos: {geom['camber_pos']:.3f})")
    print(f"   espesor_max: {geom['espesor_max']:.5f}  (pos: {geom['espesor_pos']:.3f})")
    print(f"   le_radius:   {geom['le_radius']:.5f}")
    print(f"   te_gap:      {geom['te_gap']:.5f}")

    # Sanity-clamp de parámetros para evitar perfiles degenerados
    geom_clamped = {
        'camber_max': float(np.clip(geom['camber_max'], -0.15, 0.15)),
        'camber_pos': float(np.clip(geom['camber_pos'], 0.10, 0.90)),
        'espesor_max': float(np.clip(geom['espesor_max'], 0.03, 0.30)),
        'espesor_pos': float(np.clip(geom['espesor_pos'], 0.10, 0.70)),
        'le_radius': float(np.clip(geom['le_radius'], 5e-4, 0.05)),
        'te_gap': float(np.clip(geom['te_gap'], 1e-4, 0.02)),
    }
    if any(abs(geom_clamped[k] - geom[k]) > 1e-8 for k in geom_clamped):
        print("\n [!] Algunos parámetros fueron recortados a rango físico válido.")

    # 2. Reconstruir geometría
    from reconstruir_perfil import reconstruir_perfil, guardar_selig

    print("\n Reconstruyendo geometría...")
    puntos = reconstruir_perfil(**geom_clamped)
    guardar_selig(puntos, args.output, nombre=f'IA_v{v0x}_a{alpha}_Cl{cl_target}_Cd{cd_target}')
    print(f" Perfil guardado: {args.output}  ({len(puntos)} puntos)")

    # 3. Verificar con surrogate forward
    print("\n Verificando con surrogate forward...")
    verif = verificar_surrogate(geom_clamped, v0x, alpha, Re)
    if verif:
        print(f"   Cl estimado: {verif['cl']:.4f}  (target: {cl_target})")
        print(f"   Cd estimado: {verif['cd']:.5f}  (target: {cd_target})")
        print(f"   L/D estimado: {verif['ld']:.3f}")
        err_cl = abs(verif['cl'] - cl_target) / max(abs(cl_target), 1e-6) * 100
        err_cd = abs(verif['cd'] - cd_target) / max(abs(cd_target), 1e-6) * 100
        print(f"   Error Cl: {err_cl:.1f}%  |  Error Cd: {err_cd:.1f}%")
        if err_cl > 30 or err_cd > 50:
            print("\n [!] Error elevado: dataset insuficiente para esta región.")
            print("     Ejecuta script_barrido_conditions.py para ampliar datos.")
    else:
        print("   [surrogate no disponible — entrena primero con train_ia_perfiles.py]")

    # 4. Plot opcional
    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(puntos[:, 0], puntos[:, 1], 'b-', linewidth=2)
            ax.fill(puntos[:, 0], puntos[:, 1], alpha=0.1, color='steelblue')
            titulo = (f"v={v0x} m/s  α={alpha}°  Re={Re:.0f}  "
                      f"Cl_target={cl_target}  Cd_target={cd_target}")
            if verif:
                titulo += f"\nCl_pred={verif['cl']:.4f}  Cd_pred={verif['cd']:.5f}  L/D={verif['ld']:.2f}"
            ax.set_title(titulo, fontsize=9)
            ax.set_aspect('equal')
            ax.set_xlabel('x/c')
            ax.set_ylabel('y/c')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            png = args.output.replace('.dat', '.png')
            plt.savefig(png, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\n Plot: {png}")
        except Exception as e:
            print(f" [!] Error generando plot: {e}")

    print("\n Listo.")


if __name__ == '__main__':
    main()

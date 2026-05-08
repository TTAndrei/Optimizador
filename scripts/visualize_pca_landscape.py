"""
Visualización PCA + Landscape de fitness (Cl/Cd).
- Usa Cupy para calcular PCA (SVD en GPU).
- Carga archivos `GA_history_gen{gen}.npz` generados por `RunGA.py`.
- Genera scatter en el espacio PCA y un contour/heatmap por interpolación.

Instrucciones (ejemplo):
    python visualize_pca_landscape.py --data_dir . --out landscape.png

Salida: gráfico estático con scatter + contornos de fitness.

Explicación en español.
"""

import os
import glob
import argparse
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def load_history(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "GA_history_gen*.npz")))
    X_list = []  # vectores de genes (aplanados)
    y_list = []  # fitness
    for f in files:
        try:
            d = np.load(f)
            genes = d["genes"]  # shape: (pop, n_points, 2)
            fitness = d["fitness"]
            pop, n_points, _ = genes.shape
            X_list.append(genes.reshape(pop, n_points * 2))
            y_list.append(fitness)
        except Exception as e:
            print(f"No se pudo leer {f}: {e}")
    if not X_list:
        raise RuntimeError("No se encontraron archivos GA_history_gen*.npz en el directorio.")
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y


def pca_cupy(X_cpu, n_components=2):
    X_cp = cp.asarray(X_cpu)
    mean = cp.mean(X_cp, axis=0)
    Xc = X_cp - mean
    # SVD en GPU
    U, S, Vt = cp.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]
    scores = Xc.dot(components.T)
    return scores.get(), components.get(), mean.get()


def plot_landscape(scores, fitness, out_path=None, grid_res=200):
    x = scores[:, 0]
    y = scores[:, 1]
    z = fitness

    # Scatter en el espacio PCA
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 2, 1)
    sc = ax1.scatter(x, y, c=z, cmap='viridis', s=20, edgecolor='k')
    ax1.set_title('Población (PCA) coloreada por fitness (Cl/Cd)')
    ax1.set_xlabel('PCA 1')
    ax1.set_ylabel('PCA 2')
    plt.colorbar(sc, ax=ax1, label='Fitness (Cl/Cd)')

    # Interpolación para landscape (CPU)
    ax2 = plt.subplot(1, 2, 2)
    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    XI, YI = np.meshgrid(xi, yi)
    try:
        ZI = griddata((x, y), z, (XI, YI), method='cubic')
    except Exception:
        ZI = griddata((x, y), z, (XI, YI), method='linear')

    cf = ax2.contourf(XI, YI, ZI, levels=30, cmap='viridis')
    ax2.scatter(x, y, c='k', s=4, alpha=0.4)
    ax2.set_title('Landscape de fitness (interpolado)')
    ax2.set_xlabel('PCA 1')
    ax2.set_ylabel('PCA 2')
    plt.colorbar(cf, ax=ax2, label='Fitness (Cl/Cd)')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200)
        print(f"Guardado: {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualización PCA + landscape de fitness')
    parser.add_argument('--data_dir', default='.', help='Directorio con GA_history_gen*.npz')
    parser.add_argument('--out', default=None, help='Ruta de salida para la imagen')
    args = parser.parse_args()

    X, y = load_history(args.data_dir)
    print(f"Cargados {X.shape[0]} individuos, dimensión original {X.shape[1]}")

    scores, comps, mean = pca_cupy(X, n_components=2)
    plot_landscape(scores, y, out_path=args.out)


if __name__ == '__main__':
    main()

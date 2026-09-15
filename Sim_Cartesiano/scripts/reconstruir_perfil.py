"""
reconstruir_perfil.py
======================
Genera un perfil aerodinámico en formato Selig a partir de parámetros de diseño compactos.

Parámetros de entrada:
    camber_max  — máximo del camber (curvatura media)  [fracción de cuerda]
    camber_pos  — posición x del camber máximo         [0..1]
    espesor_max — máximo espesor (extrados - intrados) [fracción de cuerda]
    espesor_pos — posición x del espesor máximo        [0..1]
    le_radius   — radio del leading edge               [fracción de cuerda]
    te_gap      — abertura del trailing edge           [fracción de cuerda]

Método:
    1. Construye camber(x) como parábola de dos tramos (antes/después de camber_pos)
    2. Construye espesor(x) como NACA-style modificado controlado por espesor_max/pos
    3. Impone radio LE en la zona de nariz mediante blending con forma circular
    4. Impone te_gap en el trailing edge
    5. Exporta formato Selig (TE superior → LE → TE inferior)

Uso como módulo:
    from scripts.reconstruir_perfil import reconstruir_perfil
    puntos = reconstruir_perfil(camber_max=0.04, camber_pos=0.4, ...)

Uso como CLI:
    python scripts/reconstruir_perfil.py --camber-max 0.04 --camber-pos 0.4 \
        --espesor-max 0.12 --espesor-pos 0.3 --le-radius 0.016 --te-gap 0.002
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_PUNTOS = 80   # puntos por superficie (total ~160 en formato Selig)


def _camber_parabola(x, camber_max, camber_pos):
    """Camber NACA 4-digit style — dos parábolas unidas en camber_pos."""
    c = np.zeros_like(x)
    p = float(np.clip(camber_pos, 0.01, 0.99))
    m = float(camber_max)

    mask1 = x <= p
    mask2 = ~mask1

    if p > 1e-6:
        c[mask1] = (m / p**2) * (2 * p * x[mask1] - x[mask1]**2)
    if (1 - p) > 1e-6:
        c[mask2] = (m / (1 - p)**2) * ((1 - 2*p) + 2*p*x[mask2] - x[mask2]**2)

    return c


def _espesor_naca(x, espesor_max, espesor_pos):
    """
    Distribución de espesor tipo NACA-4 escalada para alcanzar espesor_max en espesor_pos.
    Se normaliza para que el pico ocurra en espesor_pos.
    """
    # NACA standard thickness distribution (5 coefs)
    # t(x) = 5*t_max * (a0*sqrt(x) - a1*x - a2*x^2 + a3*x^3 - a4*x^4)
    a0, a1, a2, a3, a4 = 0.2969, 0.1260, 0.3516, 0.2843, 0.1015

    t_raw = 5.0 * (a0*np.sqrt(x) - a1*x - a2*x**2 + a3*x**3 - a4*x**4)
    t_raw = np.maximum(t_raw, 0.0)

    # Pico de t_raw en x ≈ 0.30 (NACA estándar). Escalar para que pico = espesor_max.
    x_peak = np.linspace(0, 1, 2000)
    t_peak_raw = 5.0 * (a0*np.sqrt(x_peak) - a1*x_peak - a2*x_peak**2
                         + a3*x_peak**3 - a4*x_peak**4)
    t_max_raw = float(np.max(t_peak_raw))

    if t_max_raw > 1e-9:
        t_raw *= espesor_max / t_max_raw

    # Si espesor_pos ≠ 0.30, aplicar compresión/expansión del eje x para mover el pico.
    p_std = 0.30
    p_target = float(np.clip(espesor_pos, 0.05, 0.75))
    if abs(p_target - p_std) > 0.01:
        x_warp = np.where(
            x <= p_target,
            x * (p_std / max(p_target, 1e-9)),
            p_std + (x - p_target) * ((1 - p_std) / max(1 - p_target, 1e-9))
        )
        x_warp = np.clip(x_warp, 0, 1)
        t_raw = np.interp(x_warp, x_peak, t_peak_raw) * (espesor_max / max(t_max_raw, 1e-9))

    return np.maximum(t_raw, 0.0)


def reconstruir_perfil(camber_max, camber_pos, espesor_max, espesor_pos,
                        le_radius, te_gap, n_puntos=N_PUNTOS, chord=1.0):
    """
    Genera array (N, 2) en formato Selig (TE superior → LE → TE inferior).

    Args:
        camber_max:  máximo camber [fracción de cuerda]
        camber_pos:  posición x del camber máximo [0..1]
        espesor_max: espesor máximo [fracción de cuerda]
        espesor_pos: posición x del espesor máximo [0..1]
        le_radius:   radio del leading edge [fracción de cuerda]
        te_gap:      abertura del trailing edge [fracción de cuerda]
        n_puntos:    puntos por superficie
        chord:       escala (1.0 = normalizado)

    Returns:
        np.array (2*n_puntos - 1, 2)  en formato Selig
    """
    # Distribución coseno para mayor densidad en LE y TE
    beta = np.linspace(0, np.pi, n_puntos)
    x = 0.5 * (1 - np.cos(beta))   # [0 .. 1]

    camber = _camber_parabola(x, camber_max, camber_pos)
    espesor = _espesor_naca(x, espesor_max, espesor_pos)

    # Imponer te_gap en el TE (x=1)
    espesor[-1] = float(te_gap)

    # Rampa suave de espesor hacia cero en LE (x=0)
    espesor[0] = 0.0

    # Blending con forma circular en zona de nariz para respetar le_radius.
    # Circunferencia de radio r_le tangente al perfil en x=0:
    # y_circ(x) = sqrt(max(0, r_le^2 - (x - r_le)^2))
    r = float(le_radius)
    if r > 1e-6:
        x_blend = min(4 * r, 0.15)   # zona de influencia
        mask_le = x <= x_blend
        if np.any(mask_le):
            x_le = x[mask_le]
            arg = np.maximum(0.0, r**2 - (x_le - r)**2)
            y_circ = 0.5 * np.sqrt(arg)   # mitad superior
            w = 1.0 - (x_le / x_blend)**2  # peso decreciente desde LE
            espesor[mask_le] = (1 - w) * espesor[mask_le] + w * 2 * y_circ

    espesor = np.maximum(espesor, 0.0)
    espesor[0] = 0.0

    y_upper = camber + 0.5 * espesor
    y_lower = camber - 0.5 * espesor

    # Construir array formato Selig: TE sup (x=1) → LE (x=0) → TE inf (x=1)
    x_upper = x[::-1]          # 1 → 0
    y_upper_rev = y_upper[::-1]
    x_lower = x[1:]             # 0 → 1 (omite LE duplicado)
    y_lower_fwd = y_lower[1:]

    xs = np.concatenate([x_upper, x_lower]) * chord
    ys = np.concatenate([y_upper_rev, y_lower_fwd]) * chord

    return np.column_stack([xs, ys])


def guardar_selig(puntos, filepath, nombre='IA_PREDICHO'):
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"{nombre}\n")
        for p in puntos:
            f.write(f"  {p[0]:.6f}  {p[1]:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description='Reconstruye perfil Selig desde parámetros de diseño')
    parser.add_argument('--camber-max',  type=float, required=True)
    parser.add_argument('--camber-pos',  type=float, required=True)
    parser.add_argument('--espesor-max', type=float, required=True)
    parser.add_argument('--espesor-pos', type=float, required=True)
    parser.add_argument('--le-radius',   type=float, required=True)
    parser.add_argument('--te-gap',      type=float, required=True)
    parser.add_argument('--output', default='perfil_reconstruido.dat')
    parser.add_argument('--n-puntos', type=int, default=N_PUNTOS)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    puntos = reconstruir_perfil(
        camber_max=args.camber_max,
        camber_pos=args.camber_pos,
        espesor_max=args.espesor_max,
        espesor_pos=args.espesor_pos,
        le_radius=args.le_radius,
        te_gap=args.te_gap,
        n_puntos=args.n_puntos,
    )

    guardar_selig(puntos, args.output)
    print(f" Perfil guardado: {args.output}  ({len(puntos)} puntos)")
    print(f"   camber_max={args.camber_max}  camber_pos={args.camber_pos}")
    print(f"   espesor_max={args.espesor_max}  espesor_pos={args.espesor_pos}")
    print(f"   le_radius={args.le_radius}  te_gap={args.te_gap}")

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(puntos[:, 0], puntos[:, 1], 'b-', linewidth=1.5)
        ax.fill(puntos[:, 0], puntos[:, 1], alpha=0.1, color='steelblue')
        ax.set_aspect('equal')
        ax.set_title(f'Perfil reconstruido  |  camber={args.camber_max:.4f}  espesor={args.espesor_max:.4f}')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        png = args.output.replace('.dat', '.png')
        plt.savefig(png, dpi=150)
        plt.close()
        print(f" Plot: {png}")


if __name__ == '__main__':
    main()

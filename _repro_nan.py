"""
Reproduccion directa del NaN: llama a main() con los mismos parametros
que study_simulador.py para identificar si el NaN persiste tras el fix.
"""
import sys, traceback
import Simulador2D

# Parametros identicos a study_simulador.py
alphas_test = [0, 2, 5, 8, 10, 12]
dx_test = 0.004  # El mas grueso (peor caso para TE)

for alpha in alphas_test:
    print(f"\n{'='*70}")
    print(f"  ALPHA = {alpha}°   dx_grueso = {dx_test}")
    print(f"{'='*70}")
    try:
        mesh_fina, mesh_gruesa, geometria = Simulador2D.main(
            filepath='AG24',
            iteraciones=100,    # Suficiente para detectar NaN (ocurria a iter 10)
            guardado=50,
            v0x=5,
            CFL=0.5,
            alpha_deg=alpha,
            chord=1.0,
            dx_fino=0.0015,
            dx_grueso=dx_test,
            Lx=7,
            Ly=6,
            cx=1,
            divergencia=1e-1,
            graficos=False,
            usar_wale=False,
            usar_viscosidad_estela=False,
            stop_on_convergence=False,
        )
        print(f"  OK alpha={alpha} deg: 100 iteraciones OK")
    except RuntimeError as e:
        print(f"  FAIL alpha={alpha} deg: {e}")
    except Exception as e:
        print(f"  FAIL alpha={alpha} deg ERROR INESPERADO: {e}")
        traceback.print_exc()

print("\n" + "="*70)
print("REPRODUCCION COMPLETADA")
print("="*70)

"""
Viewer en tiempo real para Simulador2D.
Se ejecuta como proceso INDEPENDIENTE mientras la simulación corre.

Uso:
    python viewer_live.py

Lee datos de memoria compartida publicados por el simulador.
No afecta el rendimiento de la simulación.
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from multiprocessing import shared_memory
import struct
import time
import sys
import os

# Nombre del bloque de memoria compartida (debe coincidir con el simulador)
SHM_NAME = "sim2d_live"
META_NAME = "sim2d_meta"

def conectar_shm(timeout=60):
    """Intenta conectar a la memoria compartida del simulador."""
    print("Esperando conexión con el simulador...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            shm_meta = shared_memory.SharedMemory(name=META_NAME, create=False)
            # Leer metadata: ny(4)+nx(4)+iter(4)+cd(4)+cl(4)+alpha(4)+timestamp(8)+v0x(4)+v0y(4) = 40 bytes
            meta_buf = shm_meta.buf
            ny = struct.unpack('i', meta_buf[0:4])[0]
            nx = struct.unpack('i', meta_buf[4:8])[0]
            if ny > 0 and nx > 0:
                shm_data = shared_memory.SharedMemory(name=SHM_NAME, create=False)
                print(f"[OK] Conectado. Malla: {ny}x{nx}")
                return shm_meta, shm_data, ny, nx
            shm_meta.close()
        except FileNotFoundError:
            pass
        time.sleep(0.5)
    print("✗ Timeout: no se encontró simulación activa.")
    sys.exit(1)

def main():
    shm_meta, shm_data, ny, nx = conectar_shm()
    
    # Preparar arrays numpy que apuntan a la memoria compartida
    # Layout de datos: speed(ny*nx*4) + solid(ny*nx*1) + vorticity(ny*nx*4)
    n_cells = ny * nx
    offset_speed = 0
    offset_solid = n_cells * 4
    offset_vort = offset_solid + n_cells
    
    # Configurar figura
    plt.ion()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('Simulador 2D — Vista en Tiempo Real', fontsize=14, fontweight='bold')
    
    ax_field = axes[0]
    ax_coefs = axes[1]
    
    # Campo de velocidad
    dummy = np.zeros((ny, nx), dtype=np.float32)
    im = ax_field.imshow(dummy, origin='lower', cmap='rainbow', aspect='auto', interpolation='bilinear')
    cbar = fig.colorbar(im, ax=ax_field, label='|V| [m/s]', shrink=0.8)
    ax_field.set_xlabel('x')
    ax_field.set_ylabel('y')
    titulo_campo = ax_field.set_title('Velocidad — iter 0')
    
    # Flecha de dirección del flujo (se actualiza con v0x, v0y)
    arrow_ref = [None]  # contenedor mutable para actualizar
    
    # Historial de Cd/Cl
    hist_iter = []
    hist_cd = []
    hist_cl = []
    hist_eff = []
    line_cd, = ax_coefs.plot([], [], 'r-', linewidth=1, label='Cd')
    line_cl, = ax_coefs.plot([], [], 'b-', linewidth=1, label='Cl')
    ax_coefs.set_xlabel('Iteración')
    ax_coefs.set_ylabel('Coeficiente')
    ax_coefs.grid(True, alpha=0.3)
    
    # Eje Y secundario para eficiencia (Cl/Cd)
    ax_eff = ax_coefs.twinx()
    line_eff, = ax_eff.plot([], [], 'g--', linewidth=1, alpha=0.7, label='Cl/Cd')
    ax_eff.set_ylabel('Cl/Cd', color='green')
    ax_eff.tick_params(axis='y', labelcolor='green')
    
    # Leyenda combinada
    lines_all = [line_cd, line_cl, line_eff]
    labels_all = [l.get_label() for l in lines_all]
    ax_coefs.legend(lines_all, labels_all, loc='upper right', fontsize=9)
    titulo_coefs = ax_coefs.set_title('Cd / Cl')
    
    fig.tight_layout()
    
    last_timestamp = 0.0
    fps_counter = 0
    fps_time = time.time()
    fps_display = 0.0
    
    print("Visualización activa. Cierra la ventana para salir.")
    
    try:
        while plt.fignum_exists(fig.number):
            # Leer metadata
            try:
                meta_buf = shm_meta.buf
                ny_now = struct.unpack('i', meta_buf[0:4])[0]
                nx_now = struct.unpack('i', meta_buf[4:8])[0]
                iteracion = struct.unpack('i', meta_buf[8:12])[0]
                cd_val = struct.unpack('f', meta_buf[12:16])[0]
                cl_val = struct.unpack('f', meta_buf[16:20])[0]
                alpha_val = struct.unpack('f', meta_buf[20:24])[0]
                timestamp = struct.unpack('d', meta_buf[24:32])[0]
                v0x_val = struct.unpack('f', meta_buf[32:36])[0]
                v0y_val = struct.unpack('f', meta_buf[36:40])[0]
            except Exception:
                time.sleep(0.05)
                continue
            
            # Solo actualizar si hay datos nuevos
            if timestamp <= last_timestamp:
                plt.pause(0.03)
                continue
            last_timestamp = timestamp
            
            # Leer campo de velocidad desde memoria compartida
            try:
                speed = np.ndarray((ny, nx), dtype=np.float32,
                                   buffer=shm_data.buf[offset_speed:offset_speed + n_cells * 4])
                solid = np.ndarray((ny, nx), dtype=np.uint8,
                                   buffer=shm_data.buf[offset_solid:offset_solid + n_cells])
                vort = np.ndarray((ny, nx), dtype=np.float32,
                                  buffer=shm_data.buf[offset_vort:offset_vort + n_cells * 4])
            except Exception:
                time.sleep(0.05)
                continue
            
            # Copiar para evitar lecturas parciales
            speed_copy = speed.copy()
            solid_copy = solid.copy().astype(bool)
            
            # Enmascarar sólido (gris oscuro)
            vmax = float(np.percentile(speed_copy[~solid_copy], 99)) if np.any(~solid_copy) else 1.0
            speed_display = speed_copy.copy()
            speed_display[solid_copy] = np.nan
            
            # Actualizar imagen
            im.set_data(speed_display)
            im.set_clim(0, max(vmax, 0.01))
            titulo_campo.set_text(f'|V| — iter {iteracion}  |  α={alpha_val:.1f}°  |  {fps_display:.0f} FPS')
            
            # Actualizar flecha de dirección del flujo
            U_mag = max(np.sqrt(v0x_val**2 + v0y_val**2), 1e-6)
            dx_arr = v0x_val / U_mag  # dirección normalizada
            dy_arr = v0y_val / U_mag
            arrow_len = nx * 0.08  # longitud visual
            x0_arr = nx * 0.05
            y0_arr = ny * 0.90
            if arrow_ref[0] is not None:
                arrow_ref[0].remove()
            arrow_ref[0] = ax_field.annotate(
                '', xy=(x0_arr + dx_arr * arrow_len, y0_arr + dy_arr * arrow_len),
                xytext=(x0_arr, y0_arr),
                arrowprops=dict(arrowstyle='->', color='white', lw=2.5),
            )
            
            # Actualizar historial Cd/Cl
            if cd_val != 0.0 or cl_val != 0.0:
                hist_iter.append(iteracion)
                hist_cd.append(cd_val)
                hist_cl.append(cl_val)
                eff_val = cl_val / cd_val if abs(cd_val) > 1e-12 else 0.0
                hist_eff.append(eff_val)
                
                # Limitar historial visible (últimos 2000 puntos)
                max_hist = 2000
                if len(hist_iter) > max_hist:
                    hist_iter = hist_iter[-max_hist:]
                    hist_cd = hist_cd[-max_hist:]
                    hist_cl = hist_cl[-max_hist:]
                    hist_eff = hist_eff[-max_hist:]
                
                line_cd.set_data(hist_iter, hist_cd)
                line_cl.set_data(hist_iter, hist_cl)
                line_eff.set_data(hist_iter, hist_eff)
                ax_coefs.relim()
                ax_coefs.autoscale_view()
                ax_eff.relim()
                ax_eff.autoscale_view()
                titulo_coefs.set_text(f'Cd={cd_val:.5f}  Cl={cl_val:.5f}  Cl/Cd={eff_val:.2f}')
            
            # FPS counter
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps_display = fps_counter / (time.time() - fps_time)
                fps_counter = 0
                fps_time = time.time()
            
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.01)
            
    except KeyboardInterrupt:
        print("\nViewer cerrado por usuario.")
    finally:
        shm_meta.close()
        shm_data.close()
        plt.close('all')
        print("Recursos liberados.")


if __name__ == "__main__":
    main()

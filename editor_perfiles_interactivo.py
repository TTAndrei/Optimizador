"""
Editor Interactivo de Perfiles Aerodinámicos
Permite cargar, editar y guardar perfiles personalizados
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backend_bases import MouseButton
import os

class EditorPerfiles:
    def __init__(self, filepath=None):
        """
        Editor interactivo de perfiles aerodinámicos.
        
        Controles:
        - Click izquierdo + arrastrar: Mover punto
        - Click derecho: Agregar punto nuevo
        - 'Delete' o 'Supr': Eliminar punto seleccionado
        - 's': Guardar perfil
        - 'r': Resetear a original
        - 'g': Toggle cuadrícula fina
        - 'h': Mostrar ayuda
        """
        self.filepath = filepath
        self.points = None
        self.original_points = None
        self.title = "Nuevo Perfil"
        
        # Estado de edición
        self.selected_point = None
        self.dragging = False
        self.snap_to_grid = True
        self.grid_spacing = 0.001
        self.fine_grid = False
        
        # Configuración visual
        self.point_size = 8
        self.selected_size = 12
        
        # Cargar archivo si se proporciona
        if filepath and os.path.exists(filepath):
            self.load_profile(filepath)
        else:
            # Crear perfil por defecto (línea recta)
            self.points = np.array([
                [0.0, 0.0],
                [0.25, 0.05],
                [0.5, 0.06],
                [0.75, 0.05],
                [1.0, 0.0],
                [0.75, -0.05],
                [0.5, -0.06],
                [0.25, -0.05]
            ])
            self.original_points = self.points.copy()
            self.title = "Perfil Personalizado"
        
        self.setup_plot()
        
    def load_profile(self, filepath):
        """Cargar perfil desde archivo - CARGA TODOS LOS PUNTOS"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Primera línea es el título
            if lines:
                self.title = lines[0].strip()
            
            # Parsear coordenadas - SIN LÍMITES, todos los puntos
            coords = []
            skipped = 0
            for i, line in enumerate(lines[1:], start=2):
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            x, y = float(parts[0]), float(parts[1])
                            coords.append([x, y])
                        except ValueError:
                            skipped += 1
                            print(f"  ⚠ Línea {i} omitida (formato inválido): {line[:30]}")
                            continue
            
            if coords:
                self.points = np.array(coords, dtype=float)
                self.original_points = self.points.copy()
                print(f"\n✓ Perfil cargado exitosamente: {filepath}")
                print(f"  📊 Total de puntos cargados: {len(self.points)}")
                print(f"  📌 Rango X: [{self.points[:, 0].min():.4f}, {self.points[:, 0].max():.4f}]")
                print(f"  📌 Rango Y: [{self.points[:, 1].min():.4f}, {self.points[:, 1].max():.4f}]")
                if skipped > 0:
                    print(f"  ⚠ Líneas omitidas: {skipped}")
                print(f"  ✅ Todos los puntos son editables\n")
            else:
                print(f"⚠ No se encontraron coordenadas válidas en {filepath}")
                
        except Exception as e:
            print(f"✗ Error al cargar archivo: {e}")
    
    def setup_plot(self):
        """Configurar la visualización interactiva con todos los puntos editables"""
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.fig.canvas.manager.set_window_title(f"Editor de Perfiles - {len(self.points)} puntos cargados")
        
        # Configurar ejes
        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(-0.15, 0.15)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True, alpha=0.3, linewidth=0.5)
        self.ax.set_xlabel('x/c', fontsize=11)
        self.ax.set_ylabel('y/c', fontsize=11)
        
        # Líneas de referencia
        self.ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        self.ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        self.ax.axvline(x=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        self.update_title()
        
        # Plot inicial
        # Mostrar TODOS los puntos
        self.line, = self.ax.plot(self.points[:, 0], self.points[:, 1], 
                                  'b-', linewidth=1.5, alpha=0.7, label=f'Perfil ({len(self.points)} pts)')
        self.scatter = self.ax.scatter(self.points[:, 0], self.points[:, 1], 
                                       c='blue', s=self.point_size**2, 
                                       picker=True, zorder=5, alpha=0.8,
                                       label=f'TODOS editables (click+arrastrar)')
        
        # Numerar algunos puntos para referencia
        for i in range(0, len(self.points), max(1, len(self.points)//20)):
            self.ax.text(self.points[i, 0], self.points[i, 1], f' {i}', 
                        fontsize=7, alpha=0.6, color='darkblue')
        
        # Punto seleccionado (invisible inicialmente)
        self.selected_scatter = self.ax.scatter([], [], c='red', s=self.selected_size**2, 
                                               zorder=10, marker='o', edgecolors='darkred', 
                                               linewidths=2)
        
        self.ax.legend(loc='upper right', fontsize=9)
        
        # Conectar eventos
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        
        # Mostrar ayuda inicial
        self.show_help()
        
    def update_title(self):
        """Actualizar título con información completa"""
        n_points = len(self.points)
        title = f"{self.title} | {n_points} puntos EDITABLES"
        if self.selected_point is not None:
            x, y = self.points[self.selected_point]
            title += f" | Seleccionado: #{self.selected_point} ({x:.4f}, {y:.4f})"
        if self.snap_to_grid:
            title += f" | Grid: {self.grid_spacing:.5f}"
        else:
            title += " | Libre"
        self.ax.set_title(title, fontsize=11, fontweight='bold')
    
    def on_pick(self, event):
        """Evento cuando se selecciona un punto"""
        if event.mouseevent.button == MouseButton.LEFT:
            ind = event.ind[0]
            self.selected_point = ind
            self.dragging = True
            
            # Resaltar punto seleccionado
            self.selected_scatter.set_offsets([self.points[ind]])
            self.fig.canvas.draw_idle()
    
    def on_click(self, event):
        """Evento de click del ratón"""
        if event.inaxes != self.ax:
            return
        
        # Click derecho: agregar punto en la posición correcta del perfil
        if event.button == MouseButton.RIGHT:
            x_click, y_click = event.xdata, event.ydata
            
            if len(self.points) < 2:
                # Si hay menos de 2 puntos, simplemente agregar al final
                x, y = x_click, y_click
                if self.snap_to_grid:
                    x = round(x / self.grid_spacing) * self.grid_spacing
                    y = round(y / self.grid_spacing) * self.grid_spacing
                x = np.clip(x, -0.05, 1.05)
                y = np.clip(y, -0.15, 0.15)
                new_point = np.array([x, y])
                self.points = np.vstack([self.points, new_point])
                insert_pos = len(self.points) - 1
            else:
                # Encontrar el segmento más cercano y proyectar el punto sobre él
                click_point = np.array([x_click, y_click])
                min_dist = float('inf')
                insert_index = len(self.points)
                projected_point = click_point
                
                for i in range(len(self.points)):
                    j = (i + 1) % len(self.points)  # Siguiente punto (circular)
                    
                    # Calcular distancia del punto al segmento i-j
                    p1 = self.points[i]
                    p2 = self.points[j]
                    
                    # Calcular proyección del click sobre el segmento
                    projection, dist = self._project_point_to_segment(click_point, p1, p2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        insert_index = j  # Insertar después del punto i
                        projected_point = projection
                
                # Usar el punto proyectado sobre la línea, no el click directo
                x, y = projected_point[0], projected_point[1]
                
                # Aplicar snap to grid si está activado
                if self.snap_to_grid:
                    x = round(x / self.grid_spacing) * self.grid_spacing
                    y = round(y / self.grid_spacing) * self.grid_spacing
                
                # Limitar a rango válido
                x = np.clip(x, -0.05, 1.05)
                y = np.clip(y, -0.15, 0.15)
                
                new_point = np.array([x, y])
                
                # Insertar punto en la posición óptima
                self.points = np.insert(self.points, insert_index, new_point, axis=0)
                insert_pos = insert_index
            
            print(f"✓ Punto añadido en posición {insert_pos}: ({x:.4f}, {y:.4f}) - Proyectado sobre línea")
            self.update_plot()
    
    def _project_point_to_segment(self, point, seg_start, seg_end):
        """
        Proyectar un punto sobre un segmento de línea.
        Retorna: (punto_proyectado, distancia)
        """
        # Vector del segmento
        segment = seg_end - seg_start
        segment_len_sq = np.dot(segment, segment)
        
        if segment_len_sq == 0:
            # El segmento es un punto
            return seg_start, np.linalg.norm(point - seg_start)
        
        # Proyección del punto sobre el segmento (parámetro t)
        t = np.clip(np.dot(point - seg_start, segment) / segment_len_sq, 0, 1)
        
        # Punto más cercano en el segmento
        projection = seg_start + t * segment
        
        # Distancia del punto a su proyección
        distance = np.linalg.norm(point - projection)
        
        return projection, distance
    
    def _point_to_segment_distance(self, point, seg_start, seg_end):
        """Calcular distancia de un punto a un segmento de línea"""
        _, distance = self._project_point_to_segment(point, seg_start, seg_end)
        return distance
    
    def on_release(self, event):
        """Evento al soltar el botón del ratón"""
        if self.dragging:
            self.dragging = False
            self.selected_point = None
            self.selected_scatter.set_offsets(np.empty((0, 2)))
            self.fig.canvas.draw_idle()
    
    def on_motion(self, event):
        """Evento de movimiento del ratón"""
        if not self.dragging or self.selected_point is None:
            return
        
        if event.inaxes != self.ax:
            return
        
        x, y = event.xdata, event.ydata
        
        # Ajustar a cuadrícula si está activado
        if self.snap_to_grid:
            x = round(x / self.grid_spacing) * self.grid_spacing
            y = round(y / self.grid_spacing) * self.grid_spacing
        
        # Limitar a rango válido
        x = np.clip(x, -0.05, 1.05)
        y = np.clip(y, -0.15, 0.15)
        
        # Actualizar posición del punto
        self.points[self.selected_point] = [x, y]
        
        # Actualizar visualización
        self.update_plot()
        self.selected_scatter.set_offsets([self.points[self.selected_point]])
        self.fig.canvas.draw_idle()
    
    def on_key(self, event):
        """Eventos de teclado"""
        
        # Guardar perfil
        if event.key == 's':
            self.save_profile()
        
        # Resetear a original
        elif event.key == 'r':
            self.points = self.original_points.copy()
            self.selected_point = None
            self.selected_scatter.set_offsets(np.empty((0, 2)))
            print("✓ Perfil reseteado al original")
            self.update_plot()
        
        # Toggle cuadrícula fina
        elif event.key == 'g':
            self.fine_grid = not self.fine_grid
            if self.fine_grid:
                self.grid_spacing = 0.0005
                self.ax.grid(True, which='both', alpha=0.2, linewidth=0.3)
                self.ax.minorticks_on()
                print(f"✓ Cuadrícula ultra-fina activada (spacing: {self.grid_spacing:.5f})")
            else:
                self.grid_spacing = 0.001
                self.ax.grid(True, alpha=0.3, linewidth=0.5)
                self.ax.minorticks_off()
                print(f"✓ Cuadrícula fina (spacing: {self.grid_spacing:.5f})")
            self.update_title()
            self.fig.canvas.draw_idle()
        
        # Toggle snap to grid
        elif event.key == 'n':
            self.snap_to_grid = not self.snap_to_grid
            status = "activado" if self.snap_to_grid else "desactivado"
            print(f"✓ Ajuste a cuadrícula {status}")
            self.update_title()
            self.fig.canvas.draw_idle()
        
        # Eliminar punto seleccionado
        elif event.key in ['delete', 'backspace', 'supr']:
            if self.selected_point is not None and len(self.points) > 3:
                deleted_point = self.points[self.selected_point].copy()
                self.points = np.delete(self.points, self.selected_point, axis=0)
                self.selected_point = None
                self.selected_scatter.set_offsets(np.empty((0, 2)))
                print(f"✓ Punto eliminado: ({deleted_point[0]:.4f}, {deleted_point[1]:.4f})")
                self.update_plot()
            elif len(self.points) <= 3:
                print("⚠ No se puede eliminar: mínimo 3 puntos requeridos")
        
        # Mostrar ayuda
        elif event.key == 'h':
            self.show_help()
        
        # Exportar coordenadas a consola
        elif event.key == 'e':
            self.export_to_console()
        
        # Ordenar puntos por x
        elif event.key == 'o':
            self.points = self.points[np.argsort(self.points[:, 0])]
            print("✓ Puntos ordenados por coordenada X")
            self.update_plot()
        
        # Cerrar perfil (conectar primer y último punto)
        elif event.key == 'c':
            if not np.allclose(self.points[0], self.points[-1]):
                self.points = np.vstack([self.points, self.points[0]])
                print("✓ Perfil cerrado")
                self.update_plot()
    
    def update_plot(self):
        """Actualizar la visualización completa - TODOS los puntos editables"""
        # Actualizar línea del perfil con TODOS los puntos
        self.line.set_data(self.points[:, 0], self.points[:, 1])
        
        # Actualizar scatter con TODOS los puntos
        self.scatter.set_offsets(self.points)
        self.scatter.set_sizes([self.point_size**2] * len(self.points))
        
        # Limpiar numeración anterior
        for txt in self.ax.texts[:]:
            txt.remove()
        
        # Numerar algunos puntos (cada N puntos para claridad)
        step = max(1, len(self.points) // 20)
        for i in range(0, len(self.points), step):
            self.ax.text(self.points[i, 0], self.points[i, 1], f' {i}', 
                        fontsize=7, alpha=0.6, color='darkblue',
                        verticalalignment='bottom')
        
        # Actualizar etiquetas
        self.line.set_label(f'Perfil completo')
        self.scatter.set_label(f'TODOS editables ({len(self.points)} pts)')
        self.ax.legend(loc='upper right', fontsize=9)
        
        # Actualizar título
        self.update_title()
        self.fig.canvas.draw_idle()
    
    def save_profile(self):
        """Guardar perfil a archivo en formato NACA estándar"""
        # Pedir nombre de archivo
        default_name = "perfil_personalizado.dat"
        filename = input(f"\n💾 Nombre del archivo [{default_name}]: ").strip()
        
        if not filename:
            filename = default_name
        
        # Asegurar extensión
        if not filename.endswith(('.dat', '.txt')):
            filename += '.dat'
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # Título (primera línea)
                f.write(f"{self.title}\n")
                
                # Coordenadas (formato: 2 espacios + x + 2 espacios + y)
                for point in self.points:
                    f.write(f"{point[0]:10.6f}{point[1]:10.6f}\n")
            
            print(f"✓ Perfil guardado en: {filename}")
            print(f"  {len(self.points)} puntos")
            
        except Exception as e:
            print(f"✗ Error al guardar: {e}")
    
    def export_to_console(self):
        """Exportar coordenadas a la consola"""
        print("\n" + "="*60)
        print(f"COORDENADAS DEL PERFIL: {self.title}")
        print("="*60)
        print(f"{'X':>12} {'Y':>12}")
        print("-"*60)
        for i, point in enumerate(self.points):
            print(f"{point[0]:12.6f} {point[1]:12.6f}")
        print("="*60)
        print(f"Total: {len(self.points)} puntos\n")
    
    def show_help(self):
        """Mostrar ayuda en consola"""
        print("\n" + "="*70)
        print(" 🎨 EDITOR DE PERFILES AERODINÁMICOS - AYUDA")
        print("="*70)
        print("\n ✅ CARGA COMPLETA:")
        print(f"  • Se han cargado TODOS los {len(self.points)} puntos del archivo")
        print(f"  • TODOS los puntos son editables mediante click+arrastrar")
        print(f"  • Use zoom/pan (toolbar) para ver detalles")
        print("\n CONTROLES DEL RATÓN:")
        print("  • Click izquierdo + arrastrar : Mover CUALQUIER punto")
        print("  • Click derecho               : Agregar punto nuevo")
        print("  • Rueda del ratón            : Zoom in/out")
        print("\n ATAJOS DE TECLADO:")
        print("  • [S]       : Guardar perfil a archivo")
        print("  • [R]       : Resetear a perfil original")
        print("  • [G]       : Toggle cuadrícula ultra-fina (0.0005) / fina (0.001)")
        print("  • [N]       : Toggle ajuste a cuadrícula")
        print("  • [O]       : Ordenar puntos por coordenada X")
        print("  • [C]       : Cerrar perfil (conectar extremos)")
        print("  • [E]       : Exportar coordenadas a consola")
        print("  • [H]       : Mostrar esta ayuda")
        print("  • [Delete]  : Eliminar punto seleccionado")
        print("\n INFORMACIÓN:")
        print(f"  • Puntos cargados  : {len(self.points)} (TODOS editables)")
        print(f"  • Snap to grid     : {'Sí' if self.snap_to_grid else 'No'}")
        print(f"  • Grid spacing     : {self.grid_spacing:.5f} (Alta precisión)")
        print("="*70 + "\n")
    
    def run(self):
        """Ejecutar el editor"""
        plt.show()


def main():
    """Función principal"""
    print("\n" + "="*70)
    print(" 🎨 EDITOR INTERACTIVO DE PERFILES AERODINÁMICOS")
    print("="*70)
    
    # Obtener directorio del script actual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    naca_file = os.path.join(script_dir, "NACA_0012")
    
    # Carga automática si existe NACA_0012 en la misma carpeta
    if os.path.exists(naca_file):
        print(f"\n ✓ Cargando automáticamente: {naca_file}")
        editor = EditorPerfiles(naca_file)
        print("\n ✓ Editor iniciado. Presiona 'H' en la ventana para ver ayuda.\n")
        editor.run()
        return
    
    # Si no existe NACA_0012, usar modo interactivo
    print("\n ⚠ No se encontró NACA_0012 en el directorio del editor")
    print("\n Opciones:")
    print("  1. Cargar perfil existente")
    print("  2. Crear perfil nuevo desde cero")
    print()
    
    opcion = input("Selecciona opción [1/2]: ").strip()
    
    if opcion == '1':
        print("\n Archivos disponibles en el directorio:")
        files = [f for f in os.listdir(script_dir) if f.endswith(('.dat', '.txt')) or 
                 ('NACA' in f and not f.endswith('.py'))]
        
        if files:
            for i, f in enumerate(files, 1):
                print(f"  {i}. {f}")
            print()
            
            file_choice = input(f"Selecciona archivo [1-{len(files)}] o escribe nombre: ").strip()
            
            try:
                idx = int(file_choice) - 1
                if 0 <= idx < len(files):
                    filepath = os.path.join(script_dir, files[idx])
                else:
                    print("⚠ Índice inválido, usando nombre como ruta")
                    filepath = file_choice if os.path.isabs(file_choice) else os.path.join(script_dir, file_choice)
            except ValueError:
                filepath = file_choice if os.path.isabs(file_choice) else os.path.join(script_dir, file_choice)
        else:
            filepath = input("Nombre del archivo: ").strip()
            if not os.path.isabs(filepath):
                filepath = os.path.join(script_dir, filepath)
        
        if not os.path.exists(filepath):
            print(f"⚠ Archivo no encontrado: {filepath}")
            print("  Creando perfil nuevo...")
            filepath = None
        
        editor = EditorPerfiles(filepath)
    else:
        print("\n Creando perfil nuevo desde cero...")
        editor = EditorPerfiles()
    
    print("\n ✓ Editor iniciado. Presiona 'H' en la ventana para ver ayuda.\n")
    editor.run()


if __name__ == "__main__":
    main()

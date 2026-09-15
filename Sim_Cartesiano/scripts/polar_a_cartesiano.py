import numpy as np

def polar_a_cartesiano(modulo, grados):
    """
    Convierte coordenadas polares a cartesianas.
    
    Parámetros:
        modulo: magnitud del vector (radio)
        grados: ángulo en grados
    
    Retorna:
        vx, vy: componentes cartesianas
    """
    # Convertir grados a radianes
    radianes = np.deg2rad(grados)
    
    # Calcular componentes cartesianas
    vx = modulo * np.cos(radianes)
    vy = modulo * np.sin(radianes)
    
    return vx, vy


if __name__ == "__main__":
    # Ejemplo de uso
    modulo = float(input("Ingrese el módulo (radio): "))
    grados = float(input("Ingrese el ángulo en grados: "))
    
    vx, vy = polar_a_cartesiano(modulo, grados)
    
    print(f"\nResultados:")
    print(f"vx = {vx:.6f}")
    print(f"vy = {vy:.6f}")
    print(f"\nVerificación:")
    print(f"Módulo = √(vx² + vy²) = {np.sqrt(vx**2 + vy**2):.6f}")
    print(f"Ángulo = arctan(vy/vx) = {np.rad2deg(np.arctan2(vy, vx)):.6f}°")

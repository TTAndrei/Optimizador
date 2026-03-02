# Controles Interactivos de Simulación

El simulador ahora incluye tres mecanismos para controlar la ejecución durante la simulación:

## 1. Ctrl+C - Finalizar con outputs completos

**Uso:**
- Presiona `Ctrl+C` en cualquier momento durante la ejecución
- La simulación se detendrá limpiamente
- Se generarán todos los gráficos y reportes con los datos simulados hasta ese momento
- Es equivalente a que la simulación termine normalmente, pero con menos iteraciones

**Qué se genera:**
- Todos los gráficos (velocidad, fuerzas, Cp, etc.)
- Reporte completo de coeficientes aerodinámicos
- Estadísticas de timing
- Datos hasta la iteración actual (Cd, Cl, divergencia, etc.)

---

## 2. Archivo PLOT_NOW.trigger - Generar gráficos sin detener

**Uso:**
```bash
# En PowerShell (mientras la simulación corre):
New-Item -Path "PLOT_NOW.trigger" -ItemType File

# O simplemente crear un archivo vacío llamado PLOT_NOW.trigger
# en la misma carpeta donde se ejecuta el script
```

**Qué sucede:**
- El simulador detecta el archivo cada 10 iteraciones
- Genera todos los gráficos con los datos actuales
- Imprime reporte de coeficientes aerodinámicos
- **Continúa la simulación** después de generar los gráficos
- Elimina automáticamente el archivo trigger

**Útil para:**
- Revisar el progreso sin detener la simulación
- Tomar "snapshots" visuales en puntos específicos
- Verificar convergencia durante ejecución larga

---

## 3. Archivo STOP_SIMULATION.trigger - Detener limpiamente

**Uso:**
```bash
# En PowerShell:
New-Item -Path "STOP_SIMULATION.trigger" -ItemType File
```

**Qué sucede:**
- El simulador detecta el archivo cada 10 iteraciones
- Detiene la simulación después de completar la iteración actual
- Genera todos los outputs y gráficos finales
- Similar a Ctrl+C pero más "suave" (espera hasta chequeo de trigger)

---

## Ejemplo de flujo de trabajo

```python
# Iniciar simulación larga
mesh_gruesa, mesh_fina, geo = main(
    iteraciones=100000,
    graficos=True,
    ...
)
```

**Durante la ejecución:**

1. **Minuto 5:** Quieres ver cómo va
   ```powershell
   New-Item -Path "PLOT_NOW.trigger" -ItemType File
   ```
   → Genera gráficos y continúa

2. **Minuto 15:** Ves que ya convergió en los gráficos anteriores
   ```powershell
   New-Item -Path "STOP_SIMULATION.trigger" -ItemType File
   ```
   → Detiene limpiamente y genera outputs finales

3. **Alternativa:** Si prefieres detener inmediatamente
   → Presiona `Ctrl+C`

---

## Mensajes de consola

Al iniciar la simulación verás:
```
═══════════════════════════════════════════════════════════════════
🎮 CONTROLES INTERACTIVOS ACTIVOS:
═══════════════════════════════════════════════════════════════════
  • Presiona Ctrl+C para finalizar limpiamente con outputs completos
  • Crea 'PLOT_NOW.trigger' para generar gráficos sin detener
  • Crea 'STOP_SIMULATION.trigger' para detener limpiamente
═══════════════════════════════════════════════════════════════════
```

Cuando actives un trigger o presiones Ctrl+C, verás mensajes confirmando la acción.

---

## Notas técnicas

- Los triggers se verifican cada 10 iteraciones (optimización de I/O)
- Los archivos trigger se eliminan automáticamente después de procesarse
- Si hay un trigger antiguo al inicio, se elimina automáticamente
- La función `generar_graficos_y_outputs()` puede llamarse programáticamente
- Compatible con checkpoints y continuación de simulaciones

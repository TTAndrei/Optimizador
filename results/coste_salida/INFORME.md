# Coste de sacar frames

## Malla dx=0.004 — 387x918 = 355 266 celdas

Baseline (`off`, ninguna via activa): **34.208 it/s**, 29.23 ms/paso, 1500 iteraciones, guardado cada 50.

| config | it/s | sobrecoste | via | s/evento | eventos | disco |
|---|---|---|---|---|---|---|
| `shm` | 33.238 | +2.9 % | shm | 0.001 s | 30 | - |
| `shm_live` | 34.183 | +0.1 % | shm | 0.001 s | 30 | - |
| `shm_c4` | 32.904 | +4.0 % | shm | 0.001 s | 15 | - |
| `dump_400` | 34.6 | -1.1 % | dump | 0.006 s | 30 | 0.15 MB/frame |
| `dump_900` | 33.431 | +2.3 % | dump | 0.021 s | 30 | 0.52 MB/frame |
| `dump_1400` | 33.073 | +3.4 % | dump | 0.021 s | 30 | 0.52 MB/frame |
| `dump_900_nc` | 34.504 | -0.9 % | dump | 0.003 s | 30 | 1.13 MB/frame |
| `frames_100` | 32.858 | +4.1 % | frames | 0.281 s | 6 | 0.02 MB/PNG |
| `frames_180` | 32.231 | +6.1 % | frames | 0.308 s | 6 | 0.05 MB/PNG |
| `frames_600` | 30.66 | +11.6 % | frames | 0.753 s | 6 | 0.20 MB/PNG |
| `off_g10` | 32.041 | +0.0 % | - | - | - | - |
| `shm_g10` | 31.265 | +2.5 % | shm | 0.001 s | 150 | - |

**Fidelidad**: todas las configuraciones dan el mismo Cl y Cd (Cl=0.516302, Cd=0.036893).

### Sobrecoste extrapolado por cadencia

Coste por evento dividido entre el coste de las iteraciones que
cubre esa cadencia. No hace falta volver a simular.

| via | s/evento | cada 200 | cada 50 | cada 10 |
|---|---|---|---|---|
| `shm` (memoria compartida) | 0.001 | 0.0 % | 0.0 % | 0.2 % |
| `shm_live` (memoria compartida) | 0.001 | 0.0 % | 0.1 % | 0.4 % |
| `shm_c4` (memoria compartida) | 0.001 | 0.0 % | 0.0 % | 0.2 % |
| `dump_400` (dump_fields (npz)) | 0.006 | 0.1 % | 0.4 % | 2.1 % |
| `dump_900` (dump_fields (npz)) | 0.021 | 0.4 % | 1.4 % | 7.1 % |
| `dump_1400` (dump_fields (npz)) | 0.021 | 0.4 % | 1.4 % | 7.1 % |
| `dump_900_nc` (dump_fields (npz)) | 0.003 | 0.0 % | 0.2 % | 1.0 % |
| `frames_100` (save_frame (PNG)) | 0.281 | 4.8 % | 19.3 % | 96.3 % |
| `frames_180` (save_frame (PNG)) | 0.308 | 5.3 % | 21.1 % | 105.4 % |
| `frames_600` (save_frame (PNG)) | 0.753 | 12.9 % | 51.5 % | 257.6 % |
| `shm_g10` (memoria compartida) | 0.001 | 0.0 % | 0.0 % | 0.2 % |
### Qué hacer con esto

- **La vista en vivo es gratis**: 0.6 ms por publicación, 0.04 % del tiempo con `guardado=50` y 0.2 % incluso publicando cada 10 pasos. No hay razón para apagarla en el modo monitor.
- Añadir la vorticidad (`live_view`) sube la publicación de 0.6 a 1.3 ms. Sigue siendo ruido.
- **El volcado a disco lo domina zlib**: 21 ms por frame comprimido frente a 3 ms sin comprimir, o sea el 86 % del coste. El precio de quitarlo es el disco: 0.52 MB por frame comprimido contra 1.13 MB en crudo (×2.2). Comprimir sale a cuenta salvo que se vuelque muy seguido.
- `max_nx` 900 y 1400 cuestan lo mismo porque la ventana refinada no llega a 900 columnas y en los dos casos el paso de submuestreo es 1. Por debajo (400) sí baja, y mucho.
- **Dibujar dentro del solver es el único camino caro**: 753 ms por figura en `frames_600`, ×36 el volcado equivalente. A una figura cada 50 pasos son 52 % de sobrecoste. Volcar campos y montar el vídeo después con `render_videos.py` da el mismo vídeo por una fracción, y además no ocupa la GPU.

El sobrecoste leído en la columna `it/s` tiene ±3 % de ruido entre corridas (se ve en que `shm` y `shm_live` salen a +2.9 % y +0.1 % midiendo casi lo mismo). Los números fiables son los de **s/evento**, que salen de los sub-cronómetros del propio bloque de salida y no de comparar dos corridas.

## Malla dx=0.002 — 713x1772 = 1 263 436 celdas

Baseline (`off`, ninguna via activa): **22.63 it/s**, 44.19 ms/paso, 1000 iteraciones, guardado cada 50.

| config | it/s | sobrecoste | via | s/evento | eventos | disco |
|---|---|---|---|---|---|---|
| `shm` | 22.493 | +0.6 % | shm | 0.002 s | 20 | - |
| `shm_live` | 22.49 | +0.6 % | shm | 0.004 s | 20 | - |
| `shm_c4` | 22.495 | +0.6 % | shm | 0.002 s | 10 | - |
| `dump_400` | 22.331 | +1.3 % | dump | 0.010 s | 20 | 0.24 MB/frame |
| `dump_900` | 22.307 | +1.4 % | dump | 0.020 s | 20 | 0.50 MB/frame |
| `dump_1400` | 21.961 | +3.0 % | dump | 0.064 s | 20 | 1.54 MB/frame |
| `dump_900_nc` | 22.364 | +1.2 % | dump | 0.003 s | 20 | 1.09 MB/frame |
| `frames_100` | 21.203 | +6.7 % | frames | 0.676 s | 4 | 0.02 MB/PNG |
| `frames_180` | 21.071 | +7.4 % | frames | 0.712 s | 4 | 0.04 MB/PNG |
| `frames_600` | 20.392 | +11.0 % | frames | 1.179 s | 4 | 0.20 MB/PNG |
| `off_g10` | 20.001 | +0.0 % | - | - | - | - |
| `shm_g10` | 20.054 | -0.3 % | shm | 0.002 s | 100 | - |

**Fidelidad**: todas las configuraciones dan el mismo Cl y Cd (Cl=0.463138, Cd=0.042711).

### Sobrecoste extrapolado por cadencia

Coste por evento dividido entre el coste de las iteraciones que
cubre esa cadencia. No hace falta volver a simular.

| via | s/evento | cada 200 | cada 50 | cada 10 |
|---|---|---|---|---|
| `shm` (memoria compartida) | 0.002 | 0.0 % | 0.1 % | 0.4 % |
| `shm_live` (memoria compartida) | 0.004 | 0.0 % | 0.2 % | 0.8 % |
| `shm_c4` (memoria compartida) | 0.002 | 0.0 % | 0.1 % | 0.4 % |
| `dump_400` (dump_fields (npz)) | 0.010 | 0.1 % | 0.4 % | 2.2 % |
| `dump_900` (dump_fields (npz)) | 0.020 | 0.2 % | 0.9 % | 4.5 % |
| `dump_1400` (dump_fields (npz)) | 0.064 | 0.7 % | 2.9 % | 14.5 % |
| `dump_900_nc` (dump_fields (npz)) | 0.003 | 0.0 % | 0.1 % | 0.7 % |
| `frames_100` (save_frame (PNG)) | 0.676 | 7.7 % | 30.6 % | 153.0 % |
| `frames_180` (save_frame (PNG)) | 0.712 | 8.1 % | 32.2 % | 161.1 % |
| `frames_600` (save_frame (PNG)) | 1.179 | 13.3 % | 53.3 % | 266.7 % |
| `shm_g10` (memoria compartida) | 0.002 | 0.0 % | 0.1 % | 0.3 % |
### Qué hacer con esto

- **La vista en vivo es gratis**: 1.8 ms por publicación, 0.08 % del tiempo con `guardado=50` y 0.4 % incluso publicando cada 10 pasos. No hay razón para apagarla en el modo monitor.
- Añadir la vorticidad (`live_view`) sube la publicación de 1.8 a 3.8 ms. Sigue siendo ruido.
- **El volcado a disco lo domina zlib**: 20 ms por frame comprimido frente a 3 ms sin comprimir, o sea el 85 % del coste. El precio de quitarlo es el disco: 0.50 MB por frame comprimido contra 1.09 MB en crudo (×2.2). Comprimir sale a cuenta salvo que se vuelque muy seguido.
- **Dibujar dentro del solver es el único camino caro**: 1178 ms por figura en `frames_600`, ×60 el volcado equivalente. A una figura cada 50 pasos son 53 % de sobrecoste. Volcar campos y montar el vídeo después con `render_videos.py` da el mismo vídeo por una fracción, y además no ocupa la GPU.

El sobrecoste leído en la columna `it/s` tiene ±3 % de ruido entre corridas (se ve en que `shm` y `shm_live` salen a +2.9 % y +0.1 % midiendo casi lo mismo). Los números fiables son los de **s/evento**, que salen de los sub-cronómetros del propio bloque de salida y no de comparar dos corridas.


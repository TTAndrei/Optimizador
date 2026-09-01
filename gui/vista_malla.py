"""Previsualizacion de malla y mascara, en CPU, antes de gastar GPU.

Todo lo que se comprueba aqui son fallos que no dan error: producen una
simulacion que corre entera y da un numero equivocado. Merece la pena pagar dos
segundos de CPU antes de pagar horas de GPU.

  - Una pared mas fina que una celda NO EXISTE. La rasterizacion es par/impar
    sobre centros de celda: si la pared del conducto no cubre ningun centro, el
    fluido se fuga por ella y el balance de masa no cierra.
  - El IBM de ghost-cell necesita unas 8 celdas de fluido entre paredes
    enfrentadas. Por debajo, los puntos imagen de las dos paredes se
    interpenetran, la reparacion se agota y la garganta se tapona.
  - Las normales solo son correctas dentro de la banda fina: el gradiente de la
    distancia se calcula con el dx MINIMO global, asi que un cuerpo en la zona
    estirada da fuerzas mal.
  - El dt no lo fija solo el CFL: el limite viscoso va con dx^2, y es lo que
    hundio la malla de dx=0.001 del estudio.
"""
from __future__ import annotations

import numpy as np
from matplotlib.path import Path
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg


def ejes(escena):
    from Simulador2D import (absorber_slivers, generar_malla_estirada,
                             generar_malla_estirada_intervalo)
    dx = escena.dx_min
    dy = escena.dy
    fe = escena.factor_expansion
    # El solver tapa el crecimiento de celda en ratio_max_malla*dx_min. Sin
    # replicarlo aqui la previsualizacion enseña una malla que no es la que se
    # simula, que es justo lo que hay que evitar.
    ratio = float(escena.solver.get("ratio_max_malla", 50))
    banda = escena.banda_fina()
    if banda is None:
        X = generar_malla_estirada(escena.Lx, escena.Lx / 2, dx, fe, escena.Lx)
        Y = generar_malla_estirada(escena.Ly, escena.Ly / 2, dy, fe, escena.Ly)
        return np.asarray(X), np.asarray(Y)
    x0, x1, y0, y1 = banda
    # Mismo tratamiento que en el solver: si no, la previsualizacion promete un
    # dt que la simulacion no va a tener.
    X = absorber_slivers(generar_malla_estirada_intervalo(
        escena.Lx, max(0.0, x0), min(escena.Lx, x1), dx, fe,
        dx_max=ratio * dx), dx)
    Y = absorber_slivers(generar_malla_estirada_intervalo(
        escena.Ly, max(0.0, y0), min(escena.Ly, y1), dy, fe,
        dx_max=ratio * dy), dy)
    return np.asarray(X), np.asarray(Y)


def mascara(escena, X, Y):
    XX, YY = np.meshgrid(X, Y, indexing="xy")
    pts = np.column_stack((XX.ravel(), YY.ravel()))
    solid = np.zeros((len(Y), len(X)), dtype=bool)
    # Las MISMAS coordenadas que va a ver el solver, incluida la extension de
    # las bocas fuera de la caja: si no, la previsualizacion ensena una mascara
    # que no es la que se simula.
    for c, x, y in escena.contornos_para_malla():
        dentro = Path(np.column_stack((x, y))).contains_points(
            pts, radius=-1e-9).reshape(solid.shape)
        solid |= (~dentro) if c.rol == "exterior" else dentro
    return solid


# Que se pinta en el modo de malla. El valor es (etiqueta, unidad).
MODOS_MALLA = {
    "tamaño":  ("tamaño de celda", "m"),
    "dx":      ("dx", "m"),
    "dy":      ("dy", "m"),
    "aspecto": ("relación de aspecto", ":1"),
}


def mapa_celda(X, Y, modo="tamaño"):
    """Mapa (ny, nx) del tamaño de celda en cada nodo de la malla.

    `np.gradient` sobre el eje da el espaciado local en el propio nodo —media de
    los dos intervalos vecinos, y el intervalo suelto en los extremos— asi que
    el mapa tiene la forma de la malla y no hay que interpolar nada.

    "tamaño" es el lado MAYOR de la celda: es el que manda en lo que se puede
    resolver ahi, y una celda de 0.004 x 0.2 no resuelve nada de 0.004.
    """
    hx = np.gradient(np.asarray(X, float))
    hy = np.gradient(np.asarray(Y, float))
    HX, HY = np.meshgrid(hx, hy, indexing="xy")
    if modo == "dx":
        return HX
    if modo == "dy":
        return HY
    if modo == "aspecto":
        return np.maximum(HX / HY, HY / HX)
    return np.maximum(HX, HY)


def diagnostico(escena):
    """(texto, avisos, X, Y, solid). Barato: todo numpy en CPU."""
    X, Y = ejes(escena)
    solid = mascara(escena, X, Y)
    ny, nx = solid.shape
    celdas = ny * nx
    avisos = list(escena.avisos())

    dx = float(np.min(np.diff(X))) if nx > 1 else escena.dx_min
    dy = float(np.min(np.diff(Y))) if ny > 1 else escena.dx_min
    h = min(dx, dy)
    nu = escena.solver.get("nu", 1e-5)
    U = escena.solver.get("v0x", 1.0)
    cfl = escena.solver.get("CFL", 0.5)
    dt_conv = cfl * h / max(U, 1e-9)
    dt_visc = 0.25 * h * h / max(nu, 1e-30)
    dt = min(dt_conv, dt_visc)

    # 9 campos float32 en GPU (u, v, p, y los de trabajo del multigrid) como
    # orden de magnitud; el pico real es mayor, pero sirve para avisar.
    mem_mb = celdas * 4 * 9 / 1e6

    if solid.any() and not solid.all():
        from scipy.ndimage import distance_transform_edt
        hueco = 2.0 * float(distance_transform_edt(~solid).max())
        grosor = 2.0 * float(distance_transform_edt(solid).max())
    else:
        hueco = grosor = float("nan")

    if escena.contornos and not solid.any():
        avisos.append("La máscara sólida ha salido vacía: la geometría no cae "
                      "dentro del dominio, o las unidades del DXF no eran las "
                      "que crees.")
    if grosor == grosor and grosor < 2.0:
        avisos.append(f"La pared más fina ocupa {grosor:.1f} celdas. Por debajo "
                      "de 2 no existe para el solver y el fluido se fuga: sube "
                      "dx_min o engorda la pared en el CAD.")
    if hueco == hueco and hueco < 8.0:
        avisos.append(f"El hueco de fluido más estrecho son {hueco:.1f} celdas. "
                      "El IBM necesita ~8: por debajo, los puntos imagen de las "
                      "dos paredes se interpenetran y la garganta se tapona.")
    # SA cuenta la distancia a pared en NUMERO DE CELDAS y la multiplica por el
    # tamano de la celda donde estas (Simulador2D._compute_sa_wall_distance), asi
    # que solo es exacta si todas las celdas entre el punto y la pared miden lo
    # mismo. Con un perfil se cumple solo: la capa limite vive dentro de la banda
    # fina uniforme. Con un conducto la cortadura ocupa TODA la seccion, asi que
    # la banda fina tiene que cubrirla. Medido en un canal: con la banda cubriendo
    # la seccion el error en d es del 0 % (3 % de pico); con la banda solo en el
    # centro y un estiramiento de x1.9 dentro del canal, 35 % mediano y 51 % de
    # pico.
    if escena.solver.get("turb_model") == "sa" and solid.any():
        fluido_y = ~solid.all(axis=1)          # filas con algo de fluido
        if fluido_y.any() and ny > 1:
            dyf = np.diff(Y)[fluido_y[:-1]]
            if dyf.size and dyf.max() > 1.5 * dyf.min():
                avisos.append(
                    f"El modelo SA necesita celdas del mismo tamaño entre el "
                    f"punto y la pared, y dentro del fluido dy va de "
                    f"{dyf.min():.4f} a {dyf.max():.4f} (×{dyf.max()/dyf.min():.1f}). "
                    f"La distancia a pared saldrá mal donde la malla se estira. "
                    f"Amplía la banda fina para que cubra toda la sección, o usa "
                    f"wale/none.")

    banda = escena.banda_fina()
    if banda:
        bx0, bx1, by0, by1 = banda
        for c in escena.contornos:
            if c.rol != "cuerpo":
                continue
            x0, y0, x1, y1 = c.bbox()
            if not (bx0 <= x0 and x1 <= bx1 and by0 <= y0 and y1 <= by1):
                avisos.append(f"El cuerpo «{c.nombre}» se sale de la banda "
                              "fina: ahí las normales de pared salen mal y las "
                              "fuerzas con ellas.")
    # Relacion de aspecto de celda. Medido en el conducto de 4.8x1 con la banda
    # fina puesta solo alrededor del cilindro: dx crece x20 (0.01 -> 0.20) contra
    # dy=0.01, y la simulacion revienta en la iteracion 790. Con la banda
    # cubriendo la seccion, o acotando el crecimiento a x8, aguanta. El
    # multigrid se degrada con malla anisotropa y aqui deja de converger.
    # Relacion de aspecto JUNTO A LA PARED. El aspecto por si solo no decide:
    # un perfil en 24x16 tiene celdas 28:1 lejos, en corriente uniforme, y corre
    # sin problema. Lo que revienta es una celda larga y plana donde hay
    # cortadura, es decir pegada a una pared. Medido en el conducto de 4.8x1:
    # con celdas 20:1 contra la pared, blowup en la iteracion 790; acotando el
    # crecimiento de dx a x8, 3000 pasos limpios.
    if nx > 1 and ny > 1:
        rx = np.diff(X)[None, :] / np.diff(Y)[:, None]
        rx = np.maximum(rx, 1.0 / rx)
        junto = np.zeros_like(rx, dtype=bool)
        if solid.any():
            from scipy.ndimage import binary_dilation
            junto |= (binary_dilation(solid) & ~solid)[:-1, :-1]
        for p in escena.parches:                 # la pared tambien puede ser el borde
            if p.tipo != "noslip":
                continue
            junto[{"bottom": np.s_[0, :], "top": np.s_[-1, :],
                   "left": np.s_[:, 0], "right": np.s_[:, -1]}[p.lado]] = True
        if junto.any():
            aspecto = float(rx[junto].max())
            if aspecto > 10.0:
                avisos.append(
                    f"Junto a la pared hay celdas de relación de aspecto "
                    f"{aspecto:.0f}:1. Por encima de ~10 el multigrid deja de "
                    f"converger y la simulación revienta. Baja «Ratio máximo "
                    f"dx_max/dx_min» o amplía la banda fina.")
    if dt_visc < dt_conv:
        avisos.append(f"El paso lo limita la viscosidad, no el CFL "
                      f"(dt_visc={dt_visc:.2e} < dt_conv={dt_conv:.2e}). "
                      "Va con dx², así que refinar sale el doble de caro de lo "
                      "que parece.")

    it = escena.solver.get("iteraciones", 0)
    texto = (
        f"Malla        {nx} x {ny} = {celdas:,} celdas\n"
        f"dx, dy mín   {dx:.5f}, {dy:.5f} m\n"
        f"dx máx       {np.max(np.diff(X)):.4f} m  (ratio {np.max(np.diff(X))/dx:.0f}:1)\n"
        f"Sólido       {100*solid.mean():.1f} % de las celdas\n"
        f"Hueco mínimo {hueco:.1f} celdas   Pared mínima {grosor:.1f} celdas\n"
        f"dt previsto  {dt:.3e} s  (conv {dt_conv:.2e} / visc {dt_visc:.2e})\n"
        f"t físico     {dt*it:.2f} s en {it} iteraciones\n"
        f"Memoria GPU  ~{mem_mb:.0f} MB en campos"
    ).replace(",", " ")
    return texto, avisos, X, Y, solid


def dialogo_malla(padre, escena):
    try:
        texto, avisos, X, Y, solid = diagnostico(escena)
    except Exception as e:
        QtWidgets.QMessageBox.critical(padre, "No se pudo generar la malla",
                                       f"{type(e).__name__}: {e}")
        return

    d = QtWidgets.QDialog(padre)
    d.setWindowTitle("Previsualización de malla")
    d.resize(1000, 700)
    v = QtWidgets.QVBoxLayout(d)

    vista = pg.PlotWidget()
    vista.setBackground("#ffffff")
    vista.setAspectLocked(True)
    img = pg.ImageItem(solid.T.astype(np.float32))
    img.setRect(QtCore.QRectF(X[0], Y[0], X[-1] - X[0], Y[-1] - Y[0]))
    img.setLookupTable(np.array([[245, 245, 242, 255], [40, 52, 64, 255]],
                                dtype=np.ubyte))
    vista.addItem(img)
    # Una de cada 20 lineas: dibujarlas todas es ilegible y lento.
    for xi in X[::20]:
        vista.addItem(pg.InfiniteLine(pos=xi, angle=90,
                                      pen=pg.mkPen("#00000018")))
    for yi in Y[::20]:
        vista.addItem(pg.InfiniteLine(pos=yi, angle=0,
                                      pen=pg.mkPen("#00000018")))
    for c in escena.contornos:
        x, y = c.arrays()
        vista.addItem(pg.PlotDataItem(x, y, pen=pg.mkPen("#c05a2a", width=2)))
    v.addWidget(vista, 1)

    info = QtWidgets.QPlainTextEdit(texto)
    info.setReadOnly(True)
    info.setStyleSheet("font-family:monospace;")
    info.setMaximumHeight(150)
    v.addWidget(info)

    if avisos:
        av = QtWidgets.QPlainTextEdit("\n".join(f"• {a}" for a in avisos))
        av.setReadOnly(True)
        av.setStyleSheet("color:#8a3b1a;background:#fff6f0;")
        av.setMaximumHeight(130)
        v.addWidget(av)

    b = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.StandardButton.Close)
    b.rejected.connect(d.reject)
    v.addWidget(b)
    d.exec()

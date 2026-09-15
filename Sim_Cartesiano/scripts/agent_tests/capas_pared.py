"""
Estudio del numero de capas de la extrapolacion de presion de pared.

QUE SE PREGUNTA
    La presion de pared se reconstruye ajustando una recta a la presion muestreada
    en N capas de fluido a lo largo de la normal, en s_k = (k-0.5)h, y evaluandola
    en s=0. El codigo usa N=5. Ese 5 nunca se justifico con una medida. Esto lo mide.

COMO
    Barrido offline sobre los campos finales ya calculados de resultados_finales/,
    que traen x, y, u, v, p y la mascara de solido. La malla se reconstruye desde
    cero con los mismos parametros de generacion y se comprueba que sale identica
    (nodos y mascara) antes de usarla, asi que las normales, la frontera y el ds
    son exactamente los de produccion. Solo se cambia N.

    No hace falta GPU-simular nada: N solo entra en la reconstruccion de p_w, que
    es post-proceso puro sobre el campo de presion.

QUE MIDE POR CADA (dx, alpha, N)
    Cl_p, Cd_p     coeficientes de la parte de PRESION. La traccion viscosa se
                   muestrea en un unico punto a 0.5 celdas y no depende de N, y
                   ademas los campos volcados no guardan nu_t, asi que el viscoso
                   no se puede reconstruir. Cl es presion-dominante; Cd no.
    rugosidad      RMS de la segunda diferencia de Cp a lo largo del contorno,
                   normalizada. Es el ruido de periodo una celda que la memoria
                   atribuye al escalonado de la rasterizacion.
    corr_subcelda  correlacion entre el residuo de Cp (quitada la tendencia suave)
                   y la distancia sub-celda de la celda frontera a la superficie,
                   leida de la SDF. Si el ruido es el escalonado, esta correlacion
                   es alta a N=1 y baja al promediar capas. Es la prueba directa
                   de la afirmacion de la memoria, no un proxy.
    residuo, b     residuo del ajuste y pendiente (gradiente normal de presion).
    toca_solido    fraccion de puntos de muestreo de cada capa cuyo estarcido
                   bilineal pisa celdas solidas. Candidato a explicar el ruido de
                   N bajo mejor que la rasterizacion.

    Ademas, con N>=3, ajuste cuadratico: separa el sesgo de curvatura (el estarcido
    mide en celdas, no en unidades fisicas, asi que su longitud fisica cambia x4
    entre la malla gruesa y la fina) del ruido.

PARTE C — VEREDICTO
    Para cada N se rehace el GCI de Roache sobre la terna 0.008/0.004/0.002 y se
    compara la sensibilidad a N con la banda de incertidumbre de malla ya publicada.
    Si mover N de 3 a 9 cambia Cl_p mucho menos que el GCI, el 5 es defendible y la
    memoria gana un numero; si no, hay hallazgo.

AVISO
    Los campos son instantaneos y de fases distintas en cada malla. La comparacion
    ENTRE N es exacta (mismo campo, mismo instante). La comparacion ENTRE MALLAS
    arrastra la misma limitacion de fase que el estudio publicado.

Uso:
    .venv/bin/python scripts/agent_tests/capas_pared.py
    .venv/bin/python scripts/agent_tests/capas_pared.py --solo-figuras
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "agent_tests"))
os.chdir(ROOT)

import numpy as np
import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import Simulador2D as S
import verificacion_numerica as vn

OUT = os.path.join(ROOT, "results", "capas_pared")
CAMPOS = os.path.join(ROOT, "resultados_finales", "naca0012", "campos")
PERFIL = os.path.join(ROOT, "resultados_finales", "perfiles", "naca0012.dat")
os.makedirs(OUT, exist_ok=True)

DXS = (0.008, 0.006, 0.004, 0.002)
ANGULOS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
NS = tuple(range(1, 13))
TERNA = (0.002, 0.004, 0.008)

# Dominio C validado, identico al de resultados_finales.py
LX, LY, CX = 24.0, 16.0, 6.0
CHORD = 1.0
FEXP = 1.1
RATIO_MAX = 50
AWX, AWY = 1.5, 1.0
RHO, U_INF = 1.0, 1.0


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def k_dx(dx):
    return f"{dx:.4f}"


# ----------------------------------------------------------------------
# Reconstruccion de la malla de produccion
# ----------------------------------------------------------------------
def construir_mesh(dx, alpha):
    """Malla identica a la del run que genero el campo. Se verifica, no se supone."""
    cy = LY / 2
    dxmax = RATIO_MAX * dx
    x_fino_min = max(0.0, (CX + CHORD * 0.5) - 0.5 * AWX)
    x_fino_max = min(LX, x_fino_min + AWX + 1.5 * CHORD)
    X1 = S.generar_malla_estirada_intervalo(
        L=LX, x_fino_min=x_fino_min, x_fino_max=x_fino_max,
        dx_min=dx, factor_expansion=FEXP, dx_max=dxmax)
    Y1 = S.generar_malla_estirada(
        L=LY, x_centro=cy, dx_min=dx, factor_expansion=FEXP,
        ancho_zona_fina=AWY, dx_max=dxmax)

    m = S.Mesh(LX, LY, 0, U_INF, 0.0, dx, dx, usar_wale=False, turb_model="sa",
               X_1d=X1, Y_1d=Y1)
    m.ibm_wall_mode = "ghost_noslip"
    m.ibm_sdf_source = "edt"
    m._disable_reforzar = True
    m.wall_treatment = "consistent"
    m.load_solids_from_file(filepath=PERFIL, chord=CHORD, x_offset=CX, y_offset=cy,
                            alpha_deg=alpha, fill=True, plot=False,
                            min_te_height=1.0 * dx)
    return m


def cargar_campo(mesh, dx, alpha):
    p = os.path.join(CAMPOS, f"dx{k_dx(dx)}_a{alpha:04.1f}_campo.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    assert np.allclose(cp.asnumpy(mesh.X_1d), d["x"], atol=1e-6), "malla X no reproducida"
    assert np.allclose(cp.asnumpy(mesh.Y_1d), d["y"], atol=1e-6), "malla Y no reproducida"
    assert bool((cp.asnumpy(mesh.solid) == d["solid"]).all()), "mascara solida no reproducida"
    mesh.u = cp.asarray(d["u"], dtype=cp.float32)
    mesh.v = cp.asarray(d["v"], dtype=cp.float32)
    mesh.p = cp.asarray(d["p"], dtype=cp.float32)
    return d


# ----------------------------------------------------------------------
# Reconstruccion de p_w con N arbitrario
#
# Replica los pasos 1-3 y 7 de Mesh.compute_surface_forces_definitive. Hay que
# duplicarlos porque alli el numero de capas lo fija el NOMBRE del modo
# (reconstruction_layers[...]) y el argumento n_extrap_layers no llega a usarse
# nunca. La duplicacion se valida a N=5 contra el metodo de produccion.
# ----------------------------------------------------------------------
def geometria_pared(mesh):
    boundary = mesh._solid_boundary_mask(use_diagonals=True)
    phi, nx_all, ny_all = mesh._signed_distance_and_normals()
    eps = cp.float32(1e-12)
    dx_loc = mesh.vol_x[cp.newaxis, :]
    dy_loc = mesh.vol_y[:, cp.newaxis]
    ds = cp.sqrt((ny_all * dx_loc) ** 2 + (nx_all * dy_loc) ** 2) + eps
    JJ, II = mesh.JJ, mesh.II
    j_face = JJ + 0.5 * nx_all
    i_face = II + 0.5 * ny_all
    x_face = mesh._bilinear_interpolate(mesh.XX, j_face, i_face)
    y_face = mesh._bilinear_interpolate(mesh.YY, j_face, i_face)
    return dict(boundary=boundary, phi=phi, nx=nx_all, ny=ny_all, ds=ds,
                x_face=x_face, y_face=y_face, JJ=JJ, II=II)


def muestrear_capas(mesh, g, n_max):
    """Presion en las n_max capas y fraccion de estarcido que pisa solido."""
    pos = np.array([0.5 + k for k in range(n_max)], dtype=np.float32)
    solid_f = mesh.solid.astype(cp.float32)
    p_layers, s_layers = [], []
    for s in pos:
        jk = g["JJ"] + cp.float32(s) * g["nx"]
        ik = g["II"] + cp.float32(s) * g["ny"]
        p_layers.append(mesh._bilinear_interpolate(mesh.p, jk, ik))
        s_layers.append(mesh._bilinear_interpolate(solid_f, jk, ik))
    return pos, p_layers, s_layers


def ajuste(pos, p_layers, N, grado=1):
    """Ajuste por minimos cuadrados sobre las N primeras capas. Devuelve a, b, residuo."""
    x = np.asarray(pos[:N], dtype=np.float64)
    y = cp.stack([p.astype(cp.float32) for p in p_layers[:N]], axis=0)
    if N == 1:
        z = cp.zeros_like(y[0])
        return y[0], z, z
    A = np.vander(x, grado + 1, increasing=True)          # [1, s, s^2...]
    pinv = cp.asarray(np.linalg.pinv(A).astype(np.float32))  # (grado+1, N)
    coef = cp.tensordot(pinv, y, axes=(1, 0))             # (grado+1, ny, nx)
    fit = sum(coef[k][cp.newaxis] * cp.asarray(x[:, None, None].astype(np.float32)) ** k
              for k in range(grado + 1))
    resid = cp.sqrt(cp.mean((y - fit) ** 2, axis=0))
    return coef[0], coef[1], resid


def debias_y_fuerzas(mesh, g, p_wall, alpha_deg):
    """Debias affine+mean y fuerza de presion. Copiado de compute_surface_forces_definitive."""
    w = g["boundary"].astype(cp.float32)
    ds = g["ds"]
    band = max(1, int(min(mesh.nx, mesh.ny) * 0.05))
    mask = cp.zeros_like(mesh.p, dtype=cp.bool_)
    mask[:band, :] = True; mask[-band:, :] = True
    mask[:, :band] = True; mask[:, -band:] = True
    mask = mask & (~mesh.solid)
    p_e = cp.asnumpy(mesh.p[mask]).astype(np.float64)
    x_e = cp.asnumpy(mesh.XX[mask]).astype(np.float64)
    y_e = cp.asnumpy(mesh.YY[mask]).astype(np.float64)
    A = np.column_stack((np.ones_like(p_e), x_e, y_e))
    c, _, _, _ = np.linalg.lstsq(A, p_e, rcond=None)
    p_bg = cp.float32(c[0]) + cp.float32(c[1]) * g["x_face"] + cp.float32(c[2]) * g["y_face"]
    p_f = p_wall - p_bg
    wds = w * ds
    p_f = p_f - cp.sum(p_f * wds) / (cp.sum(wds) + cp.float32(1e-30))

    Fx_p = float(cp.sum(-p_f * g["nx"] * wds))
    Fy_p = float(cp.sum(-p_f * g["ny"] * wds))
    # El perfil se carga ROTADO y el inflow es horizontal, asi que el angulo del
    # flujo libre es 0 y no 4 grados. Rotar aqui daria un Cd un 80% mas alto.
    a = mesh._get_freestream_angle_rad()
    drag = Fx_p * np.cos(a) + Fy_p * np.sin(a)
    lift = -Fx_p * np.sin(a) + Fy_p * np.cos(a)
    k = 2.0 / (RHO * U_INF ** 2 * CHORD)
    return k * lift, k * drag, p_f


# ----------------------------------------------------------------------
# Metricas de ruido sobre el contorno
# ----------------------------------------------------------------------
def metricas_ruido(g, cp_face, mesh, dx):
    """Rugosidad de Cp y correlacion del residuo con la posicion sub-celda."""
    b = g["boundary"]
    xb = cp.asnumpy(g["x_face"][b]).astype(np.float64)
    nyb = cp.asnumpy(g["ny"][b]).astype(np.float64)
    cpb = cp.asnumpy(cp_face[b]).astype(np.float64)
    # Distancia del centro de la celda frontera a la superficie. La SDF del solver
    # ya viene EN CELDAS (contrato de _signed_distance_and_normals), no en metros.
    # Es la distancia variable entre 0 y una celda de la que habla la memoria.
    sub = np.abs(cp.asnumpy(g["phi"][b]).astype(np.float64))

    rug, corr, n_tot = [], [], 0
    for sel in (nyb > 0, nyb <= 0):
        if sel.sum() < 12:
            continue
        o = np.argsort(xb[sel])
        y = cpb[sel][o]
        s = sub[sel][o]
        d2 = y[2:] - 2 * y[1:-1] + y[:-2]
        den = np.std(y) + 1e-12
        rug.append(np.sqrt(np.mean(d2 ** 2)) / den)
        # tendencia suave = media movil de 9; el residuo es el ruido de periodo corto
        k = 9
        ker = np.ones(k) / k
        suave = np.convolve(np.pad(y, k // 2, mode="edge"), ker, mode="valid")
        r = y - suave[:len(y)]
        if np.std(r) > 1e-12 and np.std(s) > 1e-12:
            corr.append(abs(np.corrcoef(r, s)[0, 1]))
        n_tot += sel.sum()
    return (float(np.mean(rug)) if rug else np.nan,
            float(np.mean(corr)) if corr else np.nan,
            int(n_tot))


# ----------------------------------------------------------------------
# Barrido
# ----------------------------------------------------------------------
def barrido():
    path = os.path.join(OUT, "barrido.json")
    datos = json.load(open(path)) if os.path.exists(path) else {}
    n_max = max(NS)

    for dx in DXS:
        for a in ANGULOS:
            clave = f"dx{k_dx(dx)}_a{a:.1f}"
            if clave in datos:
                continue
            if not os.path.exists(os.path.join(CAMPOS, f"dx{k_dx(dx)}_a{a:04.1f}_campo.npz")):
                continue
            t0 = time.time()
            mesh = construir_mesh(dx, a)
            if cargar_campo(mesh, dx, a) is None:
                continue
            g = geometria_pared(mesh)
            pos, p_layers, s_layers = muestrear_capas(mesh, g, n_max)
            b = g["boundary"]
            toca = [float(cp.mean((s_layers[k][b] > 1e-6).astype(cp.float32)))
                    for k in range(n_max)]

            q = 0.5 * RHO * U_INF ** 2
            reg = {"dx": dx, "alpha": a, "n_caras": int(cp.sum(b)),
                   "toca_solido_por_capa": [round(t, 5) for t in toca], "N": {}}

            for N in NS:
                a0, b1, res = ajuste(pos, p_layers, N, grado=1)
                cl, cd, p_f = debias_y_fuerzas(mesh, g, a0, a)
                rug, corr, ncar = metricas_ruido(g, p_f / cp.float32(q), mesh, dx)
                e = {"Cl_p": round(cl, 6), "Cd_p": round(cd, 6),
                     "residuo_medio": float(cp.mean(res[b])),
                     "residuo_p95": float(cp.percentile(res[b], 95)),
                     "pendiente_abs_media": float(cp.mean(cp.abs(b1[b]))),
                     "rugosidad_cp": rug, "corr_subcelda": corr}
                if N >= 3:
                    a2, _, res2 = ajuste(pos, p_layers, N, grado=2)
                    cl2, cd2, _ = debias_y_fuerzas(mesh, g, a2, a)
                    e.update({"Cl_p_cuad": round(cl2, 6), "Cd_p_cuad": round(cd2, 6),
                              "residuo_medio_cuad": float(cp.mean(res2[b]))})
                reg["N"][str(N)] = e

            datos[clave] = reg
            del mesh, g, p_layers, s_layers
            cp.get_default_memory_pool().free_all_blocks()
            n5 = reg["N"]["5"]
            log(f"{clave}: Cl_p(N=5)={n5['Cl_p']:.4f} "
                f"Cl_p(N=1)={reg['N']['1']['Cl_p']:.4f} "
                f"Cl_p(N=9)={reg['N']['9']['Cl_p']:.4f}  ({time.time()-t0:.1f}s)")
            with open(path, "w") as f:
                json.dump(datos, f, indent=1)
    return datos


def validar_contra_produccion(dx=0.004, a=4.0):
    """La reimplementacion tiene que dar lo mismo que el solver a N=5."""
    mesh = construir_mesh(dx, a)
    cargar_campo(mesh, dx, a)
    g = geometria_pared(mesh)
    pos, p_layers, _ = muestrear_capas(mesh, g, 5)
    a0, _, _ = ajuste(pos, p_layers, 5, grado=1)
    cl, cd, _ = debias_y_fuerzas(mesh, g, a0, a)
    r = mesh.compute_drag_lift(1e-5, rho=RHO, n_extrap_layers=5,
                               pressure_wall_reconstruction="linear_5")
    k = 2.0 / (RHO * U_INF ** 2 * CHORD)
    cl_ref, cd_ref = k * r["Lift_p"], k * r["Drag_p"]
    out = {"Cl_p_estudio": cl, "Cl_p_solver": cl_ref,
           "Cd_p_estudio": cd, "Cd_p_solver": cd_ref,
           "dif_rel_Cl": abs(cl - cl_ref) / max(abs(cl_ref), 1e-12),
           "dif_rel_Cd": abs(cd - cd_ref) / max(abs(cd_ref), 1e-12)}
    log(f"validacion N=5: Cl_p {cl:.6f} vs {cl_ref:.6f} "
        f"(dif {out['dif_rel_Cl']:.2e}), Cd_p {cd:.6f} vs {cd_ref:.6f} "
        f"(dif {out['dif_rel_Cd']:.2e})")
    with open(os.path.join(OUT, "validacion_n5.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


# ----------------------------------------------------------------------
# Varianza del valor reconstruido — la afirmacion central de la memoria
#
# "Promediar cinco capas en lugar de tomar una sola reduce la varianza del valor
# reconstruido". Eso seria cierto si p_w fuese la MEDIA de las capas, pero p_w es
# la ORDENADA EN EL ORIGEN de una recta ajustada en s=0.5..N-0.5, o sea una
# EXTRAPOLACION fuera del rango de datos. El intercepto es una combinacion lineal
# de las capas con pesos que cambian de signo, y eso amplifica ruido en vez de
# promediarlo. El factor exacto, con ruido blanco e independiente por capa, es
#     var(a)/sigma^2 = 1/N + s_media^2 / Sss
# que vale 1 en N=1, 2.5 en N=2, y no baja de 1 hasta N=5. Aqui se calcula ese
# factor y ademas el EMPIRICO, que usa la covarianza real entre capas medida sobre
# el contorno (las capas comparten estarcido bilineal, asi que su ruido esta
# correlacionado y el promediado rinde todavia menos).
# ----------------------------------------------------------------------
def pesos_intercepto(N):
    x = np.arange(N, dtype=np.float64) + 0.5
    A = np.column_stack((np.ones(N), x))
    return np.linalg.pinv(A)[0]          # fila del intercepto


def factor_teorico(N):
    if N == 1:
        return 1.0
    return float(np.sum(pesos_intercepto(N) ** 2))


def varianza_intercepto(dx=0.002, a=4.0, n_max=12):
    """Factor de amplificacion de ruido del intercepto, teorico y empirico."""
    mesh = construir_mesh(dx, a)
    if cargar_campo(mesh, dx, a) is None:
        return None
    g = geometria_pared(mesh)
    _, p_layers, _ = muestrear_capas(mesh, g, n_max)
    b = g["boundary"]
    xb = cp.asnumpy(g["x_face"][b]).astype(np.float64)
    nyb = cp.asnumpy(g["ny"][b]).astype(np.float64)
    P = np.stack([cp.asnumpy(pk[b]).astype(np.float64) for pk in p_layers], axis=0)

    # Ruido = lo que queda de cada capa tras quitar la tendencia suave a lo largo
    # del contorno. Es el ruido de periodo corto, que es el que la reconstruccion
    # deberia atenuar.
    res = []
    k = 9
    ker = np.ones(k) / k
    for sel in (nyb > 0, nyb <= 0):
        if sel.sum() < 3 * k:
            continue
        o = np.argsort(xb[sel])
        blk = P[:, sel][:, o]
        suave = np.stack([np.convolve(np.pad(r, k // 2, mode="edge"), ker, mode="valid")[:blk.shape[1]]
                          for r in blk], axis=0)
        res.append(blk - suave)
    R = np.concatenate(res, axis=1)
    C = np.cov(R)

    out = {"dx": dx, "alpha": a, "sigma_capa1": float(np.sqrt(C[0, 0])),
           "corr_capa1_capa2": float(C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])),
           "N": {}}
    for N in range(1, n_max + 1):
        w = np.array([1.0]) if N == 1 else pesos_intercepto(N)
        emp = float(w @ C[:N, :N] @ w) / float(C[0, 0])
        out["N"][str(N)] = {"factor_teorico": round(factor_teorico(N), 4),
                            "factor_empirico": round(emp, 4)}
    with open(os.path.join(OUT, f"varianza_dx{k_dx(dx)}.json"), "w") as f:
        json.dump(out, f, indent=1)
    log(f"varianza dx={dx:g}: teorico N=1/5/9 = "
        f"{factor_teorico(1):.2f}/{factor_teorico(5):.2f}/{factor_teorico(9):.2f}, "
        f"empirico = {out['N']['1']['factor_empirico']:.2f}/"
        f"{out['N']['5']['factor_empirico']:.2f}/{out['N']['9']['factor_empirico']:.2f}, "
        f"corr(capa1,capa2)={out['corr_capa1_capa2']:.3f}")
    return out


# ----------------------------------------------------------------------
# Parte C — GCI por N
# ----------------------------------------------------------------------
def gci_por_n(datos):
    h1, h2, h3 = TERNA
    out = {}
    for N in NS:
        filas = []
        for a in ANGULOS:
            try:
                f = [datos[f"dx{k_dx(h)}_a{a:.1f}"]["N"][str(N)] for h in (h1, h2, h3)]
            except KeyError:
                continue
            for mag in ("Cl_p", "Cd_p"):
                v = [x[mag] for x in f]
                r = vn.gci_triplete(v[0], v[1], v[2], h1, h2, h3)
                r.update({"alpha": a, "magnitud": mag})
                filas.append(r)
        util = [r for r in filas if r["convergencia_monotona"]
                and 1.0 <= r["p_observado"] <= 4.0]
        out[str(N)] = {
            "n_casos": len(filas), "n_utilizables": len(util),
            "gci_mediano_pct": round(float(np.median([r["GCI_fina_pct"] for r in util])), 3)
            if util else None,
            "casos": filas,
        }
    with open(os.path.join(OUT, "gci_por_n.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


# ----------------------------------------------------------------------
# Criterio de eleccion de N
#
# No hay verdad de referencia para p_w, asi que se usa una interna: el ajuste
# CUADRATICO sobre las mismas capas, que absorbe la curvatura de p(s) que el
# lineal no puede representar. Su Cd_p tiene meseta en N=7..9 (deja de moverse al
# anadir capas) y esa meseta se toma como referencia. El N lineal bueno es el que
# mas se le acerca. No prueba que la meseta sea el valor real, pero si mide donde
# el modelo lineal deja de ser adecuado, que es la pregunta que estaba abierta.
# ----------------------------------------------------------------------
def criterio_N(datos, dxs=(0.008, 0.004, 0.002)):
    out = {"referencia": "meseta del ajuste cuadratico, media de N=7,8,9",
           "por_N": {}, "N_optimo_por_caso": {}}
    for N in NS:
        e = []
        for dx in dxs:
            for a in ANGULOS:
                k = f"dx{k_dx(dx)}_a{a:.1f}"
                if k not in datos:
                    continue
                r = datos[k]["N"]
                plat = float(np.mean([r[str(n)]["Cd_p_cuad"] for n in (7, 8, 9)]))
                if abs(plat) < 1e-4:
                    continue
                e.append(100 * abs(r[str(N)]["Cd_p"] - plat) / abs(plat))
        out["por_N"][str(N)] = {"desvio_mediano_pct": round(float(np.median(e)), 3),
                                "desvio_p90_pct": round(float(np.percentile(e, 90)), 3),
                                "n_casos": len(e)}
    for dx in dxs:
        for a in ANGULOS:
            k = f"dx{k_dx(dx)}_a{a:.1f}"
            if k not in datos:
                continue
            r = datos[k]["N"]
            plat = float(np.mean([r[str(n)]["Cd_p_cuad"] for n in (7, 8, 9)]))
            errs = [abs(r[str(n)]["Cd_p"] - plat) for n in NS]
            out["N_optimo_por_caso"][k] = int(NS[int(np.argmin(errs))])
    vals = list(out["N_optimo_por_caso"].values())
    out["N_optimo_mediano"] = int(np.median(vals))
    with open(os.path.join(OUT, "criterio_N.json"), "w") as f:
        json.dump(out, f, indent=1)
    log(f"criterio: N optimo mediano = {out['N_optimo_mediano']}, "
        f"desvio N=5 = {out['por_N']['5']['desvio_mediano_pct']}% "
        f"frente a {out['por_N'][str(out['N_optimo_mediano'])]['desvio_mediano_pct']}%")
    return out


# ----------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------
def figuras(datos, gci):
    cols = {0.008: "#95a5a6", 0.006: "#7f8c8d", 0.004: "#2980b9", 0.002: "#c0392b"}
    Ns = np.array(NS, float)

    # 1) sensibilidad de Cl_p y Cd_p a N, referida a N=5, en alpha=4
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for mag, ax in zip(("Cl_p", "Cd_p"), axes):
        for dx in DXS:
            k = f"dx{k_dx(dx)}_a4.0"
            if k not in datos:
                continue
            v = np.array([datos[k]["N"][str(n)][mag] for n in NS], float)
            ax.plot(Ns, 100 * (v - v[4]) / abs(v[4]), "o-", color=cols[dx],
                    label=f"dx={dx:g}")
        ax.axvline(5, color="k", ls=":", lw=1)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("capas $N$"); ax.set_ylabel(f"$\\Delta${mag} respecto a $N=5$ [%]")
        ax.set_title(f"{mag} — NACA 0012, $\\alpha=4°$")
        ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "sensibilidad_N.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2) ruido: rugosidad y correlacion sub-celda
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for mag, ax, ttl in ((("rugosidad_cp"), axes[0], "rugosidad de $C_p$ (RMS $\\delta^2$/$\\sigma$)"),
                         (("corr_subcelda"), axes[1], "|corr| residuo $C_p$ vs posición sub-celda")):
        for dx in DXS:
            k = f"dx{k_dx(dx)}_a4.0"
            if k not in datos:
                continue
            v = np.array([datos[k]["N"][str(n)][mag] for n in NS], float)
            ax.plot(Ns, v, "o-", color=cols[dx], label=f"dx={dx:g}")
        ax.axvline(5, color="k", ls=":", lw=1)
        ax.set_xlabel("capas $N$"); ax.set_title(ttl); ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "ruido_N.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3) sesgo de curvatura: lineal vs cuadratico
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    for dx in DXS:
        k = f"dx{k_dx(dx)}_a4.0"
        if k not in datos:
            continue
        ns = [n for n in NS if n >= 3]
        d = [100 * (datos[k]["N"][str(n)]["Cl_p_cuad"] - datos[k]["N"][str(n)]["Cl_p"])
             / abs(datos[k]["N"][str(n)]["Cl_p"]) for n in ns]
        ax.plot(ns, d, "o-", color=cols[dx], label=f"dx={dx:g}")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(5, color="k", ls=":", lw=1)
    ax.set_xlabel("capas $N$"); ax.set_ylabel("$C_{l,p}$ cuadrático − lineal [%]")
    ax.set_title("Sesgo de curvatura del ajuste lineal — $\\alpha=4°$")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "sesgo_curvatura.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4) GCI por N
    if gci:
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        ns = [int(k) for k in sorted(gci, key=int)]
        ut = [gci[str(n)]["n_utilizables"] for n in ns]
        gm = [gci[str(n)]["gci_mediano_pct"] for n in ns]
        ax.bar(ns, ut, color="#2980b9", alpha=0.7, label="casos con $p$ utilizable")
        ax.set_xlabel("capas $N$"); ax.set_ylabel("casos utilizables (de 18)")
        ax2 = ax.twinx()
        ax2.plot(ns, gm, "o-", color="#c0392b", label="GCI mediano")
        ax2.set_ylabel("GCI mediano [%]")
        ax.axvline(5, color="k", ls=":", lw=1)
        ax.set_title("Convergencia de malla en función de $N$")
        fig.savefig(os.path.join(OUT, "gci_por_N.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 5) fraccion de estarcido que pisa solido
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    for dx in DXS:
        k = f"dx{k_dx(dx)}_a4.0"
        if k not in datos:
            continue
        t = datos[k]["toca_solido_por_capa"]
        ax.plot(np.arange(len(t)) + 0.5, 100 * np.array(t), "o-", color=cols[dx],
                label=f"dx={dx:g}")
    ax.set_xlabel("posición de la capa $s_k$ [celdas]")
    ax.set_ylabel("puntos con estarcido sobre sólido [%]")
    ax.set_title("Contaminación del muestreo por celdas sólidas")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "estarcido_solido.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_criterio(c):
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ns = [int(k) for k in sorted(c["por_N"], key=int)]
    ax.plot(ns, [c["por_N"][str(n)]["desvio_mediano_pct"] for n in ns], "o-",
            color="#c0392b", label="mediana")
    ax.plot(ns, [c["por_N"][str(n)]["desvio_p90_pct"] for n in ns], "s--",
            color="#e59866", label="percentil 90")
    ax.axvline(5, color="k", ls=":", lw=1)
    ax.set_xlabel("capas $N$ del ajuste lineal")
    ax.set_ylabel("desvío de $C_{d,p}$ vs meseta cuadrática [%]")
    ax.set_title("Ruido a $N$ bajo, sesgo de curvatura a $N$ alto")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "criterio_N.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_varianza(vs):
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ns = np.arange(1, 13)
    ax.plot(ns, [factor_teorico(n) for n in ns], "k--", label="teórico (ruido blanco)")
    for v, c in zip(vs, ("#2980b9", "#c0392b")):
        if v is None:
            continue
        ax.plot(ns, [v["N"][str(n)]["factor_empirico"] for n in ns], "o-", color=c,
                label=f"empírico dx={v['dx']:g}")
    ax.axhline(1.0, color="#7f8c8d", lw=1)
    ax.axvline(5, color="k", ls=":", lw=1)
    ax.set_xlabel("capas $N$")
    ax.set_ylabel("var($p_w$) / var(una capa)")
    ax.set_title("El ajuste extrapola: no promedia")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "varianza_intercepto.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-figuras", action="store_true")
    ap.add_argument("--solo-analisis", action="store_true")
    args = ap.parse_args()

    if not (args.solo_figuras or args.solo_analisis):
        validar_contra_produccion()
    datos = json.load(open(os.path.join(OUT, "barrido.json"))) \
        if (args.solo_figuras or args.solo_analisis) else barrido()
    g = gci_por_n(datos)
    figuras(datos, g)
    fig_criterio(criterio_N(datos))
    vs = [varianza_intercepto(dx=0.004), varianza_intercepto(dx=0.002)]
    fig_varianza(vs)
    log(f"listo -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()

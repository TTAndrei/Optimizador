r"""Validacion 1 del solver curvilineo: NACA 0012, alfa = 5, Re = 1e5, SA.

Malla de etapa 2: paso de pared 8.0e-5 y **15 capas sin crecimiento** pegadas al
cuerpo, para que el cizallamiento este resuelto con espaciado uniforme y no solo
la primera celda. El paso sale de exigir `y+ < 1` en **toda** la pared, morro
incluido; ver DN_PARED en `curvo.malla`.

Este script solo simula y vuelca. Las figuras y el video los hace `analizar.py`
a partir de lo volcado: dibujar dentro del lazo cuesta ~20 s por figura y ademas
bloquea la GPU.
"""

import json
import os
import sys
import time

import numpy as np
import cupy as cp

from curvo import malla as M, fuerzas as fz
from curvo.solver import Solver

AQUI = os.path.dirname(os.path.abspath(__file__))

# La malla ya no pasa `dn_pared`, `n_capas_pared` ni `crecimiento_pared`: desde la
# validacion 1 son los defectos de `curvo.malla`.
CASO = dict(perfil="NACA_0012_sharp", alfa=5.0, re=1.0e5, u_inf=1.0,
            distancia_lejos=None, x_salida=None,
            dt=2.5e-3, t_final=20.0, cada_historia=20, cada_campo=40)
BC = dict(oeste=None, este=None)


def figura_malla(X, Y, info, cal, caso):
    """Figura de la malla con sus parametros escritos dentro.

    Se hace en cada corrida sin pedirlo: una simulacion sin la imagen de su malla
    no se puede releer meses despues.
    """
    p = {"perfil": caso["perfil"],
         "caso": "alfa=%g  Re=%.0e  dt=%g  t*=%g" % (caso["alfa"], caso["re"],
                                                     caso["dt"], caso["t_final"]),
         "dn_pared": "%.2e c  (%d capas sin crecimiento, razon %.2f)"
                     % (info["dn_pared"], info["n_capas_pared"], info["crecimiento_pared"]),
         "crecimiento": "%.3f normal  |  dn_max %s  |  aspecto_max %g"
                        % (info["crecimiento"], info["dn_max"], info["aspecto_max"]),
         "superficie": "%d puntos de perfil + %d por lado de estela  |  x_salida %.2f c"
                       % (M.N_SUPERFICIE, M.N_ESTELA, info["x_out"]),
         "calidad": "ort_pared %.1f deg  |  oblic_p99 %.4f  |  aspecto_real %.0f  |  "
                    "crec_max %.3f  |  J<=0: %d"
                    % (cal["ortogonalidad_pared_min"], cal["oblicuidad_p99"],
                       cal["aspecto_max"], cal["crecimiento_max"], int(cal["j_negativos"]))}
    if cal["avisos"]:
        p["avisos"] = "; ".join(cal["avisos"])
    return M.dibujar(X, Y, os.path.join(AQUI, "figuras", "malla.png"), rangos=info,
                     titulo="Malla del caso", parametros=p)


def main(carpeta=None, **cambios):
    """Corre el caso. `cambios` sobreescribe `CASO`; `carpeta` elige el destino."""
    global AQUI
    if carpeta is not None:
        AQUI = os.path.abspath(carpeta)
    for sub in ("campos", "figuras", "video"):
        os.makedirs(os.path.join(AQUI, sub), exist_ok=True)
    c = dict(CASO, **cambios)
    nu = c["u_inf"] / c["re"]
    px, py = M.leer_dat(os.path.join("profiles", c["perfil"]))
    kw = {}
    if c["distancia_lejos"] is not None:
        kw["distancia_lejos"] = c["distancia_lejos"]
    if c["x_salida"] is not None:
        kw["x_out"] = c["x_salida"]
    X, Y, info = M.generar_c(px, py, **kw)
    # `perfil=`, no posicional: posicional cae en `limites` y entonces las metricas
    # de pared se miden sobre el corte de estela y ASPECTO_MAX pisa el umbral de aviso.
    cal = M.calidad(X, Y, perfil=info)
    np.savez_compressed(os.path.join(AQUI, "malla.npz"), X=X, Y=Y,
                        perfil=np.array(info["perfil"]))
    json.dump({k: (v if isinstance(v, (list, str)) else float(v))
               for k, v in cal.items()},
              open(os.path.join(AQUI, "calidad.json"), "w"), indent=1)
    print("malla %s  %d celdas  dominio %.2f x %.2f  ortogonalidad pared %.1f  "
          "oblicuidad p99 %.4f"
          % (X.shape, cal["n_celdas"], X.max() - X.min(), Y.max() - Y.min(),
             cal["ortogonalidad_pared_min"], cal["oblicuidad_p99"]), flush=True)
    figura_malla(X, Y, info, cal, c)

    s = Solver(X, Y, info, nu=nu, u_inf=c["u_inf"], alfa=c["alfa"], dt=c["dt"],
               xp=cp, dtype=np.float32, turbulento=True, cronometro=True)
    pasos = int(round(c["t_final"] / c["dt"]))
    hist, t0, marco = [], time.time(), 0
    t_prev, k_prev, ciclos = t0, 0, 0

    for k in range(pasos + 1):
        if k % c["cada_historia"] == 0:
            # Ritmo del bloque anterior. `estimadores_de_cl` baja datos a la CPU,
            # asi que la GPU ya esta sincronizada cuando se lee el reloj.
            ahora = time.time()
            it_s = (k - k_prev) / max(ahora - t_prev, 1e-9)
            t_prev, k_prev = ahora, k
            e = fz.estimadores_de_cl(s.met, s.u, s.v, s.p, info, nu,
                                     c["u_inf"], c["alfa"], bc=BC, corte=s.corte)
            f = e["fuerzas"]
            hist.append([k * c["dt"], e["superficie"], e["delta_cp"],
                         e["circulacion"], f["Cd"], f["Cd_presion"],
                         f["Cd_viscoso"], f["Cm"],
                         fz.delta_cp_te(s.met, s.p, info, c["u_inf"]),
                         e["gamma_dispersion"], float(s.nu_t.max()) / nu,
                         s.divergencia(), it_s, ahora - t0, ciclos]
                        + [e["gamma"][r] for r in sorted(e["gamma"])])
        if k % c["cada_campo"] == 0:
            np.savez_compressed(
                os.path.join(AQUI, "campos", "campo_%04d.npz" % marco),
                t=k * c["dt"], u=cp.asnumpy(s.u), v=cp.asnumpy(s.v),
                p=cp.asnumpy(s.p), nu_t=cp.asnumpy(s.nu_t))
            marco += 1
            if marco % 20 == 0:
                print("  t=%5.2f  Cl %.4f  Cd %.5f  nut/nu %5.1f  %.0fs"
                      % (k * c["dt"], hist[-1][1], hist[-1][4], hist[-1][10],
                         time.time() - t0), flush=True)
        if k < pasos:
            ciclos = s.paso()["ciclos"]

    cols = ("t Cl_sup Cl_dcp Cl_circ Cd Cd_p Cd_v Cm dCp_TE disp_gamma "
            "nut_nu div it_s segundos ciclos_poisson").split() + ["gamma_%.2f" % r for r in
                                     sorted(fz.estimadores_de_cl.__defaults__[-1])]
    np.savez_compressed(os.path.join(AQUI, "historia.npz"),
                        datos=np.array(hist), columnas=np.array(cols))
    c["dominio"] = [float(X.max() - X.min()), float(Y.max() - Y.min())]
    c["celdas"] = int(cal["n_celdas"])
    json.dump(c, open(os.path.join(AQUI, "caso.json"), "w"), indent=1)
    json.dump(s.reparto(), open(os.path.join(AQUI, "reparto.json"), "w"), indent=1)
    for n, d in s.reparto().items():
        print("  %-12s %5.1f %%  %6.2f ms/paso" % (n, d["%"], d["ms_por_paso"]))
    ritmo = np.array(hist)[1:, 12]
    print("%d marcos, %d filas de historia, %.0f s  |  %.2f it/s "
          "(mediana %.2f, min %.2f, max %.2f)"
          % (marco, len(hist), time.time() - t0, pasos / (time.time() - t0),
             np.median(ritmo), ritmo.min(), ritmo.max()))


if __name__ == "__main__":
    # correr.py [carpeta] [clave=valor ...]
    args = sys.argv[1:]
    destino = args.pop(0) if args and "=" not in args[0] else None
    cambios = {}
    for a in args:
        k, v = a.split("=", 1)
        cambios[k] = v if k == "perfil" else float(v)
    main(destino, **cambios)

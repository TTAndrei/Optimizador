r"""A/B del solver curvilineo: corre el caso de referencia y compara dos corridas.

El protocolo de las fases 1 y 2 estaba escrito (RESUMEN) pero se aplicaba a mano.
Esto es el pegamento: una corrida deja `historia.npz`, `reparto.json` y
`metricas.json`, y `--comparar` saca la tabla columna a columna con las puertas.

    PYTHONPATH=. .venv/bin/python scripts/ab_curvo.py --correr /tmp/base --pasos 800
    PYTHONPATH=. .venv/bin/python scripts/ab_curvo.py --comparar /tmp/base /tmp/variante

**Base y variante se miden seguidas, en la misma sesion**: el mismo codigo ha
dado 107.7-116.5 ms/paso entre sesiones distintas, asi que solo son comparables
los pares medidos a la vez. El ruido de una medida repetida es +-0.05 %.
"""

import argparse
import json
import os
import time

import numpy as np

from curvo import malla as M, fuerzas as fz, proyeccion as pr
from curvo.solver import Solver

CASO = dict(perfil="NACA_0012_sharp", alfa=5.0, re=1.0e5, u_inf=1.0,
            dt=2.5e-3, pasos=800, cada=20)
BC = dict(oeste=None, este=None)

COLUMNAS = ("t Cl_sup Cl_dcp Cl_circ Cd Cd_p Cd_v Cm dCp_TE disp_gamma nut_nu "
            "div it_s segundos ciclos_poisson").split()
RADIOS = (0.15, 0.3, 0.6, 1.0, 1.5)
# Reloj y diagnostico del solver: se miran aparte y no cruzan la puerta fisica.
# `ciclos_poisson` cambia de significado con la cuenta fija de ciclos (pasa a ser
# iteraciones en vez de comprobaciones), asi que compararla en % no dice nada.
DIAGNOSTICO = {"it_s", "segundos", "ciclos_poisson"}

PUERTAS = {"Cl": 0.05, "Cd": 0.05, "Cm": 0.05}      # % sobre la corrida base


def correr(destino, **cambios):
    import cupy as cp

    c = dict(CASO, **cambios)
    os.makedirs(destino, exist_ok=True)
    nu = c["u_inf"] / c["re"]
    px, py = M.leer_dat(os.path.join("profiles", c["perfil"]))
    X, Y, info = M.generar_c(px, py)
    s = Solver(X, Y, info, nu=nu, u_inf=c["u_inf"], alfa=c["alfa"], dt=c["dt"],
               xp=cp, dtype=np.float32, turbulento=True, cronometro=True)
    print("malla %s  %d celdas" % (X.shape, s.met.J.size), flush=True)

    hist, factores, ciclos_v = [], [], []
    t0 = t_prev = time.time()
    k_prev, ciclos = 0, 0
    for k in range(c["pasos"] + 1):
        if k % c["cada"] == 0:
            ahora = time.time()
            it_s = (k - k_prev) / max(ahora - t_prev, 1e-9)
            t_prev, k_prev = ahora, k
            e = fz.estimadores_de_cl(s.met, s.u, s.v, s.p, info, nu, c["u_inf"],
                                     c["alfa"], bc=BC, corte=s.corte,
                                     radios=RADIOS)
            f = e["fuerzas"]
            hist.append([k * c["dt"], e["superficie"], e["delta_cp"],
                         e["circulacion"], f["Cd"], f["Cd_presion"],
                         f["Cd_viscoso"], f["Cm"],
                         fz.delta_cp_te(s.met, s.p, info, c["u_inf"]),
                         e["gamma_dispersion"], float(s.nu_t.max()) / nu,
                         s.divergencia(), it_s, ahora - t0, ciclos]
                        + [e["gamma"][r] for r in RADIOS])
        if k < c["pasos"]:
            info_p = s.paso()
            ciclos = info_p["ciclos"]
            # `paso` devuelve la info de la ultima correccion de presion, que en
            # regimen asentado arranca ya convergida (0 ciclos, factor 0): esas
            # no dicen nada del precondicionador y estropean la mediana.
            if ciclos:
                ciclos_v.append(ciclos)
                factores.append(info_p["factor"])

    # Ritmo sin cronometro y sin historia: `_marca` sincroniza tres veces por
    # paso para poder repartir el tiempo, y esa sincronizacion **drena la
    # tuberia de lanzamientos**, que es justo lo que se esta optimizando. El
    # reparto sigue valiendo para ver donde se va el tiempo; el ms/paso que se
    # cita es este.
    s.cronometro = False
    cp.cuda.Stream.null.synchronize()
    t_puro = time.time()
    for _ in range(60):
        s.paso()
    cp.cuda.Stream.null.synchronize()
    ms_puro = 1e3 * (time.time() - t_puro) / 60

    datos = np.array(hist)
    cols = COLUMNAS + ["gamma_%.2f" % r for r in RADIOS]
    np.savez_compressed(os.path.join(destino, "historia.npz"),
                        datos=datos, columnas=np.array(cols))

    div = datos[1:, cols.index("div")]
    e = fz.estimadores_de_cl(s.met, s.u, s.v, s.p, info, nu, c["u_inf"],
                             c["alfa"], bc=BC, corte=s.corte, radios=RADIOS)
    u_cel, v_cel = s.u, s.v
    metricas = {
        "caso": c,
        "celdas": int(s.met.J.size),
        "ms_por_paso": ms_puro,
        "it_s": 1e3 / ms_puro,
        "ms_con_cronometro": 1e3 * (t_puro - t0) / c["pasos"],
        "Cl": e["superficie"], "Cl_circ": e["circulacion"],
        "Cl_dcp": e["delta_cp"],
        "Cd": e["fuerzas"]["Cd"], "Cd_p": e["fuerzas"]["Cd_presion"],
        "Cd_v": e["fuerzas"]["Cd_viscoso"], "Cm": e["fuerzas"]["Cm"],
        "dCp_TE": fz.delta_cp_te(s.met, s.p, info, c["u_inf"]),
        "gamma_dispersion": e["gamma_dispersion"],
        "div_mediana": float(np.median(div)), "div_max": float(div.max()),
        "tablero_p": pr.fraccion_par_impar(s.p),
        "tablero_u": pr.fraccion_par_impar(u_cel),
        "tablero_v": pr.fraccion_par_impar(v_cel),
        "factor_pcg": float(np.median(factores)) if factores else 0.0,
        "ciclos_pcg": float(np.median(ciclos_v)) if ciclos_v else 0.0,
        "reparto": s.reparto(),
    }
    json.dump(metricas, open(os.path.join(destino, "metricas.json"), "w"),
              indent=1)
    json.dump(s.reparto(), open(os.path.join(destino, "reparto.json"), "w"),
              indent=1)

    print("%.2f ms/paso  %.2f it/s  (con cronometro %.2f)"
          % (metricas["ms_por_paso"], metricas["it_s"],
             metricas["ms_con_cronometro"]))
    for n, d in s.reparto().items():
        print("  %-12s %5.1f %%  %6.2f ms/paso" % (n, d["%"], d["ms_por_paso"]))
    print("Cl %.9f  Cd %.9f  div_med %.3e  tablero_p %.4e  factor %.3f"
          % (metricas["Cl"], metricas["Cd"], metricas["div_mediana"],
             metricas["tablero_p"], metricas["factor_pcg"]))
    return metricas


def _rel(a, b):
    return 100.0 * (b - a) / a if abs(a) > 1e-30 else float("nan")


def comparar(base, variante):
    za, zb = [np.load(os.path.join(d, "historia.npz")) for d in (base, variante)]
    ma, mb = [json.load(open(os.path.join(d, "metricas.json")))
              for d in (base, variante)]
    cols = list(za["columnas"])
    da, db = za["datos"], zb["datos"]
    n = min(len(da), len(db))

    print("\n%-16s %16s %16s %10s" % ("columna", "base", "variante", "peor Δ%"))
    peor_fisica = 0.0
    for j, c in enumerate(cols):
        d = np.array([_rel(x, y) for x, y in zip(da[1:n, j], db[1:n, j])])
        d = d[np.isfinite(d)]
        p = float(np.abs(d).max()) if d.size else 0.0
        marca = ""
        if c not in DIAGNOSTICO and c != "t":
            peor_fisica = max(peor_fisica, p)
        else:
            marca = "  (diagnostico)"
        print("%-16s %16.9g %16.9g %9.4f%%%s" % (c, da[n-1, j], db[n-1, j], p, marca))

    print("\n%-18s %16s %16s %10s" % ("metrica", "base", "variante", "Δ%"))
    for k in ("Cl", "Cl_circ", "Cl_dcp", "Cd", "Cd_p", "Cd_v", "Cm", "dCp_TE",
              "div_mediana", "div_max", "tablero_p", "tablero_u", "tablero_v",
              "factor_pcg", "ciclos_pcg", "ms_por_paso", "it_s"):
        d = _rel(ma[k], mb[k])
        puerta = PUERTAS.get(k)
        aviso = "  FUERA" if puerta is not None and abs(d) > puerta else ""
        print("%-18s %16.9g %16.9g %9.4f%%%s" % (k, ma[k], mb[k], d, aviso))

    print("\netapas (ms/paso)")
    for etapa in sorted(set(ma["reparto"]) | set(mb["reparto"])):
        a = ma["reparto"][etapa]["ms_por_paso"]
        b = mb["reparto"][etapa]["ms_por_paso"]
        print("  %-12s %7.2f -> %7.2f   x%.3f" % (etapa, a, b, a / b))

    print("\npeor desviacion de una columna fisica: %.4f %%  (puerta 0.05 %%)"
          % peor_fisica)
    print("tiempo: %.2f -> %.2f ms/paso  (x%.3f)"
          % (ma["ms_por_paso"], mb["ms_por_paso"],
             ma["ms_por_paso"] / mb["ms_por_paso"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--correr", metavar="DESTINO")
    p.add_argument("--comparar", nargs=2, metavar=("BASE", "VARIANTE"))
    p.add_argument("--pasos", type=int, default=CASO["pasos"])
    p.add_argument("cambios", nargs="*", metavar="clave=valor")
    a = p.parse_args()
    if a.comparar:
        comparar(*a.comparar)
        return
    cambios = dict(pasos=a.pasos)
    for c in a.cambios:
        k, v = c.split("=")
        cambios[k] = type(CASO[k])(v)
    correr(a.correr, **cambios)


if __name__ == "__main__":
    main()

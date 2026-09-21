"""Campana reanudable de polares y capa limite del solver curvo.

Ejemplo:
    .venv/bin/python scripts/campana_polares_curvo.py
    .venv/bin/python scripts/campana_polares_curvo.py --backend numpy --pasos 100

Cada perfil y angulo tiene su propia carpeta. El estado se escribe despues de
cada angulo y la historia se actualiza durante la corrida, de modo que se puede
interrumpir con Ctrl-C y continuar ejecutando el mismo comando.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from curvo import fuerzas, malla
from curvo.solver import Solver


PERFILES = ("NACA_0012_sharp")
ESTACIONES = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.95)


def argumentos():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path, default=RAIZ / "resultados_polares_curvo")
    parser.add_argument("--backend", choices=("cupy", "numpy"), default="cupy")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--pasos", type=int, default=1000,
                        help="pasos por angulo; por defecto usa el valor de Caso")
    parser.add_argument("--cada", type=int, default=20,
                        help="frecuencia de guardado de historia, campos y metricas")
    parser.add_argument("--alfa-min", type=float, default=0.0)
    parser.add_argument("--alfa-max", type=float, default=15.0)
    parser.add_argument("--alfa-paso", type=float, default=1.0)
    parser.add_argument("--repetir", action="store_true",
                        help="recalcula angulos que ya tienen estado completado")
    return parser.parse_args()


def main():
    args = argumentos()
    if args.pasos <= 0 or args.cada <= 0 or args.alfa_paso <= 0:
        raise SystemExit("pasos, cada y alfa-paso deben ser positivos")
    salida = args.salida if args.salida.is_absolute() else RAIZ / args.salida
    salida.mkdir(parents=True, exist_ok=True)
    angulos = np.arange(args.alfa_min, args.alfa_max + 0.5 * args.alfa_paso,
                        args.alfa_paso)
    estado = cargar_estado(salida, args, angulos)
    try:
        for perfil in PERFILES:
            for alfa in angulos:
                clave = clave_corrida(perfil, alfa)
                carpeta = salida / perfil / clave
                if estado["corridas"].get(clave, {}).get("estado") == "completada" \
                        and not args.repetir:
                    print(f"[skip] {clave}")
                    continue
                completada = ejecutar_corrida(perfil, float(alfa), carpeta, args,
                                              estado, salida)
                estado["corridas"][clave] = {
                    "perfil": perfil, "alfa": float(alfa),
                    "estado": "completada" if completada else "fallida_malla",
                    "carpeta": str(carpeta)}
                guardar_estado(salida, estado)
                generar_graficas(salida, perfil)
    except KeyboardInterrupt:
        guardar_estado(salida, estado)
        print("\n[pausa] estado guardado; vuelve a ejecutar el mismo comando para continuar")
        return
    for perfil in PERFILES:
        generar_graficas(salida, perfil)
    guardar_estado(salida, estado)
    print(f"[fin] resultados en {salida}")


def cargar_estado(salida, args, angulos):
    ruta = salida / "estado.json"
    if ruta.is_file():
        estado = json.loads(ruta.read_text(encoding="utf-8"))
    else:
        estado = {"version": 1, "perfiles": list(PERFILES), "angulos": angulos.tolist(),
                  "parametros": {"backend": args.backend, "dtype": args.dtype,
                                 "pasos": args.pasos, "cada": args.cada},
                  "corridas": {}}
    return estado


def guardar_estado(salida, estado):
    temporal = salida / "estado.json.tmp"
    temporal.write_text(json.dumps(estado, indent=2), encoding="utf-8")
    temporal.replace(salida / "estado.json")


def clave_corrida(perfil, alfa):
    return f"{perfil}_a{alfa:+07.2f}".replace("+", "p").replace("-", "m").replace(".", "d")


def ejecutar_corrida(perfil, alfa, carpeta, args, estado, raiz_salida):
    carpeta.mkdir(parents=True, exist_ok=True)
    estado["corridas"][clave_corrida(perfil, alfa)] = {
        "perfil": perfil, "alfa": alfa, "estado": "en_curso", "carpeta": str(carpeta)}
    guardar_estado(raiz_salida, estado)
    print(f"[run] {perfil} alpha={alfa:g}")

    px, py = malla.leer_dat(RAIZ / "profiles" / perfil)
    X, Y, info = malla.generar_c(px, py)
    calidad = malla.calidad(X, Y, perfil=info)
    np.savez_compressed(carpeta / "malla.npz", X=X, Y=Y,
                        perfil=np.asarray(info["perfil"], dtype=int))
    escribir_json(carpeta / "calidad.json", calidad)
    if not calidad["valida"]:
        escribir_json(carpeta / "resumen.json", {
            "perfil": perfil, "alfa": alfa, "estado": "fallida_malla",
            "celdas": int((X.shape[0] - 1) * (X.shape[1] - 1)),
            "calidad": calidad,
            "motivo": "La malla por defecto no es valida; no se ejecuta el solver.",
        })
        print(f"  [skip] malla invalida: {', '.join(calidad['fallos'])}")
        return False

    xp = np
    if args.backend == "cupy":
        import cupy as cp
        xp = cp
    solver = Solver(X, Y, info, xp=xp, dtype=np.dtype(args.dtype), alfa=alfa)
    pasos = args.pasos
    historia = []
    capas = []
    ultima_info = {}
    inicio = time.perf_counter()
    bloque_t = inicio
    bloque_n = 0
    try:
        for paso in range(pasos + 1):
            ahora = time.perf_counter()
            bloque_n += 1
            if paso == 0 or paso % args.cada == 0 or paso == pasos:
                it_s = 0.0 if paso == 0 else bloque_n / max(ahora - bloque_t, 1e-12)
                metricas = medir(solver, info, alfa, it_s, ultima_info)
                metricas["iter"] = paso
                metricas["t"] = paso * solver.dt
                historia.append(metricas)
                guardar_historia(carpeta, historia)
                guardar_campo(carpeta, solver, paso, solver.dt)
                capas.append(medir_capa_limite(X, Y, solver, info, alfa))
                guardar_capa(carpeta, capas)
                print("  paso %d/%d: Cl=%+.5f Cd=%.5f %.2f it/s div=%.3e" %
                      (paso, pasos, metricas["Cl"], metricas["Cd"],
                       metricas["it_s"], metricas["div"]))
                bloque_t, bloque_n = ahora, 0
            if paso == pasos:
                break
            ultima_info = solver.paso()
    except KeyboardInterrupt:
        guardar_historia(carpeta, historia)
        guardar_capa(carpeta, capas)
        guardar_json_en_curso(carpeta, perfil, alfa, solver, historia)
        raise

    final = historia[-1]
    pared = medir_pared(X, Y, solver, info, alfa)
    np.savez_compressed(carpeta / "pared.npz", **pared)
    escribir_json(carpeta / "resumen.json", {
        "perfil": perfil, "alfa": alfa, "pasos": pasos,
        "estado": "completada", "celdas": int((X.shape[0] - 1) * (X.shape[1] - 1)),
        "metricas_finales": final,
        "calidad": calidad,
        "pared": {"cf_min": float(np.min(pared["cf"])),
                  "fraccion_cf_negativo": float(np.mean(pared["cf"] < 0)),
                  "cp_min": float(np.min(pared["cp"]))},
    })
    return True


def medir(solver, info, alfa, it_s, poisson):
    e = fuerzas.estimadores_de_cl(solver.met, solver.u, solver.v, solver.p, info,
                                  solver.nu, solver.u_inf, alfa, corte=solver.corte)
    f = e["fuerzas"]
    campos = solver.campos()
    nut_nu = 0.0 if solver.nu_t is None else float(np.max(
        np.asarray(solver.nu_t.get() if hasattr(solver.nu_t, "get") else solver.nu_t)) / solver.nu)
    return {"Cl": float(f["Cl"]), "Cd": float(f["Cd"]), "Cm": float(f["Cm"]),
            "Cl_superficie": float(e["superficie"]),
            "Cl_circulacion": float(e["circulacion"]),
            "Cl_delta_cp": float(e["delta_cp"]),
            "gamma_dispersion": float(e["gamma_dispersion"]),
            "dCp_TE": float(fuerzas.delta_cp_te(solver.met, solver.p, info, solver.u_inf)),
            "div": float(solver.divergencia()), "it_s": float(it_s),
            "speed_max": float(np.hypot(campos["u"], campos["v"]).max()),
            "nu_t_nu_max": nut_nu,
            "poisson_ciclos": int(poisson.get("ciclos", 0)) if isinstance(poisson, dict) else 0}


def medir_pared(X, Y, solver, info, alfa):
    campos = solver.campos()
    g = fuerzas.geometria_pared(solver.met, info)
    cx, cy = centroides(X, Y)
    i = np.arange(g["x"].size) + int(info["perfil"][0])
    dx, dy = cx[0, i] - g["x"], cy[0, i] - g["y"]
    d = dx * g["nx"] + dy * g["ny"]
    tx, ty = tangente(g, alfa)
    ut = campos["u"][0, i] * tx + campos["v"][0, i] * ty
    cp = fuerzas.coeficiente_de_presion(solver.met, solver.p, info, solver.u_inf)
    cf = 2.0 * solver.nu * ut / (d * solver.u_inf ** 2)
    return {"x_c": (g["x"] - g["x"].min()) / g["cuerda"], "y": g["y"],
            "cp": cp, "cf": cf, "d": d}


def medir_capa_limite(X, Y, solver, info, alfa):
    campos = solver.campos()
    g = fuerzas.geometria_pared(solver.met, info)
    cx, cy = centroides(X, Y)
    tx, ty = tangente(g, alfa)
    resultado = {"alfa": alfa, "estaciones": {}}
    for estacion in ESTACIONES:
        xc = (g["x"] - g["x"].min()) / g["cuerda"]
        lado = np.where(g["y"] > 0, 1, -1)
        valores = {}
        for nombre, signo in (("extrados", 1), ("intrados", -1)):
            candidatos = np.where(lado == signo)[0]
            i = candidatos[np.argmin(np.abs(xc[candidatos] - estacion))]
            col = int(info["perfil"][0]) + int(i)
            n = min(40, campos["u"].shape[0])
            dx = cx[:n, col] - g["x"][i]
            dy = cy[:n, col] - g["y"][i]
            d = dx * g["nx"][i] + dy * g["ny"][i]
            ut = campos["u"][:n, col] * tx[i] + campos["v"][:n, col] * ty[i]
            tau = solver.nu * ut[0] / d[0]
            u_tau = np.sqrt(abs(tau))
            valores[nombre] = {"d": d.tolist(), "u_uinf": (ut / solver.u_inf).tolist(),
                               "y_mas": (d * u_tau / solver.nu).tolist(),
                               "u_tau": float(u_tau), "cf": float(2 * tau / solver.u_inf ** 2)}
        resultado["estaciones"][str(estacion)] = valores
    return resultado


def centroides(X, Y):
    return (0.25 * (X[:-1, :-1] + X[:-1, 1:] + X[1:, :-1] + X[1:, 1:]),
            0.25 * (Y[:-1, :-1] + Y[:-1, 1:] + Y[1:, :-1] + Y[1:, 1:]))


def tangente(g, alfa):
    tx, ty = -g["ny"], g["nx"]
    a = np.radians(alfa)
    signo = np.where(tx * np.cos(a) + ty * np.sin(a) < 0.0, -1.0, 1.0)
    return tx * signo, ty * signo


def guardar_historia(carpeta, historia):
    nombres = ("iter", "t", "Cl", "Cd", "Cm", "Cl_superficie", "Cl_circulacion",
               "Cl_delta_cp", "gamma_dispersion", "dCp_TE", "div", "it_s",
               "speed_max", "nu_t_nu_max", "poisson_ciclos")
    datos = np.asarray([[fila.get(nombre, np.nan) for nombre in nombres] for fila in historia])
    np.savez_compressed(carpeta / "historia.npz", datos=datos, columnas=np.asarray(nombres))


def guardar_capa(carpeta, capas):
    (carpeta / "capa_limite.json").write_text(json.dumps(capas), encoding="utf-8")


def guardar_campo(carpeta, solver, paso, dt):
    campos = solver.campos()
    np.savez_compressed(carpeta / f"campo_{paso:06d}.npz", t=paso * dt, **campos)


def guardar_json_en_curso(carpeta, perfil, alfa, solver, historia):
    escribir_json(carpeta / "resumen_en_curso.json", {
        "perfil": perfil, "alfa": alfa, "estado": "interrumpida",
        "paso": int(solver.paso_n), "muestras": len(historia)})


def escribir_json(ruta, datos):
    ruta.write_text(json.dumps(_jsonable(datos), indent=2), encoding="utf-8")


def _jsonable(valor):
    if isinstance(valor, dict):
        return {str(k): _jsonable(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_jsonable(v) for v in valor]
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    if isinstance(valor, np.generic):
        return valor.item()
    if isinstance(valor, float) and not np.isfinite(valor):
        return None
    return valor


def datos_completados(salida, perfil):
    filas = []
    for resumen in sorted((salida / perfil).glob("*/resumen.json")):
        datos = json.loads(resumen.read_text(encoding="utf-8"))
        if datos.get("estado") != "completada":
            continue
        fila = dict(datos["metricas_finales"])
        fila.update({"alfa": datos["alfa"], "perfil": perfil,
                     "celdas": datos["celdas"],
                     "cf_min": datos["pared"]["cf_min"],
                     "fraccion_cf_negativo": datos["pared"]["fraccion_cf_negativo"]})
        filas.append(fila)
    return sorted(filas, key=lambda fila: fila["alfa"])


def generar_graficas(salida, perfil):
    filas = datos_completados(salida, perfil)
    if not filas:
        return
    carpeta = salida / perfil / "graficas"
    carpeta.mkdir(parents=True, exist_ok=True)
    a = np.array([f["alfa"] for f in filas])
    cl = np.array([f["Cl"] for f in filas])
    cd = np.array([f["Cd"] for f in filas])
    it_s = np.array([f["it_s"] for f in filas])
    div = np.array([f["div"] for f in filas])
    ld = cl / np.maximum(cd, 1e-12)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.plot(cd, cl, "o-"); ax.set(xlabel="Cd", ylabel="Cl", title=f"Polar {perfil}"); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(carpeta / "polar.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for eje, valores, titulo in zip(axes.flat, (cl, cd, ld, it_s, div, np.array([f["fraccion_cf_negativo"] for f in filas])),
                                    ("Cl vs alpha", "Cd vs alpha", "L/D vs alpha", "Rendimiento [it/s]", "Divergencia", "Fraccion Cf < 0")):
        eje.plot(a, valores, "o-"); eje.set_title(titulo); eje.set_xlabel("alpha [deg]"); eje.grid(alpha=.3)
        if "Divergencia" in titulo: eje.set_yscale("log")
    fig.suptitle(perfil); fig.tight_layout(); fig.savefig(carpeta / "resumen_alpha.png", dpi=150); plt.close(fig)
    generar_graficas_pared(salida, perfil, carpeta)
    generar_graficas_capa(salida, perfil, carpeta)


def generar_graficas_pared(salida, perfil, carpeta):
    filas = datos_completados(salida, perfil)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for fila in filas:
        ruta = salida / perfil / clave_corrida(perfil, fila["alfa"]) / "pared.npz"
        if not ruta.is_file():
            continue
        with np.load(ruta) as pared:
            axes[0].plot(pared["x_c"], pared["cp"], label=f"{fila['alfa']:g} deg")
            axes[1].plot(pared["x_c"], pared["cf"], label=f"{fila['alfa']:g} deg")
    axes[0].set_title("Cp sobre la pared"); axes[0].set_ylabel("Cp")
    axes[1].set_title("Cf sobre la pared"); axes[1].set_ylabel("Cf")
    for eje in axes: eje.set_xlabel("x/c desde borde de salida"); eje.grid(alpha=.3)
    axes[0].legend(ncol=2, fontsize=7); fig.tight_layout(); fig.savefig(carpeta / "pared_cp_cf.png", dpi=150); plt.close(fig)


def generar_graficas_capa(salida, perfil, carpeta):
    rutas = sorted((salida / perfil).glob("*/capa_limite.json"))
    if not rutas:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ruta in rutas:
        muestras = json.loads(ruta.read_text(encoding="utf-8"))
        if not muestras:
            continue
        datos = muestras[-1]
        alfa = datos["alfa"]
        for eje, estacion in zip(axes.flat, ("0.05", "0.2", "0.6", "0.95")):
            if estacion not in datos["estaciones"]:
                continue
            perfil_bl = datos["estaciones"][estacion]["extrados"]
            eje.plot(perfil_bl["y_mas"], perfil_bl["u_uinf"], label=f"{alfa:g} deg")
            eje.set_title(f"Capa limite extrados x/c={estacion}")
            eje.set_xlabel("y+"); eje.set_ylabel("u/Uinf"); eje.grid(alpha=.3)
    axes[0, 0].legend(ncol=2, fontsize=7); fig.tight_layout(); fig.savefig(carpeta / "capa_limite_alpha.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()

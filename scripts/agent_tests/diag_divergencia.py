"""
Donde vive la divergencia que queda despues de proyectar.

La instrumentacion del harness dice que la proyeccion solo reduce la divergencia
media x1.76 por paso, pese a que el multigrid baja el residuo del Poisson x23 en
el primer outer. Hay dos explicaciones incompatibles y este script las separa:

  A) La metrica engana. La proyeccion no corrige, por diseno, ni la interfaz
     solido-fluido (el ghost-cell del IBM la reescribe despues) ni las capas
     adyacentes a las salidas (su RHS se pone a cero a proposito). Si casi toda
     la divergencia residual vive ahi, el solver hace su trabajo y el x1.76 es un
     artefacto de promediar sobre celdas que nadie va a corregir.

  B) El solver falla. Si la divergencia esta repartida por el interior del
     dominio, lejos del perfil y de las salidas, entonces resolver bien el
     Poisson no esta quitando la divergencia, y eso apunta a que el laplaciano
     que se invierte no es el producto de la divergencia y el gradiente
     discretos que se usan de verdad.

La respuesta decide donde merece la pena invertir: en el precondicionador
(suavizador, mascara gruesa) solo si es B, y en nada de eso si es A.

Ademas comprueba directamente esa compatibilidad de operadores: aplica el
laplaciano del solver a un campo aleatorio y lo compara con divergencia(gradiente)
del mismo campo, usando los mismos operadores discretos que usa la proyeccion. Si
no coinciden, la proyeccion no puede anular la divergencia por bien que resuelva
el Poisson, y ese es el techo real del solver.

Uso:
    .venv/bin/python scripts/agent_tests/diag_divergencia.py [--iters 400]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "agent_tests"))
os.chdir(ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cupy as cp

import Simulador2D
import bench_opt as B


def _np(x):
    return cp.asnumpy(x) if isinstance(x, cp.ndarray) else np.asarray(x)


def bandas_desde_solido(solid, n_max=12):
    """Distancia en celdas al solido mas cercano, por dilatacion sucesiva.
    Suficiente para clasificar en bandas; no hace falta una EDT exacta."""
    d = np.full(solid.shape, n_max + 1, dtype=np.int16)
    d[solid] = 0
    frente = solid.copy()
    for k in range(1, n_max + 1):
        vecino = np.zeros_like(frente)
        vecino[1:, :] |= frente[:-1, :]
        vecino[:-1, :] |= frente[1:, :]
        vecino[:, 1:] |= frente[:, :-1]
        vecino[:, :-1] |= frente[:, 1:]
        nuevo = vecino & (d > n_max)
        d[nuevo] = k
        frente = frente | nuevo
    return d


def compatibilidad_operadores(mesh):
    """||L·p - D(G(p))|| / ||L·p|| con los operadores reales del solver.

    L        laplaciano de 5 puntos que invierte el multigrid
    G        gradiente que aplica la correccion de velocidad
    D        divergencia que construye el termino independiente

    Una proyeccion exacta necesita L = D·G. Cuanto se aparte de eso es la
    fraccion de divergencia que la proyeccion no puede eliminar nunca.
    """
    ny, nx = mesh.p.shape
    # El campo de prueba importa mucho y conviene decir por que. Con ruido blanco
    # el resultado no significa lo que parece: la composicion de dos derivadas
    # centradas de 3 puntos abarca de i-2 a i+2 y tiene nucleo nulo en el modo
    # tablero, asi que sobre un campo con todo el contenido en Nyquist el error
    # sale enorme por construccion. El campo de presion real es suave, asi que la
    # medida honesta es sobre un campo suave; el de ruido se deja al lado como
    # cota superior y para ver cuanto pesa el desacoplamiento par-impar.
    xs = cp.asarray(mesh.X_1d, dtype=cp.float32)[None, :]
    ys = cp.asarray(mesh.Y_1d, dtype=cp.float32)[:, None]
    Lx = float(mesh.Lx); Ly = float(mesh.Ly)
    campos = {}
    suave = (cp.sin(2 * np.pi * xs / Lx) * cp.cos(2 * np.pi * ys / Ly)
             + 0.5 * cp.sin(6 * np.pi * xs / Lx) * cp.sin(4 * np.pi * ys / Ly))
    campos["suave"] = suave.astype(cp.float32)
    rng = cp.random.RandomState(0)
    campos["ruido"] = (rng.rand(ny, nx, dtype=cp.float32) - 0.5)

    # Lp se calcula dentro del bucle, una vez por campo de prueba.

    # D(G(p)): se aplica el gradiente sobre un campo de velocidad virtual y se
    # mide su divergencia con el MISMO operador que arma el RHS del Poisson.
    u0, v0 = mesh.u.copy(), mesh.v.copy()
    salida = {}
    interior = (~mesh.solid).copy()
    interior[0, :] = False; interior[-1, :] = False
    interior[:, 0] = False; interior[:, -1] = False
    try:
        coef = cp.float32(-1.0)          # u = +grad(p), el kernel resta
        solid_flat = mesh._mg_solids_flat[0]
        nx_i, ny_i = cp.int32(nx), cp.int32(ny)
        blk = 256
        grid = ((nx * ny + blk - 1) // blk,)
        for tipo, p in campos.items():
            p = p.copy()
            p[mesh.solid] = 0.0
            p[0, :] = 0; p[-1, :] = 0; p[:, 0] = 0; p[:, -1] = 0
            kd = mesh._mg_kdims[0]
            Lp = mesh._laplacian_kernel_masked(
                mesh._mg_solids_flat[0], p.ravel(),
                mesh._mg_d2x_W[0], mesh._mg_d2x_C[0], mesh._mg_d2x_E[0],
                mesh._mg_d2y_S[0], mesh._mg_d2y_C[0], mesh._mg_d2y_N[0],
                kd['nx_i'], kd['ny_i'], size=kd['total']).reshape(ny, nx)
            for nombre, kern, extra in (
                ("centered", mesh._velocity_correction_kernel, ()),
                ("one_sided", mesh._velocity_correction_one_sided_kernel,
                 (mesh.X_1d, mesh.Y_1d)),
            ):
                mesh.u.fill(0.0); mesh.v.fill(0.0)
                kern(grid, (blk,),
                     (mesh.u.ravel(), mesh.v.ravel(), p.ravel(), solid_flat, coef,
                      mesh.d1x_W, mesh.d1x_C, mesh.d1x_E,
                      mesh.d1y_S, mesh.d1y_C, mesh.d1y_N, *extra, nx_i, ny_i))
                for dnombre, dfun in (
                    ("masked", mesh._compute_divergence_field),
                    ("face_flux", mesh._compute_flux_divergence_field_uv),
                ):
                    DG = dfun().copy()
                    a, b = Lp[interior], DG[interior]
                    # mejor escala que minimiza ||a - alpha*b||: los dos operadores
                    # pueden diferir en un factor global de signo y escala
                    den = float(cp.sum(b * b))
                    alpha = float(cp.sum(a * b)) / den if den > 0 else 0.0
                    err = float(cp.linalg.norm(a - alpha * b) /
                                max(float(cp.linalg.norm(a)), 1e-30))
                    salida[f"{tipo}/{nombre}+{dnombre}"] = {
                        "error_relativo": round(float(err), 5),
                        "escala": round(float(alpha), 5)}
    finally:
        mesh.u[:] = u0
        mesh.v[:] = v0
    return salida


def contenido_tablero(div, solid):
    """Cuanto de la divergencia residual es modo tablero (par-impar).

    Es la firma del desacoplamiento de malla colocada: con u, v y p en el mismo
    punto, el gradiente y la divergencia centrados de 3 puntos no ven el modo de
    frecuencia Nyquist, asi que la proyeccion no puede eliminarlo por muchas
    iteraciones que se le den. Si la divergencia que queda es sobre todo tablero,
    el suelo no lo fija el solver iterativo y gastar mas ciclos no sirve de nada.

    Se mide promediando en bloques 2x2: ese promedio anula exactamente el modo
    tablero, asi que lo que sobrevive es la parte suave.
    """
    d = np.where(solid, 0.0, div)
    ny, nx = d.shape
    dy, dx = ny - ny % 2, nx - nx % 2
    bloque = d[:dy, :dx].reshape(dy // 2, 2, dx // 2, 2).mean(axis=(1, 3))
    suave = np.repeat(np.repeat(bloque, 2, axis=0), 2, axis=1)
    resto = d[:dy, :dx] - suave
    n_suave = float(np.linalg.norm(suave))
    n_resto = float(np.linalg.norm(resto))
    total = np.hypot(n_suave, n_resto)
    return {
        "fraccion_tablero": round(float(n_resto / max(total, 1e-30)), 4),
        "fraccion_suave": round(float(n_suave / max(total, 1e-30)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    a = ap.parse_args()

    params = {**B.BASE, "filepath": B.PERFIL, "iteraciones": a.iters}
    prev = os.getcwd()
    os.chdir(B.SANDBOX)
    try:
        mesh = Simulador2D.main(**params)
    finally:
        os.chdir(prev)

    solid = _np(mesh.solid).astype(bool)
    ny, nx = solid.shape
    div = np.abs(_np(mesh._compute_flux_divergence_field_uv()))
    dist = bandas_desde_solido(solid)

    total = div[~solid].sum()
    def frac(m):
        return float(100.0 * div[m & ~solid].sum() / max(total, 1e-30))
    def cuenta(m):
        return float(100.0 * (m & ~solid).sum() / max((~solid).sum(), 1))

    borde = np.zeros_like(solid)
    borde[:2, :] = True; borde[-2:, :] = True
    borde[:, :2] = True; borde[:, -2:] = True

    interfaz = (dist >= 1) & (dist <= 3) & ~borde
    cerca = (dist >= 4) & (dist <= 12) & ~borde
    lejos = (dist > 12) & ~borde

    print("\n" + "=" * 66)
    print(f"REPARTO DE LA DIVERGENCIA RESIDUAL   malla {nx}x{ny}, {a.iters} iters")
    print("=" * 66)
    print(f"{'zona':<34}{'% de |div|':>12}{'% de celdas':>14}")
    print("-" * 66)
    for nombre, m in (("borde del dominio (2 capas)", borde),
                      ("interfaz del perfil (1-3 celdas)", interfaz),
                      ("cerca del perfil (4-12 celdas)", cerca),
                      ("interior lejano (>12 celdas)", lejos)):
        print(f"{nombre:<34}{frac(m):>11.2f}%{cuenta(m):>13.2f}%")

    conc = frac(borde) + frac(interfaz)
    print("-" * 66)
    print(f"{'borde + interfaz':<34}{conc:>11.2f}%"
          f"{cuenta(borde)+cuenta(interfaz):>13.2f}%")
    print("\nLectura: si 'borde + interfaz' concentra la mayor parte de |div| en una")
    print("fraccion pequena de celdas, la proyeccion esta haciendo su trabajo y el")
    print("x1.76 mide celdas que nadie corrige (caso A). Si el interior lejano pesa")
    print("mucho, el solver no esta anulando la divergencia (caso B).")

    print("\n" + "=" * 66)
    print("COMPATIBILIDAD DE OPERADORES:  ||L·p - a·D(G(p))|| / ||L·p||")
    print("=" * 66)
    compat = compatibilidad_operadores(mesh)
    for k, v in sorted(compat.items(), key=lambda kv: kv[1]["error_relativo"]):
        print(f"  {k:<26} error = {v['error_relativo']:>8.4f}   escala = {v['escala']:>9.4f}")
    print("\nLa combinacion que usa produccion es one_sided+face_flux")
    print("(wall_treatment='consistent'). Lo que cuenta es la fila 'suave/': el")
    print("campo de presion real no tiene contenido en Nyquist. La fila 'ruido/'")
    print("es la cota superior y mide cuanto pesa el desacoplamiento par-impar.")

    tab = contenido_tablero(div, solid)
    print("\n" + "=" * 66)
    print("ESTRUCTURA DE LA DIVERGENCIA RESIDUAL")
    print("=" * 66)
    print(f"  parte tablero (par-impar) : {100*tab['fraccion_tablero']:.1f}%")
    print(f"  parte suave               : {100*tab['fraccion_suave']:.1f}%")
    print("\nSi domina la parte tablero, el suelo de divergencia no lo fija el")
    print("solver iterativo sino la discretizacion colocada, y gastar mas ciclos")
    print("de multigrid no puede bajarlo.")

    salida = {
        "malla": [int(nx), int(ny)], "iters": a.iters,
        "reparto": {
            "borde": [frac(borde), cuenta(borde)],
            "interfaz_1_3": [frac(interfaz), cuenta(interfaz)],
            "cerca_4_12": [frac(cerca), cuenta(cerca)],
            "lejos": [frac(lejos), cuenta(lejos)],
        },
        "compatibilidad_operadores": compat,
        "estructura_divergencia": tab,
    }
    dst = os.path.join(B.OUT, "diag_divergencia.json")
    with open(dst, "w") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    # Mapa: donde esta la divergencia, en escala log
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    campo = np.log10(np.maximum(div, 1e-12))
    campo[solid] = np.nan
    x = _np(mesh.X_1d); y = _np(mesh.Y_1d)
    im = ax.pcolormesh(x, y, campo, cmap="inferno", shading="auto")
    ax.set_title("log10 |div(u)| tras proyeccion")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
    ax.set_xlim(4, 12); ax.set_ylim(5, 11)
    fig.colorbar(im, ax=ax)
    fig.savefig(os.path.join(B.OUT, "diag_divergencia.png"), dpi=150)
    plt.close(fig)
    print(f"\nguardado: {dst}")


if __name__ == "__main__":
    main()

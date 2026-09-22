"""Proceso hijo que ejecuta una corrida del solver curvilineo."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from curvo import convergencia as cvg
from curvo import fuerzas
from curvo import malla
from curvo.solver import Solver
from .caso import Caso
from .snapshot import SnapshotWriter


CENTINELA = "STOP_SIMULATION.trigger"


def run(caso: Caso, raiz: str | Path | None = None) -> Path:
    """Ejecuta un caso y devuelve su directorio de resultados."""
    caso.validar(raiz)
    salida = Path(caso.ejecucion.get("directorio") or "runs/gui_run")
    if not salida.is_absolute():
        salida = (Path(raiz) if raiz is not None else Path.cwd()) / salida
    (salida / "campos").mkdir(parents=True, exist_ok=True)
    caso.guardar(salida / "caso.json")

    px, py = malla.leer_dat(caso.ruta_perfil(raiz))
    X, Y, info = malla.generar_c(px, py, **caso.argumentos_malla())
    calidad = malla.calidad(X, Y, perfil=info)
    np.savez_compressed(salida / "malla.npz", X=X, Y=Y,
                        perfil=np.asarray(info["perfil"], dtype=int))
    (salida / "calidad.json").write_text(
        json.dumps(_jsonable(calidad), indent=2), encoding="utf-8")

    solver_kwargs = dict(caso.solver)
    backend = caso.ejecucion["backend"]
    xp = np
    if backend == "cupy":
        import cupy as cp
        xp = cp
    dtype = np.dtype(caso.ejecucion["dtype"])
    solver = Solver(X, Y, info, xp=xp, dtype=dtype, **solver_kwargs)
    Xpub, Ypub = X.astype(dtype), Y.astype(dtype)
    writer = SnapshotWriter()
    writer.create(Xpub, Ypub)
    (salida / "ipc.json").write_text(json.dumps({"names": writer.names}), encoding="utf-8")

    cada_historia = int(caso.ejecucion["cada_historia"])
    cada_campo = int(caso.ejecucion["cada_campo"])
    pasos = int(caso.ejecucion["pasos"])
    tol_fuerzas = float(caso.ejecucion.get("tol_fuerzas", 0.0))
    parada = cvg.ParadaFuerzas(tol=tol_fuerzas) if tol_fuerzas > 0 else None
    historia = []
    marco = 0
    inicio = time.perf_counter()
    try:
        for paso in range(pasos + 1):
            metricas = _metricas(solver, info, caso)
            metricas["it_s"] = paso / max(time.perf_counter() - inicio, 1e-9)
            if paso % cada_historia == 0:
                historia.append(metricas)
                _publicar(salida, writer, solver, metricas, Xpub, Ypub, paso)
                if parada is not None and parada.anotar(
                        metricas["t"], metricas["Cl"], metricas["Cd"])["ok"]:
                    r = parada.resumen()
                    (salida / "convergencia.json").write_text(
                        json.dumps(_jsonable(r), indent=2), encoding="utf-8")
                    print("[solver] fuerzas asentadas en t=%.2f: Cl=%+.5f "
                          "Cd=%+.5f (cola %.1e / %.1e)"
                          % (r["t"], r["Cl"], r["Cd"], r["cola"]["Cl"],
                             r["cola"]["Cd"]), flush=True)
                    break
            if paso % cada_campo == 0:
                _guardar_campo(salida, solver, paso, caso.solver["dt"])
                marco += 1
            if paso == pasos:
                break
            if (salida / CENTINELA).exists():
                print("[solver] parada solicitada", flush=True)
                break
            solver.paso()
            if paso % max(cada_historia, cada_campo) == 0:
                print("[solver] paso %d/%d Cl=%+.5f Cd=%+.5f" %
                      (paso, pasos, metricas["Cl"], metricas["Cd"]), flush=True)
    finally:
        if solver.paso_n % cada_campo != 0:
            _guardar_campo(salida, solver, solver.paso_n, solver.dt)
        writer.close()
        (salida / "historia.npz").write_bytes(_historia_npz(historia))
        (salida / "reparto.json").write_text(
            json.dumps(_jsonable(solver.reparto()), indent=2), encoding="utf-8")
        print(f"[solver] terminado: {len(historia)} muestras, {marco} campos", flush=True)
    return salida


def _metricas(solver, info, caso):
    campos = solver.campos()
    e = fuerzas.estimadores_de_cl(
        solver.met, solver.u, solver.v, solver.p, info, solver.nu,
        solver.u_inf, solver.alfa, corte=solver.corte)
    f = e["fuerzas"]
    return {"iter": int(solver.paso_n), "t": solver.paso_n * solver.dt,
            "Cl": e["superficie"], "Cd": f["Cd"], "Cm": f["Cm"],
            "div": solver.divergencia(), "speed_max": float(np.hypot(
                campos["u"], campos["v"]).max())}


def _publicar(salida, writer, solver, metricas, X, Y, paso):
    campos = solver.campos()
    writer.publish(X, Y, campos["u"].astype(X.dtype), campos["v"].astype(X.dtype),
                   campos["p"].astype(X.dtype), metricas)


def _guardar_campo(salida, solver, paso, dt):
    campos = solver.campos()
    np.savez_compressed(salida / "campos" / f"campo_{paso:06d}.npz",
                        t=paso * dt, **campos)


def _historia_npz(historia):
    if not historia:
        datos = np.empty((0, 6), dtype=float)
    else:
        nombres = ("iter", "t", "Cl", "Cd", "Cm", "div")
        datos = np.asarray([[fila[nombre] for nombre in nombres] for fila in historia])
    import io
    buffer = io.BytesIO()
    np.savez_compressed(buffer, datos=datos,
                        columnas=np.asarray(("iter", "t", "Cl", "Cd", "Cm", "div")))
    return buffer.getvalue()


def _jsonable(valor):
    if isinstance(valor, dict):
        return {str(k): _jsonable(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_jsonable(v) for v in valor]
    if isinstance(valor, np.generic):
        return valor.item()
    if isinstance(valor, float) and not np.isfinite(valor):
        return None
    return valor


def main():
    if len(sys.argv) != 2:
        raise SystemExit("uso: python -m gui.run_solver caso.json")
    ruta = Path(sys.argv[1]).resolve()
    caso = Caso.cargar(ruta)
    run(caso, Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()

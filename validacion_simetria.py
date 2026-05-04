"""
validacion_simetria.py

Banco de validacion para el simulador 2D LES (`Simulador2D.py`).
Mide en NACA0012 a alpha=0:
  - Cl, Cd (medias en regimen estacionario)
  - it/s (rendimiento)
  - div_max/U_inf final (residual proyeccion)
  - ratio nu_t/nu (si WALE activo)
  - ciclos MG promedio

Soporta dos puntos de Reynolds (100k y 1M) y guarda un JSON de baseline
para comparar antes/despues de cada paso del plan.

Uso:
    python validacion_simetria.py --baseline
    python validacion_simetria.py --tag paso2_wale
    python validacion_simetria.py --diff baseline paso2_wale
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import cupy as cp

# Importar el simulador
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Simulador2D import main as simular_main  # type: ignore

CARPETA_RESULTADOS = Path(__file__).parent / "validacion_resultados"
CARPETA_RESULTADOS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Configuracion estandar de validacion
# ---------------------------------------------------------------------------

CONFIG_BASE = dict(
    # Geometria
    filepath="NACA_0012",
    chord=1.0,
    alpha_deg=0.0,

    # Dominio (igual que el __main__ del simulador)
    Lx=12.0,
    Ly=8.0,
    cx=2.0,
    cy=None,  # auto-centrado

    # Malla
    dx_min=0.001,
    factor_expansion=1.1,
    ancho_zona_fina_x=1.2,
    ancho_zona_fina_y=1.0,

    # Flujo
    v0x=1.0,
    v0y=0.0,
    rho=1.0,
    p0=0.0,

    # Numerica
    CFL=0.5,
    divergencia=1e-1,

    # Iteraciones (corto, validacion rapida)
    iteraciones=500,
    guardado=25,

    # Modo MG
    mg_modo_turbo=True,

    # Apagar visualizacion
    graficos=False,
    save_frames=False,
    live_view=False,
    mostrar_malla=False,
    stop_on_convergence=False,

    # Desactivar band-aid de simetria para medir lift BRUTO
    corregir_deriva_vertical=False,
)


@dataclass
class RunMetrics:
    tag: str
    Re: float
    nu: float
    usar_wale: bool
    iteraciones: int
    walltime_s: float
    its_per_sec: float
    Cl_mean: float
    Cl_std: float
    Cd_mean: float
    Cd_std: float
    nx: int
    ny: int
    n_celdas: int
    div_max_final: float
    nu_t_max_ratio: float  # max(nu_t)/nu
    mg_cycles_mean: float
    notes: str = ""


def _build_run_kwargs(Re: float, usar_wale: bool, **overrides) -> dict:
    """Configura nu para alcanzar el Re objetivo con U=1, c=1."""
    nu = CONFIG_BASE["v0x"] * CONFIG_BASE["chord"] / Re
    cfg = dict(CONFIG_BASE)
    cfg["nu"] = nu
    cfg["usar_wale"] = usar_wale
    cfg.update(overrides)
    return cfg


def _suppress_stdout():
    """Context manager: silencia stdout (libera spam del simulador en logs)."""
    import contextlib
    import io
    return contextlib.redirect_stdout(io.StringIO())


def _extract_metrics(mesh, walltime_s: float, iteraciones: int,
                     Re: float, nu: float, usar_wale: bool, tag: str) -> RunMetrics:
    # Cl, Cd: descartar 30% inicial (transitorio)
    cd_arr = cp.asnumpy(mesh.cdvector).flatten()
    cl_arr = cp.asnumpy(mesh.clvector).flatten()
    mask = (cd_arr != 0) | (cl_arr != 0)
    if mask.any():
        cd_arr = cd_arr[mask]
        cl_arr = cl_arr[mask]
    n_descartar = max(1, int(len(cd_arr) * 0.3))
    cd_ss = cd_arr[n_descartar:]
    cl_ss = cl_arr[n_descartar:]
    cd_mean = float(np.mean(cd_ss)) if len(cd_ss) > 0 else 0.0
    cd_std = float(np.std(cd_ss)) if len(cd_ss) > 0 else 0.0
    cl_mean = float(np.mean(cl_ss)) if len(cl_ss) > 0 else 0.0
    cl_std = float(np.std(cl_ss)) if len(cl_ss) > 0 else 0.0

    # Divergencia residual (norma L_inf adimensional usando v0x=U_ref)
    try:
        div = mesh._compute_divergence_field()
        div_max = float(cp.max(cp.abs(div)))
    except Exception:
        div_max = float("nan")

    # Ratio nu_t/nu (si WALE)
    if usar_wale:
        try:
            nu_t = mesh.compute_wale_viscosity()
            nu_t_max = float(cp.max(nu_t))
            nu_t_ratio = nu_t_max / nu
        except Exception:
            nu_t_ratio = float("nan")
    else:
        nu_t_ratio = 0.0

    # MG cycles promedio
    try:
        mg_cycles_arr = cp.asnumpy(mesh.mg_cycles_vector)
        mg_cycles_arr = mg_cycles_arr[mg_cycles_arr > 0]
        mg_cycles_mean = float(np.mean(mg_cycles_arr)) if len(mg_cycles_arr) > 0 else 0.0
    except Exception:
        mg_cycles_mean = float("nan")

    its_per_sec = iteraciones / walltime_s if walltime_s > 0 else 0.0

    return RunMetrics(
        tag=tag,
        Re=Re,
        nu=nu,
        usar_wale=usar_wale,
        iteraciones=iteraciones,
        walltime_s=walltime_s,
        its_per_sec=its_per_sec,
        Cl_mean=cl_mean,
        Cl_std=cl_std,
        Cd_mean=cd_mean,
        Cd_std=cd_std,
        nx=int(mesh.nx),
        ny=int(mesh.ny),
        n_celdas=int(mesh.nx * mesh.ny),
        div_max_final=div_max,
        nu_t_max_ratio=nu_t_ratio,
        mg_cycles_mean=mg_cycles_mean,
    )


def run_one(Re: float, usar_wale: bool, tag: str, silenciar: bool = True,
            **overrides) -> RunMetrics:
    cfg = _build_run_kwargs(Re=Re, usar_wale=usar_wale, **overrides)
    nu = cfg["nu"]
    iteraciones = cfg["iteraciones"]

    print(f"\n[run] tag={tag}  Re={Re:.2e}  nu={nu:.2e}  WALE={usar_wale}  iter={iteraciones}")
    t0 = time.time()
    if silenciar:
        with _suppress_stdout():
            mesh = simular_main(**cfg)
    else:
        mesh = simular_main(**cfg)
    walltime = time.time() - t0
    print(f"[run]   walltime={walltime:.1f}s  it/s={iteraciones/walltime:.2f}")

    metrics = _extract_metrics(mesh, walltime, iteraciones, Re, nu, usar_wale, tag)
    return metrics


def run_baseline(tag: str = "baseline", silenciar: bool = True) -> dict:
    """Corre los 4 puntos de validacion: 2 Re x {WALE off, WALE on}."""
    resultados: list[RunMetrics] = []

    for Re in [1e5, 1e6]:
        for usar_wale in [False, True]:
            etiqueta = f"{tag}__Re{int(Re):.0e}__wale{int(usar_wale)}"
            try:
                m = run_one(Re=Re, usar_wale=usar_wale, tag=etiqueta,
                            silenciar=silenciar)
                resultados.append(m)
            except Exception as e:
                print(f"[run] ERROR en {etiqueta}: {e}")
                import traceback
                traceback.print_exc()

    # Guardar JSON
    out_path = CARPETA_RESULTADOS / f"{tag}.json"
    payload = {
        "tag": tag,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_base": CONFIG_BASE,
        "runs": [asdict(m) for m in resultados],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[save] resultados -> {out_path}")

    # Tabla resumen
    print("\n" + "=" * 96)
    print(f"{'tag':<40}{'Re':>9}{'WALE':>6}{'it/s':>8}{'Cl':>10}{'Cd':>10}"
          f"{'div_max':>10}{'nut/nu':>10}")
    print("=" * 96)
    for m in resultados:
        print(f"{m.tag[-36:]:<40}{m.Re:>9.0e}{int(m.usar_wale):>6d}"
              f"{m.its_per_sec:>8.2f}{m.Cl_mean:>10.4f}{m.Cd_mean:>10.4f}"
              f"{m.div_max_final:>10.2e}{m.nu_t_max_ratio:>10.1e}")
    print("=" * 96)

    return payload


def sweep_tol(tag: str = "sweep_tol",
              tols: tuple = (2e-2, 1e-2, 5e-3, 2e-3, 1e-3),
              Re: float = 1e5,
              usar_wale: bool = False,
              iteraciones: int = 500,
              silenciar: bool = True) -> dict:
    """
    Barre `divergencia` en la lista `tols` para encontrar el mas alto que
    cumple Cl<0.005 y it/s>=3. Ejecuta NACA0012 alpha=0 a Re fijo.

    Devuelve un dict con la tabla de resultados y la recomendacion.
    """
    print(f"\n[sweep_tol] tag={tag}  Re={Re:.0e}  WALE={usar_wale}  "
          f"iters={iteraciones}  tols={tols}")

    rows: list[RunMetrics] = []
    for tol in tols:
        etiqueta = f"{tag}__tol{tol:.0e}"
        try:
            m = run_one(Re=Re, usar_wale=usar_wale,
                        tag=etiqueta, silenciar=silenciar,
                        iteraciones=iteraciones,
                        divergencia=float(tol))
            m.notes = f"tol={tol:.0e}"
            rows.append(m)
        except Exception as e:
            print(f"[sweep_tol] ERROR en {etiqueta}: {e}")
            import traceback
            traceback.print_exc()

    # Tabla
    print("\n" + "=" * 96)
    print(f"{'tol':>10}{'it/s':>10}{'Cl':>10}{'|Cl|':>10}"
          f"{'Cd':>10}{'div_max':>12}{'mg_cyc':>10}{'OK?':>8}")
    print("=" * 96)
    candidato_optimo = None
    for r in rows:
        ok = (abs(r.Cl_mean) < 0.005) and (r.its_per_sec >= 3.0)
        flag = "OK" if ok else "-"
        # primer "OK" en orden de tols (descendente en strict, ascendente en valor)
        if ok and candidato_optimo is None:
            candidato_optimo = r
        # tomamos el de mayor tol (el primero al barrer en orden 2e-2 -> 1e-3)
        print(f"{r.notes:>10}{r.its_per_sec:>10.2f}{r.Cl_mean:>10.4f}"
              f"{abs(r.Cl_mean):>10.4f}{r.Cd_mean:>10.4f}"
              f"{r.div_max_final:>12.2e}{r.mg_cycles_mean:>10.2f}{flag:>8}")
    print("=" * 96)
    if candidato_optimo is not None:
        print(f"\n[sweep_tol] candidato optimo: {candidato_optimo.notes} "
              f"(Cl={candidato_optimo.Cl_mean:.4f}, "
              f"it/s={candidato_optimo.its_per_sec:.2f})")
    else:
        print(f"\n[sweep_tol] ningun tol cumple Cl<0.005 con it/s>=3")

    payload = {
        "tag": tag,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Re": Re,
        "usar_wale": usar_wale,
        "iteraciones": iteraciones,
        "tols": list(tols),
        "rows": [asdict(r) for r in rows],
        "candidato_optimo": asdict(candidato_optimo) if candidato_optimo else None,
    }
    out_path = CARPETA_RESULTADOS / f"{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[sweep_tol] guardado -> {out_path}")
    return payload


def diff_runs(tag_a: str, tag_b: str) -> None:
    """Compara dos runs guardados."""
    pa = CARPETA_RESULTADOS / f"{tag_a}.json"
    pb = CARPETA_RESULTADOS / f"{tag_b}.json"
    if not pa.exists() or not pb.exists():
        print(f"[diff] falta archivo: {pa if not pa.exists() else pb}")
        return
    da = json.loads(pa.read_text(encoding="utf-8"))
    db = json.loads(pb.read_text(encoding="utf-8"))

    runs_a = {r["tag"].split("__", 1)[1] if "__" in r["tag"] else r["tag"]: r for r in da["runs"]}
    runs_b = {r["tag"].split("__", 1)[1] if "__" in r["tag"] else r["tag"]: r for r in db["runs"]}

    print(f"\n[diff] {tag_a}  vs  {tag_b}")
    print("=" * 110)
    print(f"{'caso':<28}{'metrica':<14}{'antes':>14}{'despues':>14}{'delta':>14}{'%':>10}")
    print("=" * 110)
    for caso in sorted(runs_a):
        if caso not in runs_b:
            continue
        ra = runs_a[caso]
        rb = runs_b[caso]
        for metrica in ("its_per_sec", "Cl_mean", "Cd_mean",
                        "div_max_final", "nu_t_max_ratio", "mg_cycles_mean"):
            va = ra.get(metrica, float("nan"))
            vb = rb.get(metrica, float("nan"))
            try:
                delta = vb - va
                pct = 100.0 * delta / va if va not in (0, 0.0) else float("nan")
                print(f"{caso:<28}{metrica:<14}{va:>14.4g}{vb:>14.4g}{delta:>14.4g}{pct:>10.2f}")
            except Exception:
                print(f"{caso:<28}{metrica:<14}{va!s:>14}{vb!s:>14}")
    print("=" * 110)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validacion baseline NACA0012")
    parser.add_argument("--tag", type=str, default="baseline",
                        help="etiqueta del run (default: baseline)")
    parser.add_argument("--baseline", action="store_true",
                        help="alias de --tag baseline")
    parser.add_argument("--diff", nargs=2, metavar=("TAG_A", "TAG_B"),
                        help="compara dos runs guardados y termina")
    parser.add_argument("--iters", type=int, default=None,
                        help="override iteraciones (default 500)")
    parser.add_argument("--verbose", action="store_true",
                        help="no silenciar stdout del simulador")
    parser.add_argument("--sweep-tol", action="store_true",
                        help="sweep divergencia en {2e-2..1e-3} a Re=100k WALE off")
    args = parser.parse_args()

    if args.diff:
        diff_runs(args.diff[0], args.diff[1])
        return

    if args.baseline:
        args.tag = "baseline"

    if args.iters is not None:
        CONFIG_BASE["iteraciones"] = int(args.iters)

    if args.sweep_tol:
        sweep_tol(tag=args.tag if args.tag != "baseline" else "sweep_tol",
                  silenciar=not args.verbose,
                  iteraciones=CONFIG_BASE["iteraciones"])
        return

    run_baseline(tag=args.tag, silenciar=not args.verbose)


if __name__ == "__main__":
    main()

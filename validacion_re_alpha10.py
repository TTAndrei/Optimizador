"""
Re-sweep a alpha=10 deg con WALE+Cw=0.15.
Mide Cd, Cl en Re=1e5 y Re=1e6 para verificar Re-decoupling.
Salida: validacion_resultados/re_alpha10.json
"""
from __future__ import annotations
import json
import sys
import os
import time
from pathlib import Path

import numpy as np
import cupy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Simulador2D import main as simular_main

OUT = Path(__file__).parent / "validacion_resultados" / "re_alpha10.json"
OUT.parent.mkdir(exist_ok=True)


def run(Re: float, iter_n: int = 10000) -> dict:
    nu = 1.0 / Re
    cfg = dict(
        filepath="NACA_0012",
        chord=1.0,
        alpha_deg=10.0,
        Lx=12.0, Ly=8.0, cx=2.0,
        dx_min=0.001,
        factor_expansion=1.1,
        ancho_zona_fina_x=1.2,
        ancho_zona_fina_y=1.0,
        v0x=1.0, v0y=0.0,
        rho=1.0, p0=0.0,
        nu=nu,
        CFL=0.5,
        divergencia=1e-1,
        iteraciones=iter_n,
        guardado=25,
        mg_modo_turbo=True,
        graficos=False,
        save_frames=False,
        live_view=False,
        mostrar_malla=False,
        stop_on_convergence=False,
        corregir_deriva_vertical=False,
        usar_wale=True,
        wale_Cw=0.15,
    )

    print(f"\n[run] Re={Re:.0e} nu={nu:.2e} iter={iter_n}", flush=True)
    t0 = time.time()
    mesh = simular_main(**cfg)
    walltime = time.time() - t0

    cd_arr = cp.asnumpy(mesh.cdvector).flatten()
    cl_arr = cp.asnumpy(mesh.clvector).flatten()
    mask = (cd_arr != 0) | (cl_arr != 0)
    cd_arr = cd_arr[mask]; cl_arr = cl_arr[mask]
    n_desc = max(1, int(len(cd_arr) * 0.3))
    cd_ss = cd_arr[n_desc:]; cl_ss = cl_arr[n_desc:]

    try:
        nu_t = mesh.compute_wale_viscosity()
        nu_t_ratio = float(cp.max(nu_t)) / nu
    except Exception:
        nu_t_ratio = float("nan")

    return dict(
        Re=Re, nu=nu, iteraciones=iter_n,
        walltime_s=walltime,
        its_per_sec=iter_n / walltime if walltime > 0 else 0,
        Cl_mean=float(np.mean(cl_ss)),
        Cl_std=float(np.std(cl_ss)),
        Cd_mean=float(np.mean(cd_ss)),
        Cd_std=float(np.std(cd_ss)),
        nu_t_max_ratio=nu_t_ratio,
        n_samples=int(len(cd_ss)),
    )


def main():
    iter_n = int(os.environ.get("ITER", "10000"))
    runs = []
    for Re in (1e5, 1e6):
        try:
            r = run(Re, iter_n=iter_n)
            print(f"[run] Re={Re:.0e} Cl={r['Cl_mean']:.4f} Cd={r['Cd_mean']:.4f} "
                  f"nu_t/nu={r['nu_t_max_ratio']:.2f} it/s={r['its_per_sec']:.2f}",
                  flush=True)
            runs.append(r)
        except Exception as e:
            print(f"[run] ERROR Re={Re:.0e}: {e}", flush=True)
            import traceback; traceback.print_exc()

    if len(runs) == 2:
        Cd_lo, Cd_hi = runs[0]["Cd_mean"], runs[1]["Cd_mean"]
        diff = abs(Cd_lo - Cd_hi) / max(abs(Cd_hi), 1e-30)
        print(f"\n[summary] |dCd|/Cd(1e6) = {diff*100:.1f}%   (target >15% para Re-decoupling)", flush=True)
        runs_payload = {"runs": runs, "delta_Cd_rel": diff}
    else:
        runs_payload = {"runs": runs}

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(runs_payload, f, indent=2)
    print(f"[save] {OUT}", flush=True)


if __name__ == "__main__":
    main()

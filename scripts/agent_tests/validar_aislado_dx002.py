#!/usr/bin/env python3
"""Re-evalúa los ganadores del aislado (corridos a dx=0.004) a dx=0.002.

El estudio aislado optimizó con fitness a dx=0.004 (pared sub-resuelta, y+~20).
Los ganadores dan Cl 0.76-1.11 @ α=4, muy por encima de lo físico (~0.36 base).
Aquí se re-evalúan las MISMAS formas a la fidelidad de referencia (dx=0.002) para
ver si el ranking y los Cl aguantan o eran artefacto numérico.

Salida: results/convergence_study/aislado_sin_migracion_convergido/validacion_dx002.json
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import RunGA  # noqa: E402

ALPHA = 4.0
CFL = 0.5
DX = 0.002
U_FAC = 1.6
T_TARGET = 8.0
OUT_DIR = "results/convergence_study/aislado_sin_migracion_convergido"


def iters_for(dx):
    n = T_TARGET * U_FAC / (CFL * dx)
    return int(-(-n // 1000) * 1000)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def ganador_dat(seed_dir):
    cands = [p for p in glob.glob(os.path.join(seed_dir, "*.dat"))
             if "OPTIMO_PARCIAL" not in os.path.basename(p)]
    return cands[0] if cands else None


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
    with open(os.path.join(OUT_DIR, "convergencia_aislado.json")) as f:
        base = json.load(f)["ganadores"]

    cfg = dict(RunGA.CONFIG)
    cfg["dx_min"] = DX
    cfg["simulacion_iteraciones"] = iters_for(DX)
    cfg["CFL"] = CFL
    log(f"config referencia dx={DX} iters={cfg['simulacion_iteraciones']} α={ALPHA}")

    filas = []
    for seed in ["NACA_0012_sharp", "AG24", "GM15", "s1014"]:
        dat = ganador_dat(os.path.join(OUT_DIR, seed))
        if not dat:
            log(f"{seed}: sin .dat ganador, saltado")
            continue
        b = base[seed]
        log(f"{seed}: eval dx=0.002 sobre {os.path.basename(dat)} "
            f"(dx004: L/D {b['ld']:.2f} Cl {b['cl']:.3f} Cd {b['cd']:.4f}) ...")
        t0 = time.time()
        res = RunGA.simular_perfil(dat, ALPHA, cfg)
        dt = time.time() - t0
        if res is None:
            log(f"  {seed}: FALLO tras {dt/60:.1f} min")
            filas.append({"seed": seed, "error": True,
                          "ld_dx004": b["ld"], "cl_dx004": b["cl"], "cd_dx004": b["cd"]})
            continue
        fila = {
            "seed": seed, "wall_min": round(dt / 60, 1),
            "ld_dx004": b["ld"], "cl_dx004": b["cl"], "cd_dx004": b["cd"],
            "ld_dx002": round(res["ld"], 4), "cl_dx002": round(res["cl"], 4),
            "cd_dx002": round(res["cd"], 4),
            "d_ld_pct": round(100 * (res["ld"] - b["ld"]) / b["ld"], 1),
            "d_cl_pct": round(100 * (res["cl"] - b["cl"]) / b["cl"], 1),
        }
        filas.append(fila)
        log(f"  {seed}: dx002 L/D {res['ld']:.2f} Cl {res['cl']:.3f} Cd {res['cd']:.4f} "
            f"(ΔL/D {fila['d_ld_pct']:+.0f}%, ΔCl {fila['d_cl_pct']:+.0f}%, {dt/60:.1f} min)")

    ok = [f for f in filas if not f.get("error")]
    rank004 = sorted(ok, key=lambda f: f["ld_dx004"], reverse=True)
    rank002 = sorted(ok, key=lambda f: f["ld_dx002"], reverse=True)
    g004 = rank004[0]["seed"] if rank004 else None
    g002 = rank002[0]["seed"] if rank002 else None
    out = {
        "alpha": ALPHA, "dx_ref": DX, "iters": cfg["simulacion_iteraciones"],
        "filas": filas,
        "orden_dx004": [f["seed"] for f in rank004],
        "orden_dx002": [f["seed"] for f in rank002],
        "ganador_dx004": g004, "ganador_dx002": g002,
        "cambia_ganador": g004 != g002,
        "veredicto": (
            f"Ganador dx004={g004} -> dx002={g002}. "
            + ("El ranking CAMBIA al refinar (fitness dx=0.004 no fiable para estas formas)."
               if g004 != g002 else
               "El ranking se mantiene tras refinar.")),
    }
    dst = os.path.join(OUT_DIR, "validacion_dx002.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log("VEREDICTO: " + out["veredicto"])
    log(f"escrito {dst}")


if __name__ == "__main__":
    main()

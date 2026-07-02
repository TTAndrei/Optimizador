"""
Runner de tests del sistema Kutta (coarse, autoverificable).

Uso (desde la raíz del repo, SIEMPRE con el venv):
    .venv/bin/python scripts/agent_tests/run_kutta_tests.py smoke
    .venv/bin/python scripts/agent_tests/run_kutta_tests.py baseline_a5
    .venv/bin/python scripts/agent_tests/run_kutta_tests.py polygon_a5
    .venv/bin/python scripts/agent_tests/run_kutta_tests.py compare
    .venv/bin/python scripts/agent_tests/run_kutta_tests.py regression

Exit code 0 = OK, 1 = fallo. Resultados en results/agent_tests/<nombre>.json (+.npz).
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
RESULTS_DIR = os.path.join(ROOT, "results", "agent_tests")
os.makedirs(RESULTS_DIR, exist_ok=True)

COARSE = dict(
    Lx=8, Ly=5, cx=2, chord=1.0, v0x=1, v0y=0, rho=1.0, nu=1e-5,
    dx_min=0.004, ancho_zona_fina_x=1.5, ancho_zona_fina_y=1.0,
    factor_expansion=1.1, CFL=0.25, divergencia=1e-1,
    filepath="profiles/NACA_0012_sharp", min_te_height_factor=1.0,
    mg_modo_turbo_hd=True, wake_refinement_mode="long_fine_x",
    guardado=50, graficos=False, live_view=False, mostrar_malla=False,
    stop_on_convergence=False, save_frames=False,
)


def _run_sim(nombre, **overrides):
    import cupy as cp
    import numpy as np
    from Simulador2D import main

    params = {**COARSE, **overrides}
    t0 = time.time()
    mesh = main(**params)
    wall = time.time() - t0

    cl = cp.asnumpy(mesh.clvector).astype(float)
    cd = cp.asnumpy(mesh.cdvector).astype(float)
    dcp = cp.asnumpy(mesh.kutta_dcp_vector).astype(float)
    n = len(cl)
    w = max(1, n // 5)          # ventana = último 20%
    cl_w, cd_w, dcp_w = cl[-w:], cd[-w:], dcp[-w:]
    dcp_ok = dcp_w[np.isfinite(dcp_w)]
    # deriva: media del último 10% vs el 10% previo
    h = max(1, n // 10)
    deriva_cl = float(abs(np.mean(cl[-h:]) - np.mean(cl[-2 * h:-h]))) if n >= 2 * h else float("nan")

    mu = params["rho"] * params["nu"]
    circ = mesh.compute_circulation(loop_margins=(0.25,), verbose=False)
    cpd = mesh.compute_cp_diagnostics(mu, params["rho"], verbose=False) or {}
    te = mesh.debug_te_report()

    chi_max = float("nan")
    if getattr(mesh, "nu_tilde", None) is not None:
        chi_max = float(cp.max(mesh.nu_tilde)) / params["nu"]

    res = {
        "nombre": nombre,
        "params": {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))},
        "wall_s": round(wall, 1),
        "t_fisico": float(getattr(mesh, "_t_fisico", float("nan"))),
        "chi_max": chi_max,
        "n_muestras": n,
        "nan_en_cl": bool(np.any(~np.isfinite(cl))),
        "cl_medio_ventana": float(np.mean(cl_w)),
        "cl_std_ventana": float(np.std(cl_w)),
        "cd_medio_ventana": float(np.mean(cd_w)),
        "deriva_cl": deriva_cl,
        "dcp_te_medio_ventana": float(np.mean(np.abs(dcp_ok))) if dcp_ok.size else float("nan"),
        "cl_circ_025c": float(circ[0]["cl_circ"]) if circ else float("nan"),
        "cl_cp": cpd.get("cl_cp", float("nan")),
        "cp_min": cpd.get("cp_min", float("nan")),
        "x_suction": cpd.get("x_suction", float("nan")),
        "te_report": te,
    }
    path = os.path.join(RESULTS_DIR, f"{nombre}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    mesh.save_state(os.path.join(RESULTS_DIR, f"{nombre}.npz"))
    print(f"\n[{nombre}] guardado en {path}")
    print(json.dumps(res, indent=2, default=str))
    return res


def _load(nombre):
    path = os.path.join(RESULTS_DIR, f"{nombre}.json")
    with open(path) as f:
        return json.load(f)


def cmd_smoke():
    res = _run_sim("smoke", alpha_deg=0, iteraciones=3000)
    fallos = []
    if res["nan_en_cl"]:
        fallos.append("NaN en clvector")
    if abs(res["cl_medio_ventana"]) >= 0.03:
        fallos.append(f"|Cl| a α=0 = {abs(res['cl_medio_ventana']):.4f} >= 0.03")
    import math
    if math.isnan(res["dcp_te_medio_ventana"]):
        fallos.append("kutta_dcp_vector sin muestras válidas")
    return fallos


def cmd_smoke_polygon():
    res = _run_sim("smoke_polygon", alpha_deg=0, iteraciones=3000, ibm_sdf_source="polygon")
    fallos = []
    if res["nan_en_cl"]:
        fallos.append("NaN en clvector")
    if abs(res["cl_medio_ventana"]) >= 0.03:
        fallos.append(f"|Cl| a α=0 = {abs(res['cl_medio_ventana']):.4f} >= 0.03")
    if res["te_report"].get("n_irrep_te", 99) > 0:
        fallos.append(f"irreparables en TE = {res['te_report']['n_irrep_te']} (esperado 0)")
    return fallos


def cmd_baseline_a5():
    _run_sim("baseline_a5", alpha_deg=5, iteraciones=12000, ibm_sdf_source="edt")
    return []


def cmd_polygon_a5():
    _run_sim("polygon_a5", alpha_deg=5, iteraciones=12000, ibm_sdf_source="polygon")
    return []


def cmd_compare():
    b = _load("baseline_a5")
    p = _load("polygon_a5")
    fallos = []
    checks = []

    red = 1.0 - p["dcp_te_medio_ventana"] / max(b["dcp_te_medio_ventana"], 1e-12)
    checks.append(("media|ΔCp_TE| reducción >=50%",
                   f"edt={b['dcp_te_medio_ventana']:.4f} poly={p['dcp_te_medio_ventana']:.4f} (red={red:.0%})",
                   red >= 0.5))
    mejora_cl = b["cl_medio_ventana"] - p["cl_medio_ventana"]
    checks.append(("Cl baja >=0.1 hacia 0.55",
                   f"edt={b['cl_medio_ventana']:.4f} poly={p['cl_medio_ventana']:.4f}",
                   mejora_cl >= 0.1 and p["cl_medio_ventana"] > 0.3))
    checks.append(("std(Cl) ventana < 0.05", f"{p['cl_std_ventana']:.4f}", p["cl_std_ventana"] < 0.05))
    checks.append(("deriva Cl < 0.03", f"{p['deriva_cl']:.4f}", p["deriva_cl"] < 0.03))
    checks.append(("sin NaN", f"{p['nan_en_cl']}", not p["nan_en_cl"]))
    checks.append(("0 irreparables TE (poly)", f"{p['te_report'].get('n_irrep_te')}",
                   p["te_report"].get("n_irrep_te", 99) == 0))

    print(f"\n{'='*72}\nCOMPARE baseline_a5 (edt) vs polygon_a5 (polygon)\n{'='*72}")
    for nombre, detalle, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {nombre}: {detalle}")
        if not ok:
            fallos.append(nombre)
    return fallos


def cmd_regression():
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-x", "-q"], cwd=ROOT)
    return [] if r.returncode == 0 else [f"pytest exit={r.returncode}"]


def _estable(res, fallos):
    if res["nan_en_cl"]:
        fallos.append("NaN en clvector")
    if res["cl_std_ventana"] >= 0.05:
        fallos.append(f"std(Cl)={res['cl_std_ventana']:.4f} >= 0.05")
    if not res["deriva_cl"] < 0.03:
        fallos.append(f"deriva Cl={res['deriva_cl']:.4f} >= 0.03")


def cmd_sa_sanity():
    res = _run_sim("sa_sanity", alpha_deg=0, iteraciones=1500, turb_model="sa")
    fallos = []
    if res["nan_en_cl"]:
        fallos.append("NaN en clvector")
    if not res["chi_max"] < 5000:
        fallos.append(f"chi_max={res['chi_max']:.1f} (runaway o NaN)")
    return fallos


def cmd_sa_a0():
    res = _run_sim("sa_a0", alpha_deg=0, iteraciones=6000, turb_model="sa")
    fallos = []
    _estable(res, fallos)
    if abs(res["cl_medio_ventana"]) >= 0.03:
        fallos.append(f"|Cl| a α=0 = {abs(res['cl_medio_ventana']):.4f} >= 0.03")
    if not res["cd_medio_ventana"] > 0:
        fallos.append(f"Cd={res['cd_medio_ventana']:.4f} <= 0")
    return fallos


def cmd_sa_a5():
    res = _run_sim("sa_a5", alpha_deg=5, iteraciones=12000, turb_model="sa")
    fallos = []
    _estable(res, fallos)
    if not (0.4 <= res["cl_medio_ventana"] <= 0.7):
        fallos.append(f"Cl={res['cl_medio_ventana']:.4f} fuera de [0.4, 0.7]")
    if not res["dcp_te_medio_ventana"] < 0.08:
        fallos.append(f"|ΔCp_TE|={res['dcp_te_medio_ventana']:.4f} >= 0.08")
    if not res["x_suction"] < 0.15:
        fallos.append(f"x_suction={res['x_suction']:.3f} >= 0.15 (LSB no eliminada)")
    if not res["t_fisico"] >= 12:
        fallos.append(f"t_fisico={res['t_fisico']:.1f} < 12 convectivos (alargar run)")
    return fallos


def cmd_sa_a2():
    res = _run_sim("sa_a2", alpha_deg=2, iteraciones=12000, turb_model="sa")
    fallos = []
    _estable(res, fallos)
    if not (0.15 <= res["cl_medio_ventana"] <= 0.3):
        fallos.append(f"Cl={res['cl_medio_ventana']:.4f} fuera de [0.15, 0.3]")
    return fallos


def cmd_sa_a8():
    res = _run_sim("sa_a8", alpha_deg=8, iteraciones=12000, turb_model="sa")
    fallos = []
    _estable(res, fallos)
    a5 = _load("sa_a5")
    if not res["cl_medio_ventana"] > a5["cl_medio_ventana"]:
        fallos.append(f"Cl(α8)={res['cl_medio_ventana']:.3f} <= Cl(α5)={a5['cl_medio_ventana']:.3f}")
    return fallos


def cmd_sa_polar():
    import numpy as np
    alphas, cls = [], []
    for nombre, a in (("sa_a0", 0), ("sa_a2", 2), ("sa_a5", 5), ("sa_a8", 8)):
        r = _load(nombre)
        alphas.append(a)
        cls.append(r["cl_medio_ventana"])
    fallos = []
    if not all(cls[i] < cls[i + 1] for i in range(len(cls) - 1)):
        fallos.append(f"Cl(α) no monótona: {[f'{c:.3f}' for c in cls]}")
    pend = float(np.polyfit(alphas, cls, 1)[0])
    print(f"\n[sa_polar] Cl(α) = {list(zip(alphas, [round(c, 3) for c in cls]))}")
    print(f"[sa_polar] pendiente = {pend:.4f} /deg (teórico 2π·0.9 ≈ 0.099)")
    if not (0.06 <= pend <= 0.12):
        fallos.append(f"pendiente {pend:.4f} fuera de [0.06, 0.12]/deg")
    return fallos


def cmd_custom():
    # custom <nombre> key=val ...  (val evaluado como literal python)
    import ast
    nombre = sys.argv[2]
    overrides = {}
    for kv in sys.argv[3:]:
        k, v = kv.split("=", 1)
        try:
            overrides[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            overrides[k] = v
    _run_sim(nombre, **overrides)
    return []


CMDS = {
    "smoke": cmd_smoke,
    "smoke_polygon": cmd_smoke_polygon,
    "baseline_a5": cmd_baseline_a5,
    "polygon_a5": cmd_polygon_a5,
    "compare": cmd_compare,
    "regression": cmd_regression,
    "custom": cmd_custom,
    "sa_sanity": cmd_sa_sanity,
    "sa_a0": cmd_sa_a0,
    "sa_a5": cmd_sa_a5,
    "sa_a2": cmd_sa_a2,
    "sa_a8": cmd_sa_a8,
    "sa_polar": cmd_sa_polar,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print(f"Uso: run_kutta_tests.py {{{'|'.join(CMDS)}}} [nombre key=val ...]")
        sys.exit(2)
    fallos = CMDS[sys.argv[1]]()
    if fallos:
        print(f"\n[{sys.argv[1]}] FALLOS:")
        for f in fallos:
            print(f"  - {f}")
        sys.exit(1)
    print(f"\n[{sys.argv[1]}] OK")
    sys.exit(0)

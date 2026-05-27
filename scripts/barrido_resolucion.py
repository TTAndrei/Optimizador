"""
Barrido alpha × resolucion: perfil y modo configurables.

Genera una linea por resolucion en las curvas Cl, Cd y Ef vs alpha.

Uso rapido:
    python scripts/barrido_resolucion.py

Parametros editables en la seccion "Configuracion" mas abajo.

CHECKPOINT AUTONOMO
  Lee barrido_resolucion_resultados.json al arrancar y saltea
  las corridas ya completadas. Crash-safe: guarda tras cada corrida.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import csv

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR    = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _ROOT_DIR)
from Simulador2D import main as sim_main

# ─── Configuracion ────────────────────────────────────────────────────────────
ALPHAS      = list(range(-10, 11))          # -10..10 deg
ITER        = int(os.environ.get("ITER", "2000"))
DESCARTAR_FRAC = 0.30                        # fraccion inicial descartada al promediar

RESOLUCIONES = [0.001, 0.0005]                 # <-- edita aqui las resoluciones

# Modo multigrid a usar (uno solo)
MODO = dict(mg_modo_turbo=False, mg_modo_turbo_hd=True, mg_modo_turbo_ultra=False)
MODO_LABEL = "Turbo (L0)"

BASE_CFG = dict(
    filepath="profiles/NACA_0012", chord=1.0,
    Lx=12.0, Ly=8.0, cx=2.0,
    factor_expansion=1.10,
    ancho_zona_fina_x=1.2, ancho_zona_fina_y=1.0,
    v0x=1.0, v0y=0.0, rho=1.0, p0=0.0,
    nu=1e-5,
    CFL=0.5, iteraciones=ITER, guardado=50,
    divergencia=0.10,
    usar_wale=False, wale_Cw=0.1,
    graficos=False, save_frames=False, live_view=False,
    mostrar_malla=False, stop_on_convergence=False,
    corregir_deriva_vertical=False,
)
 
BASE        = Path(__file__).parent
OUT_RESULTS = BASE / "barrido_resolucion_resultados2.json"
OUT_PLAN    = BASE / "barrido_resolucion_plan2.json"
OUT_CSV     = BASE / "barrido_resolucion2.csv"

# Estilo: una linea por resolucion
_PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
DX_COLORS  = {dx: _PALETTE[i % len(_PALETTE)] for i, dx in enumerate(RESOLUCIONES)}
DX_STYLES  = {dx: (("-", "--", "-.", ":")[i % 4], ("o", "s", "^", "D")[i % 4])
              for i, dx in enumerate(RESOLUCIONES)}

# ─── Plan ─────────────────────────────────────────────────────────────────────
PLAN: list[dict] = [
    {"dx_min": dx, "alpha": float(alpha)}
    for dx in RESOLUCIONES
    for alpha in ALPHAS
]


def plan_key(dx: float, alpha: float) -> str:
    return f"{dx}|{alpha:+.1f}"


# ─── Checkpoint ───────────────────────────────────────────────────────────────
def load_results() -> dict[str, dict]:
    if not OUT_RESULTS.exists():
        return {}
    with open(OUT_RESULTS, encoding="utf-8") as f:
        rows = json.load(f)
    return {plan_key(r["dx_min"], r["alpha"]): r for r in rows}


def save_results(done: dict[str, dict]) -> None:
    ordered = [done[plan_key(s["dx_min"], s["alpha"])]
               for s in PLAN
               if plan_key(s["dx_min"], s["alpha"]) in done]
    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)


def save_plan_status(done: dict[str, dict]) -> None:
    entries = []
    for i, step in enumerate(PLAN):
        k = plan_key(step["dx_min"], step["alpha"])
        status = "done" if k in done else "pending"
        entry = {"n": i + 1, "status": status, **step}
        if status == "done" and "error" not in done[k]:
            r = done[k]
            entry["Cl"] = r.get("Cl_mean")
            entry["Cd"] = r.get("Cd_mean")
        entries.append(entry)
    n_done = sum(1 for e in entries if e["status"] == "done")
    payload = {"total": len(PLAN), "done": n_done,
               "pending": len(PLAN) - n_done, "steps": entries}
    with open(OUT_PLAN, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ─── Simulacion ───────────────────────────────────────────────────────────────
def run_step(step: dict) -> dict:
    dx, alpha = step["dx_min"], step["alpha"]
    cfg = {**BASE_CFG, "dx_min": dx, "alpha_deg": alpha, **MODO}

    t0 = time.time()
    mesh = sim_main(**cfg)
    elapsed = time.time() - t0

    def _resumen(vec_gpu):
        v = cp.asnumpy(vec_gpu).astype(float).ravel()
        if len(v) == 0:
            return float("nan"), float("nan"), float("nan")
        i0 = min(max(int(len(v) * DESCARTAR_FRAC), 0), len(v) - 1)
        tr = v[i0:]
        return float(np.mean(tr)), float(v[-1]), float(np.std(tr))

    cd_mean, cd_final, cd_std = _resumen(mesh.cdvector)
    cl_mean, cl_final, cl_std = _resumen(mesh.clvector)
    ef_mean  = cl_mean  / cd_mean  if abs(cd_mean)  > 1e-12 else float("nan")
    ef_final = cl_final / cd_final if abs(cd_final) > 1e-12 else float("nan")

    mu = BASE_CFG["rho"] * BASE_CFG["nu"]
    q  = BASE_CFG["rho"] * BASE_CFG["v0x"] ** 2 * BASE_CFG["chord"]
    forces = mesh.compute_drag_lift(mu, rho=BASE_CFG["rho"], n_extrap_layers=5)

    row = dict(
        dx_min=dx, alpha=alpha,
        nx=int(mesh.nx), ny=int(mesh.ny),
        Cl_mean=cl_mean, Cl_final=cl_final, Cl_std=cl_std,
        Cd_mean=cd_mean, Cd_final=cd_final, Cd_std=cd_std,
        Ef_mean=ef_mean, Ef_final=ef_final,
        Cl_p=float(2.0 * forces["Lift_p"] / q),
        Cl_v=float(2.0 * forces["Lift_v"] / q),
        Cd_p=float(2.0 * forces["Drag_p"] / q),
        Cd_v=float(2.0 * forces["Drag_v"] / q),
        elapsed_s=elapsed,
        its_per_s=float(ITER / elapsed) if elapsed > 0 else 0.0,
    )

    try:
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from bl_correction import compute_corrected_forces
        bl = compute_corrected_forces(mesh, filepath=BASE_CFG["filepath"],
                                      alpha_deg=float(alpha))
        row.update({
            "Cl_bl":         float(bl["Cl"]),
            "Cd_bl":         float(bl["Cd"]),
            "Ef_bl":         float(bl["Ef"]),
        })
    except Exception as _e:
        row.update({"Cl_bl": float("nan"), "Cd_bl": float("nan"),
                    "Ef_bl": float("nan")})
        print(f"  [BL correction failed: {_e}]")

    del mesh
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return row


# ─── Bucle principal ──────────────────────────────────────────────────────────
def run_all() -> dict[str, dict]:
    done = load_results()
    pending = [s for s in PLAN if plan_key(s["dx_min"], s["alpha"]) not in done]
    n_total = len(PLAN)
    n_done  = n_total - len(pending)

    print(f"\n{'='*60}")
    print(f"  Plan: {n_total} corridas  |  Completadas: {n_done}  |  Pendientes: {len(pending)}")
    print(f"  ITER={ITER}  |  Modo: {MODO_LABEL}  |  checkpoint: {OUT_RESULTS.name}")
    print(f"{'='*60}\n")

    if not pending:
        print("  Todas las corridas ya completadas.")
        return done

    nxt = pending[0]
    print(f"  Reanudando en: dx={nxt['dx_min']}  alpha={nxt['alpha']:+.0f}°\n")

    for i, step in enumerate(pending):
        dx, alpha = step["dx_min"], step["alpha"]
        k = plan_key(dx, alpha)
        global_n = n_done + i + 1

        print(f"[{global_n}/{n_total}] dx={dx}  alpha={alpha:+.0f}°  ", end="", flush=True)
        try:
            row = run_step(step)
            done[k] = row
            print(f"Cl={row['Cl_mean']:+.4f}  Cd={row['Cd_mean']:.4f}  "
                  f"Ef={row['Ef_mean']:.3f}  ({row['elapsed_s']:.0f}s)", flush=True)
        except Exception as exc:
            import traceback
            done[k] = {**step, "error": str(exc)}
            print(f"ERROR: {exc}", flush=True)
            traceback.print_exc()

        save_results(done)
        save_plan_status(done)

    return done


# ─── CSV ─────────────────────────────────────────────────────────────────────
def save_csv(done: dict[str, dict]) -> None:
    rows_ok = [r for r in done.values() if "error" not in r]
    if not rows_ok:
        return
    rows_ok.sort(key=lambda r: (r["dx_min"], r["alpha"]))

    # columnas fijas primero, luego el resto en orden alfabetico
    fixed = ["dx_min", "alpha", "nx", "ny",
             "Cl_mean", "Cl_final", "Cl_std",
             "Cd_mean", "Cd_final", "Cd_std",
             "Ef_mean", "Ef_final",
             "Cl_p", "Cl_v", "Cd_p", "Cd_v",
             "Cl_bl", "Cd_bl", "Ef_bl",
             "elapsed_s", "its_per_s"]
    extra = sorted(k for k in rows_ok[0] if k not in fixed)
    fieldnames = [f for f in fixed if f in rows_ok[0]] + extra

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_ok)
    print(f"[save] {OUT_CSV.name}  ({len(rows_ok)} filas)")


# ─── Extraccion de curvas ─────────────────────────────────────────────────────
def get_curve(done: dict, dx: float, key: str) -> tuple[np.ndarray, np.ndarray]:
    rows = [done[plan_key(dx, float(a))]
            for a in ALPHAS
            if plan_key(dx, float(a)) in done
            and "error" not in done[plan_key(dx, float(a))]]
    rows.sort(key=lambda r: r["alpha"])
    return (np.array([r["alpha"] for r in rows]),
            np.array([r[key]     for r in rows]))


# ─── Figuras ─────────────────────────────────────────────────────────────────
def generate_figures(done: dict) -> None:
    n_ok = sum(1 for r in done.values() if "error" not in r)
    if n_ok == 0:
        print("[figs] Sin datos suficientes para graficar.")
        return
    print(f"\n[figs] Generando graficas ({n_ok} corridas ok)...")

    metrics = [
        ("Cl_mean", "Cl",     "Coeficiente de sustentacion"),
        ("Cd_mean", "Cd",     "Coeficiente de resistencia"),
        ("Ef_mean", "Cl/Cd",  "Eficiencia aerodinamica"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titulo = f"Convergencia de malla — {MODO_LABEL}  (Re={int(1/BASE_CFG['nu']):.0e}, {BASE_CFG['filepath']})"
    fig.suptitle(titulo, fontsize=12)

    for ax, (key, ylabel, title) in zip(axes, metrics):
        for dx in RESOLUCIONES:
            ls, mk = DX_STYLES[dx]
            a, v = get_curve(done, dx, key)
            if len(a) == 0:
                continue
            ax.plot(a, v, color=DX_COLORS[dx], linestyle=ls, marker=mk,
                    linewidth=2, markersize=5, label=f"dx={dx}")
        ax.set_title(title)
        ax.set_xlabel("α [°]")
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="k", linewidth=0.5, linestyle=":")
        ax.axvline(0, color="k", linewidth=0.5, linestyle=":")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

    fig.tight_layout()
    out = BASE / "barrido_resolucion_curvas.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[save] {out.name}")

    # Polar Cl vs Cd
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f"Polar Cl vs Cd — {MODO_LABEL}", fontsize=12)
    for dx in RESOLUCIONES:
        ls, mk = DX_STYLES[dx]
        _, cd = get_curve(done, dx, "Cd_mean")
        _, cl = get_curve(done, dx, "Cl_mean")
        if len(cd) == 0:
            continue
        ax.plot(cd, cl, color=DX_COLORS[dx], linestyle=ls, marker=mk,
                linewidth=2, markersize=5, label=f"dx={dx}")
    ax.set_xlabel("Cd")
    ax.set_ylabel("Cl")
    ax.axhline(0, color="k", linewidth=0.5, linestyle=":")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = BASE / "barrido_resolucion_polar.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[save] {out.name}")


# ─── Entry point ─────────────────────────────────────────────────────────────
def main() -> None:
    n = len(RESOLUCIONES) * len(ALPHAS)
    print(f"Barrido resolucion: {n} corridas  |  "
          f"{len(RESOLUCIONES)} resoluciones × {len(ALPHAS)} alphas  |  "
          f"Modo: {MODO_LABEL}  |  ITER={ITER}")

    done = run_all()
    save_csv(done)
    generate_figures(done)

    n_ok  = sum(1 for r in done.values() if "error" not in r)
    n_err = sum(1 for r in done.values() if "error"     in r)
    print(f"\n{'='*60}")
    print(f"  Completado: {n_ok} ok  |  {n_err} errores")
    print(f"  Resultados: {OUT_RESULTS}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

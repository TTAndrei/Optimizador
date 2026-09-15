"""
Barrido alpha × modo_turbo × resolucion: NACA_0012 a Re=100k.

Alpha : -10 a 10 deg, paso 1 (21 puntos).
Modos : mg_modo_turbo  /  mg_modo_turbo_hd  /  mg_modo_turbo_ultra.
Grids : dx_min =  0.002 / 0.001. /0.0005 /

Total : 3 × 3 × 21 = 189 corridas.

CHECKPOINT AUTONOMO
  Al arrancar lee barrido_modos_resultados.json (si existe) y
  detecta automaticamente que corridas ya estan hechas.
  Solo ejecuta las pendientes, en el mismo orden siempre.
  Tras cada corrida guarda inmediatamente → crash-safe.

Orden fijo:
  for modo in [turbo, turbo_hd, turbo_ultra]:
    for dx_min in [0.002, 0.001, 0.0005]:
      for alpha in [-10, -9, ..., 10]:

Salida:
  barrido_modos_resultados.json     base de verdad + checkpoint
  barrido_modos_plan.json           plan completo con estado (solo lectura)
  barrido_modos_Cl/Cd/Ef.png        comparacion modos  (cols = resolucion)
  barrido_convergencia_{modo}.png   grid study por modo
  barrido_grid_Cl/Cd.png            grid 3x3 completo
  barrido_polar_Cl_Cd.png           polar todos los combos
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-cache"))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT_DIR))
import matplotlib.pyplot as plt
from Simulador2D import main as sim_main
from sim_defaults import PROJECTION_DEFAULTS, add_force_consistent_metrics

# ─── Configuracion ────────────────────────────────────────────────────────────
ALPHAS      = list(range(-10, 11))
ITER        = int(os.environ.get("ITER", "2000"))
DESCARTAR_FRAC = 0.30

MODOS = {
    "turbo":       dict(mg_modo_turbo=True,  mg_modo_turbo_hd=False, mg_modo_turbo_ultra=False),
    "turbo_hd":    dict(mg_modo_turbo=False, mg_modo_turbo_hd=True,  mg_modo_turbo_ultra=False),
    "turbo_ultra": dict(mg_modo_turbo=False, mg_modo_turbo_hd=False, mg_modo_turbo_ultra=True),
}
RESOLUCIONES = [ 0.002, 0.001, 0.0005]

BASE_CFG = dict(
    filepath="profiles/NACA_0012", chord=1.0,
    Lx=12.0, Ly=8.0, cx=2.0,
    factor_expansion=1.10,
    ancho_zona_fina_x=1.2, ancho_zona_fina_y=1.0,
    v0x=1.0, v0y=0.0, rho=1.0, p0=0.0,
    nu=1e-5,
    CFL=0.5, iteraciones=ITER, guardado=50,
    divergencia=0.10,
    usar_wale=True, wale_Cw=0.15,
    graficos=False, save_frames=False, live_view=False,
    mostrar_malla=False, stop_on_convergence=False,
    corregir_deriva_vertical=False,
    **PROJECTION_DEFAULTS,
)

BASE = ROOT_DIR / "results" / "barridos" / "barrido_modos_outer_sum"
BASE.mkdir(parents=True, exist_ok=True)
OUT_RESULTS = BASE / "summary.json"
OUT_PLAN    = BASE / "plan.json"

# ─── Estilo grafico ───────────────────────────────────────────────────────────
MODO_COLORS = {"turbo": "#2196F3", "turbo_hd": "#FF5722", "turbo_ultra": "#4CAF50"}
MODO_LABELS = {"turbo": "Turbo (L0)", "turbo_hd": "Turbo-HD (L1)", "turbo_ultra": "Turbo-Ultra (L2)"}
DX_STYLES   = {0.002: ("--", "^"), 0.001: ("-.", "s"), 0.0005: ("-", "o")}

# ─── Plan fijo de ejecucion ───────────────────────────────────────────────────
PLAN: list[dict] = [
    {"modo": modo, "dx_min": dx, "alpha": float(alpha)}
    for modo in MODOS
    for dx in RESOLUCIONES
    for alpha in ALPHAS
]


def plan_key(modo: str, dx: float, alpha: float) -> str:
    return f"{modo}|{dx}|{alpha:+.1f}"


# ─── Checkpoint: lectura y escritura ─────────────────────────────────────────
def load_results() -> dict[str, dict]:
    """Carga barrido_modos_resultados.json y devuelve {plan_key: row}."""
    if not OUT_RESULTS.exists():
        return {}
    with open(OUT_RESULTS, encoding="utf-8") as f:
        rows = json.load(f)
    return {plan_key(r["modo"], r["dx_min"], r["alpha"]): r for r in rows}


def save_results(done: dict[str, dict]) -> None:
    """Reescribe el JSON completo en el orden del plan."""
    ordered = [done[plan_key(s["modo"], s["dx_min"], s["alpha"])]
               for s in PLAN
               if plan_key(s["modo"], s["dx_min"], s["alpha"]) in done]
    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)


def save_plan_status(done: dict[str, dict]) -> None:
    """Escribe plan legible con status done/pending para inspección."""
    entries = []
    for i, step in enumerate(PLAN):
        k = plan_key(step["modo"], step["dx_min"], step["alpha"])
        status = "done" if k in done else "pending"
        entry = {"n": i + 1, "status": status, **step}
        if status == "done" and "error" not in done[k]:
            r = done[k]
            entry["Cl"] = r.get("Cl_from_Cp_force_consistent", r.get("Cl_mean"))
            entry["Cd"] = r.get("Cd_mean")
        entries.append(entry)
    n_done = sum(1 for e in entries if e["status"] == "done")
    payload = {
        "total": len(PLAN),
        "done": n_done,
        "pending": len(PLAN) - n_done,
        "steps": entries,
    }
    with open(OUT_PLAN, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ─── Simulacion ───────────────────────────────────────────────────────────────
def run_step(step: dict) -> dict:
    modo, dx, alpha = step["modo"], step["dx_min"], step["alpha"]
    cfg = {**BASE_CFG, "dx_min": dx, "alpha_deg": alpha, **MODOS[modo]}

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
        modo=modo, dx_min=dx, alpha=alpha,
        nx=int(mesh.nx), ny=int(mesh.ny),
        projection_variant=cfg["projection_variant"],
        mg_pressure_accumulation=cfg["mg_pressure_accumulation"],
        wall_pressure_gradient_mode=cfg["wall_pressure_gradient_mode"],
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
    add_force_consistent_metrics(row, mesh, cfg)

    try:
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from bl_correction import compute_corrected_forces
        bl = compute_corrected_forces(mesh, filepath=BASE_CFG["filepath"],
                                      alpha_deg=float(alpha),
                                      Cd_p_source="both")
        row.update({
            "Cl_bl":         float(bl["Cl"]),
            "Cd_bl":         float(bl["Cd"]),
            "Cd_bl_geom":    float(bl.get("Cd_bl_geom", float("nan"))),
            "Cd_p_bl":       float(bl["Cd_p"]),
            "Cd_p_geom":     float(bl.get("Cd_p_geom", float("nan"))),
            "Cd_visc_bl":    float(bl["Cd_visc"]),
            "Ef_bl":         float(bl["Ef"]),
            "trans_x_upper": float(bl["trans_x_upper"]),
            "trans_x_lower": float(bl["trans_x_lower"]),
        })
    except Exception as _e:
        row.update({"Cl_bl": float("nan"), "Cd_bl": float("nan"),
                    "Cd_p_bl": float("nan"), "Cd_visc_bl": float("nan"),
                    "Ef_bl": float("nan")})
        print(f"  [BL correction failed: {_e}]")

    del mesh
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return row


# ─── Bucle principal ──────────────────────────────────────────────────────────
def run_all() -> dict[str, dict]:
    done = load_results()

    pending = [s for s in PLAN if plan_key(s["modo"], s["dx_min"], s["alpha"]) not in done]
    n_total  = len(PLAN)
    n_done   = n_total - len(pending)

    print(f"\n{'='*60}")
    print(f"  Plan: {n_total} corridas  |  Completadas: {n_done}  |  Pendientes: {len(pending)}")
    print(f"  ITER={ITER}  |  checkpoint: {OUT_RESULTS.name}")
    print(f"{'='*60}\n")

    if not pending:
        print("  Todas las corridas ya completadas.")
        return done

    # Mostrar siguiente corrida
    nxt = pending[0]
    print(f"  Reanudando en: modo={nxt['modo']}  dx={nxt['dx_min']}  alpha={nxt['alpha']:+.0f}°\n")

    for i, step in enumerate(pending):
        modo, dx, alpha = step["modo"], step["dx_min"], step["alpha"]
        k = plan_key(modo, dx, alpha)
        global_n = n_done + i + 1

        print(f"[{global_n}/{n_total}] modo={modo}  dx={dx}  alpha={alpha:+.0f}°  ",
              end="", flush=True)
        try:
            row = run_step(step)
            done[k] = row
            print(f"Cl={row.get('Cl_from_Cp_force_consistent', row['Cl_mean']):+.4f}  Cd={row['Cd_mean']:.4f}  "
                  f"Ef={row['Ef_mean']:.3f}  ({row['elapsed_s']:.0f}s)", flush=True)
        except Exception as exc:
            import traceback
            done[k] = {**step, "error": str(exc)}
            print(f"ERROR: {exc}", flush=True)
            traceback.print_exc()

        save_results(done)
        save_plan_status(done)

    return done


# ─── Extraccion de curvas ─────────────────────────────────────────────────────
def get_curve(done: dict[str, dict], modo: str, dx: float,
              key: str) -> tuple[np.ndarray, np.ndarray]:
    rows = [done[plan_key(modo, dx, float(a))]
            for a in ALPHAS
            if plan_key(modo, dx, float(a)) in done
            and "error" not in done[plan_key(modo, dx, float(a))]]
    rows.sort(key=lambda r: r["alpha"])
    alphas = np.array([r["alpha"] for r in rows])
    vals   = np.array([r[key]    for r in rows])
    return alphas, vals


# ─── Figuras ─────────────────────────────────────────────────────────────────
def fig_comparacion_modos(done: dict, metric: str) -> None:
    """1×3 subplots (cols=dx_min), 3 lineas=modos."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(f"{metric} vs α  —  comparacion modos  (Re=100k, NACA 0012)", fontsize=13)

    for col, dx in enumerate(RESOLUCIONES):
        ax = axes[col]
        for modo in MODOS:
            key = "Cl_from_Cp_force_consistent" if metric == "Cl" else f"{metric}_mean"
            a, v = get_curve(done, modo, dx, key)
            if len(a) == 0:
                continue
            ax.plot(a, v, color=MODO_COLORS[modo], marker="o",
                    linewidth=2, markersize=5, label=MODO_LABELS[modo])
        ax.set_title(f"dx_min = {dx}")
        ax.set_xlabel("α [°]")
        ax.axhline(0, color="k", linewidth=0.5, linestyle=":")
        ax.axvline(0, color="k", linewidth=0.5, linestyle=":")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    axes[0].set_ylabel(metric)
    fig.tight_layout()
    out = BASE / f"barrido_modos_{metric}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[save] {out.name}")


def fig_convergencia_malla(done: dict, modo: str) -> None:
    """1×3 subplots (Cl/Cd/Ef), 3 lineas=dx_min."""
    metrics = [("Cl_from_Cp_force_consistent", "Cl"), ("Cd_mean", "Cd"), ("Ef_mean", "Cl/Cd")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Convergencia malla — {MODO_LABELS[modo]}  (Re=100k, NACA 0012)", fontsize=13)

    for col, (key, label) in enumerate(metrics):
        ax = axes[col]
        for dx in RESOLUCIONES:
            ls, mk = DX_STYLES[dx]
            a, v = get_curve(done, modo, dx, key)
            if len(a) == 0:
                continue
            ax.plot(a, v, linestyle=ls, marker=mk, linewidth=2,
                    markersize=5, label=f"dx={dx}")
        ax.set_title(label)
        ax.set_xlabel("α [°]")
        ax.axhline(0, color="k", linewidth=0.5, linestyle=":")
        ax.axvline(0, color="k", linewidth=0.5, linestyle=":")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = BASE / f"barrido_convergencia_{modo}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[save] {out.name}")


def fig_resumen_grid(done: dict, metric: str) -> None:
    """3×3 grid: rows=modos, cols=dx_min."""
    modos_list = list(MODOS.keys())
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True)
    fig.suptitle(f"{metric} vs α — grid completo  (Re=100k, NACA 0012)", fontsize=13)

    for ri, modo in enumerate(modos_list):
        for ci, dx in enumerate(RESOLUCIONES):
            ax = axes[ri][ci]
            key = "Cl_from_Cp_force_consistent" if metric == "Cl" else f"{metric}_mean"
            a, v = get_curve(done, modo, dx, key)
            if len(a):
                ax.plot(a, v, color=MODO_COLORS[modo], linewidth=2,
                        marker="o", markersize=3)
            ax.axhline(0, color="k", linewidth=0.5, linestyle=":")
            ax.axvline(0, color="k", linewidth=0.5, linestyle=":")
            ax.grid(True, alpha=0.25)
            if ri == 0:
                ax.set_title(f"dx={dx}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"{MODO_LABELS[modo]}\n{metric}", fontsize=8)
            if ri == 2:
                ax.set_xlabel("α [°]", fontsize=9)

    fig.tight_layout()
    out = BASE / f"barrido_grid_{metric}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[save] {out.name}")


def fig_velocidad(done: dict) -> None:
    """it/s vs alpha: 2 subplots.
    Izquierda: lineas por modo (cols fijado en dx mediano).
    Derecha:   lineas por dx_min (modo fijado en turbo_hd).
    Tabla resumen: media de it/s por (modo, dx).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Velocidad de simulacion (it/s)  —  Re=100k, NACA 0012", fontsize=13)

    # Izquierda: todos los modos a dx=0.002
    dx_ref = 0.002
    ax = axes[0]
    for modo in MODOS:
        a, v = get_curve(done, modo, dx_ref, "its_per_s")
        if len(a) == 0:
            continue
        ax.plot(a, v, color=MODO_COLORS[modo], marker="o", linewidth=2,
                markersize=5, label=MODO_LABELS[modo])
    ax.set_title(f"Modos comparados  (dx={dx_ref})")
    ax.set_xlabel("α [°]")
    ax.set_ylabel("it/s")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    # Derecha: todas las resoluciones en turbo_hd
    modo_ref = "turbo_hd"
    ax = axes[1]
    for dx in RESOLUCIONES:
        ls, mk = DX_STYLES[dx]
        a, v = get_curve(done, modo_ref, dx, "its_per_s")
        if len(a) == 0:
            continue
        ax.plot(a, v, linestyle=ls, marker=mk, linewidth=2,
                markersize=5, label=f"dx={dx}")
    ax.set_title(f"Resoluciones comparadas  ({MODO_LABELS[modo_ref]})")
    ax.set_xlabel("α [°]")
    ax.set_ylabel("it/s")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = BASE / "barrido_velocidad.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[save] {out.name}")

    # Tabla resumen it/s media en consola
    print("\n  it/s media por (modo, dx_min):")
    print(f"  {'modo':<14}", end="")
    for dx in RESOLUCIONES:
        print(f"  dx={dx:<6}", end="")
    print()
    for modo in MODOS:
        print(f"  {MODO_LABELS[modo]:<14}", end="")
        for dx in RESOLUCIONES:
            _, v = get_curve(done, modo, dx, "its_per_s")
            if len(v):
                print(f"  {np.mean(v):>8.2f}", end="")
            else:
                print(f"  {'---':>8}", end="")
        print()


def fig_polar(done: dict) -> None:
    """Polar Cl vs Cd — todos los 9 combos."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("Polar Cl vs Cd — todos los modos y resoluciones  (Re=100k, NACA 0012)", fontsize=12)

    for modo in MODOS:
        for dx in RESOLUCIONES:
            ls, mk = DX_STYLES[dx]
            _, cd = get_curve(done, modo, dx, "Cd_mean")
            _, cl = get_curve(done, modo, dx, "Cl_from_Cp_force_consistent")
            if len(cd) == 0:
                continue
            ax.plot(cd, cl, color=MODO_COLORS[modo], linestyle=ls, marker=mk,
                    linewidth=1.5, markersize=4,
                    label=f"{MODO_LABELS[modo]}  dx={dx}")

    ax.set_xlabel("Cd")
    ax.set_ylabel("Cl")
    ax.axhline(0, color="k", linewidth=0.5, linestyle=":")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = BASE / "barrido_polar_Cl_Cd.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[save] {out.name}")


def generate_figures(done: dict) -> None:
    n_ok = sum(1 for r in done.values() if "error" not in r)
    if n_ok == 0:
        print("[figs] Sin datos suficientes para graficar.")
        return
    print(f"\n[figs] Generando graficas ({n_ok} corridas ok)...")
    for metric in ("Cl", "Cd", "Ef"):
        fig_comparacion_modos(done, metric)
    for modo in MODOS:
        fig_convergencia_malla(done, modo)
    for metric in ("Cl", "Cd"):
        fig_resumen_grid(done, metric)
    fig_polar(done)
    fig_velocidad(done)


# ─── Entry point ─────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Barrido: {len(PLAN)} corridas  |  {len(MODOS)} modos × "
          f"{len(RESOLUCIONES)} resoluciones × {len(ALPHAS)} alphas  |  ITER={ITER}")

    done = run_all()
    generate_figures(done)

    n_ok  = sum(1 for r in done.values() if "error" not in r)
    n_err = sum(1 for r in done.values() if "error"     in r)
    print(f"\n{'='*60}")
    print(f"  Completado: {n_ok} ok  |  {n_err} errores")
    print(f"  Resultados: {OUT_RESULTS}")
    print(f"  Plan/estado: {OUT_PLAN}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

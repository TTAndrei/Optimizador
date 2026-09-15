"""
Sweep de caracterización del simulador para el optimizador genético:
    alpha ∈ {0, 2, 5, 8, 10} × CFL ∈ {0.25, 0.5, 0.75} × dx ∈ {0.004, 0.002}
    (30 simulaciones, config de referencia: consistent + SA + MacCormack)

Uso (desde la raíz del repo, SIEMPRE con el venv):
    .venv/bin/python scripts/agent_tests/run_cfl_sweep.py           # corre lo pendiente
    .venv/bin/python scripts/agent_tests/run_cfl_sweep.py --list    # estado de la cola
    .venv/bin/python scripts/agent_tests/run_cfl_sweep.py --summary # solo CSV+gráficas

REANUDABLE: cada sim escribe results/cfl_sweep/<nombre>.json al terminar con éxito;
al relanzar el script se saltan las que ya tienen JSON. Si una sim crashea se escribe
<nombre>.failed.json (con traceback) y se continúa con la siguiente; los fallos se
reintentan en el siguiente lanzamiento.

Por simulación se guarda:
  - <nombre>.json      : params + Cl/Cd (media/std ventana 20%), iteración y tiempo
                         físico de convergencia de Cl y Cd, it/s, Q_lazo, ΔCp_TE,
                         Cp_min, x_succión, chi_max...
  - <nombre>.npz       : estado completo (save_state: campos u/v/p + clvector/cdvector)
  - frames/<nombre>/   : frame_XXXXXX.png del campo |u| cada 1000 iteraciones (zoom perfil)
  - frames/<nombre>_streamlines.png : streamlines finales (diagnóstico Kutta/separación)
  - frames/<nombre>_series.png      : evolución temporal Cl/Cd con la convergencia marcada

Al completarse la cola (o con --summary) se genera:
  - summary.csv                 : tabla completa
  - summary_polar.png           : Cl(α) y Cd(α) por cada (dx, CFL)
  - summary_convergencia.png    : t_físico e iteraciones de convergencia vs CFL
  - summary_coste.png           : it/s y coste wall-clock por sim; error de Cl vs
                                  referencia (dx=0.002, CFL=0.25)
"""
import json
import os
import sys
import time
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT_DIR = os.path.join(ROOT, "results", "cfl_sweep")
FRAMES_DIR = os.path.join(OUT_DIR, "frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

ALPHAS = [0, 2, 5, 8, 10]
CFLS = [0.25, 0.5, 0.75]
DXS = [0.004, 0.002]   # baratas primero

T_TARGET = 12.0   # tiempos convectivos objetivo por sim
U_FAC = 1.6       # dt ≈ CFL·dx/U_FAC (medido: 1.56 en wc_mc_sa_a5_dx2)
GUARDADO = 50
FRAME_CADA = 1000

BASE = dict(
    Lx=8, Ly=5, cx=2, chord=1.0, v0x=1, v0y=0, rho=1.0, nu=1e-5,
    ancho_zona_fina_x=1.5, ancho_zona_fina_y=1.0, factor_expansion=1.1,
    filepath="profiles/NACA_0012_sharp", min_te_height_factor=1.0,
    wake_refinement_mode="long_fine_x",
    turb_model="sa", wall_treatment="consistent", advection_scheme="maccormack",
    divergencia=0.02, mg_max_outer=8, mg_niveles_max=2, mg_modo_turbo_hd=False,
    guardado=GUARDADO, graficos=False, live_view=False, mostrar_malla=False,
    stop_on_convergence=False, save_frames=True, save_frames_cada=FRAME_CADA,
    save_frame_dpi=150,
)


def sim_name(alpha, cfl, dx):
    return f"a{alpha}_cfl{int(cfl * 100):03d}_dx{int(dx * 1000)}"


def iters_for(cfl, dx):
    """Iteraciones para alcanzar T_TARGET tiempos convectivos (redondeo a 1000)."""
    n = T_TARGET * U_FAC / (cfl * dx)
    return int(-(-n // 1000) * 1000)


def build_queue():
    q = []
    for dx in DXS:
        for alpha in ALPHAS:
            for cfl in CFLS:
                q.append((alpha, cfl, dx))
    return q


def _loop_flux(mesh, params, margin):
    """Q = ∮u·n dl (saliente) en rectángulo alrededor del perfil. Debe →0."""
    import cupy as cp
    import numpy as np

    X = cp.asnumpy(mesh.X_1d).astype(float)
    Y = cp.asnumpy(mesh.Y_1d).astype(float)
    u = cp.asnumpy(mesh.u).astype(float)
    v = cp.asnumpy(mesh.v).astype(float)
    cx = float(params["cx"])
    chord = float(params["chord"])
    cy = float(params.get("cy") or params["Ly"] / 2.0)
    x0, x1 = cx - margin, cx + chord + margin
    y0, y1 = cy - (margin + chord / 2), cy + (margin + chord / 2)
    j0, j1 = np.searchsorted(X, x0), np.searchsorted(X, x1)
    i0, i1 = np.searchsorted(Y, y0), np.searchsorted(Y, y1)
    ys, xs = Y[i0:i1 + 1], X[j0:j1 + 1]
    return float(np.trapezoid(u[i0:i1 + 1, j1], ys) - np.trapezoid(u[i0:i1 + 1, j0], ys)
                 + np.trapezoid(v[i1, j0:j1 + 1], xs) - np.trapezoid(v[i0, j0:j1 + 1], xs))


def detect_convergence(serie, t_fisico, tol_abs=0.005, tol_rel=0.02):
    """
    Convergencia = último instante en que la media móvil (ventana ~0.5 conv)
    se sale de la banda ±tol alrededor de la media del último 15% del run.
    Devuelve (idx_muestra, t_conv, iter_conv, converged).
    """
    import numpy as np

    s = np.asarray(serie, dtype=float)
    n = len(s)
    if n < 10 or not np.all(np.isfinite(s)):
        return None, float("nan"), None, False
    ref = float(np.mean(s[-max(1, n * 15 // 100):]))
    tol = max(tol_abs, tol_rel * abs(ref))
    w = max(3, int(n / max(t_fisico, 1e-9) * 0.5))  # muestras en ~0.5 conv
    w = min(w, n // 3)
    kernel = np.ones(w) / w
    sm = np.convolve(s, kernel, mode="valid")       # sm[i] = media s[i:i+w]
    bad = np.where(np.abs(sm - ref) > tol)[0]
    idx = int(bad[-1] + w) if bad.size else 0       # primera ventana ya dentro de banda
    converged = idx <= 0.9 * n
    t_conv = t_fisico * idx / n
    return idx, float(t_conv), int(idx * GUARDADO), bool(converged)


def run_one(alpha, cfl, dx):
    import cupy as cp
    import numpy as np
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from Simulador2D import main

    nombre = sim_name(alpha, cfl, dx)
    frames_dir = os.path.join(FRAMES_DIR, nombre)
    iteraciones = iters_for(cfl, dx)
    params = {**BASE, "alpha_deg": alpha, "CFL": cfl, "dx_min": dx,
              "iteraciones": iteraciones, "frames_dir_refinado": frames_dir,
              "save_frame_refined_xlim": (BASE["cx"] - 0.8, BASE["cx"] + 2.2),
              "save_frame_refined_ylim": (BASE["Ly"] / 2 - 1.2, BASE["Ly"] / 2 + 1.2)}

    print(f"\n{'=' * 72}\n[{nombre}] alpha={alpha} CFL={cfl} dx={dx} iters={iteraciones}\n{'=' * 72}")
    t0 = time.time()
    mesh = main(**params)
    wall = time.time() - t0

    cl = cp.asnumpy(mesh.clvector).astype(float)
    cd = cp.asnumpy(mesh.cdvector).astype(float)
    dcp = cp.asnumpy(mesh.kutta_dcp_vector).astype(float)
    n = len(cl)
    w = max(1, n // 5)
    t_fis = float(getattr(mesh, "_t_fisico", float("nan")))
    dcp_ok = dcp[-w:][np.isfinite(dcp[-w:])]

    _, t_conv_cl, it_conv_cl, conv_cl = detect_convergence(cl, t_fis)
    _, t_conv_cd, it_conv_cd, conv_cd = detect_convergence(cd, t_fis)

    mu = params["rho"] * params["nu"]
    circ = mesh.compute_circulation(loop_margins=(0.25,), verbose=False)
    cpd = mesh.compute_cp_diagnostics(mu, params["rho"], verbose=False) or {}
    chi_max = float("nan")
    if getattr(mesh, "nu_tilde", None) is not None:
        chi_max = float(cp.max(mesh.nu_tilde)) / params["nu"]

    res = {
        "nombre": nombre,
        "alpha_deg": alpha, "CFL": cfl, "dx_min": dx, "iteraciones": iteraciones,
        "wall_s": round(wall, 1),
        "its_per_s": round(iteraciones / wall, 2),
        "t_fisico": t_fis,
        "nan_en_cl": bool(np.any(~np.isfinite(cl))),
        "cl_medio_ventana": float(np.mean(cl[-w:])),
        "cl_std_ventana": float(np.std(cl[-w:])),
        "cd_medio_ventana": float(np.mean(cd[-w:])),
        "cd_std_ventana": float(np.std(cd[-w:])),
        # convergencia (banda ±max(0.005, 2%) sobre media móvil de ~0.5 conv)
        "iter_conv_cl": it_conv_cl, "t_conv_cl": t_conv_cl, "converged_cl": conv_cl,
        "iter_conv_cd": it_conv_cd, "t_conv_cd": t_conv_cd, "converged_cd": conv_cd,
        "dcp_te_medio_ventana": float(np.mean(np.abs(dcp_ok))) if dcp_ok.size else float("nan"),
        "cl_circ_025c": float(circ[0]["cl_circ"]) if circ else float("nan"),
        "q_loop_025c": _loop_flux(mesh, params, 0.25),
        "q_loop_075c": _loop_flux(mesh, params, 0.75),
        "cl_cp": cpd.get("cl_cp", float("nan")),
        "cp_min": cpd.get("cp_min", float("nan")),
        "x_suction": cpd.get("x_suction", float("nan")),
        "chi_max": chi_max,
        "frames_dir": frames_dir,
    }

    # Streamlines finales + serie temporal Cl/Cd
    try:
        mesh.plot_streamlines(show=False,
                              save_path=os.path.join(FRAMES_DIR, f"{nombre}_streamlines.png"))
    except Exception as e:
        print(f"[WARN] streamlines: {e}")
    try:
        t_axis = np.linspace(0, t_fis, n)
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        for ax, serie, lbl, t_c in ((axes[0], cl, "Cl", t_conv_cl),
                                    (axes[1], cd, "Cd", t_conv_cd)):
            ax.plot(t_axis, serie, lw=0.8)
            if np.isfinite(t_c):
                ax.axvline(t_c, color="r", ls="--", lw=1, label=f"convergencia t={t_c:.1f}")
                ax.legend(fontsize=8)
            ax.set_ylabel(lbl)
            ax.grid(alpha=0.3)
        axes[0].set_title(f"{nombre}  Cl={res['cl_medio_ventana']:.3f}  Cd={res['cd_medio_ventana']:.4f}")
        axes[1].set_xlabel("t convectivo")
        fig.savefig(os.path.join(FRAMES_DIR, f"{nombre}_series.png"), dpi=110)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] series: {e}")

    mesh.save_state(os.path.join(OUT_DIR, f"{nombre}.npz"))
    with open(os.path.join(OUT_DIR, f"{nombre}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[{nombre}] OK  Cl={res['cl_medio_ventana']:.3f} Cd={res['cd_medio_ventana']:.4f} "
          f"t_conv_cl={t_conv_cl:.1f} ({res['its_per_s']} it/s, {wall / 60:.0f} min)")

    # Liberar GPU/figuras entre sims
    del mesh
    plt.close("all")
    cp.get_default_memory_pool().free_all_blocks()
    return res


def make_summary():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = []
    for alpha, cfl, dx in build_queue():
        p = os.path.join(OUT_DIR, f"{sim_name(alpha, cfl, dx)}.json")
        if os.path.exists(p):
            with open(p) as f:
                rows.append(json.load(f))
    if not rows:
        print("Sin resultados todavía.")
        return

    cols = ["nombre", "alpha_deg", "CFL", "dx_min", "iteraciones", "wall_s", "its_per_s",
            "t_fisico", "cl_medio_ventana", "cl_std_ventana", "cd_medio_ventana",
            "iter_conv_cl", "t_conv_cl", "converged_cl", "iter_conv_cd", "t_conv_cd",
            "converged_cd", "q_loop_025c", "dcp_te_medio_ventana", "cp_min", "x_suction",
            "chi_max", "nan_en_cl"]
    with open(os.path.join(OUT_DIR, "summary.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    def sel(dx, cfl):
        d = {r["alpha_deg"]: r for r in rows
             if r["dx_min"] == dx and r["CFL"] == cfl}
        a = sorted(d)
        return a, d

    estilos = {0.25: "-o", 0.5: "--s", 0.75: ":^"}
    colores = {0.004: "tab:orange", 0.002: "tab:blue"}

    # Polar Cl(α), Cd(α)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for dx in DXS:
        for cfl in CFLS:
            a, d = sel(dx, cfl)
            if not a:
                continue
            lbl = f"dx={dx} CFL={cfl}"
            ax1.plot(a, [d[x]["cl_medio_ventana"] for x in a], estilos[cfl],
                     color=colores[dx], label=lbl, ms=4)
            ax2.plot(a, [d[x]["cd_medio_ventana"] for x in a], estilos[cfl],
                     color=colores[dx], label=lbl, ms=4)
    ax1.set_xlabel("α (deg)"); ax1.set_ylabel("Cl"); ax1.grid(alpha=0.3)
    ax1.set_title("Polar Cl(α)"); ax1.legend(fontsize=7)
    ax2.set_xlabel("α (deg)"); ax2.set_ylabel("Cd"); ax2.grid(alpha=0.3)
    ax2.set_title("Polar Cd(α)")
    fig.savefig(os.path.join(OUT_DIR, "summary_polar.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Convergencia vs CFL
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for dx in DXS:
        for cfl in CFLS:
            a, d = sel(dx, cfl)
            if not a:
                continue
            ax1.plot(a, [d[x]["t_conv_cl"] for x in a], estilos[cfl],
                     color=colores[dx], label=f"dx={dx} CFL={cfl}", ms=4)
            ax2.plot(a, [d[x]["iter_conv_cl"] for x in a], estilos[cfl],
                     color=colores[dx], ms=4)
    ax1.set_xlabel("α (deg)"); ax1.set_ylabel("t convectivo hasta converger Cl")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=7); ax1.set_title("Tiempo físico de convergencia")
    ax2.set_xlabel("α (deg)"); ax2.set_ylabel("iteraciones hasta converger Cl")
    ax2.grid(alpha=0.3); ax2.set_title("Iteraciones de convergencia")
    fig.savefig(os.path.join(OUT_DIR, "summary_convergencia.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Coste y error vs referencia (dx=0.002, CFL=0.25)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    _, ref = sel(0.002, 0.25)
    for dx in DXS:
        for cfl in CFLS:
            a, d = sel(dx, cfl)
            if not a:
                continue
            lbl = f"dx={dx} CFL={cfl}"
            # coste útil: wall-clock hasta convergencia de Cl
            coste = [d[x]["wall_s"] * (d[x]["iter_conv_cl"] / d[x]["iteraciones"]) / 60
                     for x in a]
            ax1.plot(a, coste, estilos[cfl], color=colores[dx], label=lbl, ms=4)
            if ref:
                comunes = [x for x in a if x in ref]
                err = [d[x]["cl_medio_ventana"] - ref[x]["cl_medio_ventana"] for x in comunes]
                ax2.plot(comunes, err, estilos[cfl], color=colores[dx], label=lbl, ms=4)
    ax1.set_xlabel("α (deg)"); ax1.set_ylabel("min de GPU hasta convergencia de Cl")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=7); ax1.set_title("Coste útil por punto")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("α (deg)"); ax2.set_ylabel("ΔCl vs (dx=0.002, CFL=0.25)")
    ax2.grid(alpha=0.3); ax2.set_title("Error de Cl vs referencia")
    fig.savefig(os.path.join(OUT_DIR, "summary_coste.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Cl simulado (integración de fuerzas) vs Cl proveniente de Cp (∮ΔCp)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex="col")
    for col, dx in enumerate(DXS):
        axp, axd = axes[0, col], axes[1, col]
        for cfl in CFLS:
            a, d = sel(dx, cfl)
            if not a:
                continue
            color = {0.25: "tab:blue", 0.5: "tab:green", 0.75: "tab:red"}[cfl]
            cl_sim = [d[x]["cl_medio_ventana"] for x in a]
            cl_cp = [d[x]["cl_cp"] for x in a]
            axp.plot(a, cl_sim, "-o", color=color, ms=4, label=f"CFL={cfl} simulado")
            axp.plot(a, cl_cp, "--s", color=color, ms=4, mfc="none", label=f"CFL={cfl} Cp")
            axd.plot(a, [s - c for s, c in zip(cl_sim, cl_cp)], "-o", color=color, ms=4)
        axp.set_title(f"dx={dx}"); axp.grid(alpha=0.3)
        axd.axhline(0, color="k", lw=0.8)
        axd.set_xlabel("α (deg)"); axd.grid(alpha=0.3)
    axes[0, 0].set_ylabel("Cl"); axes[0, 0].legend(fontsize=6, ncol=2)
    axes[1, 0].set_ylabel("Cl_simulado − Cl_Cp")
    fig.suptitle("Cl integrado (fuerzas, línea sólida) vs Cl(∮ΔCp) (línea discontinua)")
    fig.savefig(os.path.join(OUT_DIR, "summary_cl_vs_cp.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSummary: {len(rows)}/{len(build_queue())} sims en {OUT_DIR}")
    print("  summary.csv, summary_polar.png, summary_convergencia.png, summary_coste.png, "
          "summary_cl_vs_cp.png")


def main_cli():
    queue = build_queue()
    if "--summary" in sys.argv:
        make_summary()
        return
    if "--list" in sys.argv:
        for alpha, cfl, dx in queue:
            nombre = sim_name(alpha, cfl, dx)
            done = os.path.exists(os.path.join(OUT_DIR, f"{nombre}.json"))
            failed = os.path.exists(os.path.join(OUT_DIR, f"{nombre}.failed.json"))
            estado = "OK" if done else ("FAILED (se reintentará)" if failed else "pendiente")
            print(f"  [{estado:>22}] {nombre}  iters={iters_for(cfl, dx)}")
        n_done = sum(os.path.exists(os.path.join(OUT_DIR, f"{sim_name(*s)}.json")) for s in queue)
        print(f"\n{n_done}/{len(queue)} completadas")
        return

    pendientes = [s for s in queue
                  if not os.path.exists(os.path.join(OUT_DIR, f"{sim_name(*s)}.json"))]
    print(f"Cola: {len(queue)} sims, {len(queue) - len(pendientes)} hechas, "
          f"{len(pendientes)} pendientes")
    for alpha, cfl, dx in pendientes:
        nombre = sim_name(alpha, cfl, dx)
        failed_path = os.path.join(OUT_DIR, f"{nombre}.failed.json")
        try:
            run_one(alpha, cfl, dx)
            if os.path.exists(failed_path):
                os.remove(failed_path)
        except KeyboardInterrupt:
            print(f"\n[INTERRUMPIDO] en {nombre}; relanza el script para continuar ahí.")
            sys.exit(130)
        except Exception:
            print(f"[FAIL] {nombre}:\n{traceback.format_exc()}")
            with open(failed_path, "w") as f:
                json.dump({"nombre": nombre, "error": traceback.format_exc()}, f, indent=2)

    make_summary()


if __name__ == "__main__":
    main_cli()

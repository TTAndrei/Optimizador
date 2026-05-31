"""
Compara benchmark_mg_results.json (alpha=10, NACA0012, Re=100k)
con polar XFoil teorico. Busca relacion constante en error/ratio.

Salida:
  results/comparativa_benchmark_vs_xfoil.png
  results/comparativa_benchmark_vs_xfoil_ratios.png
  consola: tabla con err_abs, err_rel%, ratio
"""
from __future__ import annotations
import csv
import json
import os
import sys
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np

from sim_defaults import preferred_cl_column, preferred_cd_column

BENCH     = ROOT / "results" / "benchmark_mg_results.json"
XFOIL     = ROOT / "data" / "ComparativasReales" / "NACA0012_100k_Xfoil.csv"
OUT_DIR   = ROOT / "results" / "comparativas"
ALPHA_REF = 10.0

NIVEL_COLORS = {0: "#2196F3", 1: "#FF5722", 2: "#4CAF50"}
NIVEL_LABEL  = {0: "L0", 1: "L1", 2: "L2"}


def load_xfoil_at_alpha(alpha_target: float) -> dict:
    rows = []
    with open(XFOIL, encoding="utf-8") as f:
        in_data = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Alpha,"):
                in_data = True
                continue
            if not in_data:
                continue
            parts = line.split(",")
            try:
                a = float(parts[0])
            except ValueError:
                continue
            rows.append((a, float(parts[1]), float(parts[2])))
    rows.sort(key=lambda r: r[0])
    alphas = np.array([r[0] for r in rows])
    cls    = np.array([r[1] for r in rows])
    cds    = np.array([r[2] for r in rows])
    cl_ref = float(np.interp(alpha_target, alphas, cls))
    cd_ref = float(np.interp(alpha_target, alphas, cds))
    return {"alpha": alpha_target, "Cl": cl_ref, "Cd": cd_ref,
            "Ef": cl_ref / cd_ref if cd_ref else float("nan")}


def load_bench() -> list[dict]:
    with open(BENCH, encoding="utf-8") as f:
        rows = json.load(f)
    clean = []
    for r in rows:
        if "error" in r:
            continue
        cl_key = preferred_cl_column(set(r.keys()))
        cd_key = preferred_cd_column(set(r.keys()), prefer_bl=True)
        if r.get(cl_key) is None or r.get(cd_key) is None:
            continue
        rr = dict(r)
        rr["Cl_eval"] = rr[cl_key]
        rr["Cd_eval"] = rr[cd_key]
        rr["Cl_eval_column"] = cl_key
        rr["Cd_eval_column"] = cd_key
        clean.append(rr)
    return clean


def split_id(id_str: str) -> tuple[str, int]:
    base, lvl = id_str.split("_L")
    return base, int(lvl)


def fig_comparacion(rows: list[dict], ref: dict) -> None:
    """3 subplots: Cl, Cd, Ef. Barras agrupadas por config base, color=niveles.
    Linea horizontal = XFoil teorico."""
    bases = sorted({split_id(r["id"])[0] for r in rows})
    niveles = sorted({split_id(r["id"])[1] for r in rows})

    by_id = {r["id"]: r for r in rows}
    metrics = [
        ("Cl_eval", "Cl",    ref["Cl"]),
        ("Cd_eval", "Cd",    ref["Cd"]),
        ("Ef",      "Cl/Cd", ref["Ef"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(f"Benchmark MG vs XFoil  (NACA0012, Re=100k, α={ALPHA_REF:.0f}°)",
                 fontsize=13)

    x = np.arange(len(bases))
    w = 0.25

    for col, (key, label, ref_val) in enumerate(metrics):
        ax = axes[col]
        for j, lvl in enumerate(niveles):
            vals = []
            for b in bases:
                r = by_id.get(f"{b}_L{lvl}")
                if r is None:
                    vals.append(np.nan)
                elif key == "Ef":
                    vals.append(r["Cl_eval"] / r["Cd_eval"]
                                if r.get("Cd_eval") else np.nan)
                else:
                    vals.append(r.get(key, np.nan))
            ax.bar(x + (j - 1) * w, vals, w,
                   color=NIVEL_COLORS[lvl], label=NIVEL_LABEL[lvl])
        ax.axhline(ref_val, color="k", linestyle="--", linewidth=1.5,
                   label=f"XFoil={ref_val:.4f}")
        ax.set_xticks(x)
        ax.set_xticklabels(bases)
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "comparativa_benchmark_vs_xfoil.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[save] {out}")


def fig_errores(rows: list[dict], ref: dict) -> None:
    """3 subplots: err_abs Cl, err_rel% Cd, ratio Cd/Cd_xfoil.
    Busca patron constante."""
    bases = sorted({split_id(r["id"])[0] for r in rows})
    niveles = sorted({split_id(r["id"])[1] for r in rows})
    by_id = {r["id"]: r for r in rows}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Error simulador vs XFoil — busqueda relacion constante",
                 fontsize=13)

    x = np.arange(len(bases))
    w = 0.25

    series = [
        ("err_Cl",    lambda r: r["Cl_eval"] - ref["Cl"],
         "Cl - Cl_xf  (abs)"),
        ("err_Cd_pct",lambda r: 100 * (r["Cd_eval"] - ref["Cd"]) / ref["Cd"],
         "(Cd - Cd_xf) / Cd_xf  [%]"),
        ("ratio_Cd",  lambda r: r["Cd_eval"] / ref["Cd"],
         "Cd_sim / Cd_xf"),
    ]

    for col, (_, fn, title) in enumerate(series):
        ax = axes[col]
        for j, lvl in enumerate(niveles):
            vals = [fn(by_id[f"{b}_L{lvl}"]) if f"{b}_L{lvl}" in by_id
                    else np.nan for b in bases]
            ax.bar(x + (j - 1) * w, vals, w,
                   color=NIVEL_COLORS[lvl], label=NIVEL_LABEL[lvl])
            for k, v in enumerate(vals):
                if not np.isnan(v):
                    ax.text(x[k] + (j - 1) * w, v, f"{v:.2f}",
                            ha="center", va="bottom" if v >= 0 else "top",
                            fontsize=7)
        ax.axhline(0 if col < 2 else 1, color="k", linestyle=":", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(bases)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "comparativa_benchmark_vs_xfoil_ratios.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[save] {out}")


def fig_correlacion(rows: list[dict], ref: dict) -> None:
    """Cl_sim vs Cd_sim — punto por config. Marca XFoil. Detecta clusters."""
    fig, ax = plt.subplots(figsize=(9, 7))
    for r in rows:
        _, lvl = split_id(r["id"])
        ef = r["Cl_eval"] / r["Cd_eval"]
        ax.scatter(r["Cd_eval"], r["Cl_eval"],
                   color=NIVEL_COLORS[lvl], s=80,
                   edgecolor="black", linewidth=0.5)
        ax.annotate(f"{r['id']}\nEf={ef:.1f}",
                    (r["Cd_eval"], r["Cl_eval"]),
                    fontsize=7, xytext=(5, 5), textcoords="offset points")
    ax.scatter([ref["Cd"]], [ref["Cl"]], color="red", s=200, marker="*",
               edgecolor="black", linewidth=1, label=f"XFoil (Ef={ref['Ef']:.1f})", zorder=5)
    ax.set_xlabel("Cd"); ax.set_ylabel("Cl")
    ax.set_title(f"Polar Cl vs Cd  (α={ALPHA_REF:.0f}°)  — todas configs vs XFoil")
    ax.grid(True, alpha=0.3)
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=NIVEL_COLORS[l], markersize=10,
                          label=NIVEL_LABEL[l]) for l in NIVEL_COLORS]
    handles.append(plt.Line2D([0], [0], marker="*", color="w",
                              markerfacecolor="red", markersize=15, label="XFoil"))
    ax.legend(handles=handles, fontsize=9)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "comparativa_benchmark_vs_xfoil_polar.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[save] {out}")


def print_tabla(rows: list[dict], ref: dict) -> None:
    print(f"\n  XFoil alpha={ALPHA_REF:.0f}: Cl={ref['Cl']:.4f}  Cd={ref['Cd']:.4f}  "
          f"Ef={ref['Ef']:.3f}\n")
    hdr = f"  {'id':<8} {'Cl':>8} {'Cd':>8} {'Ef':>7}  | {'dCl':>8} {'dCd':>8} "\
          f"{'dCl%':>7} {'dCd%':>7}  | {'Cl/Clxf':>8} {'Cd/Cdxf':>8}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    rows_sorted = sorted(rows, key=lambda r: (split_id(r["id"])[0],
                                              split_id(r["id"])[1]))
    ratios_cl, ratios_cd = [], []
    for r in rows_sorted:
        cl, cd = r["Cl_eval"], r["Cd_eval"]
        ef = cl / cd if cd else float("nan")
        dcl = cl - ref["Cl"]; dcd = cd - ref["Cd"]
        rcl = cl / ref["Cl"]; rcd = cd / ref["Cd"]
        ratios_cl.append(rcl); ratios_cd.append(rcd)
        print(f"  {r['id']:<8} {cl:>8.4f} {cd:>8.4f} {ef:>7.2f}  | "
              f"{dcl:>+8.4f} {dcd:>+8.4f} "
              f"{100*dcl/ref['Cl']:>+6.1f}% {100*dcd/ref['Cd']:>+6.1f}%  | "
              f"{rcl:>8.3f} {rcd:>8.3f}")
    print(f"\n  Cl_sim/Cl_xf:  mean={np.mean(ratios_cl):.3f}  "
          f"std={np.std(ratios_cl):.3f}  "
          f"range=[{min(ratios_cl):.3f}, {max(ratios_cl):.3f}]")
    print(f"  Cd_sim/Cd_xf:  mean={np.mean(ratios_cd):.3f}  "
          f"std={np.std(ratios_cd):.3f}  "
          f"range=[{min(ratios_cd):.3f}, {max(ratios_cd):.3f}]")
    cv_cl = np.std(ratios_cl) / np.mean(ratios_cl)
    cv_cd = np.std(ratios_cd) / np.mean(ratios_cd)
    print(f"  CV (std/mean): Cl={cv_cl:.3f}  Cd={cv_cd:.3f}  "
          f"-> {'CONSTANTE' if max(cv_cl, cv_cd) < 0.10 else 'NO constante'}")


def main() -> None:
    ref  = load_xfoil_at_alpha(ALPHA_REF)
    rows = load_bench()
    print(f"  Cargado: {len(rows)} configs benchmark")
    fig_comparacion(rows, ref)
    fig_errores(rows, ref)
    fig_correlacion(rows, ref)
    print_tabla(rows, ref)


if __name__ == "__main__":
    main()

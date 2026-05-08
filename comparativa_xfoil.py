"""
Comparativa Simulador2D vs XFoil NACA0012 Re=100k
"""
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Cargar XFoil ──────────────────────────────────────────────────────────────
xfoil_path = Path("ComparativasReales/NACA0012_100k_Xfoil.csv")
xf_alpha, xf_cl, xf_cd = [], [], []
with open(xfoil_path) as f:
    reader = csv.reader(f)
    in_data = False
    for row in reader:
        if not row:
            continue
        if row[0].strip() == "Alpha":
            in_data = True
            continue
        if in_data:
            try:
                xf_alpha.append(float(row[0]))
                xf_cl.append(float(row[1]))
                xf_cd.append(float(row[2]))
            except (ValueError, IndexError):
                pass

xf_alpha = np.array(xf_alpha)
xf_cl = np.array(xf_cl)
xf_cd = np.array(xf_cd)
xf_eff = xf_cl / xf_cd

# ── Cargar simulador ──────────────────────────────────────────────────────────
with open("barrido_modos_resultados.json") as f:
    data = json.load(f)

modes = ["turbo", "turbo_hd", "turbo_ultra"]
mode_labels = {"turbo": "Turbo (dx=0.002)", "turbo_hd": "Turbo HD (dx=0.0015)", "turbo_ultra": "Turbo Ultra (dx=0.001)"}
mode_colors = {"turbo": "#e74c3c", "turbo_hd": "#f39c12", "turbo_ultra": "#2ecc71"}
mode_markers = {"turbo": "o", "turbo_hd": "s", "turbo_ultra": "^"}

sim = {}
for mode in modes:
    entries = sorted([d for d in data if d["modo"] == mode], key=lambda x: x["alpha"])
    sim[mode] = {
        "alpha": np.array([d["alpha"] for d in entries]),
        "cl":    np.array([d["Cl_mean"] for d in entries]),
        "cd":    np.array([d["Cd_mean"] for d in entries]),
        "cl_std": np.array([d["Cl_std"] for d in entries]),
        "cd_std": np.array([d["Cd_std"] for d in entries]),
    }
    sim[mode]["eff"] = sim[mode]["cl"] / sim[mode]["cd"]

# Filtrar XFoil al rango del simulador
alpha_min, alpha_max = -10, 10
mask = (xf_alpha >= alpha_min) & (xf_alpha <= alpha_max)
xf_a = xf_alpha[mask]
xf_c = xf_cl[mask]
xf_d = xf_cd[mask]
xf_e = xf_eff[mask]

# ── Errores medios ────────────────────────────────────────────────────────────
print(f"{'Modo':<15} {'dCl RMS':>10} {'dCd RMS':>10} {'dEf RMS':>10}")
for mode in modes:
    a = sim[mode]["alpha"]
    xf_cl_interp = np.interp(a, xf_alpha, xf_cl)
    xf_cd_interp = np.interp(a, xf_alpha, xf_cd)
    xf_ef_interp = xf_cl_interp / xf_cd_interp
    dcl = np.sqrt(np.mean((sim[mode]["cl"] - xf_cl_interp)**2))
    dcd = np.sqrt(np.mean((sim[mode]["cd"] - xf_cd_interp)**2))
    def_ = np.sqrt(np.mean((sim[mode]["eff"] - xf_ef_interp)**2))
    print(f"{mode:<15} {dcl:>10.4f} {dcd:>10.4f} {def_:>10.4f}")

# ── Figura principal: 2x2 ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle("NACA 0012 — Simulador vs XFoil (Re = 100 000)", fontsize=15, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

ax_cl  = fig.add_subplot(gs[0, 0])
ax_cd  = fig.add_subplot(gs[0, 1])
ax_eff = fig.add_subplot(gs[1, 0])
ax_pol = fig.add_subplot(gs[1, 1])

# ─ Cl vs alpha ─
ax_cl.plot(xf_a, xf_c, "k-", lw=2, label="XFoil", zorder=10)
for mode in modes:
    a = sim[mode]["alpha"]
    cl = sim[mode]["cl"]
    std = sim[mode]["cl_std"]
    ax_cl.errorbar(a, cl, yerr=std, fmt=mode_markers[mode]+"-",
                   color=mode_colors[mode], label=mode_labels[mode],
                   capsize=3, ms=5, lw=1.4)
ax_cl.set_xlabel("α (°)")
ax_cl.set_ylabel("Cl")
ax_cl.set_title("Coeficiente de sustentación")
ax_cl.legend(fontsize=8)
ax_cl.grid(True, alpha=0.3)
ax_cl.axhline(0, color="k", lw=0.5, ls="--")

# ─ Cd vs alpha ─
ax_cd.plot(xf_a, xf_d, "k-", lw=2, label="XFoil", zorder=10)
for mode in modes:
    a = sim[mode]["alpha"]
    cd = sim[mode]["cd"]
    std = sim[mode]["cd_std"]
    ax_cd.errorbar(a, cd, yerr=std, fmt=mode_markers[mode]+"-",
                   color=mode_colors[mode], label=mode_labels[mode],
                   capsize=3, ms=5, lw=1.4)
ax_cd.set_xlabel("α (°)")
ax_cd.set_ylabel("Cd")
ax_cd.set_title("Coeficiente de arrastre")
ax_cd.legend(fontsize=8)
ax_cd.grid(True, alpha=0.3)

# ─ Eficiencia vs alpha ─
ax_eff.plot(xf_a, xf_e, "k-", lw=2, label="XFoil", zorder=10)
for mode in modes:
    a = sim[mode]["alpha"]
    eff = sim[mode]["eff"]
    ax_eff.plot(a, eff, mode_markers[mode]+"-", color=mode_colors[mode],
                label=mode_labels[mode], ms=5, lw=1.4)
ax_eff.set_xlabel("α (°)")
ax_eff.set_ylabel("Cl / Cd")
ax_eff.set_title("Eficiencia aerodinámica")
ax_eff.legend(fontsize=8)
ax_eff.grid(True, alpha=0.3)
ax_eff.axhline(0, color="k", lw=0.5, ls="--")

# ─ Polar Cl vs Cd ─
ax_pol.plot(xf_d, xf_c, "k-", lw=2, label="XFoil", zorder=10)
for mode in modes:
    cd = sim[mode]["cd"]
    cl = sim[mode]["cl"]
    ax_pol.plot(cd, cl, mode_markers[mode]+"-", color=mode_colors[mode],
                label=mode_labels[mode], ms=5, lw=1.4)
ax_pol.set_xlabel("Cd")
ax_pol.set_ylabel("Cl")
ax_pol.set_title("Polar aerodinámica (Cl vs Cd)")
ax_pol.legend(fontsize=8)
ax_pol.grid(True, alpha=0.3)

out = "comparativa_xfoil.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nGuardado: {out}")
plt.show()

# ── Figura de errores ─────────────────────────────────────────────────────────
# Cl: error absoluto (Cl cruza cero → relativo explota)
# Cd: error relativo en % (Cd siempre >0)
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Error vs XFoil (Re=100k) — Cl absoluto | Cd relativo", fontsize=13, fontweight="bold")

for mode in modes:
    a = sim[mode]["alpha"]
    xf_cl_i = np.interp(a, xf_alpha, xf_cl)
    xf_cd_i = np.interp(a, xf_alpha, xf_cd)
    err_cl_abs = sim[mode]["cl"] - xf_cl_i
    err_cd_rel = (sim[mode]["cd"] - xf_cd_i) / xf_cd_i * 100
    axes[0].plot(a, err_cl_abs, mode_markers[mode]+"-", color=mode_colors[mode],
                 label=mode_labels[mode], ms=5, lw=1.5)
    axes[1].plot(a, err_cd_rel, mode_markers[mode]+"-", color=mode_colors[mode],
                 label=mode_labels[mode], ms=5, lw=1.5)

# Cl error
axes[0].axhline(0, color="k", lw=1.2, ls="--")
axes[0].axhline(0.1, color="gray", lw=0.7, ls=":", alpha=0.6)
axes[0].axhline(-0.1, color="gray", lw=0.7, ls=":", alpha=0.6)
axes[0].set_xlabel("alfa (deg)")
axes[0].set_ylabel("Cl_sim - Cl_xfoil")
axes[0].set_title("Error absoluto Cl")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Cd error
axes[1].axhline(0, color="k", lw=1.2, ls="--")
for pct in [50, 100, 200]:
    axes[1].axhline(pct, color="gray", lw=0.7, ls=":", alpha=0.5)
axes[1].set_xlabel("alfa (deg)")
axes[1].set_ylabel("(Cd_sim - Cd_xfoil) / Cd_xfoil * 100")
axes[1].set_title("Error relativo Cd (%)")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

out2 = "comparativa_xfoil_errores.png"
fig2.tight_layout()
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Guardado: {out2}")
plt.show()

# ── Figura extra: Cl y Cd lado a lado con XFoil superpuesto ───────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle("Comparativa directa: Simulador vs XFoil", fontsize=13, fontweight="bold")

axes3[0].plot(xf_a, xf_c, "k-", lw=2.5, label="XFoil", zorder=10)
axes3[1].plot(xf_a, xf_d, "k-", lw=2.5, label="XFoil", zorder=10)
for mode in modes:
    a = sim[mode]["alpha"]
    axes3[0].plot(a, sim[mode]["cl"], mode_markers[mode]+"--",
                  color=mode_colors[mode], label=mode_labels[mode], ms=5, lw=1.4)
    axes3[1].plot(a, sim[mode]["cd"], mode_markers[mode]+"--",
                  color=mode_colors[mode], label=mode_labels[mode], ms=5, lw=1.4)

for ax, ylabel, title in zip(axes3, ["Cl", "Cd"], ["Cl vs alfa", "Cd vs alfa"]):
    ax.set_xlabel("alfa (deg)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

out3 = "comparativa_xfoil_directa.png"
fig3.tight_layout()
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Guardado: {out3}")
plt.show()

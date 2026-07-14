#!/usr/bin/env python3
"""Gráficas comparativas SA fully-turbulent vs SA-BC (transición) por perfil.

Lee los summary.csv de los barridos y genera, por perfil, Cl/Cd/Ef vs alpha
en alta resolución con ambas líneas superpuestas.

Salida: results/barridos/comparacion_transicion/<perfil>_{Cl,Cd,Ef}.png
"""
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BARRIDOS = ROOT / "results" / "barridos"
OUT = BARRIDOS / "comparacion_transicion"

CASOS = {
    "NACA0012": ("barrido_alpha_naca0012_dx002_sa", "barrido_alpha_naca0012_dx002_sabc"),
    "AG24": ("barrido_alpha_ag24_dx002_sa", "barrido_alpha_ag24_dx002_sabc"),
}
METRICAS = [
    ("Cl_final", "Cl", "Coeficiente de sustentación Cl"),
    ("Cd_final", "Cd", "Coeficiente de resistencia Cd"),
    ("Ef_final", "Ef", "Eficiencia aerodinámica L/D"),
]


def leer(suffix):
    path = BARRIDOS / suffix / "summary.csv"
    if not path.exists():
        print(f"  AVISO: falta {path}")
        return [], []
    a, filas = [], []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            a.append(float(r["alpha_deg"]))
            filas.append(r)
    orden = sorted(range(len(a)), key=lambda i: a[i])
    return [a[i] for i in orden], [filas[i] for i in orden]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    generados = []
    for perfil, (suf_sa, suf_sabc) in CASOS.items():
        a_sa, filas_sa = leer(suf_sa)
        a_bc, filas_bc = leer(suf_sabc)
        if not a_sa and not a_bc:
            continue
        for key, corto, titulo in METRICAS:
            fig, ax = plt.subplots(figsize=(9, 6))
            if a_sa:
                y = [float(r.get(key) or "nan") for r in filas_sa]
                ax.plot(a_sa, y, marker="o", linewidth=2.2, markersize=7,
                        color="#c0392b", label="SA fully-turbulent (sin transición)")
            if a_bc:
                y = [float(r.get(key) or "nan") for r in filas_bc]
                ax.plot(a_bc, y, marker="s", linewidth=2.2, markersize=7,
                        color="#2471a3", label="SA-BC (con transición, Tu=0.1%)")
            ax.axhline(0.0, color="k", linewidth=0.6, linestyle=":")
            ax.set_xlabel("α [deg]", fontsize=13)
            ax.set_ylabel(corto, fontsize=13)
            ax.set_title(f"{perfil} — {titulo} · Re=1e5, dx=0.002", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)
            ax.tick_params(labelsize=11)
            fig.tight_layout()
            dst = OUT / f"{perfil}_{corto}.png"
            fig.savefig(dst, dpi=300)
            plt.close(fig)
            generados.append(dst)
    print("Generados:")
    for g in generados:
        print(f"  {g.relative_to(ROOT)}")
    return 0 if generados else 1


if __name__ == "__main__":
    sys.exit(main())

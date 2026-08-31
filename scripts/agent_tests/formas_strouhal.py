"""Recalcula el Strouhal de todos los puntos de formas desde las series ya
guardadas. No simula: se puede correr las veces que haga falta.

POR QUE UN SEGUNDO ESTIMADOR
  El pico del FFT de Cl baila entre ~0.10 y ~0.28 segun el caso, y ~0.10 es
  justo la mitad de ~0.20. En 2D la estela a Reynolds alto sufre duplicacion de
  periodo (los vortices alternan intensidad entre lados), y entonces el
  subarmonico se lleva mas energia que el fundamental: el FFT devuelve f/2 y el
  St sale a la mitad. Contar cruces por cero es inmune a eso, porque mide el
  periodo de la oscilacion, no el reparto de energia del espectro.

  Se guardan LOS DOS con su discrepancia. Si coinciden, el St es solido. Si el
  del FFT es la mitad del de cruces, hay duplicacion de periodo y el bueno es el
  de cruces.

Uso: .venv/bin/python scripts/agent_tests/formas_strouhal.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
import numpy as np

AMP_MIN = 0.05      # por debajo, la estela no ha saturado: no hay St que medir


def estima(t, cl):
    m = np.isfinite(t) & np.isfinite(cl) & (t > 0)
    t, cl = t[m], cl[m]
    if len(t) < 64:
        return {}
    t, cl = t[len(t) // 2:], cl[len(cl) // 2:]        # fuera el transitorio
    tu = np.linspace(t[0], t[-1], len(t))
    y = np.interp(tu, t, cl)
    y = y - y.mean()
    T = tu[-1] - tu[0]
    amp = float(np.sqrt(2.0) * y.std())

    esp = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    fr = np.fft.rfftfreq(len(y), d=T / (len(tu) - 1))
    k = 1 + int(np.argmax(esp[1:]))
    st_fft = float(fr[k])
    prom = float(esp[k] / max(np.median(esp[1:]), 1e-30))

    # Cruces por cero ascendentes: cada uno es un periodo completo.
    sg = np.signbit(y)
    subidas = int(np.count_nonzero(~sg[1:] & sg[:-1]))
    st_zc = float(subidas / T) if T > 0 else None

    ok = prom >= 8.0 and amp >= AMP_MIN
    r = {"cl_amplitud": round(amp, 5), "pico_espectral": round(prom, 1),
         "st_fft": round(st_fft, 4) if ok else None,
         "st_cruces": round(st_zc, 4) if ok and st_zc else None}
    if r["st_fft"] and r["st_cruces"]:
        ratio = r["st_cruces"] / r["st_fft"]
        r["st_ratio_cruces_fft"] = round(ratio, 2)
        # ratio ~2 => el FFT cogio el subarmonico; manda el de cruces.
        r["duplicacion_periodo"] = bool(1.7 <= ratio <= 2.3)
        r["strouhal"] = r["st_cruces"] if r["duplicacion_periodo"] else r["st_fft"]
    else:
        r["strouhal"] = None
    return r


def main():
    n = 0
    for js in sorted(glob.glob("results/formas/*/metricas/*.json")):
        d = json.load(open(js))
        cambio = False
        for k, rec in d.items():
            tag = f"{k}_dx{rec['dx']:.4f}"
            serie = os.path.join("results", "formas", rec["forma"], "series",
                                 f"{tag}_serie.npz")
            if not os.path.exists(serie):
                continue
            s = np.load(serie)
            r = estima(np.asarray(s["t"], float), np.asarray(s["cl"], float))
            if r:
                rec.update(r); cambio = True; n += 1
                dup = " [duplicacion de periodo]" if r.get("duplicacion_periodo") else ""
                print(f"  {rec['forma']:9s} Re={k:7s} dx={rec['dx']}  "
                      f"St={r['strouhal']}  (fft {r['st_fft']}, cruces "
                      f"{r['st_cruces']}, amp {r['cl_amplitud']}){dup}")
        if cambio:
            json.dump(d, open(js, "w"), indent=2, ensure_ascii=False)
    print(f"{n} puntos recalculados")


if __name__ == "__main__":
    main()

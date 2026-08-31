"""Compara las dos ternas candidatas y decide cual se usa.

    0.008/0.004/0.002   r = 2, 2
    0.006/0.004/0.002   r = 1.5, 2

Criterios, por orden de peso:
  1. monotonas   cuantos (perfil, alpha, magnitud) dan serie monotona
  2. p en rango  cuantos dan orden observado entre 0.5 y 4.0
  3. GCI mediano cuanto de estrecha sale la banda en los que salen bien
  4. signo       ningun punto de la terna con el Cl cambiado de signo
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verificacion_numerica as vn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "resultados_finales")
PERFILES = ("ganador_ag", "naca0012")
ANGULOS = ("0.0", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0")
MAGS = ("cl", "cd", "ld")
TERNAS = {"0.008": (0.002, 0.004, 0.008), "0.006": (0.002, 0.004, 0.006)}


def carga(p, dx):
    f = os.path.join(OUT, p, "metricas", "polar_dx%.4f.json" % dx)
    return json.load(open(f)) if os.path.exists(f) else {}


def disponibles(terna):
    """(perfil, alpha) que existen en las tres mallas de la terna."""
    datos = {dx: {p: carga(p, dx) for p in PERFILES} for dx in terna}
    return {(p, a) for p in PERFILES for a in ANGULOS
            if all(a in datos[dx][p] for dx in terna)}


def evalua(terna, casos):
    """casos = conjunto (perfil, alpha) comun a las dos ternas. Se pasa desde
    fuera para que la comparacion sea sobre exactamente los mismos puntos: si
    una terna se juzga con mas angulos que la otra, los recuentos no son
    comparables."""
    datos = {dx: {p: carga(p, dx) for p in PERFILES} for dx in terna}
    filas, faltan = [], 0
    for p in PERFILES:
        for a in ANGULOS:
            if (p, a) not in casos:
                faltan += 1
                continue
            cls = [datos[dx][p][a]["cl"] for dx in terna]
            signo_ok = len(set(np.sign(c) for c in cls if abs(c) > 1e-3)) <= 1
            for mag in MAGS:
                f1, f2, f3 = (datos[dx][p][a][mag] for dx in terna)
                g = vn.gci_triplete(f1, f2, f3, *terna)
                filas.append({"perfil": p, "alpha": a, "mag": mag,
                              "p": g["p_observado"], "gci": g["GCI_fina_pct"],
                              "mono": g["convergencia_monotona"],
                              "signo_ok": signo_ok,
                              "f_ext": g["f_extrapolado_richardson"],
                              "f": [f3, f2, f1]})
    return filas, faltan


def resumen(nombre, filas):
    n = len(filas)
    mono = [f for f in filas if f["mono"]]
    buenos = [f for f in mono if 0.5 <= f["p"] <= 4.0 and f["signo_ok"]]
    gcis = [f["gci"] for f in buenos if np.isfinite(f["gci"])]
    ps = [f["p"] for f in mono if np.isfinite(f["p"])]
    return {"terna": nombre, "n": n, "monotonas": len(mono), "buenos": len(buenos),
            "gci_mediano": float(np.median(gcis)) if gcis else float("nan"),
            "gci_p90": float(np.percentile(gcis, 90)) if gcis else float("nan"),
            "p_mediano": float(np.median(ps)) if ps else float("nan"),
            "signo_malo": sum(1 for f in filas if not f["signo_ok"]) // len(MAGS)}


def main():
    comun = set.intersection(*(disponibles(t) for t in TERNAS.values()))
    if not comun:
        print("no hay ningun (perfil, alpha) presente en las dos ternas todavia")
        return
    total = len(PERFILES) * len(ANGULOS)
    print("comparando sobre %d de %d casos (perfil x alpha) presentes en ambas ternas"
          % (len(comun), total))
    if len(comun) < total:
        faltan = sorted({(p, a) for p in PERFILES for a in ANGULOS} - comun)
        print("  fuera por falta de datos: " +
              ", ".join("%s a=%s" % (p, a) for p, a in faltan))
    print()

    res = {}
    for nombre, terna in TERNAS.items():
        filas, _ = evalua(terna, comun)
        res[nombre] = resumen(nombre, filas)
        res[nombre]["_filas"] = filas
        res[nombre]["casos"] = sorted(comun)

    print("%-8s %5s %10s %8s %12s %10s %10s %10s" % (
        "terna", "n", "monotonas", "buenos", "GCI mediano", "GCI p90", "p mediano", "signo mal"))
    for k, r in res.items():
        print("%-8s %5d %10d %8d %11.2f%% %9.2f%% %10.2f %10d" % (
            k, r["n"], r["monotonas"], r["buenos"], r["gci_mediano"],
            r["gci_p90"], r["p_mediano"], r["signo_malo"]))

    # gana la que mas puntos buenos da; a igualdad, la de banda mas estrecha
    orden = sorted(res.values(), key=lambda r: (-r["buenos"], r["gci_mediano"]))
    g = orden[0]
    print("\nGANA LA TERNA %s  (%d/%d buenos, GCI mediano %.2f%%)" % (
        g["terna"], g["buenos"], g["n"], g["gci_mediano"]))

    print("\ndesglose de la ganadora:")
    print("%-11s %5s %4s %10s %8s %9s %6s" % (
        "perfil", "alpha", "mag", "p", "GCI%", "f_ext", "mono"))
    for f in g["_filas"]:
        print("%-11s %5s %4s %10.3f %8.2f %9.5f %6s" % (
            f["perfil"], f["alpha"], f["mag"], f["p"], f["gci"], f["f_ext"],
            "si" if f["mono"] else "NO"))

    salida = {k: {kk: vv for kk, vv in r.items() if kk != "_filas"} for k, r in res.items()}
    salida["ganadora"] = g["terna"]
    salida["detalle_ganadora"] = g["_filas"]
    d = os.path.join(OUT, "richardson")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "comparativa_ternas.json"), "w") as fh:
        json.dump(salida, fh, indent=2, ensure_ascii=False)
    print("\nescrito en %s/comparativa_ternas.json" % d)


if __name__ == "__main__":
    main()

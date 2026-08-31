"""
Desglose por componente de cada variante, sacado de los logs de los runners.

El modelo de coste dice que el paso son ~40 ms fijos mas ~5.4 ms por V-cycle, y
que el techo de acelerar solo la proyeccion es 5.7x. Esos 40 ms fijos son
adveccion, difusion, IBM y fuerzas: en cuanto la proyeccion baje, pasan a ser
ellos el cuello de botella, asi que conviene tenerlos medidos antes de decidir
donde seguir.

El simulador ya imprime la tabla al terminar cada run. Esto la recoge de los
logs en vez de instrumentar el harness, que esta en uso: no se toca nada que
este corriendo.

Uso:
    .venv/bin/python scripts/agent_tests/desglose_componentes.py
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOGS = os.path.join(ROOT, "results", "bench_opt")

# "--- nombre  opts=... " marca el comienzo de una variante;
# "  Advección....... 12.34 s  ( 7.3%)" son las filas de la tabla.
RE_VAR = re.compile(r"^\[\d\d:\d\d:\d\d\] --- ([a-z0-9_]+)\s")
RE_FILA = re.compile(r"^\s{2}([A-Za-zÁÉÍÓÚáéíóúñ/ ()]+?)\.{2,}\s+([0-9.]+) s\s+\(\s*([0-9.]+)%\)")
RE_PASO = re.compile(r"^Tiempo promedio por paso dt: ([0-9.]+) s")


def recoger():
    datos = {}
    for fichero in sorted(os.listdir(LOGS)):
        if not fichero.endswith(".log"):
            continue
        actual = None
        with open(os.path.join(LOGS, fichero), errors="replace") as f:
            for linea in f:
                linea = linea.replace("\r", "\n").split("\n")[-1] if "\r" in linea else linea
                m = RE_VAR.match(linea)
                if m:
                    actual = m.group(1)
                    datos.setdefault(actual, {"componentes": {}, "s_paso": None})
                    continue
                if actual is None:
                    continue
                m = RE_PASO.match(linea)
                if m:
                    datos[actual]["s_paso"] = float(m.group(1))
                    continue
                m = RE_FILA.match(linea)
                if m:
                    nombre = m.group(1).strip()
                    datos[actual]["componentes"][nombre] = (
                        float(m.group(2)), float(m.group(3)))
    return {k: v for k, v in datos.items() if v["componentes"]}


def main():
    datos = recoger()
    if not datos:
        print("sin desgloses en los logs todavia")
        return

    orden = ["Advección", "Difusión", "Proyección", "Recálculo dt",
             "Guardado/fuerzas", "Otros (overhead)"]
    cab = f"{'variante':<20}{'ms/paso':>9}" + "".join(f"{c[:9]:>10}" for c in orden)
    print("\n" + cab)
    print("-" * len(cab))
    for nombre, d in sorted(datos.items()):
        sp = d["s_paso"]
        fila = f"{nombre:<20}{1000*sp if sp else 0:>9.1f}"
        for c in orden:
            pct = d["componentes"].get(c, (0.0, 0.0))[1]
            # ms por paso de cada componente, que es lo comparable entre
            # variantes: el porcentaje solo dice como se reparte, no cuanto vale
            fila += f"{(pct/100*1000*sp if sp else 0):>10.1f}"
        print(fila)
    print("-" * len(cab))
    print("valores en ms por paso, no en porcentaje: el porcentaje cambia solo")
    print("porque cambia la proyeccion, y esconde que el resto se queda igual.")

    base = datos.get("baseline") or datos.get("equiv")
    if base and base["s_paso"]:
        sp = base["s_paso"]
        fijo = sum(base["componentes"].get(c, (0, 0))[1] for c in orden
                   if c != "Proyección") / 100 * 1000 * sp
        proy = base["componentes"].get("Proyección", (0, 0))[1] / 100 * 1000 * sp
        print(f"\nbaseline: proyeccion {proy:.0f} ms, resto {fijo:.0f} ms")
        print(f"  techo de acelerar solo la proyeccion: {(proy+fijo)/max(fijo,1e-9):.2f}x")
        print(f"  el mayor del resto marca donde habria que ir despues:")
        resto = sorted(((base["componentes"].get(c, (0, 0))[1] / 100 * 1000 * sp, c)
                        for c in orden if c != "Proyección"), reverse=True)
        for ms, c in resto[:3]:
            print(f"     {c:<20} {ms:6.1f} ms")


if __name__ == "__main__":
    main()

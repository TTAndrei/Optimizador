"""
Visor de progreso del estudio de resultados finales.

Dos niveles de informacion:

  ESTUDIO      cuantos de los 30 puntos estan hechos, cuales, cuanta GPU se ha
               consumido y cuanto queda. Sale de resultados_finales/ESTADO.json,
               que el runner reescribe al empezar y al terminar cada punto.

  SIMULACION   en que iteracion va la simulacion que corre ahora mismo, con su
               Cl y Cd instantaneos. Sale de la memoria compartida "sim2d_meta"
               que Simulador2D publica en cada paso, asi que funciona aunque el
               runner este redirigido a un fichero y no se vea su barra.

Uso:
    .venv/bin/python scripts/agent_tests/rf_progreso.py           # refresco continuo
    .venv/bin/python scripts/agent_tests/rf_progreso.py --once    # una vez y sale
    .venv/bin/python scripts/agent_tests/rf_progreso.py -n 5      # cada 5 s
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from multiprocessing import resource_tracker, shared_memory

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "resultados_finales")
ESTADO = os.path.join(OUT, "ESTADO.json")

# Layout de sim2d_meta, fijado en Simulador2D:
#   ny(i,0) nx(i,4) iter(i,8) cd(f,12) cl(f,16) alpha(f,20) t(d,24) v0x(f,32) v0y(f,36)
META = {"iter": ("i", 8), "cd": ("f", 12), "cl": ("f", 16),
        "alpha": ("f", 20), "t": ("d", 24)}


def lee_shm():
    try:
        shm = shared_memory.SharedMemory(name="sim2d_meta", create=False)
    except FileNotFoundError:
        return None
    # Abrir un bloque existente lo registra en el resource_tracker, que al salir
    # este proceso lo DESVINCULARIA: el visor le borraria la memoria compartida
    # al simulador que esta observando. Aqui solo se lee, nunca se es dueño.
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass
    try:
        d = {k: struct.unpack_from(f, shm.buf, off)[0] for k, (f, off) in META.items()}
    finally:
        shm.close()
    # Un bloque huerfano de una simulacion muerta seguiria ahi con su ultimo
    # valor: si el sello de tiempo no se mueve, no hay nada corriendo.
    return d if time.time() - d["t"] < 120 else None


def _vivo(pid):
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, TypeError):
        return False
    return True


def barra(frac, n=44):
    k = int(round(max(0.0, min(1.0, frac)) * n))
    return "[" + "#" * k + "." * (n - k) + "]"


def hms(s):
    s = int(max(0, s))
    return f"{s//3600:d}h{(s%3600)//60:02d}m"


def pinta():
    if not os.path.exists(ESTADO):
        return "todavia no hay ESTADO.json: el estudio no ha arrancado"
    est = json.load(open(ESTADO))
    tot, hech = est["total_puntos"], est["hechos"]
    L = ["=" * 74,
         "RESULTADOS FINALES TFG — Richardson de dos perfiles",
         "=" * 74,
         f"  {barra(hech / tot)}  {hech}/{tot} puntos  ({100*hech/tot:.0f}%)",
         f"  reusados {est['reusados']}   GPU consumida {hms(est['wall_medido_s'])}"
         f"   restante estimado {hms(est['eta_s'])}",
         f"  ESTADO.json actualizado {est['actualizado']}",
         ""]

    perfiles, dxs = [], []
    for q in est["puntos"]:
        if q["perfil"] not in perfiles:
            perfiles.append(q["perfil"])
        if q["dx"] not in dxs:
            dxs.append(q["dx"])
    for p in perfiles:
        fila = "  ".join(
            f"dx={dx:g}:" + "".join("#" if q["estado"] == "hecho" else "."
                                    for q in est["puntos"]
                                    if q["perfil"] == p and q["dx"] == dx)
            for dx in sorted(dxs, reverse=True))
        L.append(f"  {p:<10} {fila}")
    L.append("  (# hecho, . pendiente; angulos en orden 0 2 4 6 8)")

    ec = est.get("en_curso")
    pid = est.get("pid_runner")
    if ec and pid and not _vivo(pid):
        L += ["", "-" * 74,
              f"  RUNNER CAIDO: el pid {pid} ya no existe y quedo un punto a medias",
              f"    ({ec['perfil']} dx={ec['dx']:g} alpha={ec['alpha']:.0f}º, desde {ec['desde']})",
              "    relanzar con ./lanzar_resultados_finales.sh — reanuda donde estaba"]
        ec = None
    m = lee_shm()
    if ec or m:
        L += ["", "-" * 74, "  SIMULACION EN CURSO"]
    if ec:
        L.append(f"    {ec['perfil']}   dx={ec['dx']:g}   alpha={ec['alpha']:.0f}º"
                 f"   arrancada a las {ec['desde']}")
    if m:
        n = ec["iters"] if ec else 0
        if n:
            f = m["iter"] / n
            L.append(f"    {barra(f, 40)}  {m['iter']}/{n} iters  ({100*f:.1f}%)")
        else:
            L.append(f"    iteracion {m['iter']}")
        ld = m["cl"] / m["cd"] if abs(m["cd"]) > 1e-12 else float("nan")
        L.append(f"    instantaneo:  Cl={m['cl']:+.5f}   Cd={m['cd']:.5f}   L/D={ld:+.2f}")
    elif ec:
        L.append("    (sin senal de la memoria compartida; puede estar mallando"
                 " o generando salidas)")

    ult = [q for q in est["puntos"] if q["estado"] == "hecho" and not q.get("reusado")]
    if ult:
        L += ["", "-" * 74, "  ULTIMOS PUNTOS TERMINADOS"]
        for q in ult[-5:]:
            L.append(f"    {q['perfil']:<10} dx={q['dx']:<6g} α={q['alpha']:>4.0f}º"
                     f"   Cl={q['cl']:.5f}  Cd={q['cd']:.5f}  L/D={q['ld']:7.3f}"
                     f"   {hms(q['wall_s'])}")
    L += ["", "  parar limpio entre puntos:  touch PARAR_ESTUDIO.trigger", "=" * 74]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("-n", type=float, default=10.0, help="segundos entre refrescos")
    a = ap.parse_args()
    if a.once:
        print(pinta())
        return
    try:
        while True:
            sys.stdout.write("\033[H\033[J" + pinta() + "\n")
            sys.stdout.flush()
            time.sleep(a.n)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

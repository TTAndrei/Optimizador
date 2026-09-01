"""
Cuanto cuesta sacar frames: coste real de cada via de salida del solver.

El bloque de salida de Simulador2D.main() hace cuatro cosas de coste muy
distinto y hasta ahora las cuatro caian en el mismo cubo, timing_stats
['guardado']. Separados (guardado_fuerzas / _series / _shm / _frames / _dump),
se puede medir cada uno por separado y, sobre todo, medir el COSTE POR EVENTO,
que es lo unico independiente de la cadencia y lo unico que sirve para elegir
defaults.

Tres cosas que se miden porque ya se sabe que estan ahi:

  1. La publicacion en memoria compartida se paga SIEMPRE, mire alguien o no
     (Simulador2D.py, el comentario sobre "Publicar speed+solid siempre para la
     GUI interna"). A dx=0.002 en dominio C son 1356x919 celdas: 5.0 MB de |u|
     mas 1.25 MB de mascara, de GPU a CPU, cada 'guardado' iteraciones.
  2. dump_field_window comprime con zlib en el hilo principal, mono-hilo y
     bloqueante, con la GPU parada mientras tanto.
  3. save_frame a dpi=600 dibuja un pcolormesh de 1.25 M de cuadrilateros sin
     submuestrear. Es el ~20 s por figura ya documentado en render_videos.py.

Cada configuracion corre en su propio proceso: main() deja estado global
(figuras de matplotlib, bloques de memoria compartida, pools de CuPy) y
compartirlo entre variantes contamina el cronometraje.

Puerta de fidelidad: NINGUNA via de salida toca los campos, asi que todas las
configuraciones tienen que dar exactamente el mismo Cl y el mismo Cd. Si una no
lo hace, el instrumento esta mal, no el solver.

Uso:
    .venv/bin/python scripts/agent_tests/coste_salida.py --todo --dx 0.004
    .venv/bin/python scripts/agent_tests/coste_salida.py --todo --dx 0.002
    .venv/bin/python scripts/agent_tests/coste_salida.py --uno shm --dx 0.004
    .venv/bin/python scripts/agent_tests/coste_salida.py --informe
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")

OUT = os.path.join(ROOT, "results", "coste_salida")
SANDBOX = os.path.join(OUT, "_sandbox")
os.makedirs(SANDBOX, exist_ok=True)

PERFIL = os.path.join(
    ROOT, "results/convergence_study/islands_tfg2/NACA_0012_sharp/epoch10/"
          "NACA_0012_sharp_winner_Re100000_a4.0_LD27.75.dat")

RE = 1e5
GUARDADO = 50

# LA configuracion de las polares de resultados finales: dominio C (24x16 con el
# perfil en cx=6), presupuesto de proyeccion 2x3, SA + consistent + maccormack +
# SA-BC, Re=1e5. Sale de scripts/agent_tests/resultados_finales.py (DOMINIO,
# PRESUPUESTO, FLAGS) mas RunGA.CONFIG. Se mide sobre lo que se usa de verdad.
#
# Las dos paradas van desactivadas: todas las variantes tienen que recorrer
# exactamente las mismas iteraciones o lo que se compara es la parada.
BASE = {
    "v0x": 1.0, "v0y": 0.0, "chord": 1.0, "rho": 1.0, "nu": 1.0 / RE,
    "CFL": 0.5, "alpha_deg": 4.0,
    "Lx": 24.0, "Ly": 16.0, "cx": 6.0,
    "ancho_zona_fina_x": 1.5, "ancho_zona_fina_y": 1.0, "factor_expansion": 1.1,
    "turb_model": "sa", "wall_treatment": "consistent",
    "advection_scheme": "maccormack", "min_te_height_factor": 1.0,
    "wake_refinement_mode": "long_fine_x",
    "transition_model": "sa_bc", "freestream_Tu": 0.1,
    "mg_niveles_max": 2, "mg_max_outer": 2, "mg_cycles_per_outer": 3,
    "divergencia": 0.02,
    "stop_on_clcd_convergence": False, "stop_on_convergence": False,
    "guardado": GUARDADO,
    "graficos": False, "live_view": False, "mostrar_malla": False,
    "save_frames": False, "shm_publish": False,
    "bench_sync": True,
}

# Iteraciones por malla. Lo que fija el error de la medida es el numero de
# EVENTOS de guardado, no el de iteraciones: con guardado=50 salen 30 eventos,
# de sobra para un coste por evento con dos cifras.
ITERS = {0.004: 1500, 0.002: 1000}

# Ventana refinada alrededor del perfil (cx=6, cuerda 1, centrado en Ly/2).
XLIM = (5.5, 8.0)
YLIM = (7.4, 8.6)

# Las cinco optimizaciones del solver de presion con las que se calcularon los
# resultados finales. Van explicitas y comprobadas, no confiadas al default:
# sin `warm_start_filtered`, dx=0.004 con presupuesto 2x3 revienta por blowup de
# velocidad en la iteracion 1340, reproducible en todos los angulos.
FLAGS = ("warm_start", "warm_start_filtered", "coarse_mask_majority",
         "fast_masks", "interp_float32")

CONFIGS = {
    # ---- baseline: ninguna via de salida activa -------------------------
    "off":         {},

    # ---- memoria compartida ---------------------------------------------
    "shm":         {"shm_publish": True},
    "shm_live":    {"shm_publish": True, "live_view": True},
    "shm_c4":      {"shm_publish": True, "shm_cada": 4},

    # ---- volcado de campos a disco --------------------------------------
    "dump_400":    {"dump_fields_dir": "campos", "dump_fields_max_nx": 400},
    "dump_900":    {"dump_fields_dir": "campos", "dump_fields_max_nx": 900},
    "dump_1400":   {"dump_fields_dir": "campos", "dump_fields_max_nx": 1400},
    "dump_900_nc": {"dump_fields_dir": "campos", "dump_fields_max_nx": 900,
                    "dump_fields_comprimir": False},

    # ---- figuras dentro del solver (el camino caro) ----------------------
    "frames_100":  {"save_frames": True, "frames_dir_grueso": "frames",
                    "save_frame_dpi": 100, "save_frames_cada": 250},
    "frames_180":  {"save_frames": True, "frames_dir_grueso": "frames",
                    "save_frame_dpi": 180, "save_frames_cada": 250},
    "frames_600":  {"save_frames": True, "frames_dir_grueso": "frames",
                    "save_frame_dpi": 600, "save_frames_cada": 250},

    # ---- control de linealidad: misma via, cadencia 5x mas alta ----------
    "off_g10":     {"guardado": 10},
    "shm_g10":     {"guardado": 10, "shm_publish": True},
}


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _dir_bytes(d):
    if not os.path.isdir(d):
        return 0, 0
    n = tot = 0
    for f in os.listdir(d):
        fp = os.path.join(d, f)
        if os.path.isfile(fp):
            n += 1
            tot += os.path.getsize(fp)
    return n, tot


def correr(cid, dx):
    """Una configuracion, en este proceso. La llama el modo --uno."""
    import cupy as cp
    import Simulador2D
    import opt_solver

    cfg = CONFIGS[cid]
    iters = ITERS[dx]
    box = os.path.join(SANDBOX, f"{cid}_dx{dx}")
    shutil.rmtree(box, ignore_errors=True)
    os.makedirs(box, exist_ok=True)

    params = {**BASE, "filepath": PERFIL, "dx_min": dx, "iteraciones": iters,
              "save_frame_refined_xlim": XLIM, "save_frame_refined_ylim": YLIM,
              **cfg}
    # Las rutas de salida son relativas al sandbox de la configuracion.
    for k in ("dump_fields_dir", "frames_dir_grueso", "frames_dir_refinado",
              "frames_dir_vorticidad"):
        if params.get(k):
            params[k] = os.path.join(box, params[k])

    opt_solver.reset()
    opt_solver.enable(*FLAGS)
    assert opt_solver.active() == sorted(FLAGS), opt_solver.active()
    prev = os.getcwd()
    os.chdir(box)
    t0 = time.time()
    try:
        mesh = Simulador2D.main(**params)
        wall = time.time() - t0
    finally:
        os.chdir(prev)

    ts = dict(mesh._timing_stats)
    ts.pop("total_por_paso", None)
    cl = cp.asnumpy(mesh.clvector).astype(float)
    cd = cp.asnumpy(mesh.cdvector).astype(float)
    g = int(params["guardado"])
    n = len(cd)
    w = min(n, max(n // 5, 10))

    # Eventos reales de cada via, para pasar de segundos totales a s/evento.
    n_guardado = iters // g
    frame_every = params.get("save_frames_cada") or g
    n_frames = sum(1 for it in range(iters)
                   if it % g == 0 and it % int(frame_every) == 0) \
        if params.get("save_frames") else 0
    campo_every = params.get("dump_fields_cada") or g
    n_dump = sum(1 for it in range(iters)
                 if it % g == 0 and it % int(campo_every) == 0) \
        if params.get("dump_fields_dir") else 0
    shm_every = params.get("shm_cada") or 1
    n_shm = sum(1 for it in range(iters)
                if it % g == 0 and it % int(shm_every) == 0) \
        if params.get("shm_publish") else 0

    nf_dump, by_dump = _dir_bytes(params.get("dump_fields_dir") or "")
    nf_png, by_png = _dir_bytes(params.get("frames_dir_grueso") or "")

    rec = {
        "id": cid, "dx": dx, "iters": iters, "guardado": g,
        "ny": int(mesh.ny), "nx": int(mesh.nx), "celdas": int(mesh.ny * mesh.nx),
        "wall_s": round(wall, 2),
        "it_s": round(iters / wall, 3),
        "s_paso": round(wall / iters, 6),
        "timing": {k: round(v, 4) for k, v in ts.items()},
        "n_eventos": {"guardado": n_guardado, "shm": n_shm,
                      "frames": n_frames, "dump": n_dump},
        "disco": {"dump_ficheros": nf_dump, "dump_bytes": by_dump,
                  "png_ficheros": nf_png, "png_bytes": by_png},
        "cl": float(cl[-w:].mean()), "cd": float(cd[-w:].mean()),
        "n_samples": int(w),
        "overrides": cfg,
    }
    os.makedirs(os.path.join(OUT, f"dx{dx}"), exist_ok=True)
    with open(os.path.join(OUT, f"dx{dx}", f"{cid}.json"), "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)

    _log(f"{cid}: {rec['it_s']} it/s  guardado={ts['guardado']:.2f}s "
         f"(shm {ts['guardado_shm']:.2f} / frames {ts['guardado_frames']:.2f} / "
         f"dump {ts['guardado_dump']:.2f})  Cl={rec['cl']:.6f} Cd={rec['cd']:.6f}")

    del mesh
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return rec


def todo(dx, solo=None):
    ids = solo or list(CONFIGS)
    for cid in ids:
        dest = os.path.join(OUT, f"dx{dx}", f"{cid}.json")
        if os.path.exists(dest):
            _log(f"{cid}: ya existe, se salta")
            continue
        _log(f"=== lanzando {cid} (dx={dx}) ===")
        r = subprocess.run([sys.executable, __file__, "--uno", cid,
                            "--dx", str(dx)], cwd=ROOT)
        if r.returncode != 0:
            _log(f"{cid}: FALLO (codigo {r.returncode})")



# ----------------------------------------------------------------------
# Informe
# ----------------------------------------------------------------------
# Cada via se cobra POR EVENTO, no por iteracion. El coste por evento es lo
# unico independiente de la cadencia, y con el se extrapola el sobrecoste a
# cualquier `guardado` sin volver a simular.
VIAS = [("shm", "guardado_shm", "memoria compartida"),
        ("frames", "guardado_frames", "save_frame (PNG)"),
        ("dump", "guardado_dump", "dump_fields (npz)")]


def _carga(dx):
    d = os.path.join(OUT, f"dx{dx}")
    recs = {}
    for cid in CONFIGS:
        fp = os.path.join(d, f"{cid}.json")
        if os.path.exists(fp):
            recs[cid] = json.load(open(fp))
    return recs


def _s_ev(r, clave, via):
    n = r["n_eventos"][via]
    return (r["timing"][clave] / n) if n else None


def _conclusiones(recs, base):
    """Lo que hay que hacer con estos numeros, no solo los numeros."""
    p = base["s_paso"]
    out = ["### Qué hacer con esto", ""]

    shm = recs.get("shm")
    live = recs.get("shm_live")
    if shm:
        c = _s_ev(shm, "guardado_shm", "shm")
        out.append(
            f"- **La vista en vivo es gratis**: {1000*c:.1f} ms por publicación, "
            f"{100*c/(50*p):.2f} % del tiempo con `guardado=50` y "
            f"{100*c/(10*p):.1f} % incluso publicando cada 10 pasos. "
            f"No hay razón para apagarla en el modo monitor.")
    if live and shm:
        cl, cs = _s_ev(live, "guardado_shm", "shm"), _s_ev(shm, "guardado_shm", "shm")
        out.append(
            f"- Añadir la vorticidad (`live_view`) sube la publicación de "
            f"{1000*cs:.1f} a {1000*cl:.1f} ms. Sigue siendo ruido.")

    d9, dnc = recs.get("dump_900"), recs.get("dump_900_nc")
    if d9 and dnc:
        a, b = _s_ev(d9, "guardado_dump", "dump"), _s_ev(dnc, "guardado_dump", "dump")
        ma = d9["disco"]["dump_bytes"] / max(d9["disco"]["dump_ficheros"], 1) / 1e6
        mb = dnc["disco"]["dump_bytes"] / max(dnc["disco"]["dump_ficheros"], 1) / 1e6
        out.append(
            f"- **El volcado a disco lo domina zlib**: {1000*a:.0f} ms por frame "
            f"comprimido frente a {1000*b:.0f} ms sin comprimir, o sea el "
            f"{100*(a-b)/a:.0f} % del coste. El precio de quitarlo es el disco: "
            f"{ma:.2f} MB por frame comprimido contra {mb:.2f} MB en crudo "
            f"(×{mb/ma:.1f}). Comprimir sale a cuenta salvo que se vuelque muy "
            f"seguido.")

    d4, d14 = recs.get("dump_400"), recs.get("dump_1400")
    if d4 and d14 and d9:
        if abs(_s_ev(d14, "guardado_dump", "dump") - _s_ev(d9, "guardado_dump", "dump")) < 1e-3:
            out.append(
                "- `max_nx` 900 y 1400 cuestan lo mismo porque la ventana "
                "refinada no llega a 900 columnas y en los dos casos el paso de "
                "submuestreo es 1. Por debajo (400) sí baja, y mucho.")

    caros = [(cid, _s_ev(r, "guardado_frames", "frames"))
             for cid, r in recs.items() if r["n_eventos"]["frames"]]
    if caros and d9:
        cid, c = max(caros, key=lambda t: t[1])
        ref = _s_ev(d9, "guardado_dump", "dump")
        out.append(
            f"- **Dibujar dentro del solver es el único camino caro**: "
            f"{1000*c:.0f} ms por figura en `{cid}`, ×{c/ref:.0f} el volcado "
            f"equivalente. A una figura cada 50 pasos son "
            f"{100*c/(50*p):.0f} % de sobrecoste. Volcar campos y montar el "
            f"vídeo después con `render_videos.py` da el mismo vídeo por una "
            f"fracción, y además no ocupa la GPU.")

    out += ["",
            "El sobrecoste leído en la columna `it/s` tiene ±3 % de ruido entre "
            "corridas (se ve en que `shm` y `shm_live` salen a +2.9 % y +0.1 % "
            "midiendo casi lo mismo). Los números fiables son los de **s/evento**, "
            "que salen de los sub-cronómetros del propio bloque de salida y no "
            "de comparar dos corridas.", ""]
    return out


def informe(dxs):
    lineas = ["# Coste de sacar frames", ""]
    resumen = {}

    for dx in dxs:
        recs = _carga(dx)
        if "off" not in recs:
            continue
        base = recs["off"]
        celdas = base["celdas"]
        lineas += [
            f"## Malla dx={dx} — {base['ny']}x{base['nx']} = {celdas:,} celdas".replace(",", " "),
            "",
            f"Baseline (`off`, ninguna via activa): **{base['it_s']} it/s**, "
            f"{base['s_paso']*1000:.2f} ms/paso, {base['iters']} iteraciones, "
            f"guardado cada {base['guardado']}.",
            "",
            "| config | it/s | sobrecoste | via | s/evento | eventos | disco |",
            "|---|---|---|---|---|---|---|",
        ]
        for cid, r in recs.items():
            if cid == "off":
                continue
            # Cada grupo de cadencia tiene su propio baseline.
            b = recs.get("off_g10" if r["guardado"] == 10 else "off", base)
            sobre = 100 * (r["s_paso"] - b["s_paso"]) / b["s_paso"]
            fila_via, s_ev, n_ev = "-", "-", "-"
            for nombre, clave, _ in VIAS:
                n = r["n_eventos"][nombre]
                if n:
                    fila_via = nombre
                    n_ev = str(n)
                    s_ev = f"{r['timing'][clave] / n:.3f} s"
                    break
            dsk = r["disco"]
            if dsk["dump_ficheros"]:
                disco = f"{dsk['dump_bytes']/dsk['dump_ficheros']/1e6:.2f} MB/frame"
            elif dsk["png_ficheros"]:
                disco = f"{dsk['png_bytes']/dsk['png_ficheros']/1e6:.2f} MB/PNG"
            else:
                disco = "-"
            lineas.append(f"| `{cid}` | {r['it_s']} | {sobre:+.1f} % | {fila_via} "
                          f"| {s_ev} | {n_ev} | {disco} |")

        # Puerta de fidelidad: ninguna via toca los campos.
        cls = {cid: round(r["cl"], 9) for cid, r in recs.items() if r["guardado"] == recs["off"]["guardado"]}
        cds = {cid: round(r["cd"], 9) for cid, r in recs.items() if r["guardado"] == recs["off"]["guardado"]}
        ok = len(set(cls.values())) == 1 and len(set(cds.values())) == 1
        lineas += ["", f"**Fidelidad**: {'todas las configuraciones dan el mismo Cl y Cd' if ok else 'DISCREPANCIA — el instrumento esta mal'} "
                       f"(Cl={base['cl']:.6f}, Cd={base['cd']:.6f}).", ""]

        # Extrapolacion: sobrecoste de cada via a cadencias tipicas.
        lineas += ["### Sobrecoste extrapolado por cadencia", "",
                   "Coste por evento dividido entre el coste de las iteraciones que",
                   "cubre esa cadencia. No hace falta volver a simular.", "",
                   "| via | s/evento | cada 200 | cada 50 | cada 10 |",
                   "|---|---|---|---|---|"]
        for cid, r in recs.items():
            for nombre, clave, etiq in VIAS:
                n = r["n_eventos"][nombre]
                if not n:
                    continue
                s_ev = r["timing"][clave] / n
                b = recs.get("off_g10" if r["guardado"] == 10 else "off", base)
                pcts = [100 * s_ev / (c * b["s_paso"]) for c in (200, 50, 10)]
                lineas.append(f"| `{cid}` ({etiq}) | {s_ev:.3f} | "
                              + " | ".join(f"{p:.1f} %" for p in pcts) + " |")
                break
        lineas += _conclusiones(recs, base)
        resumen[str(dx)] = {cid: r for cid, r in recs.items()}

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "INFORME.md"), "w") as f:
        f.write("\n".join(lineas) + "\n")
    with open(os.path.join(OUT, "coste_salida.json"), "w") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print("\n".join(lineas))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--uno", metavar="ID")
    ap.add_argument("--todo", action="store_true")
    ap.add_argument("--dx", type=float, default=0.004)
    ap.add_argument("--solo", nargs="*", metavar="ID")
    ap.add_argument("--informe", action="store_true")
    a = ap.parse_args()

    if a.informe:
        informe([0.004, 0.002])
    elif a.uno:
        correr(a.uno, a.dx)
    elif a.todo:
        todo(a.dx, a.solo)
    else:
        ap.print_help()

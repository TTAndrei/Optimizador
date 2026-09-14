"""
Monta los videos de las corridas de video_largo_perfiles.py. Solo CPU: dibuja
desde los campos volcados, nunca desde el solver.

Dos ventanas por perfil, con tratamientos distintos porque no son la misma
imagen: la refinada es una tira de 3x1 cuerdas donde interesa la capa limite y
la burbuja, y el dominio completo es un 24x16 donde lo unico que se ve es la
estela. La escala de color es FIJA en toda la corrida y COMUN a los dos
perfiles: si cambia frame a frame el video parpadea, y si cambia entre perfiles
la comparacion entre ellos deja de valer.

Uso:
    .venv/bin/python scripts/agent_tests/render_videos_largos.py
    .venv/bin/python scripts/agent_tests/render_videos_largos.py --solo ganador_ag
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(ROOT, "results", "videos_largos")
FPS = 24

# `orient` no es cosmetico: la ventana refinada es una tira de 3x1 y el dominio
# entero un 24x16. Con aspecto real, apilar los dos paneles de la tira da una
# figura cuadrada y ponerlos lado a lado en el dominio da una figura tumbada; al
# reves salen figuras de 15 pulgadas de alto con los paneles perdidos en medio.
VENTANAS = {
    # vmax_w por ventana: el pico de vorticidad vive pegado a la pared y es dos
    # ordenes mayor que el de la estela. Con el mismo limite en las dos, o la
    # capa limite satura o la estela sale en blanco.
    "refinado": {"dir": "campos_refinado", "orient": "v", "ancho": 12.0,
                 "dpi": 150, "vmax_w": 60.0, "pad": 1.4},
    "completo": {"dir": "campos_completo", "orient": "h", "ancho": 15.0,
                 "dpi": 130, "vmax_w": 8.0, "pad": 0.8},
}
NOMBRE = {"ganador_ag": "Ganador de la campaña de optimización (AG)",
          "naca0012": "NACA 0012 sharp"}
VMAX_U = 1.8      # |u| en el punto de diseño no pasa de ~1.7 U_inf
CX, CY = 6.0, 8.0  # borde de ataque del perfil en el dominio C


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def campos(perfil, ventana):
    d = os.path.join(OUT, perfil, VENTANAS[ventana]["dir"])
    return sorted(glob.glob(os.path.join(d, "f_*.npz")))


def dibuja(perfil, ventana, out_dir, dx=0.002):
    fich = campos(perfil, ventana)
    if not fich:
        return 0
    # Incremental: con los PNG ya hechos no se vuelve a dibujar, asi que se
    # puede ir renderizando mientras la GPU sigue con el perfil que falta.
    if len(glob.glob(os.path.join(out_dir, "frame_*.png"))) == len(fich):
        return len(fich)
    os.makedirs(out_dir, exist_ok=True)
    v = VENTANAS[ventana]
    d0 = np.load(fich[0])
    # Coordenadas relativas al borde de ataque: en absoluto el perfil sale en
    # x=6, y=8 y los ejes no dicen nada de cuerdas.
    x = np.asarray(d0["x"], float) - CX
    y = np.asarray(d0["y"], float) - CY
    ar = (y[-1] - y[0]) / (x[-1] - x[0])

    if v["orient"] == "v":
        nrows, ncols = 2, 1
        figsize = (v["ancho"], 2.0 * v["ancho"] * ar + v["pad"])
    else:
        nrows, ncols = 1, 2
        figsize = (v["ancho"], 0.5 * v["ancho"] * ar + v["pad"])

    for f in fich:
        d = np.load(f)
        u = np.asarray(d["u"], np.float32)
        vv = np.asarray(d["v"], np.float32)
        solid = np.asarray(d["solid"], bool)
        spd = np.where(solid, np.nan, np.hypot(u, vv))
        w = np.gradient(vv, x, axis=1) - np.gradient(u, y, axis=0)
        w = np.where(solid, np.nan, w)

        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, layout="constrained")
        ax = np.atleast_1d(ax).ravel()
        im0 = ax[0].pcolormesh(x, y, spd, cmap="rainbow", vmin=0.0, vmax=VMAX_U,
                               shading="auto")
        fig.colorbar(im0, ax=ax[0], label="$|u|/U_\\infty$", fraction=0.03)
        im1 = ax[1].pcolormesh(x, y, w, cmap="RdBu_r", vmin=-v["vmax_w"], vmax=v["vmax_w"],
                               shading="auto")
        fig.colorbar(im1, ax=ax[1], label="$\\omega_z$", fraction=0.03)
        for a in ax:
            a.set_aspect("equal", adjustable="box")
            a.set_xlabel("$x/c$")
        ax[0].set_ylabel("$y/c$")
        if v["orient"] == "v":
            ax[1].set_ylabel("$y/c$")
        fig.suptitle(f"{NOMBRE[perfil]}   $\\alpha=4°$, $Re=10^5$, "
                     f"$dx={dx}$   t={float(d['t']):.2f}")
        fig.savefig(os.path.join(out_dir, f"frame_{int(d['iter']):06d}.png"),
                    dpi=v["dpi"])
        plt.close(fig)
    return len(fich)


def video(frames_dir, mp4, fps):
    frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frames:
        return {"status": "sin frames"}
    if shutil.which("ffmpeg") is None:
        return {"status": "ffmpeg no encontrado"}
    lista = os.path.join(frames_dir, "frames.ffconcat")
    dur = 1.0 / max(fps, 1)
    lineas = ["ffconcat version 1.0"]
    for f in frames:
        lineas += [f"file '{os.path.abspath(f)}'", f"duration {dur:.8f}"]
    lineas.append(f"file '{os.path.abspath(frames[-1])}'")
    with open(lista, "w") as fh:
        fh.write("\n".join(lineas) + "\n")
    os.makedirs(os.path.dirname(mp4), exist_ok=True)
    # scale=trunc(iw/2)*2 no es opcional: libx264 con yuv420p rechaza
    # dimensiones impares.
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "concat", "-safe", "0", "-i", lista,
           "-vf", f"fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", mp4]
    p = subprocess.run(cmd, text=True, capture_output=True)
    return {"status": "ok" if p.returncode == 0 else "fallo",
            "n": len(frames), "s": round(len(frames) / fps, 1),
            "err": p.stderr.strip()[:300]}


def render(perfiles, fps=FPS):
    hechos = []
    for perfil in perfiles:
        for ventana in VENTANAS:
            fd = os.path.join(OUT, perfil, "frames", ventana)
            n = dibuja(perfil, ventana, fd)
            if not n:
                log(f"{perfil}/{ventana}: sin campos volcados")
                continue
            mp4 = os.path.join(OUT, perfil, f"{perfil}_{ventana}.mp4")
            r = video(fd, mp4, fps)
            log(f"{perfil}/{ventana}: {r['status']} ({r.get('n')} frames, "
                f"{r.get('s')} s)")
            if r.get("err"):
                log("  " + r["err"])
            if r["status"] == "ok":
                hechos.append(mp4)
    log(f"=== {len(hechos)} videos ===")
    for m in hechos:
        log("  " + os.path.relpath(m, ROOT))
    return hechos


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solo", default="todos")
    ap.add_argument("--fps", type=int, default=FPS)
    a = ap.parse_args()
    ps = list(NOMBRE) if a.solo == "todos" else [a.solo]
    render(ps, a.fps)

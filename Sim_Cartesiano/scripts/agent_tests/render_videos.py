"""Videos de todas las corridas de formas y de tubo. Solo CPU: se lanza al final
para no competir por la GPU.

DOS FUENTES DISTINTAS
Todos los frames se dibujan AQUI, desde los campos periodicos submuestreados,
nunca en el solver. Medido: `Mesh.save_frame` tarda ~20 s por figura porque hace
pcolormesh sobre los ~2 M de celdas de la ventana refinada sin submuestrear, y
con 200 frames por corrida eso es mas tiempo de matplotlib que de simulacion, y
ademas bloqueando la GPU. Los campos volcados van a 900 columnas como mucho, asi
que aqui cada figura sale en decimas de segundo.

  - Formas: |u| y vorticidad, recortados a la ventana refinada, aspecto real.
  - Tubo: un canal de 30x1 con aspecto real es ilegible, asi que va con la
    escala vertical exagerada (aspect='auto').

El `scale=trunc(iw/2)*2` del filtro no es opcional: `bbox_inches="tight"` da
dimensiones impares que libx264 con yuv420p rechaza.

Uso:
    .venv/bin/python scripts/agent_tests/render_videos.py
    .venv/bin/python scripts/agent_tests/render_videos.py --solo formas
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

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FORMAS = os.path.join(ROOT, "results", "formas")
TUBO = os.path.join(ROOT, "results", "tubo")
FPS = 24


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def make_video(frames_dir, video_path, fps=FPS, patron="frame_*.png"):
    frames = sorted(glob.glob(os.path.join(frames_dir, patron)))
    if not frames:
        return {"status": "sin frames"}
    if shutil.which("ffmpeg") is None:
        return {"status": "ffmpeg no encontrado"}
    lista = os.path.join(frames_dir, "frames.ffconcat")
    dur = 1.0 / max(fps, 1)
    lineas = ["ffconcat version 1.0"]
    for f in frames:
        esc = os.path.abspath(f).replace("'", "'\\''")
        lineas += [f"file '{esc}'", f"duration {dur:.8f}"]
    lineas.append(f"file '{os.path.abspath(frames[-1])}'")
    with open(lista, "w") as fh:
        fh.write("\n".join(lineas) + "\n")
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "concat", "-safe", "0", "-i", lista,
           "-vf", f"fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", video_path]
    p = subprocess.run(cmd, text=True, capture_output=True)
    return {"status": "ok" if p.returncode == 0 else "fallo",
            "n": len(frames), "err": p.stderr.strip()[:300]}


def escala_robusta(campos, key, q=99.0):
    """Percentil global sobre una muestra de frames: si la escala cambia frame a
    frame el video parpadea y deja de ser comparable."""
    vals = []
    for f in campos[:: max(1, len(campos) // 12)]:
        d = np.load(f)
        a = np.asarray(d[key], dtype=np.float32)
        vals.append(np.nanpercentile(np.abs(a[np.isfinite(a)]), q))
    v = float(np.nanmax(vals)) if vals else 1.0
    return v if np.isfinite(v) and v > 0 else 1.0


def frames_forma(campos_dir, out_dir, tag, stream):
    """Un stream (|u| o vorticidad) de una corrida de forma, aspecto real."""
    campos = sorted(glob.glob(os.path.join(campos_dir, "f_*.npz")))
    if not campos:
        return 0
    # Incremental: si ya hay un PNG por cada campo, no se vuelve a dibujar. Deja
    # ir renderizando en CPU mientras la GPU sigue con las corridas que faltan.
    if len(glob.glob(os.path.join(out_dir, "frame_*.png"))) == len(campos):
        return len(campos)
    os.makedirs(out_dir, exist_ok=True)
    d0 = np.load(campos[0])
    x, y = np.asarray(d0["x"], float), np.asarray(d0["y"], float)
    # Escala fija en toda la corrida: si cambia frame a frame el video parpadea.
    if stream == "velocidad":
        vmin, vmax, cmap, etiq = 0.0, 2.2, "rainbow", "|u|"
    else:
        w = escala_robusta(campos, "v") * 25.0
        vmin, vmax, cmap, etiq = -w, w, "RdBu_r", r"$\omega_z$"
    ar = (y[-1] - y[0]) / (x[-1] - x[0])
    for f in campos:
        d = np.load(f)
        u = np.asarray(d["u"], np.float32)
        v = np.asarray(d["v"], np.float32)
        if stream == "velocidad":
            campo = np.hypot(u, v)
        else:
            campo = np.gradient(v, x, axis=1) - np.gradient(u, y, axis=0)
        campo = np.where(np.asarray(d["solid"]), np.nan, campo)
        fig, ax = plt.subplots(figsize=(11, max(3.0, 11 * ar + 1.2)))
        im = ax.pcolormesh(x, y, campo, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title(f"{tag}   {etiq}   t={float(d['t']):.2f}")
        fig.colorbar(im, ax=ax, label=etiq, fraction=0.025)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"frame_{int(d['iter']):06d}.png"), dpi=110)
        plt.close(fig)
    return len(campos)


def frames_tubo(campos_dir, out_dir, tag):
    campos = sorted(glob.glob(os.path.join(campos_dir, "f_*.npz")))
    if not campos:
        return 0
    # Incremental: si ya hay un PNG por cada campo, no se vuelve a dibujar. Deja
    # ir renderizando en CPU mientras la GPU sigue con las corridas que faltan.
    if len(glob.glob(os.path.join(out_dir, "frame_*.png"))) == len(campos):
        return len(campos)
    os.makedirs(out_dir, exist_ok=True)
    d0 = np.load(campos[0])
    x, y = np.asarray(d0["x"], float), np.asarray(d0["y"], float)
    umax = 1.6  # el maximo de Poiseuille es 1.5 U; fijarlo hace comparables los casos
    wmax = escala_robusta(campos, "u") * 12.0
    for f in campos:
        d = np.load(f)
        u = np.asarray(d["u"], np.float32)
        v = np.asarray(d["v"], np.float32)
        spd = np.hypot(u, v)
        w = np.gradient(v, x, axis=1) - np.gradient(u, y, axis=0)
        fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        ax[0].pcolormesh(x, y, spd, cmap="rainbow", vmin=0, vmax=umax, shading="auto")
        ax[0].set_aspect("auto"); ax[0].set_ylabel("y")
        ax[0].set_title(f"{tag}  |u|   t={float(d['t']):.2f}  (iter {int(d['iter'])})")
        ax[1].pcolormesh(x, y, w, cmap="RdBu_r", vmin=-wmax, vmax=wmax, shading="auto")
        ax[1].set_aspect("auto"); ax[1].set_ylabel("y"); ax[1].set_xlabel("x")
        ax[1].set_title("vorticidad")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"frame_{int(d['iter']):06d}.png"), dpi=110)
        plt.close(fig)
    return len(campos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", default="ambos", choices=("formas", "tubo", "ambos"))
    ap.add_argument("--fps", type=int, default=FPS)
    args = ap.parse_args()
    hechos = []

    if args.solo in ("formas", "ambos"):
        for campos in sorted(glob.glob(os.path.join(FORMAS, "*", "campos_t", "*"))):
            forma = campos.split(os.sep)[-3]
            tag = os.path.basename(campos)
            for stream in ("velocidad", "vorticidad"):
                fd = os.path.join(FORMAS, forma, "frames", tag, stream)
                n = frames_forma(campos, fd, f"{forma} {tag}", stream)
                if not n:
                    continue
                mp4 = os.path.join(FORMAS, forma, "videos", f"{tag}_{stream}.mp4")
                r = make_video(fd, mp4, args.fps)
                log(f"formas/{forma}/{tag}/{stream}: {r['status']} ({r.get('n')} frames)")
                if r["status"] == "ok":
                    hechos.append(mp4)

    if args.solo in ("tubo", "ambos"):
        for caso_dir in sorted(glob.glob(os.path.join(TUBO, "T*"))):
            tag = os.path.basename(caso_dir)
            campos = os.path.join(caso_dir, "campos_t")
            if not os.path.isdir(campos):
                continue
            fd = os.path.join(caso_dir, "frames")
            n = frames_tubo(campos, fd, tag)
            log(f"tubo/{tag}: {n} frames renderizados")
            if n:
                mp4 = os.path.join(caso_dir, "videos", f"{tag}.mp4")
                r = make_video(fd, mp4, args.fps)
                log(f"tubo/{tag}: {r['status']} ({r.get('n')} frames)")
                if r["status"] == "ok":
                    hechos.append(mp4)

    log(f"=== {len(hechos)} videos ===")
    for m in hechos:
        log("  " + os.path.relpath(m, ROOT))


if __name__ == "__main__":
    main()

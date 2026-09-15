"""
Corre un perfil ganador (masgen Re=1e5 o re1e3) con guardado de frames y
genera videos (velocidad dominio completo, velocidad zona refinada, presion).

Uso:
    .venv/bin/python scripts/agent_tests/video_ganador.py --tag re1e5
    .venv/bin/python scripts/agent_tests/video_ganador.py --tag re1e3
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Simulador2D import main as sim_main  # noqa: E402

OUT_ROOT = ROOT_DIR / "results" / "videos_ganadores"

GANADORES = {
    "re1e5": {
        "filepath": str(ROOT_DIR / "results/convergence_study/mixto_masgen/NACA_0012_sharp_Re100000_a4.0_LD18.18.dat"),
        "alpha_deg": 4.0,
        "nu": 1e-5,
        "dx_min": 0.002,
        "ld_refinado": 23.51,
    },
    "re1e3": {
        "filepath": str(ROOT_DIR / "results/convergence_study/mixto_re1e3/NACA_0012_sharp_Re1000_a4.0_LD5.01.dat"),
        "alpha_deg": 4.0,
        "nu": 1e-3,
        "dx_min": 0.002,
        "ld_refinado": 5.01,
    },
}

COMMON = dict(
    chord=1.0, v0x=1.0, v0y=0.0, rho=1.0, p0=0.0, CFL=0.5,
    Lx=8.0, Ly=5.0, cx=2.0, cy=None,
    factor_expansion=1.1, ancho_zona_fina_x=1.5, ancho_zona_fina_y=1.0,
    turb_model="sa", wall_treatment="consistent", advection_scheme="maccormack",
    transition_model="sa_bc", freestream_Tu=0.1,
    wake_refinement_mode="long_fine_x",
    stop_on_clcd_convergence=False,
    stop_on_convergence=False,
    graficos=False, live_view=False, mostrar_malla=False,
)

ITERACIONES = 40000
GUARDADO = 50
FRAME_EVERY = 50
FPS = 24


def make_video(frames_dir: Path, video_path: Path, fps: int) -> dict:
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        return {"status": "failed", "reason": "sin frames"}
    if shutil.which("ffmpeg") is None:
        return {"status": "failed", "reason": "ffmpeg no encontrado"}
    list_path = frames_dir / "frames.ffconcat"
    duration = 1.0 / max(fps, 1)
    lines = ["ffconcat version 1.0"]
    for f in frames:
        esc = str(f.resolve()).replace("'", "'\\''")
        lines.append(f"file '{esc}'")
        lines.append(f"duration {duration:.8f}")
    esc_last = str(frames[-1].resolve()).replace("'", "'\\''")
    lines.append(f"file '{esc_last}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-vf", f"fps={fps},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", str(video_path),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT_DIR), text=True, capture_output=True)
    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "n_frames": len(frames),
        "elapsed_s": time.time() - t0,
        "stderr": proc.stderr.strip(),
    }


def summarize(vec) -> tuple:
    try:
        arr = np.asarray(cp.asnumpy(vec), dtype=np.float64).ravel()
    except Exception:
        arr = np.asarray([], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    w = max(1, arr.size // 5)
    tail = arr[-w:]
    return float(np.mean(tail)), float(tail[-1]), float(np.std(tail))


def run(tag: str) -> None:
    g = GANADORES[tag]
    out_dir = OUT_ROOT / tag
    frames_full = out_dir / "frames_velocidad_dominio_completo"
    frames_refined = out_dir / "frames_velocidad_zona_refinada"
    for d in (frames_full, frames_refined):
        d.mkdir(parents=True, exist_ok=True)

    x_half = 0.5 * COMMON["ancho_zona_fina_x"] + 0.10
    y_half = 0.5 * COMMON["ancho_zona_fina_y"] + 0.10
    x_center = COMMON["cx"] + 0.5  # cx + chord/2
    y_center = COMMON["Ly"] / 2.0
    xlim = (max(0.0, x_center - x_half), min(COMMON["Lx"], x_center + x_half))
    ylim = (max(0.0, y_center - y_half), min(COMMON["Ly"], y_center + y_half))

    cfg = dict(COMMON)
    cfg.update(
        filepath=g["filepath"],
        alpha_deg=g["alpha_deg"],
        nu=g["nu"],
        dx_min=g["dx_min"],
        iteraciones=ITERACIONES,
        guardado=GUARDADO,
        save_frames=True,
        frames_dir_grueso=str(frames_full),
        frames_dir_refinado=str(frames_refined),
        save_frames_cada=FRAME_EVERY,
        save_frame_refined_xlim=xlim,
        save_frame_refined_ylim=ylim,
        save_frame_dpi=150,
    )

    (out_dir / "run_config.json").write_text(
        json.dumps(cfg, indent=2, default=str), encoding="utf-8"
    )

    print(f"[{tag}] Re={1.0/g['nu']:.0e}  alpha={g['alpha_deg']}  dx={g['dx_min']}  "
          f"iters={ITERACIONES}  frame_every={FRAME_EVERY}  -> {out_dir}")

    t0 = time.time()
    mesh = sim_main(**cfg)
    elapsed = time.time() - t0

    cd_mean, cd_final, cd_std = summarize(mesh.cdvector)
    cl_mean, cl_final, cl_std = summarize(mesh.clvector)

    videos = {
        "velocidad_dominio_completo": make_video(
            frames_full, out_dir / f"{tag}_velocidad_dominio_completo.mp4", FPS),
        "velocidad_zona_refinada": make_video(
            frames_refined, out_dir / f"{tag}_velocidad_zona_refinada.mp4", FPS),
    }

    meta = {
        "tag": tag, "Re": 1.0 / g["nu"], "alpha_deg": g["alpha_deg"],
        "dx_min": g["dx_min"], "iteraciones": ITERACIONES,
        "ld_refinado_estudio": g["ld_refinado"],
        "Cl_mean_sim": cl_mean, "Cl_final_sim": cl_final, "Cl_std_sim": cl_std,
        "Cd_mean_sim": cd_mean, "Cd_final_sim": cd_final, "Cd_std_sim": cd_std,
        "LD_mean_sim": cl_mean / cd_mean if abs(cd_mean) > 1e-9 else None,
        "elapsed_sim_s": elapsed,
        "videos": videos,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print(f"[{tag}] DONE en {elapsed/60:.1f} min. L/D_sim={meta['LD_mean_sim']}")
    for k, v in videos.items():
        print(f"  video {k}: {v.get('status')} ({v.get('n_frames')} frames)")
        if v.get("stderr"):
            print("   ", v["stderr"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=["re1e5", "re1e3", "all"], default="all")
    args = ap.parse_args()
    tags = ["re1e5", "re1e3"] if args.tag == "all" else [args.tag]
    for t in tags:
        run(t)

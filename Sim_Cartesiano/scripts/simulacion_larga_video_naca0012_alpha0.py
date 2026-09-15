"""
Corre una simulacion larga NACA0012 a alpha=0 y genera videos/frames
siguiendo el mismo patron que los scripts largos existentes.

Uso:
    .venv/bin/python scripts/simulacion_larga_video_naca0012_alpha0.py --force
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-cache"))

from Simulador2D import main as sim_main  # noqa: E402
from sim_defaults import PROJECTION_DEFAULTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-suffix", default="naca0012_alpha0_long_video")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--iteraciones", type=int, default=30000)
    parser.add_argument("--guardado", type=int, default=100)
    parser.add_argument("--frame-every", type=int, default=100)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--frame-dpi", type=int, default=180)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--reynolds", type=float, default=100_000.0)
    parser.add_argument("--dx-min", type=float, default=0.001)
    parser.add_argument("--u-inf", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--cfl", type=float, default=0.5)
    parser.add_argument("--lx", type=float, default=12.0)
    parser.add_argument("--ly", type=float, default=8.0)
    parser.add_argument("--cx", type=float, default=2.0)
    parser.add_argument("--cy", type=float, default=4.0)
    parser.add_argument("--factor-expansion", type=float, default=1.10)
    parser.add_argument("--fine-width-x", type=float, default=1.2)
    parser.add_argument("--fine-width-y", type=float, default=1.0)
    parser.add_argument("--ratio-max-malla", type=float, default=100.0)
    parser.add_argument("--divergencia", type=float, default=0.10)
    parser.add_argument("--refined-pad-x", type=float, default=0.10)
    parser.add_argument("--refined-pad-y", type=float, default=0.10)
    parser.add_argument("--no-wale", action="store_true")
    parser.add_argument("--wale-cw", type=float, default=0.15)
    parser.add_argument("--long-fine-x", action="store_true")
    parser.add_argument("--no-turbo-hd", action="store_true")
    return parser.parse_args()


def safe_suffix(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())


def output_dir(args: argparse.Namespace) -> Path:
    suffix = safe_suffix(args.output_suffix or "naca0012_alpha0_long_video")
    return ROOT_DIR / "results" / "videos" / suffix


def refined_limits(args: argparse.Namespace) -> tuple[tuple[float, float], tuple[float, float]]:
    chord = 1.0
    x_center = float(args.cx) + 0.5 * chord
    y_center = float(args.cy)
    x_half = 0.5 * float(args.fine_width_x) + float(args.refined_pad_x)
    y_half = 0.5 * float(args.fine_width_y) + float(args.refined_pad_y)
    xlim = (max(0.0, x_center - x_half), min(float(args.lx), x_center + x_half))
    ylim = (max(0.0, y_center - y_half), min(float(args.ly), y_center + y_half))
    return xlim, ylim


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    extra = sorted({k for row in rows for k in row if k not in keys})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys + extra, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_vector(vec: Any, discard_frac: float = 0.30) -> tuple[float, float, float]:
    try:
        arr = np.asarray(cp.asnumpy(vec) if hasattr(vec, "shape") else vec, dtype=np.float64).ravel()
    except Exception:
        arr = np.asarray([], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    start = int(min(arr.size - 1, max(0, math.floor(discard_frac * arr.size))))
    tail = arr[start:] if start < arr.size else arr[-1:]
    return float(np.mean(tail)), float(tail[-1]), float(np.std(tail))


def make_video(frames_dir: Path, video_path: Path, fps: int) -> dict[str, Any]:
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        return {"status": "failed", "reason": "sin frames", "video": str(video_path)}
    if shutil.which("ffmpeg") is None:
        return {"status": "failed", "reason": "ffmpeg no encontrado en PATH", "video": str(video_path)}

    list_path = frames_dir / "frames.ffconcat"
    duration = 1.0 / max(int(fps), 1)
    lines = ["ffconcat version 1.0"]
    for frame in frames:
        escaped = str(frame.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
        lines.append(f"duration {duration:.8f}")
    escaped_last = str(frames[-1].resolve()).replace("'", "'\\''")
    lines.append(f"file '{escaped_last}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-vf",
        f"fps={int(fps)},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        str(video_path),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT_DIR), text=True, capture_output=True)
    return {
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "video": str(video_path.relative_to(ROOT_DIR)) if video_path.exists() else str(video_path),
        "n_frames": len(frames),
        "elapsed_s": time.time() - t0,
        "stderr": proc.stderr.strip(),
    }


def build_config(
    args: argparse.Namespace,
    frames_full: Path,
    frames_refined: Path,
    frames_pressure: Path,
) -> dict[str, Any]:
    xlim, ylim = refined_limits(args)
    nu = float(args.u_inf) / max(float(args.reynolds), 1e-30)
    return {
        **PROJECTION_DEFAULTS,
        "filepath": "profiles/NACA_0012",
        "chord": 1.0,
        "alpha_deg": float(args.alpha),
        "Lx": float(args.lx),
        "Ly": float(args.ly),
        "cx": float(args.cx),
        "cy": float(args.cy),
        "dx_min": float(args.dx_min),
        "factor_expansion": float(args.factor_expansion),
        "ancho_zona_fina_x": float(args.fine_width_x),
        "ancho_zona_fina_y": float(args.fine_width_y),
        "ratio_max_malla": float(args.ratio_max_malla),
        "v0x": float(args.u_inf),
        "v0y": 0.0,
        "rho": float(args.rho),
        "p0": 0.0,
        "nu": nu,
        "CFL": float(args.cfl),
        "iteraciones": int(args.iteraciones),
        "guardado": int(args.guardado),
        "divergencia": float(args.divergencia),
        "usar_wale": not bool(args.no_wale),
        "wale_Cw": float(args.wale_cw),
        "graficos": False,
        "save_frames": True,
        "frames_dir_grueso": str(frames_full),
        "frames_dir_refinado": str(frames_refined),
        "frames_dir_presion": str(frames_pressure),
        "save_frames_cada": int(args.frame_every),
        "save_frame_refined_xlim": xlim,
        "save_frame_refined_ylim": ylim,
        "save_frame_dpi": int(args.frame_dpi),
        "live_view": False,
        "mostrar_malla": False,
        "stop_on_convergence": False,
        "corregir_deriva_vertical": False,
        "ibm_wall_mode": "ghost_noslip",
        "wake_refinement_mode": "long_fine_x" if args.long_fine_x else "base",
        "mg_modo_turbo_hd": not bool(args.no_turbo_hd),
        "mg_modo_turbo_ultra": False,
    }


def main() -> int:
    args = parse_args()
    out_dir = output_dir(args)
    frames_full = out_dir / "frames_velocidad_dominio_completo"
    frames_refined = out_dir / "frames_velocidad_zona_refinada"
    frames_pressure = out_dir / "frames_presion_dominio_completo"
    video_full = out_dir / "naca0012_alpha0_velocidad_dominio_completo.mp4"
    video_refined = out_dir / "naca0012_alpha0_velocidad_zona_refinada.mp4"
    video_pressure = out_dir / "naca0012_alpha0_presion_dominio_completo.mp4"

    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_full.mkdir(parents=True, exist_ok=True)
    frames_refined.mkdir(parents=True, exist_ok=True)
    frames_pressure.mkdir(parents=True, exist_ok=True)

    cfg = build_config(args, frames_full, frames_refined, frames_pressure)
    (out_dir / "run_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"Salida: {out_dir.relative_to(ROOT_DIR)}")
    print(f"Frames velocidad dominio completo: {frames_full.relative_to(ROOT_DIR)}")
    print(f"Frames velocidad zona refinada: {frames_refined.relative_to(ROOT_DIR)}")
    print(f"Frames presion dominio completo: {frames_pressure.relative_to(ROOT_DIR)}")
    print(f"Recorte refinado x={cfg['save_frame_refined_xlim']} y={cfg['save_frame_refined_ylim']}")

    t0 = time.time()
    mesh = sim_main(**cfg)
    elapsed = time.time() - t0

    cd_mean, cd_final, cd_std = summarize_vector(mesh.cdvector)
    cl_mean, cl_final, cl_std = summarize_vector(mesh.clvector)
    div_mean, div_final, div_std = summarize_vector(getattr(mesh, "divvector_flux", []))
    wall_mean, wall_final, wall_std = summarize_vector(getattr(mesh, "wall_leak_max_vector", []))

    summary = [{
        "status": "ok",
        "alpha_deg": float(args.alpha),
        "Re": float(args.reynolds),
        "dx_min": float(args.dx_min),
        "iteraciones": int(args.iteraciones),
        "guardado": int(args.guardado),
        "frame_every": int(args.frame_every),
        "n_frames_velocity_full": len(list(frames_full.glob("frame_*.png"))),
        "n_frames_velocity_refined": len(list(frames_refined.glob("frame_*.png"))),
        "n_frames_pressure_full": len(list(frames_pressure.glob("frame_*.png"))),
        "Cd_mean": cd_mean,
        "Cd_final": cd_final,
        "Cd_std": cd_std,
        "Cl_mean": cl_mean,
        "Cl_final": cl_final,
        "Cl_std": cl_std,
        "div_flux_mean": div_mean,
        "div_flux_final": div_final,
        "div_flux_std": div_std,
        "wall_leak_max_mean": wall_mean,
        "wall_leak_max_final": wall_final,
        "wall_leak_max_std": wall_std,
        "elapsed_sim_s": elapsed,
    }]
    write_csv(out_dir / "summary.csv", summary)

    videos = {
        "velocidad_dominio_completo": make_video(frames_full, video_full, int(args.fps)),
        "velocidad_zona_refinada": make_video(frames_refined, video_refined, int(args.fps)),
        "presion_dominio_completo": make_video(frames_pressure, video_pressure, int(args.fps)),
    }
    meta = {
        "output_dir": str(out_dir.relative_to(ROOT_DIR)),
        "frames_velocity_full": str(frames_full.relative_to(ROOT_DIR)),
        "frames_velocity_refined": str(frames_refined.relative_to(ROOT_DIR)),
        "frames_pressure_full": str(frames_pressure.relative_to(ROOT_DIR)),
        "videos": videos,
        "summary": summary[0],
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print("Videos:")
    for key, info in videos.items():
        print(f"- {key}: {info.get('status')} {info.get('video')}")
        if info.get("stderr"):
            print(info["stderr"])
    return 0 if all(v.get("status") == "ok" for v in videos.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

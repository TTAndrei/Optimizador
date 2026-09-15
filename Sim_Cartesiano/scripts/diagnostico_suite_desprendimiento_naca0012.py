"""
Suite para aislar el origen del desprendimiento espurio en NACA0012.

La suite reutiliza el diagnostico nocturno existente para guardar metricas,
auditoria de capa limite, campos finales, frames y videos por cada caso.

Uso recomendado para dejarlo por la noche:
    .venv/bin/python scripts/diagnostico_suite_desprendimiento_naca0012.py --preset full --force

Reanudar sin repetir casos ya terminados:
    .venv/bin/python scripts/diagnostico_suite_desprendimiento_naca0012.py --preset full
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

from diagnostico_nocturno_boundary_layer_naca0012 import (
    Case,
    ROOT_DIR,
    build_base_config,
    load_existing,
    run_case,
    safe_name,
    save_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-suffix", default="naca0012_suite_desprendimiento")
    parser.add_argument("--preset", choices=("smoke", "targeted", "full"), default="targeted")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--reynolds", type=float, default=100_000.0)
    parser.add_argument(
        "--target-time",
        type=float,
        default=15.0,
        help="Tiempo convectivo U*t/c comun para comparar casos con distinto CFL o dx.",
    )
    parser.add_argument(
        "--dt-target",
        type=float,
        default=2.5e-4,
        help="Paso temporal objetivo para los casos de dx variable a dt constante.",
    )
    parser.add_argument("--iteraciones", type=int, default=30_000)
    parser.add_argument("--guardado", type=int, default=100)
    parser.add_argument("--frame-every", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frame-dpi", type=int, default=160)
    parser.add_argument("--dx-min", type=float, default=0.001)
    parser.add_argument("--u-inf", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--cfl", type=float, default=0.50)
    parser.add_argument("--lx", type=float, default=12.0)
    parser.add_argument("--ly", type=float, default=8.0)
    parser.add_argument("--cx", type=float, default=2.0)
    parser.add_argument("--cy", type=float, default=4.0)
    parser.add_argument("--factor-expansion", type=float, default=1.10)
    parser.add_argument("--fine-width-x", type=float, default=1.2)
    parser.add_argument("--fine-width-y", type=float, default=1.0)
    parser.add_argument("--ratio-max-malla", type=float, default=100.0)
    parser.add_argument("--divergencia", type=float, default=0.10)
    parser.add_argument("--wale-cw", type=float, default=0.15)
    return parser.parse_args()


def ceil_to_multiple(value: float, multiple: int) -> int:
    step = max(int(multiple), 1)
    return max(step, int(math.ceil(float(value) / step) * step))


def dt_from_cfl(cfl: float, dx_min: float, u_inf: float) -> float:
    return float(cfl) * float(dx_min) / max(float(u_inf), 1e-30)


def iterations_for(args: argparse.Namespace, cfl: float, dx_min: float) -> int:
    dt = dt_from_cfl(cfl, dx_min, args.u_inf)
    return ceil_to_multiple(float(args.target_time) / max(dt, 1e-30), int(args.guardado))


def case_with_timing(
    args: argparse.Namespace,
    case_id: str,
    description: str,
    overrides: dict[str, Any],
    *,
    cfl: float | None = None,
    dx_min: float | None = None,
    dt: float | None = None,
) -> Case:
    cfg = dict(overrides)
    case_cfl = float(cfg.get("CFL", args.cfl if cfl is None else cfl))
    case_dx = float(cfg.get("dx_min", args.dx_min if dx_min is None else dx_min))
    if cfl is not None:
        cfg["CFL"] = float(cfl)
    if dx_min is not None:
        cfg["dx_min"] = float(dx_min)
    if dt is not None:
        case_cfl = float(dt) * float(args.u_inf) / max(case_dx, 1e-30)
        cfg["CFL"] = case_cfl
        cfg["iteraciones"] = ceil_to_multiple(float(args.target_time) / max(float(dt), 1e-30), int(args.guardado))
    else:
        cfg["iteraciones"] = iterations_for(args, case_cfl, case_dx)
    cfg["save_frames_cada"] = int(args.frame_every)
    return Case(case_id, description, cfg)


def build_cases(args: argparse.Namespace) -> list[Case]:
    if args.preset == "smoke":
        return [
            Case("base_smoke", "Validacion corta de escritura de metricas y frames", {"iteraciones": 200, "save_frames_cada": 50}),
            Case(
                "cfl025_smoke",
                "Validacion corta de caso CFL bajo",
                {"CFL": 0.25, "iteraciones": 200, "save_frames_cada": 50},
            ),
        ]

    cfl_cases = [
        case_with_timing(args, "cfl_050", "Baseline temporal: CFL 0.50, dx base", {}, cfl=0.50),
        case_with_timing(args, "cfl_040", "CFL 0.40, mismo dx y mismo tiempo fisico", {}, cfl=0.40),
        case_with_timing(args, "cfl_035", "CFL 0.35, mismo dx y mismo tiempo fisico", {}, cfl=0.35),
        case_with_timing(args, "cfl_030", "CFL 0.30, mismo dx y mismo tiempo fisico", {}, cfl=0.30),
        case_with_timing(args, "cfl_025", "CFL 0.25, mismo dx y mismo tiempo fisico", {}, cfl=0.25),
    ]
    dt_resolution_cases = [
        case_with_timing(
            args,
            "dt025_dx0750",
            "dx_min 0.00075 con dt constante; separa resolucion de paso temporal",
            {},
            dx_min=0.00075,
            dt=args.dt_target,
        ),
        case_with_timing(
            args,
            "dt025_dx0500",
            "dx_min 0.00050 con dt constante; comprueba si y+ y delta99 arreglan el campo",
            {"ancho_zona_fina_y": max(float(args.fine_width_y), 1.4)},
            dx_min=0.00050,
            dt=args.dt_target,
        ),
    ]
    te_ibm_cases = [
        case_with_timing(
            args,
            "te_factor_4",
            "Borde de salida mas grueso numericamente: min_te_height_factor 4",
            {"min_te_height_factor": 4.0},
            cfl=0.50,
        ),
        case_with_timing(
            args,
            "te_factor_8",
            "Borde de salida mucho mas grueso numericamente: min_te_height_factor 8",
            {"min_te_height_factor": 8.0},
            cfl=0.50,
        ),
        case_with_timing(
            args,
            "ibm_solid_zero",
            "Diagnostico IBM: fuerza solido a velocidad cero sin ghost noslip",
            {"ibm_wall_mode": "solid_zero_only"},
            cfl=0.50,
        ),
    ]
    wake_cases = [
        case_with_timing(
            args,
            "long_wake_cfl050",
            "Estela refinada larga con CFL 0.50",
            {"wake_refinement_mode": "long_fine_x", "ancho_zona_fina_x": 2.4},
            cfl=0.50,
        ),
        case_with_timing(
            args,
            "long_wake_cfl025",
            "Estela refinada larga + CFL 0.25",
            {"wake_refinement_mode": "long_fine_x", "ancho_zona_fina_x": 2.4},
            cfl=0.25,
        ),
    ]

    if args.preset == "targeted":
        return [
            cfl_cases[0],
            cfl_cases[2],
            cfl_cases[4],
            dt_resolution_cases[1],
            te_ibm_cases[1],
            wake_cases[1],
        ]
    return cfl_cases + dt_resolution_cases + te_ibm_cases + wake_cases


def enrich_row(row: dict[str, Any], case: Case, args: argparse.Namespace) -> dict[str, Any]:
    cfg = case.overrides
    cfl = float(row.get("CFL", cfg.get("CFL", args.cfl)))
    dx_min = float(row.get("dx_min", cfg.get("dx_min", args.dx_min)))
    iterations = int(row.get("iteraciones", cfg.get("iteraciones", args.iteraciones)))
    dt_est = dt_from_cfl(cfl, dx_min, args.u_inf)
    row.update(
        {
            "dt_est": dt_est,
            "physical_time_est": dt_est * iterations,
            "target_time": float(args.target_time),
            "dt_target": float(args.dt_target),
            "min_te_height_factor": cfg.get("min_te_height_factor", 2.0),
            "ibm_wall_mode": cfg.get("ibm_wall_mode", "ghost_noslip"),
        }
    )
    case_summary = ROOT_DIR / str(row.get("case_dir", "")) / "case_summary.json"
    if case_summary.exists():
        case_summary.write_text(json.dumps(row, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    return row


def write_suite_notes(out_dir: Path, args: argparse.Namespace, cases: list[Case]) -> None:
    lines = [
        "# Suite desprendimiento NACA0012",
        "",
        "Objetivo: aislar si el desprendimiento espurio aparece por paso temporal, resolucion normal, borde de salida/IBM o refinamiento de estela.",
        "",
        f"- alpha: {args.alpha:g} deg",
        f"- Re: {args.reynolds:g}",
        f"- tiempo fisico objetivo U*t/c: {args.target_time:g}",
        f"- frame cada: {args.frame_every} iteraciones",
        f"- videos: {'no' if args.no_videos else 'si'}",
        "",
        "| case | iteraciones | CFL | dx_min | dt_est | notas |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for case in cases:
        cfg = case.overrides
        cfl = float(cfg.get("CFL", args.cfl))
        dx = float(cfg.get("dx_min", args.dx_min))
        iters = int(cfg.get("iteraciones", args.iteraciones))
        lines.append(
            f"| {case.case_id} | {iters} | {cfl:g} | {dx:g} | {dt_from_cfl(cfl, dx, args.u_inf):.6g} | {case.description} |"
        )
    lines.extend(
        [
            "",
            "Lectura esperada:",
            "- Si al bajar CFL desaparece o se retrasa el desprendimiento, el origen principal es temporal/numerico.",
            "- Si `dt025_dx0500` mejora frente a `cfl_025`, faltaba resolucion de capa limite.",
            "- Si `te_factor_*` cambia mucho el resultado, el borde de salida esta inyectando perturbaciones numericas.",
            "- Si `ibm_solid_zero` cambia mucho el resultado, el acoplamiento IBM/ghost debe revisarse.",
            "- Si `long_wake_cfl025` mejora pero `cfl_025` no, la transicion de estela/malla contamina el campo.",
            "",
            "Artefactos por caso:",
            "- `history.csv` y `history.png`: Cl, Cd, divergencia y fuga de pared.",
            "- `surface_boundary_layer.csv` y `boundary_layer_audit.png`: y+, delta99 teorica, Cf y proxy de separacion.",
            "- `cp_profile_mean.csv`: Cp medio e integral de Cp.",
            "- `state_final.npz`: u, v, p, solido y malla final.",
            "- `videos/velocity_full.mp4`, `videos/velocity_zoom.mp4`, `videos/pressure_full.mp4`: evolucion del campo.",
        ]
    )
    (out_dir / "suite_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = ROOT_DIR / "results" / "diagnosticos" / safe_name(args.output_suffix)
    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(args)
    base_cfg = build_base_config(args, out_dir)
    base_cfg["iteraciones"] = int(args.iteraciones)
    base_cfg["save_frames_cada"] = int(args.frame_every)
    write_suite_notes(out_dir, args, cases)
    (out_dir / "suite_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "base_config": base_cfg,
                "cases": [c.__dict__ for c in cases],
            },
            indent=2,
            ensure_ascii=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    rows_by_id = load_existing(out_dir / "summary.json")
    print(f"Salida: {out_dir.relative_to(ROOT_DIR)}", flush=True)
    print(f"Casos: {', '.join(c.case_id for c in cases)}", flush=True)
    print(f"Videos: {'desactivados' if args.no_videos else 'activados'}; frame cada {args.frame_every} iteraciones", flush=True)
    for case in cases:
        if case.case_id in rows_by_id and not args.force:
            print(f"[skip] {case.case_id}: ya completado", flush=True)
            rows_by_id[case.case_id] = enrich_row(rows_by_id[case.case_id], case, args)
            continue
        try:
            rows_by_id[case.case_id] = enrich_row(run_case(case, base_cfg, out_dir, args), case, args)
        except Exception as exc:
            rows_by_id[case.case_id] = {
                "case_id": case.case_id,
                "status": "failed",
                "description": case.description,
                "error": str(exc),
            }
            print(f"[failed] {case.case_id}: {exc}", flush=True)
        save_summary(out_dir, rows_by_id, cases)

    print(f"Resumen: {(out_dir / 'summary.csv').relative_to(ROOT_DIR)}", flush=True)
    print(f"Informe: {(out_dir / 'analysis_report.md').relative_to(ROOT_DIR)}", flush=True)
    print(f"Notas: {(out_dir / 'suite_notes.md').relative_to(ROOT_DIR)}", flush=True)
    failed = [row for row in rows_by_id.values() if row.get("status") != "ok"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

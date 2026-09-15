from __future__ import annotations

import numpy as np
import cupy as cp
import inspect
import pytest

from Simulador2D import Mesh, main as sim_main


def _cuda_available() -> bool:
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="CUDA device required")


def _make_mesh(nonuniform: bool = False) -> Mesh:
    if nonuniform:
        x = np.array([0.0, 0.12, 0.29, 0.5, 0.78, 1.0], dtype=np.float64)
        y = np.array([0.0, 0.18, 0.37, 0.63, 1.0], dtype=np.float64)
    else:
        x = np.linspace(0.0, 1.0, 6, dtype=np.float64)
        y = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    mesh = Mesh(
        Lx=1.0,
        Ly=1.0,
        p0=0.0,
        v0x=0.0,
        v0y=0.0,
        dx=float(np.min(np.diff(x))),
        dy=float(np.min(np.diff(y))),
        usar_wale=False,
        X_1d=x,
        Y_1d=y,
    )
    mesh._U_ref = 1.0
    return mesh


@requires_cuda
def test_flux_faces_preserve_uniform_field_and_zero_divergence():
    mesh = _make_mesh(nonuniform=False)
    mesh.u[:] = cp.float32(1.25)
    mesh.v[:] = cp.float32(-0.75)

    u_face_x, v_face_y = mesh._build_projection_faces()
    assert float(cp.max(cp.abs(u_face_x - cp.float32(1.25)))) < 1e-6
    assert float(cp.max(cp.abs(v_face_y - cp.float32(-0.75)))) < 1e-6

    div = mesh._compute_flux_divergence_field_faces(
        u_face_x,
        v_face_y,
        boundary_u_w=mesh.u[:, 0],
        boundary_u_e=mesh.u[:, -1],
        boundary_v_s=mesh.v[0, :],
        boundary_v_n=mesh.v[-1, :],
    )
    assert float(cp.max(cp.abs(div))) < 1e-6


@requires_cuda
def test_pressure_gradient_correction_is_uniform_on_nonuniform_mesh():
    mesh = _make_mesh(nonuniform=True)
    grad_px = 2.0
    coef = 0.1
    mesh.u[:] = cp.float32(0.0)
    mesh.v[:] = cp.float32(0.0)
    mesh.p[:] = cp.float32(grad_px) * mesh.XX

    u_face_x, v_face_y = mesh._build_projection_faces()
    mesh._apply_pressure_correction_to_faces(u_face_x, v_face_y, mesh.p, cp.float32(coef))
    u_new, v_new = mesh._reconstruct_centered_velocity_from_faces(
        u_face_x, v_face_y, u_prev=mesh.u, v_prev=mesh.v)

    expected_u = -coef * grad_px
    assert float(cp.max(cp.abs(u_face_x - cp.float32(expected_u)))) < 1e-5
    assert float(cp.max(cp.abs(u_new[:, 1:-1] - cp.float32(expected_u)))) < 1e-5
    assert float(cp.max(cp.abs(v_new[1:-1, :]))) < 1e-6


def _make_wall_mesh() -> Mesh:
    x = np.linspace(0.0, 1.0, 33, dtype=np.float64)
    y = np.linspace(0.0, 1.0, 33, dtype=np.float64)
    mesh = Mesh(
        Lx=1.0,
        Ly=1.0,
        p0=0.0,
        v0x=0.0,
        v0y=0.0,
        dx=float(np.min(np.diff(x))),
        dy=float(np.min(np.diff(y))),
        usar_wale=False,
        X_1d=x,
        Y_1d=y,
    )
    mesh.add_solid_rectangle(0.42, 0.42, 0.58, 0.58)
    mesh._U_ref = 1.0
    mesh.u[:] = cp.float32(0.0)
    mesh.v[:] = cp.float32(0.25)
    return mesh


@requires_cuda
def test_compatible_flux_reduces_wall_leak_on_simple_wall():
    legacy = _make_wall_mesh()
    compat = _make_wall_mesh()

    legacy.project_multigrid(
        rho_sim=1.0,
        dt=0.05,
        tol_div=1e-4,
        max_outer=2,
        cycles_per_outer=2,
        niveles_max=0,
        projection_variant="legacy_centered",
        verbose=False,
    )
    compat.project_multigrid(
        rho_sim=1.0,
        dt=0.05,
        tol_div=1e-4,
        max_outer=2,
        cycles_per_outer=2,
        niveles_max=0,
        projection_variant="compatible_flux",
        verbose=False,
    )

    _, legacy_max = legacy.compute_wall_leak_metrics()
    _, compat_max = compat.compute_wall_leak_metrics()
    assert compat_max <= legacy_max + 1e-5


def test_mg_pressure_accumulation_default_is_outer_sum():
    mesh_sig = inspect.signature(Mesh.project_multigrid)
    main_sig = inspect.signature(sim_main)
    assert mesh_sig.parameters["mg_pressure_accumulation"].default == "outer_sum"
    assert main_sig.parameters["mg_pressure_accumulation"].default == "outer_sum"


@requires_cuda
def test_surface_force_audit_matches_pressure_force_integral():
    mesh = _make_wall_mesh()
    mesh.p[:] = cp.sin(cp.float32(np.pi) * mesh.XX) * cp.cos(cp.float32(np.pi) * mesh.YY)

    audit = mesh.extract_surface_force_audit(mu=0.0, rho=1.0, chord=1.0, n_extrap_layers=3)
    rows = audit["rows"]
    summary = audit["summary"]
    forces = mesh.compute_drag_lift(mu=0.0, rho=1.0, n_extrap_layers=3)

    assert rows
    assert summary["n_surface_faces"] == len(rows)
    assert abs(summary["Lift_p_audit"] - forces["Lift_p"]) < 1e-6
    assert abs(sum(r["dLift_p"] for r in rows) - forces["Lift_p"]) < 1e-6

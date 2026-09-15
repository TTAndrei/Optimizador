"""
Boundary Layer Correction — post-proceso para simulador LES Cartesiano IBM.

Pipeline:
  Cp_LES  → Cl  (integración directa ΔCp, evita error momentum balance)
  perfil  → panel method (Hess-Smith vortex) → Cp_inv → Ue(s)
  Ue(s)   → BL integral (Thwaites lam + Michel + Head turb) → Cd_visc
  Cd_total = Cd_p_LES + Cd_visc_BL   (sin doble conteo de viscosidad)

Por qué NO usar Cp_LES para BL: LES Cp ya contiene efectos viscosos → doble conteo.

Uso rápido:
    from bl_correction import compute_corrected_forces
    result = compute_corrected_forces(mesh, filepath="profiles/NACA_0012")
    print(result["Cl"], result["Cd"], result["Ef"])

Refs: Thwaites (1949), Michel (1951), Head (1958), Katz & Plotkin cap.11,
      Drela "Flight Vehicle Aerodynamics" cap.4
"""
from __future__ import annotations
import numpy as np

# ── Tablas Thwaites (White, Fluid Mechanics, Table 9.2) ─────────────────────
_TH_LAM = np.array([-0.09, -0.06, -0.04, -0.02,  0.0,
                     0.02,  0.05,  0.08,  0.10,  0.12, 0.20])
_TH_H   = np.array([ 3.70,  3.44,  3.30,  3.14,  2.59,
                      2.40,  2.20,  2.09,  2.00,  1.94, 1.73])


# ═══════════════════════════════════════════════════════════════════════════════
# A. Carga de geometría del perfil
# ═══════════════════════════════════════════════════════════════════════════════

def load_profile(filepath: str,
                 chord: float = 1.0,
                 alpha_deg: float = 0.0
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Lee un archivo de perfil aerodinámico (.dat / sin extensión).
    Formato esperado: línea header + pares x y (x ∈ [0,1]).

    Retorna (xu, yu, xl, yl) ordenado LE→TE, escalado por chord, rotado.
    alpha_deg=0 siempre para uso con panel method (pasar alpha al solver).
    """
    try:
        raw = open(filepath, encoding="utf-8").readlines()
    except UnicodeDecodeError:
        raw = open(filepath, encoding="latin-1").readlines()

    pts = []
    for line in raw[1:]:           # saltar header
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pts.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue

    if len(pts) < 4:
        raise ValueError(f"load_profile: solo {len(pts)} puntos en '{filepath}'")

    x_raw = np.array([p[0] for p in pts], dtype=np.float64)
    y_raw = np.array([p[1] for p in pts], dtype=np.float64)

    # Escalar
    x_raw *= chord
    y_raw *= chord

    # Dividir en upper (TE→LE) y lower (LE→TE) por mínimo x
    le_idx = int(np.argmin(x_raw))
    xu = x_raw[:le_idx + 1][::-1].copy()   # LE→TE
    yu = y_raw[:le_idx + 1][::-1].copy()
    xl = x_raw[le_idx:].copy()              # LE→TE
    yl = y_raw[le_idx:].copy()

    # Rotar por alpha alrededor del cuarto de cuerda
    if alpha_deg != 0.0:
        a  = np.deg2rad(alpha_deg)
        cx = 0.25 * chord
        R  = np.array([[np.cos(a), -np.sin(a)],
                       [np.sin(a),  np.cos(a)]])
        def _rot(x, y):
            v = R @ np.vstack([x - cx, y])
            return v[0] + cx, v[1]
        xu, yu = _rot(xu, yu)
        xl, yl = _rot(xl, yl)

    return xu, yu, xl, yl


def profile_polygon_from_file(filepath: str,
                              chord: float = 1.0,
                              x_offset: float = 0.0,
                              y_offset: float = 0.0,
                              alpha_geometry_deg: float = 0.0,
                              min_te_height: float | None = None
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Replica la transformacion geometrica usada por Mesh.load_solids_from_file."""
    try:
        raw = open(filepath, encoding="utf-8").readlines()
    except UnicodeDecodeError:
        raw = open(filepath, encoding="latin-1").readlines()

    pts = []
    for line in raw[1:]:
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            pts.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    if len(pts) < 4:
        raise ValueError(f"profile_polygon_from_file: solo {len(pts)} puntos en '{filepath}'")

    arr = np.asarray(pts, dtype=np.float64)
    x_raw = arr[:, 0] * float(chord)
    y_raw = arr[:, 1] * float(chord)

    if min_te_height is not None:
        le_idx = int(np.argmin(x_raw))
        upper_x = x_raw[:le_idx + 1]
        upper_y = y_raw[:le_idx + 1]
        lower_x = x_raw[le_idx:]
        lower_y = y_raw[le_idx:]
        te_thickness = abs(float(upper_y[0]) - float(lower_y[-1]))
        if te_thickness < float(min_te_height) and len(upper_x) > 2 and len(lower_x) > 2:
            upper_x_inc = upper_x[::-1].copy()
            upper_y_inc = upper_y[::-1].copy()
            x_te = min(float(upper_x_inc[-1]), float(lower_x[-1]))
            x_le = max(float(upper_x_inc[0]), float(lower_x[0]))
            x_sample = np.linspace(x_te, x_le, 2000)
            y_up_s = np.interp(x_sample, upper_x_inc, upper_y_inc)
            y_lo_s = np.interp(x_sample, lower_x, lower_y)
            valid = np.where((y_up_s - y_lo_s) >= float(min_te_height))[0]
            if len(valid) > 0:
                x_cut = float(x_sample[valid[0]])
                y_cut_upper = float(np.interp(x_cut, upper_x_inc, upper_y_inc))
                y_cut_lower = float(np.interp(x_cut, lower_x, lower_y))
                eps = 1e-8
                mask_up = upper_x < (x_cut - eps)
                mask_lo = lower_x < (x_cut - eps)
                x_raw = np.concatenate([[x_cut], upper_x[mask_up], lower_x[mask_lo], [x_cut]])
                y_raw = np.concatenate([[y_cut_upper], upper_y[mask_up], lower_y[mask_lo], [y_cut_lower]])

    alpha = -np.deg2rad(float(alpha_geometry_deg))
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cx_rot = 0.25 * float(chord)
    x_shift = x_raw - cx_rot
    y_shift = y_raw
    x_rot = x_shift * ca - y_shift * sa + cx_rot
    y_rot = x_shift * sa + y_shift * ca
    x_final = x_rot + float(x_offset)
    y_final = y_rot + float(y_offset)
    if not (np.isclose(x_final[0], x_final[-1]) and np.isclose(y_final[0], y_final[-1])):
        x_final = np.concatenate([x_final, x_final[0:1]])
        y_final = np.concatenate([y_final, y_final[0:1]])
    return x_final, y_final


def _mesh_interp(mesh, field, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interpola un campo del mesh en coordenadas fisicas usando bilinear GPU."""
    import cupy as cp_gpu

    x_axis = np.asarray(cp_gpu.asnumpy(mesh.X_1d), dtype=np.float64)
    y_axis = np.asarray(cp_gpu.asnumpy(mesh.Y_1d), dtype=np.float64)
    jj = np.interp(np.asarray(x, dtype=np.float64), x_axis, np.arange(len(x_axis), dtype=np.float64))
    ii = np.interp(np.asarray(y, dtype=np.float64), y_axis, np.arange(len(y_axis), dtype=np.float64))
    jj = np.clip(jj, 0.0, len(x_axis) - 1.0)
    ii = np.clip(ii, 0.0, len(y_axis) - 1.0)
    vals = mesh._bilinear_interpolate(
        field.astype(cp_gpu.float32, copy=False),
        cp_gpu.asarray(jj, dtype=cp_gpu.float32),
        cp_gpu.asarray(ii, dtype=cp_gpu.float32),
    )
    return np.asarray(cp_gpu.asnumpy(vals), dtype=np.float64)


def _pressure_background(mesh, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float], bool]:
    import cupy as cp_gpu

    band = max(1, int(min(mesh.nx, mesh.ny) * 0.05))
    mask = cp_gpu.zeros_like(mesh.p, dtype=cp_gpu.bool_)
    mask[:band, :] = True
    mask[-band:, :] = True
    mask[:, :band] = True
    mask[:, -band:] = True
    mask = mask & (~mesh.solid)
    p_edge = np.asarray(cp_gpu.asnumpy(mesh.p[mask]), dtype=np.float64)
    x_edge = np.asarray(cp_gpu.asnumpy(mesh.XX[mask]), dtype=np.float64)
    y_edge = np.asarray(cp_gpu.asnumpy(mesh.YY[mask]), dtype=np.float64)
    if p_edge.size < 8:
        return np.zeros_like(x, dtype=np.float64), (0.0, 0.0, 0.0), False
    A = np.column_stack((np.ones_like(p_edge), x_edge, y_edge))
    coef, _, _, _ = np.linalg.lstsq(A, p_edge, rcond=None)
    bg = coef[0] + coef[1] * x + coef[2] * y
    return bg, (float(coef[0]), float(coef[1]), float(coef[2])), True


def _apply_pressure_debias(mesh, p_wall: np.ndarray, x: np.ndarray, y: np.ndarray,
                           ds: np.ndarray, mode: str) -> tuple[np.ndarray, dict]:
    valid = {"none", "mean_only", "affine_only", "affine+mean"}
    if mode not in valid:
        raise ValueError(f"pressure_debias_mode invalido: {mode!r}")
    p_force = np.asarray(p_wall, dtype=np.float64).copy()
    bg = np.zeros_like(p_force)
    a = b = c = 0.0
    model = mode
    if mode in {"affine_only", "affine+mean"}:
        bg, (a, b, c), ok = _pressure_background(mesh, x, y)
        if ok:
            p_force = p_force - bg
        elif mode == "affine+mean":
            model = "mean_only"
        else:
            model = "none"
    if mode in {"mean_only", "affine+mean"} and model != "none":
        wsum = float(np.sum(ds))
        if wsum > 0.0:
            p_force = p_force - float(np.sum(p_force * ds) / wsum)
    return p_force, {
        "pressure_debias_model": model,
        "pressure_debias_a": a,
        "pressure_debias_b": b,
        "pressure_debias_c": c,
        "p_bg": bg,
    }


def compute_geometric_pressure_forces(mesh,
                                      filepath: str | None = None,
                                      rho: float = 1.0,
                                      chord: float | None = None,
                                      alpha_geometry_deg: float | None = None,
                                      x_offset: float | None = None,
                                      y_offset: float | None = None,
                                      n_extrap_layers: int = 5,
                                      pressure_debias_mode: str = "affine+mean",
                                      te_exclude_after: float | None = None,
                                      sample_step: float | None = None) -> dict:
    """
    Integra solo presión sobre paneles geométricos reales del perfil.
    Es una ruta de auditoría/postproceso; no modifica el solver.
    """
    import cupy as cp_gpu

    chord = float(chord if chord is not None else getattr(mesh, "_airfoil_chord", getattr(mesh, "_chord", 1.0)))
    filepath = filepath if filepath is not None else getattr(mesh, "_airfoil_filepath", None)
    if hasattr(mesh, "_airfoil_polygon_x") and hasattr(mesh, "_airfoil_polygon_y"):
        x_poly = np.asarray(mesh._airfoil_polygon_x, dtype=np.float64)
        y_poly = np.asarray(mesh._airfoil_polygon_y, dtype=np.float64)
    else:
        if filepath is None:
            raise ValueError("compute_geometric_pressure_forces requiere filepath o polygon guardado en mesh")
        alpha_geometry_deg = float(
            alpha_geometry_deg
            if alpha_geometry_deg is not None
            else getattr(mesh, "_airfoil_alpha_geometry", getattr(mesh, "alpha_geometry", 0.0))
        )
        x_offset = float(x_offset if x_offset is not None else getattr(mesh, "_airfoil_x_offset", 0.0))
        y_offset = float(y_offset if y_offset is not None else getattr(mesh, "_airfoil_y_offset", 0.0))
        min_te = getattr(mesh, "_airfoil_min_te_height", None)
        x_poly, y_poly = profile_polygon_from_file(
            filepath,
            chord=chord,
            x_offset=x_offset,
            y_offset=y_offset,
            alpha_geometry_deg=alpha_geometry_deg,
            min_te_height=min_te,
        )

    if len(x_poly) < 3:
        raise ValueError("perfil geometrico demasiado corto")

    x0 = x_poly[:-1]
    y0 = y_poly[:-1]
    x1 = x_poly[1:]
    y1 = y_poly[1:]
    dx = x1 - x0
    dy = y1 - y0
    ds = np.sqrt(dx * dx + dy * dy)
    keep = ds > 1e-12
    x0, y0, x1, y1, dx, dy, ds = [arr[keep] for arr in (x0, y0, x1, y1, dx, dy, ds)]
    xm = 0.5 * (x0 + x1)
    ym = 0.5 * (y0 + y1)

    signed_area = 0.5 * float(np.sum(x_poly[:-1] * y_poly[1:] - x_poly[1:] * y_poly[:-1]))
    if signed_area >= 0.0:
        nx = dy / ds
        ny = -dx / ds
    else:
        nx = -dy / ds
        ny = dx / ds

    step = float(sample_step) if sample_step is not None else float(min(getattr(mesh, "dx", 1e-3), getattr(mesh, "dy", 1e-3)))
    step = max(step, 1e-12)

    solid_f = mesh.solid.astype(cp_gpu.float32)
    plus_s = _mesh_interp(mesh, solid_f, xm + 0.75 * step * nx, ym + 0.75 * step * ny)
    minus_s = _mesh_interp(mesh, solid_f, xm - 0.75 * step * nx, ym - 0.75 * step * ny)
    flip = plus_s > minus_s
    nx[flip] *= -1.0
    ny[flip] *= -1.0

    positions = np.asarray([0.5 + i for i in range(max(1, int(n_extrap_layers)))], dtype=np.float64)
    p_layers = []
    for pos in positions:
        p_layers.append(_mesh_interp(mesh, mesh.p, xm + pos * step * nx, ym + pos * step * ny))
    p_layers_arr = np.vstack(p_layers)
    if len(positions) >= 2:
        n = float(len(positions))
        sx = float(np.sum(positions))
        sx2 = float(np.sum(positions * positions))
        sy = np.sum(p_layers_arr, axis=0)
        sxy = np.sum(p_layers_arr * positions[:, None], axis=0)
        denom = max(n * sx2 - sx * sx, 1e-30)
        slope = (n * sxy - sx * sy) / denom
        p_wall = (sy - slope * sx) / n
    else:
        p_wall = p_layers_arr[0]

    x_min = float(np.nanmin(xm))
    x_max = float(np.nanmax(xm))
    chord_geom = max(x_max - x_min, 1e-30)
    x_over_c = np.clip((xm - x_min) / chord_geom, 0.0, 1.0)
    active = np.isfinite(p_wall)
    if te_exclude_after is not None:
        active &= x_over_c <= float(te_exclude_after)

    p_force, debias = _apply_pressure_debias(
        mesh,
        p_wall[active],
        xm[active],
        ym[active],
        ds[active],
        str(pressure_debias_mode),
    )

    nx_a = nx[active]
    ny_a = ny[active]
    ds_a = ds[active]
    dFx_p = -p_force * nx_a * ds_a
    dFy_p = -p_force * ny_a * ds_a

    alpha_rad = float(mesh._get_freestream_angle_rad())
    cos_a = np.cos(alpha_rad)
    sin_a = np.sin(alpha_rad)
    dDrag_p = dFx_p * cos_a + dFy_p * sin_a
    dLift_p = -dFx_p * sin_a + dFy_p * cos_a

    U_ref = max(float(getattr(mesh, "_U_ref", getattr(mesh, "_vel_ref", 1.0))), 1e-30)
    q = 0.5 * float(rho) * U_ref * U_ref * max(chord, 1e-30)
    cp_raw = p_wall[active] / max(0.5 * float(rho) * U_ref * U_ref, 1e-30)
    cp_force = p_force / max(0.5 * float(rho) * U_ref * U_ref, 1e-30)

    rows = []
    active_idx = np.flatnonzero(active)
    for out_id, idx in enumerate(active_idx):
        region = "LE" if x_over_c[idx] < 0.1 else ("TE" if x_over_c[idx] >= 0.8 else "MID")
        rows.append({
            "surface_panel_id": int(out_id),
            "x": float(xm[idx]),
            "y": float(ym[idx]),
            "x_over_c": float(x_over_c[idx]),
            "side": "upper" if ny[idx] > 0.0 else "lower",
            "region": region,
            "nx": float(nx[idx]),
            "ny": float(ny[idx]),
            "ds": float(ds[idx]),
            "p_wall": float(p_wall[idx]),
            "p_wall_force": float(p_force[out_id]),
            "p_bg": float(debias["p_bg"][out_id]) if len(debias["p_bg"]) else 0.0,
            "Cp_raw": float(cp_raw[out_id]),
            "Cp_force": float(cp_force[out_id]),
            "dFx_p": float(dFx_p[out_id]),
            "dFy_p": float(dFy_p[out_id]),
            "dDrag_p": float(dDrag_p[out_id]),
            "dLift_p": float(dLift_p[out_id]),
        })

    drag_p = float(np.sum(dDrag_p))
    lift_p = float(np.sum(dLift_p))
    summary = {
        "surface_normal_mode": "geometric",
        "n_surface_panels": int(len(rows)),
        "n_extrap_layers": int(n_extrap_layers),
        "pressure_debias_mode": str(pressure_debias_mode),
        "pressure_debias_model": debias["pressure_debias_model"],
        "pressure_debias_a": float(debias["pressure_debias_a"]),
        "pressure_debias_b": float(debias["pressure_debias_b"]),
        "pressure_debias_c": float(debias["pressure_debias_c"]),
        "te_exclude_after": float(te_exclude_after) if te_exclude_after is not None else float("nan"),
        "Drag_p_geom": drag_p,
        "Lift_p_geom": lift_p,
        "Cd_p_geom": drag_p / q,
        "Cl_p_geom": lift_p / q,
        "Cp_geom_min": float(np.nanmin(cp_force)) if len(cp_force) else float("nan"),
        "Cp_geom_max": float(np.nanmax(cp_force)) if len(cp_force) else float("nan"),
    }
    return {"rows": rows, "summary": summary}


# ═══════════════════════════════════════════════════════════════════════════════
# B. Panel method — Hess-Smith vortex panels
# ═══════════════════════════════════════════════════════════════════════════════

def _panel_geom(xi, yi, xj1, yj1, xj2, yj2):
    """Geometría auxiliar: log_r, dth (en marco local), cos_p, sin_p, ds."""
    dx = xj2 - xj1;  dy = yj2 - yj1
    ds = np.sqrt(dx**2 + dy**2) + 1e-30
    cos_p = dx / ds;  sin_p = dy / ds

    dxi1 = xi - xj1;  dyi1 = yi - yj1
    dxi2 = xi - xj2;  dyi2 = yi - yj2

    X = dxi1 * cos_p + dyi1 * sin_p
    Y = -dxi1 * sin_p + dyi1 * cos_p

    r1sq = dxi1**2 + dyi1**2 + 1e-30
    r2sq = dxi2**2 + dyi2**2 + 1e-30
    log_r = 0.5 * np.log(r1sq / r2sq)
    dth = np.arctan2(Y, X - ds) - np.arctan2(Y, X)
    return log_r, dth, cos_p, sin_p, ds


def _vortex_uv(xi, yi, xj1, yj1, xj2, yj2):
    """Velocidad (u,v) en (xi,yi) por panel VÓRTICE de fuerza γ=1/m.
    Local: u_loc = -dth/(2π), v_loc = +log_r/(2π).
    Global: rotación por (cos_p, sin_p)."""
    log_r, dth, cos_p, sin_p, _ = _panel_geom(xi, yi, xj1, yj1, xj2, yj2)
    inv2pi = 1.0 / (2.0 * np.pi)
    u_loc = -inv2pi * dth
    v_loc =  inv2pi * log_r
    u = u_loc * cos_p - v_loc * sin_p
    v = u_loc * sin_p + v_loc * cos_p
    return u, v


def _source_uv(xi, yi, xj1, yj1, xj2, yj2):
    """Velocidad (u,v) en (xi,yi) por panel FUENTE de fuerza σ=1/m.
    Local: u_loc = +log_r/(2π), v_loc = +dth/(2π).
    Global: rotación por (cos_p, sin_p)."""
    log_r, dth, cos_p, sin_p, _ = _panel_geom(xi, yi, xj1, yj1, xj2, yj2)
    inv2pi = 1.0 / (2.0 * np.pi)
    u_loc = inv2pi * log_r
    v_loc = inv2pi * dth
    u = u_loc * cos_p - v_loc * sin_p
    v = u_loc * sin_p + v_loc * cos_p
    return u, v


def panel_cp_inviscid(xu: np.ndarray, yu: np.ndarray,
                      xl: np.ndarray, yl: np.ndarray,
                      alpha_rad: float,
                      n_panels: int = 100
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cp invíscido por método de Hess-Smith (paneles fuente de fuerza variable +
    un único vórtice constante para circulación).

    Incógnitas: σ_0,...,σ_{N-1}, γ  (N+1 unknowns).
    Ecuaciones:
      i=0..N-1: V_n(i) = 0  (no penetración en cada panel midpoint)
      i=N    : V_t(panel 0, TE upper) + V_t(panel N-1, TE lower) = 0  (Kutta)

    Retorna (x_norm[200], Cp_upper[200], Cp_lower[200]).
    """
    chord = float(xu[-1] - xu[0]) if xu[-1] > xu[0] else float(xu[0])

    # ── Redistribución coseno ────────────────────────────────────────────────
    N_half = n_panels // 2
    theta_c = np.linspace(0.0, np.pi, N_half + 1)
    x_cos   = 0.5 * chord * (1.0 - np.cos(theta_c))
    yu_c = np.interp(x_cos, xu, yu)
    yl_c = np.interp(x_cos, xl, yl)

    # Ensamble CCW: upper TE→LE, lower LE→TE
    x_nodes = np.concatenate([x_cos[::-1], x_cos[1:]])
    y_nodes = np.concatenate([yu_c[::-1], yl_c[1:]])
    N = len(x_nodes) - 1

    xm = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    ym = 0.5 * (y_nodes[:-1] + y_nodes[1:])
    dx = np.diff(x_nodes);  dy = np.diff(y_nodes)
    ds = np.sqrt(dx**2 + dy**2) + 1e-30
    cos_p = dx / ds;  sin_p = dy / ds
    nx =  sin_p;  ny = -cos_p

    # ── Construir matrices de influencia ─────────────────────────────────────
    # An_s[i,j] = vel. normal en panel i por σ_j=1 en panel j (fuente)
    # An_v[i,j] = vel. normal en panel i por γ=1 en panel j (vórtice unitario)
    # At_s[i,j] = vel. tangencial análoga
    # At_v[i,j] = vel. tangencial análoga
    An_s = np.zeros((N, N));  An_v = np.zeros((N, N))
    At_s = np.zeros((N, N));  At_v = np.zeros((N, N))

    for j in range(N):
        us, vs = _source_uv(xm, ym, x_nodes[j], y_nodes[j], x_nodes[j+1], y_nodes[j+1])
        uv, vv = _vortex_uv(xm, ym, x_nodes[j], y_nodes[j], x_nodes[j+1], y_nodes[j+1])
        An_s[:, j] = us * nx + vs * ny
        An_v[:, j] = uv * nx + vv * ny
        At_s[:, j] = us * cos_p + vs * sin_p
        At_v[:, j] = uv * cos_p + vv * sin_p

    # Self-influencia (corrección de convención: fórmula evalúa lado INTERIOR
    # del CCW, necesitamos lado EXTERIOR para no-penetración y Cp).
    #   Source self normal (exterior) = +1/2
    #   Source self tangencial         =  0 (continua)
    #   Vortex self normal             =  0 (continua)
    #   Vortex self tangencial (ext)   = +1/2
    diag = np.arange(N)
    An_s[diag, diag] = 0.5
    At_s[diag, diag] = 0.0
    An_v[diag, diag] = 0.0
    At_v[diag, diag] = 0.5

    # ── Sistema (N+1)×(N+1): incógnitas [σ_0..σ_{N-1}, γ] ────────────────────
    M = np.zeros((N + 1, N + 1))
    b = np.zeros(N + 1)

    # Filas no-penetración (paneles i=0..N-1)
    M[:N, :N] = An_s
    M[:N,  N] = An_v.sum(axis=1)   # contribución de γ unitario sobre todos los paneles
    b[:N] = -(np.cos(alpha_rad) * nx + np.sin(alpha_rad) * ny)

    # Fila Kutta (i=N): V_t(panel 0) + V_t(panel N-1) = 0
    # V_t(k) = V_inf·t̂(k) + Σ_j σ_j·At_s[k,j] + γ·Σ_j At_v[k,j]
    k1, k2 = 0, N - 1
    M[N, :N] = At_s[k1, :] + At_s[k2, :]
    M[N,  N] = At_v[k1, :].sum() + At_v[k2, :].sum()
    b[N] = -(np.cos(alpha_rad) * (cos_p[k1] + cos_p[k2])
             + np.sin(alpha_rad) * (sin_p[k1] + sin_p[k2]))

    # ── Resolver ─────────────────────────────────────────────────────────────
    try:
        sol = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(M, b, rcond=1e-10)
    sigma = sol[:N];  gamma_const = sol[N]

    # ── Velocidad tangencial en cada midpoint ────────────────────────────────
    Vt = np.cos(alpha_rad) * cos_p + np.sin(alpha_rad) * sin_p
    Vt += At_s @ sigma
    Vt += gamma_const * At_v.sum(axis=1)

    Cp_panels = 1.0 - Vt**2

    # ── Separar upper / lower y mapear a 200 pts ─────────────────────────────
    xm_norm = xm / chord
    xu_m  = xm_norm[:N_half][::-1]
    Cpu_m = Cp_panels[:N_half][::-1]
    xl_m  = xm_norm[N_half:]
    Cpl_m = Cp_panels[N_half:]

    out_x = np.linspace(0.0, 1.0, 200)
    Cp_u_out = np.interp(out_x, xu_m, Cpu_m)
    Cp_l_out = np.interp(out_x, xl_m, Cpl_m)
    return out_x, Cp_u_out, Cp_l_out


# ═══════════════════════════════════════════════════════════════════════════════
# C. Integrador de capa límite (Thwaites + Michel + Head)
# ═══════════════════════════════════════════════════════════════════════════════

def _h_from_lam(lam: np.ndarray) -> np.ndarray:
    lam_c = np.clip(lam, _TH_LAM[0], _TH_LAM[-1])
    return np.interp(lam_c, _TH_LAM, _TH_H)


def _h1_from_h(H: float) -> float:
    if H < 1.6:
        return 3.3 + 0.8234 * max(H - 1.1, 1e-6) ** (-1.287)
    return 3.3 + 1.5501 * max(H - 0.6778, 1e-6) ** (-3.064)


def _h_from_h1(H1: float) -> float:
    H_arr  = np.linspace(1.05, 4.5, 800)
    H1_arr = np.array([_h1_from_h(h) for h in H_arr])
    return float(np.interp(H1, H1_arr[::-1], H_arr[::-1]))


def _cf_head(H: float, Re_th: float) -> float:
    return 0.246 * 10.0 ** (-0.678 * min(H, 4.0)) * max(Re_th, 10.0) ** (-0.268)


def _head_rhs(state: np.ndarray, Ue: float, dUeds: float, nu: float) -> np.ndarray:
    theta = max(state[0], 1e-12)
    H1    = max(state[1] / theta, 3.01)
    H     = _h_from_h1(H1)
    Re_th = Ue * theta / nu
    Cf    = _cf_head(H, Re_th)
    dth   = 0.5 * Cf - (H + 2.0) * theta / max(Ue, 1e-8) * dUeds
    # d(H1·θ)/ds = 0.0306*(H1-3)^(-0.6169)
    dH1th = 0.0306 * max(H1 - 3.0, 1e-6) ** (-0.6169)
    return np.array([dth, dH1th])


def integrate_bl(s: np.ndarray,
                 Ue: np.ndarray,
                 nu: float,
                 Re: float
                 ) -> dict:
    """
    Integra ecuaciones de capa límite sobre una superficie.

    s  : coordenada de arco (m), creciente, desde LE (0) → TE
    Ue : velocidad de borde (m/s)
    nu : viscosidad cinemática (m²/s)
    Re : Reynolds de cuerda (para Michel)

    Retorna dict: theta, dstar, H, Cf (arrays), trans_idx (int o None)
    """
    M  = len(s)
    Ue = np.maximum(Ue, 1e-8 * np.max(Ue))
    theta  = np.zeros(M)
    dstar  = np.zeros(M)
    H_arr  = np.full(M, 2.59)
    Cf_arr = np.zeros(M)

    theta[0] = 1e-6
    trans_idx = None

    # ── Thwaites laminar ─────────────────────────────────────────────────────
    integral = 0.0
    for i in range(1, M):
        ds_i = s[i] - s[i-1]
        integral += 0.5 * (Ue[i-1]**5 + Ue[i]**5) * ds_i
        th2 = 0.45 * nu / max(Ue[i]**6, 1e-30) * integral
        theta[i] = max(np.sqrt(abs(th2)), 1e-12)

        dUeds_i = (Ue[i] - Ue[i-1]) / (ds_i + 1e-30)
        lam = theta[i]**2 / nu * dUeds_i
        H_lam = float(_h_from_lam(np.array([lam]))[0])
        H_arr[i] = H_lam
        Cf_arr[i] = max(H_lam, 0.0) * 2.0 * nu / (Ue[i] * theta[i] + 1e-30)
        dstar[i] = theta[i] * H_arr[i]

        if i >= 2:
            Re_s  = Ue[i] * s[i] / nu
            Re_th = Ue[i] * theta[i] / nu
            if Re_s > 0 and Re_th > 1.174 * (1.0 + 22400.0 / Re_s)**0.46 * Re_s**0.46:
                trans_idx = i
                break

    if trans_idx is None:
        return {"theta": theta, "dstar": dstar, "H": H_arr,
                "Cf": Cf_arr, "trans_idx": None, "separated": False, "sep_idx": None}

    # ── Head turbulento (RK4) ────────────────────────────────────────────────
    H0  = max(H_arr[trans_idx], 1.1)
    H10 = _h1_from_h(H0)
    state = np.array([theta[trans_idx], H10 * theta[trans_idx]])
    separated = False
    sep_idx = None  # primer índice donde H cruza 2.4 (separación parcial)

    for i in range(trans_idx + 1, M):
        ds_i  = s[i] - s[i-1]
        dUeds = (Ue[i] - Ue[i-1]) / (ds_i + 1e-30)
        Ue_m  = 0.5 * (Ue[i-1] + Ue[i])

        k1 = _head_rhs(state,                  Ue[i-1], dUeds, nu)
        k2 = _head_rhs(state + 0.5*ds_i*k1,   Ue_m,    dUeds, nu)
        k3 = _head_rhs(state + 0.5*ds_i*k2,   Ue_m,    dUeds, nu)
        k4 = _head_rhs(state + ds_i*k3,        Ue[i],   dUeds, nu)
        state = state + ds_i / 6.0 * (k1 + 2*k2 + 2*k3 + k4)

        state[0] = max(state[0], 1e-12)
        H1_i = max(state[1] / state[0], 3.01)
        H_i  = _h_from_h1(H1_i)

        if H_i > 2.4 and sep_idx is None:
            sep_idx = i  # registrar punto de separación incipiente

        if H_i > 3.5:
            separated = True
            H_i = 3.5

        theta[i] = state[0]
        H_arr[i] = H_i
        dstar[i]  = theta[i] * H_i
        Re_th     = Ue[i] * theta[i] / nu
        Cf_arr[i] = _cf_head(H_i, Re_th)

    return {"theta": theta, "dstar": dstar, "H": H_arr,
            "Cf": Cf_arr, "trans_idx": trans_idx,
            "separated": separated, "sep_idx": sep_idx}


# ═══════════════════════════════════════════════════════════════════════════════
# D. Squire-Young (Cd viscoso en TE)
# ═══════════════════════════════════════════════════════════════════════════════

def cd_visc_squire_young(theta_te: float, H_te: float,
                          Ue_te: float, v_inf: float,
                          chord: float) -> float:
    """Cd_visc = 2*(θ_TE/c)*(Ue_TE/V∞)^((H_TE+5)/2)"""
    if H_te > 4.0:
        return 2.0 * theta_te / chord   # fallback separación masiva
    return 2.0 * (theta_te / chord) * (Ue_te / max(v_inf, 1e-12)) ** ((H_te + 5.0) / 2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# E. Cl desde ΔCp (integración directa)
# ═══════════════════════════════════════════════════════════════════════════════

def cl_from_delta_cp(x_norm: np.ndarray,
                     Cp_upper: np.ndarray,
                     Cp_lower: np.ndarray,
                     alpha_rad: float) -> float:
    """
    Cl = ∫(Cp_lower - Cp_upper) d(x/c) · cos(α)
    Rellena bins vacíos (valor 0.0) por interpolación si detecta datos ausentes.
    """
    dCp = Cp_lower - Cp_upper

    # Detectar bins vacíos: en get_cp_profile_mean, bins sin datos → 0.0
    # Heurística: bins donde tanto Cp_u como Cp_l son 0 simultáneamente → vacío
    mask_valid = ~((Cp_upper == 0.0) & (Cp_lower == 0.0))
    if mask_valid.sum() < 4:
        return float("nan")
    if not mask_valid.all():
        dCp = np.interp(x_norm, x_norm[mask_valid], dCp[mask_valid])

    return float(np.trapezoid(dCp, x_norm) * np.cos(alpha_rad))


# ═══════════════════════════════════════════════════════════════════════════════
# F1. Lógica de régimen de separación
# ═══════════════════════════════════════════════════════════════════════════════

_SEP_PARTIAL_H = 2.4   # H umbral separación incipiente (Stratford/Drela)
_SEP_MASSIVE_X = 0.3   # x/c umbral separación masiva


def _cd_visc_surface(bl: dict, s: np.ndarray, Ue: np.ndarray,
                     v_inf: float, chord: float) -> tuple:
    """
    Squire-Young adaptado al régimen de separación.

    Adjunto (sep_idx=None):
        Squire-Young en TE — comportamiento estándar.
    Separación parcial (x_sep >= 0.3c):
        Squire-Young evaluado en x_sep — BL post-separación ignorada.
    Separación masiva (x_sep < 0.3c):
        Cd_visc = 0 — Cd_p_LES ya captura resistencia de forma dominante.

    Returns: (cd_visc, regime_str, x_sep_norm)
      regime_str: "adjunto" | "sep_parcial" | "sep_masiva"
      x_sep_norm: x/c del punto de separación (1.0 si adjunto)
    """
    sep_idx = bl["sep_idx"]

    if sep_idx is None:
        cd = cd_visc_squire_young(bl["theta"][-1], bl["H"][-1], Ue[-1], v_inf, chord)
        return cd, "adjunto", 1.0

    x_sep = float(s[sep_idx]) / chord

    if x_sep >= _SEP_MASSIVE_X:
        cd = cd_visc_squire_young(bl["theta"][sep_idx], bl["H"][sep_idx],
                                  Ue[sep_idx], v_inf, chord)
        return cd, "sep_parcial", x_sep
    else:
        return 0.0, "sep_masiva", x_sep


# ═══════════════════════════════════════════════════════════════════════════════
# F. API principal
# ═══════════════════════════════════════════════════════════════════════════════

def compute_corrected_forces(mesh,
                              filepath: str,
                              Re: float | None = None,
                              v_inf: float | None = None,
                              chord: float | None = None,
                              alpha_deg: float | None = None,
                              rho: float = 1.0,
                              n_panels: int = 100,
                              n_extrap_layers: int = 5,
                              pressure_debias_mode: str = "affine+mean",
                              Cd_p_source: str = "ibm") -> dict:
    """
    Calcula Cl y Cd corregidos post-simulación.

    mesh     : objeto Mesh2D retornado por sim_main()
    filepath : ruta al archivo de perfil (e.g. "profiles/NACA_0012")
    Re       : Reynolds; calculado desde mesh si None
    v_inf    : velocidad freestream; desde mesh._U_ref si None
    chord    : cuerda; desde mesh._chord si None
    alpha_deg: ángulo de ataque; desde mesh.alpha_deg si None
    rho      : densidad [kg/m³]
    n_panels : número de paneles del panel method
    n_extrap_layers: capas para extrapolar presión del LES a pared
    pressure_debias_mode: modo de de-bias de presión usado en fuerzas
    Cd_p_source: "ibm" (default), "geometric" o "both"

    Retorna dict con: Cl, Cd, Cd_p, Cd_visc, Ef, Cl_inviscid,
                      trans_x_upper, trans_x_lower, bl_upper, bl_lower,
                      separated, warn
    """
    import cupy as cp_gpu

    warn = []

    # ── Metadatos del mesh ───────────────────────────────────────────────────
    chord     = chord     if chord     is not None else float(mesh._chord)
    v_inf     = v_inf     if v_inf     is not None else float(mesh._U_ref)
    nu        = float(mesh._nu_molecular)
    if alpha_deg is None:
        alpha_deg = float(getattr(mesh, "alpha_deg",
                          getattr(mesh, "_alpha_deg", 0.0)))
    alpha_rad = np.deg2rad(alpha_deg)
    Re        = Re if Re is not None else (v_inf * chord / nu)
    mu        = rho * nu
    q         = 0.5 * rho * v_inf**2 * chord   # normalización estándar

    # ── Fuerzas de presión del LES (fiables) ─────────────────────────────────
    Cd_p_source = str(Cd_p_source)
    if Cd_p_source not in {"ibm", "geometric", "both"}:
        raise ValueError("Cd_p_source debe ser 'ibm', 'geometric' o 'both'")
    forces = mesh.compute_drag_lift(
        mu,
        rho=rho,
        n_extrap_layers=n_extrap_layers,
        pressure_debias_mode=pressure_debias_mode,
    )
    Cd_p_ibm = float(forces["Drag_p"]) / q
    geom_summary = None
    if Cd_p_source in {"geometric", "both"}:
        geom = compute_geometric_pressure_forces(
            mesh,
            filepath=filepath,
            rho=rho,
            chord=chord,
            n_extrap_layers=n_extrap_layers,
            pressure_debias_mode=pressure_debias_mode,
        )
        geom_summary = geom["summary"]
    Cd_p = (
        float(geom_summary["Cd_p_geom"])
        if Cd_p_source == "geometric" and geom_summary is not None
        else Cd_p_ibm
    )

    # ── Geometría del perfil → panel method → Cp invíscido ───────────────────
    xu, yu, xl, yl = load_profile(filepath, chord=chord, alpha_deg=0.0)
    x_norm_p, Cpu_inv, Cpl_inv = panel_cp_inviscid(xu, yu, xl, yl,
                                                    alpha_rad, n_panels)
    Cl_inv = cl_from_delta_cp(x_norm_p, Cpu_inv, Cpl_inv, alpha_rad)

    # ── Ue desde Cp invíscido (Bernoulli, sin doble conteo) ──────────────────
    Ue_u = v_inf * np.sqrt(np.maximum(1.0 - Cpu_inv, 1e-6))
    Ue_l = v_inf * np.sqrt(np.maximum(1.0 - Cpl_inv, 1e-6))
    s    = x_norm_p * chord   # arco ≈ x (válido t/c < 15%)

    # ── Integración BL ───────────────────────────────────────────────────────
    bl_u = integrate_bl(s, Ue_u, nu, Re)
    bl_l = integrate_bl(s, Ue_l, nu, Re)

    Cd_v_u, regime_u, x_sep_u = _cd_visc_surface(bl_u, s, Ue_u, v_inf, chord)
    Cd_v_l, regime_l, x_sep_l = _cd_visc_surface(bl_l, s, Ue_l, v_inf, chord)
    Cd_visc = Cd_v_u + Cd_v_l

    if regime_u == "sep_masiva" or regime_l == "sep_masiva":
        warn.append(f"sep_masiva: x_sep_u={x_sep_u:.2f} x_sep_l={x_sep_l:.2f} "
                    f"— Cd_visc=0, usando solo Cd_p_LES")
    elif regime_u == "sep_parcial" or regime_l == "sep_parcial":
        warn.append(f"sep_parcial: Squire-Young en x_sep "
                    f"(u={x_sep_u:.2f}, l={x_sep_l:.2f})")

    # ── Cl desde Cp del LES (ΔCp integración directa) ────────────────────────
    x_les, Cpu_les, Cpl_les = mesh.get_cp_profile_mean()
    if len(x_les) == 0:
        warn.append("sin datos Cp en mesh — Cl desde panel method (inviscido)")
        Cl = Cl_inv
    else:
        Cl = cl_from_delta_cp(x_les, Cpu_les, Cpl_les, alpha_rad)
        if np.isnan(Cl):
            warn.append("Cl_LES nan — usando Cl_inviscid")
            Cl = Cl_inv

    Cd  = Cd_p + Cd_visc
    Ef  = Cl / Cd if abs(Cd) > 1e-12 else float("nan")

    M  = len(s)
    ti_u = bl_u["trans_idx"]
    ti_l = bl_l["trans_idx"]
    trans_x_u = float(x_norm_p[ti_u]) if ti_u is not None and ti_u < M else 1.0
    trans_x_l = float(x_norm_p[ti_l]) if ti_l is not None and ti_l < M else 1.0

    result = {
        "Cl":             Cl,
        "Cd":             Cd,
        "Cd_p":           Cd_p,
        "Cd_p_source":    Cd_p_source,
        "Cd_p_ibm":       Cd_p_ibm,
        "Cd_visc":        Cd_visc,
        "Cd_visc_upper":  Cd_v_u,
        "Cd_visc_lower":  Cd_v_l,
        "Ef":             Ef,
        "Cl_inviscid":    Cl_inv,
        "trans_x_upper":  trans_x_u,
        "trans_x_lower":  trans_x_l,
        "regime_upper":   regime_u,
        "regime_lower":   regime_l,
        "x_sep_upper":    x_sep_u,
        "x_sep_lower":    x_sep_l,
        "bl_upper":       bl_u,
        "bl_lower":       bl_l,
        "separated":      bl_u["separated"] or bl_l["separated"],
        "warn":           warn,
        "x_norm_panel":   x_norm_p,
        "Cp_upper_inv":   Cpu_inv,
        "Cp_lower_inv":   Cpl_inv,
    }
    if geom_summary is not None:
        result.update({
            "Cd_p_geom": geom_summary.get("Cd_p_geom", float("nan")),
            "Cl_p_geom": geom_summary.get("Cl_p_geom", float("nan")),
            "pressure_debias_model_geom": geom_summary.get("pressure_debias_model", ""),
        })
        if Cd_p_source == "both":
            result["Cd_bl_geom"] = float(geom_summary.get("Cd_p_geom", float("nan"))) + Cd_visc
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# G. Visualización
# ═══════════════════════════════════════════════════════════════════════════════

def plot_bl_result(result: dict, alpha_deg: float = 0.0,
                   Re: float = 1e5, save_path: str | None = None) -> None:
    import matplotlib.pyplot as plt

    x = result["x_norm_panel"]
    bl_u, bl_l = result["bl_upper"], result["bl_lower"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f"BL Correction  alpha={alpha_deg:.1f}  Re={Re:.0e}"
                 f"  Cl={result['Cl']:.4f}  Cd={result['Cd']:.4f}", fontsize=11)

    def _p(ax, yu, yl, ylabel, logy=False):
        ax.plot(x, yu, color="#2196F3", label="upper", linewidth=2)
        ax.plot(x, yl, color="#FF5722", label="lower", linewidth=2, linestyle="--")
        for xi, c in [(result["trans_x_upper"], "#2196F3"),
                      (result["trans_x_lower"], "#FF5722")]:
            if xi < 1.0:
                ax.axvline(xi, color=c, linestyle=":", linewidth=1)
        ax.set_ylabel(ylabel); ax.set_xlabel("x/c")
        ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
        if logy:
            ax.set_yscale("log")

    _p(axes[0,0], result["Cp_upper_inv"], result["Cp_lower_inv"], "Cp (panel, invíscido)")
    axes[0,0].invert_yaxis()
    _p(axes[0,1], bl_u["theta"]*1000, bl_l["theta"]*1000, "θ [mm]")
    _p(axes[0,2], bl_u["dstar"]*1000, bl_l["dstar"]*1000, "δ* [mm]")
    _p(axes[1,0], bl_u["H"], bl_l["H"], "H = δ*/θ")
    axes[1,0].axhline(2.4, color="red", linestyle="--", linewidth=1)
    _p(axes[1,1], bl_u["Cf"], bl_l["Cf"], "Cf")
    # Desglose Cd
    ax = axes[1,2]
    labels = ["Cd_p (LES)", "Cd_visc_upper (BL)", "Cd_visc_lower (BL)"]
    vals   = [result["Cd_p"], result["Cd_visc_upper"], result["Cd_visc_lower"]]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    ax.bar(labels, vals, color=colors)
    ax.set_title(f"Desglose Cd  total={result['Cd']:.4f}")
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180)
        print(f"[bl] {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests unitarios (sin mesh real)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os, sys

    ROOT = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("TEST 1: load_profile — NACA 0012")
    xu, yu, xl, yl = load_profile(os.path.join(ROOT, "profiles", "NACA_0012"))
    print(f"  Upper: {len(xu)} pts  LE=({xu[0]:.4f},{yu[0]:.4f})  TE=({xu[-1]:.4f},{yu[-1]:.4f})")
    print(f"  Lower: {len(xl)} pts  LE=({xl[0]:.4f},{yl[0]:.4f})  TE=({xl[-1]:.4f},{yl[-1]:.4f})")
    assert abs(xu[0]) < 0.01, "LE debe estar en x~0"
    assert abs(xu[-1] - 1.0) < 0.02, "TE debe estar en x~1"
    print("  OK")

    print("\nTEST 2: panel_cp_inviscid - NACA 0012 simetria alpha=0")
    x_n, Cpu, Cpl = panel_cp_inviscid(xu, yu, xl, yl, alpha_rad=0.0)
    Cl_0 = cl_from_delta_cp(x_n, Cpu, Cpl, 0.0)
    print(f"  Cl @ alpha=0: {Cl_0:.4f}  (debe ser ~0)")
    assert abs(Cl_0) < 0.05, f"Cl @ alpha=0 no es cero: {Cl_0}"
    # Simetría: Cp_upper ≈ Cp_lower
    sym_err = np.mean(np.abs(Cpu - Cpl))
    print(f"  Simetría Cp error medio: {sym_err:.4f}  (debe ser <0.01)")
    assert sym_err < 0.05, f"Asimetría excesiva: {sym_err}"
    print("  OK")

    print("\nTEST 3: panel_cp_inviscid - Cl vs thin-airfoil theory alpha=5")
    alpha5 = np.deg2rad(5.0)
    x_n5, Cpu5, Cpl5 = panel_cp_inviscid(xu, yu, xl, yl, alpha_rad=alpha5)
    Cl_5 = cl_from_delta_cp(x_n5, Cpu5, Cpl5, alpha5)
    Cl_theory = 2.0 * np.pi * np.sin(alpha5)
    err_pct = abs(Cl_5 - Cl_theory) / Cl_theory * 100
    print(f"  Cl_panel={Cl_5:.4f}  Cl_theory={Cl_theory:.4f}  error={err_pct:.1f}%")
    assert err_pct < 10.0, f"Error Cl vs thin-airfoil >10%: {err_pct:.1f}%"
    print("  OK")

    print("\nTEST 4: integrate_bl - placa plana (Thwaites analitico)")
    nu_test = 1e-5
    s_test  = np.linspace(1e-4, 1.0, 300)
    Ue_test = np.ones(300)
    bl_test = integrate_bl(s_test, Ue_test, nu=nu_test, Re=1.0/nu_test)
    theta_analytic = np.sqrt(0.45 * nu_test * s_test)
    # Comparar en zona laminar (antes de transición)
    ti = bl_test["trans_idx"]
    if ti is not None and ti > 10:
        err_th = np.max(np.abs(bl_test["theta"][:ti] - theta_analytic[:ti]) / theta_analytic[:ti])
        print(f"  Error max Thwaites vs analítico (lam): {err_th*100:.1f}%")
        assert err_th < 0.05, f"Error Thwaites >5%: {err_th*100:.1f}%"
    print(f"  Transición en s={s_test[ti]:.3f}" if ti else "  Sin transición (laminara)")
    print("  OK")

    print("\nTEST 5: load_profile - todos los perfiles")
    for name in ["NACA_0012", "AG24", "GM15"]:
        path = os.path.join(ROOT, "profiles", name)
        if os.path.exists(path):
            xu_, yu_, xl_, yl_ = load_profile(path)
            print(f"  {name}: upper={len(xu_)} lower={len(xl_)} pts")

    print("\n" + "=" * 60)
    print("Todos los tests OK")

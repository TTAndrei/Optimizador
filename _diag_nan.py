"""
Diagnostic: Find exactly WHERE and WHY NaN occurs with AG24 at alpha=2°.
Traces NaN through each step of the first few iterations.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import cupy as cp

# Import the simulator
from Simulador2D import Mesh

chord = 1.0
dx = 0.004
Lx = 7.0
Ly = 6.0
p0 = 0.0
v0x = 5.0
v0y = 0.0
alpha_deg = 5.0
rho = 1.225
nu = 1.5e-5

print("=" * 70)
print(f"DIAGNÓSTICO NaN: AG24, alpha={alpha_deg}, dx={dx}")
print("=" * 70)

# Create mesh
mesh = Mesh(Lx, Ly, p0, v0x, v0y, dx, dx, usar_wale=False)

# Load profile WITHOUT trim (to confirm NaN happens)
min_te = 0.0  # effectively disable trim
cx = 2.0
cy = Ly / 2.0

print(f"\n--- Cargando perfil AG24 ---")
info = mesh.load_solids_from_file(
    filepath="AG24",
    chord=chord,
    x_offset=cx, y_offset=cy,
    alpha_deg=alpha_deg, fill=True, plot=False,
    min_te_height=min_te
)
print(f"Info: {info}")

# Set boundaries  
mesh.set_boundary("left", "inflow", (v0x, v0y))
mesh.set_boundary("right", "outflow", p0)
mesh.set_boundary("top", "slip")
mesh.set_boundary("bottom", "slip")
mesh.apply_boundaries()

# Check ghost cell setup
print(f"\n--- Ghost Cell IBM ---")
print(f"Ghost cell ready: {mesh._ghost_cell_ready}")
if mesh._ghost_cell_ready:
    n_ghost = mesh._n_ghost
    print(f"Num ghost cells: {n_ghost}")
    
    # Check image points
    j_img = mesh._image_j_idx
    i_img = mesh._image_i_idx
    
    # Are any image points inside solid?
    j_img_int = cp.clip(j_img.astype(cp.int32), 0, mesh.nx - 1)
    i_img_int = cp.clip(i_img.astype(cp.int32), 0, mesh.ny - 1)
    img_in_solid = mesh.solid[i_img_int, j_img_int]
    n_img_in_solid = int(cp.sum(img_in_solid))
    print(f"Image points landing in solid: {n_img_in_solid} / {n_ghost} ({100*n_img_in_solid/max(n_ghost,1):.1f}%)")
    
    if n_img_in_solid > 0:
        print("  *** CRITICAL: Image points in solid = ghost cell interpolation from solid values! ***")
        
        # Check bilinear neighbors too
        j0 = j_img.astype(cp.int32)
        i0 = i_img.astype(cp.int32)
        j1 = cp.minimum(j0 + 1, mesh.nx - 1)
        i1 = cp.minimum(i0 + 1, mesh.ny - 1)
        s00 = mesh.solid[i0, j0]
        s10 = mesh.solid[i0, j1]
        s01 = mesh.solid[i1, j0]
        s11 = mesh.solid[i1, j1]
        any_solid = s00 | s10 | s01 | s11
        n_bilinear_in_solid = int(cp.sum(any_solid))
        print(f"  Image bilinear stencil touching solid: {n_bilinear_in_solid} / {n_ghost}")
        
        # Which ghost cells have image in solid?
        problematic = cp.where(img_in_solid)[0]
        if len(problematic) > 0:
            print(f"\n  First 20 problematic ghost cells (image in solid):")
            for k in range(min(20, len(problematic))):
                idx = int(problematic[k])
                gi = int(mesh._ghost_i[idx])
                gj = int(mesh._ghost_j[idx])
                ii = float(i_img[idx])
                ij = float(j_img[idx])
                x_ghost = (gj + 0.5) * dx + 0  # no offset in mesh
                y_ghost = (gi + 0.5) * dx + 0
                print(f"    ghost[{idx}] at (i={gi},j={gj}) -> image at ({ii:.2f},{ij:.2f}), "
                      f"img_solid={bool(img_in_solid[idx])}")

# Check: what does _vel_ref say?
print(f"\nvel_ref: {getattr(mesh, '_vel_ref', 'NOT SET')}")

# Initial field state
print(f"\n--- Initial field state ---")
print(f"u: min={float(cp.min(mesh.u)):.4f}, max={float(cp.max(mesh.u)):.4f}, has_nan={bool(cp.isnan(mesh.u).any())}")
print(f"v: min={float(cp.min(mesh.v)):.4f}, max={float(cp.max(mesh.v)):.4f}, has_nan={bool(cp.isnan(mesh.v).any())}")
print(f"p: min={float(cp.min(mesh.p)):.4f}, max={float(cp.max(mesh.p)):.4f}, has_nan={bool(cp.isnan(mesh.p).any())}")

# Run a few iterations and check after each step
CFL = 0.8
dt_base = CFL * dx / np.sqrt(v0x**2 + v0y**2)
print(f"\ndt_base = {dt_base:.6e}")

def check_field(mesh, label):
    u_nan = bool(cp.isnan(mesh.u).any())
    v_nan = bool(cp.isnan(mesh.v).any())
    p_nan = bool(cp.isnan(mesh.p).any())
    u_max = float(cp.max(cp.abs(mesh.u)))
    v_max = float(cp.max(cp.abs(mesh.v)))
    p_max = float(cp.max(cp.abs(mesh.p)))
    
    status = "OK" if not (u_nan or v_nan or p_nan) else "*** NaN ***"
    if u_max > 100 or v_max > 100:
        status = f"*** BLOWUP ***"
    
    print(f"  [{label}] |u|={u_max:.4f} |v|={v_max:.4f} |p|={p_max:.4f} {status}")
    
    if u_nan or v_nan:
        # Find WHERE the NaN is
        if u_nan:
            nan_locs = cp.where(cp.isnan(mesh.u))
            n_nan = len(nan_locs[0])
            print(f"    NaN in u: {n_nan} locations")
            if n_nan > 0 and n_nan <= 50:
                for k in range(min(10, n_nan)):
                    i = int(nan_locs[0][k])
                    j = int(nan_locs[1][k])
                    is_solid = bool(mesh.solid[i, j])
                    is_ghost = bool(mesh._ghost_mask[i, j]) if hasattr(mesh, '_ghost_mask') else '?'
                    print(f"      (i={i},j={j}) solid={is_solid} ghost={is_ghost}")
        if v_nan:
            nan_locs = cp.where(cp.isnan(mesh.v))
            print(f"    NaN in v: {len(nan_locs[0])} locations")
    
    return u_nan or v_nan or p_nan or u_max > 100

print(f"\n--- Running {200} iterations ---")
for it in range(200):
    dt_use = dt_base
    
    # Adaptive dt
    speed = cp.sqrt(mesh.u**2 + mesh.v**2)
    fluid = ~mesh.solid
    if cp.any(fluid):
        Umax = float(cp.max(speed[fluid]))
        Umax = max(Umax, v0x)
    else:
        Umax = v0x
    dt_adv = CFL * dx / max(Umax, 1e-12)
    dt_use = min(dt_adv, dt_base * 2.0)
    
    print(f"\n  Iter {it}: dt={dt_use:.4e}, Umax={Umax:.4f}")
    
    # Step 1: Advection
    mesh.advect_velocities(dt_use)
    mesh.apply_boundaries(after_projection=False)
    if check_field(mesh, "after advect"):
        print("  >>> NaN/blowup after ADVECTION")
        break
    
    # Step 2: Diffusion
    mesh.diffuse_velocity(nu, dt_use, usar_wale=False)
    mesh.apply_boundaries(after_projection=False)
    if check_field(mesh, "after diffuse"):
        print("  >>> NaN/blowup after DIFFUSION")
        break
    
    # Step 3: Projection
    try:
        mg_info = mesh.project_multigrid(rho, dt_use, tol_div=1e-1, verbose=False)
        mesh.apply_boundaries(after_projection=True)
        if check_field(mesh, "after project"):
            print("  >>> NaN/blowup after PROJECTION")
            break
    except Exception as e:
        print(f"  >>> EXCEPTION in projection: {e}")
        break

print("\n--- Done ---")

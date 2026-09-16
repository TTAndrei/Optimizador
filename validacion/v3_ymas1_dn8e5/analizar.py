r"""Figuras, perfiles de capa limite y video a partir de lo que volco `correr.py`.

Todo se dibuja fuera del lazo de simulacion: cada figura cuesta ~0.2 s desde los
campos guardados frente a ~20 s si se hace dentro, y ademas no bloquea la GPU.
"""

import json
import os
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AQUI = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(AQUI, "figuras")
VID = os.path.join(AQUI, "video")
ESTACIONES = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95)   # x/c de los perfiles de BL

# Polares XFOIL de AirfoilTools ya cacheadas en el repo. `Ncrit` fija lo agresiva
# que es la transicion: 5 es tunel ruidoso, 9 es el valor limpio de libro.
XFOIL = {5: "Sim_Cartesiano/data/xfoil/xf-n0012-il-100000-n5.csv",
         9: "Sim_Cartesiano/data/ComparativasReales/NACA0012_100k_Xfoil.csv"}
RAIZ = os.path.abspath(os.path.join(AQUI, "..", ".."))


def xfoil(alfa):
    """Cl, Cd, Cdp, Cm y punto de transicion de XFOIL al angulo pedido."""
    fuera = {}
    for n, f in XFOIL.items():
        d = np.genfromtxt(os.path.join(RAIZ, f), delimiter=",", skip_header=11,
                          names=["Alpha", "Cl", "Cd", "Cdp", "Cm", "Tx", "Bx"])
        d = d[np.isfinite(d["Cl"])]
        i = int(np.argmin(np.abs(d["Alpha"] - alfa)))
        fuera[n] = {k: float(d[k][i]) for k in d.dtype.names}
    return fuera


# ---------------------------------------------------------------------------
# Geometria de pared, reconstruida del mismo modo que `curvo.fuerzas`
# ---------------------------------------------------------------------------
def geometria(X, Y, perfil):
    i0, i1 = int(perfil[0]), int(perfil[1])
    c = slice(i0, i1 - 1)
    # Vector de area de la cara eta de j=0, igual que en `Metrica`.
    Sx = -(Y[0, i0 + 1:i1] - Y[0, i0:i1 - 1])
    Sy = X[0, i0 + 1:i1] - X[0, i0:i1 - 1]
    ds = np.hypot(Sx, Sy)
    x = 0.5 * (X[0, i0:i1 - 1] + X[0, i0 + 1:i1])
    y = 0.5 * (Y[0, i0:i1 - 1] + Y[0, i0 + 1:i1])
    xc = 0.5 * (X[:-1, c.start:c.stop] + X[1:, c.start:c.stop])
    return dict(c=c, ds=ds, nx=Sx / ds, ny=Sy / ds, x=x, y=y,
                cuerda=float(X[0, i0:i1].max() - X[0, i0:i1].min()),
                i_le=int(np.argmin(x)), i0=i0, i1=i1)


def tangente(g, alfa):
    """Tangente unitaria de cada cara de pared, orientada **aguas abajo**.

    Girar la normal exterior 90 grados da una tangente cuyo sentido se invierte
    entre extrados e intrados, y con ella el `Cf` de una cara sale con el signo
    cambiado respecto a la otra. Se fija proyectando sobre la corriente libre:
    asi `Cf > 0` es flujo adherido y `Cf < 0` es flujo invertido en los dos
    lados, que es el convenio con el que se lee una separacion. Cerca del punto
    de remanso la superficie es casi vertical y la proyeccion se anula, pero ahi
    `Cf` tiende a cero de todas formas.
    """
    tx, ty = -g["ny"], g["nx"]
    a = np.radians(alfa)
    signo = np.where(tx * np.cos(a) + ty * np.sin(a) < 0.0, -1.0, 1.0)
    return tx * signo, ty * signo


def centroides(X, Y):
    return (0.25 * (X[:-1, :-1] + X[:-1, 1:] + X[1:, :-1] + X[1:, 1:]),
            0.25 * (Y[:-1, :-1] + Y[:-1, 1:] + Y[1:, :-1] + Y[1:, 1:]))


def perfil_bl(X, Y, u, v, g, x_sobre_c, lado, nu, u_inf, alfa=0.0, n=40):
    """Perfil de capa limite en una estacion: distancia normal, u_t, y+, u+.

    Se recorre la linea eta de la columna cuyo centro de cara esta mas cerca de
    la estacion pedida. La distancia se mide **normal a la pared** proyectando el
    centroide sobre la normal de la cara, que es lo que compara con Blasius o con
    la ley de la pared; el arco a lo largo de eta seria otra cosa en cuanto la
    malla deja de ser ortogonal.
    """
    x0 = g["x"].min()
    xc = (g["x"] - x0) / g["cuerda"]
    arriba = g["y"] > 0.0 if lado == "extrados" else g["y"] <= 0.0
    cand = np.where(arriba)[0]
    i = cand[np.argmin(np.abs(xc[cand] - x_sobre_c))]
    col = g["i0"] + i

    cx, cy = centroides(X, Y)
    dx = cx[:n, col] - g["x"][i]
    dy = cy[:n, col] - g["y"][i]
    d = dx * g["nx"][i] + dy * g["ny"][i]            # normal exterior

    tx, ty = tangente(g, alfa)
    ut = u[:n, col] * tx[i] + v[:n, col] * ty[i]

    # u_tau del gradiente de pared, que es lo que el esquema ve: la cara esta a
    # media celda del centro y en la pared u_t = 0.
    tau = nu * ut[0] / d[0]
    u_tau = np.sqrt(abs(tau))
    return dict(x_c=float(xc[i]), d=d, ut=ut, u_tau=float(u_tau),
                y_mas=d * u_tau / nu, u_mas=ut / max(u_tau, 1e-30),
                cf=2.0 * tau / u_inf ** 2)


def cf_cp_pared(X, Y, u, v, p, g, nu, u_inf, alfa=0.0):
    """Cp y Cf sobre la pared, con el mismo gradiente de primer orden del esquema."""
    cx, cy = centroides(X, Y)
    i = np.arange(g["i0"], g["i1"] - 1)
    dx, dy = cx[0, i] - g["x"], cy[0, i] - g["y"]
    d = dx * g["nx"] + dy * g["ny"]
    tx, ty = tangente(g, alfa)
    ut = u[0, i] * tx + v[0, i] * ty
    return (p[0, i] / (0.5 * u_inf ** 2), 2.0 * nu * ut / (d * u_inf ** 2), d)


# ---------------------------------------------------------------------------
def _escala(a, datos, cols, nombres, t, desde=2.0):
    """Encuadra el panel con lo que pasa **despues** del transitorio de arranque.

    El pico inicial del Cd es diez veces el valor convergido y con el dentro no
    se lee nada; la curva se sigue dibujando entera.
    """
    v = np.concatenate([datos[t >= desde, cols.index(n)] for n in nombres])
    m, M = v.min(), v.max()
    h = max(M - m, 1e-6) * 0.15
    a.set_ylim(m - h, M + h)


def historia(datos, cols, caso):
    t = datos[:, cols.index("t")]
    xf = xfoil(caso["alfa"])
    fig, ax = plt.subplots(3, 2, figsize=(13, 11), sharex=True)

    a = ax[0, 0]
    for n, e in (("Cl_sup", "superficie"), ("Cl_dcp", "$\\oint\\Delta C_p$"),
                 ("Cl_circ", "circulacion (Kutta-Joukowski)")):
        a.plot(t, datos[:, cols.index(n)], label=e, lw=1.2)
    for n, e in xf.items():
        a.axhline(e["Cl"], ls="--", lw=.8, color="k" if n == 9 else "gray",
                  label="XFOIL Ncrit=%d (%.3f)" % (n, e["Cl"]))
    a.set_ylabel("$C_l$"); a.legend(fontsize=8); a.grid(alpha=.3)
    a.set_title("Sustentacion: tres estimadores")

    a = ax[0, 1]
    for n, e in (("Cd", "total"), ("Cd_p", "presion"), ("Cd_v", "viscoso")):
        a.plot(t, datos[:, cols.index(n)], label=e, lw=1.2)
    for n, e in xf.items():
        a.axhline(e["Cd"], ls="--", lw=.8, color="k" if n == 9 else "gray",
                  label="XFOIL Ncrit=%d (%.4f)" % (n, e["Cd"]))
    a.set_ylabel("$C_d$"); a.legend(fontsize=8); a.grid(alpha=.3)
    _escala(a, datos, cols, ("Cd", "Cd_p", "Cd_v"), t)
    a.set_title("Resistencia (encuadrado tras el transitorio)")

    a = ax[1, 0]
    a.plot(t, datos[:, cols.index("Cm")], lw=1.2, color="C3")
    for n, e in xf.items():
        a.axhline(e["Cm"], ls="--", lw=.8, color="k" if n == 9 else "gray",
                  label="XFOIL Ncrit=%d" % n)
    a.legend(fontsize=7)
    a.set_ylabel("$C_m$ (c/4)"); a.grid(alpha=.3); a.set_title("Momento")
    _escala(a, datos, cols, ("Cm",), t)

    a = ax[1, 1]
    a.plot(t, datos[:, cols.index("dCp_TE")], lw=1.2, color="C4")
    a.axhline(-0.092, ls="--", c="k", lw=.8,
              label="cartesiano CON parche de Kutta")
    a.set_ylabel("$\\Delta C_p$ en el TE"); a.legend(fontsize=8); a.grid(alpha=.3)
    a.set_ylim(-0.10, 0.005)
    a.set_title("Condicion de Kutta, sin imponerla")

    a = ax[2, 0]
    for n in [c for c in cols if c.startswith("gamma_")]:
        a.plot(t, datos[:, cols.index(n)], lw=1.0, label=n.replace("gamma_", "r="))
    a.set_ylabel("$\\Gamma$"); a.set_xlabel("$t^*$"); a.legend(fontsize=7, ncol=2)
    a.grid(alpha=.3); a.set_title("Circulacion por tamano de lazo")

    a = ax[2, 1]
    a.plot(t, datos[:, cols.index("nut_nu")], lw=1.2, color="C5", label="$\\nu_t/\\nu$ max")
    a.set_ylabel("$\\nu_t/\\nu$"); a.set_xlabel("$t^*$"); a.grid(alpha=.3)
    a.legend(loc="center right", fontsize=8)
    b = a.twinx()
    b.semilogy(t, datos[:, cols.index("div")], lw=1.0, color="C7", ls=":")
    b.set_ylabel("divergencia relativa", color="C7")
    a.set_title("Turbulencia y divergencia")

    fig.suptitle("NACA 0012, $\\alpha=%g^\\circ$, Re=%.0e, Spalart-Allmaras, "
                 "%d celdas, $y^+\\!\\approx\\!1$"
                 % (caso["alfa"], caso["re"], caso["celdas"]), y=.995)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "historia_coeficientes.png"), dpi=140)
    plt.close(fig)

    fig, a = plt.subplots(figsize=(9, 3.4))
    a.plot(t[1:], datos[1:, cols.index("it_s")], lw=1.0)
    a.axhline(np.median(datos[1:, cols.index("it_s")]), ls="--", c="k", lw=.8,
              label="mediana %.2f it/s" % np.median(datos[1:, cols.index("it_s")]))
    a.set_xlabel("$t^*$"); a.set_ylabel("it/s"); a.grid(alpha=.3); a.legend()
    a.set_title("Ritmo de calculo (RTX 3070 Ti, float32)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ritmo.png"), dpi=140)
    plt.close(fig)


def cl_cd(datos, cols, caso, desde=1.0):
    """Solo Cl y Cd, y el Cd **recortado por delante**.

    El pico de arranque del Cd es veinte veces el valor convergido: con el dentro
    el eje se estira y no se ve ni la bajada ni donde se estabiliza. El recorte es
    solo de dibujo, la serie entera esta en `historia.npz`.
    """
    t = datos[:, cols.index("t")]
    xf = xfoil(caso["alfa"])
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))

    a = ax[0]
    a.plot(t, datos[:, cols.index("Cl_sup")], lw=1.4, color="C0")
    a.set_xlabel("$t^*$"); a.set_ylabel("$C_l$"); a.grid(alpha=.3)
    a.set_title("Sustentacion")

    k = t >= desde
    a = ax[1]
    a.plot(t[k], datos[k, cols.index("Cd")], lw=1.4, color="C1", label="total")
    a.plot(t[k], datos[k, cols.index("Cd_p")], lw=1.0, color="C2", label="presion")
    a.plot(t[k], datos[k, cols.index("Cd_v")], lw=1.0, color="C4", label="viscoso")
    a.set_xlabel("$t^*$"); a.set_ylabel("$C_d$"); a.grid(alpha=.3)
    a.set_title("Resistencia (desde $t^*=%g$, sin el pico de arranque)" % desde)

    for n, e in xf.items():
        col = "k" if n == 9 else "gray"
        ax[0].axhline(e["Cl"], ls="--", lw=.9, color=col,
                      label="XFOIL Ncrit=%d: %.4f" % (n, e["Cl"]))
        ax[1].axhline(e["Cd"], ls="--", lw=.9, color=col,
                      label="XFOIL Ncrit=%d: %.5f" % (n, e["Cd"]))
    for a, v in ((ax[0], datos[-1, cols.index("Cl_sup")]),
                 (ax[1], datos[-1, cols.index("Cd")])):
        a.annotate("%.4f" % v, (t[-1], v), textcoords="offset points",
                   xytext=(-6, 6), ha="right", fontsize=9)
        a.legend(fontsize=8)

    fig.suptitle("NACA 0012, $\\alpha=%g^\\circ$, Re=%.0e, Spalart-Allmaras, "
                 "%d celdas, $y^+\\!\\approx\\!1$"
                 % (caso["alfa"], caso["re"], caso["celdas"]))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "cl_cd.png"), dpi=150)
    plt.close(fig)


def malla_dominio(X, Y, g, caso, ruta):
    """La malla entera, con las cotas del dominio a la vista."""
    fig, a = plt.subplots(figsize=(10, 6))
    a.plot(X[-1], Y[-1], "k-", lw=1.2)
    a.plot(X[:, 0], Y[:, 0], "k-", lw=.8)
    a.plot(X[:, -1], Y[:, -1], "k-", lw=.8)
    a.plot(X[::6].T, Y[::6].T, color="C0", lw=.25)
    a.plot(X[:, ::12], Y[:, ::12], color="C0", lw=.25)
    a.set_aspect("equal")
    lx, ly = X.max() - X.min(), Y.max() - Y.min()
    a.set_title("Malla %dx%d  —  $L_x$=%.2f c, $L_y$=%.2f c, "
                "$x\\in[%.2f, %.2f]$, $|y|\\leq%.2f$"
                % (X.shape[0], X.shape[1], lx, ly, X.min(), X.max(), Y.max()))
    a.set_xlabel("x/c"); a.set_ylabel("y/c")
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def campo(X, Y, d, g, caso, ruta, vista, vmax):
    """Un marco del modulo de la velocidad. `vista` = (x0, x1, y0, y1) o None."""
    mod = np.hypot(d["u"], d["v"])
    fig, a = plt.subplots(figsize=(9, 5.4) if vista else (10, 5.6))
    m = a.pcolormesh(X, Y, mod, cmap="turbo", vmin=0.0, vmax=vmax,
                     shading="auto", rasterized=True)
    i0, i1 = g["i0"], g["i1"]
    a.fill(X[0, i0:i1], Y[0, i0:i1], color="w", ec="k", lw=.7, zorder=5)
    if vista:
        a.set_xlim(vista[0], vista[1]); a.set_ylim(vista[2], vista[3])
    a.set_aspect("equal")
    fig.colorbar(m, ax=a, label="$|u|/U_\\infty$", shrink=.85)
    a.set_title("$t^*=%5.2f$   $\\alpha=%g^\\circ$  Re=%.0e"
                % (float(d["t"]), caso["alfa"], caso["re"]))
    a.set_xlabel("x/c"); a.set_ylabel("y/c")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def marco_bl(X, Y, d, g, caso, ruta, nu):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    u, v = d["u"], d["v"]
    for x_c in ESTACIONES:
        b = perfil_bl(X, Y, u, v, g, x_c, "extrados", nu, caso["u_inf"],
                      caso["alfa"])
        ax[0].plot(b["ut"] / caso["u_inf"], b["d"], marker=".", ms=3, lw=1,
                   label="x/c=%.2f" % b["x_c"])
        ok = b["y_mas"] > 0
        ax[1].semilogx(b["y_mas"][ok], b["u_mas"][ok], marker=".", ms=3, lw=1)
    ax[0].set_ylim(0, 0.08); ax[0].set_xlabel("$u_t/U_\\infty$")
    ax[0].set_ylabel("distancia normal a la pared"); ax[0].grid(alpha=.3)
    ax[0].legend(fontsize=7); ax[0].set_title("Perfiles de capa limite (extrados)")

    yp = np.logspace(0, 3, 50)
    ax[1].semilogx(yp, yp, "k--", lw=.8, label="$u^+=y^+$")
    ax[1].semilogx(yp, np.log(yp) / 0.41 + 5.0, "k:", lw=.8,
                   label="$\\frac{1}{\\kappa}\\ln y^+ + 5$")
    ax[1].set_xlim(0.3, 1e3); ax[1].set_ylim(0, 25)
    ax[1].set_xlabel("$y^+$"); ax[1].set_ylabel("$u^+$"); ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=7); ax[1].set_title("Ley de la pared")

    cp, cf, d1 = cf_cp_pared(X, Y, u, v, d["p"], g, nu, caso["u_inf"],
                             caso["alfa"])
    xc = (g["x"] - g["x"].min()) / g["cuerda"]
    ar = g["y"] > 0
    ax[2].plot(xc[ar], cf[ar], lw=1, label="extrados")
    ax[2].plot(xc[~ar], cf[~ar], lw=1, label="intrados")
    ax[2].axhline(0, c="k", lw=.6)
    ax[2].set_xlabel("x/c"); ax[2].set_ylabel("$C_f$"); ax[2].grid(alpha=.3)
    ax[2].legend(fontsize=7); ax[2].set_ylim(-0.005, 0.025)
    ax[2].set_title("Friccion de pared")
    fig.suptitle("$t^*=%5.2f$" % float(d["t"]))
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def video(patron, salida, fps=20):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", patron, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", salida], check=True)


# ---------------------------------------------------------------------------
def main():
    caso = json.load(open(os.path.join(AQUI, "caso.json")))
    nu = caso["u_inf"] / caso["re"]
    m = np.load(os.path.join(AQUI, "malla.npz"))
    X, Y = m["X"], m["Y"]
    g = geometria(X, Y, m["perfil"])

    h = np.load(os.path.join(AQUI, "historia.npz"))
    cols = list(h["columnas"])
    historia(h["datos"], cols, caso)
    cl_cd(h["datos"], cols, caso)

    marcos = sorted(os.listdir(os.path.join(AQUI, "campos")))
    for sub in ("dominio", "cerca", "bl"):
        os.makedirs(os.path.join(VID, sub), exist_ok=True)
    for k, n in enumerate(marcos):
        d = np.load(os.path.join(AQUI, "campos", n))
        campo(X, Y, d, g, caso, os.path.join(VID, "dominio", "%04d.png" % k),
              (-2.5, 5.0, -2.5, 2.5), 1.6)
        campo(X, Y, d, g, caso, os.path.join(VID, "cerca", "%04d.png" % k),
              (-0.25, 1.35, -0.45, 0.45), 1.6)
        marco_bl(X, Y, d, g, caso, os.path.join(VID, "bl", "%04d.png" % k), nu)
    for sub, nombre in (("dominio", "velocidad_dominio.mp4"),
                        ("cerca", "velocidad_perfil.mp4"),
                        ("bl", "capa_limite.mp4")):
        video(os.path.join(VID, sub, "%04d.png"), os.path.join(VID, nombre))

    # Estado final: figuras fijas de mas resolucion.
    d = np.load(os.path.join(AQUI, "campos", marcos[-1]))
    campo(X, Y, d, g, caso, os.path.join(FIG, "velocidad_dominio_final.png"),
          None, 1.6)
    campo(X, Y, d, g, caso, os.path.join(FIG, "velocidad_estela_final.png"),
          (-2.5, 5.0, -2.5, 2.5), 1.6)
    malla_dominio(X, Y, g, caso, os.path.join(FIG, "malla_dominio.png"))
    campo(X, Y, d, g, caso, os.path.join(FIG, "velocidad_perfil_final.png"),
          (-0.25, 1.35, -0.45, 0.45), 1.6)
    marco_bl(X, Y, d, g, caso, os.path.join(FIG, "capa_limite_final.png"), nu)

    cp, cf, d1 = cf_cp_pared(X, Y, d["u"], d["v"], d["p"], g, nu,
                             caso["u_inf"], caso["alfa"])
    b = [perfil_bl(X, Y, d["u"], d["v"], g, x, "extrados", nu, caso["u_inf"],
                   caso["alfa"]) for x in ESTACIONES]
    y1 = np.array([p["y_mas"][0] for p in b])
    np.savez_compressed(os.path.join(AQUI, "pared_final.npz"), x_c=(
        g["x"] - g["x"].min()) / g["cuerda"], y=g["y"], cp=cp, cf=cf,
        d_primera=d1, estaciones=np.array(ESTACIONES), y_mas_primera=y1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    xc = (g["x"] - g["x"].min()) / g["cuerda"]
    ar = g["y"] > 0
    ax[0].plot(xc[ar], cp[ar], lw=1.2, label="extrados")
    ax[0].plot(xc[~ar], cp[~ar], lw=1.2, label="intrados")
    ax[0].invert_yaxis(); ax[0].set_xlabel("x/c"); ax[0].set_ylabel("$C_p$")
    ax[0].grid(alpha=.3); ax[0].legend(); ax[0].set_title("Presion en la pared")
    ax[1].plot(xc, d1 * np.sqrt(np.abs(cf) / 2.0) * caso["u_inf"] / nu, ".", ms=3)
    ax[1].axhline(1.0, ls="--", c="k", lw=.8, label="$y^+=1$")
    ax[1].set_yscale("log"); ax[1].set_xlabel("x/c")
    ax[1].set_ylabel("$y^+$ del primer centro"); ax[1].grid(alpha=.3)
    ax[1].legend(); ax[1].set_title("Resolucion de pared alcanzada")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "pared_cp_ymas.png"), dpi=140)
    plt.close(fig)
    u = h["datos"][-1]
    xf = xfoil(caso["alfa"])
    yp = d1 * np.sqrt(np.abs(cf) / 2.0) / nu
    comp = {"nuestro": {n: float(u[cols.index(n)]) for n in
                        ("Cl_sup", "Cl_dcp", "Cl_circ", "Cd", "Cd_p", "Cd_v",
                         "Cm", "dCp_TE", "disp_gamma", "nut_nu", "div", "it_s")},
            "xfoil": {str(n): v for n, v in xf.items()},
            "y_mas_primer_centro": {"mediana": float(np.median(yp)),
                                    "p95": float(np.percentile(yp, 95)),
                                    "max": float(yp.max())},
            "placa_plana": {"laminar_2_caras": 2 * 1.328 / np.sqrt(caso["re"]),
                            "turbulenta_2_caras": 2 * 0.074 / caso["re"] ** 0.2},
            "perfil_delgado_Cl": float(2 * np.pi * np.sin(np.radians(caso["alfa"])))}
    for n in (5, 9):
        comp["xfoil"][str(n)]["error_Cl_%"] = 100 * (
            comp["nuestro"]["Cl_sup"] / xf[n]["Cl"] - 1)
        comp["xfoil"][str(n)]["error_Cd_%"] = 100 * (
            comp["nuestro"]["Cd"] / xf[n]["Cd"] - 1)
        comp["xfoil"][str(n)]["error_Cdp_%"] = 100 * (
            comp["nuestro"]["Cd_p"] / xf[n]["Cdp"] - 1)
    json.dump(comp, open(os.path.join(AQUI, "resultados.json"), "w"), indent=1)

    print("y+ del primer centro: mediana %.2f, p95 %.2f, max %.2f"
          % (np.median(yp),
             np.percentile(yp, 95), yp.max()))
    print(json.dumps(comp["xfoil"], indent=1))


if __name__ == "__main__":
    main()

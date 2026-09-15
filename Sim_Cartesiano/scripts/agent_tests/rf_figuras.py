"""
Figuras del estudio de resultados finales.

Separado del runner a proposito: las 18 h de GPU del estudio no se pueden
repetir para retocar una leyenda. Todo lo que hay aqui se regenera desde los
.npz ya volcados, asi que `resultados_finales.py --solo-figuras` rehace las 300
imagenes en minutos sin tocar el solver.

Por punto (perfil x malla x angulo):
    velocidad_zoom / velocidad_dominio     modulo de la velocidad
    streamlines_zoom / streamlines_dominio lineas de corriente coloreadas
    presion_zoom                           coeficiente de presion
    vorticidad_zoom                        vorticidad, que es donde se ve la
                                           burbuja de separacion y la estela
    vectores_zoom                          campo de vectores sobre el modulo
    fuerzas / eficiencia / residuos        historia temporal

La zona fina y el dominio completo van los dos porque responden a preguntas
distintas: el zoom enseña la capa limite y la burbuja, y el dominio entero es lo
unico que enseña si la estela sale por el contorno sin reflejarse, que es la
justificacion del dominio 24x16.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator
from scipy.interpolate import RegularGridInterpolator

import resultados_finales as RF

DPI = 160
VMAX = 1.8
CX, CY = RF.DOMINIO["cx"], RF.DOMINIO["Ly"] / 2.0

ZOOM = (CX - 0.45, CX + 1.85, CY - 0.70, CY + 0.70)
TODO = (0.0, RF.DOMINIO["Lx"], 0.0, RF.DOMINIO["Ly"])


def _remuestrear(npz, ventana, n=900):
    """La malla es cartesiana estirada; streamplot y np.gradient exigen paso
    constante, asi que se interpola a una rejilla uniforme de la ventana pedida."""
    x, y = np.asarray(npz["x"], float), np.asarray(npz["y"], float)
    x0, x1, y0, y1 = ventana
    xi = np.linspace(x0, x1, n)
    yi = np.linspace(y0, y1, max(40, int(n * (y1 - y0) / (x1 - x0))))
    XI, YI = np.meshgrid(xi, yi)
    pts = np.column_stack([YI.ravel(), XI.ravel()])
    out = []
    for k in ("u", "v", "p", "solid"):
        f = np.asarray(npz[k], float)
        g = RegularGridInterpolator((y, x), f, bounds_error=False, fill_value=None)
        out.append(g(pts).reshape(XI.shape))
    return (xi, yi) + tuple(out)


def _cuerpo(ax, xi, yi, mask):
    ax.contourf(xi, yi, mask, levels=[0.5, 2.0], colors=["#111111"], zorder=5)
    ax.contour(xi, yi, mask, levels=[0.5], colors=["#111111"], linewidths=1.0, zorder=6)


def _ejes(ax, titulo, ventana, paso=None, relativo=False):
    x0, x1, y0, y1 = ventana
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    if relativo:
        # El perfil vive en x=cx, y=Ly/2 dentro del dominio. En el zoom interesa
        # leer distancias desde el borde de ataque, no la coordenada absoluta,
        # asi que se reetiquetan los ticks sin tocar los datos.
        ax.xaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t - CX:.2f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t - CY:.2f}"))
        ax.set_xlabel("x/c   (0 = borde de ataque)")
        ax.set_ylabel("y/c   (0 = línea de cuerda)")
    else:
        ax.set_xlabel("x [c]")
        ax.set_ylabel("y [c]")
    ax.set_title(titulo, fontsize=11)
    if paso:
        ax.xaxis.set_major_locator(MultipleLocator(paso))
        ax.yaxis.set_major_locator(MultipleLocator(paso))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.tick_params(which="both", direction="out", top=False, right=False)


def _t_final(p, dx, a):
    """Tiempo fisico realmente alcanzado. El nominal (t~20) sale de un dt medio
    estimado; el dt es adaptativo por CFL, asi que el real difiere y es el que
    debe aparecer en la figura."""
    f = RF.serie_path(p, dx, a)
    if not os.path.exists(f):
        return None
    t = np.load(f)["t"]
    return float(t[-1]) if len(t) else None


def _cab(p, dx, a, meta):
    tf = _t_final(p, dx, a)
    return (f"{RF.PERFILES[p]['nombre']} — α={a:.0f}°, Re=1e5, dx={dx:g}, "
            f"dominio {RF.DOMINIO['Lx']:g}×{RF.DOMINIO['Ly']:g}\n"
            f"campo final: {meta.get('iters_efectivas')} iters"
            + (f", t={tf:.1f}" if tf else "") +
            f"   Cl={meta.get('cl'):.4f}  Cd={meta.get('cd'):.5f}  L/D={meta.get('ld'):.2f}")


def _fig_dir(p):
    return os.path.join(RF.d_perfil(p), "figuras")


def _guarda(fig, p, sub, nombre):
    f = os.path.join(_fig_dir(p), sub, nombre)
    fig.savefig(f, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Campos
# ----------------------------------------------------------------------
def figuras_campo(p, dx, a, meta):
    f = RF.campo_path(p, dx, a)
    if not os.path.exists(f):
        return
    npz = np.load(f)
    base = f"dx{RF.k_dx(dx)}_a{a:04.1f}"
    cab = _cab(p, dx, a, meta)

    for etiq, ventana, paso in (("zoom", ZOOM, 0.5), ("dominio", TODO, 2.0)):
        xi, yi, u, v, pr, mask = _remuestrear(npz, ventana)
        dentro = mask > 0.5
        speed = np.where(dentro, np.nan, np.hypot(u, v))

        # En el dominio completo casi todo el campo vale ~1: con la escala fija
        # 0-1.8 del zoom, la estela y la deflexion quedan invisibles. Se escala
        # por percentiles para que se vea lo unico que esa vista tiene que
        # demostrar: que la estela sale por el contorno sin reflejarse.
        if etiq == "zoom":
            vlo, vhi = 0.0, VMAX
        else:
            vlo = float(np.nanpercentile(speed, 0.2))
            vhi = float(np.nanpercentile(speed, 99.8))
        fig, ax = plt.subplots(figsize=(10.5, 6.8))
        im = ax.pcolormesh(xi, yi, speed, cmap="turbo", shading="auto",
                           vmin=vlo, vmax=vhi)
        _cuerpo(ax, xi, yi, mask)
        if etiq != "zoom":
            ax.add_patch(Rectangle((ZOOM[0], ZOOM[2]), ZOOM[1] - ZOOM[0],
                                   ZOOM[3] - ZOOM[2], fill=False, ec="w",
                                   lw=1.2, ls="--", zorder=7))
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label(
            r"$|\mathbf{u}|/U_\infty$ [-]")
        _ejes(ax, "Campo de velocidad — " + cab, ventana, paso, etiq == "zoom")
        fig.tight_layout()
        _guarda(fig, p, "campos", f"{base}_velocidad_{etiq}.png")

        fig, ax = plt.subplots(figsize=(10.5, 6.8))
        ax.set_facecolor("#f7f7f7")
        st = ax.streamplot(xi, yi, np.where(dentro, 0.0, u), np.where(dentro, 0.0, v),
                           color=speed, cmap="turbo", norm=Normalize(0.0, VMAX),
                           density=2.4 if etiq == "zoom" else 3.0,
                           linewidth=0.9, arrowsize=0.8)
        _cuerpo(ax, xi, yi, mask)
        fig.colorbar(st.lines, ax=ax, fraction=0.035, pad=0.02).set_label(
            r"$|\mathbf{u}|/U_\infty$ [-]")
        _ejes(ax, "Líneas de corriente — " + cab, ventana, paso, etiq == "zoom")
        fig.tight_layout()
        _guarda(fig, p, "campos", f"{base}_streamlines_{etiq}.png")

        if etiq != "zoom":
            continue

        # Cp = 2(p - p_inf) con rho=1 y U_inf=1. p_inf se toma de la columna de
        # entrada del propio campo remuestreado, no de cero: el solver deja la
        # presion definida salvo constante.
        cp_ = np.where(dentro, np.nan, 2.0 * (pr - np.nanmedian(pr[:, 0])))
        lim = float(np.nanpercentile(np.abs(cp_), 99)) or 1.0
        fig, ax = plt.subplots(figsize=(10.5, 6.8))
        im = ax.pcolormesh(xi, yi, cp_, cmap="RdBu_r", shading="auto",
                           norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim))
        ax.contour(xi, yi, np.nan_to_num(cp_), levels=14, colors="k",
                   linewidths=0.35, alpha=0.5)
        _cuerpo(ax, xi, yi, mask)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label(r"$C_p$ [-]")
        _ejes(ax, "Coeficiente de presión — " + cab, ventana, paso, True)
        fig.tight_layout()
        _guarda(fig, p, "campos", f"{base}_presion_zoom.png")

        # Vorticidad: donde se ve la burbuja de separacion laminar y el
        # desprendimiento, que es lo que rompe el Richardson del Cd.
        dy = yi[1] - yi[0]
        dxx = xi[1] - xi[0]
        w = np.gradient(v, dxx, axis=1) - np.gradient(u, dy, axis=0)
        w = np.where(dentro, np.nan, w)
        lw = float(np.nanpercentile(np.abs(w), 98)) or 1.0
        fig, ax = plt.subplots(figsize=(10.5, 6.8))
        im = ax.pcolormesh(xi, yi, w, cmap="RdBu_r", shading="auto",
                           norm=TwoSlopeNorm(vcenter=0.0, vmin=-lw, vmax=lw))
        _cuerpo(ax, xi, yi, mask)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label(
            r"$\omega_z\,c/U_\infty$ [-]")
        _ejes(ax, "Vorticidad — " + cab, ventana, paso, True)
        fig.tight_layout()
        _guarda(fig, p, "campos", f"{base}_vorticidad_zoom.png")

        s = 14
        fig, ax = plt.subplots(figsize=(10.5, 6.8))
        im = ax.pcolormesh(xi, yi, speed, cmap="Blues", shading="auto",
                           vmin=0.0, vmax=VMAX, alpha=0.75)
        ax.quiver(xi[::s], yi[::s], np.where(dentro, np.nan, u)[::s, ::s],
                  np.where(dentro, np.nan, v)[::s, ::s],
                  scale=28, width=0.0022, color="#111111", zorder=4)
        _cuerpo(ax, xi, yi, mask)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label(
            r"$|\mathbf{u}|/U_\infty$ [-]")
        _ejes(ax, "Campo de vectores — " + cab, ventana, paso, True)
        fig.tight_layout()
        _guarda(fig, p, "campos", f"{base}_vectores_zoom.png")


# ----------------------------------------------------------------------
# Historia temporal
# ----------------------------------------------------------------------
def _acota(ax, t, v, frac=0.12):
    """Acota el eje Y al rango del flujo ya arrancado.

    El impulso de arranque da picos de Cl~17 y Cd~4 en las primeras decenas de
    pasos. Con el eje completo, el 99% de la serie —que es lo que se promedia y
    lo unico que interesa— queda comprimido en una franja de un pixel. Se corta
    el primer `frac` del tiempo para fijar los limites, pero la curva se dibuja
    entera y se avisa en el propio eje.
    """
    if len(t) < 10:
        return
    m = t >= t[0] + frac * (t[-1] - t[0])
    w = np.asarray(v)[m]
    w = w[np.isfinite(w)]
    if not len(w):
        return
    lo, hi = float(np.min(w)), float(np.max(w))
    if hi <= lo:
        lo, hi = lo - 1e-3, hi + 1e-3
    pad = 0.12 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.text(0.995, 0.04, f"transitorio de arranque (t<{t[0]+frac*(t[-1]-t[0]):.1f}) "
            "fuera de escala", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="#666666")


def figuras_historia(p, dx, a, meta):
    f = RF.serie_path(p, dx, a)
    if not os.path.exists(f):
        return
    z = np.load(f)
    t, cl, cd = np.asarray(z["t"]), np.asarray(z["cl"]), np.asarray(z["cd"])
    base = f"dx{RF.k_dx(dx)}_a{a:04.1f}"
    tit = (f"{RF.PERFILES[p]['nombre']} — α={a:.0f}°, dx={dx:g}, Re=1e5")
    # Ventana de promediado: el ultimo 20%, igual que la que produce las metricas.
    t_win = t[int(0.8 * len(t)):]

    fig, ax = plt.subplots(2, 1, figsize=(10.0, 6.4), sharex=True)
    for k, (v, et, col, med, ci) in enumerate((
            (cl, r"$C_l$", "#c0392b", meta.get("cl"), meta.get("cl_ci95")),
            (cd, r"$C_d$", "#1f4e79", meta.get("cd"), meta.get("cd_ci95")))):
        ax[k].plot(t, v, color=col, lw=0.9)
        _acota(ax[k], t, v)
        if med is not None:
            ax[k].axhline(med, color="k", ls="--", lw=1.0,
                          label=f"media ventana = {med:.5f}" +
                                (f" ± {ci:.5f} (CI95)" if ci else ""))
        if len(t_win):
            ax[k].axvspan(t_win[0], t[-1], color="#999999", alpha=0.15,
                          label="ventana de promediado (último 20%)")
        ax[k].set_ylabel(et)
        ax[k].grid(alpha=0.25)
        ax[k].legend(fontsize=8, loc="best")
    ax[1].set_xlabel(r"$t\,U_\infty/c$ [-]")
    ax[0].set_title("Historia de fuerzas — " + tit, fontsize=11)
    fig.tight_layout()
    _guarda(fig, p, "historia", f"{base}_fuerzas.png")

    ld = np.where(np.abs(cd) > 1e-12, cl / cd, np.nan)
    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    ax.plot(t, ld, color="#1e8449", lw=0.9)
    _acota(ax, t, ld)
    if meta.get("ld") is not None:
        ax.axhline(meta["ld"], color="k", ls="--", lw=1.0,
                   label=f"media ventana = {meta['ld']:.3f}" +
                         (f" ± {meta['ld_ci95']:.3f} (CI95)" if meta.get("ld_ci95") else ""))
    if len(t_win):
        ax.axvspan(t_win[0], t[-1], color="#999999", alpha=0.15,
                   label="ventana de promediado")
    if meta.get("t_parada_offline"):
        ax.axvline(meta["t_parada_offline"], color="#e67e22", ls=":", lw=1.4,
                   label=f"criterio de parada habría cortado en t={meta['t_parada_offline']:.1f}")
    ax.set_xlabel(r"$t\,U_\infty/c$ [-]")
    ax.set_ylabel(r"$C_l/C_d$ [-]")
    ax.set_title("Eficiencia aerodinámica — " + tit, fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    _guarda(fig, p, "historia", f"{base}_eficiencia.png")

    if "res" in z.files:
        r = np.asarray(z["res"])
        fig, ax = plt.subplots(figsize=(10.0, 4.2))
        for j, et in enumerate(("residuo u", "residuo v", "divergencia")[:r.shape[1]]):
            ax.semilogy(t, np.maximum(np.abs(r[:, j]), 1e-16), lw=0.9, label=et)
        ax.set_xlabel(r"$t\,U_\infty/c$ [-]")
        ax.set_ylabel("residuo [-]")
        ax.set_title("Residuos — " + tit, fontsize=11)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        _guarda(fig, p, "historia", f"{base}_residuos.png")


def figuras_punto(p, dx, a, meta):
    figuras_campo(p, dx, a, meta)
    figuras_historia(p, dx, a, meta)


# ----------------------------------------------------------------------
# Polares por perfil
# ----------------------------------------------------------------------
COLOR_DX = {0.008: "#e67e22", 0.006: "#16a085", 0.004: "#8e44ad",
            0.002: "#1f4e79", 0.001: "#c0392b"}
MARCA_DX = {0.008: "D", 0.006: "v", 0.004: "^", 0.002: "s", 0.001: "o"}
# Con .get: anadir una malla no debe tumbar un estudio de horas por un color.
_COL = lambda dx: COLOR_DX.get(dx, "#555555")
_MAR = lambda dx: MARCA_DX.get(dx, "x")


def figuras_polar(p):
    datos = {dx: RF.cargar(p, dx) for dx in RF.DXS}
    if not any(datos.values()):
        return
    for mag, et, lab in (("cl", "cl_ci95", r"$C_l$"),
                         ("cd", "cd_ci95", r"$C_d$"),
                         ("ld", "ld_ci95", r"$C_l/C_d$")):
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        for dx in RF.DXS:
            d = datos[dx]
            aa = sorted(d, key=float)
            if not aa:
                continue
            ax.errorbar([float(k) for k in aa], [d[k][mag] for k in aa],
                        yerr=[d[k].get(et) or 0.0 for k in aa],
                        color=_COL(dx), marker=_MAR(dx), ms=5, lw=1.4,
                        capsize=3, label=f"dx = {dx:g}")
        ax.set_xlabel(r"$\alpha$ [°]")
        ax.set_ylabel(lab)
        ax.set_title(f"{lab} frente a α — {RF.PERFILES[p]['nombre']}\n"
                     "tres mallas, barras = CI95", fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        _guarda(fig, p, "polar", f"polar_{mag}.png")

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    for dx in RF.DXS:
        d = datos[dx]
        aa = sorted(d, key=float)
        if not aa:
            continue
        ax.plot([d[k]["cd"] for k in aa], [d[k]["cl"] for k in aa],
                color=_COL(dx), marker=_MAR(dx), ms=5, lw=1.4,
                label=f"dx = {dx:g}")
        for k in aa:
            ax.annotate(f"{float(k):.0f}°", (d[k]["cd"], d[k]["cl"]),
                        fontsize=7, xytext=(4, -2), textcoords="offset points")
    ax.set_xlabel(r"$C_d$ [-]")
    ax.set_ylabel(r"$C_l$ [-]")
    ax.set_title(f"Polar — {RF.PERFILES[p]['nombre']}", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _guarda(fig, p, "polar", "polar_cl_cd.png")


# ----------------------------------------------------------------------
# Geometria y comparativa entre perfiles
# ----------------------------------------------------------------------
def figura_perfiles():
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    for p, cfg in RF.PERFILES.items():
        f = os.path.join(RF.OUT, "perfiles", f"{p}.dat")
        if not os.path.exists(f):
            continue
        xy = np.loadtxt(f, skiprows=1)
        # El .dat no repite el primer punto y el ganador tiene el borde de salida
        # romo: sin cerrar el contorno la figura enseña un perfil abierto.
        xy = np.vstack([xy, xy[:1]])
        ax.plot(xy[:, 0], xy[:, 1], color=cfg["color"], lw=1.5, label=cfg["nombre"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title("Geometría de los dos perfiles del estudio", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RF.OUT, "perfiles", "perfiles_comparados.png"),
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def figuras_comparativa():
    """Los dos perfiles en la malla fina, que es la unica comparacion honesta."""
    dx = RF.TERNA[0]                      # la mas fina de la terna en uso
    datos = {p: RF.cargar(p, dx) for p in RF.PERFILES}
    if not all(datos.values()):
        return
    fig, axs = plt.subplots(1, 3, figsize=(15.0, 4.4))
    for ax, (mag, et, lab) in zip(axs, (("cl", "cl_ci95", r"$C_l$"),
                                        ("cd", "cd_ci95", r"$C_d$"),
                                        ("ld", "ld_ci95", r"$C_l/C_d$"))):
        for p, cfg in RF.PERFILES.items():
            d = datos[p]
            aa = sorted(d, key=float)
            ax.errorbar([float(k) for k in aa], [d[k][mag] for k in aa],
                        yerr=[d[k].get(et) or 0.0 for k in aa],
                        color=cfg["color"], marker="o", ms=5, lw=1.5, capsize=3,
                        label=p)
        ax.set_xlabel(r"$\alpha$ [°]")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle(f"Ganador de la optimización frente a NACA 0012 — malla dx={dx:g}, "
                 f"Re=1e5, barras = CI95",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RF.OUT, "comparativa", f"ganador_ag_vs_naca0012_dx{RF.k_dx(dx)}.png"),
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def figuras_richardson():
    f = os.path.join(RF.OUT, "richardson", "richardson.json")
    if not os.path.exists(f):
        return
    todo = json.load(open(f))
    for p, res in todo.items():
        if not res:
            continue
        aa = sorted(res, key=float)
        # Una lamina por magnitud: las tres juntas dejaban cada panel a 5 pulgadas
        # con nueve curvas dentro y no se distinguia el tramo extrapolado.
        for mag, et in RF.MAGNITUDES:
            fig, ax = plt.subplots(figsize=(7.4, 5.4))
            # Las mallas salen de la terna configurada, no de una lista fija: con
            # dx=0.001 escrita a mano esta figura seguia dibujando la malla que se
            # descarto por colapso de dt y se dejaba fuera la de 0.008.
            h = list(RF.TERNA[::-1])          # gruesa -> fina
            for ka in aa:
                g = res[ka][mag]
                y = [g["f_gruesa"], g["f_media"], g["f_fina"]]
                fiable = g.get("p_fiable", g["citable"])
                l, = ax.plot(h, y, marker="o", ms=4, lw=1.2,
                             ls="-" if fiable else ":",
                             label=f"α={float(ka):.0f}°" + ("" if fiable else " (p forzado)"))
                # Extrapolado robusto: el de Richardson puro se dispara cuando el
                # orden observado tiende a 0 (Cl del ganador en alpha=6 daba 2.91)
                # y cruza a negativo en el Cd.
                ext = g.get("f_extrapolado_robusto", g["f_extrapolado_richardson"])
                # Tramo malla fina -> h=0 discontinuo: es extrapolacion, no un dato
                # medido; sin la linea el salto a la estrella se lee como ruido.
                ax.plot([h[-1], 0.0], [y[-1], ext], ls="--", lw=1.0, alpha=.75,
                        color=l.get_color())
                ax.plot(0.0, ext, marker="*", ms=11, color=l.get_color())
            ax.set_xlabel("dx  (tramo discontinuo = extrapolación a h→0 ★)")
            ax.set_ylabel(et)
            ax.set_xlim(-0.0004, max(h) * 1.12)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
            ax.set_title(f"Convergencia de malla en {et} — {RF.PERFILES[p]['nombre']}\n"
                         "línea continua = orden observado utilizable; "
                         "punteada = p forzado al nominal", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(RF.OUT, "richardson", "figuras",
                                     f"convergencia_{p}_{mag}.png"),
                        dpi=DPI, bbox_inches="tight")
            plt.close(fig)


def todas():
    figura_perfiles()
    for p in RF.PERFILES:
        for dx in RF.DXS:
            d = RF.cargar(p, dx)
            for ka, meta in d.items():
                figuras_punto(p, dx, float(ka), meta)
        figuras_polar(p)
    figuras_comparativa()
    figuras_richardson()


if __name__ == "__main__":
    todas()

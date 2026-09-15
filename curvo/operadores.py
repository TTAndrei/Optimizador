r"""Operadores de volumenes finitos sobre la malla curvilinea.

Todo es conservativo por construccion: lo unico que se calcula son **flujos por
cara**, y la divergencia es su balance. Lo que entra por una cara sale por la
misma cara de la celda vecina, con lo que cualquier cantidad transportada se
conserva a nivel discreto. Es la propiedad que el semi-Lagrangiano no tiene
(backtrace + interpolacion bilineal) y el motivo de cambiar de esquema.

El laplaciano lleva **metrica completa**: los terminos cruzados xi-eta van
dentro, no se desprecian. Sobre las mallas generadas la oblicuidad p99 vale
0.03-0.06 en perfiles reales pero 0.33 de mediana sobre el historial del GA, asi
que la aproximacion ortogonal (ADI, splitting direccional) se cae justo en las
geometrias a las que tiende el optimizador.

Condiciones de frontera. Los cuatro lados del rectangulo computacional se pasan
como valores de Dirichlet **en el centro de la cara**; `None` significa gradiente
nulo. En la malla C: `sur` = pared (j=0), `norte` = campo lejano, `oeste` y
`este` = plano de salida.
"""

from __future__ import annotations

from .metrica import xp_de

__all__ = [
    "dif_xi_caras",
    "dif_eta_caras",
    "a_caras_xi",
    "a_caras_eta",
    "divergencia",
    "gradiente",
    "laplaciano",
]


# ---------------------------------------------------------------------------
# Derivadas en caras
# ---------------------------------------------------------------------------
def dif_xi_caras(phi, oeste=None, este=None):
    """d(phi)/d(xi) en las caras xi. phi (ny,nx) -> (ny, nx+1).

    En las caras de borde la distancia al centro de celda es media celda, de ahi
    el factor 2. Sin valor de Dirichlet se toma gradiente nulo.
    """
    xp = xp_de(phi)
    ny, nx = phi.shape
    d = xp.zeros((ny, nx + 1), dtype=phi.dtype)
    d[:, 1:-1] = phi[:, 1:] - phi[:, :-1]
    if oeste is not None:
        d[:, 0] = 2.0 * (phi[:, 0] - oeste)
    if este is not None:
        d[:, -1] = 2.0 * (este - phi[:, -1])
    return d


def dif_eta_caras(phi, sur=None, norte=None):
    """d(phi)/d(eta) en las caras eta. phi (ny,nx) -> (ny+1, nx)."""
    xp = xp_de(phi)
    ny, nx = phi.shape
    d = xp.zeros((ny + 1, nx), dtype=phi.dtype)
    d[1:-1, :] = phi[1:, :] - phi[:-1, :]
    if sur is not None:
        d[0, :] = 2.0 * (phi[0, :] - sur)
    if norte is not None:
        d[-1, :] = 2.0 * (norte - phi[-1, :])
    return d


def a_caras_xi(g_eta):
    """Lleva una magnitud de caras eta (ny+1,nx) a caras xi (ny,nx+1).

    Media de las cuatro caras eta que rodean a la cara xi, replicando en el
    borde. Es el mismo estencil con el que `Metrica` promedia `S_eta`, asi que
    coeficiente y derivada se evaluan en el mismo sitio.
    """
    xp = xp_de(g_eta)
    p = xp.concatenate([g_eta[:, :1], g_eta, g_eta[:, -1:]], axis=1)
    return 0.25 * (p[:-1, :-1] + p[:-1, 1:] + p[1:, :-1] + p[1:, 1:])


def a_caras_eta(g_xi):
    """Lleva una magnitud de caras xi (ny,nx+1) a caras eta (ny+1,nx)."""
    xp = xp_de(g_xi)
    p = xp.concatenate([g_xi[:1, :], g_xi, g_xi[-1:, :]], axis=0)
    return 0.25 * (p[:-1, :-1] + p[:-1, 1:] + p[1:, :-1] + p[1:, 1:])


# ---------------------------------------------------------------------------
# Divergencia y gradiente
# ---------------------------------------------------------------------------
def divergencia(F_xi, F_eta):
    """Balance de flujos por celda. F_xi (ny,nx+1), F_eta (ny+1,nx) -> (ny,nx).

    Integrada sobre el volumen: no se divide por J. Un campo de flujos de una
    corriente uniforme da cero en maquina por telescopado de las areas de cara.
    """
    return (F_xi[:, 1:] - F_xi[:, :-1]) + (F_eta[1:, :] - F_eta[:-1, :])


def gradiente(met, phi, oeste=None, este=None, sur=None, norte=None):
    """Gradiente por celda (Green-Gauss). Devuelve (gx, gy), ambos (ny,nx).

    Exacto para campos lineales: la suma de `phi_cara * S_cara` sobre un
    poligono cerrado reproduce el gradiente constante sin error.
    """
    xp = xp_de(phi)
    ny, nx = phi.shape

    f_xi = xp.zeros((ny, nx + 1), dtype=phi.dtype)
    f_xi[:, 1:-1] = 0.5 * (phi[:, :-1] + phi[:, 1:])
    f_xi[:, 0] = phi[:, 0] if oeste is None else oeste
    f_xi[:, -1] = phi[:, -1] if este is None else este

    f_eta = xp.zeros((ny + 1, nx), dtype=phi.dtype)
    f_eta[1:-1, :] = 0.5 * (phi[:-1, :] + phi[1:, :])
    f_eta[0, :] = phi[0, :] if sur is None else sur
    f_eta[-1, :] = phi[-1, :] if norte is None else norte

    gx = divergencia(f_xi * met.Sx_xi, f_eta * met.Sx_eta) / met.J
    gy = divergencia(f_xi * met.Sy_xi, f_eta * met.Sy_eta) / met.J
    return gx, gy


# ---------------------------------------------------------------------------
# Laplaciano de metrica completa
# ---------------------------------------------------------------------------
def _nu_en_caras(met, nu):
    """Viscosidad en caras xi y eta. Acepta escalar o campo por celda."""
    if not hasattr(nu, "shape") or nu.shape != met.J.shape:
        return nu, nu
    xp = met.xp
    p_i = xp.concatenate([nu[:, :1], nu, nu[:, -1:]], axis=1)
    p_j = xp.concatenate([nu[:1, :], nu, nu[-1:, :]], axis=0)
    return 0.5 * (p_i[:, :-1] + p_i[:, 1:]), 0.5 * (p_j[:-1, :] + p_j[1:, :])


def flujos_difusivos(met, phi, nu=1.0, oeste=None, este=None, sur=None, norte=None):
    """Flujos difusivos por cara, con terminos cruzados.

        grad(phi) . S_xi  = a_xi  * phi_xi  + b_xi  * phi_eta
        grad(phi) . S_eta = a_eta * phi_eta + b_eta * phi_xi

    Devuelve (F_xi, F_eta) ya multiplicados por nu.
    """
    d_xi = dif_xi_caras(phi, oeste, este)
    d_eta = dif_eta_caras(phi, sur, norte)
    nu_xi, nu_eta = _nu_en_caras(met, nu)
    F_xi = nu_xi * (met.a_xi * d_xi + met.b_xi * a_caras_xi(d_eta))
    F_eta = nu_eta * (met.a_eta * d_eta + met.b_eta * a_caras_eta(d_xi))
    return F_xi, F_eta


def laplaciano(met, phi, nu=1.0, oeste=None, este=None, sur=None, norte=None,
               integrado=False):
    """div(nu grad phi). Por unidad de volumen salvo `integrado=True`.

    Un campo lineal da cero en maquina: el flujo de un gradiente constante
    telescopa sobre la celda cerrada igual que la corriente uniforme. Es el test
    barato que caza cualquier error de metrica.
    """
    F_xi, F_eta = flujos_difusivos(met, phi, nu, oeste, este, sur, norte)
    d = divergencia(F_xi, F_eta)
    return d if integrado else d / met.J

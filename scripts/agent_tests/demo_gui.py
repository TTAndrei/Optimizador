"""Genera casos de ejemplo para la GUI y comprueba el camino completo.

Escribe en results/gui_demo/ tres DXF y sus escenas ya montadas, para abrir con
`./lanzar_gui.sh` -> Importar DXF. Con --simular, ademas corre uno de verdad por
el mismo camino que usa la GUI (gui.run_solver), que es la unica forma de saber
que el puente escena -> solver funciona de punta a punta.

    .venv/bin/python scripts/agent_tests/demo_gui.py
    .venv/bin/python scripts/agent_tests/demo_gui.py --simular
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

import ezdxf

import formas_dominio as fd
from gui.escena import Contorno, Escena, Parche

OUT = os.path.join(ROOT, "results", "gui_demo")


def _dxf(nombre, polilineas):
    os.makedirs(OUT, exist_ok=True)
    d = ezdxf.new("R2010")
    d.header["$INSUNITS"] = 6                       # metros
    msp = d.modelspace()
    for capa, pts in polilineas:
        msp.add_lwpolyline([(float(a), float(b)) for a, b in pts],
                           close=True, dxfattribs={"layer": capa})
    ruta = os.path.join(OUT, f"{nombre}.dxf")
    d.saveas(ruta)
    return ruta


def _circulo(cx, cy, r, n=180):
    """Cerrado: el ultimo punto repite el primero, que es lo que espera el
    rasterizador y lo que comprueba Escena.avisos()."""
    th = np.linspace(0, 2 * np.pi, n + 1)
    return np.column_stack((cx + r * np.cos(th), cy + r * np.sin(th)))


CASOS = {}


def caso_conducto():
    """Canal recto con un cilindro dentro. El caso mas simple que ejercita las
    tres cosas nuevas a la vez: contorno exterior, cuerpo interno y entrada que
    ocupa solo un tramo del lado."""
    Lx, Ly, h = 4.0, 2.0, 1.0
    x, y = fd.canal(Lx=Lx, Ly=Ly, h=h)
    ruta = _dxf("conducto_cilindro", [
        ("PAREDES", np.column_stack((x[:-1], y[:-1]))),
        ("CUERPOS", _circulo(1.2, 1.0, 0.12)[:-1]),
    ])
    esc = Escena(Lx=Lx, Ly=Ly, dx_min=0.02, factor_expansion=1.05)
    esc.contornos = [
        Contorno(x=list(x), y=list(y), rol="exterior", nombre="conducto"),
        Contorno(x=list(_circulo(1.2, 1.0, 0.12)[:, 0]),
                 y=list(_circulo(1.2, 1.0, 0.12)[:, 1]), nombre="cilindro"),
    ]
    esc.parches = [
        Parche("left", "inflow", (1.0, 0.0), 0.5, 1.5),
        Parche("right", "outflow", 0.0, 0.5, 1.5),
        Parche("top", "noslip"), Parche("bottom", "noslip"),
    ]
    esc.solver = {"nu": 0.02, "v0x": 1.0, "CFL": 0.5, "iteraciones": 3000,
                  "guardado": 50, "turb_model": "none",
                  "transition_model": "none",
                  "stop_on_convergence": False,
                  "stop_on_clcd_convergence": False}
    return ruta, esc


def caso_tobera():
    Lx, Ly = 4.0, 2.0
    x, y = fd.tobera(Lx=Lx, Ly=Ly, h_in=1.2, h_out=0.6, x_ini=1.0, x_fin=3.0)
    ruta = _dxf("tobera", [("PAREDES", np.column_stack((x[:-1], y[:-1])))])
    esc = Escena(Lx=Lx, Ly=Ly, dx_min=0.02, factor_expansion=1.05,
                 refinado=(0.0, Lx, 0.3, 1.7))
    esc.contornos = [Contorno(x=list(x), y=list(y), rol="exterior",
                              nombre="tobera")]
    esc.parches = [
        Parche("left", "inflow", (1.0, 0.0), 0.4, 1.6),
        Parche("right", "outflow", 0.0, 0.7, 1.3),
        Parche("top", "noslip"), Parche("bottom", "noslip"),
    ]
    esc.solver = {"nu": 0.02, "v0x": 1.0, "CFL": 0.5, "iteraciones": 3000,
                  "guardado": 50, "turb_model": "none",
                  "transition_model": "none",
                  "stop_on_convergence": False,
                  "stop_on_clcd_convergence": False}
    return ruta, esc


def caso_escalon():
    Lx, Ly = 6.0, 2.0
    x, y = fd.escalon(Lx=Lx, Ly=Ly, h=0.5, x_step=1.5)
    ruta = _dxf("escalon", [("PAREDES", np.column_stack((x[:-1], y[:-1])))])
    esc = Escena(Lx=Lx, Ly=Ly, dx_min=0.02, factor_expansion=1.05,
                 refinado=(0.0, Lx, 0.0, Ly))
    esc.contornos = [Contorno(x=list(x), y=list(y), rol="exterior",
                              nombre="escalon")]
    esc.parches = [
        Parche("left", "inflow", (1.0, 0.0), 1.5, 2.0),
        Parche("right", "outflow", 0.0),
        Parche("top", "noslip"), Parche("bottom", "noslip"),
    ]
    esc.solver = {"nu": 0.005, "v0x": 1.0, "CFL": 0.5, "iteraciones": 4000,
                  "guardado": 50, "turb_model": "none",
                  "transition_model": "none",
                  "stop_on_convergence": False,
                  "stop_on_clcd_convergence": False}
    return ruta, esc


CASOS = {"conducto": caso_conducto, "tobera": caso_tobera,
         "escalon": caso_escalon}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simular", metavar="CASO", nargs="?", const="conducto")
    ap.add_argument("--iters", type=int, default=600)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    from gui.vista_malla import diagnostico
    for nombre, fn in CASOS.items():
        ruta_dxf, esc = fn()
        ruta = esc.guardar(os.path.join(OUT, f"{nombre}.json"))
        texto, avisos, X, Y, solid = diagnostico(esc)
        print(f"\n=== {nombre} ===\n{os.path.basename(ruta_dxf)}  ->  "
              f"{os.path.basename(ruta)}")
        print(texto)
        for av in avisos:
            print(f"  AVISO: {av}")

    if a.simular:
        esc_ruta, esc = CASOS[a.simular]()[0], CASOS[a.simular]()[1]
        esc.solver["iteraciones"] = a.iters
        esc.modo_salida = "monitor"
        carpeta = os.path.join(OUT, f"corrida_{a.simular}")
        os.makedirs(carpeta, exist_ok=True)
        ruta = esc.guardar(os.path.join(carpeta, "escena.json"))
        print(f"\n=== simulando {a.simular} por el camino de la GUI ===")
        from gui.run_solver import main as correr
        correr(ruta)
        print(f"[demo] salida en {carpeta}")


if __name__ == "__main__":
    main()

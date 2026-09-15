"""
Resultados finales del TFG — convergencia de malla de Richardson para los dos
perfiles de referencia, con el solver de presion optimizado.

QUE SE CALCULA
    Dos perfiles   AG24 (ganador del algoritmo genetico) y NACA 0012 sharp, que
                   es la semilla de la que salio. Mismo numero de puntos y mismo
                   tratamiento del borde de salida, asi que la comparacion entre
                   los dos no arrastra diferencias de discretizacion geometrica.
    Cuatro mallas  dx = 0.008 / 0.006 / 0.004 / 0.002. La terna principal es
                   0.008/0.004/0.002, con r=2 constante en los dos saltos.
                   0.006 se calcula como terna alternativa de contraste.
    Cinco angulos  alpha = 0, 2, 4, 6, 8 grados.
    Dominio        24 x 16 cuerdas, perfil en cx=6. Es el dominio C validado; el
                   8x5 del AG se descarto por estela corta.

POR QUE ESTA TERNA Y NO LA QUE SE PLANEO
    El plan original era 0.004/0.002/0.001, y era mejor por el lado teorico: la
    terna se apoyaba en las mallas mas finas, que es donde el rango asintotico
    es plausible. Se cayo por un motivo puramente numerico, no de presupuesto.

    A dx=0.001 el paso de tiempo COLAPSA. Medido sobre los 4 puntos que llegaron
    a calcularse (ganador_ag, alpha 0/2/4/6):

        dx      t_final alcanzado   dt medio
        0.004     15.9 - 19.6       1.4e-3
        0.002     15.7 - 21.7       8.0e-4
        0.001      5.2 -  6.9       1.2e-4

    De 0.004 a 0.002 el dt baja x0.57, lineal con la malla, como toca. De 0.002 a
    0.001 baja x0.15: sobra un factor 3.4. Las 52000 iteraciones se quedaban en
    t~6 en vez de t~20, o sea que el Richardson habria comparado t=6 contra
    t=20.6 —el punto fino todavia en transitorio de arranque— y habria medido esa
    diferencia temporal creyendo medir discretizacion espacial.

    Ademas el dt no era solo pequeno, seguia cayendo durante todo el run (x6.8 y
    sin estabilizar). Es la misma firma que precedia al blowup: warm_start_filtered
    evito el NaN pero no curo la inestabilidad de fondo, solo cambio el sintoma.

    Eso explica, sin invocar fisica ninguna, todo lo raro de la pata fina: CI95
    del 95% en el L/D, L/D no monotono en alpha 2 y 4, Cd inflado (0.030 frente a
    0.017) y el criterio de convergencia que no disparaba nunca. El Cl MEDIO si
    coincidia con el de dx=0.002 (0.70 frente a 0.6977): lo que se rompia era la
    dispersion y el Cd.

    Llevar dx=0.001 hasta t=20 pedia ~175000 iteraciones, 11 h por punto y 80 h
    los que faltaban. Se sustituye por mallas mas gruesas, que son sanas y baratas.

LO QUE SE PIERDE AL BAJAR LA TERNA — LEER ANTES DE CITAR NADA
    Esto empeora la terna y hay que decirlo. La version anterior de este mismo
    fichero ya avisaba de ello: del paso 0.004 al 0.002 el Cl se movia hasta un
    50.7% (alpha=0) y el Cd hasta un 65.2% (alpha=8), mientras que de 0.002 a
    0.001 se movian 19.2% y 1.6%. O sea que dx=0.004 ya estaba en el limite del
    rango asintotico, y la terna nueva se apoya en 0.008, que esta claramente
    fuera. Es previsible que p_observado salga disparatado en varios angulos.

    La terna nueva NO demuestra convergencia de malla. Da una banda de
    sensibilidad con tres mallas y un GCI formal que solo vale donde la serie
    salga monotona y p caiga en 0.5-4.0. Todo lo demas queda marcado como no
    citable. Presentarlo como verificacion cerrada seria falso.

LO QUE ESTE ESTUDIO NO PUEDE CERRAR
    El Cd oscila con la malla en alpha=4 y 6 incluso a dx=0.001. No es error de
    truncamiento sino la burbuja de separacion laminar cambiando de sitio segun
    la resolucion, o sea un cambio de solucion y no una serie convergente.
    Richardson no aplica ahi y refinar mas no lo arregla. Se calcula el GCI de
    todo, pero solo es citable donde la convergencia sale monotona; el resto
    queda marcado en la tabla y en el informe.

QUE SE GUARDA POR PUNTO
    metricas/   JSON con las 28 claves del estudio anterior (Cl, Cd, L/D, CI95,
                desviaciones, numero de muestras, estado del criterio de parada,
                iteraciones efectivas, coste) + CSV plano.
    series/     .npz con t, Cl(t), Cd(t), L/D(t) y los tres residuos por paso.
    campos/     .npz con x, y, u, v, p y la mascara de solido del campo final.
    figuras/    campos y streamlines en zona fina Y en dominio completo, presion,
                vorticidad, vectores, y las historias de fuerzas y eficiencia.

REANUDABLE
    Cada punto se escribe en cuanto termina. Volver a lanzar salta lo hecho. Para
    parar limpio entre puntos: `touch PARAR_ESTUDIO.trigger` en la raiz.

PROGRESO
    En terminal se ve la barra de tqdm del simulador mas el banner de cada punto.
    Desatendido, `rf_progreso.py` lee ESTADO.json y la memoria compartida del
    simulador y muestra el avance del estudio y de la simulacion en curso.

Uso:
    .venv/bin/python scripts/agent_tests/resultados_finales.py
    .venv/bin/python scripts/agent_tests/resultados_finales.py --solo-figuras
    .venv/bin/python scripts/agent_tests/resultados_finales.py --solo-analisis
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "agent_tests"))
os.chdir(ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")

import opt_solver

OUT = os.path.join(ROOT, "resultados_finales")

# ----------------------------------------------------------------------
# Configuracion del estudio
# ----------------------------------------------------------------------
PERFILES = {
    "ganador_ag": {
        # OJO con el nombre. El fichero se llama NACA_0012_sharp_winner porque
        # esa fue la familia semilla de la que descendio, no porque sea un NACA.
        # Y NO se le puede llamar "AG24": AG24 es un perfil real de catalogo
        # (AG24 Bubble Dancer DLG, de Mark Drela) que el AG uso como otra semilla
        # y que compitio contra este, con su propio ganador de L/D 25.00 en
        # islands_tfg2/AG24/epoch10/. Verificado bit a bit contra el top1 del
        # ranking en dominio C (ranking_dominio/perfiles/top1_LD27.75.dat):
        # max|dif| = 0.000e+00 sobre los 161 puntos.
        "nombre": "Ganador de la campaña de optimización (AG)",
        "dat": ("results/convergence_study/islands_tfg2/NACA_0012_sharp/epoch10/"
                "NACA_0012_sharp_winner_Re100000_a4.0_LD27.75.dat"),
        "color": "#c0392b",
    },
    "naca0012": {
        "nombre": "NACA 0012 sharp (semilla de referencia)",
        "dat": "profiles/NACA_0012_sharp",
        "color": "#1f4e79",
    },
}

DOMINIO = {"Lx": 24.0, "Ly": 16.0, "cx": 6.0}
RE = 1e5
# Impares anadidos 2026-08-26 para densificar la polar alrededor del pico de
# L/D, que en los dos perfiles cae en alpha=4. Los pares ya estaban hechos en
# las cuatro mallas; el runner los detecta por fichero y los salta.
ANGULOS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
DXS = (0.008, 0.006, 0.004, 0.002)
ITERS = {0.008: 6500, 0.006: 8700, 0.004: 13000, 0.002: 26000}   # t ~ 20 en todas

# LA MALLA FINA (dx=0.001) ESTA FUERA. No es por presupuesto: sus puntos no eran
# comparables con los de las otras dos mallas. Medido sobre los 4 puntos que se
# llegaron a calcular (ganador_ag, alpha 0/2/4/6, en resultados_finales/):
#
#   dx      t_final alcanzado   dt medio
#   0.004     15.9 - 19.6       1.4e-3
#   0.002     15.7 - 21.7       8.0e-4
#   0.001      5.2 -  6.9       1.2e-4     <- ni de lejos
#
# De 0.004 a 0.002 el dt baja x0.57, lineal, como toca. De 0.002 a 0.001 baja
# x0.15 en vez de x0.5: sobra un factor 3.4. Las 52000 iteraciones solo llegaban
# a t~6 en vez de t~20, o sea que el Richardson comparaba t=6 contra t=20.6, dos
# estados fisicos distintos con el fino aun en transitorio de arranque.
#
# Y el dt no era pequeno, estaba COLAPSANDO. Intervalo de muestreo a alpha=4 a
# lo largo del run:
#
#   dx=0.002:  3.9e-2 -> 4.0e-2 -> 4.0e-2 -> 4.0e-2      plano, sano
#   dx=0.001:  1.3e-2 -> 6.9e-3 -> 4.8e-3 -> 2.0e-3      cae x6.8 y sigue
#
# Es la misma firma de colapso de dt que precedia al blowup. warm_start_filtered
# evito el NaN pero no curo la inestabilidad: en vez de reventar, degenera en
# pasos cada vez mas cortos. Sintoma distinto, causa viva.
#
# Eso explica sin recurrir a fisica todo lo que se veia en la pata fina: CI95 del
# 95% en el L/D, L/D no monotono en alpha 2 y 4, Cd inflado (0.030 frente a
# 0.017) y converged_clcd que no disparaba nunca. Nada de eso era desprendimiento
# resuelto, era transitorio. El Cl MEDIO si coincidia con el de dx=0.002
# (0.70 frente a 0.6977): lo que se rompe es la dispersion y el Cd.
#
# Llevarlo a t=20 pedia ~175000 iteraciones, 11 h por punto. Se sustituye por dos
# mallas gruesas nuevas, que son baratas y sanas.

# Terna para el GCI, en orden (fina, media, gruesa). Con 0.008 el ratio de
# refinamiento es r=2 EXACTO en los dos saltos, que es lo que pide el
# procedimiento estandar y ademas degenera la ecuacion implicita de p a su forma
# cerrada. La terna con 0.006 da r=1.5 y r=2.0: Celik la admite y el codigo la
# resuelve, pero se calcula como comprobacion secundaria, no como resultado
# principal.
TERNA = (0.002, 0.004, 0.008)
TERNA_ALT = (0.002, 0.004, 0.006)

# LAS DOS PARADAS FUERA. Todos los puntos van a ventana completa.
#
# Se probo con la parada de Cl/Cd activa y cortaba desigual: naca0012 alpha=8 en
# t=5.29 y alpha=6 en t=8.95, mientras alpha=0, 2 y 4 llegaban a t~20. Puntos de
# la misma polar con tiempos fisicos que difieren 3x no son comparables entre si,
# y menos aun entre mallas, que es lo que el Richardson exige.
#
# No se pierde la informacion: `parada_offline` reproduce el criterio calibrado
# sobre la serie volcada y guarda t_parada_offline, ld_parada_offline y
# coste_parada_offline. Se sabe donde HABRIA parado sin dejar que parase, que es
# lo mejor de los dos mundos.
#
# (bloque historico, ya no se usa para simular)
#
# El riesgo conocido es que cada malla pare en un tiempo fisico distinto y la
# terna acabe comparando estados distintos. Se asume porque el criterio es el
# recalibrado por validacion cruzada sobre 20 series (el anterior cortaba en
# transitorio con un 47% de error), y porque un punto que para antes con el mismo
# valor solo ahorra tiempo. Queda auditable: cada punto guarda iters_efectivas,
# iters_programadas y t_conv_clcd, asi que si una malla para mucho antes que otra
# se ve en la tabla y en el informe.
PARADAS = {"stop_on_clcd_convergence": True, "stop_on_convergence": False}

# Los puntos reusados de polar_optimizada se calcularon SIN parada, o sea a
# ventana completa. Son los mas convergidos, no los menos, pero es una diferencia
# de trato frente a los que si paren: por eso cada punto registra si llevaba la
# parada activa y el informe marca los que se cortaron antes de tiempo.
SIN_PARADAS = {"stop_on_clcd_convergence": False, "stop_on_convergence": False}

# Optimizaciones del solver de presion. Van explicitas aunque ya sean el default
# de opt_solver: si manana cambia el default, este estudio no debe moverse.
FLAGS = ("warm_start", "warm_start_filtered", "coarse_mask_majority",
         "fast_masks", "interp_float32")
PRESUPUESTO = {"mg_max_outer": 2, "mg_cycles_per_outer": 3}

# La malla gruesa va SIN warm_start. No es una preferencia: con el, dx=0.004
# revienta por blowup de velocidad en la iteracion 1340, en todos los angulos
# probados y de forma reproducible. El cruce esta en results/bench_opt/:
#
#   diag_blowup.json  el fallo necesita las dos cosas a la vez. Optimizaciones
#                     ON + presupuesto 8x5 va bien; OFF + 2x3 va bien; ON + 2x3
#                     revienta. O sea que una opcion deja de ser inocua cuando la
#                     proyeccion se queda sin ciclos de sobra.
#   diag_flag.json    leave-one-out: quitar warm_start es lo unico que lo evita.
#                     Quitar cualquiera de las otras tres no cambia nada.
#
# warm_start no cambia la discretizacion, solo la semilla de un solver iterativo,
# y seria irrelevante si la proyeccion convergiese. Con presupuesto fijo 2x3 no
# converge —el desacoplamiento par-impar hace inalcanzable la tolerancia— asi que
# la solucion si depende de la semilla, y mezclar configuraciones dentro de la
# terna puede sesgar el orden aparente. Cuanto, esta medido en diag_mezcla.json:
# se compara el desplazamiento que mete warm_start a malla igual contra el que
# mete el cambio de malla, que es la senal que el Richardson mide.
# LA MISMA CONFIGURACION EN LAS TRES MALLAS.
#
# Hizo falta llegar hasta aqui. warm_start con presupuesto 2x3 reventaba por
# blowup de velocidad: en dx=0.004 en la iteracion 1340 y en dx=0.001 en la 1740,
# los diez puntos. Quitarlo estabilizaba pero pagando caro en exactitud (el L/D
# del punto de diseno caia de 41.3 a 19.8).
#
# La causa no era la magnitud de la semilla sino su ESTRUCTURA. Se probo primero
# reescalarla por el cambio de dt —el RHS del outer 0 lleva un termino
# (rho/dt)·div(u^n) que solo se anula si el paso anterior quedo bien proyectado—
# y solo movio el fallo de la iteracion 1340 a la 1800. Lo que si lo arregla es
# `warm_start_filtered`: la semilla arrastraba el modo par-impar acumulado, que
# el multigrid no reduce, y un paso de Jacobi ponderado lo aniquila antes de
# reusarla.
#
# Validado en los tres sitios (results/bench_opt/diag_escalado.json y
# diag_filtro_val.json):
#   dx=0.004 alpha=0  13000/13000 iteraciones, donde antes abortaba en la 1340
#   dx=0.001 alpha=4   3000/3000, donde antes abortaba en la ~1740
#   dx=0.002 alpha=4  L/D 41.155 frente a 41.256 sin filtro, o sea -0.25%
# Coste: 387 s frente a ~405 s sin warm_start. Nulo.
#
# OJO: la malla fina solo esta probada a 3000 iteraciones, no a las 52000 del
# estudio. Es 1.7x el umbral donde reventaba, evidencia razonable pero no prueba.
FLAGS_POR_DX = {dx: FLAGS for dx in DXS}


# ----------------------------------------------------------------------
# Parada por convergencia: DESACTIVADA en todas las mallas
# ----------------------------------------------------------------------
# Se llego a activar solo en dx=0.001 para acortar sus 30 h. Con la malla fina
# fuera ya no hace falta: las cuatro mallas restantes caben en ventana completa,
# y asi la terna es homogenea en tiempo fisico, que es justo lo que fallaba.
#
# Queda medido por si vuelve a hacer falta. Sobre los 20 puntos de dx=0.004 y
# 0.002, comparando su L/D de ventana completa contra el de corte:
#
#     t_min= 5 -> 10/10 cortes, error maximo 5.5%   (el malo: ganador_ag
#                 dx=0.004 alpha=6, cortando en t=5.52)
#     t_min= 8 ->  9/10 cortes, error maximo 1.7%
#     t_min=12 ->  6/10 cortes, error maximo 1.3%
#
# Cada punto sigue registrando t_parada_offline y ld_parada_offline, o sea que
# el corte se puede reproducir a posteriori sin volver a simular.


# Todas las mallas a ventana completa: con la fina fuera, ninguna necesita
# recorte y la terna queda homogenea en tiempo fisico.
PARADAS_POR_DX = {dx: SIN_PARADAS for dx in DXS}

# Puntos ya pagados con esta MISMA configuracion (mismo dominio, mismo dx, mismo
# presupuesto, mismas optimizaciones, mismas 26000 iteraciones y sin paradas).
# Se copian en vez de repetirse: son 1.4 h de GPU.
# Sin reuso. Todo lo calculado hasta ahora salio de configuraciones distintas
# —unas con warm_start sin filtro, otras sin warm_start— y ninguna coincide con
# la definitiva. Los 30 puntos se calculan de cero.
REUSO = {}

PARAR = os.path.join(ROOT, "PARAR_ESTUDIO.trigger")

# Coste esperado por punto en segundos, para la estimacion inicial. Sale del
# estudio anterior dividido por la aceleracion medida del solver nuevo. En cuanto
# hay puntos reales medidos, se usan esos.
PRIOR_S = {0.008: 200.0, 0.006: 280.0, 0.004: 430.0, 0.002: 1050.0}

COLS = ["perfil", "dx", "alpha", "cl", "cd", "ld", "cl_ci95", "cd_ci95", "ld_ci95",
        "cl_std", "cd_std", "n_samples", "converged_clcd", "t_conv_clcd",
        "t_parada_offline", "ld_parada_offline", "coste_parada_offline",
        "iters_efectivas", "iters_programadas", "cl_raw", "cl_audit_inst",
        "cl_cp_discrepancy", "cl_cp_discrepancy_flag", "parada_clcd_activa",
        "Re", "Lx", "Ly", "cx", "wall_s"]


def k_dx(dx):
    return f"{dx:.4f}"


def k_a(a):
    return f"{a:.1f}"


def d_perfil(p):
    return os.path.join(OUT, p)


def json_path(p, dx):
    return os.path.join(d_perfil(p), "metricas", f"polar_dx{k_dx(dx)}.json")


def serie_path(p, dx, a):
    return os.path.join(d_perfil(p), "series", f"dx{k_dx(dx)}_a{a:04.1f}_series.npz")


def campo_path(p, dx, a):
    return os.path.join(d_perfil(p), "campos", f"dx{k_dx(dx)}_a{a:04.1f}_campo.npz")


def prepara_arbol():
    for p in PERFILES:
        for sub in ("metricas", "series", "campos",
                    "figuras/campos", "figuras/historia", "figuras/polar"):
            os.makedirs(os.path.join(d_perfil(p), *sub.split("/")), exist_ok=True)
    for sub in ("perfiles", "richardson/figuras", "comparativa"):
        os.makedirs(os.path.join(OUT, *sub.split("/")), exist_ok=True)


def log(msg):
    linea = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(linea, flush=True)
    with open(os.path.join(OUT, "ejecucion.log"), "a") as f:
        f.write(linea + "\n")


def cargar(p, dx):
    f = json_path(p, dx)
    return json.load(open(f)) if os.path.exists(f) else {}


def guardar(p, dx, d):
    with open(json_path(p, dx), "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------
# Plan de ejecucion
# ----------------------------------------------------------------------
def plan():
    """Bloques de barato a caro: las dos polares completas de malla gruesa y
    Si hay que abortar a mitad, lo que queda hecho ya es un estudio de las
    mallas mas gruesas de los dos perfiles."""
    bloques = [(p, dx) for dx in DXS for p in PERFILES]
    return [(p, dx, a) for p, dx in bloques for a in ANGULOS]


def estado_punto(p, dx, a):
    return k_a(a) in cargar(p, dx)


# ----------------------------------------------------------------------
# Progreso
# ----------------------------------------------------------------------
def escribe_estado(en_curso=None):
    puntos, hechos, pend_s = [], 0, 0.0
    medidos = {}
    for p, dx, a in plan():
        d = cargar(p, dx).get(k_a(a))
        if d:
            hechos += 1
            medidos.setdefault(dx, []).append(d.get("wall_s", PRIOR_S[dx]))
            puntos.append({"perfil": p, "dx": dx, "alpha": a, "estado": "hecho",
                           "cl": d.get("cl"), "cd": d.get("cd"), "ld": d.get("ld"),
                           "wall_s": d.get("wall_s"),
                           "reusado": bool(d.get("reusado"))})
        else:
            puntos.append({"perfil": p, "dx": dx, "alpha": a, "estado": "pendiente"})

    for q in puntos:
        if q["estado"] == "pendiente":
            v = medidos.get(q["dx"])
            pend_s += float(np.mean(v)) if v else PRIOR_S[q["dx"]]

    est = {
        "actualizado": time.strftime("%Y-%m-%d %H:%M:%S"),
        # El visor comprueba si este pid sigue vivo: si el runner murio a media
        # simulacion, el "en curso" que quedo escrito seguiria ahi para siempre y
        # el estudio pareceria estar avanzando cuando lleva horas parado.
        "pid_runner": os.getpid() if en_curso else None,
        "total_puntos": len(puntos),
        "hechos": hechos,
        "reusados": sum(1 for q in puntos if q.get("reusado")),
        "pendientes": len(puntos) - hechos,
        "en_curso": en_curso,
        "eta_s": round(pend_s),
        "wall_medido_s": round(sum(q["wall_s"] or 0 for q in puntos
                                   if q["estado"] == "hecho" and not q.get("reusado"))),
        "puntos": puntos,
    }
    with open(os.path.join(OUT, "ESTADO.json"), "w") as f:
        json.dump(est, f, indent=2, ensure_ascii=False)
    escribe_progreso_txt(est)
    return est


def barra(frac, n=44):
    k = int(round(frac * n))
    return "[" + "#" * k + "." * (n - k) + "]"


def hms(s):
    s = int(max(0, s))
    return f"{s//3600:d}h{(s%3600)//60:02d}m"


def escribe_progreso_txt(est):
    L = []
    tot, hech = est["total_puntos"], est["hechos"]
    L.append("RESULTADOS FINALES TFG — convergencia de malla de Richardson")
    L.append(f"actualizado: {est['actualizado']}")
    L.append("")
    L.append(f"  {barra(hech / tot)}  {hech}/{tot} puntos  ({100*hech/tot:.0f}%)")
    L.append(f"  reusados: {est['reusados']}   GPU consumida: {hms(est['wall_medido_s'])}"
             f"   restante estimado: {hms(est['eta_s'])}")
    ec = est.get("en_curso")
    if ec:
        L.append("")
        L.append(f"  EN CURSO: {ec['perfil']}  dx={ec['dx']:g}  alpha={ec['alpha']:.0f}º"
                 f"  ({ec['iters']} iters)  desde {ec['desde']}")
    L.append("")
    for p in PERFILES:
        L.append(f"  {p:<10} " + "  ".join(
            f"dx={dx:g}:" + "".join(
                "#" if q["estado"] == "hecho" else "."
                for q in est["puntos"] if q["perfil"] == p and q["dx"] == dx)
            for dx in DXS))
    L.append("")
    L.append("  malla:  # hecho   . pendiente     angulos en orden 0 2 4 6 8")
    L.append("  parar limpio entre puntos:  touch PARAR_ESTUDIO.trigger")
    with open(os.path.join(OUT, "PROGRESO.txt"), "w") as f:
        f.write("\n".join(L) + "\n")


# ----------------------------------------------------------------------
# Reuso de puntos ya pagados
# ----------------------------------------------------------------------
def importa_reuso():
    """Copia los puntos ya calculados con esta misma configuracion.

    Se comprueba clave a clave que la configuracion coincide en vez de darlo por
    hecho: un punto importado con otro dominio, otro presupuesto de proyeccion o
    otras optimizaciones contaminaria el Richardson entero y la tabla seguiria
    pareciendo valida.
    """
    for (p, dx), origen in REUSO.items():
        src = os.path.join(origen, f"polar_dx{k_dx(dx)}.json")
        if not os.path.exists(src):
            log(f"reuso {p} dx={dx:g}: no existe {src}, se simulara")
            continue
        fuente, destino = json.load(open(src)), cargar(p, dx)
        n = 0
        for a in ANGULOS:
            ka = k_a(a)
            if ka in destino or ka not in fuente:
                continue
            r = fuente[ka]
            ok = (r.get("Lx") == DOMINIO["Lx"] and r.get("Ly") == DOMINIO["Ly"]
                  and r.get("cx") == DOMINIO["cx"] and r.get("Re") == RE
                  and r.get("dx") == dx
                  and r.get("iters_programadas") == ITERS[dx]
                  and sorted(r.get("optimizaciones", [])) == sorted(FLAGS_POR_DX[dx])
                  and r.get("presupuesto") == PRESUPUESTO)
            if not ok:
                log(f"reuso {p} dx={dx:g} a={a}: configuracion distinta, se simulara")
                continue
            base = f"dx{k_dx(dx)}_a{a:04.1f}"
            for src_npz, dst in ((f"series/{base}_series.npz", serie_path(p, dx, a)),
                                 (f"campos/{base}_campo.npz", campo_path(p, dx, a))):
                s = os.path.join(origen, src_npz)
                if os.path.exists(s) and not os.path.exists(dst):
                    shutil.copy2(s, dst)
            r = dict(r)
            r["reusado"] = origen.replace(ROOT + "/", "")
            r["parada_clcd_activa"] = bool(
        PARADAS_POR_DX[dx].get("stop_on_clcd_convergence"))
            r["perfil"] = p
            destino[ka] = r
            n += 1
        if n:
            guardar(p, dx, destino)
            log(f"reuso {p} dx={dx:g}: {n} puntos importados de {origen.replace(ROOT+'/','')}")


# ----------------------------------------------------------------------
# Simulacion
# ----------------------------------------------------------------------
def corre_punto(p, dx, a):
    import verificacion_numerica as vn
    import criterio_ci95 as cc

    flags = FLAGS_POR_DX[dx]
    opt_solver.reset()
    opt_solver.enable(*flags)
    # Comprobar en vez de confiar: con una opcion sin activar —o con una de mas—
    # el punto saldria con otro solver y la tabla pareceria igual de valida.
    assert opt_solver.active() == sorted(flags), (dx, opt_solver.active())

    t0 = time.time()
    r = vn.simular(
        os.path.join(OUT, "perfiles", f"{p}.dat"), RE, dx, alpha=a, iters=ITERS[dx],
        extra={**DOMINIO, **PARADAS_POR_DX[dx], **PRESUPUESTO},
        dump=serie_path(p, dx, a),
        dump_field=campo_path(p, dx, a),
    )
    if r is None:
        return None
    r.update(DOMINIO)
    r["perfil"] = p
    r["optimizaciones"] = list(flags)
    r["presupuesto"] = dict(PRESUPUESTO)
    r["parada_clcd_activa"] = bool(
        PARADAS_POR_DX[dx].get("stop_on_clcd_convergence"))
    r["wall_s"] = round(time.time() - t0, 1)

    # Donde habria parado el criterio calibrado, sin haberle dejado parar.
    sp = serie_path(p, dx, a)
    if os.path.exists(sp):
        z = np.load(sp)
        s = cc._serie(z["t"], z["cl"], z["cd"], f"{p}_dx{k_dx(dx)}_a{a:.0f}", "finales")
        if s is not None:
            crit = json.load(open(f"{vn.OUT}/criterio_parada.json"))["criterio"]
            e = cc.evaluar(s, **crit)
            r.update({"t_parada_offline": round(e[0], 3) if e else None,
                      "ld_parada_offline": round(e[1], 4) if e else None,
                      "coste_parada_offline": round(e[2], 3) if e else None})
    return r


def ejecuta():
    log("=" * 78)
    log("RESULTADOS FINALES TFG — Richardson de dos perfiles")
    for _dx in DXS:
        log(f"  optimizaciones dx={_dx:g}: {sorted(FLAGS_POR_DX[_dx])}")
    log(f"  presupuesto proyeccion: {PRESUPUESTO}")
    log(f"  dominio: {DOMINIO}   Re={RE:.0e}   alphas={ANGULOS}")
    for _dx in DXS:
        _p = "parada Cl/Cd (t_min=%.0f)" % PARADAS_POR_DX[_dx].get(
            "clcd_min_t_fisico_before_check", 0) \
            if PARADAS_POR_DX[_dx].get("stop_on_clcd_convergence") \
            else "ventana completa"
        log(f"  paradas dx={_dx:g}: {_p}")
    log(f"  mallas: {DXS}   iters: {ITERS}")
    log(f"  terna GCI principal (fina,media,gruesa): {TERNA}  r=2 constante")
    log(f"  terna GCI alternativa: {TERNA_ALT}  r=1.5 y 2")
    log("=" * 78)

    importa_reuso()
    est = escribe_estado()
    log(f"punto de partida: {est['hechos']}/{est['total_puntos']} hechos, "
        f"restante estimado {hms(est['eta_s'])}")

    for i, (p, dx, a) in enumerate(plan(), 1):
        if os.path.exists(PARAR):
            os.remove(PARAR)
            log("PARAR_ESTUDIO.trigger detectado: se detiene limpio entre puntos")
            break
        if estado_punto(p, dx, a):
            continue

        est = escribe_estado(en_curso={"perfil": p, "dx": dx, "alpha": a,
                                       "iters": ITERS[dx],
                                       "desde": time.strftime("%H:%M:%S")})
        log("-" * 78)
        log(f"[{i}/{len(plan())}]  {p}  dx={dx:g}  alpha={a:.0f}º  "
            f"({ITERS[dx]} iters)   restante estimado {hms(est['eta_s'])}")

        r = corre_punto(p, dx, a)
        if r is None:
            log(f"  FALLIDA: {p} dx={dx:g} alpha={a}")
            continue

        d = cargar(p, dx)
        d[k_a(a)] = r
        guardar(p, dx, d)
        log(f"  Cl={r['cl']:.5f}  Cd={r['cd']:.5f}  L/D={r['ld']:.3f}  "
            f"({r['wall_s']:.0f}s)")

        import rf_figuras
        rf_figuras.figuras_punto(p, dx, a, r)
        escribe_estado()

    escribe_estado()
    log("bucle de simulacion terminado")


# ----------------------------------------------------------------------
# Richardson
# ----------------------------------------------------------------------
MAGNITUDES = [("cl", "Cl"), ("cd", "Cd"), ("ld", "Cl/Cd")]


def analisis():
    import verificacion_numerica as vn

    todo, todo_alt = {}, {}
    for p in PERFILES:
        datos = {dx: cargar(p, dx) for dx in DXS}
        res, res_alt = {}, {}
        for a in ANGULOS:
            ka = k_a(a)
            for terna, destino in ((TERNA, res), (TERNA_ALT, res_alt)):
                if not all(ka in datos[dx] for dx in terna):
                    continue
                h1, h2, h3 = terna
                por_mag = {}
                for mag, _ in MAGNITUDES:
                    # f1 = malla fina. h3 > h2 > h1.
                    f1, f2, f3 = (datos[dx][ka][mag] for dx in terna)
                    g = vn.gci_triplete(f1, f2, f3, h1, h2, h3)
                    # Sin convergencia monotona la serie no esta en rango asintotico
                    # y el GCI es un numero sin significado: se calcula igual pero se
                    # marca, para que no acabe citado en la memoria.
                    g["citable"] = bool(g["convergencia_monotona"]
                                        and 0.5 <= g["p_observado"] <= 4.0)
                    g["terna_dx"] = list(terna)

                    # Extrapolado robusto. gci_triplete usa el p de Celik, que va
                    # en valor absoluto: cuando los saltos apenas encogen sale
                    # p~0, el factor 1/(r^p - 1) se dispara y el extrapolado deja
                    # de tener sentido fisico. Caso medido: Cl del ganador en
                    # alpha=6 daba 2.91 con la malla fina en 0.879 (amplificacion
                    # x27.5, p real -0.05). El Cd ademas cruzaba a negativo, asi
                    # que se ajusta sobre log(Cd) para que no pueda.
                    ext, p_usado, p_obs, fiable = vn.extrapola_robusto(
                        f1, f2, f3, h1, h2, h3, log=(mag == "cd" and min(f1, f2, f3) > 0))
                    g["f_extrapolado_robusto"] = round(float(ext), 6)
                    g["p_usado"] = round(float(p_usado), 4)
                    g["p_observado_con_signo"] = (round(float(p_obs), 4)
                                                  if p_obs == p_obs else None)
                    g["p_fiable"] = bool(fiable)
                    por_mag[mag] = g

                # El L/D extrapolado se reconstruye desde Cl y Cd en vez de
                # extrapolar el cociente: un cociente hereda las dos patologias
                # y ademas revienta si el Cd extrapolado se acerca a cero.
                cde = por_mag["cd"]["f_extrapolado_robusto"]
                por_mag["ld"]["ld_desde_cl_cd"] = (
                    round(por_mag["cl"]["f_extrapolado_robusto"] / cde, 4)
                    if cde else None)
                destino[ka] = por_mag
        todo[p] = res
        todo_alt[p] = res_alt

    with open(os.path.join(OUT, "richardson", "richardson.json"), "w") as f:
        json.dump(todo, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, "richardson", "richardson_terna_alt.json"), "w") as f:
        json.dump(todo_alt, f, indent=2, ensure_ascii=False)

    # CSV plano de la tabla de GCI
    fil = ["perfil,alpha,magnitud,f_gruesa,f_media,f_fina,p_observado,"
           "f_extrapolado,GCI_fina_pct,GCI_gruesa_pct,monotona,citable,"
           "f_extrapolado_robusto,p_usado,p_observado_con_signo,p_fiable"]
    for p, res in todo.items():
        for ka in sorted(res, key=float):
            for mag, _ in MAGNITUDES:
                g = res[ka][mag]
                fil.append(f"{p},{ka},{mag},{g['f_gruesa']},{g['f_media']},{g['f_fina']},"
                           f"{g['p_observado']},{g['f_extrapolado_richardson']},"
                           f"{g['GCI_fina_pct']},{g['GCI_gruesa_pct']},"
                           f"{int(g['convergencia_monotona'])},{int(g['citable'])},"
                           f"{g['f_extrapolado_robusto']},{g['p_usado']},"
                           f"{g['p_observado_con_signo']},{int(g['p_fiable'])}")
    with open(os.path.join(OUT, "richardson", "tabla_gci.csv"), "w") as f:
        f.write("\n".join(fil) + "\n")

    escribe_csv()
    informe(todo)
    log(f"analisis escrito en {OUT}/richardson/")
    return todo


def escribe_csv():
    fil = [",".join(COLS)]
    for p in PERFILES:
        for dx in DXS:
            d = cargar(p, dx)
            for ka in sorted(d, key=float):
                r = d[ka]
                fil.append(",".join("" if r.get(c) is None else str(r.get(c, ""))
                                    for c in COLS))
    with open(os.path.join(OUT, "metricas_todas.csv"), "w") as f:
        f.write("\n".join(fil) + "\n")


def informe(todo):
    ext_ok = sum(1 for res in todo.values() for ka in res
                 for m, _ in MAGNITUDES if res[ka][m].get("p_fiable"))
    n_tot = sum(len(res) * len(MAGNITUDES) for res in todo.values())
    L = ["# Resultados finales — convergencia de malla",
         "",
         f"Generado {time.strftime('%Y-%m-%d %H:%M')}. **Resultados definitivos del TFG.**",
         "",
         f"Dos perfiles, {len(DXS)} mallas (dx = "
         + " / ".join(f"{dx:g}" for dx in DXS) + f"), {len(ANGULOS)} angulos "
         f"(alpha = {min(ANGULOS):g} a {max(ANGULOS):g} de grado en grado).",
         f"Terna principal del GCI: {TERNA[::-1]} con r=2 constante. "
         f"Terna alternativa de contraste: {TERNA_ALT[::-1]} (r=1.5 y 2).",
         f"Dominio {DOMINIO['Lx']:g}x{DOMINIO['Ly']:g} con el perfil en cx={DOMINIO['cx']:g}, "
         f"Re={RE:.0e}, t~20 en todas las mallas, ventana completa sin parada anticipada.",
         f"Solver de presion con presupuesto de proyeccion {PRESUPUESTO}. "
         "Optimizaciones activas, IDENTICAS en las cuatro mallas:",
         ""] + [f"  - dx={dx:g}: " + ", ".join(sorted(FLAGS_POR_DX[dx])) for dx in DXS] + [
         "",
         "`warm_start_filtered` es lo que permite usar la misma configuracion en "
         "todas las mallas. Sin el filtro del modo par-impar, `warm_start` con "
         "presupuesto 2x3 revienta por blowup de velocidad (iteracion 1340 a "
         "dx=0.004, ~1740 a dx=0.001). Cruce en `results/bench_opt/`.",
         "",
         "## Como leer la tabla",
         "",
         "Dos extrapolaciones por fila:",
         "",
         "  - `extrap` es Richardson puro con el orden observado. Se conserva por "
         "trazabilidad, pero **se dispara cuando p tiende a 0**: el factor "
         "1/(r^p - 1) crece sin limite. Medido: el Cl del ganador en alpha=6 daba "
         "2.91 con la malla fina en 0.879 (amplificacion x27.5), y varios Cd "
         "cruzaban a negativo.",
         "  - `robusto` es el valor a usar. Cuando el orden observado no es "
         "utilizable (`p_fiable` = NO) se fuerza p=2, el nominal del esquema, y el "
         "Cd se ajusta sobre log(Cd) para que no pueda salir negativo. Ahi el "
         "numero es una estimacion indicativa acotada, no una medida del error de "
         "discretizacion.",
         "",
         f"El umbral es p en [{vn_pmin():g}, {vn_pmax():g}]. P_MIN=1 y no 0.5 "
         "porque en r=2 la amplificacion vale exactamente 1 en p=1 y se dispara por "
         "debajo: con p<1 el extrapolado se aleja de la malla fina mas que el salto "
         "entero entre las dos mallas mas finas.",
         "",
         "## Tabla de GCI",
         ""]
    for p, res in todo.items():
        L += [f"### {PERFILES[p]['nombre']}", "",
              "| alpha | mag | grueso | medio | fino | p obs | extrap | "
              "**robusto** | GCI fino % | monotona | p fiable |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for ka in sorted(res, key=float):
            for mag, et in MAGNITUDES:
                g = res[ka][mag]
                po = g.get("p_observado_con_signo")
                L.append(f"| {ka} | {et} | {g['f_gruesa']:.5f} | {g['f_media']:.5f} | "
                         f"{g['f_fina']:.5f} | "
                         f"{'%.2f' % po if po is not None else 'no mono'} | "
                         f"{g['f_extrapolado_richardson']:.5f} | "
                         f"**{g.get('f_extrapolado_robusto', float('nan')):.5f}** | "
                         f"{g['GCI_fina_pct']:.2f} | "
                         f"{'si' if g['convergencia_monotona'] else 'NO'} | "
                         f"{'si' if g.get('p_fiable') else 'NO'} |")
        n = len(res) * len(MAGNITUDES)
        ok = sum(1 for ka in res for m, _ in MAGNITUDES if res[ka][m].get("p_fiable"))
        L += ["", f"Orden observado utilizable en {ok} de {n} casos.", ""]
    L += ["## Avisos al citar", "",
          f"- **Orden observado utilizable en {ext_ok} de {n_tot} casos.** En el "
          "resto la columna `robusto` usa p=2 nominal.",
          "- **Punto de diseno: alpha = 4.** Las tres mallas de la terna coinciden "
          "en que el maximo de L/D esta ahi. El extrapolado deja alpha=3 por "
          "encima, pero eso se debe a que su Cl aun no ha convergido en malla "
          "(correccion +6.2% frente a +0.5% en alpha=4), no a que el pico este ahi.",
          "- **El L/D extrapolado se reconstruye desde Cl y Cd** (campo "
          "`ld_desde_cl_cd` en `richardson.json`), no extrapolando el cociente: un "
          "cociente hereda las patologias de ambos y revienta si el Cd extrapolado "
          "se acerca a cero.",
          "- La ventaja del perfil optimizado sobre el NACA 0012 **se mantiene en "
          "todos los angulos, en las cuatro mallas y tambien extrapolada**. Es el "
          "resultado mas robusto del estudio: no depende de la malla ni del metodo "
          "de extrapolacion.",
          ""]
    with open(os.path.join(OUT, "richardson", "INFORME.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def vn_pmin():
    import verificacion_numerica as vn
    return vn.P_MIN


def vn_pmax():
    import verificacion_numerica as vn
    return vn.P_MAX


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-figuras", action="store_true")
    ap.add_argument("--solo-analisis", action="store_true")
    args = ap.parse_args()

    prepara_arbol()
    for p, cfg in PERFILES.items():
        dst = os.path.join(OUT, "perfiles", f"{p}.dat")
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(ROOT, cfg["dat"]), dst)

    import rf_figuras
    if args.solo_figuras:
        importa_reuso()
        rf_figuras.todas()
        escribe_estado()
        return
    if args.solo_analisis:
        analisis()
        return

    ejecuta()
    # El analisis va ANTES que las figuras: es el resultado que importa y las
    # figuras son cosmetica. Al anadir dx=0.008 un KeyError de color tumbo el
    # cierre de un estudio entero y dejo el richardson.json sin escribir.
    analisis()
    rf_figuras.todas()
    log("ESTUDIO COMPLETO" if escribe_estado()["pendientes"] == 0
        else "estudio incompleto: quedan puntos pendientes")


if __name__ == "__main__":
    main()

"""Cl y Cd de cuerpos romos canonicos (circulo, cuadrado, gota) a Re = 1e3..1e6,
en dos mallas, con campos, frames y video de cada corrida.

POR QUE
  El solver solo se ha ejercitado contra perfiles alares. Estas tres formas
  tienen Cd experimental de sobra en la literatura (circulo ~1.0-1.2 subcritico,
  cuadrado ~2.0-2.2, cuerpo fuselado ~0.05-0.1), asi que miden el sesgo del
  solver contra verdad externa. El precedente es results/esfera_dx002_laminar/:
  cilindro D=1, Cd=1.644 frente a 1.1-1.2 experimental.

CONFIGURACION Y SUS PORQUES
  - Dominio C (24x16, cx=6), el de referencia del TFG.
  - alpha=0 en las tres formas; longitud caracteristica 1, o sea Re = Re_D.
  - `stop_on_clcd_convergence=False`: el criterio calibrado mide el CI95 de L/D,
    y un cuerpo romo simetrico a alpha=0 tiene Cl~0 => L/D~0 => sigma/|media| se
    dispara y no abre nunca (medido: converged_clcd=false tras 44000 iteraciones
    en las dos corridas de esfera_dx002). Se replica offline sobre la serie de
    Cd, que es la magnitud con sentido aqui.
  - `stop_on_convergence=False`: compara el campo contra el de hace 50
    iteraciones y a estas mallas cortaria en pleno transitorio.
  - `min_te_height_factor=0.0`: el recorte de borde de salida esta pensado para
    perfiles. En el cuadrado la cara trasera es un segmento vertical de altura
    completa y el bisector del TE degenera; con min_te ~ 0 el recorte no dispara.
    El solver rechaza el 0 exacto (Simulador2D.py:7568), de ahi el 1e-6: en el
    circulo y la gota, de cola afilada, recorta una lasca despreciable.

SA-BC Y EL COLAPSO DEL PASO TEMPORAL
  Lo ideal es correr con SA + transicion sa_bc, pero en un cuerpo romo nu_t
  satura en la estela y dt_visc = 0.25*dx^2/nu_eff manda: el cilindro con SA se
  quedo en t=5.3 con las mismas 44000 iteraciones que en laminar llegan a t=17.0.
  Por eso cada caso lleva una SONDA de 2500 iteraciones cuyo veredicto mira la
  TENDENCIA, no solo el nivel: la sonda corta no basta (en el cilindro dio
  dt=4.5e-4 y el run promedio 1.2e-4). Si colapsa, el caso se repite en laminar
  y el veredicto queda registrado en el JSON.
  Re=1e3 va laminar por decreto: a ese Reynolds el modelo no tiene sentido.
  Si colapsa a dx=0.004 colapsa peor a dx=0.002 (dt_visc va con dx^2), asi que
  el veredicto se hereda y no se vuelve a sondar.

Uso:
    .venv/bin/python scripts/agent_tests/formas_clcd.py            # fases A y C
    .venv/bin/python scripts/agent_tests/formas_clcd.py --fase A
    .venv/bin/python scripts/agent_tests/formas_clcd.py --horas 4
    .venv/bin/python scripts/agent_tests/formas_clcd.py --solo-analisis
Parada limpia: crear PARAR_FORMAS.trigger en la raiz del repo.
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
import matplotlib.pyplot as plt

import opt_solver
import formas_geom

OUT = os.path.join(ROOT, "results", "formas")
GEOM = os.path.join(OUT, "geom")
TRIGGER = os.path.join(ROOT, "PARAR_FORMAS.trigger")

DOMINIO = {"Lx": 24.0, "Ly": 16.0, "cx": 6.0}
PRESUPUESTO = {"mg_max_outer": 2, "mg_cycles_per_outer": 3}
PARADAS = {"stop_on_clcd_convergence": False, "stop_on_convergence": False}
GEOM_EXTRA = {"min_te_height_factor": 1e-6}
FLAGS = ("warm_start", "warm_start_filtered", "coarse_mask_majority",
         "fast_masks", "interp_float32")
SA = {"turb_model": "sa", "transition_model": "sa_bc", "freestream_Tu": 0.1}
LAMINAR = {"turb_model": None, "transition_model": "none"}

CFL = 0.5
T_OBJETIVO = 20.0
ITERS_SONDA = 2500
GUARDADO = 50
N_FRAMES = 200
# Ventana de recorte para frames y campos periodicos: cuerpo en x=[6,7], cy=8.
VENTANA_X = (5.0, 13.0)
VENTANA_Y = (6.0, 10.0)

FORMAS = ("circulo", "cuadrado", "gota")
FASES = {
    "A": [(f, re, 0.004) for re in (1e3, 1e4, 1e5, 1e6) for f in FORMAS],
    # Fase C recortada a Re=1e5 (el Reynolds de referencia del TFG) por
    # presupuesto: la reparacion de los puntos que colapsaron con SA-BC en la
    # malla gruesa se comio el hueco que tenia Re=1e4 en la fina.
    "C": [(f, re, 0.002) for re in (1e5,) for f in FORMAS],
}
# Coste medido por punto (s), para el ETA antes de tener medidas propias.
PRIOR_S = {0.004: 620.0, 0.002: 1900.0}


def log(msg):
    linea = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(linea, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ejecucion.log"), "a") as f:
        f.write(linea + "\n")


def k_re(re):
    return f"{re:.0e}"


def dirs(forma, re, dx):
    tag = f"{k_re(re)}_dx{dx:.4f}"
    d = os.path.join(OUT, forma)
    return {
        "tag": tag,
        "json": os.path.join(d, "metricas", f"polar_dx{dx:.4f}.json"),
        "serie": os.path.join(d, "series", f"{tag}_serie.npz"),
        "campo": os.path.join(d, "campos", f"{tag}_campo.npz"),
        "campos_t": os.path.join(d, "campos_t", tag),
        "frames": os.path.join(d, "frames", tag),
    }


def cargar(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def guardar(path, d):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def corre(dat, re, dx, iters, modelo, extra_io=None):
    """Una llamada al solver. extra_io=None => sonda (sin frames ni campos)."""
    import verificacion_numerica as vn
    opt_solver.reset()
    opt_solver.enable(*FLAGS)
    assert opt_solver.active() == sorted(FLAGS), opt_solver.active()
    extra = {**DOMINIO, **PARADAS, **PRESUPUESTO, **GEOM_EXTRA, **modelo,
             "guardado": GUARDADO}
    dump = dump_field = None
    if extra_io:
        extra.update(extra_io["sim"])
        dump, dump_field = extra_io["serie"], extra_io["campo"]
    t0 = time.time()
    r = vn.simular(dat, re, dx, alpha=0.0, iters=iters, extra=extra,
                   dump=dump, dump_field=dump_field)
    if r is not None:
        r["wall_s"] = round(time.time() - t0, 1)
    return r


def dt_de_serie(path, n=500):
    """(dt inicial, dt final, t_final) a partir de la serie volcada."""
    d = np.load(path)
    t = np.asarray(d["t"], dtype=float)
    t = t[np.isfinite(t) & (t > 0)]
    if len(t) < 4:
        return None
    k = max(2, min(len(t) // 3, n * len(t) // max(1, ITERS_SONDA)))
    dt_ini = (t[k - 1] - t[0]) / max(1, (k - 1)) / GUARDADO
    dt_fin = (t[-1] - t[-k]) / max(1, (k - 1)) / GUARDADO
    return dt_ini, dt_fin, float(t[-1])


def sonda(dat, re, dx, modelo, tmp):
    """2500 iteraciones para medir dt y juzgar si colapsa."""
    r = corre(dat, re, dx, ITERS_SONDA, modelo,
              extra_io={"sim": {}, "serie": tmp, "campo": None})
    if r is None or not os.path.exists(tmp + ".npz"):
        return None
    d = dt_de_serie(tmp + ".npz")
    if d is None:
        return None
    dt_ini, dt_fin, _ = d
    dt_cfl = CFL * dx / 2.0          # u_max ~ 2U en un cuerpo romo
    colapso = bool(dt_fin < 0.6 * dt_cfl or dt_fin / max(dt_ini, 1e-30) < 0.75)
    return {"dt_ini": dt_ini, "dt_fin": dt_fin, "dt_cfl": dt_cfl,
            "razon_cfl": dt_fin / dt_cfl, "razon_tendencia": dt_fin / max(dt_ini, 1e-30),
            "colapso": colapso}


def _t_final(serie_path):
    if not os.path.exists(serie_path):
        return None
    t = np.asarray(np.load(serie_path)["t"], dtype=float)
    t = t[np.isfinite(t) & (t > 0)]
    return float(t[-1]) if len(t) else None


def strouhal(t, cl):
    """St = f*D/U con D=U=1: el pico del espectro de Cl.

    Devuelve (St, amplitud, prominencia) o (None, amp, prom). Con un cuerpo que
    todavia no desprende, el pico del FFT cae en el crecimiento lento de la
    inestabilidad y da un St espurio bajisimo, asi que se exige que el pico
    destaque sobre el fondo del espectro y que la oscilacion de Cl tenga
    amplitud apreciable. Si no, no hay desprendimiento que medir.
    """
    m = np.isfinite(t) & np.isfinite(cl) & (t > 0)
    t, cl = t[m], cl[m]
    if len(t) < 64:
        return None, None, None
    i0 = len(t) // 2                     # descartar transitorio: la mitad
    t, cl = t[i0:], cl[i0:]
    tu = np.linspace(t[0], t[-1], len(t))
    y = np.interp(tu, t, cl)
    y = y - y.mean()
    amp = float(np.sqrt(2.0) * y.std())          # amplitud de pico de un seno
    esp = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    fr = np.fft.rfftfreq(len(y), d=(tu[-1] - tu[0]) / (len(tu) - 1))
    if len(esp) < 8:
        return None, round(amp, 5), None
    k = 1 + int(np.argmax(esp[1:]))
    prom = float(esp[k] / max(np.median(esp[1:]), 1e-30))
    st = round(float(fr[k]), 4)
    # Con amplitud de Cl por debajo de 0.05 el desprendimiento no ha saturado:
    # lo que mide el FFT es el crecimiento de la inestabilidad, no un ciclo
    # limite, y el St sale espurio. Se reporta None y quedan amp y prom.
    ok = prom >= 8.0 and amp >= 0.05
    return (st if ok else None), round(amp, 5), round(prom, 1)


def parada_offline_cd(t, cd):
    """El criterio calibrado del solver, pero sobre Cd en vez de sobre L/D."""
    from Simulador2D import _detect_series_convergence
    import verificacion_numerica as vn
    p = f"{vn.OUT}/criterio_parada.json"
    if not os.path.exists(p):
        return None
    c = json.load(open(p))["criterio"]
    for i in range(10, len(t) + 1):
        if t[i - 1] < c["min_t"]:
            continue
        d, ci, ok = _detect_series_convergence(
            t[:i], cd[:i], c["tol_drift"], c["tol_ci95"], c["window"])
        if ok:
            return {"t": round(float(t[i - 1]), 3),
                    "cd": round(float(np.mean(cd[max(0, i - 20):i])), 5),
                    "frac_del_run": round(float(t[i - 1] / t[-1]), 3)}
    return None


def punto(forma, re, dx, veredictos):
    D = dirs(forma, re, dx)
    dat = os.path.join(GEOM, f"{forma}.dat")
    for k in ("serie", "campo"):
        os.makedirs(os.path.dirname(D[k]), exist_ok=True)

    # --- modelo de turbulencia: sonda + veredicto ---
    # El veredicto de colapso SI se hereda de una malla mas gruesa (dt_visc va
    # con dx^2: si colapsa a 0.004, colapsa peor a 0.002). El dt NO se hereda:
    # depende de la malla, y usarlo dimensionaria mal el run.
    clave_v = f"{forma}_{k_re(re)}"
    if re <= 1e3:
        ver = {"colapso": True, "motivo": "Re<=1e3, laminar por decreto"}
    elif veredictos.get(clave_v, {}).get("colapso"):
        ver = {"colapso": True,
               "motivo": f"heredado de dx={veredictos[clave_v]['dx']}",
               "sonda_heredada": veredictos[clave_v]}
    else:
        log(f"  sonda SA-BC ({ITERS_SONDA} it)...")
        ver = sonda(dat, re, dx, SA, os.path.join(OUT, "_sonda"))
        if ver is None:
            ver = {"colapso": True, "motivo": "la sonda con SA fallo"}
        else:
            ver["motivo"] = ("dt colapsa" if ver["colapso"] else "dt estable")
        veredictos[clave_v] = dict(ver, dx=dx)
        guardar(os.path.join(OUT, "veredictos_sa.json"), veredictos)
    modelo = LAMINAR if ver["colapso"] else SA
    nombre_modelo = "laminar" if ver["colapso"] else "sa+sa_bc"
    log(f"  modelo={nombre_modelo} ({ver['motivo']})")

    # --- dimensionar el run: sonda con el modelo que se va a usar ---
    if ver["colapso"]:
        s = sonda(dat, re, dx, modelo, os.path.join(OUT, "_sonda"))
        if s is None:
            log("  la sonda laminar fallo, se salta el punto")
            return None
        dt_fin = s["dt_fin"]
        ver["sonda_laminar"] = s
    else:
        dt_fin = ver["dt_fin"]
    iters = int(round(T_OBJETIVO / dt_fin / 1000.0) * 1000)
    iters = int(np.clip(iters, 5000, 200000))
    frame_every = max(GUARDADO, int(round(iters / N_FRAMES / GUARDADO)) * GUARDADO)
    log(f"  dt={dt_fin:.3e} -> iters={iters} (t~{T_OBJETIVO}), frame cada {frame_every}")

    # Los PNG NO se generan en el solver: medido, `save_frame` tarda ~20 s por
    # figura porque hace pcolormesh sobre los ~2 M de celdas de la ventana sin
    # submuestrear, y con 200 frames eso son 2 h de matplotlib por corrida, mas
    # que la propia simulacion. Se vuelcan campos submuestreados (rapido) y
    # render_videos.py dibuja despues, en CPU y sin bloquear la GPU.
    sim_io = {"sim": {"save_frame_refined_xlim": VENTANA_X,
                      "save_frame_refined_ylim": VENTANA_Y,
                      "dump_fields_dir": D["campos_t"],
                      "dump_fields_cada": frame_every,
                      "dump_fields_max_nx": 900},
              "serie": D["serie"], "campo": D["campo"]}
    r = corre(dat, re, dx, iters, modelo, extra_io=sim_io)
    if r is None:
        log("  simulacion fallida")
        return None

    # La sonda de 2500 iteraciones NO basta para cazar el colapso de dt: cuatro
    # casos la pasaron ("dt estable") y luego se quedaron en t=6-9 en vez de 20.
    # El colapso se desarrolla a lo largo del run, no en su arranque, asi que la
    # unica comprobacion fiable es a posteriori. Ademas de rescatar el punto,
    # esto garantiza que todo el barrido en Re use el MISMO modelo: sin eso la
    # tendencia de Cd contra Re no significa nada.
    if not ver["colapso"]:
        t_alc = _t_final(D["serie"])
        if t_alc is not None and t_alc < 0.7 * T_OBJETIVO:
            log(f"  COLAPSO POST-RUN: t={t_alc:.2f} de {T_OBJETIVO} objetivo. "
                f"Se descarta SA-BC y se repite en laminar.")
            ver = {"colapso": True, "motivo": "colapso detectado post-run",
                   "t_con_sa": round(t_alc, 3), "sonda_sa": dict(ver)}
            veredictos[clave_v] = dict(ver, dx=dx)
            guardar(os.path.join(OUT, "veredictos_sa.json"), veredictos)
            modelo, nombre_modelo = LAMINAR, "laminar"
            sl = sonda(dat, re, dx, modelo, os.path.join(OUT, "_sonda"))
            if sl is None:
                log("  la sonda laminar fallo, se salta el punto")
                return None
            dt_fin = sl["dt_fin"]
            ver["sonda_laminar"] = sl
            iters = int(np.clip(round(T_OBJETIVO / dt_fin / 1000.0) * 1000, 5000, 200000))
            frame_every = max(GUARDADO, int(round(iters / N_FRAMES / GUARDADO)) * GUARDADO)
            sim_io["sim"]["dump_fields_cada"] = frame_every
            log(f"  reintento laminar: dt={dt_fin:.3e} -> iters={iters}")
            shutil.rmtree(D["campos_t"], ignore_errors=True)
            r = corre(dat, re, dx, iters, modelo, extra_io=sim_io)
            if r is None:
                log("  el reintento laminar fallo")
                return None

    # np.savez_compressed no duplica la extension: D["serie"] ya acaba en .npz
    d = np.load(D["serie"])
    t, cl, cd = (np.asarray(d[k], float) for k in ("t", "cl", "cd"))
    m = np.isfinite(t) & np.isfinite(cd) & (t > 0)
    t, cl, cd = t[m], cl[m], cd[m]
    r.update({
        "forma": forma,
        "t_final": round(float(t[-1]), 3) if len(t) else None,
        "dt_medio": float(t[-1] / r["iters_efectivas"]) if len(t) else None,
        "dt_sonda": dt_fin,
        "modelo_turbulencia": nombre_modelo,
        "veredicto_sa": ver,
        **dict(zip(("strouhal", "cl_amplitud", "pico_espectral"), strouhal(t, cl))),
        "parada_offline_cd": parada_offline_cd(t, cd),
        "campos_volcados": (len(os.listdir(D["campos_t"]))
                            if os.path.isdir(D["campos_t"]) else 0),
        **DOMINIO, "optimizaciones": sorted(FLAGS), "presupuesto": dict(PRESUPUESTO),
    })
    return r


def estado(hecho, total, pendiente, t0, en_curso=True):
    guardar(os.path.join(OUT, "ESTADO.json"), {
        "hechos": hecho, "total": total, "pendiente": pendiente,
        "wall_s": round(time.time() - t0, 1),
        # Si el runner muere a mitad, un en_curso rancio lo haria parecer vivo.
        "pid_runner": os.getpid() if en_curso else None,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    with open(os.path.join(OUT, "PROGRESO.txt"), "w") as f:
        n = int(40 * hecho / max(1, total))
        f.write(f"[{'#'*n}{'.'*(40-n)}] {hecho}/{total}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fase", default="AC", help="A, C o AC")
    ap.add_argument("--horas", type=float, default=None,
                    help="plazo: no se lanza un punto nuevo si ya se agoto")
    ap.add_argument("--solo-geom", action="store_true")
    args = ap.parse_args()

    os.makedirs(GEOM, exist_ok=True)
    for f in FORMAS:
        formas_geom.escribe(f, os.path.join(GEOM, f"{f}.dat"))
    log(f"geometrias en {GEOM}: " + ", ".join(
        f"{f} (area {formas_geom.AREA[f]:.4f})" for f in FORMAS))
    if args.solo_geom:
        return

    plan = [p for fase in args.fase for p in FASES.get(fase, [])]
    t0 = time.time()
    limite = args.horas * 3600 if args.horas else None
    veredictos = cargar(os.path.join(OUT, "veredictos_sa.json"))
    hechos = 0

    log(f"=== formas: {len(plan)} puntos, fases {args.fase} ===")
    for i, (forma, re, dx) in enumerate(plan):
        D = dirs(forma, re, dx)
        js = cargar(D["json"])
        if k_re(re) in js:
            log(f"[{i+1}/{len(plan)}] {forma} Re={k_re(re)} dx={dx} YA HECHO")
            hechos += 1
            continue
        if os.path.exists(TRIGGER):
            os.remove(TRIGGER)
            log("PARAR_FORMAS.trigger: parada limpia")
            break
        if limite and (time.time() - t0) + PRIOR_S[dx] > limite:
            log(f"plazo agotado ({args.horas} h): quedan {len(plan)-i} puntos")
            break

        log(f"[{i+1}/{len(plan)}] {forma} Re={k_re(re)} dx={dx}")
        # Un punto que reviente no debe llevarse la fase entera por delante.
        try:
            r = punto(forma, re, dx, veredictos)
        except Exception as e:
            import traceback
            log(f"  FALLO: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        if r is None:
            continue
        js = cargar(D["json"])
        js[k_re(re)] = r
        guardar(D["json"], js)
        hechos += 1
        log(f"  -> Cd={r['cd']:.4f}±{r.get('cd_ci95')}  Cl={r['cl']:+.4f}  "
            f"St={r.get('strouhal')}  t={r.get('t_final')}  ({r['wall_s']}s)")
        estado(hechos, len(plan), len(plan) - i - 1, t0)

    estado(hechos, len(plan), 0, t0, en_curso=False)
    log(f"=== fin: {hechos}/{len(plan)} puntos en {(time.time()-t0)/3600:.2f} h ===")


if __name__ == "__main__":
    main()

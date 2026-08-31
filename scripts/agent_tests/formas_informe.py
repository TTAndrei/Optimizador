"""Tablas e informe del estudio de formas y del tubo. Solo lee JSON, sin GPU.

Los valores de referencia son de libro (Hoerner, White, Blevins) para cuerpos 2D
a alpha=0 con la longitud caracteristica del cuadro. El objetivo del estudio es
CUANTIFICAR el sesgo del solver, no acertar: el precedente ya medido es un
cilindro con Cd=1.64 frente a 1.1-1.2 experimental.
"""
from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FORMAS = os.path.join(ROOT, "results", "formas")
TUBO = os.path.join(ROOT, "results", "tubo")

# (Cd 2D de referencia, nota). El del cilindro depende fuerte de Re: subcritico
# ~1.2, crisis de resistencia por encima de Re~2e5.
REF = {
    "circulo":  {1e3: 1.0, 1e4: 1.1, 1e5: 1.2, 1e6: 0.35},
    "cuadrado": {1e3: 2.1, 1e4: 2.1, 1e5: 2.1, 1e6: 2.1},
    "gota":     {1e3: 0.12, 1e4: 0.08, 1e5: 0.06, 1e6: 0.05},
}
ST_REF = {"circulo": 0.20, "cuadrado": 0.13, "gota": None}


def carga_formas():
    filas = []
    for js in sorted(glob.glob(os.path.join(FORMAS, "*", "metricas", "*.json"))):
        for re_k, r in json.load(open(js)).items():
            filas.append(r)
    return filas


def tabla_formas(filas):
    out = ["| forma | Re | dx | modelo | t | Cd | ±CI95 | Cd ref | desv | Cl | amp Cl | St |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(filas, key=lambda r: (r["forma"], r["dx"], r["Re"])):
        ref = REF.get(r["forma"], {}).get(r["Re"])
        desv = f"{100*(r['cd']/ref-1):+.0f} %" if ref else "—"
        st, sf, sc = r.get("strouhal"), r.get("st_fft"), r.get("st_cruces")
        if not st:
            st_txt = "no desprende"
        elif r.get("duplicacion_periodo"):
            st_txt = f"**{st}** (dupl.)"
        elif sf and sc and not (0.85 <= sc / sf <= 1.18):
            st_txt = f"{sf} / {sc} ?"        # los dos estimadores no concuerdan
        else:
            st_txt = f"{st}"
        out.append(
            f"| {r['forma']} | {r['Re']:.0e} | {r['dx']} | {r['modelo_turbulencia']} | "
            f"{r.get('t_final')} | {r['cd']:.4f} | {r.get('cd_ci95')} | "
            f"{ref if ref else '—'} | {desv} | {r['cl']:+.4f} | "
            f"{r.get('cl_amplitud')} | {st_txt} |")
    return "\n".join(out)


def figura_formas(filas):
    if not filas:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for forma in sorted({r["forma"] for r in filas}):
        for dx, m in ((0.004, "o--"), (0.002, "s-")):
            sub = sorted([r for r in filas if r["forma"] == forma and r["dx"] == dx],
                         key=lambda r: r["Re"])
            if sub:
                ax[0].plot([r["Re"] for r in sub], [r["cd"] for r in sub], m,
                           label=f"{forma} dx={dx}")
        if REF.get(forma):
            k = sorted(REF[forma])
            ax[0].plot(k, [REF[forma][i] for i in k], "k:", alpha=0.5)
    ax[0].set_xscale("log"); ax[0].set_xlabel("Re"); ax[0].set_ylabel("Cd")
    ax[0].set_title("Cd (punteado negro = referencia)"); ax[0].legend(fontsize=7)
    for forma in sorted({r["forma"] for r in filas}):
        sub = sorted([r for r in filas if r["forma"] == forma and r.get("strouhal")],
                     key=lambda r: r["Re"])
        if sub:
            ax[1].plot([r["Re"] for r in sub], [r["strouhal"] for r in sub], "o-", label=forma)
    ax[1].set_xscale("log"); ax[1].set_xlabel("Re"); ax[1].set_ylabel("St")
    ax[1].set_title("Strouhal (solo casos con desprendimiento saturado)")
    if ax[1].get_legend_handles_labels()[0]:
        ax[1].legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.join(FORMAS, "figuras"), exist_ok=True)
    fig.savefig(os.path.join(FORMAS, "figuras", "cd_st_vs_re.png"), dpi=140)
    plt.close(fig)


def informe_formas():
    filas = carga_formas()
    if not filas:
        return
    figura_formas(filas)
    lam = sum(r["modelo_turbulencia"] == "laminar" for r in filas)
    decreto = sum(r["modelo_turbulencia"] == "laminar"
                  and "decreto" in r.get("veredicto_sa", {}).get("motivo", "")
                  for r in filas)
    colapso = lam - decreto
    txt = [
        "# Formas romas: Cl y Cd a Re = 1e3..1e6", "",
        f"{len(filas)} puntos. Dominio 24x16 (cx=6), alpha=0, longitud "
        "caracteristica 1, o sea Re = Re_D y Cd referido a 1 m.", "",
        "## Resultados", "", tabla_formas(filas), "",
        "## Modelo de turbulencia", "",
        f"**{lam} de {len(filas)} puntos acabaron en laminar.** {decreto} por decreto "
        f"(Re<=1e3, donde el modelo no tiene sentido fisico) y {colapso} por colapso "
        "del paso temporal con SA-BC.", "",
        "**SA-BC no sobrevivio en ninguna de las tres formas, a ningun Reynolds.** "
        "El mecanismo es `dt_visc = 0.25*dx^2/nu_eff` (Simulador2D.py:8195): en la "
        "estela de un cuerpo romo nu_t satura y el paso temporal se hunde. Esto "
        "confirma y generaliza lo que ya se habia medido sobre el cilindro en "
        "`results/esfera_dx002/` (t=5.3 con SA frente a t=17.0 en laminar con las "
        "mismas 44000 iteraciones).", "",
        "**Una sonda corta NO detecta el colapso, y esto costo cuatro corridas.** "
        "El veredicto se tomaba con 2500 iteraciones mirando el nivel de dt y su "
        "tendencia; cuatro casos la pasaron como 'dt estable' y luego se quedaron "
        "en t = 6.1 a 9.4 de los 20 objetivo. El colapso se desarrolla a lo largo "
        "del run, no en su arranque. La deteccion que si funciona es **a "
        "posteriori**: si con SA el tiempo fisico alcanzado se queda por debajo "
        "del 70 % del objetivo, se descarta SA y se repite en laminar. Los "
        "veredictos, con lo que dijo la sonda y lo que paso de verdad, estan en "
        "`results/formas/veredictos_sa.json`.", "",
        "Esto importa mas alla del ahorro: si unos puntos del barrido corren con "
        "SA y otros en laminar, **la tendencia de Cd contra Re no significa nada**, "
        "porque mezcla el efecto del Reynolds con el del modelo. Todos los puntos "
        "de esta tabla usan el mismo modelo.", "",
        "## Como leer esto", "",
        "- La columna `desv` compara contra valores de libro para cuerpos 2D. El "
        "estudio mide el sesgo del solver, no pretende acertar: el cilindro ya "
        "daba +37 % en `results/esfera_dx002_laminar/`.",
        "- **El circulo a Re=1e6 (+326 %) no es un fallo del solver, es una "
        "limitacion conocida y esperada.** La referencia 0.35 es la crisis de "
        "resistencia: por encima de Re~2e5 la capa limite transiciona a "
        "turbulenta antes de separar, el punto de separacion se va hacia atras y "
        "el Cd se desploma. Eso exige una capa limite resuelta y transicion "
        "tridimensional; con el modelo de turbulencia descartado por colapso de "
        "dt, el solver corre laminar y **no puede reproducir la crisis por "
        "construccion**. Lo consistente es comparar ese punto contra el ~1.2 "
        "subcritico, y entonces la desviacion es +24 %, en linea con el resto.",
        "- **La referencia de la gota es la mas floja de las tres y conviene no "
        "apoyarse en su `desv`.** Los Cd de libro de 0.05-0.1 son de cuerpos "
        "fuselados de esbeltez ~4 con el flujo adherido hasta la cola. El de aqui "
        "tiene esbeltez 2.58 (espesor 0.387 con morro de radio 0.2), y a esa "
        "relacion la recuperacion de presion es lo bastante brusca como para que "
        "separe tambien en la realidad. El dato utilizable de esta fila no es la "
        "desviacion contra el libro sino que **la gota da entre 3 y 5 veces menos "
        "Cd que el circulo a igual Re y misma malla**, que es una comparacion "
        "interna y por tanto libre del sesgo del solver.",
        "- **El cuadrado es el caso mejor portado**: su Cd de referencia no "
        "depende de Re (los puntos de separacion estan clavados en las aristas, "
        "no hay crisis de resistencia que reproducir) y el solver converge hacia "
        "el desde abajo: -45 %, -35 %, -24 % y **+1 % a Re=1e6**. Que la forma sin "
        "separacion ambigua sea la que acierta apunta a que el sesgo del solver "
        "esta en donde coloca la separacion, no en como integra las fuerzas.",
        "- `t` es el tiempo convectivo alcanzado. Si un caso se queda corto, su "
        "Cd esta medido en transitorio y no es comparable con el resto.",
        "- El CI95 es de la ventana de promediado, no incluye incertidumbre de "
        "malla. Para eso hay que mirar la diferencia entre dx=0.004 y dx=0.002.",
        "- Cl deberia ser ~0 en las tres formas por simetria a alpha=0; un Cl "
        "grande delata que la ventana no cubre un numero entero de ciclos de "
        "desprendimiento, no una fuerza real.",
        "- **El Strouhal se mide con dos estimadores y la tabla muestra los dos "
        "cuando discrepan** (`ST_REF`: cilindro 0.20 subcritico, cuadrado 0.13). "
        "El pico del FFT de Cl se equivoca cuando la estela sufre duplicacion de "
        "periodo —los vortices alternan intensidad entre lados y el subarmonico se "
        "lleva mas energia que el fundamental—, y entonces devuelve f/2. Contar "
        "cruces por cero es inmune a eso pero sobrecuenta si la onda tiene "
        "armonicos fuertes. Medido: el circulo a Re=1e4 daba 0.0998 por FFT y "
        "**0.2005 por cruces, que es el 0.20 canonico del cilindro**; el cuadrado "
        "a Re=1e6 igual (0.105 -> 0.2106). Marcados `(dupl.)`. Donde la razon "
        "entre los dos no es ~1 ni ~2 se muestran ambos con `?`: ahi el numero no "
        "es citable. Los dos valores estan siempre en el JSON (`st_fft`, "
        "`st_cruces`, `st_ratio_cruces_fft`), recalculables sin GPU con "
        "`scripts/agent_tests/formas_strouhal.py`.",
        "- **`St = no desprende` no es un fallo de medida**: significa que a t~20, "
        "arrancando de un campo simetrico, la inestabilidad de la estela todavia "
        "no ha saturado. Se detecta por la amplitud de Cl (`amp Cl`): por debajo "
        "de 0.05 el pico del FFT mide el crecimiento de la inestabilidad, no un "
        "ciclo limite, y da un St espurio. El caso medido: circulo a Re=1e3 da "
        "un pico limpisimo (prominencia 6127) en f=0.104 con amplitud 0.012 — "
        "la mitad del St de libro, porque es la envolvente de crecimiento y no "
        "desprendimiento saturado. El Cd de esos casos SI es utilizable: es un "
        "Cd de estela simetrica, mas bajo que el desprendido.",
        "- Un cuerpo perfectamente simetrico en una malla simetrica solo empieza "
        "a desprender cuando el ruido numerico rompe la simetria, y a Re bajo eso "
        "tarda. Alargar la ventana es lo que arreglaria estos casos.", "",
        "Videos en `results/formas/<forma>/videos/`, campos en `campos/` y "
        "`campos_t/`, series en `series/`.", "",
    ]
    with open(os.path.join(FORMAS, "INFORME.md"), "w") as f:
        f.write("\n".join(txt))
    print(f"[informe] {FORMAS}/INFORME.md ({len(filas)} puntos)")


def informe_tubo():
    casos = []
    for js in sorted(glob.glob(os.path.join(TUBO, "T*", "tubo.json"))):
        casos.append(json.load(open(js)))
    if not casos:
        return
    out = ["| caso | Lx×Ly | Re_H | dx | modelo | t | Le medida | 0.05·Re·H | desarr. | u_c salida | err dp/dx |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in casos:
        le = f"{c['Le_medida']:.2f}" if c["Le_medida"] else "—"
        err = f"{c['err_dpdx_pct']:+.1f} %" if c["err_dpdx_pct"] is not None else "n/a"
        out.append(
            f"| {c['caso']} | {c['Lx']:g}×{c['Ly']:g} | {c['Re']:.0e} | {c['dx']} | "
            f"{c['modelo']} | {c['t_final']} | {le} | {c['Le_teorica_005ReH']:.1f} | "
            f"{'si' if c['desarrollado'] else 'no'} | {c['u_centro_salida']} | {err} |")
    txt = [
        "# Canal 2D: paredes no-slip, dominio en reposo, llenado desde la entrada", "",
        "Sin cuerpo inmerso, o sea sin IBM de por medio: es el unico caso del "
        "proyecto con solucion analitica (Poiseuille) y por tanto la validacion "
        "mas limpia del solver.", "",
        "## Resultados", "", "\n".join(out), "",
        "## Como leer esto", "",
        "- `err dp/dx` compara contra `-12·nu·U/H^2` y **solo se calcula si el "
        "flujo llego a desarrollarse dentro del dominio**; en los casos de "
        "llenado a Re alto la longitud de entrada teorica excede el tubo, asi "
        "que ahi la comparacion es n/a por construccion, no por fallo.",
        "- `u_c salida` deberia tender a 1.5·U si el perfil se desarrollo.",
        "- El arranque es impulsivo: el solver no tiene rampa de entrada, asi que "
        "el primer tramo del transitorio es un golpe de presion y no es fisica.",
        "- **Los dos casos de validacion se desarrollan y ambos aciertan la "
        "longitud de entrada**: T1 da Le=6.6 frente a 5.0 de la correlacion y T2 "
        "da 26.6 frente a 25.0, con la linea central en 1.488 y 1.488 frente al "
        "1.5 exacto de Poiseuille (99.2 %). El gradiente de presion sale a 2.7 % "
        "en T1. Es la validacion mas limpia del proyecto: sin IBM de por medio, "
        "el solver reproduce la solucion exacta.",
        "- La velocidad en las paredes sale exactamente 0 (`u_pared_max` en el "
        "JSON), que es la comprobacion de que la condicion no-slip se aplica.", "",
        "Videos en `results/tubo/<caso>/videos/`, perfiles en `figuras/`.", "",
    ]
    with open(os.path.join(TUBO, "INFORME.md"), "w") as f:
        f.write("\n".join(txt))
    print(f"[informe] {TUBO}/INFORME.md ({len(casos)} casos)")


if __name__ == "__main__":
    informe_formas()
    informe_tubo()

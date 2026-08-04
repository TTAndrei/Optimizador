#!/usr/bin/env python3
"""Segunda pasada de acentuacion, apoyada en el diccionario hunspell espanol.

Para cada palabra ASCII que hunspell rechaza, genera los candidatos que resultan
de anadir una tilde o cambiar `n` por `n con virgulilla` y aplica el cambio solo
si **exactamente uno** de esos candidatos es palabra valida. Las palabras que
hunspell ya acepta tal cual no se tocan: pueden ser correctas sin tilde y
decidirlo requiere contexto (lo hace `acentuar.py`).

Requiere `spylls` y un diccionario hunspell `es_ES`:

    pip install spylls
    python3 acentuar2.py memoria.txt
"""

import itertools
import os
import re
import sys

from spylls.hunspell import Dictionary

RUTAS_DIC = [
    "/var/snap/firefox/common/snap-hunspell/es_ES",
    "/usr/share/hunspell/es_ES",
    "/usr/share/myspell/dicts/es_ES",
]

TILDE = {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú"}

# Formas que hunspell acepta sin tilde pero que en esta memoria son otra palabra.
FORZADAS = {
    "adveccion": "advección", "advectiva": "advectiva",
    "elite": "élite", "elites": "élites",
    "acerco": "acercó", "acoto": "acotó", "alcanzo": "alcanzó",
    "aplico": "aplicó", "comparo": "comparó", "comprobo": "comprobó",
    "concluyo": "concluyó", "consumio": "consumió", "corrigio": "corrigió",
    "cuantifico": "cuantificó", "cubrio": "cubrió", "descarto": "descartó",
    "empeoro": "empeoró", "encontro": "encontró", "exonero": "exoneró",
    "identifico": "identificó", "implemento": "implementó",
    "localizo": "localizó", "movio": "movió", "oscilo": "osciló",
    "perdio": "perdió", "permitio": "permitió", "provoco": "provocó",
    "recupero": "recuperó", "resulto": "resultó", "separo": "separó",
    "situo": "situó", "subio": "subió", "selecciono": "seleccionó",
    "retraso": "retrasó", "reactivo": "reactivó", "obligo": "obligó",
    "senalo": "señaló", "anadio": "añadió", "hundia": "hundía",
    "tardo": "tardó", "volvio": "volvió", "salio": "salió",
    "sirvio": "sirvió", "disparo": "disparó", "mostro": "mostró",
    "fracaso": "fracasó", "penso": "pensó", "opero": "operó",
}

# Cambios que hunspell no puede resolver y que aqui son inequivocos.
EXTRA = {
    "fenomenologia": "fenomenología", "metodologica": "metodológica",
    "metodologico": "metodológico", "metodologicas": "metodológicas",
    "metodologicos": "metodológicos", "desviatorica": "desviatórica",
    "anisotropo": "anisótropo", "subsonica": "subsónica",
    "sobrecirculacion": "sobrecirculación", "reimposicion": "reimposición",
    "constatacion": "constatación", "falsacion": "falsación",
    "precomputo": "precómputo", "postsuavizado": "postsuavizado",
    "presuavizado": "presuavizado", "multipunto": "multipunto",
    "fluidodinamicos": "fluidodinámicos", "bidimensionalidad": "bidimensionalidad",
    "tridimensionalidad": "tridimensionalidad",
    "estadisticamente": "estadísticamente", "geometricamente": "geométricamente",
    "numericamente": "numéricamente", "energeticamente": "energéticamente",
    "tipicamente": "típicamente", "practicamente": "prácticamente",
    "automaticamente": "automáticamente", "basicamente": "básicamente",
    "historicamente": "históricamente", "economicamente": "económicamente",
    "sistematicamente": "sistemáticamente", "explicitamente": "explícitamente",
    "cualitativamente": "cualitativamente", "aerodinamicamente": "aerodinámicamente",
    "ambientalmente": "ambientalmente", "cientificamente": "científicamente",
    "clúster": "clúster",
}

PROTEGIDO = re.compile(r"(\$\$.*?\$\$|\$[^$\n]+\$|`[^`\n]*`|^//.*$)", re.MULTILINE)
PALABRA = re.compile(r"\b[A-Za-z]{3,}\b")


def candidatos(w):
    """Variantes con una tilde y/o virgulillas."""
    pos_v = [i for i, c in enumerate(w) if c in TILDE]
    pos_n = [i for i, c in enumerate(w) if c == "n"]
    salida = set()
    for nn in range(len(pos_n) + 1):
        for combo_n in itertools.combinations(pos_n, nn):
            base = list(w)
            for i in combo_n:
                base[i] = "ñ"
            for i in pos_v:
                v = list(base)
                v[i] = TILDE[v[i]]
                salida.add("".join(v))
            if combo_n:
                salida.add("".join(base))
    salida.discard(w)
    return salida


def cargar():
    for r in RUTAS_DIC:
        if os.path.exists(r + ".dic"):
            return Dictionary.from_files(r)
    sys.exit("no se encuentra un diccionario hunspell es_ES")


def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else "memoria.txt"
    d = cargar()
    texto = open(ruta, encoding="utf-8").read()
    cache, cambios = {}, {}

    def resolver(w):
        bajo = w.lower()
        if bajo in EXTRA:
            return EXTRA[bajo]
        if bajo in FORZADAS:
            return FORZADAS[bajo]
        if bajo in cache:
            return cache[bajo]
        r = None
        if not d.lookup(bajo):
            validos = [c for c in candidatos(bajo) if d.lookup(c)]
            if len(validos) == 1:
                r = validos[0]
        cache[bajo] = r
        return r

    def sust(m):
        w = m.group(0)
        r = resolver(w)
        if not r:
            return w
        if w[0].isupper():
            r = r[0].upper() + r[1:]
        cambios[w] = r
        return r

    partes = PROTEGIDO.split(texto)
    for i in range(0, len(partes), 2):
        partes[i] = PALABRA.sub(sust, partes[i])
    open(ruta, "w", encoding="utf-8").write("".join(partes))
    print(f"{ruta}: {len(cambios)} formas corregidas")
    for k in sorted(cambios)[:40]:
        print(f"   {k} -> {cambios[k]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Restaura tildes y enes en `memoria.txt`.

El fuente se escribe en ASCII para no depender de la codificacion del terminal.
Este paso lo convierte a castellano correcto usando `/usr/share/dict/spanish`
para las palabras cuya version sin tilde no es a su vez una palabra valida, mas
un diccionario manual para el resto.

No toca lo que va entre `$...$` (matematicas), entre acentos graves (codigo) ni
las lineas de comentario del fuente.
"""

import re
import sys
import unicodedata

DICC_SISTEMA = "/usr/share/dict/spanish"

# Entradas erroneas o raras del diccionario del sistema que no deben aplicarse.
VETO = {"segun", "solo", "este", "esta", "estas", "estos", "aquel", "ese",
        "esa", "esos", "esas", "continua", "continuo", "critico", "critica",
        "publico", "publica", "practico", "domino", "limite", "limites",
        "termino", "terminos", "calculo", "calculos", "valido", "valida",
        "numero", "numeros", "ordenes", "articulo", "capitulo", "titulo"}

# Palabras que el diccionario del sistema no cubre, o que cubre mal.
MANUAL = {
    "segun": "según", "ano": "año", "anos": "años",
    "senales": "señales", "senalar": "señalar", "senalada": "señalada",
    "senalado": "señalado", "senalados": "señalados", "senaladas": "señaladas",
    "senala": "señala", "senalo": "señaló", "ensenanza": "enseñanza",
    "tecnica": "técnica", "tecnicas": "técnicas", "tecnico": "técnico",
    "tecnicos": "técnicos", "numero": "número", "numeros": "números",
    "calculo": "cálculo", "calculos": "cálculos", "limite": "límite",
    "limites": "límites", "termino": "término", "terminos": "términos",
    "capitulo": "capítulo", "capitulos": "capítulos", "titulo": "título",
    "titulos": "títulos", "articulo": "artículo", "ordenes": "órdenes",
    "critico": "crítico", "critica": "crítica", "criticas": "críticas",
    "practico": "práctico", "practicos": "prácticos",
    "continua": "continua", "cuestion": "cuestión", "cuestiones": "cuestiones",
    "clúster": "clúster", "vease": "véase", "esten": "estén",
    "estandar": "estándar", "estandares": "estándares",
    "linea": "línea", "lineas": "líneas", "lineal": "lineal",
    "energia": "energía", "energias": "energías", "energetica": "energética",
    "energetico": "energético", "energeticos": "energéticos",
    "energeticamente": "energéticamente",
    "aereo": "aéreo", "aereos": "aéreos", "aerea": "aérea",
    "atras": "atrás", "detras": "detrás", "quiza": "quizá",
    "unica": "única", "unicas": "únicas", "unico": "único", "unicos": "únicos",
    "unicamente": "únicamente", "geometria": "geometría",
    "geometrias": "geometrías", "geometrico": "geométrico",
    "geometrica": "geométrica", "geometricos": "geométricos",
    "geometricas": "geométricas", "geometricamente": "geométricamente",
    "ingenieria": "ingeniería", "aerodinamica": "aerodinámica",
    "aerodinamicas": "aerodinámicas", "aerodinamico": "aerodinámico",
    "aerodinamicos": "aerodinámicos", "aerodinamicamente": "aerodinámicamente",
    "numerica": "numérica", "numericas": "numéricas", "numerico": "numérico",
    "numericos": "numéricos", "numericamente": "numéricamente",
    "asintotico": "asintótico", "asintotica": "asintótica",
    "eliptica": "elíptica", "eliptico": "elíptico",
    "parametrica": "paramétrica", "parametrico": "paramétrico",
    "parametro": "parámetro", "parametros": "parámetros",
    "atomica": "atómica", "automatica": "automática",
    "automatico": "automático", "automaticamente": "automáticamente",
    "estatico": "estático", "caracteristica": "característica",
    "caracteristicas": "características", "caracteristico": "característico",
    "electrica": "eléctrica", "electrico": "eléctrico",
    "electricidad": "electricidad", "informatica": "informática",
    "matematica": "matemática", "matematicas": "matemáticas",
    "estadistica": "estadística", "estadisticos": "estadísticos",
    "estadistico": "estadístico", "analitica": "analítica",
    "sintesis": "síntesis", "hipotesis": "hipótesis", "tesis": "tesis",
    "energeticas": "energéticas", "logica": "lógica", "logico": "lógico",
    "canonico": "canónico", "generico": "genérico",
    "academica": "académica", "academico": "académico",
    "docente": "docente", "etica": "ética", "eticas": "éticas",
    "etico": "ético", "eticos": "éticos", "publicacion": "publicación",
    "deberia": "debería", "deberian": "deberían", "habria": "habría",
    "habrian": "habrían", "podria": "podría", "podrian": "podrían",
    "seria": "sería", "serian": "serían", "tendria": "tendría",
    "requeriria": "requeriría", "exigiria": "exigiría",
    "produciria": "produciría", "supondria": "supondría",
    "convertiria": "convertiría", "costaria": "costaría",
    "rendiria": "rendiría", "aportaria": "aportaría",
    "afectaria": "afectaría", "ahorraria": "ahorraría",
    "duplicaria": "duplicaría", "eliminaria": "eliminaría",
    "concentraria": "concentraría", "introduciria": "introduciría",
    "asignaria": "asignaría", "tenderia": "tendería",
    "permitiria": "permitiría", "obligaria": "obligaría",
    "situaria": "situaría", "creceria": "crecería",
    "reduciria": "reduciría", "apuntaria": "apuntaría",
    "desplazaria": "desplazaría", "anularia": "anularía",
    "quedaria": "quedaría", "resultaria": "resultaría",
    "dispararia": "dispararía", "cubriria": "cubriría",
    "escalaria": "escalaría", "propondria": "propondría",
    "mediria": "mediría", "haria": "haría", "iria": "iría",
    "irian": "irían", "recuperaria": "recuperaría",
    "compensaria": "compensaría", "ahorrarian": "ahorrarían",
    "costarian": "costarían", "perderia": "perdería",
    "veria": "vería", "verian": "verían", "dejaria": "dejaría",
    "multiplicaria": "multiplicaría", "exigirian": "exigirían",
    "competiria": "competiría", "orientaria": "orientaría",
    "produciran": "producirán", "seguiria": "seguiría",
    "arrastraria": "arrastraría",
}

# Palabras cuya forma sin tilde tambien es valida: se resuelven por contexto.
CONTEXTO = [
    (r"\bcampana\b", "campaña"),
    (r"\bcampanas\b", "campañas"),
    (r"\bmas\b(?! bien\b)", "más"),
    (r"\ben perdida\b", "en pérdida"),
    (r"\buna perdida\b", "una pérdida"),
    (r"\bla perdida de\b", "la pérdida de"),
    (r"\by perdida\b", "y pérdida"),
    (r"\bser valida\b", "ser válida"),
    (r"\bes valida\b", "es válida"),
    (r"\bsea valida\b", "sea válida"),
    (r"\bde ser válida\b", "de ser válida"),
    (r"\besta\b(?= (?:es|en|la|el|situad|acotad|pendiente|justificad|"
     r"implementad|documentad|bien|por|dentro|fuera|dominad|limitad|"
     r"entrenad|entre|entrenando|calibrad|activad|desactivad))", "está"),
    (r"\bno esta\b", "no está"), (r"\bque esta\b", "que está"),
    (r"\bestan\b", "están"), (r"\baun asi\b", "aun así"),
    (r"\baun cuando\b", "aun cuando"), (r"\by aun\b", "y aún"),
    (r"\bmenos aun\b", "menos aún"),
    (r"\bsolo\b", "solo"),
]


def deacc(w):
    return "".join(c for c in unicodedata.normalize("NFD", w)
                   if unicodedata.category(c) != "Mn").replace("ñ", "n")


def construir_mapa():
    palabras = [l.strip() for l in open(DICC_SISTEMA, encoding="utf-8")
                if l.strip()]
    llanas = {w.lower() for w in palabras if deacc(w.lower()) == w.lower()}
    cand = {}
    for w in palabras:
        wl = w.lower()
        d = deacc(wl)
        if d == wl or d in llanas or d in VETO:
            continue
        if d in cand and cand[d] != wl:
            cand[d] = None                    # ambiguo: no se toca
        else:
            cand.setdefault(d, wl)
    mapa = {k: v for k, v in cand.items() if v}
    mapa.update(MANUAL)
    return mapa


def aplicar_mayusculas(origen, destino):
    if origen[0].isupper():
        return destino[0].upper() + destino[1:]
    return destino


PROTEGIDO = re.compile(r"(\$\$.*?\$\$|\$[^$\n]+\$|`[^`\n]*`|\[[A-Z]+\](?=\s)|^//.*$)",
                       re.MULTILINE)


def acentuar(texto, mapa):
    partes = PROTEGIDO.split(texto)
    palabra = re.compile(r"\b[A-Za-z]{2,}\b")

    def sust(m):
        w = m.group(0)
        d = w.lower()
        if d in mapa:
            return aplicar_mayusculas(w, mapa[d])
        return w

    for i in range(0, len(partes), 2):          # los impares son protegidos
        t = palabra.sub(sust, partes[i])
        for pat, rep in CONTEXTO:
            t = re.sub(pat, rep, t)
            t = re.sub(pat[:2] + pat[2].upper() + pat[3:],
                       rep[0].upper() + rep[1:], t)
        partes[i] = t
    return "".join(partes)


def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else "memoria.txt"
    mapa = construir_mapa()
    texto = open(ruta, encoding="utf-8").read()
    nuevo = acentuar(texto, mapa)
    open(ruta, "w", encoding="utf-8").write(nuevo)
    print(f"{ruta}: {len(mapa)} entradas en el mapa, "
          f"{sum(1 for a, b in zip(texto, nuevo) if a != b)} caracteres tocados")


if __name__ == "__main__":
    main()

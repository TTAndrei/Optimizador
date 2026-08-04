#!/usr/bin/env python3
"""Genera `TFG.docx` a partir de `memoria.txt` usando `MaquetaTFG.docx` como plantilla.

La plantilla aporta estilos, numeracion automatica de titulos, encabezados y margenes
oficiales de la EETAC. Este script solo sustituye el cuerpo (`word/document.xml`).

    python3 build_tfg.py [--plantilla RUTA] [--salida RUTA]

Marcas de `memoria.txt`:

    #1 / #2 / #3 / #4   titulo numerado de nivel 1..4
    #0                  titulo de nivel 1 sin numerar (Bibliografia, Anexos, ...)
    (linea normal)      parrafo de cuerpo
    * texto             vineta
    1. texto            lista numerada manual (el numero se escribe literal)
    $$ latex $$         ecuacion centrada y numerada (capitulo.n)
    [FIG] pie           hueco de figura + pie "Figura c.n: pie"
    [TBL] pie           pie de tabla "Tabla c.n: pie" (va tras la tabla)
    | a | b | c |       fila de tabla (la primera del bloque es cabecera)
    [TOC] [TOF] [TOT]   campos de indice de contenidos / figuras / tablas
    [PB]                salto de pagina
    [SECBODY]           salto de seccion: empieza el cuerpo (encabezados + pagina 1)
    ---                 linea en blanco explicita

    En linea: **negrita**, *cursiva*, `monoespaciado`, $latex$
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from latex2omml import latex_to_omml, latex_to_omml_inline  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
TITULO_TFG = ("Simulador CFD 2D acelerado por GPU y optimización genética "
              "de perfiles aerodinámicos")

SECT_PRELIM = (
    '<w:sectPr><w:pgSz w:w="11906" w:h="16838" w:code="9"/>'
    '<w:pgMar w:top="1418" w:right="1701" w:bottom="1418" w:left="1701"'
    ' w:header="709" w:footer="709" w:gutter="0"/><w:cols w:space="708"/>'
    '<w:docGrid w:linePitch="360"/></w:sectPr>')

SECT_BODY = (
    '<w:sectPr><w:headerReference w:type="even" r:id="rId8"/>'
    '<w:headerReference w:type="default" r:id="rId9"/>'
    '<w:headerReference w:type="first" r:id="rId10"/>'
    '<w:type w:val="oddPage"/><w:pgSz w:w="11906" w:h="16838" w:code="9"/>'
    '<w:pgMar w:top="1418" w:right="1701" w:bottom="1418" w:left="1701"'
    ' w:header="709" w:footer="709" w:gutter="0"/>'
    '<w:pgNumType w:start="1"/><w:cols w:space="708"/>'
    '<w:docGrid w:linePitch="360"/></w:sectPr>')

DOC_OPEN = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<w:document '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
    'mc:Ignorable="w14" '
    'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006">'
    "<w:body>")


def esc(t):
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


# --------------------------------------------------------------------------- runs

TOKEN_RE = re.compile(r"(\*\*.+?\*\*|(?<!\*)\*[^*\n]+?\*(?!\*)|`[^`\n]+?`|\$[^$\n]+?\$)")


def runs(text, base=""):
    """Convierte texto con marcas en linea a una lista de runs OOXML."""
    out = []
    for frag in TOKEN_RE.split(text):
        if not frag:
            continue
        if frag.startswith("**") and frag.endswith("**"):
            out.append(_run(frag[2:-2], "<w:b/>" + base))
        elif frag.startswith("`") and frag.endswith("`"):
            out.append(_run(frag[1:-1],
                            '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>'
                            '<w:sz w:val="18"/>' + base))
        elif frag.startswith("$") and frag.endswith("$"):
            out.append(latex_to_omml_inline(frag[1:-1]))
        elif frag.startswith("*") and frag.endswith("*"):
            out.append(_run(frag[1:-1], "<w:i/>" + base))
        else:
            out.append(_run(frag, base))
    return "".join(out)


def _run(text, rpr=""):
    pr = f"<w:rPr>{rpr}</w:rPr>" if rpr else ""
    return f'<w:r>{pr}<w:t xml:space="preserve">{esc(text)}</w:t></w:r>'


def para(text, style=None, ppr_extra="", base=""):
    ppr = ""
    if style or ppr_extra:
        ppr = "<w:pPr>"
        if style:
            ppr += f'<w:pStyle w:val="{style}"/>'
        ppr += ppr_extra + "</w:pPr>"
    return f"<w:p>{ppr}{runs(text, base)}</w:p>"


def field(instr, placeholder, rpr=""):
    pr = f"<w:rPr>{rpr}</w:rPr>" if rpr else ""
    return ('<w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r>'
            f'<w:r>{pr}<w:instrText xml:space="preserve">{esc(instr)}'
            "</w:instrText></w:r>"
            '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
            f'<w:r>{pr}<w:t xml:space="preserve">{esc(placeholder)}</w:t></w:r>'
            '<w:r><w:fldChar w:fldCharType="end"/></w:r>')


def numero_auto(cap, n, seq, rpr=""):
    """`cap.n` como campos STYLEREF+SEQ, para que Word lo renumere solo."""
    # El numero de capitulo va literal: STYLEREF no lo resuelve LibreOffice y el
    # numero de capitulo es estable. El contador dentro del capitulo si es un campo.
    return (_run(f"{cap}.", rpr)
            + field(rf" SEQ {seq}{cap} \* ARABIC ", str(n), rpr))


# --------------------------------------------------------------------------- tablas

def tabla(filas):
    ncol = max(len(f) for f in filas)
    ancho = 9070
    w = ancho // ncol
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for _ in range(ncol))
    borde = ('<w:tblBorders>'
             + "".join(f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
                       for s in ("top", "left", "bottom", "right",
                                 "insideH", "insideV"))
             + "</w:tblBorders>")
    out = ('<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/>'
           '<w:jc w:val="center"/>' + borde +
           '<w:tblLook w:val="04A0"/></w:tblPr>'
           f"<w:tblGrid>{grid}</w:tblGrid>")
    for i, f in enumerate(filas):
        cab = i == 0
        out += '<w:trPr><w:tblHeader/></w:trPr>' if cab else ""
        out += "<w:tr>"
        for j in range(ncol):
            celda = f[j] if j < len(f) else ""
            sombra = ('<w:shd w:val="clear" w:color="auto" w:fill="EDEDED"/>'
                      if cab else "")
            ppr = ('<w:pPr><w:spacing w:before="40" w:after="40"/>'
                   '<w:jc w:val="center"/></w:pPr>')
            texto = runs(celda, "<w:b/>" if cab else "")
            out += (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{sombra}'
                    '<w:vAlign w:val="center"/></w:tcPr>'
                    f"<w:p>{ppr}{texto}</w:p></w:tc>")
        out += "</w:tr>"
    return out + "</w:tbl>"


# --------------------------------------------------------------------------- parser

class Builder:
    def __init__(self):
        self.xml = []
        self.cap = 0          # capitulo actual
        self.n_eq = 0
        self.n_fig = 0
        self.n_tab = 0
        self.pend_tabla = []
        self.avisos = []

    def add(self, s):
        self.xml.append(s)

    def flush_tabla(self):
        if self.pend_tabla:
            self.add(tabla(self.pend_tabla))
            self.pend_tabla = []

    def heading(self, lvl, texto):
        self.flush_tabla()
        if lvl == 0:
            extra = ('<w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr>'
                     '<w:ind w:left="431" w:hanging="431"/>')
            self.add(para(texto, "Ttulo1", extra))
            return
        if lvl == 1:
            self.cap += 1
            self.n_eq = self.n_fig = self.n_tab = 0
        self.add(para(texto, f"Ttulo{lvl}"))

    def ecuacion(self, tex):
        self.flush_tabla()
        self.n_eq += 1
        omml = latex_to_omml(tex, display=False)
        ppr = ('<w:pPr><w:tabs><w:tab w:val="center" w:pos="4253"/>'
               '<w:tab w:val="right" w:pos="8494"/></w:tabs>'
               '<w:spacing w:before="120" w:after="120"/>'
               '<w:jc w:val="left"/></w:pPr>')
        rpr = '<w:sz w:val="18"/>'
        num = (_run("(", rpr)
               + numero_auto(self.cap, self.n_eq, "Ecuacion", rpr)
               + _run(")", rpr))
        self.add(f"<w:p>{ppr}<w:r><w:tab/></w:r>{omml}"
                 f"<w:r><w:tab/></w:r>{num}</w:p>")

    def figura(self, pie):
        self.flush_tabla()
        self.n_fig += 1
        hueco = ('<w:pPr><w:jc w:val="center"/>'
                 '<w:pBdr><w:top w:val="dashed" w:sz="4" w:space="4" w:color="808080"/>'
                 '<w:left w:val="dashed" w:sz="4" w:space="4" w:color="808080"/>'
                 '<w:bottom w:val="dashed" w:sz="4" w:space="4" w:color="808080"/>'
                 '<w:right w:val="dashed" w:sz="4" w:space="4" w:color="808080"/></w:pBdr>'
                 '<w:spacing w:before="120" w:after="60"/></w:pPr>')
        self.add(f"<w:p>{hueco}" + _run(
            f"[PENDIENTE: insertar aqui la figura {self.cap}.{self.n_fig}]",
            '<w:i/><w:color w:val="808080"/>') + "</w:p>")
        self.add("<w:p><w:pPr><w:pStyle w:val=\"Figura\"/></w:pPr>"
                 + _run("Figura ")
                 + numero_auto(self.cap, self.n_fig, "Figura")
                 + runs(": " + pie) + "</w:p>")

    def pie_tabla(self, pie):
        self.flush_tabla()
        self.n_tab += 1
        self.add("<w:p><w:pPr><w:pStyle w:val=\"Taula\"/></w:pPr>"
                 + _run("Tabla ")
                 + numero_auto(self.cap, self.n_tab, "Tabla")
                 + runs(": " + pie) + "</w:p>")


def construir(fuente):
    b = Builder()
    lineas = fuente.splitlines()
    i = 0
    while i < len(lineas):
        ln = lineas[i].rstrip()
        s = ln.strip()
        i += 1

        if not s:
            b.flush_tabla()
            continue

        if s.startswith("//"):                       # comentario del fuente
            continue

        if s.startswith("#0b "):        # subtitulo en negrita, fuera del indice
            b.flush_tabla()
            b.add(para(s[4:].strip(), None,
                       '<w:spacing w:before="240" w:after="60"/>'
                       '<w:jc w:val="left"/>', base="<w:b/>"))
            continue

        if s.startswith("#") and len(s) > 2 and s[1].isdigit() and s[2] == " ":
            b.heading(int(s[1]), s[3:].strip())
            continue

        if s == "[PB]":
            b.flush_tabla()
            b.add('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
            continue

        if s == "[SECBODY]":
            b.flush_tabla()
            b.add(f"<w:p><w:pPr>{SECT_PRELIM}</w:pPr></w:p>")
            continue

        if s == "---":
            b.flush_tabla()
            b.add("<w:p/>")
            continue

        if s in ("[TOC]", "[TOF]", "[TOT]"):
            b.flush_tabla()
            instr = {"[TOC]": r' TOC \o "1-3" \h \z \u ',
                     "[TOF]": r' TOC \h \z \c "Figura" ',
                     "[TOT]": r' TOC \h \z \c "Tabla" '}[s]
            ph = "[Actualiza el indice en Word: Ctrl+E, F9]"
            b.add(f"<w:p>{field(instr, ph)}</w:p>")
            continue

        if s.startswith("$$") and s.endswith("$$") and len(s) > 4:
            b.ecuacion(s[2:-2].strip())
            continue

        if s.startswith("[FIG]"):
            b.figura(s[5:].strip())
            continue

        if s.startswith("[TBL]"):
            b.pie_tabla(s[5:].strip())
            continue

        if s.startswith("|") and s.endswith("|"):
            b.pend_tabla.append([c.strip() for c in s[1:-1].split("|")])
            continue

        b.flush_tabla()

        if s.startswith("* "):
            b.add(para(s[2:], "Prrafodelista",
                       '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>'))
            continue

        if s.startswith("@ ") and " | " in s:
            term, desc = s[2:].split(" | ", 1)
            b.add("<w:p><w:pPr><w:tabs><w:tab w:val=\"left\" w:pos=\"1418\"/></w:tabs>"
                  '<w:spacing w:before="40"/><w:ind w:left="1418" w:hanging="1418"/>'
                  "<w:jc w:val=\"left\"/></w:pPr>"
                  + runs(term.strip()) + "<w:r><w:tab/></w:r>"
                  + runs(desc.strip()) + "</w:p>")
            continue

        if s.startswith("[REF] "):
            b.add(para(s[6:], "Prrafodelista",
                       '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="15"/></w:numPr>'
                       '<w:spacing w:before="60"/><w:jc w:val="left"/>'
                       '<w:ind w:left="567" w:hanging="567"/>'))
            continue

        m = re.match(r"^(\d+)\.\s+(.*)$", s)
        if m:
            b.add(para(f"{m.group(1)}. {m.group(2)}", "Prrafodelista",
                       '<w:ind w:left="720" w:hanging="360"/>'))
            continue

        b.add(para(s))

    b.flush_tabla()
    return "".join(b.xml)


# --------------------------------------------------------------------------- salida

def empaquetar(plantilla, salida, cuerpo):
    tmp = tempfile.mkdtemp(prefix="tfgdocx_")
    try:
        with zipfile.ZipFile(plantilla) as z:
            z.extractall(tmp)

        doc = os.path.join(tmp, "word", "document.xml")
        with open(doc, "w", encoding="utf-8") as f:
            f.write(DOC_OPEN + cuerpo + SECT_BODY + "</w:body></w:document>")

        hdr = os.path.join(tmp, "word", "header1.xml")
        with open(hdr, encoding="utf-8") as f:
            h = f.read()
        h = h.replace("Títol del TFG", esc(TITULO_TFG))
        with open(hdr, "w", encoding="utf-8") as f:
            f.write(h)

        if os.path.exists(salida):
            os.remove(salida)
        with zipfile.ZipFile(salida, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(tmp):
                for name in files:
                    ruta = os.path.join(root, name)
                    z.write(ruta, os.path.relpath(ruta, tmp))
    finally:
        shutil.rmtree(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuente", default=os.path.join(AQUI, "memoria.txt"))
    ap.add_argument("--plantilla",
                    default=os.path.expanduser("~/Descargas/MaquetaTFG.docx"))
    ap.add_argument("--salida", default=os.path.join(AQUI, "TFG.docx"))
    ap.add_argument("--pdf", action="store_true", help="convertir a PDF al terminar")
    a = ap.parse_args()

    with open(a.fuente, encoding="utf-8") as f:
        cuerpo = construir(f.read())
    empaquetar(a.plantilla, a.salida, cuerpo)
    print(f"escrito {a.salida} ({os.path.getsize(a.salida)/1024:.0f} kB)")

    if a.pdf:
        subprocess.run(["soffice", "--headless", "--convert-to", "pdf",
                        "--outdir", os.path.dirname(a.salida), a.salida],
                       check=True, capture_output=True)
        print("PDF generado")


if __name__ == "__main__":
    main()

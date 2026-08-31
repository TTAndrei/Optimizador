#!/usr/bin/env python3
"""Construye el capitulo 4 reordenado como documento suelto (docx + odt).

Parches respecto de build_tfg.py:
  - el capitulo arranca en 4 (self.cap = 3 antes del primer #1)
  - numeros de titulo, ecuacion, figura y tabla literales: los campos SEQ
    reiniciarian la numeracion en un extracto
  - [FIG] admite sufijo "||ruta" y el hueco indica que fichero insertar
"""
import os
import re
import subprocess
import sys

AQUI = "/home/ttandrei/Proyectos/Optimizador/Optimizador/docs/tfg"
sys.path.insert(0, AQUI)
import build_tfg as B


def heading(self, lvl, texto):
    self.flush_tabla()
    if lvl == 0:
        extra = ('<w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr>'
                 '<w:ind w:left="431" w:hanging="431"/>')
        self.add(B.para(texto, "Ttulo1", extra))
        return
    if lvl == 1:
        self.cap += 1
        self.n_eq = self.n_fig = self.n_tab = 0
        self.sub = [0, 0, 0]
        num = f"{self.cap}."
    else:
        i = lvl - 2
        self.sub[i] += 1
        for j in range(i + 1, 3):
            self.sub[j] = 0
        num = f"{self.cap}." + ".".join(str(x) for x in self.sub[:i + 1])
    extra = ('<w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr>'
             '<w:ind w:left="431" w:hanging="431"/>')
    self.add(B.para(f"{num} {texto}", f"Ttulo{lvl}", extra))


def figura(self, pie):
    self.flush_tabla()
    self.n_fig += 1
    ruta = None
    if "||" in pie:
        pie, ruta = pie.split("||", 1)
        pie, ruta = pie.strip(), ruta.strip()
    etiqueta = (f"[Figura {self.cap}.{self.n_fig}: insertar {ruta}]" if ruta
                else f"[Figura {self.cap}.{self.n_fig}: pendiente de generar]")
    hueco = ('<w:pPr><w:jc w:val="center"/>'
             '<w:pBdr><w:top w:val="dashed" w:sz="4" w:space="4" w:color="808080"/>'
             '<w:left w:val="dashed" w:sz="4" w:space="4" w:color="808080"/>'
             '<w:bottom w:val="dashed" w:sz="4" w:space="4" w:color="808080"/>'
             '<w:right w:val="dashed" w:sz="4" w:space="4" w:color="808080"/></w:pBdr>'
             '<w:spacing w:before="120" w:after="60"/></w:pPr>')
    self.add(f"<w:p>{hueco}"
             + B._run(etiqueta, '<w:i/><w:color w:val="808080"/>') + "</w:p>")
    self.add('<w:p><w:pPr><w:pStyle w:val="Figura"/></w:pPr>'
             + B._run(f"Figura {self.cap}.{self.n_fig}")
             + B.runs(": " + pie) + "</w:p>")


def pie_tabla(self, pie):
    self.flush_tabla()
    self.n_tab += 1
    self.add('<w:p><w:pPr><w:pStyle w:val="Taula"/></w:pPr>'
             + B._run(f"Tabla {self.cap}.{self.n_tab}")
             + B.runs(": " + pie) + "</w:p>")


_init = B.Builder.__init__


def init(self):
    _init(self)
    self.cap = 3          # el primer #1 pasa a ser el capitulo 4
    self.sub = [0, 0, 0]


B.Builder.__init__ = init
B.Builder.heading = heading
B.Builder.figura = figura
B.Builder.pie_tabla = pie_tabla
B.numero_auto = lambda cap, n, seq, rpr="": B._run(f"{cap}.{n}", rpr)
B.SECT_BODY = B.SECT_PRELIM   # sin encabezados: es un documento suelto

fuente = os.path.join(AQUI, "cap4_reordenado.txt")
salida = os.path.join(AQUI, "Cap4_reordenado.docx")
plantilla = os.path.expanduser("~/Descargas/MaquetaTFG.docx")

with open(fuente, encoding="utf-8") as f:
    cuerpo = B.construir(f.read())
B.empaquetar(plantilla, salida, cuerpo)
print(f"docx {salida} ({os.path.getsize(salida)/1024:.0f} kB)")

subprocess.run(["soffice", "--headless", "--convert-to", "odt",
                "--outdir", AQUI, salida], check=True, capture_output=True)
odt = salida.replace(".docx", ".odt")
print(f"odt  {odt} ({os.path.getsize(odt)/1024:.0f} kB)")
subprocess.run(["soffice", "--headless", "--convert-to", "pdf",
                "--outdir", AQUI, odt], check=True, capture_output=True)
pdf = salida.replace(".docx", ".pdf")
print(f"pdf  {pdf} ({os.path.getsize(pdf)/1024:.0f} kB)")

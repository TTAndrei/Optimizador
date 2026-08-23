"""Conversor LaTeX -> OMML (Office Math Markup) para el subconjunto que usa la memoria.

No pretende cubrir LaTeX: cubre exactamente lo que aparece en `memoria.txt`. Si algo no
esta soportado lanza ValueError, para que el fallo salga en la generacion y no en Word.
"""

import re

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ", "eta": "η",
    "theta": "θ", "kappa": "κ", "lambda": "λ", "mu": "μ",
    "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ",
    "tau": "τ", "phi": "φ", "varphi": "ϕ", "chi": "χ", "psi": "ψ",
    "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ",
    "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ",
    "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}

SYMBOL = {
    "partial": "∂", "nabla": "∇", "infty": "∞", "cdot": "·",
    "times": "×", "approx": "≈", "simeq": "≃", "sim": "∼",
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠",
    "ne": "≠", "to": "→", "rightarrow": "→", "Rightarrow": "⇒",
    "pm": "±", "mp": "∓", "propto": "∝", "ll": "≪", "gg": "≫",
    "in": "∈", "notin": "∉", "subset": "⊂", "cup": "∪",
    "cap": "∩", "forall": "∀", "exists": "∃", "equiv": "≡",
    "ldots": "…", "dots": "…", "cdots": "⋯", "prime": "′",
    "circ": "∘", "deg": "°", "perp": "⊥", "parallel": "∥",
    "langle": "⟨", "rangle": "⟩", "emptyset": "∅",
    "lesssim": "≲", "gtrsim": "≳", "ast": "∗", "star": "⋆",
    "div": "÷", "setminus": "\\", "vert": "|", "|": "‖",
    "leftarrow": "←", "leftrightarrow": "↔", "mapsto": "↦",
    "oplus": "⊕", "otimes": "⊗", "sqrtsym": "√", "angle": "∠",
    "hbar": "ℏ", "ell": "ℓ", "Re": "ℜ", "Im": "ℑ",
    "prop": "∝", "therefore": "∴", "because": "∵",
}

NARY = {"int": "∫", "oint": "∮", "iint": "∬", "sum": "∑",
        "prod": "∏", "lim": None}

FUNCS = {"ln", "log", "exp", "sin", "cos", "tan", "max", "min", "sgn", "erf",
         "tanh", "sinh", "cosh", "arctan", "lim", "sup", "inf", "det", "diag"}

SPACES = {" ": " ", "~": " ", ",": " ", ";": " ", ":": " ", "quad": " ", "qquad": "  ", "!": ""}

MATHFONT = '<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr>'


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def run(text, upright=False):
    if text == "":
        return ""
    pr = '<m:rPr><m:sty m:val="p"/></m:rPr>' if upright else ""
    return f'<m:r>{pr}{MATHFONT}<m:t xml:space="preserve">{esc(text)}</m:t></m:r>'


class Tok:
    def __init__(self, kind, val):
        self.kind, self.val = kind, val

    def __repr__(self):
        return f"{self.kind}:{self.val}"


def tokenize(s):
    toks, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
        elif c == "\\":
            m = re.match(r"\\([A-Za-z]+|.)", s[i:])
            toks.append(Tok("cmd", m.group(1)))
            i += m.end()
        elif c in "{}^_&":
            toks.append(Tok(c, c))
            i += 1
        else:
            m = re.match(r"[0-9]+(?:[.,][0-9]+)?", s[i:])
            if m:
                toks.append(Tok("num", m.group(0)))
                i += m.end()
            else:
                toks.append(Tok("chr", c))
                i += 1
    return toks


class Parser:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def next(self):
        tk = self.t[self.i]
        self.i += 1
        return tk

    def group(self):
        """Un argumento: {..} o un unico token."""
        tk = self.peek()
        if tk is None:
            raise ValueError("argumento vacio al final de la expresion")
        if tk.kind == "{":
            self.next()
            out = self.seq(stop="}")
            self.next()
            return out
        return self.atom()

    def seq(self, stop=None):
        out = []
        while True:
            tk = self.peek()
            if tk is None or (stop and tk.kind == stop):
                break
            out.append(self.script())
        return "".join(out)

    def script(self):
        base = self.atom()
        sub = sup = None
        while True:
            tk = self.peek()
            if tk is None or tk.kind not in ("_", "^"):
                break
            self.next()
            if tk.kind == "_":
                sub = self.group()
            else:
                sup = self.group()
        if sub is not None and sup is not None:
            return (f"<m:sSubSup><m:e>{base}</m:e><m:sub>{sub}</m:sub>"
                    f"<m:sup>{sup}</m:sup></m:sSubSup>")
        if sub is not None:
            return f"<m:sSub><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>"
        if sup is not None:
            return f"<m:sSup><m:e>{base}</m:e><m:sup>{sup}</m:sup></m:sSup>"
        return base

    def nary(self, glyph):
        sub = sup = ""
        while True:
            tk = self.peek()
            if tk is None or tk.kind not in ("_", "^"):
                break
            self.next()
            if tk.kind == "_":
                sub = self.group()
            else:
                sup = self.group()
        body = self.script()
        hide = ""
        if not sub:
            hide += '<m:subHide m:val="1"/>'
        if not sup:
            hide += '<m:supHide m:val="1"/>'
        return (f'<m:nary><m:naryPr><m:chr m:val="{glyph}"/><m:limLoc m:val="subSup"/>'
                f'{hide}<m:ctrlPr>{MATHFONT}</m:ctrlPr></m:naryPr>'
                f"<m:sub>{sub}</m:sub><m:sup>{sup}</m:sup><m:e>{body}</m:e></m:nary>")

    def delim(self, open_ch):
        """\\left( ... \\right)"""
        body_toks = []
        depth = 1
        close_ch = ")"
        while self.i < len(self.t):
            tk = self.t[self.i]
            if tk.kind == "cmd" and tk.val == "left":
                depth += 1
            elif tk.kind == "cmd" and tk.val == "right":
                depth -= 1
                if depth == 0:
                    self.i += 1
                    close_ch = self.next().val
                    if close_ch == ".":
                        close_ch = ""
                    break
            body_toks.append(tk)
            self.i += 1
        body = Parser(body_toks).seq()
        if open_ch == ".":
            open_ch = ""
        return (f'<m:d><m:dPr><m:begChr m:val="{esc(open_ch)}"/>'
                f'<m:endChr m:val="{esc(close_ch)}"/><m:ctrlPr>{MATHFONT}</m:ctrlPr></m:dPr>'
                f"<m:e>{body}</m:e></m:d>")

    def atom(self):
        tk = self.next()
        if tk.kind in ("num",):
            return run(tk.val, upright=True)
        if tk.kind == "chr":
            return run(tk.val)
        if tk.kind == "{":
            body = self.seq(stop="}")
            self.next()
            return body
        if tk.kind == "cmd":
            c = tk.val
            if c in ("frac", "dfrac", "tfrac"):
                num, den = self.group(), self.group()
                return (f"<m:f><m:fPr><m:ctrlPr>{MATHFONT}</m:ctrlPr></m:fPr>"
                        f"<m:num>{num}</m:num><m:den>{den}</m:den></m:f>")
            if c == "sqrt":
                return (f'<m:rad><m:radPr><m:degHide m:val="1"/>'
                        f"<m:ctrlPr>{MATHFONT}</m:ctrlPr></m:radPr>"
                        f"<m:deg/><m:e>{self.group()}</m:e></m:rad>")
            if c in ("mathcal", "mathbb", "mathscr"):
                return self.group()       # sin fuente caligrafica: se deja recto
            if c in ("mathbf", "bm", "vec"):
                inner = self.group()
                return inner.replace(MATHFONT, '<w:rPr><w:rFonts w:ascii="Cambria Math"'
                                                ' w:hAnsi="Cambria Math"/><w:b/></w:rPr>')
            if c in ("text", "mathrm", "mathit", "operatorname"):
                tk2 = self.peek()
                if tk2 is not None and tk2.kind == "{":
                    self.next()
                    buf = ""
                    depth = 1
                    while self.i < len(self.t):
                        t2 = self.t[self.i]
                        if t2.kind == "{":
                            depth += 1
                        elif t2.kind == "}":
                            depth -= 1
                            if depth == 0:
                                self.i += 1
                                break
                        buf += (" " if t2.kind == "cmd" and t2.val == " " else t2.val)
                        self.i += 1
                    return run(buf, upright=True)
                return run(self.next().val, upright=True)
            if c == "bar" or c == "overline":
                return (f'<m:bar><m:barPr><m:pos m:val="top"/><m:ctrlPr>{MATHFONT}</m:ctrlPr>'
                        f"</m:barPr><m:e>{self.group()}</m:e></m:bar>")
            if c in ("hat", "tilde", "dot", "ddot"):
                ch = {"hat": "̂", "tilde": "̃", "dot": "̇", "ddot": "̈"}[c]
                return (f'<m:acc><m:accPr><m:chr m:val="{ch}"/><m:ctrlPr>{MATHFONT}</m:ctrlPr>'
                        f"</m:accPr><m:e>{self.group()}</m:e></m:acc>")
            if c in ("big", "Big", "bigg", "Bigg", "biggl", "biggr",
                     "bigl", "bigr", "displaystyle", "limits", "nolimits"):
                return ""                 # sugerencias de tamano: sin efecto en OMML
            if c == "left":
                return self.delim(self.next().val)
            if c == "right":
                raise ValueError("\\right sin \\left")
            if c in NARY:
                if c == "lim":
                    return run("lim", upright=True)
                return self.nary(NARY[c])
            if c in GREEK:
                return run(GREEK[c])
            if c in SYMBOL:
                return run(SYMBOL[c], upright=True)
            if c in SPACES:
                return run(SPACES[c], upright=True)
            if c in FUNCS:
                return run(c, upright=True)
            if c in ("{", "}", "%", "#", "&", "_"):
                return run(c, upright=True)
            raise ValueError(f"comando LaTeX no soportado: \\{c}")
        return run(tk.val)


def latex_to_omml(src, display=True):
    body = Parser(tokenize(src)).seq()
    tag = "oMathPara" if display else "oMath"
    if display:
        return ("<m:oMathPara><m:oMathParaPr>"
                '<m:jc m:val="center"/></m:oMathParaPr>'
                f"<m:oMath>{body}</m:oMath></m:oMathPara>")
    return f"<m:oMath>{body}</m:oMath>"


def latex_to_omml_inline(src):
    return f"<m:oMath>{Parser(tokenize(src)).seq()}</m:oMath>"

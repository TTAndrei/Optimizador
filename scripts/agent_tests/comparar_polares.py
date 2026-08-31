"""
Compara la polar del solver optimizado contra la publicada, angulo a angulo.

Las dos se calcularon con la misma configuracion fisica —mismo perfil, dominio
24x16 con cx=6, dx=0.002, Re=1e5, CFL=0.5, 26000 iteraciones, paradas
desactivadas— y por la misma ruta de codigo, asi que las 28 metricas son
directamente comparables sin conversiones.

Lo que se mira, por orden de importancia:

  1. Si el cambio supera la incertidumbre estadistica del punto publicado. Un
     Cl que se mueve menos que su propio CI95 no es una diferencia, es ruido.
  2. Si el cambio en Cd sigue un patron con el angulo. La hipotesis es que la
     divergencia espuria junto a la pared infla la resistencia, asi que su
     efecto deberia crecer con la carga del perfil. Un patron monotono apoya el
     mecanismo; uno erratico lo desmiente.
  3. Si la divergencia residual baja en todos los puntos. Es la medida directa
     de que la proyeccion converge mejor, independiente de las fuerzas.

Uso:
    .venv/bin/python scripts/agent_tests/comparar_polares.py
"""
from __future__ import annotations

import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

NUEVA = os.path.join(ROOT, "results/polar_optimizada/polar_dx0.0020.json")
PUBLI = os.path.join(ROOT, "results/richardson_polar_domC/polar_dx0.0020.json")
# alpha=10 no esta en el estudio de Richardson; su unico homologo quedo truncado
TRUNC = os.path.join(ROOT, "results/polar_2grados_dom24x16/polar_ganador_tfg2.json")


def carga(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    n, p, t = carga(NUEVA), carga(PUBLI), carga(TRUNC)
    if not n:
        print("todavia no hay resultados nuevos")
        return

    print("=" * 92)
    print("POLAR: solver optimizado frente al publicado")
    print("  perfil ganador AG, dominio 24x16 cx=6, dx=0.002, Re=1e5, t~20 (26000 iters)")
    print("=" * 92)
    hdr = (f"{'alpha':>6}{'Cl pub':>10}{'Cl opt':>10}{'dCl':>9}{'CI95':>7}"
           f"{'Cd pub':>10}{'Cd opt':>10}{'dCd':>9}{'CI95':>7}"
           f"{'L/D pub':>9}{'L/D opt':>9}{'dL/D':>9}")
    print(hdr)
    print("-" * len(hdr))

    filas = []
    for ka in sorted(n, key=float):
        a = float(ka)
        nn = n[ka]
        ref, origen = (p.get(ka), "richardson") if ka in p else (t.get(ka), "truncado")
        if ref is None:
            print(f"{a:>6.1f}   (sin referencia)")
            continue
        def d(k):
            return 100.0 * (nn[k] - ref[k]) / ref[k] if ref.get(k) else float("nan")
        def ci(k):
            v = ref.get(f"{k}_ci95")
            return 100.0 * v / ref[k] if v and ref.get(k) else float("nan")
        marca = "" if origen == "richardson" else "  <- ref truncada"
        print(f"{a:>6.1f}{ref['cl']:>10.5f}{nn['cl']:>10.5f}{d('cl'):>8.2f}%{ci('cl'):>6.1f}%"
              f"{ref['cd']:>10.5f}{nn['cd']:>10.5f}{d('cd'):>8.2f}%{ci('cd'):>6.1f}%"
              f"{ref['ld']:>9.3f}{nn['ld']:>9.3f}{d('ld'):>8.2f}%{marca}")
        filas.append((a, d('cl'), d('cd'), d('ld'), ci('cl'), ci('cd'), origen))

    print("-" * len(hdr))
    print("dCl/dCd/dL/D = cambio frente al publicado.  CI95 = incertidumbre estadistica")
    print("del punto publicado: un cambio por debajo de su CI95 no es distinguible del ruido.")

    val = [f for f in filas if f[6] == "richardson"]
    if len(val) >= 3:
        print("\n" + "=" * 92)
        print("PATRON DE Cd CON EL ANGULO")
        print("=" * 92)
        print("  La hipotesis es que la divergencia espuria junto a la pared infla la")
        print("  resistencia, luego su efecto deberia crecer con la carga del perfil.\n")
        for a, _, dcd, _, _, _, _ in val:
            barra = "#" * int(abs(dcd))
            print(f"   alpha={a:>4.1f}   dCd = {dcd:>7.2f}%  {barra}")
        seq = [f[2] for f in val]
        mono = all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))
        print(f"\n  monotono (cae mas segun sube alpha): {'SI, apoya el mecanismo' if mono else 'NO'}")

    # Referencia externa. XFOIL es de un NACA0012, no del perfil optimizado, asi
    # que los valores absolutos no son comparables; lo que sirve es la FORMA de
    # la cubeta de resistencia y de que lado del referente cae cada polar. El
    # simulador tiene un sesgo documentado de sobrestimar Cd (validacion previa:
    # +124% de media), asi que acercarse a XFOIL es indicio de correccion, no de
    # casualidad.
    xf = os.path.join(ROOT, "results/1eraGranOptimizacion/verificacion_numerica/validacion_xfoil.json")
    if os.path.exists(xf):
        x = {f"{q['alpha']:.1f}": q for q in json.load(open(xf))["puntos"]}
        comunes = [k for k in sorted(n, key=float) if k in x and k in p]
        if comunes:
            print("\n" + "=" * 92)
            print("DISTANCIA A XFOIL  (NACA0012 Re=1e5; referencia de forma, no de valor)")
            print("=" * 92)
            print(f"{'alpha':>6}{'Cd XFOIL':>11}{'Cd pub':>10}{'x sobre':>9}"
                  f"{'Cd opt':>10}{'x sobre':>9}{'mejora':>9}")
            print("-" * 64)
            for k in comunes:
                cx_ = x[k]["cd_xfoil"]
                rp_, rn = p[k]["cd"] / cx_, n[k]["cd"] / cx_
                print(f"{float(k):>6.1f}{cx_:>11.5f}{p[k]['cd']:>10.5f}{rp_:>9.2f}"
                      f"{n[k]['cd']:>10.5f}{rn:>9.2f}{'si' if rn < rp_ else 'NO':>9}")
            print("-" * 64)
            print("  'x sobre' = cuantas veces el Cd simulado supera al de XFOIL.")
            print("  Mas cerca de 1 es mejor, pero ojo: son perfiles distintos.")

    print("\n" + "=" * 92)
    print("COSTE Y CONVERGENCIA")
    print("=" * 92)
    print(f"{'alpha':>6}{'wall pub':>11}{'wall opt':>11}{'x vel':>8}{'n muestras':>12}")
    print("-" * 48)
    tp = tn = 0.0
    for ka in sorted(n, key=float):
        ref = p.get(ka)
        if not ref:
            continue
        tp += ref['wall_s']; tn += n[ka]['wall_s']
        print(f"{float(ka):>6.1f}{ref['wall_s']:>11.0f}{n[ka]['wall_s']:>11.0f}"
              f"{ref['wall_s']/n[ka]['wall_s']:>8.2f}{n[ka]['n_samples']:>12}")
    if tn:
        print("-" * 48)
        print(f"{'TOTAL':>6}{tp:>11.0f}{tn:>11.0f}{tp/tn:>8.2f}")
        print(f"\n  {tp/3600:.2f} h  ->  {tn/3600:.2f} h")


if __name__ == "__main__":
    main()

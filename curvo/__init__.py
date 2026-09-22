"""Solver curvilineo sobre malla C monobloque adaptada al cuerpo.

Modulo aparte: `Simulador2D.py` (cartesiano + IBM) no se toca.

    malla        generador de malla C por marcha hiperbolica
    metrica      areas de cara, volumenes y coeficientes, en numpy o cupy
    operadores   divergencia, gradiente Green-Gauss, laplaciano de metrica completa
    multigrid    correccion aditiva (ACM) + kernels CUDA del suavizador
    conveccion   conveccion-difusion implicita conservativa con limitador TVD
    proyeccion   proyeccion de presion, Poisson compacto resuelto con PCG
    turbulencia  Spalart-Allmaras
    solver       Navier-Stokes incompresible por paso fraccionado
    fuerzas      Cl, Cd, Cm y Cp integrados sobre la linea j=0
    convergencia parada por fuerzas: hasta que Cl y Cd fijan el tercer decimal

Todo funciona igual con numpy y con cupy: los modulos usan el `xp` del array que
reciben. En GPU conviene float32 (ver `multigrid.tolerancia`).
"""

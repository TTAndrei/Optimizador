"""Especificacion declarativa de los controles de la GUI curvilinea.

Tres niveles: PESTANAS -> bloques -> parametros. Cada parametro es

    (clave, etiqueta, tipo, ayuda [, opciones])

con tipo en "f" (float), "i" (entero), "b" (casilla), "c" (desplegable, y
entonces el 5o elemento son las opciones) y "s" (texto).

`ayuda` es lo que sale en el tooltip, y sale **resumido de los docstrings de
`curvo/malla.py`**: son numeros medidos, no estimaciones. Si cambia el defecto
o la medida, se cambian los dos sitios.
"""

PESTANAS = [
    ("Malla", [
        ("Pared", [
            ("dn_pared", "Paso de pared [c]", "f",
             "Espesor de la primera celda pegada al perfil. Es el parametro que "
             "manda: fija la resolucion de la capa limite y con ella casi todo lo "
             "demas.\n"
             "  2e-3  ->  y+ ~ 10 a Re=1e5\n"
             "  8e-5  ->  y+ < 1 en TODA la pared a Re=1e5  (defecto)\n"
             "Bajarlo mete mas capas y mas ocasiones de que la marcha se pliegue."),
            ("n_capas_pared", "Capas de pared", "i",
             "Capas pegadas a la pared que crecen a 'Crecimiento en pared' en vez "
             "de al crecimiento normal: un bloque de capa limite con espaciado casi "
             "constante.\n"
             "Medido: 25 capas dan el mismo y+ y el mismo reparto que 15; las 10 "
             "extra se van al campo intermedio."),
            ("crecimiento_pared", "Crecimiento en pared", "f",
             "Razon de crecimiento dentro del bloque de capa limite.\n"
             "1.0 = espaciado constante (defecto)."),
            ("ajustar_dn_pared", "Ajuste por radio de morro", "f",
             "Recorta el paso de pared a este factor por el radio del borde de "
             "ataque. 0 = desactivado.\n"
             "Con 0.5, un perfil de morro afilado usa medio radio de paso.\n"
             "AVISO medido: sobre el historial del GA EMPEORA (12.1 % de mallas "
             "validas frente a 15.9 % con paso fijo), porque bajar el paso sin "
             "bajar el crecimiento mete mas capas."),
        ]),
        ("Superficie", [
            ("n_sup", "Puntos de superficie", "i",
             "Puntos que recorren el perfil, de borde de salida a borde de salida. "
             "Fija el espaciado tangencial medio (perimetro / n).\n"
             "Medido: 256 -> 512 solo pasa la ortogonalidad de 57.3 a 59.9 grados y "
             "cuesta celdas en proporcion directa. NO es la palanca para arreglar "
             "una malla mala."),
            ("razon_le", "Refino LE", "f",
             "Espaciado en el borde de ataque, como fraccion del espaciado maximo. "
             "Mas bajo = mas puntos concentrados en el morro."),
            ("ancho_le", "Ancho de refino LE", "f",
             "Anchura de la zona de agrupamiento del borde de ataque, en fraccion "
             "de la longitud de arco total."),
            ("razon_te", "Refino TE", "f",
             "Lo mismo que 'Refino LE' pero en el borde de salida."),
            ("ancho_te", "Ancho de refino TE", "f",
             "Anchura de la zona de agrupamiento del borde de salida, en fraccion "
             "de la longitud de arco total."),
        ]),
        ("Borde de salida", [
            ("cola_te", "Cola de cierre TE [gaps]", "f",
             "Longitud de la cola con que se cierra una base roma, en multiplos del "
             "gap. Solo actua si el perfil tiene base finita.\n"
             "Colgar el corte de estela del punto medio de una base plana (0) deja "
             "un nodo de 90 grados que la marcha no puede abanicar: jacobiano "
             "negativo en las dos uniones base-estela.\n"
             "Medido: AG24 pasa de 6 celdas cruzadas y 26 grados de ortogonalidad a "
             "0 y 83. La ventana util es 2.5 a 6; 4 es el centro.\n"
             "La cola es geometria inventada: se avisa por encima del 1 % de cuerda."),
        ]),
        ("Estela y dominio", [
            ("n_estela", "Puntos por lado de estela", "i",
             "Puntos a lo largo de CADA lado del corte de estela. Tiene que ser el "
             "mismo por los dos lados para que el emparejamiento i <-> N-1-i del "
             "corte sea una permutacion de indices y no una interpolacion."),
            ("x_out", "Plano de salida [c]", "f",
             "Hasta donde llega la malla aguas abajo, en cuerdas desde el borde de "
             "ataque.\n"
             "Manda mas que la extension lateral: el estudio de dominio del solver "
             "cartesiano midio que truncar la estela es el error dominante."),
            ("distancia_lejos", "Distancia campo lejano [c]", "f",
             "Hasta donde marcha la malla desde la pared, en cuerdas. Fija la "
             "frontera exterior; el numero de capas se deduce de este valor y del "
             "crecimiento."),
            ("crecimiento", "Crecimiento normal", "f",
             "Razon geometrica del espaciado normal entre capas consecutivas, fuera "
             "del bloque de pared.\n"
             "Mas bajo = transicion mas suave y mas precision lejos de la pared, a "
             "cambio de mas capas y mas coste."),
            ("dn_max", "Paso normal maximo [c]", "f",
             "Tope absoluto del espaciado normal. Vacio o 'inf' = sin tope "
             "(defecto).\n"
             "Sin tope el paso crece geometricamente hasta la ultima capa, pero el "
             "crecimiento es autosemejante: la celda mas gruesa mide ~1/8 de su "
             "distancia a la pared."),
            ("aspecto_max", "Aspecto maximo 1a capa", "f",
             "Relacion de aspecto que no se deja pasar a la primera capa. La regla "
             "es dn(xi) = max(dn_pared, ds(xi) / aspecto_max).\n"
             "Solo actua donde el espaciado tangencial se dispara, o sea en la "
             "estela lejana, y NO toca el perfil. Sin el, el corte de estela "
             "arrastraba el paso de pared 18 cuerdas aguas abajo: aspecto 9142.\n"
             "Bajarlo mucho se come la resolucion de la estela cercana."),
        ]),
        ("Marcha (avanzado)", [
            ("mezcla_volumen", "Suavizado de volumen", "f",
             "Cuanto se suaviza a lo largo de la superficie el volumen de celda que "
             "la marcha impone.\n"
             "0 = el area de cada celda sigue el espaciado local exacto; 1 = sigue "
             "una version suavizada.\n"
             "Subirlo alisa el campo lejano (medido: 94 -> 30 celdas cruzadas), a "
             "cambio de adaptarse menos a la geometria."),
            ("disipacion", "Disipacion de marcha", "f",
             "Amortiguacion implicita de la marcha, sobre el INCREMENTO de cada "
             "paso. Es lo que la sostiene en zonas concavas.\n"
             "Actua sobre el incremento y no sobre la posicion: suavizar posiciones "
             "es lo que colapsa el borde de ataque."),
            ("disipacion_curva", "Disipacion por curvatura", "f",
             "Refuerzo de la amortiguacion donde la superficie es concava.\n"
             "Medido: reforzarla tambien en las esquinas convexas EMPEORA (con 25 "
             "aparecen 41 celdas cruzadas en un caso que con 8 sale limpio). "
             "Subirlo mucho tampoco arregla un borde de salida romo."),
        ]),
    ]),
    ("Fisica", [
        ("Corriente libre", [
            ("u_inf", "Velocidad U infinito", "f",
             "Modulo de la velocidad de la corriente libre. Con cuerda 1 y esta "
             "velocidad, el Reynolds es u_inf / nu."),
            ("nu", "Viscosidad cinematica", "f",
             "Viscosidad cinematica. Con cuerda 1 y u_inf 1, nu = 1e-5 es Re = 1e5."),
            ("alfa", "Angulo de ataque [grados]", "f",
             "En malla adaptada al cuerpo el angulo se aplica INCLINANDO LA "
             "CORRIENTE, no la geometria: una sola malla sirve para la polar "
             "entera."),
        ]),
        ("Turbulencia", [
            ("turbulento", "Spalart-Allmaras", "b",
             "Activa el modelo de turbulencia de una ecuacion. Sin el, la corrida "
             "es laminar."),
            ("nu_tilde_inf", "Nu tilde infinito", "f",
             "Valor de la variable de Spalart-Allmaras en la corriente libre, en "
             "multiplos de nu."),
        ]),
        ("Integracion temporal", [
            ("dt", "Paso temporal", "f",
             "Paso de tiempo del solver. El limitador que suele mandar es el "
             "viscoso, dt_visc = 0.25 dx^2 / nu_ef."),
            ("bdf2", "BDF2", "b",
             "Integracion de segundo orden en el tiempo en vez de Euler atras. "
             "Mas preciso en transitorios; para buscar un estacionario no hace "
             "falta."),
            ("correcciones", "Correcciones de velocidad", "i",
             "Pasadas de correccion diferida de los terminos cruzados en el paso de "
             "momento."),
            ("correcciones_p", "Correcciones de presion", "i",
             "Pasadas de correccion de los terminos cruzados en la proyeccion de "
             "presion."),
        ]),
    ]),
    ("Ejecucion", [
        ("Duracion y muestreo", [
            ("pasos", "Pasos", "i",
             "Numero de pasos temporales de la corrida."),
            ("cada_historia", "Muestreo de Cl/Cd", "i",
             "Cada cuantos pasos se apunta un punto de la historia de fuerzas."),
            ("cada_campo", "Muestreo de campos", "i",
             "Cada cuantos pasos se manda un campo completo a la vista.\n"
             "Bajarlo mucho cuesta: volcar y dibujar bloquea la GPU."),
            ("tol_fuerzas", "Parada por fuerzas", "f",
             "Para la corrida cuando Cl y Cd dejan de moverse por encima de\n"
             "esta tolerancia absoluta (5e-4 = media unidad del tercer decimal).\n"
             "En 0 se corren los pasos pedidos, que es lo que hay que hacer en\n"
             "un estudio de malla."),
        ]),
        ("Backend", [
            ("backend", "Backend", "c",
             "numpy = CPU, cupy = GPU.\n"
             "Medido: con 21k celdas los kernels fundidos dan x15.9 en GPU.",
             ("numpy", "cupy")),
            ("dtype", "Precision", "c",
             "En la 3070 Ti fp64 va a 1/64 de fp32, asi que float32 es la unica "
             "opcion util en GPU: en float64 la GPU solo gana x4.\n"
             "Medido: float64 baja Cl 0.96 % y Cd 0.71 %, irrelevante frente al "
             "error de malla.",
             ("float32", "float64")),
            ("salida", "Salida", "c",
             "monitor = campos a la vista en vivo; grabar = ademas a disco; "
             "ninguno = solo fuerzas.",
             ("monitor", "grabar", "ninguno")),
        ]),
    ]),
]

# Orden de pestanas en el que aparece el perfil, que vive fuera de ellas.
PERFIL = ("perfil", "Perfil .dat", "s",
          "Fichero de geometria en formato Selig. No se escala, no se rota y no "
          "se recorta el borde de salida.")


def parametros():
    """Todos los parametros en plano, como (clave, etiqueta, tipo, ayuda, opciones)."""
    for _, bloques in PESTANAS:
        for _, entradas in bloques:
            for entrada in entradas:
                clave, etiqueta, tipo, ayuda, *resto = entrada
                yield clave, etiqueta, tipo, ayuda, (resto[0] if resto else None)


def defectos() -> dict[str, object]:
    """Defaults planos para construir formularios y casos nuevos."""
    from .caso import Caso

    caso = Caso()
    resultado = {"perfil": caso.perfil}
    resultado.update(caso.malla)
    resultado.update(caso.solver)
    resultado.update(caso.ejecucion)
    return resultado

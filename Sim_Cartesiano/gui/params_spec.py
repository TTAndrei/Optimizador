"""Que parametros de main() se exponen en la GUI, agrupados y con tipo.

Declarativo a proposito: `main()` tiene ~200 argumentos y la GUI vieja cableaba
55 a mano, con lo que se quedo desactualizada en cuanto el solver crecio. Aqui
se anade una linea por parametro nuevo y el formulario se regenera solo.

Cada entrada: (clave, etiqueta, tipo, defecto, ayuda).
tipo: "f" float, "i" int, "b" bool, "s" texto, ("c", [opciones]) desplegable.
"""

GRUPOS = [
 ("Flujo", [
  ("v0x", "Velocidad U∞ [m/s]", "f", 1.0, "Velocidad de la corriente libre."),
  ("alpha_deg", "Ángulo de ataque [°]", "f", 4.0, "Rota la geometría, no el flujo."),
  ("rho", "Densidad ρ", "f", 1.0, "Con ρ=1, U=1 y c=1, ν=1/Re."),
  ("nu", "Viscosidad ν", "f", 1e-5, "ν = U·c/Re. 1e-5 es Re=1e5."),
  ("chord", "Longitud de referencia c", "f", 1.0,
   "Normaliza Cl y Cd. Con un conducto no significa nada: mira la fuerza."),
 ]),
 ("Malla", [
  ("dx_min", "dx mínimo [m]", "f", 0.004,
   "Manda en todo: el coste va con 1/dx², y dt con dx² por el límite viscoso."),
  ("dy_min", "dy mínimo [m]", "f", None,
   "Vacío = igual que dx. Separarlos sirve para resolver una capa límite sin "
   "pagar la misma resolución a lo largo del flujo, pero celdas muy planas "
   "junto a una pared hacen que el multigrid deje de converger."),
  ("factor_expansion", "Factor de expansión", "f", 1.1,
   "Crecimiento de celda fuera de la zona fina. Por encima de 1.2 el "
   "multigrid se degrada."),
  ("ancho_zona_fina_x", "Ancho zona fina X", "f", 1.5, ""),
  ("ancho_zona_fina_y", "Ancho zona fina Y", "f", 1.0, ""),
  ("ratio_max_malla", "Ratio máximo dx_max/dx_min", "i", 50, ""),
  ("wake_refinement_mode", "Refinado de estela", ("c", ["base", "long_fine_x"]),
   "long_fine_x", "long_fine_x alarga la banda fina aguas abajo."),
 ]),
 ("Física del solver", [
  ("turb_model", "Turbulencia", ("c", ["none", "wale", "sa"]), "sa",
   "SA es la configuración validada. Con contorno exterior importado NO usar "
   "SA: la distancia a pared sale mal en malla estirada."),
  ("transition_model", "Transición", ("c", ["none", "sa_bc"]), "sa_bc",
   "Intermitencia algebraica de Bas-Cakmakcıoğlu. Requiere SA."),
  ("freestream_Tu", "Turbulencia de corriente libre Tu [%]", "f", 0.1, ""),
  ("advection_scheme", "Advección", ("c", ["sl", "maccormack"]), "maccormack",
   "maccormack corrige la difusión numérica del semi-lagrangiano bilineal."),
  ("wall_treatment", "Tratamiento de pared", ("c", ["legacy", "consistent"]),
   "consistent", "consistent = flujo por caras + gradiente one-sided."),
  ("ibm_wall_mode", "Modo de pared IBM",
   ("c", ["ghost_noslip", "slip_only", "solid_zero_only"]), "ghost_noslip", ""),
  ("CFL", "CFL", "f", 0.5, "El dt real es el mínimo entre CFL y el límite viscoso."),
 ]),
 ("Proyección", [
  ("mg_max_outer", "Iteraciones externas", "i", 2,
   "2×3 es el presupuesto validado. Por debajo, degrada monótonamente."),
  ("mg_cycles_per_outer", "Ciclos V por externa", "i", 3, ""),
  ("mg_niveles_max", "Niveles de multigrid", "i", 2,
   "Más niveles EMPEORAN con malla anisótropa. Con conductos estrechos, el "
   "engrosado 3×3 puede cerrar el canal."),
  ("divergencia", "Tolerancia de divergencia", "f", 0.02, ""),
 ]),
 ("Presupuesto y parada", [
  ("iteraciones", "Iteraciones", "i", 13000, ""),
  ("guardado", "Muestrear cada", "i", 50,
   "Cada cuántas iteraciones se calculan Cl/Cd y se publica la vista en vivo."),
  ("stop_on_clcd_convergence", "Parar cuando Cl/Cd converjan", "b", False,
   "Desactivar en estudios de malla: corta a tiempo físico distinto en cada "
   "malla y contamina la comparación."),
  ("stop_on_convergence", "Parar en estado estacionario", "b", False, ""),
 ]),
]

# Parametros que la GUI gestiona por su cuenta (geometria, fronteras, salida) y
# que por eso NO salen en el formulario.
GESTIONADOS = {
    "Lx", "Ly", "cx", "cy", "filepath", "escena",
    "boundary_left", "boundary_right", "boundary_top", "boundary_bottom",
    "save_frames", "frames_dir_grueso", "frames_dir_refinado",
    "frames_dir_vorticidad", "dump_fields_dir", "dump_fields_cada",
    "dump_fields_max_nx", "shm_publish", "shm_cada", "live_view",
    "graficos", "mostrar_malla",
}


def defectos():
    return {k: d for _, ps in GRUPOS for (k, _e, _t, d, _a) in ps}

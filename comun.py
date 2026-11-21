"""Funciones compartidas por exacto.py, columnas.py y dos_viajes.py."""

import csv
import math
import os

import pyomo.environ as pyo

COLUMNAS_CSV = ["instancia", "metodo", "var_continuas", "var_enteras", "restricciones", "objetivo",
                "cota_inferior", "gap", "tiempo_s", "estado", "iteraciones", "rutas_generadas", "lb_a", "lb_b"]


def guardar_resultado(fila):
    """Agrega una fila a resultados.csv (lo crea con encabezado si no existe)."""
    nuevo = not os.path.exists("resultados.csv")
    with open("resultados.csv", "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS_CSV, restval="")
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def contar(modelo):
    """Devuelve (variables continuas, variables enteras/binarias, restricciones)."""
    variables = list(modelo.component_data_objects(pyo.Var))
    continuas = sum(1 for v in variables if v.is_continuous())
    restricciones = sum(1 for _ in modelo.component_data_objects(pyo.Constraint, active=True))
    return continuas, len(variables) - continuas, restricciones


def leer_mip(res):
    """Del resultado de HiGHS devuelve (objetivo, cota, estado). objetivo es None si no hay solución."""
    objetivo = res.problem.upper_bound
    if objetivo is None or not math.isfinite(objetivo):
        objetivo = None
    return objetivo, res.problem.lower_bound, str(res.solver.termination_condition)


def max_viajes_tipo2(d):
    """Valor de N (Big M) para (7) y (10) de F1 y (34), (37) y (38) de P1. El paper no lo da.

    Toda ruta tiene un viaje tipo 1 (dura al menos min t1) y cada viaje tipo 2 dura al menos
    min t2, sin contar la recolección. Por (5), en un día entran a lo sumo
    floor((T - min t1) / min t2) viajes tipo 2, así que y2_vmd nunca supera ese valor y N es
    lo bastante grande para que (7) y (10) no corten soluciones factibles.
    """
    return math.floor((d["T"] - min(d["t1"].values())) / min(d["t2"].values()))


def nueva_ruta(d, p, a, visitados, dia, tipo1, x1, y2, x2):
    """Arma una ruta con los parámetros del paper: p_k, a_km, g_kmd y h_kd.

    tipo1, x1, y2 y x2 son el detalle de los viajes, solo para imprimir la solución.
    """
    D = range(1, d["D"] + 1)
    return {"p": p,
            "a": {i: a.get(i, 0) for i in d["M"]},
            "g": {(i, t): 1 if (i in visitados and t == dia) else 0 for i in d["M"] for t in D},
            "h": {t: 1 if t == dia else 0 for t in D},
            "tipo1": tipo1, "x1": x1, "y2": y2, "x2": x2}


def resolver_im1(d, rutas, limite, gap=None):
    """IM1: maestro con variables enteras y solo las rutas dadas. limite en segundos; gap es la
    tolerancia relativa del MIP (None = la de HiGHS por defecto). Lo usan CG y dos viajes.

    Como en la sección 5.1, cada ruta puede usarse en cualquier día: r_vkd = 1 si el vehículo v
    hace la ruta k el día d. Entonces h_kd vale 1 en el día en que se hace, y g_kmd vale 1 ese día
    para los clientes que visita la ruta (visita[k, m] = 1 si g_kmd = 1 en su día original).
    """
    N = d["V"]  # N en (19) = |V|
    visita = {(k, i): sum(r["g"][i, t] for t in range(1, d["D"] + 1)) for k, r in enumerate(rutas) for i in d["M"]}
    m = pyo.ConcreteModel()
    m.V = pyo.RangeSet(d["V"])
    m.K = pyo.Set(initialize=range(len(rutas)))
    m.M = pyo.Set(initialize=d["M"])
    m.D = pyo.RangeSet(d["D"])

    m.r = pyo.Var(m.V, m.K, m.D, within=pyo.Binary)   # (23) entera
    m.w = pyo.Var(m.M, m.D, within=pyo.Binary)        # (24) entera
    m.z = pyo.Var(within=pyo.NonNegativeIntegers)     # (25) entera

    # (17)
    m.obj = pyo.Objective(expr=d["cveh"] * m.z + sum(
        rutas[k]["p"] * m.r[v, k, t] for v in m.V for k in m.K for t in m.D))
    # (18)
    m.r18 = pyo.Constraint(m.M, rule=lambda m, i: sum(
        rutas[k]["a"][i] * m.r[v, k, t] for v in m.V for k in m.K for t in m.D) >= d["Q"][i])
    # (19) El paper no escribe IM1 con días libres: la sección 4.1 lo define como M1 con r_vk entera
    # (días fijos) y la 5.1 solo dice en palabras que cada ruta puede usarse en cualquier día. Para
    # ese caso, g_kmd tiene que valer 1 el día en que se hace la ruta y no en su día original, así
    # que se usa visita[k, m] (1 si la ruta k visita a m) en lugar de g_kmd. Es interpretación nuestra.
    m.r19 = pyo.Constraint(m.M, m.D, rule=lambda m, i, t: sum(
        visita[k, i] * m.r[v, k, t] for v in m.V for k in m.K) <= N * m.w[i, t])
    # (20)
    m.r20 = pyo.Constraint(m.M, rule=lambda m, i: sum(m.w[i, t] for t in m.D) <= d["W"])
    # (21)
    m.r21 = pyo.Constraint(m.V, m.D, rule=lambda m, v, t: sum(m.r[v, k, t] for k in m.K) <= 1)
    # (22)
    m.r22 = pyo.Constraint(m.D, rule=lambda m, t: sum(m.r[v, k, t] for v in m.V for k in m.K) <= m.z)

    opciones = {"mip_rel_gap": gap} if gap is not None else None
    res = pyo.SolverFactory("highs").solve(m, timelimit=limite, options=opciones, load_solutions=False)
    return m, res


def describir(ruta):
    if ruta["tipo1"] is None:
        return "super ruta (recoge toda la demanda)"
    viajes = [f"tipo 1 a {ruta['tipo1']} ({ruta['x1']:.2f} t)"]
    viajes += [f"{n} x tipo 2 a {i} ({ruta['x2'][i]:.2f} t)" for i, n in ruta["y2"].items()]
    return ", ".join(viajes)


def imprimir_solucion_rutas(m, rutas):
    """Imprime la solución de IM1 (o del MIP de dos viajes): qué ruta hace cada vehículo cada día."""
    print("\nSolución (por día y vehículo):")
    for t in m.D:
        print(f"Día {t}:")
        for v in m.V:
            for k in m.K:
                if pyo.value(m.r[v, k, t]) > 0.5:
                    print(f"  vehículo {v}: {describir(rutas[k])}")

"""Modelo MIP exacto F1 (sección 3 del paper). |V| se toma de datos.py."""

import time

import pyomo.environ as pyo

from comun import contar, guardar_resultado, leer_mip, max_viajes_tipo2
from datos import cargar


def construir_f1(d, num_vehiculos, N):
    m = pyo.ConcreteModel()
    m.V = pyo.RangeSet(num_vehiculos)
    m.M = pyo.Set(initialize=d["M"])
    m.D = pyo.RangeSet(d["D"])

    m.y1 = pyo.Var(m.V, m.M, m.D, within=pyo.Binary)                # (12)
    m.y2 = pyo.Var(m.V, m.M, m.D, within=pyo.NonNegativeIntegers)   # (13)
    m.x1 = pyo.Var(m.V, m.M, m.D, within=pyo.NonNegativeReals)      # (14)
    m.x2 = pyo.Var(m.V, m.M, m.D, within=pyo.NonNegativeReals)      # (14)
    m.w = pyo.Var(m.M, m.D, within=pyo.Binary)                      # (15)
    m.z = pyo.Var(within=pyo.NonNegativeIntegers)                   # (16)

    # (1) 
    m.obj = pyo.Objective(expr=d["cveh"] * m.z + sum(
        d["ch"] * (d["t1"][i] * m.y1[v, i, t] + d["t2"][i] * m.y2[v, i, t])
        for v in m.V for i in m.M for t in m.D))
    # (2) 
    m.r2 = pyo.Constraint(m.V, m.M, m.D, rule=lambda m, v, i, t: m.x1[v, i, t] <= d["L"] * m.y1[v, i, t])
    # (3) 
    m.r3 = pyo.Constraint(m.V, m.M, m.D, rule=lambda m, v, i, t: m.x2[v, i, t] <= d["L"] * m.y2[v, i, t])
    # (4) 
    m.r4 = pyo.Constraint(m.M, rule=lambda m, i: sum(
        m.x1[v, i, t] + m.x2[v, i, t] for v in m.V for t in m.D) == d["Q"][i])
    # (5)
    m.r5 = pyo.Constraint(m.V, m.D, rule=lambda m, v, t: sum(
        d["s"][i] * (m.x1[v, i, t] + m.x2[v, i, t]) + d["t1"][i] * m.y1[v, i, t] + d["t2"][i] * m.y2[v, i, t]
        for i in m.M) <= d["T"])
    # (6) 
    m.r6 = pyo.Constraint(m.V, m.D, rule=lambda m, v, t: sum(m.y1[v, i, t] for i in m.M) <= 1)
    # (7) 
    m.r7 = pyo.Constraint(m.V, m.M, m.D, rule=lambda m, v, i, t: N * sum(m.y1[v, j, t] for j in m.M) >= m.y2[v, i, t])
    # (8)
    m.r8 = pyo.Constraint(m.D, rule=lambda m, t: m.z >= sum(m.y1[v, i, t] for v in m.V for i in m.M))
    # (9) 
    m.r9 = pyo.Constraint(m.V, m.M, m.D, rule=lambda m, v, i, t: m.w[i, t] >= m.y1[v, i, t])
    # (10)
    m.r10 = pyo.Constraint(m.V, m.M, m.D, rule=lambda m, v, i, t: N * m.w[i, t] >= m.y2[v, i, t])
    # (11)
    m.r11 = pyo.Constraint(m.M, rule=lambda m, i: sum(m.w[i, t] for t in m.D) <= d["W"])
    return m


def imprimir_solucion(m):
    print("\nSolución (por día y vehículo):")
    for t in m.D:
        print(f"Día {t}:")
        for v in m.V:
            viajes = []
            for i in m.M:
                if pyo.value(m.y1[v, i, t]) > 0.5:
                    viajes.append(f"tipo 1 a {i} ({pyo.value(m.x1[v, i, t]):.2f} t)")
            for i in m.M:
                n2 = round(pyo.value(m.y2[v, i, t]))
                if n2 > 0:
                    viajes.append(f"{n2} x tipo 2 a {i} ({pyo.value(m.x2[v, i, t]):.2f} t)")
            if viajes:
                print(f"  vehículo {v}: " + ", ".join(viajes))


def correr_f1(instancia, limite):
    """Resuelve F1 con un límite de tiempo en segundos."""
    d = cargar(instancia)
    num_vehiculos = d["V"]

    N = max_viajes_tipo2(d)
    m = construir_f1(d, num_vehiculos, N)
    continuas, enteras, restricciones = contar(m)

    inicio = time.perf_counter()
    res = pyo.SolverFactory("highs").solve(m, timelimit=limite, load_solutions=False)
    tiempo = time.perf_counter() - inicio

    objetivo, cota, estado = leer_mip(res)
    gap = (objetivo - cota) / objetivo if objetivo is not None else None

    print(f"\nF1 {instancia}  (|V|={num_vehiculos}, N={N})")
    print(f"Variables continuas: {continuas}, enteras/binarias: {enteras}, restricciones: {restricciones}")
    print(f"Estado: {estado}")
    print(f"Objetivo: {objetivo:.2f}" if objetivo is not None else "Objetivo: sin solución factible")
    print(f"Cota inferior: {cota:.2f}")
    print(f"Gap: {gap:.2%}" if gap is not None else "Gap: -")
    print(f"Tiempo: {tiempo:.1f} s")

    if objetivo is not None:
        m.solutions.load_from(res)
        imprimir_solucion(m)

    guardar_resultado({
        "instancia": instancia, "metodo": "F1", "var_continuas": continuas, "var_enteras": enteras,
        "restricciones": restricciones, "objetivo": objetivo if objetivo is not None else "",
        "cota_inferior": cota, "gap": gap if gap is not None else "", "tiempo_s": round(tiempo, 2), "estado": estado,
    })

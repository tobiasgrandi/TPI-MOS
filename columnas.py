"""Generación de columnas (secciones 4.1 a 4.3 del paper): maestro M1, pricing P1, IM1 y cotas LB A y LB B.

Los duales que devuelve HiGHS son d(objetivo)/d(lado derecho), así que con las restricciones escritas
como en el paper quedan lambda >= 0 y gamma, tau, eta <= 0, y (26) se usa tal cual.
"""

import math
import time

import pyomo.environ as pyo

from comun import (contar, guardar_resultado, imprimir_solucion_rutas, leer_mip, max_viajes_tipo2,
                   nueva_ruta, resolver_im1)
from datos import cargar

COSTO_SUPER = 1e6    # costo de la super ruta (4.2)
TOL = 1e-6           # una ruta entra al maestro si su costo reducido es < -TOL


def rutas_iniciales(d):
    """Rutas iniciales (4.2): una por cliente y por día con un solo viaje tipo 1, más la super ruta."""
    rutas = []
    for t in range(1, d["D"] + 1):
        for i in d["M"]:
            x1 = min(d["Q"][i], d["L"], (d["T"] - d["t1"][i]) / d["s"][i])
            rutas.append(nueva_ruta(d, d["ch"] * d["t1"][i], {i: x1}, [i], t, i, x1, {}, {}))
    # Super ruta: recoge toda la demanda, en el día 1, con costo muy alto
    rutas.append(nueva_ruta(d, COSTO_SUPER, d["Q"], d["M"], 1, None, 0, {}, {}))
    return rutas


def resolver_maestro(d, rutas, objetivo):
    """Maestro LP M1 (17)-(25). objetivo: "M1" usa (17), "M1a" minimiza z y "M1b" el costo de las rutas.

    Devuelve el valor del maestro y los duales de (18), (19), (21) y (22).
    """
    N = d["V"]  # N en (19) = |V|
    m = pyo.ConcreteModel()
    m.V = pyo.RangeSet(d["V"])
    m.K = pyo.Set(initialize=range(len(rutas)))
    m.M = pyo.Set(initialize=d["M"])
    m.D = pyo.RangeSet(d["D"])

    m.r = pyo.Var(m.V, m.K, bounds=(0, 1))       # (23)
    m.w = pyo.Var(m.M, m.D, bounds=(0, 1))       # (24)
    m.z = pyo.Var(within=pyo.NonNegativeReals)   # (25)

    costo_rutas = sum(rutas[k]["p"] * m.r[v, k] for v in m.V for k in m.K)
    if objetivo == "M1":      # (17)
        m.obj = pyo.Objective(expr=d["cveh"] * m.z + costo_rutas)
    elif objetivo == "M1a":   # (M1a)
        m.obj = pyo.Objective(expr=m.z)
    else:                     # (M1b)
        m.obj = pyo.Objective(expr=costo_rutas)

    # (18)
    m.r18 = pyo.Constraint(m.M, rule=lambda m, i: sum(
        rutas[k]["a"][i] * m.r[v, k] for v in m.V for k in m.K) >= d["Q"][i])
    # (19)
    m.r19 = pyo.Constraint(m.M, m.D, rule=lambda m, i, t: sum(
        rutas[k]["g"][i, t] * m.r[v, k] for v in m.V for k in m.K) <= N * m.w[i, t])
    # (20)
    m.r20 = pyo.Constraint(m.M, rule=lambda m, i: sum(m.w[i, t] for t in m.D) <= d["W"])
    # (21)
    m.r21 = pyo.Constraint(m.V, m.D, rule=lambda m, v, t: sum(
        rutas[k]["h"][t] * m.r[v, k] for k in m.K) <= 1)
    # (22)
    m.r22 = pyo.Constraint(m.D, rule=lambda m, t: sum(
        rutas[k]["h"][t] * m.r[v, k] for v in m.V for k in m.K) <= m.z)

    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    pyo.SolverFactory("highs").solve(m)

    lam = {i: m.dual[m.r18[i]] for i in m.M}
    gam = {(i, t): m.dual[m.r19[i, t]] for i in m.M for t in m.D}
    tau = {(v, t): m.dual[m.r21[v, t]] for v in m.V for t in m.D}
    eta = {t: m.dual[m.r22[t]] for t in m.D}
    return pyo.value(m.obj), (lam, gam, tau, eta)


def resolver_pricing(d, duales, objetivo, N):
    """Pricing P1 (26)-(45). Devuelve el costo reducido mínimo y la ruta correspondiente.

    Para M1a se saca el primer término de (26) (4.3). N en (34), (37) y (38) es el de F1.
    """
    lam, gam, tau, eta = duales
    p = pyo.ConcreteModel()
    p.M = pyo.Set(initialize=d["M"])
    p.D = pyo.RangeSet(d["D"])
    p.V = pyo.RangeSet(d["V"])

    p.y1 = pyo.Var(p.M, within=pyo.Binary)                 # (40)
    p.y2 = pyo.Var(p.M, within=pyo.NonNegativeIntegers)    # (41)
    p.q2 = pyo.Var(p.M, within=pyo.Binary)                 # (42)
    p.x1 = pyo.Var(p.M, within=pyo.NonNegativeReals)       # (43)
    p.x2 = pyo.Var(p.M, within=pyo.NonNegativeReals)       # (43)
    p.g = pyo.Var(p.M, p.D, within=pyo.Binary)             # (44)
    p.h = pyo.Var(p.D, within=pyo.Binary)                  # (45)

    costo = sum(d["ch"] * d["t1"][i] * p.y1[i] + d["ch"] * d["t2"][i] * p.y2[i] for i in p.M)
    # (26)
    p.obj = pyo.Objective(expr=(costo if objetivo != "M1a" else 0)
                          - sum(lam[i] * (p.x1[i] + p.x2[i]) for i in p.M)
                          - sum(gam[i, t] * p.g[i, t] for i in p.M for t in p.D)
                          - sum(tau[v, t] * p.h[t] for v in p.V for t in p.D)
                          - sum(eta[t] * p.h[t] for t in p.D))

    # (27)
    p.r27 = pyo.Constraint(p.M, rule=lambda p, i: p.x1[i] <= d["L"] * p.y1[i])
    # (28)
    p.r28 = pyo.Constraint(p.M, rule=lambda p, i: p.x2[i] <= d["L"] * p.y2[i])
    # (29)
    p.r29 = pyo.Constraint(p.M, rule=lambda p, i: p.x1[i] + p.x2[i] <= d["Q"][i])
    # (30)
    p.r30 = pyo.Constraint(expr=sum(
        d["s"][i] * (p.x1[i] + p.x2[i]) + d["t1"][i] * p.y1[i] + d["t2"][i] * p.y2[i] for i in p.M) <= d["T"])
    # (31)
    p.r31 = pyo.Constraint(expr=sum(p.y1[i] for i in p.M) == 1)
    # (32)
    p.r32 = pyo.Constraint(expr=sum(p.h[t] for t in p.D) == 1)
    # (33)
    p.r33 = pyo.Constraint(p.M, p.D, rule=lambda p, i, t: p.g[i, t] + 1 >= p.y1[i] + p.h[t])
    # (34)
    p.r34 = pyo.Constraint(p.M, p.D, rule=lambda p, i, t: N * p.g[i, t] + 1 >= p.y2[i] + p.h[t])
    # (35)
    p.r35 = pyo.Constraint(p.M, p.D, rule=lambda p, i, t: p.g[i, t] <= p.h[t])
    # (36)
    p.r36 = pyo.Constraint(p.M, p.D, rule=lambda p, i, t: p.g[i, t] <= p.y1[i] + p.y2[i])
    # (37)
    dif = {i: d["t_dep"][i] - d["t_disp"][i] for i in d["M"]}
    p.r37 = pyo.Constraint(p.M, p.M, rule=lambda p, i, j:
                           dif[i] - N * (1 - p.y1[i]) <= dif[j] + N * (1 - p.q2[j]))
    # (38)
    p.r38 = pyo.Constraint(p.M, rule=lambda p, i: N * p.q2[i] >= p.y2[i])
    # (39)
    p.r39 = pyo.Constraint(p.M, rule=lambda p, i: p.q2[i] <= p.y2[i])

    pyo.SolverFactory("highs").solve(p)

    # Pasaje de la solución a la nueva columna k: p_k, a_km, g_kmd y h_kd
    dia = next(t for t in p.D if pyo.value(p.h[t]) > 0.5)
    visitados = [i for i in p.M if pyo.value(p.g[i, dia]) > 0.5]
    a = {i: pyo.value(p.x1[i] + p.x2[i]) for i in p.M}
    tipo1 = next(i for i in p.M if pyo.value(p.y1[i]) > 0.5)
    y2 = {i: round(pyo.value(p.y2[i])) for i in p.M if pyo.value(p.y2[i]) > 0.5}
    ruta = nueva_ruta(d, pyo.value(costo), a, visitados, dia, tipo1, pyo.value(p.x1[tipo1]),
                      y2, {i: pyo.value(p.x2[i]) for i in y2})
    return pyo.value(p.obj), ruta


def generar_columnas(d, objetivo):
    """Generación de columnas con el maestro indicado ("M1", "M1a" o "M1b").

    Devuelve las rutas (iniciales + generadas), el valor del último maestro LP y las iteraciones.
    """
    rutas = rutas_iniciales(d)
    N = max_viajes_tipo2(d)
    iteraciones = 0
    while True:
        iteraciones += 1
        valor, duales = resolver_maestro(d, rutas, objetivo)
        reducido, ruta = resolver_pricing(d, duales, objetivo, N)
        print(f"  {objetivo} iteración {iteraciones}: maestro = {valor:.4f}, costo reducido = {reducido:.4f}")
        if reducido >= -TOL:
            return rutas, valor, iteraciones
        rutas.append(ruta)


def correr_cg(instancia, limite_im1, gap_im1):
    """Corre CG + IM1 y calcula LB A y LB B. Devuelve (lb_a, lb_b) para el gap de dos viajes.

    limite_im1 (segundos) y gap_im1 (tolerancia relativa) son los criterios de corte de IM1.
    """
    d = cargar(instancia)
    n_iniciales = len(rutas_iniciales(d))

    # Generación de columnas con M1 y solución entera IM1
    inicio = time.perf_counter()
    rutas, lb_a, iteraciones = generar_columnas(d, "M1")
    m, res = resolver_im1(d, rutas, limite_im1, gap_im1)
    tiempo = time.perf_counter() - inicio
    continuas, enteras, restricciones = contar(m)
    objetivo, _, estado = leer_mip(res)

    # LB B (4.3): ceil(z de M1a) * cveh + valor de M1b. El -TOL evita que un z = 6.0000001 redondee a 7.
    inicio_lb = time.perf_counter()
    _, z_a, _ = generar_columnas(d, "M1a")
    _, costo_b, _ = generar_columnas(d, "M1b")
    lb_b = d["cveh"] * math.ceil(z_a - TOL) + costo_b
    tiempo_lb = time.perf_counter() - inicio_lb

    cota = max(lb_a, lb_b)
    gap = (objetivo - cota) / objetivo if objetivo is not None else None

    print(f"\nCG {instancia}  (|V|={d['V']})")
    print(f"Iteraciones: {iteraciones}, rutas generadas: {len(rutas) - n_iniciales}")
    print(f"IM1: variables continuas: {continuas}, enteras/binarias: {enteras}, restricciones: {restricciones}")
    print(f"Estado IM1: {estado}")
    print(f"Objetivo: {objetivo:.2f}" if objetivo is not None else "Objetivo: sin solución factible")
    print(f"LB A: {lb_a:.2f}")
    print(f"LB B: {lb_b:.2f}  (z de M1a = {z_a:.4f}, costo de rutas de M1b = {costo_b:.2f})")
    print(f"Gap: {gap:.2%}" if gap is not None else "Gap: -")
    print(f"Tiempo CG + IM1: {tiempo:.1f} s (cálculo de LB B: {tiempo_lb:.1f} s)")

    if objetivo is not None:
        m.solutions.load_from(res)
        print(f"Vehículos: {round(pyo.value(m.z))}")
        imprimir_solucion_rutas(m, rutas)

    guardar_resultado({
        "instancia": instancia, "metodo": "CG", "var_continuas": continuas, "var_enteras": enteras,
        "restricciones": restricciones, "objetivo": objetivo if objetivo is not None else "",
        "cota_inferior": cota, "gap": gap if gap is not None else "", "tiempo_s": round(tiempo, 2),
        "estado": estado, "iteraciones": iteraciones, "rutas_generadas": len(rutas) - n_iniciales,
        "lb_a": lb_a, "lb_b": lb_b,
    })
    return lb_a, lb_b

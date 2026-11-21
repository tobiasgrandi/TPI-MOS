"""Heurística de dos viajes T1 (sección 4.4 del paper).

1. Se generan todas las rutas de uno o dos viajes con las fórmulas de la p. 14.
2. Un MIP igual a IM1 (ver resolver_im1 en comun.py) elige qué ruta hace cada vehículo cada día
   con N = |V| y visita_km = 1 si la ruta k visita al cliente m.
3. El gap se calcula con LB A y LB B de CG (sección 5.2): las de la misma ejecución si CG corrió
   antes, o si no las de la última corrida de CG para la instancia en resultados.csv.
"""

import csv
import os
import time

import pyomo.environ as pyo

from comun import contar, guardar_resultado, imprimir_solucion_rutas, leer_mip, nueva_ruta, resolver_im1
from datos import cargar


def generar_rutas(d):
    """Rutas de uno o dos viajes (4.4). El día no importa: el MIP elige en qué día se hace cada ruta."""
    T, L, Q, s, t1, t2, ch = d["T"], d["L"], d["Q"], d["s"], d["t1"], d["t2"], d["ch"]
    rutas = []

    # Un viaje: tipo 1 al cliente m
    for m in d["M"]:
        x1 = min(Q[m], L, (T - t1[m]) / s[m])
        rutas.append(nueva_ruta(d, ch * t1[m], {m: x1}, [m], 1, m, x1, {}, {}))

    # Dos viajes: tipo 1 a m y tipo 2 a m', con (t_dep_m - t_disp_m) <= (t_dep_m' - t_disp_m').
    # Se recorre cada par una sola vez (m' >= m por índice) y el tipo 1 va al cliente con menor
    # (t_dep - t_disp); si empatan, al de menor índice. El caso m = m' va literal aunque x1 + x2 > Q_m.
    dif = {m: d["t_dep"][m] - d["t_disp"][m] for m in d["M"]}
    for i in d["M"]:
        for j in d["M"]:
            if j < i:
                continue
            m, mp = (i, j) if dif[i] <= dif[j] else (j, i)
            costo = ch * (t1[m] + t2[mp])

            # Variante 1: primero se llena el viaje tipo 1
            x1 = min(Q[m], L, (T - t1[m] - t2[mp]) / s[m])
            x2 = min(Q[mp], L, (T - t1[m] - t2[mp] - x1 * s[m]) / s[mp])
            rutas.append(ruta_dos_viajes(d, costo, m, mp, x1, x2))

            # Variante 2: primero se llena el viaje tipo 2.
            # El paper divide por s_m' en x̄1_m; parece errata y se usa s_m.
            x2 = min(Q[mp], L, (T - t1[m] - t2[mp]) / s[mp])
            x1 = min(Q[m], L, (T - t1[m] - t2[mp] - x2 * s[mp]) / s[m])
            rutas.append(ruta_dos_viajes(d, costo, m, mp, x1, x2))
    return rutas


def ruta_dos_viajes(d, costo, m, mp, x1, x2):
    """Ruta con un viaje tipo 1 a m (recoge x1) y un viaje tipo 2 a mp (recoge x2)."""
    a = {m: x1}
    a[mp] = a.get(mp, 0) + x2  # si m = mp, a_m = x1 + x2
    return nueva_ruta(d, costo, a, [m, mp], 1, m, x1, {mp: 1}, {mp: x2})


def leer_cotas(instancia):
    """LB A y LB B de la última corrida de columnas.py para la instancia, o None si no hay."""
    if not os.path.exists("resultados.csv"):
        return None
    with open("resultados.csv", newline="", encoding="utf-8") as f:
        filas = [r for r in csv.DictReader(f) if r["instancia"] == instancia and r["metodo"] == "CG"]
    if not filas:
        return None
    return float(filas[-1]["lb_a"]), float(filas[-1]["lb_b"])


def correr_dos_viajes(instancia, limite, cotas=None):
    """Corre la heurística con un límite de tiempo en segundos.

    cotas = (LB A, LB B) de CG en la misma ejecución, o None para leerlas del CSV.
    """
    d = cargar(instancia)

    inicio = time.perf_counter()
    rutas = generar_rutas(d)
    print(f"Rutas generadas: {len(rutas)}")
    m, res = resolver_im1(d, rutas, limite)
    tiempo = time.perf_counter() - inicio
    continuas, enteras, restricciones = contar(m)
    objetivo, _, estado = leer_mip(res)

    origen = "de CG en esta ejecución"
    if cotas is None:
        cotas = leer_cotas(instancia)
        origen = "de resultados.csv"
    cota = max(cotas) if cotas else None
    gap = (objetivo - cota) / objetivo if objetivo is not None and cota is not None else None

    print(f"\nDos viajes {instancia}  (|V|={d['V']})")
    print(f"Variables continuas: {continuas}, enteras/binarias: {enteras}, restricciones: {restricciones}")
    print(f"Estado: {estado}")
    print(f"Objetivo: {objetivo:.2f}" if objetivo is not None else "Objetivo: sin solución factible")
    if cotas:
        print(f"LB A: {cotas[0]:.2f}, LB B: {cotas[1]:.2f} ({origen})")
    else:
        print("No corrió CG y no hay corrida de CG para esta instancia en resultados.csv: no se calcula el gap")
    print(f"Gap: {gap:.2%}" if gap is not None else "Gap: -")
    print(f"Tiempo: {tiempo:.1f} s")

    if objetivo is not None:
        m.solutions.load_from(res)
        print(f"Vehículos: {round(pyo.value(m.z))}")
        imprimir_solucion_rutas(m, rutas)

    guardar_resultado({
        "instancia": instancia, "metodo": "dos_viajes", "var_continuas": continuas, "var_enteras": enteras,
        "restricciones": restricciones, "objetivo": objetivo if objetivo is not None else "",
        "cota_inferior": cota if cota is not None else "", "gap": gap if gap is not None else "",
        "tiempo_s": round(tiempo, 2), "estado": estado,
        "lb_a": cotas[0] if cotas else "", "lb_b": cotas[1] if cotas else "",
    })

# TPI-MOS

Implementación en Python (Pyomo + HiGHS) de los métodos del paper *Tactical Waste Collection*:

- **F1**: modelo MIP exacto.
- **CG**: generación de columnas, con las cotas LB A y LB B.
- **dos_viajes**: heurística de dos viajes.

## Instalación

Hace falta [uv](https://docs.astral.sh/uv/). Desde la carpeta del proyecto:

```
uv sync
```

## Uso

1. Abrir `main.py` y elegir qué correr en las variables de arriba:

   ```python
   INSTANCIA = ["PMDa"]                   # lista con "PMDa", "PMDb" y/o "PCa"
   METODOS = ["F1", "CG", "dos_viajes"]   # se corren en este orden

   LIMITE_F1 = ...          # segundos
   LIMITE_IM1 = ...         # segundos
   GAP_IM1 = ...            # tolerancia de gap de IM1
   LIMITE_DOS_VIAJES = ...  # segundos
   ```

2. Correr:

   ```
   uv run main.py
   ```

Para que dos_viajes calcule su gap necesita las cotas de CG. Si CG está en `METODOS`, conviene dejarlo antes que dos_viajes. Si no está, dos_viajes busca la última corrida de CG de esa instancia en `resultados.csv`. Si no encuentra ninguna, deja el gap vacío.

## Resultados

- **En pantalla**: tamaño del modelo, estado, objetivo, cotas, gap, tiempo y la solución día por día.
- **En `resultados.csv`**: cada corrida agrega una fila.

## Archivos

| Archivo | Qué tiene |
|---|---|
| `main.py` | Configuración y ejecución |
| `datos.py` | Datos de las instancias |
| `exacto.py` | Modelo F1 |
| `columnas.py` | Generación de columnas |
| `dos_viajes.py` | Heurística de dos viajes |
| `comun.py` | Funciones compartidas |

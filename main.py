from columnas import correr_cg
from dos_viajes import correr_dos_viajes
from exacto import correr_f1

INSTANCIA = ["PMDb", "PCa"]                   # "PMDa", "PMDb" y/o "PCa"
METODOS = ["F1", "CG", "dos_viajes"]   # se corren en este orden; CG antes que dos_viajes para el gap

LIMITE_F1 = 3600            # segundos
LIMITE_IM1 = 120            # segundos (sección 5.1)
GAP_IM1 = 0.004             # 0,4 % (sección 5.1)
LIMITE_DOS_VIAJES = 600     # segundos (sección 5.2)


def main():
    for instancia in INSTANCIA:
        cotas = None
        for metodo in METODOS:
            print(f"\n========== {metodo} - {instancia} ==========")
            if metodo == "F1":
                correr_f1(instancia, LIMITE_F1)
            elif metodo == "CG":
                cotas = correr_cg(instancia, LIMITE_IM1, GAP_IM1)
            elif metodo == "dos_viajes":
                correr_dos_viajes(instancia, LIMITE_DOS_VIAJES, cotas)


if __name__ == "__main__":
    main()

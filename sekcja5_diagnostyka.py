# -*- coding: utf-8 -*-
"""
SEKCJA 5: DIAGNOSTYKA IMPUTACJI + PODSUMOWANIE                        OSOBA 5
Punkt 5 polecenia (miara diagnostyczna z uzasadnieniem) + tabela porównawcza z punktu 4.
Osoba 5 składa też raport (.html/.pdf) i wydruk do oddania.

jak w wykładzie (zab_cz_1): s. 153-155 (gamma, lambda, efektywność; wzory 46-48),
s. 87 (porównanie wyników przy różnych założeniach)
kalibracja w decylach: gains_table jak w kodzie "DM_Approach_py (1).ipynb"

Uruchomienie samej sekcji:  python sekcja5_diagnostyka.py  (wymaga gotowych sekcji 2-4)
"""
import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401

import sekcja2_mechanizm
import sekcja3_modele
import sekcja4_innowacje
from wspolne import M_IMP, wczytaj_dane  # noqa: F401


def uruchom(df, wyniki1, wyniki2, wyniki3, wyniki4):
    """TODO (osoba 5):
      - z wyniki3["kontrast_imputacje"] (oraz dla parametrów):
          gamma  = (1+1/M)*B / W                      wzrost wariancji     jak w wykładzie (zab_cz_1), wzór 46
          lambda = (gamma + 2/(df+3)) / (gamma + 1)   brakująca informacja jak w wykładzie (zab_cz_1), wzór 47
          RE     = (1 + lambda/M)^(-1)                efektywność          jak w wykładzie (zab_cz_1), wzór 48
      - uzasadnienie wyboru lambda kontrastu jako miary (porównać z % braków Y)
      - (dodatkowo) kalibracja modelu imputacyjnego w decylach
      - tabela porównawcza: wynik_cc vs wynik_mi vs wynik_ipw (OR, CI, p) + wykres
    """
    print("\n=== 5. DIAGNOSTYKA I PODSUMOWANIE ===")
    wyniki = {}
    # ... kod osoby 5 ...
    return wyniki


if __name__ == "__main__":
    df = wczytaj_dane()
    w2 = sekcja2_mechanizm.uruchom(df)
    w3 = sekcja3_modele.uruchom(df, w2.get("zmienne_imputacji"))
    w4 = sekcja4_innowacje.uruchom(df, w2, w3)
    uruchom(df, {}, w2, w3, w4)

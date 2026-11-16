# -*- coding: utf-8 -*-
"""
SEKCJA 1: STRUKTURA DANYCH                                            OSOBA 1
Punkt 1 polecenia: analiza struktury (cechy ciągłe i skokowe), graficzna ocena wyników badania,
zakres i rozkład braków danych.

jak w wykładzie (zab_cz_1), s. 17-18 (histogram, wykres pudełkowy), s. 46 (skale pomiarowe),
s. 73-74 (macierz braków), s. 89-90 (wzorzec braków)
wykresy (sns.histplot) jak w kodzie "DM_Approach_py (1).ipynb"

Uruchomienie samej sekcji:  python sekcja1_struktura.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from wspolne import OUT_DIR, wczytaj_dane  # noqa: E402,F401


def uruchom(df):
    """TODO (osoba 1):
      - podział cech: ciągła (age) i skokowe (education, house_loan, poutcome, contact, y_newdeposit)
      - age: statystyki opisowe, histogram, boxplot (także wg M)
      - cechy skokowe: tabele częstości z brakami (value_counts(dropna=False)), wykresy słupkowe
      - wynik badania: tabela contact x y_newdeposit (z brakami), % zakupów wg kanału + 95% CI
      - zakres braków: liczba i % braków w każdej zmiennej; wzorzec braków (monotoniczny?)
      - wykresy zapisujemy do OUT_DIR / "fig1_....png", tabele drukujemy print(...)

    ZWRACA słownik (dla osoby 5, do raportu), klucze:
      tab_struktura_age, tab_czestosci (dict: zmienna -> DataFrame), tab_wynik_kanal, tab_braki
    """
    print("\n=== 1. STRUKTURA DANYCH ===")
    wyniki = {}
    # ... kod osoby 1 ...
    return wyniki


if __name__ == "__main__":
    uruchom(wczytaj_dane())

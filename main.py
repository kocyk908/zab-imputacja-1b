# -*- coding: utf-8 -*-
"""
Zaawansowana analityka biznesowa, metody imputacji danych
Zadanie 1, wersja B: brak danych w zmiennej skokowej Y_NEWDEPOSIT (zbiór kb19)

Uruchamia wszystkie sekcje po kolei i zapisuje pełny wydruk do wyniki/wydruk.txt
(to jest "wydruk" z punktu c) wymagań do oddania).

    python main.py
"""
import sys

import sekcja1_struktura
import sekcja2_mechanizm
import sekcja3_modele
import sekcja4_innowacje
import sekcja5_diagnostyka
from wspolne import OUT_DIR, wczytaj_dane


class Tee:
    """Wypisuje jednocześnie na ekran i do pliku z wydrukiem."""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


if __name__ == "__main__":
    sys.stdout = Tee(OUT_DIR / "wydruk.txt")
    df = wczytaj_dane()
    w1 = sekcja1_struktura.uruchom(df)
    w2 = sekcja2_mechanizm.uruchom(df)
    w3 = sekcja3_modele.uruchom(df, w2.get("zmienne_imputacji"))
    w4 = sekcja4_innowacje.uruchom(df, w2, w3)
    w5 = sekcja5_diagnostyka.uruchom(df, w1, w2, w3, w4)
    print(f"\nGotowe. Wydruk: {OUT_DIR / 'wydruk.txt'}")

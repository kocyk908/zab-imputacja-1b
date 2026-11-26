# -*- coding: utf-8 -*-
"""
WSPÓLNE ustawienia dla wszystkich sekcji: lokalizacja robocza, wczytanie danych, model z zadania.
"""
from pathlib import Path

import pandas as pd

# lokalizacja robocza (zmień tylko WORK_DIR, jeśli dane leżą gdzie indziej)
WORK_DIR = Path(__file__).resolve().parent
DATA_FILE = WORK_DIR / "kb19.csv"
OUT_DIR = WORK_DIR / "wyniki"
OUT_DIR.mkdir(exist_ok=True)

SEED = 20261116   # wspólne ziarno losowania
ALPHA = 0.05
M_IMP = 100       # liczba imputacji


FORMULA = ("y_newdeposit ~ C(contact, Treatment(0)) + age"
           " + C(house_loan, Treatment('yes')) + C(poutcome, Treatment('unknown'))")
PARAM_A = "C(contact, Treatment(0))[T.1]"
PARAM_B = "C(contact, Treatment(0))[T.2]"


def wczytaj_dane():
    """Wczytuje kb19.csv i dodaje indykator braków M (M=1 -> brak Y)."""
    df = pd.read_csv(DATA_FILE)
    df["M"] = df["y_newdeposit"].isna().astype(int)
    return df

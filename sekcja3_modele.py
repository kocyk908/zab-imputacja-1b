# -*- coding: utf-8 -*-
"""
SEKCJA 3: MODEL LOGITOWY PRZY MCAR (CC) I MAR (MI)                    OSOBA 3
Punkt 3 polecenia: estymacja modelu przy MCAR i MAR + specyfikacja modelu imputacyjnego.

jak w wykładzie (zab_cz_1): s. 105-109 (CC), s. 132, 143-149 (MI), s. 151-155 (reguły Rubina)
model logitowy: smf.glm(..., family=sm.families.Binomial()) jak w kodzie "DM_Approach_py (1).ipynb"

Uruchomienie samej sekcji:  python sekcja3_modele.py
"""
import numpy as np  # noqa: F401
import statsmodels.api as sm  # noqa: F401
import statsmodels.formula.api as smf  # noqa: F401
from scipy import stats  # noqa: F401

from wspolne import FORMULA, M_IMP, PARAM_A, PARAM_B, SEED, wczytaj_dane  # noqa: F401

# Tymczasowo, dopóki osoba 2 nie poda swoich zmiennych (main.py podstawi wynik sekcji 2):
ZMIENNE_IMPUTACJI_TYMCZASOWE = ("C(contact) + age + C(house_loan) + C(poutcome) + C(education)")


def rubin(Q, U, n, k):
    """TODO: reguły Rubina dla jednego parametru, jak w wykładzie (zab_cz_1), s. 151-152, 155.
    Q - oszacowania z M imputacji, U - ich wariancje, n - liczba obserwacji, k - liczba parametrów.
    Zwraca dict: est, se, W, B, T, df (wzór 49), low, high, p."""
    raise NotImplementedError


def imputuj(df, zmienne_imputacji, rng, delta=0.0):
    """TODO: jedna imputacja Y_NEWDEPOSIT (wzorzec monotoniczny -> jedno równanie logistyczne,
    odpowiednik PROC MI; MONOTONE LOGISTIC):
      1. model logit y_newdeposit ~ zmienne_imputacji na obserwowanych Y
      2. losuj beta* ~ N(beta_hat, V_hat)        jak w wykładzie (zab_cz_1), s. 132, 144
      3. dla braków losuj Y* ~ Bernoulli(expit(x'beta* + delta))   jak w wykładzie (zab_cz_1), s. 132
         (delta = 0 to MAR; osoba 4 użyje delta != 0 do analizy wrażliwości: WŁASNE)
    Zwraca kopię df z uzupełnionym y_newdeposit."""
    raise NotImplementedError


def uruchom(df, zmienne_imputacji=None):
    """TODO (osoba 3):
      a) MCAR: smf.glm(FORMULA, ...) na kompletnych przypadkach; kontrast beta1 - beta2:
         est = b[PARAM_A] - b[PARAM_B], var = V_AA + V_BB - 2 V_AB, OR = exp(est), CI, p
      b) MAR: M_IMP razy imputuj(...) -> dopasuj FORMULA -> zapisz kontrast i jego wariancję
      c) połącz rubin(...) dla każdego parametru i dla kontrastu
      d) w raporcie: specyfikacja modelu imputacyjnego (zmienne, metoda, M, ziarno)

    ZWRACA słownik, klucze:
      wynik_cc = dict(est, se, OR, OR_low, OR_high, p)
      wynik_mi = dict(est, se, OR, OR_low, OR_high, p, W, B, T, df)
      tab_param_cc, tab_param_mi
      kontrast_imputacje = (Q, U)  - oszacowania i wariancje kontrastu z każdej imputacji (dla osoby 5)
    """
    zmienne_imputacji = zmienne_imputacji or ZMIENNE_IMPUTACJI_TYMCZASOWE
    print("\n=== 3. MODELE: MCAR (CC) i MAR (MI) ===")
    print("Model imputacyjny: y_newdeposit ~", zmienne_imputacji)
    wyniki = {}
    # ... kod osoby 3 ...
    return wyniki


if __name__ == "__main__":
    uruchom(wczytaj_dane())

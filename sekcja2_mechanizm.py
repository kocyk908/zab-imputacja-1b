# -*- coding: utf-8 -*-
"""
SEKCJA 2: MECHANIZM BRAKÓW DANYCH                                     OSOBA 2
Punkt 2 polecenia: zależność między prawdopodobieństwem braku a pozostałymi cechami;
które cechy pozwalają skorygować oszacowania przez imputację.

jak w wykładzie (zab_cz_1): s. 91-96 (MCAR/MAR/MNAR), s. 94 (metody Balickiego),
s. 103 (model skłonności do odpowiedzi)

Uruchomienie samej sekcji:  python sekcja2_mechanizm.py
"""
import statsmodels.api as sm  # noqa: F401
import statsmodels.formula.api as smf  # noqa: F401
from scipy import stats  # noqa: F401

from wspolne import wczytaj_dane


def uruchom(df):
    """TODO (osoba 2):
      - Metoda I (Balicki): porównanie średniego age dla M=0 i M=1 (test t)
      - Metoda II (Balicki): M vs cechy skokowe, test chi2 + miara asocjacji (V Cramera)
      - model logitowy P(M=1 | X): smf.glm("M ~ ...", data=df, family=sm.families.Binomial()).fit(),
        jak w kodzie "DM_Approach_py (1).ipynb" (sprawdzić nieliniowość wieku, np. I(age**2))
      - wniosek: czy MCAR odrzucamy; które cechy wolno użyć do korekty (imputacji)
      - czy education (spoza modelu z zadania) wiąże się z Y -> zmienna pomocnicza do imputacji

    ZWRACA słownik, klucze:
      tab_testy_brakow  - tabela testów
      model_brakow      - dopasowany model P(M=1|X)   (osoba 4 użyje go do IPW)
      zmienne_imputacji - prawa strona formuły modelu imputacyjnego (dla osoby 3), np.
                          "C(contact) + age + I(age**2) + C(house_loan) + C(poutcome) + C(education)"
    """
    print("\n=== 2. MECHANIZM BRAKÓW ===")
    wyniki = {"tab_testy_brakow": None, "model_brakow": None, "zmienne_imputacji": None}
    # ... kod osoby 2 ...
    return wyniki


if __name__ == "__main__":
    uruchom(wczytaj_dane())

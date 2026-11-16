# -*- coding: utf-8 -*-
"""
SEKCJA 4: INNOWACYJNOŚĆ: WRAŻLIWOŚĆ (MNAR), IPW, KONTROLA W SAS        OSOBA 4
Część punktu 4 polecenia + "aspekt innowacyjny" (2 pkt).

jak w wykładzie (zab_cz_1): s. 75, 87, 97, 156 (analiza wrażliwości, tipping point),
s. 101-103 (metody wykorzystujące wagi, propensity score)
implementacja delta-adjustment i IPW dla modelu logitowego: WŁASNE

Uruchomienie samej sekcji:  python sekcja4_innowacje.py  (wymaga gotowych sekcji 2 i 3)
"""
import numpy as np  # noqa: F401
import statsmodels.api as sm  # noqa: F401
import statsmodels.formula.api as smf  # noqa: F401

import sekcja2_mechanizm
import sekcja3_modele
from wspolne import FORMULA, SEED, wczytaj_dane  # noqa: F401


def uruchom(df, wyniki2, wyniki3):
    """TODO (osoba 4):
      - tipping point: sekcja3_modele.imputuj(..., delta=d) dla d w [-2, 2] (np. co 0.5),
        osobno dla braków w kanale A i w kanale B; łączenie przez sekcja3_modele.rubin;
        przy jakim delta zmienia się wniosek o A vs B? wykres OUT_DIR / "fig_wrazliwosc.png"
      - IPW: wagi 1 / P(M=0|X) z wyniki2["model_brakow"], ważony logit FORMULA na kompletnych
        przypadkach (odporne SE: cov_type="HC0")
      - (opcjonalnie) ten sam model w SAS: PROC MI (MONOTONE LOGISTIC) + PROC LOGISTIC
        + PROC MIANALYZE wg przykładu z Zadanie1_b.pdf, porównanie z Pythonem

    ZWRACA słownik, klucze:
      tab_wrazliwosc, wynik_ipw (ten sam format co wynik_cc z sekcji 3)
    """
    print("\n=== 4. WRAŻLIWOŚĆ (MNAR) i IPW ===")
    wyniki = {}
    # ... kod osoby 4 ...
    return wyniki


if __name__ == "__main__":
    df = wczytaj_dane()
    w2 = sekcja2_mechanizm.uruchom(df)
    w3 = sekcja3_modele.uruchom(df, w2.get("zmienne_imputacji"))
    uruchom(df, w2, w3)

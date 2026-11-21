# -*- coding: utf-8 -*-
"""
SEKCJA 2: MECHANIZM BRAKÓW DANYCH                                     OSOBA 2
Punkt 2 polecenia: zależność między prawdopodobieństwem braku a pozostałymi cechami;
które cechy pozwalają skorygować oszacowania przez imputację.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from wspolne import wczytaj_dane

def cramers_v(ct):
    """Oblicza miarę asocjacji V Craméra dla tablicy kontyngencji."""
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.sum().sum()
    return np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

def uruchom(df):
    print("\n=== 2. MECHANIZM BRAKÓW ===")
    wyniki = {"tab_testy_brakow": None, "model_brakow": None, "zmienne_imputacji": None}
    
    testy_lista = []

    # 1. Metoda I (Balicki) - zmienna ciągła (Test t-Studenta dla wariancji niejednorodnych/Welcha)
    age_m0 = df[df['M'] == 0]['age']
    age_m1 = df[df['M'] == 1]['age']
    t_stat, p_val_t = stats.ttest_ind(age_m0, age_m1, equal_var=False)
    
    testy_lista.append({
        'Zmienna': 'age', 
        'Test': 't-Studenta (Welcha)', 
        'Statystyka': round(t_stat, 4), 
        'p-value': p_val_t, 
        'V Cramera': None
    })
    print(f"Metoda I: Test t-Studenta dla 'age': t={t_stat:.4f}, p-val={p_val_t:.4e}")

    # 2. Metoda II (Balicki) - cechy skokowe (Chi-kwadrat i V Cramera)
    cechy_skokowe = ['education', 'house_loan', 'poutcome', 'contact']
    for col in cechy_skokowe:
        ct = pd.crosstab(df['M'], df[col])
        chi2, p_val_chi2, dof, ex = stats.chi2_contingency(ct)
        v = cramers_v(ct)
        testy_lista.append({
            'Zmienna': col, 
            'Test': 'Chi-kwadrat', 
            'Statystyka': round(chi2, 4), 
            'p-value': p_val_chi2, 
            'V Cramera': round(v, 4)
        })
        print(f"Metoda II: Test Chi2 dla '{col:10s}': chi2={chi2:8.2f}, p-val={p_val_chi2:.4e}, V={v:.4f}")

    wyniki["tab_testy_brakow"] = pd.DataFrame(testy_lista)

    # 3. Model logitowy skłonności do braków danych (propensity model dla M=1)
    print("\nEstymacja modelu logitowego P(M=1|X):")
    # Formuła bez składnika kwadratowego wieku
    formula_m = "M ~ age + C(education) + C(house_loan) + C(poutcome) + C(contact)"
    model_brakow = smf.glm(formula_m, data=df, family=sm.families.Binomial()).fit()
    print(model_brakow.summary())
    wyniki["model_brakow"] = model_brakow

    # 4. Sprawdzenie zmiennej pomocniczej 'education'
    ct_edy = pd.crosstab(df[df['M'] == 0]['y_newdeposit'], df[df['M'] == 0]['education'])
    chi2_edy, p_edy, _, _ = stats.chi2_contingency(ct_edy)
    print(f"\nTest wpływu zmiennej 'education' na target Y (tylko obserwacje pełne M=0): p-val = {p_edy:.4e}")

    # 5. Podsumowanie mechanizmu i definicja zmiennych do modelu imputacyjnego
    print("\nWNIOSEK:")
    print("Testy jednowymiarowe oraz model logitowy wykazują istotny wpływ ")
    print("wieku ('age') oraz kanału ('contact') na prawdopodobieństwo braku danych (M).")
    print("Stanowi to podstawę do odrzucenia założenia MCAR na rzecz mechanizmu MAR.")
    print("Zmienna 'education' istotnie determinuje samą zmienną 'y_newdeposit', stanowiąc")
    print("doskonałą zmienną pomocniczą do korekty oszacowań podczas imputacji.")
    
    # Składamy prawą stronę równania bez I(age**2)
    formula_imp = "C(contact) + age + C(house_loan) + C(poutcome) + C(education)"
    wyniki["zmienne_imputacji"] = formula_imp
    
    return wyniki


if __name__ == "__main__":
    df_test = wczytaj_dane()
    uruchom(df_test)
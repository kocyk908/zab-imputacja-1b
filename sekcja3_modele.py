# -*- coding: utf-8 -*-
import numpy as np  # noqa: F401
import statsmodels.api as sm  # noqa: F401
import statsmodels.formula.api as smf  # noqa: F401
from scipy import stats  # noqa: F401

from wspolne import FORMULA, M_IMP, PARAM_A, PARAM_B, SEED, wczytaj_dane  # noqa: F401

ZMIENNE_IMPUTACJI_TYMCZASOWE = (
    "C(contact) + age + C(house_loan) "
    "+ C(poutcome) + C(education)"
)

def rubin(Q, U, n, k):
    """
    Łączy wyniki imputacji regułami Rubina.

    Q - oszacowania parametru
    U - wariancje oszacowań
    n - liczba obserwacji
    k - liczba parametrów modelu
    """

    Q = np.asarray(Q, dtype=float)
    U = np.asarray(U, dtype=float)

    M = len(Q)

    if M < 2 or len(U) != M:
        raise ValueError("Potrzebne są co najmniej 2 pary Q, U.")

    if not np.all(np.isfinite(Q)) or not np.all(np.isfinite(U)):
        raise ValueError("Q i U muszą zawierać liczby skończone.")

    if np.any(U <= 0) or n <= k:
        raise ValueError("Wariancje muszą być dodatnie, a n > k.")

    est = np.mean(Q)

    W = np.mean(U)

    B = np.var(Q, ddof=1)

    T = W + (1 + 1 / M) * B

    se = np.sqrt(T)

    # Stopnie swobody Rubina
    if B > 0:
        r = (1 + 1 / M) * B / W
        v = (M - 1) * (1 + 1 / r)**2
    else:
        v = np.inf

    # Korekta stopni swobody
    vcom = n - k
    tau = (1 + 1 / M) * B / T

    vobs = (
        (1 - tau)
        * vcom
        * (vcom + 1)
        / (vcom + 3)
    )

    df = 1 / (1 / v + 1 / vobs)

    # Test t-Studenta
    t = est / se
    p = 2 * stats.t.sf(abs(t), df)

    # Przedział ufności 95%
    t_kryt = stats.t.ppf(0.975, df)

    low = est - t_kryt * se
    high = est + t_kryt * se

    # Wyniki
    return {
        "est": float(est),
        "se": float(se),
        "W": float(W),
        "B": float(B),
        "T": float(T),
        "df": float(df),
        "low": float(low),
        "high": float(high),
        "p": float(p)
    }


def imputuj(df, zmienne_imputacji, rng, delta=0.0):

    from patsy import build_design_matrices
    from scipy.special import expit

    dane = df.copy()

    braki = dane["y_newdeposit"].isna()

    if not braki.any():
        return dane

    obserwowane = dane.loc[~braki].copy()

    formula_imp = (
        "y_newdeposit ~ " + zmienne_imputacji
    )

    model_imp = smf.glm(
        formula=formula_imp,
        data=obserwowane,
        family=sm.families.Binomial()
    ).fit()

    beta = model_imp.params.to_numpy()

    V = model_imp.cov_params().to_numpy()

    beta_star = rng.multivariate_normal(
        mean=beta,
        cov=V
    )

   
    design_info = getattr(
        model_imp.model.data, "model_spec", None
    )

    if design_info is None:
        design_info = model_imp.model.data.design_info


    X_missing = build_design_matrices(
        [design_info],
        dane.loc[braki],
        return_type="dataframe"
    )[0]

    logit = X_missing.to_numpy() @ beta_star

    logit = logit + delta

    prawdopodobienstwa = expit(logit)

    y_star = rng.binomial(
        n=1,
        p=prawdopodobienstwa
    )

    dane.loc[braki, "y_newdeposit"] = y_star

    return dane



def uruchom(df, zmienne_imputacji=None):
    import pandas as pd

    if zmienne_imputacji is None:
        zmienne_imputacji = ZMIENNE_IMPUTACJI_TYMCZASOWE

    print("\n=== 3. MODELE CC I MI ===")

    df_cc = df.dropna(
        subset=["y_newdeposit"]
    ).copy()

    formula_podstawowa = (
        "y_newdeposit ~ " + zmienne_imputacji
    )

    # Rozszerzenie o kwadrat wieku
    skladnik_age2 = "I(age ** 2)"

    if skladnik_age2 in zmienne_imputacji:
        raise ValueError(
            "Formuła wejściowa zawiera już age². "
            "Przekaż podstawową formułę z sekcji 2."
        )

    formula_rozszerzona = (
        formula_podstawowa + " + " + skladnik_age2
    )

    # Dopasowanie modeli imputacyjnych
    model_podstawowy = smf.glm(
        formula=formula_podstawowa,
        data=df_cc,
        family=sm.families.Binomial()
    ).fit()

    model_rozszerzony = smf.glm(
        formula=formula_rozszerzona,
        data=df_cc,
        family=sm.families.Binomial()
    ).fit()

    # Kryterium AIC
    aic_1 = model_podstawowy.aic
    aic_2 = model_rozszerzony.aic

    # Test ilorazu wiarygodnosci LRT
    LR = 2 * (
        model_rozszerzony.llf
        - model_podstawowy.llf
    )

    df_diff = int(
        model_rozszerzony.df_model
        - model_podstawowy.df_model
    )

    p_LRT = stats.chi2.sf(LR, df_diff)

    print("\n--- WYBOR MODELU IMPUTACYJNEGO ---")
    print(f"AIC bez age²: {aic_1:.4f}")
    print(f"AIC z age²: {aic_2:.4f}")
    print(f"Statystyka LRT: {LR:.4f}")
    print(f"Stopnie swobody: {df_diff}")
    print(f"p-value LRT: {p_LRT:.6e}")

    # Przyjęta reguła wyboru
    if aic_2 < aic_1 and p_LRT < 0.05:
        zmienne_imputacji = (
            zmienne_imputacji + " + " + skladnik_age2
        )
        print("Wybrano model z age².")
    else:
        print("Wybrano model podstawowy.")

    print("\nModel imputacyjny: y_newdeposit ~",
          zmienne_imputacji)
    print("Liczba imputacji:", M_IMP)
    print("Seed:", SEED)

    # COMPLETE CASE (CC)

    model_cc = smf.glm(
        formula=FORMULA,
        data=df_cc,
        family=sm.families.Binomial()
    ).fit()

    b = model_cc.params
    V = model_cc.cov_params()

    est_cc = b[PARAM_A] - b[PARAM_B]

    var_cc = (
        V.loc[PARAM_A, PARAM_A]
        + V.loc[PARAM_B, PARAM_B]
        - 2 * V.loc[PARAM_A, PARAM_B]
    )

    se_cc = np.sqrt(var_cc)
    z_cc = est_cc / se_cc
    p_cc = 2 * stats.norm.sf(abs(z_cc))

    # Przedział ufności 95%
    low_cc = est_cc - 1.96 * se_cc
    high_cc = est_cc + 1.96 * se_cc

    wynik_cc = {
        "est": float(est_cc),
        "se": float(se_cc),
        "OR": float(np.exp(est_cc)),
        "OR_low": float(np.exp(low_cc)),
        "OR_high": float(np.exp(high_cc)),
        "p": float(p_cc)
    }

    tab_param_cc = model_cc.summary2().tables[1]

    print("\n--- WYNIK CC ---")
    print("Liczba obserwacji:", len(df_cc))
    print(f"OR A/B: {wynik_cc['OR']:.4f}")
    print(
        f"95% CI: ({wynik_cc['OR_low']:.4f}, "
        f"{wynik_cc['OR_high']:.4f})"
    )
    print(f"p-value: {wynik_cc['p']:.4f}")

    # MULTIPLE IMPUTATION (MAR)

    rng = np.random.default_rng(SEED)

    Q = []
    U = []

    param_Q = []
    param_U = []

    print(f"\nRozpoczynam {M_IMP} imputacji...")

    for i in range(M_IMP):

        df_imp = imputuj(
            df,
            zmienne_imputacji,
            rng,
            delta=0.0
        )

        model = smf.glm(
            formula=FORMULA,
            data=df_imp,
            family=sm.families.Binomial()
        ).fit()

        b_imp = model.params
        V_imp = model.cov_params()

        est_imp = b_imp[PARAM_A] - b_imp[PARAM_B]

        var_imp = (
            V_imp.loc[PARAM_A, PARAM_A]
            + V_imp.loc[PARAM_B, PARAM_B]
            - 2 * V_imp.loc[PARAM_A, PARAM_B]
        )

        Q.append(float(est_imp))
        U.append(float(var_imp))

        param_Q.append(b_imp)

        param_U.append(
            pd.Series(
                np.diag(V_imp),
                index=b_imp.index
            )
        )

        if (i + 1) % 10 == 0:
            print(f"Imputacja {i + 1}/{M_IMP}")

    Q = np.array(Q)
    U = np.array(U)

    # REGUŁY RUBINA

    wynik_rubin = rubin(
        Q,
        U,
        n=len(df),
        k=len(model.params)
    )

    wynik_mi = {
        **wynik_rubin,
        "OR": float(np.exp(wynik_rubin["est"])),
        "OR_low": float(np.exp(wynik_rubin["low"])),
        "OR_high": float(np.exp(wynik_rubin["high"]))
    }

    # TABELA PARAMETRÓW MI

    param_Q = pd.DataFrame(param_Q)
    param_U = pd.DataFrame(param_U)

    wiersze = []

    for nazwa in param_Q.columns:

        wynik_param = rubin(
            param_Q[nazwa].to_numpy(),
            param_U[nazwa].to_numpy(),
            n=len(df),
            k=len(model.params)
        )

        wiersze.append({
            "Parametr": nazwa,
            "Coef.": wynik_param["est"],
            "Std.Err.": wynik_param["se"],
            "P>|t|": wynik_param["p"],
            "[0.025": wynik_param["low"],
            "0.975]": wynik_param["high"],
            "OR": float(np.exp(wynik_param["est"]))
        })

    tab_param_mi = pd.DataFrame(
        wiersze
    ).set_index("Parametr")

    # WYŚWIETLANIE WYNIKÓW

    print("\n--- WYNIK MAR / MI ---")
    print(f"OR A/B: {wynik_mi['OR']:.4f}")
    print(
        f"95% CI: ({wynik_mi['OR_low']:.4f}, "
        f"{wynik_mi['OR_high']:.4f})"
    )
    print(f"p-value: {wynik_mi['p']:.4f}")

    print("\n--- PARAMETRY MI ---")
    print(tab_param_mi.round(4))


    wyniki = {
        "wynik_cc": wynik_cc,
        "wynik_mi": wynik_mi,
        "tab_param_cc": tab_param_cc,
        "tab_param_mi": tab_param_mi,
        "kontrast_imputacje": (Q, U),
        "model_imputacyjny": zmienne_imputacji,
        "wybor_modelu": {
            "AIC_podstawowy": float(aic_1),
            "AIC_age2": float(aic_2),
            "LRT": float(LR),
            "df_LRT": df_diff,
            "p_LRT": float(p_LRT)
        }
    }

    return wyniki

#if __name__ == "__main__":
#    uruchom(wczytaj_dane())



if __name__ == "__main__":

    import sekcja2_mechanizm

    df = wczytaj_dane()

    wyniki2 = sekcja2_mechanizm.uruchom(df)

    zmienne = wyniki2["zmienne_imputacji"]

    wyniki3 = uruchom(df, zmienne)

    wymagane = [
        "wynik_cc",
        "wynik_mi",
        "tab_param_cc",
        "tab_param_mi",
        "kontrast_imputacje"
    ]

    for klucz in wymagane:
        assert klucz in wyniki3, f"Brak: {klucz}"

    Q, U = wyniki3["kontrast_imputacje"]

    assert len(Q) == M_IMP
    assert len(U) == M_IMP

    print("\nSEKCJA 3: TEST ZAKONCZONY POMYSLNIE")


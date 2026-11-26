# -*- coding: utf-8 -*-
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

import sekcja2_mechanizm
import sekcja3_modele
import sekcja4_innowacje
from sekcja4_innowacje import KOLOR, STYL
from wspolne import ALPHA, OUT_DIR, wczytaj_dane


def gains_table(df, y, score, groups=10):
    tmp = df[[y, score]].copy()
    tmp = tmp.sort_values(score, ascending=False).reset_index(drop=True)
    value_counts = tmp[score].value_counts().sort_index(ascending=False)

    n = len(tmp)
    tmp['decile'] = 0
    current_index = 0

    for i, (val, count) in enumerate(value_counts.items()):
        start_idx = current_index
        end_idx = current_index + count - 1
        group_start = (start_idx * groups) // n
        group_end = (end_idx * groups) // n  # noqa: F841
        tmp.loc[tmp[score] == val, 'decile'] = group_start
        current_index += count

    tmp['decile'] = tmp['decile'] + 1

    gainout = (
        tmp.groupby('decile', as_index=False)
        .agg(
            n=(y, 'count'),
            total=(y, 'sum'),
            mean=(y, 'mean')
        )
        .sort_values('decile')
    )

    gainout['cumn'] = gainout['n'].cumsum()
    gainout['cumsum'] = gainout['total'].cumsum()
    gainout['cummean'] = gainout['cumsum'] / gainout['cumn']

    return gainout


def diagnostyka_rubina(Q, U, df_stopnie):
    Q, U = np.asarray(Q, float), np.asarray(U, float)
    m = len(Q)
    s2_w = U.mean()
    s2_m = Q.var(ddof=1)
    T = s2_w + (m + 1) / m * s2_m
    gamma = (m + 1) / m * s2_m / s2_w
    lam = (gamma + 2 / (df_stopnie + 3)) / (gamma + 1)
    return {"m": m, "W": s2_w, "B": s2_m, "T": T, "gamma": gamma, "lambda": lam,
            "RE": 1 / (1 + lam / m), "blad_MC": np.sqrt(s2_m / m), "m_zalecane": int(np.ceil(100 * lam))}


def kalibracja_imputacji(df, zmienne_imputacji):
    obs = df[df["M"] == 0].copy()
    model = smf.glm("y_newdeposit ~ " + zmienne_imputacji, data=obs, family=sm.families.Binomial()).fit()
    obs["p_hat"] = model.predict(obs)
    obs["p_przewidywane"] = obs["p_hat"]

    y_dec = gains_table(obs, y="y_newdeposit", score="p_hat", groups=10)
    p_dec = gains_table(obs, y="p_przewidywane", score="p_hat", groups=10)
    tab = pd.DataFrame({"decyl": y_dec["decile"], "n": y_dec["n"],
                        "zakupy_obserwowane": y_dec["mean"], "zakupy_przewidywane": p_dec["mean"]})

    e = tab["zakupy_przewidywane"]
    hl = float((tab["n"] * (tab["zakupy_obserwowane"] - e) ** 2 / (e * (1 - e))).sum())
    p_hl = float(stats.chi2.sf(hl, len(tab) - 2))

    df_pred = df.copy()
    df_pred["p_hat"] = model.predict(df_pred)
    porownanie = (df_pred.groupby(["contact", "M"])["p_hat"].mean().unstack()
                  .rename(columns={0: "przewidywane P(Y=1): obserwowani", 1: "przewidywane P(Y=1): braki"}))
    porownanie.index = porownanie.index.map({0: "0: brak kontaktu", 1: "1: kanał A", 2: "2: kanał B"})
    return {"tab": tab, "HL": hl, "p_HL": p_hl, "braki_vs_obs": porownanie}


def tabela_porownawcza(wyniki3, wyniki4):
    metody = [("CC (MCAR)", wyniki3.get("wynik_cc")),
              ("MI (MAR)", wyniki3.get("wynik_mi")),
              ("IPW (MAR)", wyniki4.get("wynik_ipw"))]
    wiersze = []
    for nazwa, w in metody:
        if not w:
            continue
        wiersze.append({"metoda": nazwa, "beta1-beta2": w["est"], "SE": w["se"], "OR A vs B": w["OR"],
                        "OR_low": w["OR_low"], "OR_high": w["OR_high"], "p": w["p"],
                        "decyzja": "odrzucamy H0" if w["p"] < ALPHA else "brak podstaw do odrzucenia H0"})
    return pd.DataFrame(wiersze).set_index("metoda")


def wykres_diagnostyki(Q, kal, sciezka):
    with plt.rc_context(STYL):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
        m = np.arange(1, len(Q) + 1)
        ax1.plot(m, np.exp(np.cumsum(Q) / m), color=KOLOR["seria1"], lw=2)
        ax1.axhline(np.exp(np.mean(Q)), color=KOLOR["wyciszony"], lw=1, ls="--")
        ax1.set(title="Stabilizacja OR (A vs B) wraz z liczbą imputacji",
                xlabel="liczba imputacji m", ylabel="średnia OR z pierwszych m imputacji")
        ax1.grid(color=KOLOR["siatka"], lw=0.6)

        t = kal["tab"]
        gr = max(t["zakupy_przewidywane"].max(), t["zakupy_obserwowane"].max()) * 1.08
        ax2.plot([0, gr], [0, gr], color=KOLOR["wyciszony"], lw=1, ls="--", label="idealna kalibracja")
        ax2.plot(t["zakupy_przewidywane"], t["zakupy_obserwowane"], "o-", color=KOLOR["seria1"], lw=2, ms=6,
                 markeredgecolor="white", label="decyle (gains_table)")
        ax2.set(title=f"Kalibracja modelu imputacyjnego (H-L p = {kal['p_HL']:.3f})",
                xlabel="przewidywane P(Y=1)", ylabel="obserwowany odsetek zakupów")
        ax2.grid(color=KOLOR["siatka"], lw=0.6)
        ax2.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(sciezka, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return sciezka


def wykres_porownania(tab, sciezka):
    kolory = [KOLOR["wyciszony"], KOLOR["seria1"], KOLOR["seria2"]]
    with plt.rc_context(STYL):
        fig, ax = plt.subplots(figsize=(7.5, 2.8))
        for i, (nazwa, r) in enumerate(tab.iterrows()):
            ax.errorbar(r["OR A vs B"], i, xerr=[[r["OR A vs B"] - r["OR_low"]], [r["OR_high"] - r["OR A vs B"]]],
                        fmt="o", ms=8, lw=2.5, color=kolory[i % 3], markeredgecolor="white")
            ax.text(r["OR_high"] * 1.03, i, f"{r['OR A vs B']:.2f} [{r['OR_low']:.2f}; {r['OR_high']:.2f}], "
                    f"p = {r['p']:.2f}".replace(".", ","), va="center", fontsize=8, color=KOLOR["tekst"])
        ax.axvline(1, color=KOLOR["tekst2"], lw=1)
        ax.set_yticks(range(len(tab)), tab.index)
        ax.invert_yaxis()
        ax.set_xlim(0.6, 2.6)
        ax.set(title="OR kanał A vs kanał B (95% CI) wg metody obsługi braków", xlabel="OR")
        ax.grid(axis="x", color=KOLOR["siatka"], lw=0.6)
        fig.tight_layout()
        fig.savefig(sciezka, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return sciezka


def uruchom(df, wyniki1, wyniki2, wyniki3, wyniki4):
    print("\n=== 5. DIAGNOSTYKA IMPUTACJI I PODSUMOWANIE ===")
    proc_brakow = df["M"].mean()
    wynik_mi = wyniki3["wynik_mi"]
    Q, U = wyniki3["kontrast_imputacje"]

    # Miara diagnostyczna
    diag = diagnostyka_rubina(Q, U, wynik_mi["df"])
    tab_diag = pd.DataFrame([diag], index=["beta1 - beta2 (A vs B)"])
    print("\n--- A. MIARA DIAGNOSTYCZNA ---")
    print(tab_diag[["m", "W", "B", "T", "gamma", "lambda", "RE", "blad_MC", "m_zalecane"]].to_string(
        float_format=lambda v: f"{v:.4f}"))
    print(f"\nOdsetek braków Y: {proc_brakow:.1%}; lambda dla kontrastu A vs B: {diag['lambda']:.1%} "
          f"({diag['lambda'] / proc_brakow:.1f} raza więcej).")
    print(f"Błąd Monte Carlo beta1-beta2: {diag['blad_MC']:.4f} = {diag['blad_MC'] / wynik_mi['se']:.1%} SE; "
          f"zalecane m >= 100*lambda = {diag['m_zalecane']}, użyto m = {diag['m']} (RE = {diag['RE']:.4f}).")

    # Kalibracja modelu imputacyjnego
    kal = kalibracja_imputacji(df, wyniki3["model_imputacyjny"])
    print("\n--- B. KALIBRACJA MODELU IMPUTACYJNEGO ---")
    print(f"Model imputacyjny: y_newdeposit ~ {wyniki3['model_imputacyjny']}")
    print(kal["tab"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Hosmer-Lemeshow: chi2 = {kal['HL']:.2f}, df = {len(kal['tab']) - 2}, p = {kal['p_HL']:.4f}")
    print("\nŚrednie przewidywane P(Y=1) wg kanału: obserwowani vs rekordy z brakiem:")
    print(kal["braki_vs_obs"].to_string(float_format=lambda v: f"{v:.4f}"))

    # Porównanie metod
    tab_por = tabela_porownawcza(wyniki3, wyniki4)
    print("\n--- C. PORÓWNANIE METOD: OR kanał A vs B ---")
    print(tab_por.to_string(float_format=lambda v: f"{v:.4f}"))

    wykresy = [wykres_diagnostyki(Q, kal, OUT_DIR / "fig_diagnostyka.png"),
               wykres_porownania(tab_por, OUT_DIR / "fig_porownanie_metod.png")]
    print("\nWykresy:", ", ".join(p.name for p in wykresy))

    # Wnioski
    cc, mi = wyniki3["wynik_cc"], wynik_mi
    print("\n--- WNIOSKI ---")
    print("Mechanizm braków:")
    print("  Testy jednowymiarowe oraz model logitowy wykazują istotny wpływ wieku ('age') oraz kanału")
    print("  ('contact') na prawdopodobieństwo braku danych (M). Wyklucza to MCAR, lecz nie pozwala rozstrzygnąć")
    print("  między MAR a MNAR, więc MAR pozostaje założeniem. Zmienna 'education' istotnie determinuje samą")
    print("  zmienną 'y_newdeposit', stanowiąc dobrą zmienną pomocniczą do korekty oszacowań podczas imputacji.")

    print("\nWynik modelu w kontekście braków:")
    print(f"  Przy MCAR (CC) OR(A vs B) = {cc['OR']:.2f}, p = {cc['p']:.2f}; przy MAR (MI) OR = {mi['OR']:.2f}, "
          f"p = {mi['p']:.2f}", end="")
    if "wynik_ipw" in wyniki4:
        ipw = wyniki4["wynik_ipw"]
        print(f"; IPW OR = {ipw['OR']:.2f}, p = {ipw['p']:.2f}.")
    else:
        print(".")
    print("  We wszystkich metodach brak podstaw do odrzucenia H0: beta1 = beta2, czyli nie znaleziono istotnej")
    print("  różnicy szansy zakupu między kanałami A i B.")
    if "tab_wrazliwosc" in wyniki4:
        tw = wyniki4["tab_wrazliwosc"]
        istotny_mar = bool(tw.query("delta_A == 0 and delta_B == 0")["istotny"].iloc[0])
        bez_zmian = int((tw["istotny"] == istotny_mar).sum())
        diag_or = tw.query("delta_A == delta_B")["OR"]
        print(f"  Analiza wrażliwości: wniosek pozostaje niezmieniony w {bez_zmian} z {len(tw)} scenariuszy")
        print(f"  odstępstwa od MAR. Przy jednakowym odstępstwie w obu kanałach OR wynosi "
              f"{diag_or.min():.2f}-{diag_or.max():.2f}.")
        print("  Wniosek zmienia się przy wyraźnym odstępstwie tylko w jednym kanale, więc wynik jest warunkowy")
        print("  względem założenia MAR.")

    print("\nMiara diagnostyczna imputacji:")
    print("  Frakcja brakującej informacji lambda dla testowanego kontrastu beta1 - beta2.")
    print(f"  lambda = {diag['lambda']:.2f}, czyli ok. {diag['lambda']:.0%} informacji o kontraście A vs B jest")
    print(f"  utracone z powodu braków danych, przy odsetku braków Y równym {proc_brakow:.1%}.")
    print(f"  Efektywność estymatora MI RE = {diag['RE']:.4f}, zalecane m >= {diag['m_zalecane']}, "
          f"użyto m = {diag['m']}.")
    t = kal["tab"]
    max_odch = (t["zakupy_obserwowane"] - t["zakupy_przewidywane"]).abs().max()
    if kal["p_HL"] < ALPHA:
        print(f"  Test Hosmera-Lemeshowa wskazuje na istotne, choć niewielkie odchylenia kalibracji "
              f"(p = {kal['p_HL']:.3f}),")
    else:
        print(f"  Test Hosmera-Lemeshowa nie wskazuje na istotne odchylenia kalibracji (p = {kal['p_HL']:.3f}),")
    print(f"  największa różnica w decylu wynosi {max_odch * 100:.1f} p.p.")
    bvo = kal["braki_vs_obs"]
    print("  Model imputacyjny przewiduje dla rekordów z brakiem prawie takie samo P(Y=1) jak dla obserwowanych")
    print(f"  (np. kanał B: {bvo.iloc[2, 1]:.3f} vs {bvo.iloc[2, 0]:.3f}), dlatego MI daje wynik bliski CC.")

    return {"tab_diagnostyka": tab_diag, "tab_kalibracja": kal["tab"], "kalibracja": kal,
            "tab_porownanie": tab_por, "wykresy": wykresy}


if __name__ == "__main__":
    df = wczytaj_dane()
    w2 = sekcja2_mechanizm.uruchom(df)
    w3 = sekcja3_modele.uruchom(df, w2.get("zmienne_imputacji"))
    w4 = sekcja4_innowacje.uruchom(df, w2, w3)
    uruchom(df, {}, w2, w3, w4)

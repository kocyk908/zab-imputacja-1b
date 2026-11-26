# -*- coding: utf-8 -*-
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from scipy import stats

import sekcja2_mechanizm
import sekcja3_modele
from wspolne import ALPHA, FORMULA, M_IMP, OUT_DIR, PARAM_A, PARAM_B, SEED, wczytaj_dane

DELTY = np.linspace(-2.0, 2.0, 9)
M_WRAZLIWOSC = 20
PERCENTYL_PRZYCIECIA = 99
GRUPY_SKLONNOSCI = 5
SKLADNIK_AGE2 = "I(age ** 2)"

SCENARIUSZE = {
    "tylko kanał A": ("delta_B == 0", "delta_A", r"Braki tylko w kanale A ($\delta_B$ = 0)"),
    "tylko kanał B": ("delta_A == 0", "delta_B", r"Braki tylko w kanale B ($\delta_A$ = 0)"),
    "oba kanały, to samo delta": ("delta_A == delta_B", "delta_A", r"Braki w A i B, to samo $\delta$"),
}

KOLOR = {
    "tlo": "#fcfcfb", "tekst": "#0b0b0b", "tekst2": "#52514e", "wyciszony": "#898781",
    "siatka": "#e1e0d9", "os": "#c3c2b7",
    "seria1": "#2a78d6", "seria2": "#eb6834",
    "minus": "#e34948", "zero": "#f0efec", "plus": "#2a78d6",
}
STYL = {
    "figure.facecolor": KOLOR["tlo"], "axes.facecolor": KOLOR["tlo"], "savefig.facecolor": KOLOR["tlo"],
    "axes.edgecolor": KOLOR["os"], "axes.labelcolor": KOLOR["tekst2"], "axes.titlecolor": KOLOR["tekst"],
    "axes.titlelocation": "left", "axes.titlesize": 10, "axes.labelsize": 9, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": KOLOR["os"], "ytick.color": KOLOR["os"],
    "xtick.labelcolor": KOLOR["tekst2"], "ytick.labelcolor": KOLOR["tekst2"],
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.frameon": False, "legend.fontsize": 8,
}
OR_PODZIALKA = [0.25, 0.33, 0.5, 0.67, 0.8, 1, 1.25, 1.5, 2, 3, 4]


def _liczba(x, wzor="{:.2f}"):
    return wzor.format(x).replace(".", ",").replace("-", "\u2212")


def kontrast_ab(model):
    b, V = model.params, model.cov_params()
    est = b[PARAM_A] - b[PARAM_B]
    var = V.loc[PARAM_A, PARAM_A] + V.loc[PARAM_B, PARAM_B] - 2 * V.loc[PARAM_A, PARAM_B]
    return float(est), float(var)


def wynik_walda(est, se):
    z = stats.norm.ppf(1 - ALPHA / 2)
    return {"est": float(est), "se": float(se), "OR": float(np.exp(est)),
            "OR_low": float(np.exp(est - z * se)), "OR_high": float(np.exp(est + z * se)),
            "p": float(2 * stats.norm.sf(abs(est / se)))}


def siatka_wrazliwosci(df, zmienne_imputacji):
    braki = df["y_newdeposit"].isna()
    kanal = df.loc[braki, "contact"].to_numpy()
    braki_A, braki_B = kanal == 1, kanal == 2

    wiersze, Q_mar = [], None
    # Imputacja z przesunięciem logitu dla każdej komórki siatki delta_A x delta_B
    for d_A in DELTY:
        for d_B in DELTY:
            delta = np.where(braki_A, d_A, 0.0) + np.where(braki_B, d_B, 0.0)
            rng = np.random.default_rng(SEED)
            Q, U, zakup_A, zakup_B = [], [], [], []
            for _ in range(M_WRAZLIWOSC):
                df_imp = sekcja3_modele.imputuj(df, zmienne_imputacji, rng, delta=delta)
                model = smf.glm(FORMULA, data=df_imp, family=sm.families.Binomial()).fit()
                est, var = kontrast_ab(model)
                Q.append(est)
                U.append(var)
                y_imp = df_imp.loc[braki, "y_newdeposit"].to_numpy()
                zakup_A.append(y_imp[braki_A].mean())
                zakup_B.append(y_imp[braki_B].mean())
            # Reguły Rubina
            r = sekcja3_modele.rubin(Q, U, n=len(df), k=len(model.params))
            if d_A == 0 and d_B == 0:
                Q_mar = np.array(Q)
            wiersze.append({
                "delta_A": d_A, "delta_B": d_B, "est": r["est"], "se": r["se"],
                "OR": np.exp(r["est"]), "OR_low": np.exp(r["low"]), "OR_high": np.exp(r["high"]),
                "p": r["p"], "zakup_braki_A": np.mean(zakup_A), "zakup_braki_B": np.mean(zakup_B),
            })
        print(f"  delta_A = {d_A:+.1f}: policzono {len(DELTY)} komórek", flush=True)

    tab = pd.DataFrame(wiersze)
    tab["istotny"] = tab["p"] < ALPHA
    return tab, Q_mar


def scenariusz(tab, nazwa):
    warunek, kolumna, _ = SCENARIUSZE[nazwa]
    s = tab.query(warunek).sort_values(kolumna).reset_index(drop=True)
    s["delta"] = s[kolumna]
    return s


def punkty_zwrotne(tab):
    wiersze = []
    for nazwa in SCENARIUSZE:
        s = scenariusz(tab, nazwa)
        i0 = int(np.flatnonzero(s["delta"] == 0)[0])
        wniosek_mar = bool(s.loc[i0, "istotny"])
        for kierunek, krok in (("delta < 0", -1), ("delta > 0", 1)):
            wiersz = {"scenariusz": nazwa, "kierunek": kierunek, "delta_zwrotne": np.nan,
                      "szansa_x": np.nan, "OR": np.nan, "zakup_braki_A": np.nan, "zakup_braki_B": np.nan}
            i = i0 + krok
            while 0 <= i < len(s):
                # Interpolacja log(p) między punktami siatki
                if bool(s.loc[i, "istotny"]) != wniosek_mar:
                    j = i - krok
                    t = ((np.log(ALPHA) - np.log(s.loc[j, "p"]))
                         / (np.log(s.loc[i, "p"]) - np.log(s.loc[j, "p"])))
                    w = {k: s.loc[j, k] + t * (s.loc[i, k] - s.loc[j, k])
                         for k in ("delta", "OR", "zakup_braki_A", "zakup_braki_B")}
                    wiersz.update(delta_zwrotne=w["delta"], szansa_x=np.exp(w["delta"]), OR=w["OR"],
                                  zakup_braki_A=w["zakup_braki_A"], zakup_braki_B=w["zakup_braki_B"])
                    break
                i += krok
            wiersze.append(wiersz)
    return pd.DataFrame(wiersze)


def ipw(df, model_brakow):
    # Wybór modelu braków: wiek liniowo albo z age² (AIC i LRT)
    formula_lin = model_brakow.model.formula
    if SKLADNIK_AGE2 in formula_lin:
        raise ValueError("Model braków z sekcji 2 ma już age²; oczekiwany model z wiekiem liniowo.")
    model_age2 = smf.glm(f"{formula_lin} + {SKLADNIK_AGE2}", data=df, family=sm.families.Binomial()).fit()
    LR = 2 * (model_age2.llf - model_brakow.llf)
    df_LR = int(model_age2.df_model - model_brakow.df_model)
    p_LR = stats.chi2.sf(LR, df_LR)
    modele = {"wiek liniowo (sekcja 2)": model_brakow, "z age²": model_age2}
    wybrany = "z age²" if (model_age2.aic < model_brakow.aic and p_LR < ALPHA) else "wiek liniowo (sekcja 2)"
    wybor = {"formula_liniowa": formula_lin, "AIC_liniowy": float(model_brakow.aic),
             "AIC_age2": float(model_age2.aic), "LRT": float(LR), "df_LRT": df_LR, "p_LRT": float(p_LR),
             "wybrany": wybrany}

    # Odsetek braków wg grup wieku: obserwowany i przewidywany
    grupy = pd.cut(df["age"], [0, 30, 40, 50, 60, 70, 200],
                   labels=["18-30", "31-40", "41-50", "51-60", "61-70", "71+"])
    kalibracja = df.groupby(grupy, observed=True)["M"].agg(n="size", obserwowany="mean")
    for nazwa, m in modele.items():
        przewidywane = pd.Series(np.asarray(m.predict(df)), index=df.index)
        kalibracja[nazwa] = przewidywane.groupby(grupy, observed=True).mean()

    # Wagi IPW i ważony model logitowy na kompletnych przypadkach
    odpowiedz = (df["M"] == 0).to_numpy()
    cc = df[odpowiedz]
    p_odp_ogolem = odpowiedz.mean()
    wiersze, wagi_ht, suma_wag = [], {}, {}
    for nazwa, m in modele.items():
        phi = 1 - np.asarray(m.predict(df))
        w_ht = 1 / phi[odpowiedz]
        wagi_ht[nazwa] = w_ht
        suma_wag[nazwa] = float(w_ht.sum())
        w_stab = p_odp_ogolem * w_ht
        grupa = pd.qcut(phi, GRUPY_SKLONNOSCI, labels=False, duplicates="drop")
        phi_grupy = pd.Series(odpowiedz).groupby(grupa).mean().to_numpy()
        w_grupy = p_odp_ogolem / phi_grupy[grupa[odpowiedz]]
        warianty = {"pełne": w_stab,
                    f"przycięte (p{PERCENTYL_PRZYCIECIA})":
                        np.minimum(w_stab, np.percentile(w_stab, PERCENTYL_PRZYCIECIA)),
                    f"{GRUPY_SKLONNOSCI} grup skłonności": w_grupy}
        for wariant, w in warianty.items():
            # Odporne błędy standardowe (HC0)
            model = smf.glm(FORMULA, data=cc, family=sm.families.Binomial(),
                            var_weights=w).fit(cov_type="HC0")
            est, var = kontrast_ab(model)
            wiersze.append({"model braków": nazwa, "wagi": wariant,
                            "max waga 1/phi": float(w.max() / p_odp_ogolem),
                            "ESS": float(w.sum() ** 2 / (w ** 2).sum()),
                            **wynik_walda(est, np.sqrt(var))})
    tab = pd.DataFrame(wiersze)
    glowny = tab[(tab["model braków"] == wybrany) & (tab["wagi"] == "pełne")].iloc[0]
    wynik_ipw = {k: float(glowny[k]) for k in ("est", "se", "OR", "OR_low", "OR_high", "p")}

    # Kontrola: wagi niestabilizowane dają to samo OR
    m_ht = smf.glm(FORMULA, data=cc, family=sm.families.Binomial(),
                   var_weights=wagi_ht[wybrany]).fit(cov_type="HC0")
    est_ht, var_ht = kontrast_ab(m_ht)
    roznica_stab = max(abs(est_ht - wynik_ipw["est"]), abs(np.sqrt(var_ht) - wynik_ipw["se"]))

    return {"tab": tab, "wynik_ipw": wynik_ipw, "wybor": wybor, "kalibracja": kalibracja,
            "modele": modele, "wagi_ht": wagi_ht, "suma_wag": suma_wag, "roznica_stab": roznica_stab}


def _os_or(os_wykresu, zakres):
    podzialka = [v for v in OR_PODZIALKA if zakres[0] <= v <= zakres[1]]
    os_wykresu.set_major_locator(FixedLocator(podzialka))
    os_wykresu.set_major_formatter(FuncFormatter(lambda v, _: _liczba(v, "{:g}")))
    os_wykresu.set_minor_locator(NullLocator())


def wykres_wrazliwosci(tab, sciezka):
    or_mat = tab.pivot(index="delta_A", columns="delta_B", values="OR")
    ist = tab.pivot(index="delta_A", columns="delta_B", values="istotny").to_numpy(dtype=bool)
    log_or = np.log(or_mat.to_numpy())
    zakres = np.abs(log_or).max()
    n, k0 = len(DELTY), int(np.flatnonzero(DELTY == 0)[0])
    opisy = ["0" if d == 0 else _liczba(d, "{:+.1f}") for d in DELTY]

    with plt.rc_context(STYL):
        fig, ax = plt.subplots(figsize=(7.0, 6.2))
        skala = LinearSegmentedColormap.from_list("or", [KOLOR["minus"], KOLOR["zero"], KOLOR["plus"]])
        im = ax.imshow(log_or, cmap=skala, vmin=-zakres, vmax=zakres, origin="lower")
        ax.set_xticks(np.arange(-0.5, n), minor=True)
        ax.set_yticks(np.arange(-0.5, n), minor=True)
        ax.grid(which="minor", color=KOLOR["tlo"], lw=1.5)
        ax.tick_params(which="both", length=0)

        # Obrys obszaru istotnej różnicy
        for i, j in zip(*np.nonzero(ist)):
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n and ist[ni, nj]:
                    continue
                if di:
                    xs, ys = [j - 0.5, j + 0.5], [i + di / 2] * 2
                else:
                    xs, ys = [j + dj / 2] * 2, [i - 0.5, i + 0.5]
                ax.plot(xs, ys, color=KOLOR["tekst"], lw=2, solid_capstyle="projecting", clip_on=False)

        ax.plot(k0, k0, "o", ms=7, mfc=KOLOR["tekst"], mec=KOLOR["tlo"], mew=1.5)

        ax.set_xticks(range(n), opisy)
        ax.set_yticks(range(n), opisy)
        for krawedz in ax.spines.values():
            krawedz.set_visible(False)
        ax.set_xlabel(r"$\delta_B$: przesunięcie logitu zakupu dla braków w kanale B")
        ax.set_ylabel(r"$\delta_A$: przesunięcie logitu zakupu dla braków w kanale A")
        ax.set_title("Analiza wrażliwości na MNAR: OR zakupu, kanał A vs kanał B", pad=10)

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        podzialka = [v for v in OR_PODZIALKA if abs(np.log(v)) <= zakres]
        cb.set_ticks(np.log(podzialka), labels=[_liczba(v, "{:g}") for v in podzialka])
        cb.set_label("OR (A vs B), skala log")
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0)
        for y, tekst, va in ((1.01, "A lepszy", "bottom"), (-0.01, "B lepszy", "top")):
            cb.ax.text(0.5, y, tekst, transform=cb.ax.transAxes, ha="center", va=va,
                       fontsize=7, color=KOLOR["tekst2"])

        ax.legend(handles=[
            Line2D([], [], color=KOLOR["tekst"], lw=2,
                   label=f"obszar p < {_liczba(ALPHA, '{:g}')} (istotna różnica A vs B)"),
            Line2D([], [], lw=0, marker="o", ms=7, mfc=KOLOR["tekst"], mec=KOLOR["tlo"], mew=1.5,
                   label=r"MAR ($\delta_A = \delta_B = 0$): OR = " + _liczba(or_mat.iloc[k0, k0])),
        ], loc="upper left", bbox_to_anchor=(0, -0.1))
        fig.savefig(sciezka, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return sciezka


def wykres_scenariuszy(tab, tab_tp, sciezka):
    zakres = (tab["OR_low"].min(), tab["OR_high"].max())
    with plt.rc_context(STYL):
        fig, osie = plt.subplots(1, 3, figsize=(10.5, 3.7), sharey=True)
        for ax, nazwa in zip(osie, SCENARIUSZE):
            s = scenariusz(tab, nazwa)
            x, OR, ist = s["delta"].to_numpy(), s["OR"].to_numpy(), s["istotny"].to_numpy()
            ax.grid(axis="y", color=KOLOR["siatka"], lw=0.8)
            ax.set_axisbelow(True)
            ax.axhline(1, color=KOLOR["tekst2"], lw=1)
            ax.axvline(0, color=KOLOR["os"], lw=1)
            ax.annotate("MAR", (0, 0), xycoords=("data", "axes fraction"), xytext=(3, 3),
                        textcoords="offset points", fontsize=7, color=KOLOR["wyciszony"])
            ax.fill_between(x, s["OR_low"], s["OR_high"], color=KOLOR["seria1"], alpha=0.12, lw=0)
            ax.plot(x, OR, color=KOLOR["seria1"], lw=2)
            ax.plot(x[~ist], OR[~ist], "o", ms=6, mfc=KOLOR["tlo"], mec=KOLOR["seria1"], mew=1.8)
            ax.plot(x[ist], OR[ist], "o", ms=6, mfc=KOLOR["seria1"], mec=KOLOR["seria1"], mew=1.8)
            zwrotne = tab_tp.loc[(tab_tp["scenariusz"] == nazwa) & tab_tp["delta_zwrotne"].notna(), "delta_zwrotne"]
            for d in zwrotne:
                ax.axvline(d, color=KOLOR["tekst"], lw=1)
            podtytul = ("punkt zwrotny: " + "; ".join(r"$\delta^*$ ≈ " + _liczba(d) for d in zwrotne)
                        if len(zwrotne) else "wniosek bez zmian w całym zakresie")
            ax.set_title(SCENARIUSZE[nazwa][2], pad=17)
            ax.text(0, 1.02, podtytul, transform=ax.transAxes, fontsize=8, color=KOLOR["tekst2"], va="bottom")
            ax.set_xlabel(r"$\delta$: przesunięcie logitu zakupu dla braków")
            ax.set_xticks(DELTY[::2], ["0" if d == 0 else _liczba(d, "{:+.0f}") for d in DELTY[::2]])
        osie[0].set_yscale("log")
        _os_or(osie[0].yaxis, zakres)
        osie[0].set_ylabel("OR (A vs B) z 95% CI, skala log")
        fig.legend(handles=[
            Line2D([], [], color=KOLOR["seria1"], lw=0, marker="o", ms=6, mfc=KOLOR["seria1"],
                   label=f"p < {_liczba(ALPHA, '{:g}')}"),
            Line2D([], [], color=KOLOR["seria1"], lw=0, marker="o", ms=6, mfc=KOLOR["tlo"], mew=1.8,
                   label=f"p ≥ {_liczba(ALPHA, '{:g}')}"),
            Patch(color=KOLOR["seria1"], alpha=0.12, label="95% CI"),
        ], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
        fig.tight_layout()
        fig.savefig(sciezka, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return sciezka


def wykres_wag(df, wyn_ipw, sciezka):
    modele, wagi, tab = wyn_ipw["modele"], wyn_ipw["wagi_ht"], wyn_ipw["tab"]
    kolory = dict(zip(modele, (KOLOR["seria1"], KOLOR["seria2"])))
    with plt.rc_context(STYL):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.3, 1]})

        obs = df.groupby("age")["M"].mean()
        ax1.grid(axis="y", color=KOLOR["siatka"], lw=0.8)
        ax1.set_axisbelow(True)
        ax1.scatter(obs.index, obs.to_numpy(), s=12, color=KOLOR["wyciszony"], lw=0, zorder=3,
                    label="obserwowany (wg roku życia)")
        for nazwa, m in modele.items():
            pred = pd.Series(np.asarray(m.predict(df)), index=df.index).groupby(df["age"]).mean()
            ax1.plot(pred.index, pred.to_numpy(), color=kolory[nazwa], lw=2, label=f"model braków: {nazwa}")
        ax1.set_ylim(0, 1)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax1.set_xlabel("wiek")
        ax1.set_ylabel("odsetek braków Y")
        ax1.set_title("Odsetek braków wg wieku: dane i modele braków")
        ax1.legend(loc="upper left")

        w_max = max(w.max() for w in wagi.values())
        przedzialy = np.geomspace(1, w_max * 1.05, 40)
        for nazwa, w in wagi.items():
            wiersz = tab[(tab["model braków"] == nazwa) & (tab["wagi"] == "pełne")].iloc[0]
            ax2.hist(w, bins=przedzialy, histtype="step", lw=2, color=kolory[nazwa],
                     label=f"{nazwa}: max {_liczba(w.max(), '{:.1f}')}, ESS {wiersz['ESS']:.0f}")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        podzialka = [v for v in (1, 1.5, 2, 3, 5, 10, 20, 50, 100) if v <= w_max * 1.05]
        ax2.xaxis.set_major_locator(FixedLocator(podzialka))
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _liczba(v, "{:g}")))
        ax2.xaxis.set_minor_locator(NullLocator())
        ax2.set_xlabel(r"waga IPW = 1 / $\varphi$ = 1 / P(odpowiedź | X), skala log")
        ax2.set_ylabel("liczba osób, skala log")
        ax2.set_title("Rozkład wag IPW (kompletne przypadki)")
        ax2.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(sciezka, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return sciezka


def uruchom(df, wyniki2, wyniki3):
    print("\n=== 4. WRAŻLIWOŚĆ (MNAR) i IPW ===")

    # ---------------------------------------------------------------- A. wrażliwość
    zmienne_imputacji = wyniki3["model_imputacyjny"]
    print("\n--- A. ANALIZA WRAŻLIWOŚCI NA MNAR (delta-adjustment, tipping point) ---")
    print(f"Model imputacyjny (z sekcji 3): y_newdeposit ~ {zmienne_imputacji}")
    print(f"Siatka: delta_A i delta_B od {DELTY[0]:+.1f} do {DELTY[-1]:+.1f} co {DELTY[1] - DELTY[0]:.1f}, "
          f"m = {M_WRAZLIWOSC} imputacji na komórkę, seed {SEED}")
    print("delta = przesunięcie logitu zakupu u osób bez wyniku: szansa zakupu x exp(delta) względem MAR")
    tab_wr, Q_mar = siatka_wrazliwosci(df, zmienne_imputacji)

    Q_sekcja3 = np.asarray(wyniki3["kontrast_imputacje"][0])[:M_WRAZLIWOSC]
    zgodne = Q_mar is not None and len(Q_sekcja3) == M_WRAZLIWOSC and np.allclose(Q_mar, Q_sekcja3)
    print(f"Kontrola odtwarzalności: komórka (0, 0) = pierwsze {M_WRAZLIWOSC} imputacji z sekcji 3: "
          f"{'TAK' if zgodne else 'NIE'}")

    def siatka_do_druku(teksty):
        piv = tab_wr.assign(tekst=teksty).pivot(index="delta_A", columns="delta_B", values="tekst").iloc[::-1]
        piv.index = [f"delta_A {d:+.1f}" for d in piv.index]
        piv.columns = [f"{d:+.1f}" for d in piv.columns]
        return piv.to_string()

    print(f"\nOR kanał A vs B (wiersze: delta_A, kolumny: delta_B; * = p < {ALPHA}):")
    print(siatka_do_druku([f"{o:.2f}{'*' if i else ' '}" for o, i in zip(tab_wr["OR"], tab_wr["istotny"])]))
    print("\nWartość p testu H0: beta1 - beta2 = 0 (reguły Rubina):")
    print(siatka_do_druku([f"{p:.3f}" for p in tab_wr["p"]]))

    tab_tp = punkty_zwrotne(tab_wr)
    mar = tab_wr[(tab_wr["delta_A"] == 0) & (tab_wr["delta_B"] == 0)].iloc[0]
    obserwowani = df[df["M"] == 0].groupby("contact")["y_newdeposit"].mean()
    print("\nPunkty zwrotne (pierwsza zmiana wniosku, interpolacja log(p) między punktami siatki):")
    print(tab_tp.to_string(index=False, formatters={
        "delta_zwrotne": lambda v: "-" if pd.isna(v) else f"{v:+.2f}",
        "szansa_x": lambda v: "-" if pd.isna(v) else f"{v:.2f}",
        "OR": lambda v: "-" if pd.isna(v) else f"{v:.3f}",
        "zakup_braki_A": lambda v: "-" if pd.isna(v) else f"{v:.1%}",
        "zakup_braki_B": lambda v: "-" if pd.isna(v) else f"{v:.1%}",
    }))
    print(f"Odniesienie, odsetek zakupów: obserwowani A = {obserwowani[1]:.1%}, B = {obserwowani[2]:.1%}; "
          f"braki przy MAR: A = {mar['zakup_braki_A']:.1%}, B = {mar['zakup_braki_B']:.1%}")

    # ---------------------------------------------------------------- B. IPW
    print("\n--- B. IPW: WAŻENIE ODWROTNOŚCIĄ PRAWDOPODOBIEŃSTWA ODPOWIEDZI ---")
    wyn_ipw = ipw(df, wyniki2["model_brakow"])
    wybor, tab_ipw, wynik_ipw = wyn_ipw["wybor"], wyn_ipw["tab"], wyn_ipw["wynik_ipw"]
    print(f"Model braków z sekcji 2: {wybor['formula_liniowa']}")
    print("Wybór modelu braków (ta sama reguła co w sekcji 3: AIC i LRT):")
    print(f"  AIC wiek liniowo: {wybor['AIC_liniowy']:.2f}   AIC z age²: {wybor['AIC_age2']:.2f}")
    print(f"  LRT = {wybor['LRT']:.2f}, df = {wybor['df_LRT']}, p = {wybor['p_LRT']:.3e}"
          f"  ->  wybrano: {wybor['wybrany']}")
    print("\nKalibracja wg wieku: odsetek braków obserwowany vs przewidywany przez model braków:")
    print(wyn_ipw["kalibracja"].to_string(formatters={
        k: "{:.1%}".format for k in wyn_ipw["kalibracja"].columns if k != "n"}))
    print(f"\nSuma wag 1/phi na kompletnych przypadkach (oszacowanie liczebności próby; "
          f"powinna być ok. {len(df)}):")
    for nazwa, suma in wyn_ipw["suma_wag"].items():
        print(f"  {nazwa}: {suma:.0f}")

    print(f"\nWarianty IPW: ważony logit na {int((df['M'] == 0).sum())} kompletnych przypadkach, "
          "wagi stabilizowane P(M=0) / phi, odporne SE (HC0);")
    print(f"phi = P(M=0|X) z modelu skłonności albo odsetek odpowiedzi w {GRUPY_SKLONNOSCI} grupach skłonności "
          ":")
    druk = tab_ipw.assign(CI=[f"({lo:.3f}, {hi:.3f})" for lo, hi in zip(tab_ipw["OR_low"], tab_ipw["OR_high"])])
    druk = druk.rename(columns={"CI": "95% CI"})
    print(druk[["model braków", "wagi", "max waga 1/phi", "ESS", "OR", "95% CI", "p"]].to_string(
        index=False, formatters={"max waga 1/phi": "{:.2f}".format, "ESS": "{:.0f}".format,
                                 "OR": "{:.4f}".format, "p": "{:.4f}".format}))
    print(f"Kontrola: wagi niestabilizowane 1/phi dają to samo OR i SE (max różnica {wyn_ipw['roznica_stab']:.1e}); "
          "stabilizacja zmienia tylko skalę wag.")
    print(f"\nWYNIK IPW (model braków: {wybor['wybrany']}, wagi pełne):")
    print(f"OR A/B: {wynik_ipw['OR']:.4f}")
    print(f"95% CI: ({wynik_ipw['OR_low']:.4f}, {wynik_ipw['OR_high']:.4f})")
    print(f"p-value: {wynik_ipw['p']:.4f}")

    # ---------------------------------------------------------------- C. wykresy
    wykresy = [
        wykres_wrazliwosci(tab_wr, OUT_DIR / "fig_wrazliwosc.png"),
        wykres_scenariuszy(tab_wr, tab_tp, OUT_DIR / "fig_wrazliwosc_scenariusze.png"),
        wykres_wag(df, wyn_ipw, OUT_DIR / "fig_ipw_wagi.png"),
    ]
    print("\nWykresy:", ", ".join(p.name for p in wykresy))

    # ---------------------------------------------------------------- D. wnioski
    def wniosek(istotny):
        return "istotna różnica A vs B" if istotny else "brak istotnej różnicy A vs B"

    print("\n--- WNIOSKI (sekcja 4) ---")
    print(f"1. MAR (delta = 0, m = {M_WRAZLIWOSC}): OR = {mar['OR']:.3f}, p = {mar['p']:.3f} "
          f"-> {wniosek(mar['istotny'])}. "
          f"Sekcja 3 (m = {M_IMP}): OR = {wyniki3['wynik_mi']['OR']:.3f}, p = {wyniki3['wynik_mi']['p']:.3f}.")
    bez_zmian = int((tab_wr["istotny"] == mar["istotny"]).sum())
    print(f"2. Wniosek z MAR utrzymuje się w {bez_zmian} z {len(tab_wr)} komórek siatki delta_A x delta_B.")
    diag = scenariusz(tab_wr, "oba kanały, to samo delta")
    print(f"3. To samo delta w obu kanałach: OR od {diag['OR'].min():.3f} do {diag['OR'].max():.3f}, "
          f"{'wniosek bez zmian' if (diag['istotny'] == mar['istotny']).all() else 'wniosek się zmienia'}.")
    print("4. Punkty zwrotne:")
    for _, r in tab_tp.iterrows():
        if pd.isna(r["delta_zwrotne"]):
            print(f"   {r['scenariusz']}, {r['kierunek']}: wniosek bez zmian dla delta od "
                  f"{DELTY[0]:+.1f} do {DELTY[-1]:+.1f}")
        else:
            print(f"   {r['scenariusz']}, {r['kierunek']}: delta ok. {r['delta_zwrotne']:+.2f} "
                  f"(szansa zakupu braków x{r['szansa_x']:.2f}), OR = {r['OR']:.2f}; zakupy wśród braków "
                  f"A / B: {r['zakup_braki_A']:.1%} / {r['zakup_braki_B']:.1%}")
    print(f"5. IPW ({wybor['wybrany']}): OR = {wynik_ipw['OR']:.3f}, 95% CI ({wynik_ipw['OR_low']:.3f}, "
          f"{wynik_ipw['OR_high']:.3f}), p = {wynik_ipw['p']:.3f} -> {wniosek(wynik_ipw['p'] < ALPHA)}. "
          f"Dla porównania CC: OR = {wyniki3['wynik_cc']['OR']:.3f}, MI: OR = {wyniki3['wynik_mi']['OR']:.3f}.")
    grupy_ipw = tab_ipw[(tab_ipw["model braków"] == wybor["wybrany"])
                        & (tab_ipw["wagi"] == f"{GRUPY_SKLONNOSCI} grup skłonności")].iloc[0]
    print(f"   Wariant z {GRUPY_SKLONNOSCI} grupami skłonności: OR = {grupy_ipw['OR']:.3f}, "
          f"p = {grupy_ipw['p']:.3f}.")
    kal = wyn_ipw["kalibracja"].iloc[-1]
    maks = tab_ipw[tab_ipw["wagi"] == "pełne"].set_index("model braków")["max waga 1/phi"]
    print(f"6. Odsetek braków w grupie {wyn_ipw['kalibracja'].index[-1]}: obserwowany {kal['obserwowany']:.0%}, "
          f"model liniowy {kal['wiek liniowo (sekcja 2)']:.0%}, model z age² {kal['z age²']:.0%}; "
          f"największa waga 1/phi: {maks['wiek liniowo (sekcja 2)']:.1f} vs {maks['z age²']:.1f}.")

    return {
        "tab_wrazliwosc": tab_wr,
        "tab_tipping_point": tab_tp,
        "wynik_ipw": wynik_ipw,
        "tab_ipw": tab_ipw,
        "tab_kalibracja_brakow": wyn_ipw["kalibracja"],
        "wybor_modelu_brakow": wybor,
        "wykresy": wykresy,
    }


if __name__ == "__main__":
    df = wczytaj_dane()
    w2 = sekcja2_mechanizm.uruchom(df)
    w3 = sekcja3_modele.uruchom(df, w2.get("zmienne_imputacji"))
    uruchom(df, w2, w3)

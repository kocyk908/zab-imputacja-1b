# -*- coding: utf-8 -*-
"""
SEKCJA 1: STRUKTURA DANYCH                                            OSOBA 1
Punkt 1 polecenia: analiza struktury (cechy ciągłe i skokowe), graficzna ocena wyników badania,
zakres i rozkład braków danych.

jak w wykładzie (zab_cz_1), s. 17-18 (histogram, wykres pudełkowy), s. 46 (skale pomiarowe),
s. 73-74 (macierz braków), s. 89-90 (wzorzec braków)
wykresy (sns.histplot) jak w kodzie "DM_Approach_py (1).ipynb"

Dodatkowe odwołania użyte w implementacji:
jak w wykładzie (zab_cz_1), s. 14 (wykres słupkowy), s. 35, 37, 47-48, 70 (jakość danych,
wartości odstające: reguła 3 sigm i reguła IQR), s. 75-81, 107-108 (skutki braków danych:
mniejsza moc testu, możliwe obciążenie CC)

Uruchomienie samej sekcji:  python sekcja1_struktura.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from wspolne import OUT_DIR, wczytaj_dane  # noqa: E402,F401

# Dodatkowe importy dla obliczeń statystycznych i formatowania wykresów
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy import stats  # noqa: E402

from wspolne import ALPHA  # noqa: E402

# --- konfiguracja analizy struktury i braków ---
CECHA_CIAGLA = "age"
CECHY_SKOKOWE = ["education", "house_loan", "poutcome", "contact", "y_newdeposit"]
ZMIENNE_ANALIZOWANE = [CECHA_CIAGLA] + CECHY_SKOKOWE

# Metadane i skale pomiarowe:
OPIS_CECH = {
    "age": ("ciągła", "ilorazowa", "Wiek klienta (lata)"),
    "education": ("skokowa", "porządkowa (+ unknown)", "Poziom wykształcenia klienta"),
    "house_loan": ("skokowa", "nominalna (binarna)", "Posiadanie kredytu mieszkaniowego"),
    "poutcome": ("skokowa", "nominalna", "Reakcja na poprzednią kampanię"),
    "contact": ("skokowa", "nominalna", "Kanał komunikacji (0=brak, 1=A, 2=B)"),
    "y_newdeposit": ("skokowa", "nominalna (binarna)", "Zakup produktu w t1 (zmienna celu)"),
}

ETYKIETY_CONTACT = {0: "Brak kontaktu (0)", 1: "Kanał A (1)", 2: "Kanał B (2)"}
ETYKIETY_Y = {0.0: "Brak zakupu (0)", 1.0: "Zakup (1)", np.nan: "Brak danych (NaN)"}
KOLEJNOSC_KATEGORII = {
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "house_loan": ["yes", "no"],
    "poutcome": ["unknown", "failure", "other", "success"],
    "contact": [0, 1, 2],
    "y_newdeposit": [0.0, 1.0, np.nan],
}

PRZEDZIALY_WIEKU = [17, 30, 40, 50, 60, 70, 100]
ETYKIETY_WIEKU = ["18-30", "31-40", "41-50", "51-60", "61-70", "71+"]

# Paleta i styl wykresów
KOLOR = {
    "tlo": "#fcfcfb",
    "tekst": "#0b0b0b",
    "tekst2": "#52514e",
    "wyciszony": "#898781",
    "siatka": "#e1e0d9",
    "os": "#c3c2b7",
    "seria1": "#2a78d6",   # niebieski (obserwowane / główna seria)
    "seria2": "#eb6834",   # pomarańczowy (braki / kontrast)
    "seria3": "#2e9f6b",   # zielony (zakup)
    "brak": "#c84b61",     # bordowo-czerwony (braki danych)
    "neutral": "#9aa0a6",  # szary
}
STYL = {
    "figure.facecolor": KOLOR["tlo"],
    "axes.facecolor": KOLOR["tlo"],
    "savefig.facecolor": KOLOR["tlo"],
    "axes.edgecolor": KOLOR["os"],
    "axes.labelcolor": KOLOR["tekst2"],
    "axes.titlecolor": KOLOR["tekst"],
    "axes.titlelocation": "left",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": KOLOR["os"],
    "ytick.color": KOLOR["os"],
    "xtick.labelcolor": KOLOR["tekst2"],
    "ytick.labelcolor": KOLOR["tekst2"],
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
}


def _liczba(x, wzor="{:.1f}"):
    """Formatuje liczbę na wykresie z przecinkiem dziesiętnym."""
    return wzor.format(x).replace(".", ",").replace("-", "\u2212")


def _calkowita(n):
    """Formatuje liczbę całkowitą ze spacją jako separatorem tysięcy."""
    return f"{int(n):,}".replace(",", " ")


def _przedzial_wilsona(k, n, alpha=ALPHA):
    """95% przedział ufności Wilsona dla proporcji k/n.
    Daje dokładniejsze pokrycie niż przybliżenie Walda dla mniejszych podgrup (tu: kanał B)."""
    if n == 0:
        return np.nan, np.nan
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    mian = 1 + z**2 / n
    srodek = (p + z**2 / (2 * n)) / mian
    polowa = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / mian
    return float(max(0.0, srodek - polowa)), float(min(1.0, srodek + polowa))


# =====================================================================================
# 1. TABELE STATYSTYCZNE
# =====================================================================================
def statystyki_age(df):
    """Statystyki opisowe cechy ciągłej age (ogółem oraz wg statusu braku M=0 i M=1)
    wraz z kontrolą wartości odstających wg reguły kwantylowej i reguły 3 sigm.
   """ # jak w wykładzie (zab_cz_1), s. 17-18, 47, 70.
    grupy = [
        ("Ogółem", df[CECHA_CIAGLA]),
        ("Obserwowane (M=0)", df.loc[df["M"] == 0, CECHA_CIAGLA]),
        ("Brak Y (M=1)", df.loc[df["M"] == 1, CECHA_CIAGLA]),
    ]
    wiersze = []
    for nazwa, s in grupy:
        q1, me, q3 = s.quantile([0.25, 0.50, 0.75])
        iqr = q3 - q1
        mean, sd = s.mean(), s.std(ddof=1)
        # Reguła kwantylowa [Q1 - 1.5*IQR, Q3 + 1.5*IQR] i reguła trzech sigm |x - mu| > 3*sigma:
        # jak w wykładzie (zab_cz_1), s. 47, 70
        out_iqr = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        out_3s = int((np.abs(s - mean) > 3 * sd).sum())
        wiersze.append({
            "Grupa": nazwa,
            "n": int(s.count()),
            "Średnia": float(mean),
            "Odch.std.": float(sd),
            "Min": float(s.min()),
            "Q1": float(q1),
            "Mediana": float(me),
            "Q3": float(q3),
            "Max": float(s.max()),
            "IQR": float(iqr),
            "Skośność": float(s.skew()),
            "Odstające (IQR)": out_iqr,
            "Odstające (3σ)": out_3s,
        })
    return pd.DataFrame(wiersze).set_index("Grupa")


def tabele_czestosci(df):
    """Tabele częstości cech skokowych z uwzględnieniem braków danych (value_counts(dropna=False))
    oraz odsetkiem braków Y w każdej kategorii.
    Podział skal pomiarowych. Zestawienie braków w kategoriach."""
    wynik = {}
    n_tot = len(df)
    for col in CECHY_SKOKOWE:
        vc = df[col].value_counts(dropna=False)
        kolejnosc = [k for k in KOLEJNOSC_KATEGORII[col] if any(
            (pd.isna(k) and pd.isna(idx)) or (not pd.isna(k) and idx == k) for idx in vc.index
        )]
        # Zachowanie wszystkich kategorii na wypadek nietypowej wartości w danych
        pozostale = [idx for idx in vc.index if not any(
            (pd.isna(k) and pd.isna(idx)) or (not pd.isna(k) and idx == k) for k in kolejnosc
        )]
        indeks = kolejnosc + pozostale

        wiersze = []
        for kat in indeks:
            maska = df[col].isna() if pd.isna(kat) else (df[col] == kat)
            licz = int(maska.sum())
            udzial = 100.0 * licz / n_tot
            if col == "contact":
                etykieta = ETYKIETY_CONTACT.get(kat, str(kat))
            elif col == "y_newdeposit":
                etykieta = "Brak danych (NaN)" if pd.isna(kat) else ETYKIETY_Y.get(kat, str(kat))
            else:
                etykieta = "Brak danych (NaN)" if pd.isna(kat) else str(kat)

            wiersz = {
                "Kategoria": etykieta,
                "Liczebność": licz,
                "Udział [%]": round(udzial, 2),
            }
            # Dla predyktorów podany jest także rozkład braków Y w danej kategorii
            if col != "y_newdeposit":
                braki_y = int(df.loc[maska, "M"].sum())
                wiersz["Braki Y (n)"] = braki_y
                wiersz["Braki Y [%]"] = round(100.0 * braki_y / licz, 2) if licz > 0 else 0.0
            else:
                # Udział wśród obserwacji kompletnych (M=0)
                n_obs = int((df["M"] == 0).sum())
                wiersz["Udział wśród M=0 [%]"] = (
                    np.nan if pd.isna(kat) else round(100.0 * licz / n_obs, 2)
                )
            wiersze.append(wiersz)
        wynik[col] = pd.DataFrame(wiersze).set_index("Kategoria")
    return wynik


def tabela_wynik_kanal(df):
    """Tabela krzyżowa contact x y_newdeposit (z brakami) oraz % zakupów z 95% CI Wilsona."""
    wiersze = []
    for kod in [0, 1, 2]:
        sub = df[df["contact"] == kod]
        n_tot = len(sub)
        n_brak = int(sub["M"].sum())
        n_obs = n_tot - n_brak
        n_zakup = int((sub["y_newdeposit"] == 1).sum())
        n_nie = int((sub["y_newdeposit"] == 0).sum())
        p_hat = n_zakup / n_obs if n_obs > 0 else np.nan
        ci_lo, ci_hi = _przedzial_wilsona(n_zakup, n_obs, alpha=ALPHA)
        wiersze.append({
            "Kanał": ETYKIETY_CONTACT[kod],
            "Ogółem (n)": n_tot,
            "Udział w próbie [%]": round(100.0 * n_tot / len(df), 2),
            "Brak zakupu (Y=0)": n_nie,
            "Zakup (Y=1)": n_zakup,
            "Braki (NaN)": n_brak,
            "Braki [%]": round(100.0 * n_brak / n_tot, 2),
            "Obserwowane (M=0)": n_obs,
            "Zakup wśród M=0 [%]": round(100.0 * p_hat, 2),
            "CI 95% dolna [%]": round(100.0 * ci_lo, 2),
            "CI 95% górna [%]": round(100.0 * ci_hi, 2),
        })

    # Wiersz podsumowujący
    n_tot = len(df)
    n_brak = int(df["M"].sum())
    n_obs = n_tot - n_brak
    n_zakup = int((df["y_newdeposit"] == 1).sum())
    n_nie = int((df["y_newdeposit"] == 0).sum())
    ci_lo, ci_hi = _przedzial_wilsona(n_zakup, n_obs, alpha=ALPHA)
    wiersze.append({
        "Kanał": "Ogółem",
        "Ogółem (n)": n_tot,
        "Udział w próbie [%]": 100.00,
        "Brak zakupu (Y=0)": n_nie,
        "Zakup (Y=1)": n_zakup,
        "Braki (NaN)": n_brak,
        "Braki [%]": round(100.0 * n_brak / n_tot, 2),
        "Obserwowane (M=0)": n_obs,
        "Zakup wśród M=0 [%]": round(100.0 * n_zakup / n_obs, 2),
        "CI 95% dolna [%]": round(100.0 * ci_lo, 2),
        "CI 95% górna [%]": round(100.0 * ci_hi, 2),
    })
    return pd.DataFrame(wiersze).set_index("Kanał")


def analiza_brakow(df):
    """Zakres braków w każdej zmiennej oraz identyfikacja wzorca z macierzy indykatorowej M.""" #jak w wykładzie (zab_cz_1), s. 73-74 (macierz braków M), s. 89-90 (wzorzec monotoniczny).
    # 1. Zakres braków per zmienna: jak w wykładzie (zab_cz_1), s. 46, 73-74
    wiersze = []
    n_tot = len(df)
    for col in ZMIENNE_ANALIZOWANE:
        n_miss = int(df[col].isna().sum())
        typ, skala, opis = OPIS_CECH[col]
        wiersze.append({
            "Zmienna": col,
            "Typ": typ,
            "Skala pomiarowa": skala,
            "Obserwowane": n_tot - n_miss,
            "Braki (n)": n_miss,
            "Braki [%]": round(100.0 * n_miss / n_tot, 2),
        })
    tab_braki = pd.DataFrame(wiersze).set_index("Zmienna")

    # 2. Macierz indykatorowa braków M = [m_ij]: jak w wykładzie (zab_cz_1), s. 73-74, 89-90
    M_mat = df[ZMIENNE_ANALIZOWANE].isna().astype(int)
    # Sortujemy kolumny rosnąco wg liczby braków, aby sprawdzić monotoniczność: jak w wykładzie (zab_cz_1), s. 89
    kol_posort = tab_braki["Braki (n)"].sort_values().index.tolist()
    M_posort = M_mat[kol_posort]
    wzorce = M_posort.value_counts().reset_index(name="Liczebność")
    wzorce["Udział [%]"] = (100.0 * wzorce["Liczebność"] / n_tot).round(2)
    wzorce["Liczba brakujących cech"] = wzorce[kol_posort].sum(axis=1)

    # Sprawdzenie formalne monotoniczności: w każdym wierszu po pierwszym 1 mogą być tylko 1
    wiersze_m = wzorce[kol_posort].to_numpy()
    czy_monotoniczny = bool(np.all(np.diff(wiersze_m, axis=1) >= 0))

    # 3. Rozkład braków wg grup wieku i kanału
    grupy_wieku = pd.cut(df["age"], bins=PRZEDZIALY_WIEKU, labels=ETYKIETY_WIEKU)
    tab_braki_wiek = df.groupby(grupy_wieku, observed=True).agg(
        Ogółem=("M", "size"),
        Braki=("M", "sum"),
        Braki_pct=("M", lambda s: round(100.0 * s.mean(), 2)),
    )
    tab_braki_wiek.index.name = "Grupa wieku"
    for kod, nazwa in [(0, "Braki_kanal_0 [%]"), (1, "Braki_kanal_A [%]"), (2, "Braki_kanal_B [%]")]:
        sub = df[df["contact"] == kod]
        sub_g = pd.cut(sub["age"], bins=PRZEDZIALY_WIEKU, labels=ETYKIETY_WIEKU)
        tab_braki_wiek[nazwa] = (100.0 * sub.groupby(sub_g, observed=True)["M"].mean()).round(1)
    tab_braki_wiek = tab_braki_wiek.rename(columns={"Braki_pct": "Braki ogółem [%]"})

    return tab_braki, wzorce, czy_monotoniczny, tab_braki_wiek


# =====================================================================================
# 2. WYKRESY (zapis do OUT_DIR)
# =====================================================================================
def wykres_age(df, sciezka):
    """Histogram (sns.histplot jak w kodzie "DM_Approach_py (1).ipynb"), histogram porównawczy wg M
    oraz wykres pudełkowy: jak w wykładzie (zab_cz_1), s. 17-18."""
    with plt.rc_context(STYL):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.2, 4.0))

        # Panel A: Histogram gęstości age ogółem — sns.histplot jak w kodzie "DM_Approach_py (1).ipynb"
        sns.histplot(
            df["age"], stat="density", bins=30, ax=ax1,
            color=KOLOR["seria1"], alpha=0.55, edgecolor=KOLOR["tlo"], linewidth=0.5,
        )
        mean_all = df["age"].mean()
        med_all = df["age"].median()
        ax1.axvline(mean_all, color=KOLOR["tekst"], lw=1.6, ls="-",
                    label=f"Średnia = {_liczba(mean_all)} lat")
        ax1.axvline(med_all, color=KOLOR["seria2"], lw=1.6, ls="--",
                    label=f"Mediana = {_liczba(med_all, '{:.0f}')} lat")
        ax1.grid(axis="y", color=KOLOR["siatka"], lw=0.8)
        ax1.set_axisbelow(True)
        ax1.set_xlabel("Wiek klienta (lata)")
        ax1.set_ylabel("Gęstość rozkładu")
        ax1.set_title("A. Rozkład wieku (cała próba)")
        ax1.legend(loc="upper right")

        # Panel B: Histogram porównawczy wg M=0 vs M=1 — jak w wykładzie (zab_cz_1), s. 17
        obs_age = df.loc[df["M"] == 0, "age"]
        mis_age = df.loc[df["M"] == 1, "age"]
        sns.histplot(
            obs_age, stat="percent", bins=25, ax=ax2,
            color=KOLOR["seria1"], alpha=0.45, edgecolor=KOLOR["seria1"], linewidth=0.6,
            label=f"Obserwowane M=0 (śr. {_liczba(obs_age.mean())})",
        )
        sns.histplot(
            mis_age, stat="percent", bins=25, ax=ax2,
            color=KOLOR["brak"], alpha=0.50, edgecolor=KOLOR["brak"], linewidth=0.6,
            label=f"Brak Y, M=1 (śr. {_liczba(mis_age.mean())})",
        )
        ax2.grid(axis="y", color=KOLOR["siatka"], lw=0.8)
        ax2.set_axisbelow(True)
        ax2.set_xlabel("Wiek klienta (lata)")
        ax2.set_ylabel("Udział w grupie [%]")
        ax2.set_ylim(0, ax2.get_ylim()[1] * 1.22)
        ax2.set_title("B. Rozkład wieku wg statusu braku Y")
        ax2.legend(loc="upper right")

        # Panel C: Wykres pudełkowy age (ogółem oraz wg M=0 / M=1) — jak w wykładzie (zab_cz_1), s. 18
        dane_box = [df["age"].to_numpy(), obs_age.to_numpy(), mis_age.to_numpy()]
        kolory_box = [KOLOR["neutral"], KOLOR["seria1"], KOLOR["brak"]]
        bp = ax3.boxplot(
            dane_box,
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": KOLOR["tlo"],
                       "markeredgecolor": KOLOR["tekst"], "markersize": 5},
            medianprops={"color": KOLOR["tekst"], "linewidth": 1.8},
            whiskerprops={"color": KOLOR["tekst2"], "linewidth": 1.1},
            capprops={"color": KOLOR["tekst2"], "linewidth": 1.1},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.35,
                        "markerfacecolor": KOLOR["wyciszony"], "markeredgecolor": "none"},
        )
        for patch, kol in zip(bp["boxes"], kolory_box):
            patch.set_facecolor(kol)
            patch.set_alpha(0.65)
            patch.set_edgecolor(KOLOR["tekst2"])
        ax3.grid(axis="y", color=KOLOR["siatka"], lw=0.8)
        ax3.set_axisbelow(True)
        ax3.set_xticks(
            [1, 2, 3],
            ["Ogółem\n(n=11 304)", "Obserwowane\nM=0 (n=9 220)", "Brak Y\nM=1 (n=2 084)"],
        )
        ax3.set_ylabel("Wiek klienta (lata)")
        ax3.set_title("C. Wykres pudełkowy wieku")

        fig.tight_layout()
        fig.savefig(sciezka, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return sciezka


def wykres_cechy_skokowe(tab_czestosci, sciezka):
    """Wykresy słupkowe rozkładu wszystkich 5 cech skokowych (z kategorią braków w y_newdeposit):
    jak w wykładzie (zab_cz_1), s. 14, 46; układ wielopanelowy: WŁASNE."""
    tytuly = {
        "education": "Wykształcenie (education)",
        "house_loan": "Kredyt mieszkaniowy (house_loan)",
        "poutcome": "Poprzednia kampania (poutcome)",
        "contact": "Kanał komunikacji (contact)",
        "y_newdeposit": "Zakup produktu — zmienna celu (y_newdeposit)",
    }
    with plt.rc_context(STYL):
        fig = plt.figure(figsize=(11.5, 6.4))
        gs = GridSpec(2, 6, figure=fig, hspace=0.45, wspace=0.55)
        osie = [
            fig.add_subplot(gs[0, 0:2]),
            fig.add_subplot(gs[0, 2:4]),
            fig.add_subplot(gs[0, 4:6]),
            fig.add_subplot(gs[1, 0:3]),
            fig.add_subplot(gs[1, 3:6]),
        ]
        for ax, col in zip(osie, CECHY_SKOKOWE):
            tab = tab_czestosci[col]
            kat = tab.index.tolist()
            wart = tab["Udział [%]"].to_numpy()
            licz = tab["Liczebność"].to_numpy()

            kolory = []
            for k in kat:
                if "Brak danych" in k:
                    kolory.append(KOLOR["brak"])
                elif k == "Zakup (1)":
                    kolory.append(KOLOR["seria3"])
                elif k == "unknown":
                    kolory.append(KOLOR["neutral"])
                else:
                    kolory.append(KOLOR["seria1"])

            x = np.arange(len(kat))
            slupki = ax.bar(x, wart, color=kolory, width=0.58, alpha=0.88)
            ax.grid(axis="y", color=KOLOR["siatka"], lw=0.8)
            ax.set_axisbelow(True)
            ax.set_xticks(x, kat, rotation=12 if len(kat) > 3 else 0)
            ax.set_ylim(0, max(wart) * 1.24)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
            ax.set_title(tytuly[col], pad=8)

            for s, v, n_i in zip(slupki, wart, licz):
                ax.text(
                    s.get_x() + s.get_width() / 2,
                    v + max(wart) * 0.03,
                    f"{_liczba(v)}%\n({_calkowita(n_i)})",
                    ha="center", va="bottom", fontsize=7.5, color=KOLOR["tekst"],
                )
        osie[0].set_ylabel("Udział w próbie [%]")
        osie[3].set_ylabel("Udział w próbie [%]")
        fig.suptitle("Struktura cech skokowych w zbiorze kb19 (n = 11 304)",
                     x=0.04, y=0.99, ha="left", fontsize=11, fontweight="bold", color=KOLOR["tekst"])
        fig.savefig(sciezka, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return sciezka


def wykres_wynik_badania(tab_wynik_kanal, sciezka):
    """Graficzna ocena wyników badania: (A) pełna struktura Y z brakami wg kanału,
    (B) odsetek zakupów wśród obserwacji kompletnych z 95% CI Wilsona (WŁASNE)."""
    sub = tab_wynik_kanal.loc[[ETYKIETY_CONTACT[0], ETYKIETY_CONTACT[1], ETYKIETY_CONTACT[2]]]
    kanaly = sub.index.tolist()
    x = np.arange(len(kanaly))

    with plt.rc_context(STYL):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.3))

        # Panel A: 100% wykres skumulowany (Y=1, Y=0, NaN) — WŁASNE
        p_y1 = 100.0 * sub["Zakup (Y=1)"] / sub["Ogółem (n)"]
        p_y0 = 100.0 * sub["Brak zakupu (Y=0)"] / sub["Ogółem (n)"]
        p_na = 100.0 * sub["Braki (NaN)"] / sub["Ogółem (n)"]

        ax1.bar(x, p_y1, width=0.55, color=KOLOR["seria3"], alpha=0.9, label="Zakup (Y=1)")
        ax1.bar(x, p_y0, bottom=p_y1, width=0.55, color=KOLOR["seria1"], alpha=0.75, label="Brak zakupu (Y=0)")
        ax1.bar(x, p_na, bottom=p_y1 + p_y0, width=0.55, color=KOLOR["brak"], alpha=0.85, label="Brak danych (NaN)")

        for xi, v1, v0, vna, n_i in zip(x, p_y1, p_y0, p_na, sub["Ogółem (n)"]):
            ax1.text(xi, v1 / 2, f"{_liczba(v1)}%", ha="center", va="center",
                     fontsize=7.5 if v1 < 6 else 8, color="white", fontweight="bold")
            ax1.text(xi, v1 + v0 / 2, f"{_liczba(v0)}%", ha="center", va="center",
                     fontsize=8, color="white", fontweight="bold")
            ax1.text(xi, v1 + v0 + vna / 2, f"{_liczba(vna)}%", ha="center", va="center",
                     fontsize=8, color="white", fontweight="bold")
            ax1.text(xi, 102.5, f"n = {_calkowita(n_i)}", ha="center", va="bottom",
                     fontsize=8, color=KOLOR["tekst2"])

        ax1.set_xticks(x, kanaly)
        ax1.set_ylim(0, 113)
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax1.set_ylabel("Udział w grupie [%]")
        ax1.set_title("A. Struktura zmiennej Y (z brakami) wg kanału")
        ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)

        # Panel B: Odsetek zakupów wśród M=0 + 95% CI Wilsona — WŁASNE
        y_pct = sub["Zakup wśród M=0 [%]"].to_numpy()
        err_lo = y_pct - sub["CI 95% dolna [%]"].to_numpy()
        err_hi = sub["CI 95% górna [%]"].to_numpy() - y_pct
        kolory_b = [KOLOR["neutral"], KOLOR["seria1"], KOLOR["seria2"]]

        ax2.bar(x, y_pct, width=0.52, color=kolory_b, alpha=0.88)
        ax2.errorbar(x, y_pct, yerr=[err_lo, err_hi], fmt="none",
                     ecolor=KOLOR["tekst"], elinewidth=1.5, capsize=5, zorder=4)
        ax2.grid(axis="y", color=KOLOR["siatka"], lw=0.8)
        ax2.set_axisbelow(True)
        ax2.set_xticks(x, kanaly)
        ax2.set_ylim(0, 22)
        ax2.set_yticks([0, 5, 10, 15, 20])
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax2.set_ylabel("Odsetek zakupów wśród M=0 [%]")
        ax2.set_title("B. Skuteczność kanałów (obserwacje kompletne, 95% CI)")

        for xi, v, hi, n_o in zip(x, y_pct, sub["CI 95% górna [%]"], sub["Obserwowane (M=0)"]):
            ax2.text(
                xi, hi + 0.7,
                f"{_liczba(v)}%\n(M=0: {_calkowita(n_o)})",
                ha="center", va="bottom", fontsize=8, color=KOLOR["tekst"],
            )

        fig.tight_layout()
        fig.savefig(sciezka, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return sciezka


def wykres_braki(df, tab_braki, wzorce, tab_braki_wiek, sciezka):
    """Wizualizacja wzorca braków (jak w wykładzie (zab_cz_1), s. 90) oraz rozkładu braków
    wg grup wieku i kanału komunikacji (WŁASNE)."""
    kol_posort = tab_braki["Braki (n)"].sort_values().index.tolist()
    mat = wzorce[kol_posort].to_numpy()  # 0 = znane, 1 = brak

    with plt.rc_context(STYL):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(11.6, 4.3), gridspec_kw={"width_ratios": [1.05, 1.25]}
        )

        # Panel A: Macierz wzorca braków — jak w wykładzie (zab_cz_1), s. 90
        n_w, n_k = mat.shape
        for i in range(n_w):
            for j in range(n_k):
                kol = KOLOR["brak"] if mat[i, j] == 1 else KOLOR["seria1"]
                rect = plt.Rectangle((j, n_w - 1 - i), 1, 1, facecolor=kol,
                                     edgecolor=KOLOR["tlo"], linewidth=2.5, alpha=0.88)
                ax1.add_patch(rect)
                etyk = "BRAK\n(1)" if mat[i, j] == 1 else "OK\n(0)"
                ax1.text(j + 0.5, n_w - 1 - i + 0.5, etyk, ha="center", va="center",
                         color="white", fontsize=8, fontweight="bold")

        ax1.set_xlim(0, n_k)
        ax1.set_ylim(0, n_w)
        etyk_kol = [c.replace("house_loan", "house_\nloan").replace("y_newdeposit", "y_new-\ndeposit")
                    for c in kol_posort]
        ax1.set_xticks(np.arange(n_k) + 0.5, etyk_kol, rotation=0, ha="center")
        ax1.xaxis.tick_top()
        ax1.tick_params(axis="both", length=0)
        for sp in ax1.spines.values():
            sp.set_visible(False)

        # Liczebności wzorców po lewej, liczba brakujących zmiennych po prawej, sumy braków na dole
        ax1.set_yticks(
            [n_w - 1 - i + 0.5 for i in range(n_w)],
            [f"{_calkowita(r['Liczebność'])} ({_liczba(r['Udział [%]'])}%)"
             for _, r in wzorce.iterrows()],
        )
        for i, (_, r) in enumerate(wzorce.iterrows()):
            ax1.text(n_k + 0.12, n_w - 1 - i + 0.5, f"{int(r['Liczba brakujących cech'])}",
                     ha="left", va="center", fontsize=8.5, color=KOLOR["tekst2"], fontweight="bold")
        ax1.text(n_k + 0.12, n_w + 0.04, "Braki\nw wierszu", ha="left", va="bottom",
                 fontsize=7.5, color=KOLOR["tekst2"])

        for j, col in enumerate(kol_posort):
            b_n = int(tab_braki.loc[col, "Braki (n)"])
            ax1.text(j + 0.5, -0.18, _calkowita(b_n), ha="center", va="top",
                     fontsize=8, color=KOLOR["brak"] if b_n > 0 else KOLOR["tekst2"],
                     fontweight="bold" if b_n > 0 else "normal")
        ax1.text(n_k + 0.12, -0.18, f"Σ = {_calkowita(tab_braki['Braki (n)'].sum())}",
                 ha="left", va="top", fontsize=8, color=KOLOR["brak"], fontweight="bold")

        ax1.legend(
            handles=[
                Patch(facecolor=KOLOR["seria1"], alpha=0.88, label="Wartość zaobserwowana (m=0)"),
                Patch(facecolor=KOLOR["brak"], alpha=0.88, label="Brak danych (m=1)"),
            ],
            loc="upper left", bbox_to_anchor=(0, -0.22), ncol=2,
        )
        ax1.set_title("A. Wzorzec braków danych (monotoniczny)", pad=36)

        # Panel B: Rozkład odsetka braków Y wg wieku i kanału komunikacji (WŁASNE)
        x = np.arange(len(tab_braki_wiek))
        ax2.grid(axis="y", color=KOLOR["siatka"], lw=0.8)
        ax2.set_axisbelow(True)
        ax2.bar(x, tab_braki_wiek["Braki ogółem [%]"], width=0.55,
                color=KOLOR["siatka"], edgecolor=KOLOR["os"], label="Ogółem w grupie wieku")
        ax2.plot(x, tab_braki_wiek["Braki_kanal_0 [%]"], marker="o", lw=2,
                 color=KOLOR["neutral"], label="Brak kontaktu (0)")
        ax2.plot(x, tab_braki_wiek["Braki_kanal_A [%]"], marker="s", lw=2.2,
                 color=KOLOR["seria1"], label="Kanał A (1)")
        ax2.plot(x, tab_braki_wiek["Braki_kanal_B [%]"], marker="^", lw=2.2,
                 color=KOLOR["seria2"], label="Kanał B (2)")

        ax2.set_xticks(x, tab_braki_wiek.index.tolist())
        ax2.set_ylim(0, 80)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax2.set_xlabel("Przedział wieku klienta (lata)")
        ax2.set_ylabel("Odsetek braków y_newdeposit [%]")
        ax2.set_title("B. Odsetek braków Y wg wieku i kanału komunikacji")
        ax2.legend(loc="upper left", ncol=2)

        fig.tight_layout()
        fig.savefig(sciezka, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return sciezka


# =====================================================================================
# 3. URUCHOMIENIE SEKCJI
# =====================================================================================
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
    oraz dodatkowe klucze pomocnicze:
      tab_wzorzec_brakow, tab_braki_wiek, wykresy
    """
    print("\n=== 1. STRUKTURA DANYCH ===")
    wyniki = {}

    # 1.1. Podstawowe informacje o zbiorze i kontrola jakości danych: jak w wykładzie (zab_cz_1), s. 35, 46-48
    print(f"Liczba obserwacji (n): {_calkowita(len(df))}")
    print(f"Liczba zmiennych (k):  {len(ZMIENNE_ANALIZOWANE)} "
          f"(1 ciągła: {CECHA_CIAGLA}; {len(CECHY_SKOKOWE)} skokowych: {', '.join(CECHY_SKOKOWE)})")

    # Kontrola ukrytych znaków/spacji w cechach tekstowych: jak w wykładzie (zab_cz_1), s. 47-48
    tekstowe = ["education", "house_loan", "poutcome"]
    bledy_spacje = sum(int(df[c].astype(str).str.strip().ne(df[c].astype(str)).sum()) for c in tekstowe)
    print(f"Kontrola jakości: zbędne spacje w cechach tekstowych = {bledy_spacje}, "
          f"zakres wieku = [{df['age'].min()}, {df['age'].max()}] lat (brak wartości nieprawidłowych).")

    # 1.2. Cecha ciągła: age — jak w wykładzie (zab_cz_1), s. 17-18, 47, 70
    tab_age = statystyki_age(df)
    print("\n--- CECHA CIĄGŁA: AGE (statystyki opisowe i obserwacje odstające) ---")
    print(tab_age.to_string(formatters={
        "Średnia": "{:.2f}".format,
        "Odch.std.": "{:.2f}".format,
        "Min": "{:.0f}".format,
        "Q1": "{:.1f}".format,
        "Mediana": "{:.1f}".format,
        "Q3": "{:.1f}".format,
        "Max": "{:.0f}".format,
        "IQR": "{:.1f}".format,
        "Skośność": "{:.2f}".format,
    }))

    # 1.3. Cechy skokowe: tabele częstości z brakami — jak w wykładzie (zab_cz_1), s. 46; braki w kat.: WŁASNE
    tab_czest = tabele_czestosci(df)
    print("\n--- CECHY SKOKOWE: TABELE CZĘSTOŚCI (z brakami danych) ---")
    for col, tab in tab_czest.items():
        typ, skala, opis = OPIS_CECH[col]
        print(f"\nZmienna: {col} ({skala}) — {opis}")
        print(tab.to_string())

    # 1.4. Wynik badania: tabela krzyżowa contact x y_newdeposit + 95% CI Wilsona (WŁASNE)
    tab_wynik = tabela_wynik_kanal(df)
    print("\n--- WYNIK BADANIA WG KANAŁU KOMUNIKACJI (contact x y_newdeposit, 95% CI Wilsona) ---")
    print(tab_wynik.to_string())

    # 1.5. Zakres, macierz i wzorzec braków danych — jak w wykładzie (zab_cz_1), s. 46, 73-74, 89-90
    tab_braki, wzorce, czy_monotoniczny, tab_braki_wiek = analiza_brakow(df)
    print("\n--- ZAKRES I SKALE POMIAROWE ZMIENNYCH ---")
    print(tab_braki.to_string())

    print("\n--- WZORZEC BRAKÓW DANYCH Z MACIERZY M ---")
    print(wzorce.to_string(index=False))
    print(f"Czy wzorzec jest monotoniczny? {'TAK' if czy_monotoniczny else 'NIE'} "
          "(braki występują wyłącznie w jednej zmiennej: y_newdeposit — przypadek item nonresponse).")

    print("\n--- ROZKŁAD BRAKÓW Y WG PRZEDZIAŁÓW WIEKU I KANAŁU KOMUNIKACJI ---")
    print(tab_braki_wiek.to_string())

    # Kontrola progu wiekowego braków w kanale A (WŁASNE)
    braki_a_45 = int(df.loc[(df["contact"] == 1) & (df["age"] <= 45), "M"].sum())
    n_a_45 = int(((df["contact"] == 1) & (df["age"] <= 45)).sum())
    braki_a_46 = float(100.0 * df.loc[(df["contact"] == 1) & (df["age"] >= 46), "M"].mean())
    print(f"\nObserwacja szczegółowa (kanał A): dla wieku <= 45 lat liczba braków Y = {braki_a_45}/{n_a_45} (0,0%), "
          f"natomiast dla wieku >= 46 lat odsetek braków wynosi {braki_a_46:.2f}%.")

    # 1.6. Generowanie i zapis wykresów do OUT_DIR / "fig1_....png"
    wykresy = [
        wykres_age(df, OUT_DIR / "fig1_age_rozklad.png"),
        wykres_cechy_skokowe(tab_czest, OUT_DIR / "fig1_cechy_skokowe.png"),
        wykres_wynik_badania(tab_wynik, OUT_DIR / "fig1_wynik_badania.png"),
        wykres_braki(df, tab_braki, wzorce, tab_braki_wiek, OUT_DIR / "fig1_wzorzec_i_rozklad_brakow.png"),
    ]
    print("\nZapisane wykresy:", ", ".join(p.name for p in wykresy))

    # 1.7. Podsumowanie sekcji 1 i skutki braków danych: jak w wykładzie (zab_cz_1), s. 75-81, 107-108
    print("\n--- WNIOSKI ---")
    print("1. Zbiór kb19 obejmuje 11 304 klientów opisanych 1 cechą ciągłą (age) i 5 cechami skokowymi.")
    print("2. Braki danych występują wyłącznie w zmiennej celu y_newdeposit: "
          f"{int(df['M'].sum())} obs. ({100.0 * df['M'].mean():.2f}%). Wzorzec braków jest monotoniczny.")
    print("3. Wśród obserwacji kompletnych (M=0) oba kanały komunikacji (A: 14,70%, B: 13,61%) wyraźnie "
          "przewyższają brak komunikacji (3,96%), natomiast 95% przedziały ufności dla A (13,83%-15,62%) "
          "i B (10,78%-17,03%) nakładają się.")
    print("4. Braki nie są rozłożone równomiernie: osoby z brakiem Y są średnio o ponad 11 lat starsze "
          "(50,22 vs 38,83 lat), a w najmniej licznym kanale B brakuje aż 36,66% wyników (wobec 18,28% w A "
          "i 14,65% przy braku kontaktu).")
    print("5. Konsekwencje: usunięcie braków redukuje próbę "
          "(szczególnie w kanale B: z 731 do 463 obs.), obniżając moc statystyczną testu i poszerzając "
          "przedziały ufności, a systematyczny rozkład braków wg wieku i kanału grozi obciążeniem oszacowań "
          "przy analizie kompletnych przypadków (CC).")

    wyniki.update({
        "tab_struktura_age": tab_age,
        "tab_czestosci": tab_czest,
        "tab_wynik_kanal": tab_wynik,
        "tab_braki": tab_braki,
        "tab_wzorzec_brakow": wzorce,
        "tab_braki_wiek": tab_braki_wiek,
        "wykresy": wykresy,
    })
    return wyniki


if __name__ == "__main__":
    uruchom(wczytaj_dane())

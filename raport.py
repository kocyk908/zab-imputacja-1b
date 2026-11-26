# -*- coding: utf-8 -*-
import base64
import html
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from wspolne import ALPHA, DATA_FILE, FORMULA, M_IMP, OUT_DIR, SEED

NAZWY = {
    "Intercept": "β0: wyraz wolny",
    "C(contact, Treatment(0))[T.1]": "β1: CONTACT_A (kanał A)",
    "C(contact, Treatment(0))[T.2]": "β2: CONTACT_B (kanał B)",
    "age": "β3: AGE",
    "C(house_loan, Treatment('yes'))[T.no]": "β4: HOUSING_LOAN = no",
    "C(poutcome, Treatment('unknown'))[T.failure]": "β5: POUTCOME = failure",
    "C(poutcome, Treatment('unknown'))[T.success]": "β6: POUTCOME = success",
    "C(poutcome, Treatment('unknown'))[T.other]": "β7: POUTCOME = other",
}

CSS = (Path(__file__).with_name("raport.css")).read_text(encoding="utf-8")


def _n(v, d=2):
    return f"{v:,.{d}f}".replace(",", " ").replace(".", ",")


def _p(p):
    return "&lt; 0,001" if p < 0.001 else _n(p, 3)


def _pz(p):
    """Cały zapis wartości p do tekstu: "p = 0,438" albo "p < 0,001"."""
    return "p &lt; 0,001" if p < 0.001 else f"p = {_n(p, 3)}"


M_STAB = 40

KOLUMNY_P = ("p", "p-value", "P>|z|", "P>|t|")


def _proc(v, d=1):
    return _n(100 * v, d) + "%"


def _tbl(d, dec=4, index=True, calkowite=(), podpis=None):
    if podpis:
        return f'<p class="podpis-tab">{html.escape(podpis)}</p>' + _tbl(d, dec, index, calkowite)
    d = d.copy()
    for c in d.columns:
        if c in KOLUMNY_P and pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "-" if pd.isna(v) else ("< 0,001" if v < 0.001 else _n(v, 3)))
        elif c in calkowite:
            d[c] = d[c].map(lambda v: "-" if pd.isna(v) else f"{v:,.0f}".replace(",", " "))
        elif pd.api.types.is_integer_dtype(d[c]):
            d[c] = d[c].map(lambda v: f"{v:,}".replace(",", " "))
        elif pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "-" if pd.isna(v) else _n(v, dec))
    return d.to_html(index=index, border=0, classes="t", escape=True)


def _img(nazwa, podpis, max_szer=None):
    sciezka = OUT_DIR / nazwa
    if not sciezka.exists():
        return f'<p class="todo">Brak wykresu <code>{html.escape(nazwa)}</code>.</p>'
    b64 = base64.b64encode(sciezka.read_bytes()).decode()
    styl = f' style="max-width:min(100%,{max_szer}px)"' if max_szer else ""
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(podpis)}"{styl}>'
            f"<figcaption>{podpis}</figcaption></figure>")


def _parametry(tab, kol_p):
    t = pd.DataFrame({"β": tab["Coef."], "SE": tab["Std.Err."], "p": tab[kol_p],
                      "OR": np.exp(tab["Coef."]), "OR 95% CI: od": np.exp(tab["[0.025"]),
                      "do": np.exp(tab["0.975]"])})
    t = t.reindex([k for k in NAZWY if k in t.index] + [k for k in t.index if k not in NAZWY])
    t.index = [NAZWY.get(i, i) for i in t.index]
    return t


def _autorzy():
    plik = Path(__file__).with_name("autorzy.txt")
    if not plik.exists():
        return ""
    osoby = []
    for linia in plik.read_text(encoding="utf-8").splitlines():
        pola = linia.split()
        if len(pola) >= 3:
            osoby.append(f"{html.escape(pola[1])} {html.escape(pola[0])} ({html.escape(pola[2])})")
    if not osoby:
        return ""
    lista = "".join(f"<li>{o}</li>" for o in osoby)
    return f'<p class="sub" style="margin-bottom:0"><b>Skład grupy:</b></p><ul class="fakty sub">{lista}</ul>'


def _sekcja1(w1):
    h = ['<h2 id="s1">1. Analiza struktury</h2>']
    if not w1:
        return "\n".join(h)
    wymagane = ("tab_struktura_age", "tab_czestosci", "tab_wynik_kanal", "tab_braki", "tab_wzorzec_brakow",
                "tab_braki_wiek")
    if all(k in w1 for k in wymagane):
        return _sekcja1_pelna(w1, h)
    for klucz, wartosc in w1.items():
        if isinstance(wartosc, pd.DataFrame):
            h.append(f"<h3>{html.escape(klucz)}</h3>{_tbl(wartosc, 3)}")
        elif isinstance(wartosc, pd.Series):
            h.append(f"<h3>{html.escape(klucz)}</h3>{_tbl(wartosc.to_frame(), 3)}")
        elif isinstance(wartosc, dict):
            for k2, v2 in wartosc.items():
                if isinstance(v2, (pd.DataFrame, pd.Series)):
                    v2 = v2.to_frame() if isinstance(v2, pd.Series) else v2
                    h.append(f"<h3>{html.escape(klucz)}: {html.escape(str(k2))}</h3>{_tbl(v2, 3)}")
    for png in sorted(OUT_DIR.glob("fig1*.png")):
        h.append(_img(png.name, png.stem))
    return "\n".join(h)


TYTULY_CZESTOSCI = {"education": "education (wykształcenie)", "house_loan": "house_loan (kredyt mieszkaniowy)",
                    "poutcome": "poutcome (wynik poprzedniej kampanii)", "contact": "contact (kanał komunikacji)",
                    "y_newdeposit": "y_newdeposit (zakup lokaty, zmienna celu)"}


def _sekcja1_pelna(w1, h):
    ta, tw, tb = w1["tab_struktura_age"], w1["tab_wynik_kanal"], w1["tab_braki"]
    tw_ = w1["tab_braki_wiek"]
    n = int(ta.loc["Ogółem", "n"])
    yb = tb.loc["y_newdeposit"]
    sr0, sr1 = ta.loc["Obserwowane (M=0)", "Średnia"], ta.loc["Brak Y (M=1)", "Średnia"]
    kan = {k: tw.loc[k] for k in tw.index}
    A, B, Z = kan["Kanał A (1)"], kan["Kanał B (2)"], kan["Brak kontaktu (0)"]
    mono = len(w1["tab_wzorzec_brakow"]) <= 2

    h.append(
        f"<p>Zbiór <code>kb19</code> liczy <b>{_n(n, 0)}</b> klientów opisanych przez 6 zmiennych, w tym jedną ciągłą (<code>age</code>) "
        "i pięć skokowych (<code>education</code>, <code>house_loan</code>, <code>poutcome</code>, "
        "<code>contact</code>, <code>y_newdeposit</code>).</p>")

    h.append("<h3>1.1. Zakres braków i skale pomiarowe</h3>")
    h.append(_tbl(tb, 2, calkowite=("Obserwowane", "Braki (n)"), podpis="Tabela 1.1. Zakres braków danych i skale pomiarowe zmiennych"))
    h.append(
        f"<p>Braki występują <b>wyłącznie w zmiennej celu</b> <code>y_newdeposit</code> i obejmują {_n(yb['Braki (n)'], 0)} obs. "
        f"({_n(yb['Braki [%]'], 1)}%). Jest to brak odpowiedzi na pytanie (<i>item nonresponse</i>), a wzorzec braków "
        f"z macierzy M jest {'<b>monotoniczny</b>' if mono else 'niemonotoniczny'}. Pozwala to imputować jedną "
        "zmienną modelem logitowym (odpowiednik <code>PROC MI MONOTONE LOGISTIC</code>).</p>")
    h.append(_tbl(w1["tab_wzorzec_brakow"], 2, index=False, podpis="Tabela 1.2. Wzorce braków danych z macierzy M (1 = brak)"))

    h.append("<h3>1.2. Wiek (AGE) jako cecha ciągła</h3>")
    h.append(_tbl(ta, 2, calkowite=("n", "Min", "Max", "Odstające (IQR)", "Odstające (3σ)"),
                   podpis="Tabela 1.3. Statystyki opisowe wieku (AGE) ogółem i wg braku odpowiedzi"))
    h.append(
        f"<p>Klienci z brakiem odpowiedzi są średnio <b>o {_n(sr1 - sr0, 1)} roku starsi</b> "
        f"({_n(sr1, 2)} vs {_n(sr0, 2)} lat). To pierwszy sygnał, że braki nie są losowe (MCAR) i zależą od "
        "obserwowanych cech. Formalnie sprawdza to rozdział 2.</p>")
    h.append(_img("fig1_age_rozklad.png", "Rys. 1.1. Rozkład wieku ogółem i wg braku odpowiedzi (M)"))

    h.append("<h3>1.3. Rozkłady cech skokowych i odsetek braków Y w kategoriach</h3>")
    for nr, (col, t) in enumerate(w1["tab_czestosci"].items(), start=4):
        h.append(_tbl(t, 2, calkowite=("Liczebność", "Braki Y (n)"),
                      podpis=f"Tabela 1.{nr}. Rozkład zmiennej {TYTULY_CZESTOSCI.get(col, col)}"))
    nr_tab = 4 + len(w1["tab_czestosci"])
    h.append(_img("fig1_cechy_skokowe.png", "Rys. 1.2. Rozkłady cech skokowych i odsetek braków Y w kategoriach"))

    h.append("<h3>1.4. Wynik badania wg kanału komunikacji</h3>")
    h.append(_tbl(tw, 2, calkowite=("Ogółem (n)", "Brak zakupu (Y=0)", "Zakup (Y=1)", "Braki (NaN)",
                                    "Obserwowane (M=0)"),
                  podpis=f"Tabela 1.{nr_tab}. Wynik badania wg kanału komunikacji (z 95% przedziałami Wilsona)"))
    h.append(
        f"<p>Wśród obserwowanych odpowiedzi kupuje {_n(A['Zakup wśród M=0 [%]'], 2)}% klientów kanału A i "
        f"{_n(B['Zakup wśród M=0 [%]'], 2)}% kanału B, a bez kontaktu tylko {_n(Z['Zakup wśród M=0 [%]'], 2)}%. "
        f"95% przedziały Wilsona dla A ({_n(A['CI 95% dolna [%]'], 2)}-{_n(A['CI 95% górna [%]'], 2)}%) i B "
        f"({_n(B['CI 95% dolna [%]'], 2)}-{_n(B['CI 95% górna [%]'], 2)}%) <b>nakładają się</b>. Najmniejszy kanał B ma przy tym najwięcej braków, aż <b>{_n(B['Braki [%]'], 2)}%</b>, wobec "
        f"{_n(A['Braki [%]'], 2)}% w kanale A i {_n(Z['Braki [%]'], 2)}% przy braku kontaktu. Po usunięciu braków (CC) zostaje w nim "
        f"{int(B['Obserwowane (M=0)'])} z {int(B['Ogółem (n)'])} obserwacji.</p>")
    h.append(_img("fig1_wynik_badania.png", "Rys. 1.3. Odsetek zakupów wg kanału (M=0) z 95% CI i udział braków"))

    h.append("<h3>1.5. Rozkład braków wg wieku i kanału</h3>")
    h.append(_tbl(tw_, 1, calkowite=("Ogółem", "Braki"),
                  podpis=f"Tabela 1.{nr_tab + 1}. Braki Y wg grup wieku ogółem i w kanałach komunikacji"))
    a_mlodzi = tw_.loc[[i for i in tw_.index if i in ("18-30", "31-40")], "Braki_kanal_A [%]"]
    if len(a_mlodzi) and (a_mlodzi == 0).all():
        a = tw_["Braki_kanal_A [%]"]
        polowa = a[a > 50]
        h.append(f"<p>W kanale A <b>nie ma ani jednego braku do 45. roku życia</b>, a w grupach "
                 f"{polowa.index[0]} i starszych brakuje ponad połowy odpowiedzi ({_n(polowa.min(), 1)}-"
                 f"{_n(polowa.max(), 1)}%). Brak odpowiedzi zależy więc silnie od wieku i kanału, czyli od cech "
                 "obserwowanych. Jest to niespójne z MCAR i zgodne z MAR. Zależność od wieku nie jest "
                 "liniowa, dlatego w modelu imputacyjnym (punkt 3b) i w modelu braków dla IPW (punkt 4b) "
                 "sprawdzono dodatkowo kwadrat wieku (AGE²), który pozwala opisać zależność zakrzywioną.</p>")
    h.append(_img("fig1_wzorzec_i_rozklad_brakow.png", "Rys. 1.4. Wzorzec braków (macierz M) i rozkład braków "
                                                       "wg wieku i kanału"))
    return "\n".join(h)


def _nazwy_zmiennych(formula):
    return [c.strip().removeprefix("C(").removesuffix(")") for c in formula.split("+")]


def _sekcja2(df, w2):
    tt = w2["tab_testy_brakow"].set_index("Zmienna")
    model = w2.get("model_brakow")
    tab_m = model.summary2().tables[1] if model is not None else w2.get("model_brakow_tab")
    tab_m = pd.DataFrame({"β": tab_m["Coef."], "SE": tab_m["Std.Err."], "p": tab_m["P>|z|"],
                          "OR braku": np.exp(tab_m["Coef."])})
    brak_wiek = df.groupby(pd.cut(df["age"], [17, 39, 49, 100], labels=["do 39", "40-49", "50+"]),
                           observed=True)["M"].mean()
    brak_kanal = df.groupby("contact")["M"].mean()
    return f"""
<h2 id="s2">2. Zależność prawdopodobieństwa braku danych od pozostałych cech</h2>
<p>Zdefiniowano indykator braku <code>M</code> (M = 1, gdy Y_NEWDEPOSIT jest brakujące) i oceniono
jego związek z pozostałymi cechami dwiema metodami Balickiego oraz modelem logitowym skłonności do braku.</p>
<h3>2a. Testy jednowymiarowe</h3>
<p class="src">Dla wieku (AGE) zastosowano test t Welcha (metoda I), a dla cech skokowych test χ² z miarą V Craméra (metoda II).</p>
{_tbl(tt, 4, podpis="Tabela 2.1. Testy zależności braku Y od pozostałych cech")}
<p>Odsetek braków rośnie z wiekiem, od {_proc(brak_wiek.iloc[0])} do 39 lat przez {_proc(brak_wiek.iloc[1])} w wieku
40-49 do {_proc(brak_wiek.iloc[2])} od 50 lat. Różni się też między kanałami i wynosi {_proc(brak_kanal.loc[0])} przy
braku kontaktu, {_proc(brak_kanal.loc[1])} w kanale A i {_proc(brak_kanal.loc[2])} w kanale B.</p>
<h3>2b. Model logitowy P(M = 1 | X)</h3>
{_tbl(tab_m, 4, podpis="Tabela 2.2. Model logitowy prawdopodobieństwa braku Y")}
<p>Do modelu imputacyjnego przekazano zmienne {', '.join(_nazwy_zmiennych(w2['zmienne_imputacji']))}.</p>
"""


def _sekcja3(df, w2, w3):
    cc, mi, wyb = w3["wynik_cc"], w3["wynik_mi"], w3["wybor_modelu"]
    n_cc = int((df["M"] == 0).sum())
    n_brak = int(df["M"].sum())
    return f"""
<h2 id="s3">3. Model logitowy przy założeniach MCAR i MAR</h2>
<p>Model z treści zadania:</p>
<div class="wzor"><span>logit P(Y_NEWDEPOSIT = 1 | X) = β0 + β1·CONTACT_A + β2·CONTACT_B + β3·AGE</span><br>
<span>+ β4·HOUSING_LOAN + β5·POUTCOME_failure</span><br>
<span>+ β6·POUTCOME_success + β7·POUTCOME_other</span></div>
<p>gdzie CONTACT_A i CONTACT_B to zmienne 0/1 dla kanału A i B (kategoria odniesienia to brak komunikacji, CONTACT = 0),
HOUSING_LOAN = 1, gdy klient nie ma kredytu mieszkaniowego (kategoria odniesienia yes), a POUTCOME_failure, _success i
_other to zmienne 0/1 dla wyniku poprzedniej kampanii (kategoria odniesienia unknown). Odniesienie dla CONTACT wybrano
inaczej niż w przykładowym kodzie R/SAS (tam kanał B), tak aby β1 i β2 odpowiadały kanałom A i B jak we wzorze
z zadania. Na test różnicy β1 - β2 nie ma to wpływu.</p>
<p>Hipoteza i miara efektu:</p>
<div class="wzor">H0: β1 - β2 = 0 &nbsp;&nbsp;&nbsp;&nbsp; OR(A vs B) = exp(β1 - β2)<br>
Var(β̂1 - β̂2) = Var(β̂1) + Var(β̂2) - 2·Cov(β̂1, β̂2)</div>
<p>gdzie OR(A vs B) to iloraz szans zakupu w kanale A względem kanału B, a Var i Cov to wariancje i kowariancja
oszacowań parametrów.</p>
<p class="src">Poniżej ten sam model w zapisie kodu (składnia formuł statsmodels). Po lewej od „~” stoi zmienna
objaśniana, po prawej zmienne objaśniające, a <code>C()</code> oznacza zmienną kategoryczną.</p>
<span class="f">{html.escape(FORMULA)}</span>

<h3>3a. Analiza kompletnych przypadków (CC) przy MCAR, n = {n_cc}</h3>
<p>Przy MCAR rekordy z brakiem można pominąć, bo kompletne przypadki są podpróbą losową.</p>
{_tbl(_parametry(w3['tab_param_cc'], 'P>|z|'), 4, podpis='Tabela 3.1. Oszacowania parametrów modelu przy MCAR (CC)')}
<p>Kontrast A vs B w analizie CC wynosi β̂1 - β̂2 = {_n(cc['est'], 4)} (SE = {_n(cc['se'], 4)}), {_pz(cc['p'])};
OR = {_n(cc['OR'], 3)} [{_n(cc['OR_low'], 3)}; {_n(cc['OR_high'], 3)}].</p>

<h3>3b. Imputacja wielokrotna (MI) przy MAR i specyfikacja modelu imputacyjnego</h3>
<p>Imputowana jest binarna zmienna Y_NEWDEPOSIT, w której brakuje {n_brak} wartości. Braki są tylko w Y, więc
wzorzec jest monotoniczny i wystarcza jedno równanie logistyczne, odpowiednik <code>PROC MI; MONOTONE LOGISTIC</code>.</p>
<p>Imputacja uwzględnia dwa źródła losowości. Najpierw dopasowano model logitowy na rekordach z obserwowanym Y.
Następnie w każdej imputacji losowano parametry β* ~ N(β̂, V̂(β̂)), czyli z przybliżonego rozkładu a posteriori,
a potem wartości Y* ~ Bernoulli(expit(x′β*)) dla rekordów z brakiem.</p>
<p>Predyktorami są wszystkie zmienne modelu z zadania, zmienna pomocnicza EDUCATION (rozdział 2) oraz kwadrat wieku
(AGE²), który opisuje nieliniowy wpływ wieku. O dodaniu kwadratu wieku zdecydowało porównanie modeli. AIC spadło
z {_n(wyb['AIC_podstawowy'], 1)} bez AGE² do {_n(wyb['AIC_age2'], 1)} z AGE², a test LR = {_n(wyb['LRT'], 2)}
(df = {wyb['df_LRT']}, {_pz(wyb['p_LRT'])}). Pełny model imputacyjny w zapisie kodu ma postać</p>
<span class="f">y_newdeposit ~ {html.escape(w3['model_imputacyjny'])}</span>
<p>Wykonano m = {M_IMP} imputacji z ziarnem losowania {SEED}. Na każdym z m zbiorów dopasowano model z zadania,
a wyniki połączono regułami Rubina (z poprawionymi stopniami swobody).</p>
{_tbl(_parametry(w3['tab_param_mi'], 'P>|t|'), 4, podpis=f'Tabela 3.2. Oszacowania parametrów modelu przy MAR (MI, m = {M_IMP})')}
<p>Kontrast A vs B po imputacji wielokrotnej wynosi β̂1 - β̂2 = {_n(mi['est'], 4)} (SE = {_n(mi['se'], 4)}), df = {_n(mi['df'], 0)},
{_pz(mi['p'])}; OR = {_n(mi['OR'], 3)} [{_n(mi['OR_low'], 3)}; {_n(mi['OR_high'], 3)}].</p>
"""


def _sekcja4(w3, w4, w5):
    tp = w4["tab_tipping_point"].copy()
    tp_druk = pd.DataFrame({
        "scenariusz": tp["scenariusz"], "kierunek": tp["kierunek"],
        "δ zwrotne": tp["delta_zwrotne"], "szansa zakupu u braków ×": tp["szansa_x"],
        "OR A vs B w punkcie": tp["OR"], "% zakupów braki A": 100 * tp["zakup_braki_A"],
        "% zakupów braki B": 100 * tp["zakup_braki_B"]}).set_index(["scenariusz", "kierunek"])
    tw = w4["tab_wrazliwosc"]
    ist_mar = bool(tw.query("delta_A == 0 and delta_B == 0")["istotny"].iloc[0])
    bez_zmian = int((tw["istotny"] == ist_mar).sum())
    diag_or = tw.query("delta_A == delta_B")["OR"]
    ipw_tab = w4["tab_ipw"][["model braków", "wagi", "max waga 1/phi", "ESS", "OR", "OR_low", "OR_high", "p"]].rename(
        columns={"max waga 1/phi": "max waga 1/φ", "ESS": "efektywna liczebność", "OR_low": "95% CI: od",
                 "OR_high": "do"})
    ipw_tab["model braków"] = ipw_tab["model braków"].str.replace("sekcja", "rozdział")
    wyb = w4["wybor_modelu_brakow"]
    ipw = w4["wynik_ipw"]
    por = w5["tab_porownanie"].rename(columns={"beta1-beta2": "β1 - β2", "OR A vs B": "OR (A vs B)",
                                               "OR_low": "95% CI: od", "OR_high": "do"})
    return f"""
<h2 id="s4">4. Wyniki modelu w kontekście braków danych</h2>
<p>Założeń o mechanizmie braków nie da się zweryfikować, dlatego wyniki otrzymane przy różnych założeniach
porównano ze sobą (analiza wrażliwości).</p>
{_tbl(por, 4, podpis="Tabela 4.1. Kontrast A vs B wg metody obsługi braków")}
{_img('fig_porownanie_metod.png', 'Rys. 4.1. OR kanał A vs kanał B (95% CI) wg metody obsługi braków. Pionowa linia OR = 1 oznacza brak różnicy między kanałami, a przedział, który ją przecina, oznacza różnicę nieistotną')}

<h3>4a. Analiza wrażliwości na MNAR (tipping point)</h3>
<p>Do logitu modelu imputacyjnego dodano przesunięcie δ, osobno dla braków w kanale A (δ_A) i w kanale B (δ_B).
Oznacza to, że osoby bez wyniku mają exp(δ) razy inną szansę zakupu, niż przewiduje model przy MAR. Siatka δ_A × δ_B od -2
do 2 (81 komórek, po 20 imputacji).</p>
{_img('fig_wrazliwosc.png', 'Rys. 4.2. OR kanał A vs B na siatce δ_A × δ_B', max_szer=680)}
{_img('fig_wrazliwosc_scenariusze.png', 'Rys. 4.3. OR kanał A vs B w trzech przekrojach siatki z punktami zwrotnymi')}
{_tbl(tp_druk, 2, podpis="Tabela 4.2. Punkty zwrotne analizy wrażliwości")}
<p class="src">„-” oznacza, że w tym kierunku wniosek nie zmienia się w całym badanym zakresie δ ∈ [-2; 2].
δ zwrotne interpolowane liniowo między punktami siatki (krok 0,5).</p>
<ul>
<li>Wniosek z MAR utrzymuje się w <b>{bez_zmian} z {len(tw)}</b> komórek siatki. W punkcie MAR (δ_A = δ_B = 0)
OR = {_n(float(tw.query("delta_A == 0 and delta_B == 0")["OR"].iloc[0]))}, a nie {_n(w3['wynik_mi']['OR'])} jak w punkcie 3b,
bo na komórkę przypada 20 imputacji zamiast {M_IMP}; różnica mieści się w błędzie losowania.</li>
<li>Gdy osoby bez wyniku w obu kanałach odbiegają od MAR tak samo, OR wynosi {_n(diag_or.min())}-{_n(diag_or.max())},
więc wniosek się nie zmienia.</li>
<li>Wniosek zmienia się przy odstępstwie dotyczącym jednego kanału, czyli gdy
{'; gdy '.join(_wrazliwosc_opis(w4)[0])}.</li>
</ul>

<h3>4b. Ważenie odwrotnością prawdopodobieństwa odpowiedzi (IPW)</h3>
<p>IPW to inna metoda przy tym samym założeniu MAR, w której nie imputuje się Y, tylko waży kompletne przypadki przez 1/φ_i, gdzie
φ_i = P(M = 0 | X) (estymator Horvitza-Thompsona z modelem skłonności do odpowiedzi). Model braków z rozdziału 2 (wiek liniowo) zawyża odsetek braków u najstarszych i daje skrajne wagi,
dlatego wybrano wariant z kwadratem wieku (AGE²), dla którego AIC spadło z {_n(wyb['AIC_liniowy'], 1)} do {_n(wyb['AIC_age2'], 1)} (LR = {_n(wyb['LRT'], 1)},
{_pz(wyb['p_LRT'])}). Kontrolnie policzono też wariant z 5 grupami skłonności.</p>
{_tbl(ipw_tab.set_index(['model braków', 'wagi']), 3, calkowite=('efektywna liczebność',), podpis='Tabela 4.3. Wyniki IPW dla dwóch modeli braków i trzech wariantów wag')}
{_img('fig_ipw_wagi.png', 'Rys. 4.4. Odsetek braków wg wieku w danych i w dwóch modelach braków (lewy) oraz rozkład wag IPW (prawy)')}
<p>Dla modelu braków {html.escape(wyb['wybrany'])} i pełnych wag IPW daje OR = {_n(ipw['OR'], 3)}
[{_n(ipw['OR_low'], 3)}; {_n(ipw['OR_high'], 3)}], {_pz(ipw['p'])}.</p>
"""


def _sekcja5(df, w3, w5):
    d = w5["tab_diagnostyka"].iloc[0]
    proc = df["M"].mean()
    kal = w5["kalibracja"]
    t = w5["tab_kalibracja"]
    max_odch = (t["zakupy_obserwowane"] - t["zakupy_przewidywane"]).abs().max()
    bvo = kal["braki_vs_obs"]
    Q = np.asarray(w3["kontrast_imputacje"][0])
    biezace_or = np.exp(np.cumsum(Q) / np.arange(1, len(Q) + 1))[M_STAB - 1:]
    stab_min, stab_max = biezace_or.min(), biezace_or.max()
    diag_druk = w5["tab_diagnostyka"][["m", "W", "B", "T", "gamma", "lambda", "RE", "blad_MC", "m_zalecane"]].rename(
        columns={"W": "s²_w", "B": "s²_m", "T": "s²", "gamma": "γ",
                 "lambda": "λ", "RE": "efektywność", "blad_MC": "błąd Monte Carlo",
                 "m_zalecane": "m ≥ 100·λ"})
    return f"""
<h2 id="s5">5. Miara diagnostyczna oceny imputacji</h2>
<p>Do oceny imputacji w kontekście analizowanego modelu wykorzystano frakcję brakującej informacji λ, liczoną
dla testowanego kontrastu β1 - β2. Wynika ona z podziału wariancji estymatora MI na część wewnątrzimputacyjną
i międzyimputacyjną:</p>
<div class="wzor">γ = ((m+1)/m)·s<sup>2</sup><sub>m</sub> / s<sup>2</sup><sub>w</sub><br>
λ = (γ + 2/(v+3)) / (γ + 1)<br>
efektywność = (1 + λ/m)<sup>-1</sup></div>
<p>gdzie γ to względny wzrost wariancji spowodowany brakami, λ to brakująca część informacji, a efektywność to
efektywność estymatora MI przy m imputacjach; m to liczba imputacji, s<sup>2</sup><sub>w</sub> wariancja
wewnątrzimputacyjna, s<sup>2</sup><sub>m</sub> wariancja międzyimputacyjna, a v stopnie swobody z reguł Rubina.
Wariancja całkowita to s<sup>2</sup> = s<sup>2</sup><sub>w</sub> + ((m+1)/m)·s<sup>2</sup><sub>m</sub>.</p>
{_tbl(diag_druk, 4, podpis="Tabela 5.1. Składniki wariancji i miary diagnostyczne imputacji dla kontrastu β1 - β2")}
<p>Wybraliśmy λ, ponieważ odnosi się bezpośrednio do parametru, na podstawie którego podejmujemy decyzję. Sam
odsetek braków Y ({_proc(proc)}) nie mówi, ile informacji o różnicy między kanałami utracono. Dla kontrastu
β1 - β2 λ = {_n(d['lambda'], 3)}, co oznacza, że ok. {_proc(d['lambda'], 0)} informacji o kontraście A vs B jest
utracone z powodu braków danych. To {_n(d['lambda'] / proc, 1)} raza więcej, niż wynikałoby z samego odsetka braków. Wynika to z koncentracji
braków w małym kanale B i wśród starszych klientów (rozdział 2). Wartość λ zależy też od jakości modelu
imputacyjnego, bo im lepiej predyktory wyjaśniają Y, tym mniejsza wariancja międzyimputacyjna i niższe λ.</p>
<p>Miara ma też zastosowanie praktyczne. Na jej podstawie wyznacza się efektywność estymatora MI
({_n(d['RE'], 4)} przy m = {int(d['m'])}) oraz potrzebną liczbę imputacji (m ≥ 100·λ = {int(d['m_zalecane'])},
White i in., 2011). Błąd Monte Carlo oszacowania kontrastu wynosi {_n(d['blad_MC'], 4)}, czyli
{_proc(d['blad_MC'] / w3['wynik_mi']['se'])} SE, więc wynik nie zależy od losowania. λ podaje też standardowo
procedura PROC MIANALYZE w SAS, co ułatwia porównanie z innymi analizami.</p>
<h3>Diagnostyki uzupełniające</h3>
<p>λ opisuje koszt niepewności imputacji, ale nie ocenia, czy same imputacje są trafne. Dlatego
sprawdzono też kalibrację modelu imputacyjnego. Obserwacje z obserwowanym Y podzielono na 10 grup (decyli)
według przewidywanego P(Y = 1) i w każdej porównano obserwowany odsetek zakupów z przewidywanym. Wynik pokazują
prawy panel rys. 5.1 i tabela 5.2. Im bliżej siebie są obie kolumny (i im bliżej przekątnej leżą punkty), tym
lepiej skalibrowany model.</p>
{_img('fig_diagnostyka.png', 'Rys. 5.1. Stabilizacja OR wraz z liczbą imputacji (lewy) i kalibracja modelu imputacyjnego w decylach (prawy)')}
{_tbl(t.set_index('decyl').rename(columns={"zakupy_obserwowane": "obserwowany odsetek zakupów",
                                           "zakupy_przewidywane": "przewidywane P(Y = 1)"}), 4,
       podpis="Tabela 5.2. Kalibracja modelu imputacyjnego w decylach przewidywanego P(Y = 1)")}
<ul>
<li>Test Hosmera-Lemeshowa {'wskazuje na statystycznie istotne, choć niewielkie odchylenia kalibracji' if kal['p_HL'] < ALPHA else 'nie wskazuje na istotne odchylenia kalibracji'}
(χ² = {_n(kal['HL'], 2)}, {_pz(kal['p_HL'])}). Największa różnica w decylu wynosi {_n(100 * max_odch, 1)} p.p.,
a przy n &gt; 9000 test wychwytuje nawet tak małe odchylenia.</li>
<li>Po {M_STAB} imputacjach średnia OR(A vs B) zmienia się już tylko w zakresie {_n(stab_min, 3)}-{_n(stab_max, 3)},
więc m = {M_IMP} wystarcza.</li>
<li>Model imputacyjny przewiduje dla rekordów z brakiem prawie takie samo P(Y = 1) jak dla obserwowanych
(w kanale B {_n(bvo.iloc[2, 1], 3)} wobec {_n(bvo.iloc[2, 0], 3)}). To wyjaśnia, dlaczego MI daje wynik bliski CC.</li>
</ul>
{_tbl(bvo.rename_axis(index="kanał", columns=None), 4, podpis="Tabela 5.3. Średnie przewidywane P(Y = 1) wg kanału dla rekordów obserwowanych i z brakiem")}
"""


def _wrazliwosc_opis(w4):
    tp = w4["tab_tipping_point"]
    zmienia, stabilne = [], []
    zakres = _n(float(np.exp(w4["tab_wrazliwosc"]["delta_A"].abs().max())), 1)
    oba_stabilne = False
    for _, r in tp.iterrows():
        oba = r["scenariusz"].startswith("oba")
        kanal = r["scenariusz"][-1]
        kto = "tylko w kanale " + kanal + " szansa zakupu osób bez wyniku jest"
        kierunek = "wyższa" if r["kierunek"].endswith("> 0") else "niższa"
        if pd.isna(r["delta_zwrotne"]):
            if oba:
                oba_stabilne = True
            else:
                stabilne.append(f"{kto} {kierunek} niż przewidywana przy MAR, nawet ok. {zakres} raza")
            continue
        razy = r["szansa_x"] if r["szansa_x"] >= 1 else 1 / r["szansa_x"]
        lepszy = "A" if r["OR"] > 1 else "B"
        zmienia.append(f"{kto} ok. {_n(razy, 1)} raza {kierunek} niż przy MAR (δ_{kanal} = {_n(r['delta_zwrotne'], 2)}), "
                       f"a wtedy kanał {lepszy} jest istotnie lepszy")
    if oba_stabilne:
        stabilne.append(f"w obu kanałach szansa zakupu osób bez wyniku jest w tym samym stopniu wyższa albo niższa "
                        f"niż przewidywana przy MAR, nawet ok. {zakres} raza")
    tw = w4["tab_wrazliwosc"]
    ist_mar = bool(tw.query("delta_A == 0 and delta_B == 0")["istotny"].iloc[0])
    return zmienia, stabilne, int((tw["istotny"] == ist_mar).sum()), len(tw)


def _wnioski(df, w1, w3, w4, w5):
    cc, mi, ipw = w3["wynik_cc"], w3["wynik_mi"], w4["wynik_ipw"]
    d = w5["tab_diagnostyka"].iloc[0]
    zmienia, stabilne, bez_zmian, n_kom = _wrazliwosc_opis(w4)
    brak_kanal = df.groupby("contact")["M"].mean()
    sr = df.groupby("M")["age"].mean()
    n_b = int((df["contact"] == 2).sum())
    n_b_cc = int(((df["contact"] == 2) & (df["M"] == 0)).sum())
    return f"""
<h2 id="s6">6. Wnioski</h2>
<h3>6.1. Struktura danych i braki</h3>
<p>Braki dotyczą wyłącznie zmiennej celu Y_NEWDEPOSIT ({_proc(df['M'].mean())} rekordów) i mają wzorzec
monotoniczny. Nie są rozłożone równomiernie, bo osoby bez wyniku są średnio starsze ({_n(sr.loc[1], 1)} vs
{_n(sr.loc[0], 1)} lat), a najwięcej braków ma najmniejszy kanał B ({_proc(brak_kanal.loc[2])} wobec
{_proc(brak_kanal.loc[1])} w kanale A). Analiza kompletnych przypadków zmniejsza próbę w kanale B z {n_b} do {n_b_cc}
obserwacji, co obniża moc testu, a systematyczny rozkład braków grozi obciążeniem oszacowań.</p>
<h3>6.2. Mechanizm braków</h3>
<p>Testy jednowymiarowe oraz model logitowy wykazują istotny wpływ wieku (<code>age</code>) oraz kanału
(<code>contact</code>) na prawdopodobieństwo braku danych (M). Zmienna <code>education</code> istotnie
determinuje samą zmienną <code>y_newdeposit</code>, stanowiąc dobrą zmienną pomocniczą do korekty oszacowań podczas
imputacji. Istotna zależność braku od obserwowanych zmiennych wyklucza MCAR, lecz nie pozwala rozstrzygnąć między MAR
a MNAR. Przyjęcie MAR pozostaje więc założeniem, którego nie da się zweryfikować na podstawie danych, i z tego
powodu wykonano analizę wrażliwości.</p>
<h3>6.3. Model przy MCAR i MAR</h3>
<p>Przy MCAR (CC) OR kanału A wobec B wynosi {_n(cc['OR'], 3)} ({_pz(cc['p'])}), a przy MAR (imputacja
wielokrotna) {_n(mi['OR'], 3)} ({_pz(mi['p'])}). Uwzględnienie braków przesuwa oszacowanie nieznacznie na
korzyść kanału A, ale nie zmienia decyzji. Nadal brak podstaw do odrzucenia hipotezy β1 = β2, czyli <b>nie znaleziono istotnej różnicy szansy zakupu
między kanałami A i B</b>. Oba kanały istotnie
zwiększają szansę zakupu względem braku komunikacji.</p>
<h3>6.4. Wyniki w kontekście braków</h3>
<p>Przy założeniu MAR wniosek potwierdza druga, niezależna metoda, IPW (OR = {_n(ipw['OR'], 3)},
{_pz(ipw['p'])}). Analiza wrażliwości pokazuje jednak, że <b>wniosek zależy od założenia MAR</b>. Odstępstwo od MAR opisuje,
ile razy szansa zakupu osób bez wyniku różni się od przewidywanej przy MAR. W {bez_zmian} z {n_kom} rozpatrzonych
scenariuszy odstępstwa wniosek o braku istotnej różnicy pozostaje niezmieniony. W pozostałych {n_kom - bez_zmian}
scenariuszach, przy wystarczająco asymetrycznych odchyleniach w jednym kanale, pojawia się istotna przewaga jednego
z kanałów.</p>
<p>Wniosek się nie zmienia w następujących scenariuszach.</p>
<ul>{''.join(f'<li>{s[0].upper() + s[1:]}.</li>' for s in stabilne)}</ul>
<p>Wniosek zmienia się w następujących scenariuszach.</p>
<ul>{''.join(f'<li>{s[0].upper() + s[1:]}.</li>' for s in zmienia)}</ul>
<p>Danymi nie da się wykluczyć takich scenariuszy, więc brak istotnej różnicy między kanałami jest wynikiem
warunkowym względem założenia MAR.</p>
<h3>6.5. Ocena imputacji</h3>
<p>Frakcja brakującej informacji o testowanym kontraście wynosi λ = {_n(d['lambda'], 2)}, czyli ok.
{_proc(d['lambda'], 0)} informacji o kontraście A vs B jest utracone z powodu braków danych. To
{_n(d['lambda'] / df['M'].mean(), 1)} raza więcej, niż sugeruje sam odsetek braków. Przy m = {int(d['m'])}
efektywność estymatora MI wynosi {_n(d['RE'], 4)}, a błąd Monte Carlo to tylko
{_proc(d['blad_MC'] / mi['se'])} SE. Test Hosmera-Lemeshowa wskazuje na statystycznie istotne, choć niewielkie odchylenia
kalibracji modelu imputacyjnego, bo największa różnica w decylu wynosi
{_n(100 * w5['tab_kalibracja'].eval('zakupy_obserwowane - zakupy_przewidywane').abs().max(), 1)} p.p. Dla rekordów z brakiem
model przewiduje podobne P(Y = 1) jak dla obserwowanych, dlatego wynik MI jest bliski CC.</p>
<div class="box ok"><p><b>Wniosek końcowy.</b> Na podstawie dostępnych danych nie znaleziono istotnej różnicy
szansy zakupu między kanałami komunikacji A i B. Wynik jest spójny w analizie kompletnych przypadków przy
założeniu MCAR (OR = {_n(cc['OR'], 2)}, {_pz(cc['p'])}), w imputacji wielokrotnej przy założeniu MAR
(OR = {_n(mi['OR'], 2)}, {_pz(mi['p'])}) oraz w metodzie IPW (OR = {_n(ipw['OR'], 2)}, {_pz(ipw['p'])}). Oba
kanały istotnie zwiększają szansę zakupu względem braku kontaktu.</p>
<p style="margin-bottom:0">Braki odpowiedzi zależą od wieku i kanału, więc dane są niespójne z MCAR, a
założenia MAR nie da się zweryfikować na podstawie samych danych. Analiza wrażliwości pokazuje, że wniosek o braku
istotnej różnicy jest stabilny przy jednakowych odchyleniach od MAR w obu kanałach, ale może się zmienić przy
wystarczająco dużym odchyleniu szansy zakupu osób bez odpowiedzi tylko w jednym z kanałów. W {bez_zmian} z {n_kom}
rozpatrzonych scenariuszy odstępstwa od MAR wniosek o braku istotnej różnicy pozostaje niezmieniony. W pozostałych
{n_kom - bez_zmian}, przy wystarczająco asymetrycznych odchyleniach w jednym z kanałów, pojawia się istotna przewaga
jednego z kanałów. Wynik należy więc interpretować jako warunkowy względem założenia MAR.</p></div>
"""


def zbuduj(df, w1, w2, w3, w4, w5, sciezka=None):
    sciezka = sciezka or OUT_DIR / "raport.html"
    mi, cc = w3["wynik_mi"], w3["wynik_cc"]
    n_brak = int(df["M"].sum())
    d = w5["tab_diagnostyka"].iloc[0]
    czesci = [f"""<!doctype html><html lang="pl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Imputacja danych: zadanie 1B</title><style>{CSS}</style></head><body>
<p class="naglowek">Zaawansowana analityka biznesowa, metody imputacji danych<br>
{date.today().strftime('%d.%m.%Y')}</p>
<h1>Zadanie 1, wersja B: brak danych w zmiennej skokowej</h1>
<p class="sub">Zbiór danych: <code>{DATA_FILE.name}</code></p>
{_autorzy()}
<nav class="spis"><h3>Spis treści</h3><ol>
<li><a href="#s1">Analiza struktury</a></li>
<li><a href="#s2">Zależność prawdopodobieństwa braku danych od pozostałych cech</a></li>
<li><a href="#s3">Model logitowy przy założeniach MCAR i MAR</a></li>
<li><a href="#s4">Wyniki modelu w kontekście braków danych</a></li>
<li><a href="#s5">Miara diagnostyczna oceny imputacji</a></li>
<li><a href="#s6">Wnioski</a></li>
<li><a href="#s7">Odtwarzalność i źródła</a></li>
</ol></nav>""".replace(f"{len(df):,}", f"{len(df):,}".replace(",", " ")).replace(f"{n_brak:,}", f"{n_brak:,}".replace(",", " ")),
              _sekcja1(w1), _sekcja2(df, w2), _sekcja3(df, w2, w3), _sekcja4(w3, w4, w5), _sekcja5(df, w3, w5),
              _wnioski(df, w1, w3, w4, w5),
              f"""
<h2 id="s7">7. Odtwarzalność i źródła</h2>
<ul>
<li>Aby odtworzyć wyniki, należy uruchomić <code>pip install -r requirements.txt</code>, a potem <code>python main.py</code>. Ścieżka do danych
jest zdefiniowana na początku <code>wspolne.py</code> (<code>WORK_DIR</code>, <code>DATA_FILE</code>). Ziarno
losowania {SEED}, wersje pakietów przypięte w <code>requirements.txt</code>, alternatywnie obraz Docker.</li>
<li>Pełny wydruk wyników zapisuje się w <code>wyniki/wydruk.txt</code>, a wykresy w folderze <code>wyniki/</code>.</li>
</ul>
<h3>Literatura</h3>
<ul class="src">
<li>White I.R., Royston P., Wood A.M. (2011). Multiple imputation using chained equations: issues and guidance
for practice. <i>Statistics in Medicine</i> 30 (reguła m ≥ 100·λ).</li>
</ul>
</body></html>"""]
    sciezka.write_text("\n".join(czesci), encoding="utf-8")
    print(f"Raport: {sciezka}")
    return sciezka

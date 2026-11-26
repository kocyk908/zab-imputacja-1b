# Zadanie 1B: plan pracy

Zbiór: `kb19.csv` (11 304 obs.). Braki tylko w `y_newdeposit` (2 084 obs., 18,4%).
Pytanie badawcze: czy kanał A i kanał B różnią się szansą zakupu? Test H0: β1 − β2 = 0 w modelu logitowym.

Oznaczenia źródeł:
- **„jak w wykładzie (zab_cz_1), s. X”**: slajd z wykładu `zab_cz_1.pdf`
- **„jak w kodzie "..."”**: kod prowadzącego z `exercises_cwiczenia/`
- **„WŁASNE”**: nie ma tego ani na wykładzie, ani w kodach prowadzącego

> Uwaga o kodach: w `exercises_cwiczenia/` **nie ma kodu do imputacji**. Są tam kody do CLV,
> migracji, k-średnich, analizy przeżycia i podejścia data mining. Wykład odsyła do plików
> `S03MDPattern01`, `S03MechTesting01/02`, `S05Imp01.py`, `S05MultipleImputation001` itd.,
> ale tych plików nie mamy. **Warto je pobrać z Teams/Moodle.** Z obecnych kodów przydaje się
> tylko `DM_Approach_py (1).ipynb`: model logitowy przez `smf.glm(..., family=sm.families.Binomial())`,
> zestaw importów, histogramy `sns.histplot` i funkcja `gains_table` (decyle).

---

## Co jest do zrobienia (punkty z polecenia)

### 1. Analiza struktury i braków
| Krok | Źródło |
|---|---|
| Cecha ciągła `age`: statystyki, histogram, wykres pudełkowy | jak w wykładzie (zab_cz_1), s. 17–18, styl wykresów jak w kodzie "DM_Approach_py (1).ipynb" |
| Cechy skokowe: tabele częstości | jak w wykładzie (zab_cz_1), s. 46 skale pomiarowe |
| Wynik badania: % zakupów wg `contact` (0/A/B) | WŁASNE, zwykła tabela krzyżowa |
| Macierz braków M i wzorzec braków. Tu wzorzec jest **monotoniczny**, bo brakuje tylko Y | jak w wykładzie (zab_cz_1), s. 73–74, 89–90 |
| Skutki braków: mniejsza moc testu, możliwe obciążenie | jak w wykładzie (zab_cz_1), s. 75–81, 107–108 |

### 2. Zależność P(brak) od pozostałych cech
| Krok | Źródło |
|---|---|
| Indykator braku `M = 1` dla brakującego Y | jak w wykładzie (zab_cz_1), s. 73 |
| **Metoda I (Balicki)**: porównanie średniego `age` u osób z brakiem i bez braku (test t) | jak w wykładzie (zab_cz_1), s. 94 |
| **Metoda II (Balicki)**: związek M z cechami skokowymi, χ² i V Cramera | jak w wykładzie (zab_cz_1), s. 94 |
| Model logitowy P(M = 1 \| X), czyli model skłonności do odpowiedzi | jak w wykładzie (zab_cz_1), s. 103 |
| Wniosek: MCAR odrzucone, zakładamy MAR. Danych nie da się użyć do rozstrzygnięcia MAR vs MNAR | jak w wykładzie (zab_cz_1), s. 92–96 |

Wyniki, które już mamy: braki zależą głównie od **wieku** (do 40 lat ok. 5%, od 50 lat ok. 46–49%)
i od **kanału** (B: 37%, A: 18%, brak kontaktu: 15%). Po kontroli wieku `house_loan` i `education`
przestają być istotne. `education` nie należy do modelu, ale wyjaśnia Y, więc jest dobrą
**zmienną pomocniczą** do imputacji.

### 3. Model logitowy przy MCAR i MAR, ze specyfikacją modelu imputacyjnego
| Krok | Źródło |
|---|---|
| **MCAR**: analiza kompletnych przypadków (CC) | jak w wykładzie (zab_cz_1), s. 105–109. Na s. 109 jest wyjątek: w regresji CC bywa nieobciążone |
| Postać modelu (ref.: contact = 0, house_loan = yes, poutcome = unknown) | treść zadania + przykładowy kod R/SAS z PDF zadania. W Pythonie: `smf.glm` jak w kodzie "DM_Approach_py (1).ipynb" |
| **MAR**: imputacja wielokrotna. Wzorzec monotoniczny, więc wystarczy jedno równanie logistyczne (odpowiednik `PROC MI; MONOTONE LOGISTIC`) | jak w wykładzie (zab_cz_1), s. 148–149, 161, 163 |
| Dwa źródła losowości: (1) losujemy β* z rozkładu a posteriori, (2) losujemy Y* ~ Bernoulli | jak w wykładzie (zab_cz_1), s. 132, 143–144. Na wykładzie jest wersja dla regresji liniowej, my przenosimy ją na logit: **WŁASNE** |
| Łączenie wyników regułami Rubina: średnia, W, B, T, df | jak w wykładzie (zab_cz_1), s. 151–152, 155 (df wg wzoru 49) |
| Test kontrastu β1 − β2 na imputowanych danych (odpowiednik `estimate contact 1 -1` + MIANALYZE) | PDF zadania + jak w wykładzie (zab_cz_1), s. 150. W Pythonie trzeba to policzyć samodzielnie: **WŁASNE** |
| Wybór zmiennych do modelu imputacyjnego (AIC: + education, + age²) | **WŁASNE** |

### 4. Podsumowanie modelu właściwego w kontekście braków
| Krok | Źródło |
|---|---|
| Tabela CC vs MI: OR(A vs B), CI, p | jak w wykładzie (zab_cz_1), s. 87: porównanie wyników przy różnych założeniach |
| **Analiza wrażliwości (MNAR, tipping point)** | jak w wykładzie (zab_cz_1), s. 75 (tipping point), 87, 97 (analiza wrażliwości). Wykład tylko wspomina tę metodę, implementacja delta-adjustment to **WŁASNE** |
| (opcjonalnie) IPW, czyli ważenie odwrotnością P(odpowiedzi) | jak w wykładzie (zab_cz_1), s. 101–103, Horvitz-Thompson / propensity |

### 5. Miara diagnostyczna imputacji
| Krok | Źródło |
|---|---|
| **λ (frakcja brakującej informacji)** dla kontrastu β1 − β2, γ (wzrost wariancji) i efektywność | jak w wykładzie (zab_cz_1), s. 153–155, wzory 46–48 |
| Uzasadnienie: λ kontrastu ≈ 0,37 przy zaledwie 18% braków, bo braki skupiają się w małym kanale B | interpretacja: **WŁASNE** |
| (dodatkowo) kalibracja modelu imputacyjnego w decylach | `gains_table` jak w kodzie "DM_Approach_py (1).ipynb" |

---

## Podział pracy (5 osób)

| Osoba | Zakres | Źródła | Daje dalej |
|---|---|---|---|
| **1. Struktura danych** | pkt 1: statystyki `age`, tabele częstości, wynik badania wg `contact`, wykresy, macierz braków i wzorzec | jak w wykładzie (zab_cz_1), s. 17–18, 73–74, 89–90, jak w kodzie "DM_Approach_py (1).ipynb" | wykresy, tabele i 1–2 strony tekstu |
| **2. Mechanizm braków** | pkt 2: testy Balickiego (I i II), model logitowy P(brak), wniosek MCAR/MAR, wybór zmiennej pomocniczej `education` | jak w wykładzie (zab_cz_1), s. 91–96, 103 | tabela testów, model braków i lista zmiennych dla osoby 3 |
| **3. Modele CC i MI** | pkt 3: model CC, model imputacyjny, MI (M = 100), reguły Rubina, test β1 − β2, specyfikacja imputacji w tekście | jak w wykładzie (zab_cz_1), s. 105–109, 132, 143–155 | tabele oszacowań CC i MI |
| **4. Innowacyjność** | pkt 4 (część): analiza wrażliwości MNAR (tipping point), IPW, kontrola wyników w SAS (`PROC MI` / `PROC MIANALYZE` wg przykładu z PDF; SAS jest zainstalowany na tym komputerze) | jak w wykładzie (zab_cz_1), s. 75, 87, 97, 101–103, 149–150 | wykres wrażliwości, tabela IPW, porównanie Python vs SAS |
| **5. Diagnostyka i raport** | pkt 5 (λ, γ, RE, kalibracja) + pkt 4 (tabela porównawcza i wnioski) + złożenie raportu HTML/PDF, wydruku i kodu, upload na Teams | jak w wykładzie (zab_cz_1), s. 153–155 | gotowa paczka do oddania |

Kolejność: osoby 1 i 2 zaczynają od razu. Osoba 3 rusza po wyniku osoby 2 (zmienne do imputacji).
Osoby 4 i 5 korzystają z wyników osoby 3. Osoba 5 przez cały czas pilnuje spójnego stylu i odwołań
do slajdów. Każda osoba pisze w swoim pliku `sekcjaX_....py`, a `main.py` uruchamia całość.

## Co jest potrzebne

1. **Sprawdzić w skoroszycie z grupami**, czy naszym zbiorem na pewno jest `kb19`.
2. **Pobrać kody z wykładu** (`S03MechTesting01/02`, `S05Imp01.py`, `S05MultipleImputation001`),
   aby jak najwięcej opierać na kodzie prowadzącego. W `exercises_cwiczenia/` ich nie ma.
3. Python z pakietami: `pandas numpy statsmodels scipy matplotlib seaborn` (lista w `requirements.txt`).
   Na tym laptopie Windows Smart App Control blokuje numpy, więc uruchamiamy przez Docker
   (`Dockerfile`). Na innych komputerach wystarczy `pip install -r requirements.txt`.
4. Do oddania:
   a) raport `.html`/`.pdf`,
   b) kod `.py` ze ścieżką do danych na początku,
   c) wydruk wyników.
   Wszystko wrzucamy na MS Teams.
5. Punktacja: **innowacyjność to 2 pkt**. Tu liczą się analiza wrażliwości MNAR, IPW i wybór
   modelu imputacyjnego przez AIC.

## Stan obecny

- `sekcja1–5_*.py` + `main.py` to **szkielet do pisania od nowa**. Każda osoba ma swój plik z listą TODO
  oraz ustalonymi nazwami zmiennych na wejściu i wyjściu.
- `wersja_robocza/` (tylko lokalnie, nie ma jej w repo) to pełna działająca wersja (`analiza.py`, `raport.py`, `wyniki/`).
  Traktujemy ją jako ściągę i do sprawdzania swoich wyników, nie do oddania.

| Metoda | OR (A vs B) | 95% CI | p |
|---|---|---|---|
| CC (MCAR) | 1,08 | 0,80–1,45 | 0,61 |
| MI (MAR) | 1,13 | 0,84–1,52 | 0,40 |
| IPW (MAR) | 1,12 | 0,82–1,53 | 0,46 |

Wniosek: **brak istotnej różnicy między A i B** niezależnie od założenia. Oba kanały są jednak
istotnie lepsze od braku komunikacji (OR ≈ 2,6–2,9). λ dla kontrastu = 0,37.

Do zrobienia w wersji roboczej:
- ujednolicić nazwy z wykładem: γ zamiast „RIV”, λ zamiast „FMI”;
- dopisać do raportu odwołania do slajdów;
- przepiąć fragmenty na kod prowadzącego, kiedy go zdobędziemy.

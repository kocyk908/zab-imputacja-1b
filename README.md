# ZAB: metody imputacji danych, zadanie 1B

Projekt grupowy (5 osób) z przedmiotu *Zaawansowana analityka biznesowa, metody imputacji danych*.

## Co robimy

Mamy wyniki badania marketingowego (`kb19.csv`, 11 304 klientów). Klienci zostali losowo przydzieleni
do trzech grup: **kanał komunikacji A**, **kanał komunikacji B** albo **brak komunikacji** (`contact` = 1 / 2 / 0).
Zmienna celu `y_newdeposit` mówi, czy klient kupił produkt (1/0). **U 2 084 klientów (18,4%) wynik jest nieznany.**

**Pytanie badawcze:** czy szansa zakupu przy kanale A różni się od szansy przy kanale B?

Odpowiadamy modelem logitowym:

```
logit P(Y=1) = b0 + b1·CONTACT_A + b2·CONTACT_B + b3·AGE + b4·HOUSING_LOAN
             + b5·POUTCOME_failure + b6·POUTCOME_success + b7·POUTCOME_other
```

i testem **H0: b1 − b2 = 0** (OR kanał A vs B = exp(b1 − b2)).

Całe zadanie dotyczy tego, **jak braki w Y wpływają na wynik**. Liczymy model przy dwóch założeniach:
- **MCAR**: braki są całkowicie losowe, więc wystarczy usunąć rekordy z brakami (analiza kompletnych przypadków, CC);
- **MAR**: braki zależą od innych cech (np. wieku), więc uzupełniamy je **imputacją wielokrotną (MI)**.

Pełna treść zadania: `Zadanie1_b.pdf`. Szczegółowy plan z odwołaniami do slajdów: [`PLAN.md`](PLAN.md).

## Pięć punktów zadania i kto co robi

| # | Punkt z polecenia | Plik | Osoba |
|---|---|---|---|
| 1 | Analiza struktury (cechy ciągłe/skokowe), wykresy, zakres i rozkład braków | `sekcja1_struktura.py` | Osoba 1 |
| 2 | Od czego zależy brak danych? Które cechy pozwolą skorygować wynik imputacją? | `sekcja2_mechanizm.py` | Osoba 2 |
| 3 | Model logitowy przy MCAR i MAR + specyfikacja modelu imputacyjnego | `sekcja3_modele.py` | Osoba 3 |
| 4 | Podsumowanie wyników w kontekście braków + aspekt innowacyjny (analiza wrażliwości, IPW, SAS) | `sekcja4_innowacje.py` | Osoba 4 |
| 5 | Miara diagnostyczna imputacji z uzasadnieniem + złożenie raportu i wydruku | `sekcja5_diagnostyka.py` | Osoba 5 |

Wpiszcie tu swoje imiona:
- Osoba 1: …
- Osoba 2: …
- Osoba 3: …
- Osoba 4: …
- Osoba 5: …

## Struktura repo

```
wspolne.py              ścieżki, wczytanie danych, model z zadania (wspólne, nie zmieniamy bez uzgodnienia)
sekcja1_struktura.py    osoba 1
sekcja2_mechanizm.py    osoba 2
sekcja3_modele.py       osoba 3
sekcja4_innowacje.py    osoba 4
sekcja5_diagnostyka.py  osoba 5
main.py                 uruchamia wszystko po kolei i zapisuje wyniki/wydruk.txt
PLAN.md                 szczegółowy plan, odwołania do wykładu
kb19.csv                dane (nasz indywidualny zbiór)
Zadanie1_b.pdf          treść zadania
```

Wykładu (`zab_cz_1.pdf`) i kodów prowadzącego (`exercises_cwiczenia/`) **nie ma w repo**, bo repo jest
publiczne. Każdy ma je z Teams/Moodle. Trzymajcie je lokalnie w tym samym folderze, `.gitignore` je pomija.

Każda sekcja ma funkcję `uruchom(...)`. Funkcja zwraca słownik z wynikami **o ustalonych nazwach**
(są opisane w docstringu). Trzymajcie się tych nazw, bo z nich korzystają kolejne sekcje.

Zależności między sekcjami:
- sekcja 3 bierze `zmienne_imputacji` z sekcji 2 (do tego czasu ma wersję tymczasową);
- sekcja 4 używa `imputuj()` i `rubin()` z sekcji 3 oraz `model_brakow` z sekcji 2;
- sekcja 5 zbiera wyniki ze wszystkich.

## Jak uruchomić

```bash
pip install -r requirements.txt
python sekcja1_struktura.py     # tylko swoja sekcja
python main.py                  # całość -> wyniki/wydruk.txt
```

## Jak pracujemy w Git

1. Jednorazowo: `git clone <link do repo>`.
2. Przed pracą: `git pull`.
3. Edytujesz **tylko swój plik** `sekcjaX_....py`. Dzięki temu nie będzie konfliktów.
4. Po pracy:
   ```bash
   git add sekcjaX_....py
   git commit -m "sekcja X: co zrobione"
   git pull
   git push
   ```
5. Nie commitujemy folderu `wyniki/`, bo każdy generuje go sobie sam.

## Zasada odwołań w kodzie

Przy każdym kroku piszemy w komentarzu, skąd pochodzi:
- `jak w wykładzie (zab_cz_1), s. 94`: metoda ze slajdu;
- `jak w kodzie "DM_Approach_py (1).ipynb"`: kod prowadzącego z `exercises_cwiczenia/`;
- `WŁASNE`: nie ma tego ani w wykładzie, ani w kodach prowadzącego.

Uwaga: w `exercises_cwiczenia/` **nie ma kodu do imputacji**. Wykład odsyła do plików `S03MechTesting01/02`,
`S05Imp01.py` i `S05MultipleImputation001`.

## Co oddajemy (MS Teams)

a) raport z wynikami, opisem i wnioskami (`.html` lub `.pdf`);
b) kod (`.py`) ze ścieżką do danych na początku, czyli `wspolne.py` + sekcje + `main.py`;
c) wydruk wyników, czyli `wyniki/wydruk.txt` z `python main.py`.

Punktacja: poprawność i interpretacja 3 pkt (waga 2), **innowacyjność 2 pkt**,
programowanie (odtwarzalność, czytelność, komentarze) 1,5 pkt, estetyka i kompletność 0,5 pkt.

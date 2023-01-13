# **Wheat Detection Model - WDM**
![Logo](https://drive.google.com/uc?export=view&id=1xwnJd3MACXy-L-Rs1TBoWTd3J1RJ7-Se)

## Charakterystyka
Szacowanie wielkości plonu w oparciu o fotograficzną dokumentację in-situ.
Oprogramowanie pozwala w łatwy sposób wyznaczyć liczbę kłosów na zdjęciu wykonanym w terenie.
Program ma na celu usprawnienie i przyśpieszenie pracy terenowej w trakcie kontroli na miejscu.
Program szacuje plonowanie na podstawie obsady(liczby roślin w ramie pomiarowej 25x25cm).
Gdy obsada jest zbyt duża, rośliny wykształcają mniejsze (krótsze) kłosy, które zawierają drobniejsze ziarno, o mniejszej masie 1000 ziarn i gęstości.
Oznacza to, że w takim wypadku plon będzie mniejszy, a jego jakość nie będzie najlepsza. Zwiększa się także podatność roślin na wyleganie.
Jeśli natomiast obsada kłosów jest zbyt mała, dłuższe kłosy, zawierające większe ziarniaki, nie pozwolą uzyskać zadowalającego plonu.
Ważne jest także, by obsada kłosów była równomierna dla całej plantacji. 

## Prawa autorskie
W oprogramowaniu  zostały zastosowane autorskie rozwiązania oraz gotowe implementacje architektury modelu.
### Autorzy
- [@dominowak](https://www.github.com/dominowak)
- [@kasiakrupa146](https://www.github.com/kasiakrupa146)
- [@dominikMAI](https://www.github.com/dominikMAI)

### Warunki licencyjne
Implementacja modelu  (YOLO_v3) wykorzystana w ramach licencji: 
[Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)\
W skryptach udostępnionych na licencji Apache 2.0 nie wprowadzono żadnych zmian. 

Architektura modelu detekcyjnego według artykułu autorstwa J. Redmona i A. Farhada **“YOLOv3: An Incremental Improvement”**
Metoda szacowania plonowania na podstawie opracowania **"Szacowanie plonów rolniczych"** autorstwa prof. dr hab. Bogdana Kuliga, Uniwersytet Rolniczy w Krakowie.\

Niniejsze oprogramowanie udostępniane jest na licencji **MIT 2023**.
[MIT](https://choosealicense.com/licenses/mit/)

## Specyfika wymagań
| indeks | opis                                                                                | priorytet  | kategoria        |
|--------|-------------------------------------------------------------------------------------|------------|------------------|
| W1     | Program musi być łatwy w użyciu i instalacji                                        | wymagane   | pozafunkcjonalne |
| W2     | Program powinien zawierać instrukcję obsługi                                        | wymagane   | pozafunkcjonalne |
| W3     | Oprogramowanie powinno być zdolne do zaczytania dowolnego plik graficznego          | wymagane   | funkcjonalne     |
| W4     | Oprogramowanie powinno wskazywać lokalizację kłosów pszenicy na obrazie             | wymagane   | funkcjonalne     |
| W5     | Model musi cechować zdolność do odróżniania kłosów pszenicy wśród innych roślin     | wymagane   | funkcjonalne     |
| W6     | Model powinien wykrywać przynajmniej 95% wszystkich kłosów pszenicy                 | wymagane   | funkcjonalne     |
| W7     | Oprogramowanie musi umożliwiać uruchomienie modelu z wykorzystaniem procesora GPU   | wymagane   | funkcjonalne     |
| W8     | Oprogramowanie musi odczytywać wprowadzane  wartości funkcji plonowania             | wymagane   | funkcjonalne     |
| W9     | Oprogramowanie musi prawidłowo obliczać szacowane wartości uzysku plonu             | wymagane   | funkcjonalne     |
| P1     | Program umożliwia zapis ilustracji z wynikiem detekcji w przestrzeni dyskowej       | przydatne  | funkcjonalne     |
| P2     | Program dostosowuje rozmiar interfejsu do rozmiarów monitora użytkownika            | przydatne  | funkcjonalne     |
| P3     | Dla poprawy czytelności użytkownik może zmienić kolor obrysu zasięgu                | przydatne  | funkcjonalne     |
| O1     | Oprogramowanie automatycznie przeprowadza predykcję po zaczytaniu pliku graficznego | opcjonalne | funkcjonalne     |
| O2     | Oprogramowanie powinno przeprowadzić detekcję w czasię nie dłuższym niż 3 sekundy   | opcjonalne | pozafunkcjonalne |
| O3     | Interfes dopasowuje język opogramowania do kraju w którym jest wykorzystywany       | opcjonalne | funkcjonalne     |
| O4     | Język interfejsu wybierany jest z listy                                             | opcjonalne | funkcjonalne     |
| O5     | Plik graficzny może zostać zaczytany bezpośrednio z lokalizacji sieciowej           | opcjonalne | funkcjonalne     |
| O6	 | Aplikacja powinna wykonywać detekcję na obrazie prosto z kamery urządzenia		   | opcjonalne | funkcjonalne	   |

## Stos technologiczny

Struktura katalogów
```{bash}
├── main.py
├── LICENSE.md
├── README.md
├── requirements.txt
├── src
│   ├── detect.py
│   ├── model_f.py
├── static
│   ├── css
│   │   └── styles.css
│   ├── favicon.ico
│   ├── js
│   │   └── utils.js
│   └── temp
└── templates
    └── index.html
```

### Architektura rozwoju i uruchomieniowa
Kontrola wersji: git 2.17.1 \
IDLE: jupyter lab 3.1.7 \
Środowisko języka Python: Anaconda 4.10.3 \
Język oprogramowania: Python 3.7.11 \
Framework: Flask 1.1.2 \
Deep Learning: tensorflow 1.13.1 \
Przetwarzanie obrazu: opencv 4.5.2.52, Albumentations 1.0.3 \
Analiza danych: numpy 1.21.5, scikit-learn 1.1.3\
Wizualizacja: matplotlib 3.4.2\
Monitoring procesów treningu modeli: Tensorboard 2.10.1\
Środowisko obsługi procesorów graficznych - CUDA 11.4 \

Oprogramowanie rozwijano i testowano na maszynie wirtualnej z systemem **Ubuntu 18.04.6 LTS**

## Testowanie wymagań

**Wymaganie W1** \
Uruchomienie zawiera się w trzech krokach począwszy od pobrania repozytorium do uruchomienia w wierszu poleceń:
1. Pobranie repozytorium
Projekt można pobrać z repozytorium za pomocą komendy:
```bash
  git clone https://github.com/DominikMAI/WheatDetectionModel.git
  cd WheatDetection Model
```

2. Pobranie wag modelu detekcyjnego
Do prawidłowego działania modelu potrzebne są wagi. Do pobrania wag potrzebne jest narzędzie curl..
Wagi z treningu można pobrać z lokalizacji sieciowej:
```bash
cd weights
curl -o weights.h5  -L 'https://drive.google.com/uc?export=download&confirm=yes&id=1t9_0HlgSjF9UpboXfd2sJsuN2anM7Baq'

python main.py --weights_dir=./weights/yolov5.hdf5
```

3. Instalacja potrzebnych bibliotek za pomocą polecenia anacondy
```bash
conda install --file requirements.txt
```

4. Uruchomienie interfejsu aplikacji (z domyślną konfiguracją).
```bash
python main.py
```

**Wymaganie W2**
1. Aby podejrzeć dostępne argumenty należy uruchomić program z flagą --help. Użycie argumentu pomocy pozwala na podejrzenie
dostępnych funkcji oraz podgląd domyślnych wartości argumentów.
```bash
python main.py --help
```

**Wymaganie W3**
Aby odczytać obraz wejściowy należy:
Uruchomić aplikację
```bash
python mainy.py
```
1. W adresie przeglądarki internetowej wpisz adres 127.0.0.1:5000 oraz kliknij Enter,
2. W uruchomionej aplikacji należy kliknąć przycisk "Przeglądaj", w otwartym oknie dialogowym należy wybrać zdjęcie terenowe na lokalnym dysku,
3. Zdjęcie po zaczytaniu powinno się wyświetlić w interfejsie graficznym aplikacji,

**Wymaganie W4**
1. Wczytaniu zdjęcie terenowe do interfejsu aplikacji
2. Następni kliknij "Predykcja",
3. Po ok. 2s wyświetlony obraz w interfejsie powinien zostać zastąpiony obrazem z zasięgami kłosów pszenicy

**Wymaganie W5**
Weryfikacja wymagania obejmuje kroki:
1. Po kliknięciu "Predykcja" należy zwrócić czy model wskazuje poprawnie lokalizację większości kłosów pszenicy
2. Wsród wyników detekcji nie powinny znaleźć się inne rośliny niż pszenica


**Wymaganie W6**
1. Uruchom aplikację
2. Przeprowadź detekcję na obrazie z obrazu będącego w katalogu danych testowych 


**Wymaganie W7**
1. Aby wykorzystać w predykcji procesor graficzny, w wierszu poleceń należy umieścić argument gpu.
Użycie procesora graficznego GPU brak-"", jedno urządzenie - "0", dwa urządzenia -"0,1" itd
```bash
python main.py --gpu='0'
```

**Wymaganie W8**
1. Aby wprowadzić wartości należy wpisać wartości w pola kolejno:
- Średnia liczba ziaren w kłosie - W trakcie pracy terenowej należy odnotować średnią liczbę ziaren w kłosie.
- Masa 1000 nasion - Masa 1000 nasion w jednostce gram g.
- Wielkość szkód -  wielkość szkody, odnotowanej wcześniej straty plonu. Stratę należy podać w punktach procentowych.\


**Wymaganie W9**
1. Po wprowadzeniu wartości należy kliknąć przycisk "Oblicz plonowanie".
2. Po kliknięciu przycisku natychmiast powinna pojawić się wartość oszacowanego plonu, wraz z całym komunikatem np Plonowanie wynosi: 27.086 dt/ha.\
Wielkość oszacowanego plonu podawana jest w jednostce dt/ha.
Dla poszczególnych gatunków pszenicy przyjmujemy różne zakresy wartości:

| Gatunek         | Liczba kłosów na 1m2 | Liczba ziaren w kłosie | MTN       | Plon ziarna (dt/ha) |
|-----------------|----------------------|------------------------|-----------|---------------------|
| Pszenica ozima  | 450-750              | 32.2-50.1              | 41.7-52.9 | 64.5-104.1          |
| Pszenica jara   | 480-750              | 21.8-31.8              | 35.7-47.1 | 46.5-83.0           |
| Pszenica twarda | 520-635              | 16.5-22.5              | 48.9-54.5 | 42.5-70.2           |


**Wymaganie P1**
1. Przeprowadź opisaną powyżej detekcję.
2. W katalogu ./temp powinien znaleźć się zapisany plik graficzny z wynikami detekcji

**Wymaganie P2**
1. Po uruchomieniu aplikacji (patrz wymaganie W2) oraz zaczytaniu obrazu interfejs w całości powinien mieścić się w obszarze monitora
Wszystkie elementy interfejsu powinny być widoczne dla użytkownika.

**Wymaganie P3**
1. Aby skonfigurować kolor zasięgów detekcyjnych w poleceniu należy użyć argumentu --bbox_color. 
```bash
python main.py --bbox_color='darkred'
```
Źródło dostępnych kolorów znajduje się pod adresem: https://matplotlib.org/3.1.1/_images/sphx_glr_named_colors_003.png
Opcja została wprowadzona, ze względu na to, że różne obrazy mogą mieć różne jasności i zasięgi mogą nie być widoczne 
w domyślnych kolorach zasięgów.


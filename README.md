# 🧠 SPJ – Spracovanie prirodzeného jazyka

Repozitár obsahuje praktické cvičenia z predmetu **Spracovanie prirodzeného jazyka (SPJ)**, vypracované v jazyku Python. Úlohy postupne pokrývajú tokenizáciu, reprezentáciu textu, klasifikáciu SMS správ, latentné sémantické indexovanie, statické a kontextové embeddingy, neurónové siete a architektúru Transformer.

## 📁 Štruktúra repozitára

| Priečinok | Obsah |
|---|---|
| `NLP_tokenizacia/` | Tokenizácia slovenských textov zo súboru CoNLL-U a tvorba riedkej matice tokenov. |
| `NLP_spam/` | Základná klasifikácia SMS správ na `ham` a `spam` pomocou binárnej reprezentácie a logistickej regresie. |
| `TF-IDF/` | TF-IDF váhovanie, vyhľadávanie podobných dokumentov a klasifikácia SMS. |
| `TF-IDF+LSI+Else/` | Rozšírené experimenty: LSI, Word2Vec (CBOW a Skip-gram), CNN, RNN a LSTM vrátane porovnania výsledkov. |
| `Contextual_Embeddings_Transformer/` | Kontextové embeddingy pomocou BERT, výpočet kosínusovej podobnosti, vizualizácia výsledkov a teoretické spracovanie ELMo a Transformer architektúry. |

---

## 🔬 Cvičenie 4 – LSI, Word2Vec a neurónové siete

Projekt `TF-IDF+LSI+Else/` nadväzuje na predchádzajúce cvičenie s TF-IDF. Používa dataset **SMS Spam Collection** s **5 572** správami (`ham` / `spam`). Dáta sa delia na trénovaciu (80 %, 4 457 správ) a testovaciu (20 %, 1 115 správ) množinu, so stratifikáciou podľa triedy.

- **LSI:** TF-IDF matica a `TruncatedSVD` s 20, 50 a 100 latentnými témami.
- **Word2Vec:** trénovanie CBOW a Skip-gram modelov, 100-rozmerné embeddingy, priemerovanie vektorov slov na úrovni dokumentu.
- **Neurónové siete:** CNN, SimpleRNN a LSTM nad sekvenciami tokenov, inicializované embeddingmi zo Skip-gram Word2Vec.
- **Vyhodnotenie:** Accuracy, Precision, Recall a F1 pre klasifikáciu triedy `spam`.

### Výsledky experimentov

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| TF-IDF | 0,9865 | 1,0000 | 0,8993 | 0,9470 |
| LSI – 20 tém | 0,9812 | 0,9776 | 0,8792 | 0,9258 |
| LSI – 50 tém | 0,9830 | 0,9710 | 0,8993 | 0,9338 |
| LSI – 100 tém | 0,9883 | 0,9857 | 0,9262 | 0,9550 |
| Word2Vec – CBOW | 0,9740 | 0,9225 | 0,8792 | 0,9003 |
| Word2Vec – Skip-gram | 0,9830 | 0,9851 | 0,8859 | 0,9329 |
| CNN | 0,9901 | 0,9792 | 0,9463 | 0,9625 |
| **RNN** | **0,9919** | **0,9861** | **0,9530** | **0,9693** |
| LSTM | 0,9901 | 0,9859 | 0,9396 | 0,9622 |

**Najvyššie F1 skóre v tomto experimente dosiahla jednoduchá RNN: 0,9693.**

Hodnoty pochádzajú z uskutočnených behov a môžu sa pri opakovanom trénovaní mierne líšiť.

### Spustenie cvičenia 4

V termináli prejdite z koreňového priečinka repozitára do projektu:

```powershell
cd 'TF-IDF+LSI+Else'
python -m pip install -r requirements.txt
```

Jednotlivé skripty spúšťajte v nasledujúcom poradí:

```powershell
python main.py
python test_project.py
python -m experiments.lsi
python -m experiments.word2vec_model
python -m experiments.neural_networks
python comparison.py
```

**Poznámky:**

- `data/spam.csv` obsahuje vstupné dáta.
- `src/` obsahuje spoločné funkcie na spracovanie textu, váhovanie, klasifikáciu a vyhľadávanie.
- `experiments/` obsahuje experimenty LSI, Word2Vec a neurónových sietí.
- `models/` obsahuje uložené Word2Vec modely. Skript neurónových sietí načítava `models/word2vec_1.model`, preto musí byť tento model dostupný.
- `results/neural_results.csv` a `results/comparison_results.csv` obsahujú výsledky porovnania.
- `comparison.py` používa zaznamenané hodnoty klasických modelov a načítava metriky neurónových sietí z `results/neural_results.csv`.
- Príkazy treba spúšťať z priečinka `TF-IDF+LSI+Else/`, aby relatívne cesty fungovali správne.

---

## 🧠 Cvičenie 5 – Kontextové embeddingy a Transformer

Projekt `Contextual_Embeddings_Transformer/` sa zameriava na kontextové vektorové reprezentácie slov, architektúru ELMo a Transformer a rozdiely medzi enkóderovými a dekóderovými jazykovými modelmi.

### 1. Kontextové embeddingy

V praktickej časti sa využíva predtrénovaný model **BERT (`bert-base-uncased`)** na vytváranie kontextových embeddingov anglického slova `bank`.

Experiment používa štyri vety s dvoma rôznymi významami cieľového slova:

| Označenie | Veta | Význam slova |
|---|---|---|
| S1 | I deposited money in the bank. | Finančná inštitúcia |
| S2 | She works at the bank. | Finančná inštitúcia |
| S3 | We sat on the bank of the river. | Breh rieky |
| S4 | The fisherman walked along the river bank. | Breh rieky |

Model BERT vytvára pre každé použitie slova samostatný embedding s rozmerom **768**. Ak je slovo rozdelené na viacero subtokenov, ich reprezentácie sa spriemerujú.

Na porovnanie výsledných reprezentácií sa používa kosínusová podobnosť (Cosine Similarity).

### 2. Výsledky experimentu

| Porovnanie | Kosínusová podobnosť |
|---|---:|
| S1 – S2 (finančná inštitúcia) | 0,7580 |
| S3 – S4 (breh rieky) | 0,8159 |
| S1 – S3 | 0,4987 |
| S1 – S4 | 0,5169 |
| S2 – S3 | 0,4792 |
| S2 – S4 | 0,5261 |

Priemerná kosínusová podobnosť medzi reprezentáciami s rôznymi významami je **0,5052**.

Výsledky ukazujú, že reprezentácie slova s rovnakým významom dosahujú v tomto experimente vyššiu podobnosť než reprezentácie s rôznymi významami.

### 3. Vizualizácia výsledkov

Na prehľadné zobrazenie podobnosti medzi jednotlivými embeddingmi sa používa tepelná mapa (Heatmap).

Výstupné súbory:

- `results/similarity_matrix.csv` – matica kosínusových podobností.
- `results/similarity_heatmap.png` – grafické znázornenie podobností medzi vetami.

### 4. Teoretická časť

Súčasťou cvičenia je aj teoretické spracovanie nasledujúcich tém:

- **Kontextové embeddingy:** rozdiel medzi statickými a kontextovými reprezentáciami slov.
- **ELMo:** úloha obojsmernej siete BiLSTM, jednotlivé vrstvy modelu a kombinovanie ich reprezentácií.
- **Transformer:** Self-Attention, Multi-Head Attention, Feed Forward Network a rozdiel medzi enkóderom a dekóderom.
- **Jazykové modely:** obojsmerný a kauzálny attention, Masked Language Modeling (MLM), Next Sentence Prediction (NSP) a Next Token Prediction (NTP).
- **Porovnanie modelov:** výhody a obmedzenia enkóderových a dekóderových jazykových modelov.

Teoretické odpovede sú uložené v súbore `theory.md`.

### 5. Štruktúra projektu

```text
Contextual_Embeddings_Transformer/
│
├── experiments/
│   ├── __init__.py
│   ├── contextual_embeddings.py
│   ├── similarity.py
│   └── visualization.py
│
├── results/
│   ├── similarity_matrix.csv
│   └── similarity_heatmap.png
│
├── main.py
├── requirements.txt
├── theory.md
└── README.md
```

### 6. Spustenie cvičenia 5

V termináli prejdite z koreňového priečinka repozitára do projektu:

```powershell
cd 'Contextual_Embeddings_Transformer'
```

Vytvorte a aktivujte virtuálne prostredie (ak ešte neexistuje):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Nainštalujte potrebné knižnice:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Spustite experiment:

```powershell
python main.py
```

Pri prvom spustení sa z internetu automaticky stiahne predtrénovaný model BERT `bert-base-uncased` z Hugging Face.

**Poznámky:**

- `experiments/contextual_embeddings.py` obsahuje načítanie modelu BERT a extrakciu kontextových embeddingov cieľového slova.
- `experiments/similarity.py` obsahuje výpočet kosínusovej podobnosti a tvorbu matice podobností.
- `experiments/visualization.py` zabezpečuje vytvorenie a uloženie tepelnej mapy.
- `main.py` spúšťa celý experiment, spracováva štyri vstupné vety a vyhodnocuje podobnosť získaných reprezentácií.
- `results/` obsahuje vygenerované výsledky experimentu.
- `theory.md` obsahuje teoretické odpovede k jednotlivým témam cvičenia.
- Príkazy treba spúšťať z priečinka `Contextual_Embeddings_Transformer/`, aby relatívne cesty k výstupným súborom fungovali správne.

---

## 🛠️ Technológie

V jednotlivých cvičeniach sú použité nasledujúce technológie a knižnice:

- **Python** – hlavný programovací jazyk.
- **NumPy** – numerické výpočty a práca s vektormi.
- **pandas** – spracovanie dát a tvorba tabuliek.
- **SciPy** – vedecké a matematické výpočty.
- **scikit-learn** – klasifikácia, TF-IDF a LSI.
- **Gensim** – implementácia Word2Vec.
- **TensorFlow/Keras** – implementácia neurónových sietí.
- **PyTorch** – práca s neurónovými sieťami a embeddingmi.
- **Hugging Face Transformers** – načítanie a využitie predtrénovaného modelu BERT.
- **Matplotlib** – vizualizácia výsledkov a tvorba heatmap.

Závislosti jednotlivých cvičení sú uvedené v príslušných súboroch `requirements.txt` v jednotlivých projektových priečinkoch.

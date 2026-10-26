# 🧠 SPJ – Spracovanie prirodzeného jazyka

Repozitár obsahuje praktické cvičenia z predmetu **Spracovanie prirodzeného jazyka (SPJ)**, vypracované v jazyku Python. Úlohy postupne pokrývajú tokenizáciu, reprezentáciu textu, klasifikáciu SMS správ, latentné sémantické indexovanie, statické embeddingy a neurónové siete.

## 📁 Štruktúra repozitára

| Priečinok | Obsah |
|---|---|
| `NLP_tokenizacia/` | Tokenizácia slovenských textov zo súboru CoNLL-U a tvorba riedkej matice tokenov. |
| `NLP_spam/` | Základná klasifikácia SMS správ na `ham` a `spam` pomocou binárnej reprezentácie a logistickej regresie. |
| `TF-IDF/` | TF-IDF váhovanie, vyhľadávanie podobných dokumentov a klasifikácia SMS. |
| `TF-IDF+LSI+Else/` | Rozšírené experimenty: LSI, Word2Vec (CBOW a Skip-gram), CNN, RNN a LSTM vrátane porovnania výsledkov. |

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

**Najvyššie F1 skóre v tomto experimente dosiahla jednoduchá RNN: 0,9693.** Hodnoty pochádzajú z uskutočnených behov a môžu sa pri opakovanom trénovaní mierne líšiť.

## ▶️ Spustenie cvičenia 4

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

## 🛠️ Technológie

Python, NumPy, pandas, SciPy, scikit-learn, Gensim a TensorFlow/Keras. Závislosti pre cvičenie 4 sú uvedené v `TF-IDF+LSI+Else/requirements.txt`.

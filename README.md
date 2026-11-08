# 🧠 SPJ – Spracovanie prirodzeného jazyka

Repozitár obsahuje praktické cvičenia z predmetu **Spracovanie prirodzeného jazyka (SPJ)**, vypracované v jazyku Python. Projekt sa zameriava na spracovanie textu, jeho reprezentáciu, klasifikáciu, jazykové modely a neurónové siete.

## 📁 Štruktúra repozitára

| Priečinok | Obsah |
|---|---|
| `NLP_tokenizacia/` | Tokenizácia slovenských textov zo súboru CoNLL-U a tvorba riedkej matice tokenov. |
| `NLP_spam/` | Klasifikácia SMS správ na `ham` a `spam` pomocou binárnej reprezentácie a logistickej regresie. |
| `TF-IDF/` | TF-IDF váhovanie, vyhľadávanie podobných dokumentov a klasifikácia SMS. |
| `TF-IDF+LSI+Else/` | LSI, Word2Vec (CBOW a Skip-gram), CNN, RNN a LSTM. |
| `Contextual_Embeddings_Transformer/` | Kontextové embeddingy pomocou BERT, výpočet kosínusovej podobnosti, vizualizácia výsledkov a teoretické spracovanie ELMo a Transformer architektúry. |

## 🛠️ Použité technológie

- **Python** – hlavný programovací jazyk.
- **NumPy, pandas, SciPy** – práca s dátami a numerické výpočty.
- **scikit-learn** – TF-IDF, LSI a klasifikácia.
- **Gensim** – Word2Vec.
- **TensorFlow/Keras** – neurónové siete.
- **PyTorch** – práca s neurónovými sieťami a embeddingmi.
- **Hugging Face Transformers** – predtrénované jazykové modely.
- **Matplotlib** – vizualizácia výsledkov.

Jednotlivé cvičenia obsahujú vlastnú projektovú štruktúru a súbory `requirements.txt` s potrebnými závislosťami.
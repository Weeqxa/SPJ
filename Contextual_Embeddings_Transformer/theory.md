
# 5. cvičenie – Kontextové embeddingy a Transformer

## 1. Kontextové embeddingy

### 1.1 Statické a kontextové embeddingy

Statické embeddingy predstavujú slová pomocou vektorov, ktoré zostávajú rovnaké bez ohľadu na kontext, v ktorom sa slovo nachádza.

Príkladom statických embeddingov sú Word2Vec a GloVe.

Problém statických embeddingov spočíva v tom, že jedno slovo má vždy rovnakú vektorovú reprezentáciu, aj keď môže mať viacero významov.

Napríklad anglické slovo „bank“ môže označovať:
- finančnú inštitúciu,
- breh rieky.

Pri použití statických embeddingov bude mať slovo „bank“ v oboch prípadoch rovnaký vektor.

Kontextové embeddingy tento problém riešia. Vektorová reprezentácia slova sa vytvára na základe kontextu, v ktorom sa slovo nachádza.

Preto môže mať rovnaké slovo v rôznych vetách odlišné vektorové reprezentácie.

Príkladom modelov využívajúcich kontextové embeddingy sú ELMo a BERT.

### 1.2 Praktická realizácia pomocou BERT

V praktickej časti sme použili predtrénovaný model BERT (bert-base-uncased) na vytvorenie kontextových embeddingov anglického slova „bank“.

Použili sme štyri vety:

1. I deposited money in the bank.
2. She works at the bank.
3. We sat on the bank of the river.
4. The fisherman walked along the river bank.

V prvých dvoch vetách slovo „bank“ označuje finančnú inštitúciu. V ďalších dvoch vetách označuje breh rieky.

Model BERT vytvoril pre každé použitie slova samostatnú vektorovú reprezentáciu s rozmerom 768.

Na porovnanie reprezentácií sme použili kosínusovú podobnosť (Cosine Similarity).

Výsledky experimentu:

| Porovnanie | Kosínusová podobnosť |
|---|---:|
| Finančná inštitúcia – S1 a S2 | 0.7580 |
| Breh rieky – S3 a S4 | 0.8159 |
| Priemerná podobnosť rôznych významov | 0.5052 |

Výsledky ukázali, že reprezentácie slova s rovnakým významom dosahujú vyššiu podobnosť ako reprezentácie s rôznymi významami.

Experiment tak demonštruje schopnosť modelu BERT vytvárať kontextové reprezentácie slov.

---

## 2. ELMo

### 2.1 Úloha siete BiLSTM

ELMo (Embeddings from Language Models) je model na vytváranie kontextových vektorových reprezentácií slov.

Je založený na obojsmernej rekurentnej neurónovej sieti BiLSTM (Bidirectional Long Short-Term Memory).

BiLSTM pozostáva z dvoch LSTM sietí:

- Forward LSTM spracováva text zľava doprava.
- Backward LSTM spracováva text sprava doľava.

Vďaka tomu model získava informácie z oboch smerov a dokáže lepšie pochopiť význam slova v kontexte celej vety.

Napríklad pri spracovaní slova „bank“ môže model využiť predchádzajúce aj nasledujúce slová na určenie jeho významu.

### 2.2 Informácie zachytávané jednotlivými vrstvami

Model ELMo využíva viacero vrstiev BiLSTM, pričom jednotlivé vrstvy zachytávajú rôzne typy jazykových informácií.

- Prvá vrstva zachytáva predovšetkým syntaktické informácie, napríklad gramatické vlastnosti slov a ich vzťahy v texte.
- Druhá vrstva zachytáva viac sémantických informácií, napríklad význam slov v konkrétnom kontexte.

Vstupná reprezentácia slov je vytvorená pomocou znakovej konvolučnej neurónovej siete (Character CNN).

Kombinácia týchto reprezentácií umožňuje modelu vytvárať bohatšie kontextové embeddingy.

### 2.3 Kombinovanie reprezentácií jednotlivých vrstiev

ELMo kombinuje reprezentácie z jednotlivých vrstiev pomocou naučeného váženého súčtu.

Každá vrstva má svoju váhu, ktorá určuje jej príspevok k výslednej reprezentácii.

Výsledný embedding možno vyjadriť vzťahom:

ELMo_k = γ Σ (s_j * h_k,j)

Kde:
- h_k,j predstavuje reprezentáciu slova k z vrstvy j,
- s_j predstavuje naučenú normalizovanú váhu danej vrstvy,
- γ je naučený skalárny parameter, ktorý upravuje veľkosť výslednej reprezentácie.

Vďaka tomuto mechanizmu môže model využívať rôzne jazykové informácie podľa konkrétnej úlohy.

---

## 3. Transformer architektúra – základné princípy

### 3.1 Attention mechanizmus

Attention je mechanizmus, ktorý umožňuje modelu určiť, na ktoré slová vo vstupnej sekvencii sa má pri spracovaní konkrétneho slova sústrediť.

Transformer využíva mechanizmus Self-Attention, pri ktorom každý token môže získavať informácie aj z ostatných tokenov v sekvencii.

Self-Attention pracuje s tromi typmi reprezentácií:

- Query (Q) – reprezentuje informáciu, ktorú token hľadá.
- Key (K) – reprezentuje informáciu, podľa ktorej sa vyhodnocuje relevantnosť ostatných tokenov.
- Value (V) – obsahuje informácie, ktoré sa následne kombinujú.

Výpočet attention možno vyjadriť vzťahom:

Attention(Q, K, V) = softmax(QKᵀ / √d_k) V

Kde d_k predstavuje rozmer vektorov Key.

Transformer využíva aj Multi-Head Attention, ktorý umožňuje modelu súčasne sledovať rôzne vzťahy medzi slovami.

### 3.2 Dopredná neurónová sieť (Feed Forward Network)

Druhou základnou súčasťou Transformer bloku je Feed Forward Network (FFN).

Ide o doprednú neurónovú sieť, ktorá spracováva reprezentáciu každého tokenu samostatne.

Typicky pozostáva z dvoch lineárnych vrstiev a nelineárnej aktivačnej funkcie.

Jej úlohou je transformovať reprezentácie získané pomocou attention mechanizmu a zachytiť ďalšie zložitejšie vzťahy v dátach.

### 3.3 Enkóder a dekóder v pôvodnej architektúre

Pôvodná architektúra Transformer pozostáva z dvoch hlavných častí:

**Enkóder (Encoder)**

Enkóder spracováva vstupnú sekvenciu a vytvára jej kontextové reprezentácie.

Jeho základné komponenty sú:
- Multi-Head Self-Attention.
- Feed Forward Network.
- Residual Connections.
- Layer Normalization.

Enkóder využíva obojsmerný attention, takže pri spracovaní tokenu môže využívať informácie z ľavej aj pravej strany.

Vstupom enkódera je zdrojová sekvencia tokenov. Výstupom sú kontextové vektorové reprezentácie týchto tokenov.

**Dekóder (Decoder)**

Dekóder využíva výstupy enkódera na generovanie cieľovej sekvencie.

Jeho základné komponenty sú:
- Masked Multi-Head Self-Attention.
- Encoder-Decoder Attention.
- Feed Forward Network.
- Residual Connections.
- Layer Normalization.

Masked Self-Attention zabezpečuje, že dekóder pri generovaní tokenu nemôže využívať informácie o budúcich tokenoch.

Encoder-Decoder Attention umožňuje dekóderu získavať informácie z výstupov enkódera.

Vstupom dekódera je predchádzajúca časť cieľovej sekvencie spolu s reprezentáciami enkódera. Výstupom sú pravdepodobnosti nasledujúcich tokenov.

### 3.4 Prečo moderné modely využívajú iba enkóder alebo dekóder?

Moderné jazykové modely často využívajú iba jednu časť pôvodnej Transformer architektúry, pretože rôzne úlohy vyžadujú odlišný spôsob spracovania textu.

**Enkóderové modely:**

Príkladom je BERT.

Sú vhodné najmä na:
- klasifikáciu textu,
- analýzu sentimentu,
- extrakciu informácií,
- porozumenie textu.

Dokážu spracovávať kontext z oboch strán textu.

**Dekóderové modely:**

Príkladom je GPT.

Sú vhodné najmä na:
- generovanie textu,
- odpovedanie na otázky,
- tvorbu súvislých textov,
- konverzáciu.

Generujú text postupne, pričom každý nový token závisí od predchádzajúcich tokenov.

Použitie iba jednej časti architektúry umožňuje prispôsobiť model konkrétnemu typu úlohy.

---

## 4. Enkóderové a dekóderové jazykové modely

### 4.1 Obojsmerný a kauzálny Attention

Hlavným rozdielom medzi enkóderovými a dekóderovými modelmi je spôsob využívania informácií z textu.

**Obojsmerný Attention (Bidirectional Attention)**

Používa sa v enkóderových modeloch, napríklad BERT.

Každý token môže pri spracovaní využívať informácie zo všetkých ostatných tokenov v sekvencii vrátane predchádzajúcich aj nasledujúcich tokenov.

Tento prístup umožňuje lepšie porozumenie kontextu a významu slov.

**Kauzálny Attention (Causal alebo Masked Attention)**

Používa sa v dekóderových modeloch, napríklad GPT.

Každý token môže využívať iba informácie z predchádzajúcich tokenov a aktuálneho tokenu.

Prístup k budúcim tokenom je zablokovaný pomocou attention masky.

Týmto spôsobom model dokáže generovať text postupne bez znalosti budúcich slov.

### 4.2 Masked Language Modeling (MLM)

Masked Language Modeling je tréningová úloha používaná napríklad pri modeli BERT.

Počas tréningu sa v texte náhodne zamaskujú niektoré tokeny a úlohou modelu je predpovedať ich pôvodnú podobu na základe kontextu.

Príklad:

Pôvodná veta:
"The cat sits on the mat."

Veta po maskovaní:
"The cat sits on the [MASK]."

Model sa snaží predpovedať chýbajúce slovo „mat“.

Táto úloha umožňuje modelu učiť sa vzťahy medzi slovami a využívať informácie z oboch smerov.

### 4.3 Next Sentence Prediction (NSP)

Next Sentence Prediction je tréningová úloha používaná v pôvodnom modeli BERT.

Úlohou modelu je určiť, či druhá veta prirodzene nasleduje po prvej vete v pôvodnom texte.

Príklad:

Veta A: "I went to the store."

Veta B: "I bought some milk."

Tieto vety môžu tvoriť logicky nadväzujúcu dvojicu.

Model sa učí klasifikovať, či druhá veta skutočne nasleduje po prvej, alebo ide o náhodne vybranú vetu.

NSP pomáha modelu učiť sa vzťahy medzi vetami.

### 4.4 Next Token Prediction (NTP)

Next Token Prediction je základná tréningová úloha dekóderových jazykových modelov, napríklad GPT.

Úlohou modelu je predpovedať nasledujúci token na základe všetkých predchádzajúcich tokenov.

Príklad:

Vstup:
"I like to drink"

Možné pokračovanie:
"coffee"

Model postupne predpovedá ďalšie tokeny a vytvára tak súvislý text.

Na rozdiel od MLM sa pri tejto úlohe využíva kauzálny attention a model nemá prístup k budúcim tokenom.

### 4.5 Výhody a obmedzenia enkóderových a dekóderových modelov

**Enkóderové modely (napríklad BERT)**

Výhody:
- Využívajú obojsmerný kontext.
- Efektívne zachytávajú význam slov a viet.
- Sú vhodné na klasifikáciu a porozumenie textu.
- Dokážu vytvárať kvalitné kontextové reprezentácie.

Obmedzenia:
- Nie sú primárne navrhnuté na postupné generovanie textu.
- Ich obojsmerný attention nie je vhodný na štandardné autoregresívne generovanie.

**Dekóderové modely (napríklad GPT)**

Výhody:
- Sú vhodné na generovanie prirodzeného textu.
- Dokážu postupne vytvárať dlhšie textové sekvencie.
- Sú vhodné na konverzáciu a tvorbu obsahu.
- Môžu využívať predchádzajúci kontext pri predikcii ďalšieho tokenu.

Obmedzenia:
- Pri predikcii tokenu nemôžu využívať budúci kontext.
- Generovanie prebieha postupne, čo môže byť výpočtovo náročné.
- Pri dlhých sekvenciách rastú nároky na pamäť a výpočtový výkon.

### 4.6 Zhrnutie

Enkóderové a dekóderové jazykové modely sa líšia predovšetkým spôsobom spracovania kontextu a cieľom, na ktorý sú navrhnuté.

Enkóderové modely využívajú obojsmerný attention a sú vhodné na porozumenie textu.

Dekóderové modely využívajú kauzálny attention a sú vhodné na generovanie textu.

Oba typy modelov predstavujú dôležitú súčasť moderného spracovania prirodzeného jazyka (NLP).

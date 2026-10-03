import re
from scipy.sparse import lil_matrix
import csv


sentence = ""
sentences = []

original_sentence = []
original_sentences = []

skip_until = 0


with open("sk_snk-ud-train.conllu", "r", encoding="utf-8") as file:
    for line in file:

        if line.startswith("#"):
            continue

        if line.strip() == "":
            sentences.append(sentence)
            original_sentences.append(original_sentence)

            sentence = ""
            original_sentence = []

            skip_until = 0
            continue

        parts = line.strip().split("\t")

        token_id = parts[0]

        form = parts[1]
        misc = parts[9]

        if "-" in token_id:
            start, end = token_id.split("-")
            skip_until = int(end)

            original_sentence.append(form)

        elif token_id.isdigit() and int(token_id) <= skip_until:
            continue

        else:
            original_sentence.append(form)

        if "SpaceAfter=No" in misc:
            sentence += form

        else:
            sentence += form + " "  

all_tokens = set()
sentences_tokens = []

for sentence in sentences:
    tokens = re.findall(r"\w+|--|[^\w\s]", sentence, re.UNICODE)
    sentences_tokens.append(tokens)
    for token in tokens:
        all_tokens.add(token)

unique_tokens = sorted(all_tokens)

print("Number of unique tokens:", len(unique_tokens))


token_to_index = {}

for i, token in enumerate(unique_tokens):
    token_to_index[token] = i


# --------------------------------
# Створення sparse matrix
# --------------------------------

matrix = lil_matrix(
    (len(sentences_tokens), len(unique_tokens)),
    dtype=int
)


# --------------------------------
# Заповнення matrix
# --------------------------------

for sentence_index, tokens in enumerate(sentences_tokens):

    for token in set(tokens):

        token_index = token_to_index[token]

        matrix[sentence_index, token_index] = 1


# Перетворення LIL -> CSR
matrix = matrix.tocsr()


# --------------------------------
# Запис у CSV
# --------------------------------

with open("token_matrix.csv", "w", encoding="utf-8-sig", newline="") as file:

    writer = csv.writer(file)

    writer.writerow(["Sentence"] + unique_tokens)

    for sentence_index, sentence in enumerate(sentences):

        row = [0] * len(unique_tokens)

        for token_index in matrix[sentence_index].indices:
            row[token_index] = 1

        writer.writerow([sentence] + row)


print("Matrix saved to token_matrix.csv")
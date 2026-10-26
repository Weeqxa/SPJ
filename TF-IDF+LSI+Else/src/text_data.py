
import re
import pandas as pd
from scipy.sparse import lil_matrix


def load_data(filename):
    # Nacitanie datasetu
    data = pd.read_csv(filename)
    return data["Message"], data["Category"]


def tokenize(text):
    # Rozdelenie textu na tokeny
    return re.findall(r"\w+", text.lower())


def build_vocabulary(documents):
    # Vytvorenie slovnika zo vsetkych dokumentov
    all_tokens = set()

    for document in documents:
        all_tokens.update(tokenize(document))

    unique_tokens = sorted(all_tokens)

    return {
        token: index
        for index, token in enumerate(unique_tokens)
    }


def create_frequency_matrix(documents, vocabulary):
    # Vytvorenie frekvencnej term-dokument matice
    matrix = lil_matrix(
        (len(documents), len(vocabulary)),
        dtype=int
    )

    for document_index, document in enumerate(documents):
        tokens = tokenize(document)

        for token in tokens:
            if token in vocabulary:
                token_index = vocabulary[token]
                matrix[document_index, token_index] += 1

    return matrix.tocsr()


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .text_data import create_frequency_matrix
from .weighting import calculate_tfidf


def search_documents(
    query,
    messages,
    categories,
    vocabulary,
    tfidf_matrix,
    tf_type,
    idf,
    top_n=10
):
    # Tokenizacia a frekvencny vektor dopytu
    query_matrix = create_frequency_matrix(
        [query], vocabulary
    )

    # Vypocet TF-IDF vah dopytu
    query_tfidf = calculate_tfidf(
        query_matrix, tf_type, idf
    )

    # Kosinusova podobnost
    similarities = cosine_similarity(
        query_tfidf, tfidf_matrix
    ).ravel()

    # Zoradenie dokumentov zostupne
    sorted_indices = np.argsort(-similarities, kind="stable")

    print("\nQuery:", query)
    print("Top", top_n, "similar documents:")

    for i in sorted_indices[:top_n]:
        print()
        print("Document:", i)
        print("Category:", categories.iloc[i])
        print("Similarity:", round(similarities[i], 4))
        print("Message:", messages.iloc[i])

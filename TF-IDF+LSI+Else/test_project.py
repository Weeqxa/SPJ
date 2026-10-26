
import numpy as np

from src.text_data import (
    load_data,
    build_vocabulary,
    create_frequency_matrix
)

from src.weighting import (
    TF_VARIANTS,
    IDF_VARIANTS,
    calculate_idf,
    calculate_tfidf
)

from sklearn.metrics.pairwise import cosine_similarity


def test_all_weightings():
    # Test vsetkych kombinacii TF a IDF
    messages, _ = load_data("data/spam.csv")
    vocabulary = build_vocabulary(messages)
    matrix = create_frequency_matrix(messages, vocabulary)

    print("Testovanie TF-IDF kombinacii:")

    successful = 0
    failed = 0

    for tf_type in TF_VARIANTS:
        for idf_type in IDF_VARIANTS:
            try:
                idf = calculate_idf(matrix, idf_type)
                tfidf = calculate_tfidf(matrix, tf_type, idf)

                assert tfidf.shape == matrix.shape
                assert np.all(np.isfinite(tfidf.data))

                print("OK:", tf_type, "+", idf_type)
                successful += 1

            except Exception as error:
                print("ERROR:", tf_type, "+", idf_type)
                print("Reason:", error)
                failed += 1

    print()
    print("Successful:", successful)
    print("Failed:", failed)


def test_unknown_words():
    # Test dopytu s neznamymi slovami
    messages, _ = load_data("data/spam.csv")
    vocabulary = build_vocabulary(messages)
    matrix = create_frequency_matrix(messages, vocabulary)

    query = "xyznonexistentword123 abcunknown456"

    query_matrix = create_frequency_matrix(
        [query], vocabulary
    )

    idf = calculate_idf(matrix, "vyhladena")
    query_tfidf = calculate_tfidf(
        query_matrix, "logaritmicka", idf
    )

    document_tfidf = calculate_tfidf(
        matrix, "logaritmicka", idf
    )

    similarities = cosine_similarity(
        query_tfidf, document_tfidf
    ).ravel()

    assert query_matrix.nnz == 0
    assert np.all(similarities == 0)

    print("\nTest neznamych slov: OK")
    print("Similarity:", similarities[:5])


def main():
    test_all_weightings()
    test_unknown_words()


if __name__ == "__main__":
    main()

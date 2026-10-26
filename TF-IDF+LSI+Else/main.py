
from src.text_data import (
    load_data,
    build_vocabulary,
    create_frequency_matrix
)

from src.weighting import calculate_idf, calculate_tfidf

from src.search import search_documents

from src.classification import (
    prepare_data,
    train_model,
    evaluate_model,
    compare_weightings,
    predict_message
)


# Nastavenia
TF_TYPE = "logaritmicka"
IDF_TYPE = "vyhladena"

QUERY = "Congratulations you won a free prize"
NEW_MESSAGE = "You have won a free cash prize"


def main():
    # 1. Nacitanie a tokenizacia dokumentov
    messages, categories = load_data("data/spam.csv")
    vocabulary = build_vocabulary(messages)

    print("Messages:", len(messages))
    print("Unique tokens:", len(vocabulary))

    # 2. Frekvencna term-dokument matica
    matrix = create_frequency_matrix(messages, vocabulary)

    print("\nFrekvencna matica:")
    print("Matrix shape:", matrix.shape)
    print("Non-zero values:", matrix.nnz)

    # 3. TF-IDF vahovanie
    idf = calculate_idf(matrix, IDF_TYPE)
    tfidf_matrix = calculate_tfidf(matrix, TF_TYPE, idf)

    print("\nTF-IDF vahovanie:")
    print("TF:", TF_TYPE)
    print("IDF:", IDF_TYPE)
    print("Matrix shape:", tfidf_matrix.shape)

    # 4. Vyhladavanie dokumentov
    search_documents(
        QUERY,
        messages,
        categories,
        vocabulary,
        tfidf_matrix,
        TF_TYPE,
        idf
    )

    # 5. Priprava dat na klasifikaciu
    (
        train_matrix,
        test_matrix,
        y_train,
        y_test,
        train_vocabulary
    ) = prepare_data(messages, categories)

    # IDF sa pocita iba na trenovacej mnozine
    train_idf = calculate_idf(train_matrix, IDF_TYPE)

    X_train = calculate_tfidf(
        train_matrix, TF_TYPE, train_idf
    )

    X_test = calculate_tfidf(
        test_matrix, TF_TYPE, train_idf
    )

    # 6. Trenovanie a vyhodnotenie modelu
    model = train_model(X_train, y_train)

    predictions, metrics = evaluate_model(
        model, X_test, y_test
    )

    print("\nKlasifikacia:")
    print("First 10 predictions:", predictions[:10])

    for name, value in metrics.items():
        print(name + ":", round(value, 4))

    # 7. Predikcia noveho zaznamu
    prediction = predict_message(
        NEW_MESSAGE,
        model,
        train_vocabulary,
        TF_TYPE,
        train_idf
    )

    print("\nNew message:", NEW_MESSAGE)
    print("Predicted category:", prediction)

    # 8. Porovnanie roznych sposobov vahovania
    results = compare_weightings(
        train_matrix,
        test_matrix,
        y_train,
        y_test
    )

    print("\nPorovnanie TF-IDF vahovania:")
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

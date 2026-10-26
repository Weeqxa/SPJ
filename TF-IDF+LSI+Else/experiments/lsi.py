
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from src.text_data import load_data
from src.classification import prepare_data, train_model, evaluate_model
from src.weighting import calculate_idf, calculate_tfidf


# Nastavenia
TF_TYPE = "logaritmicka"
IDF_TYPE = "vyhladena"
TOPICS = [20, 50, 100]


def main():
    # 1. Nacitanie a rozdelenie dat
    messages, categories = load_data("data/spam.csv")

    train_matrix, test_matrix, y_train, y_test, vocabulary = prepare_data(
        messages, categories
    )

    # 2. TF-IDF reprezentacia
    train_idf = calculate_idf(train_matrix, IDF_TYPE)

    X_train = calculate_tfidf(train_matrix, TF_TYPE, train_idf)
    X_test = calculate_tfidf(test_matrix, TF_TYPE, train_idf)

    print("Train documents:", X_train.shape[0])
    print("Test documents:", X_test.shape[0])
    print("TF-IDF shape:", X_train.shape)

    # 3. LSI s roznym poctom tem
    results = []

    for k in TOPICS:
        svd = TruncatedSVD(n_components=k, random_state=42)

        train_lsi = svd.fit_transform(X_train)
        test_lsi = svd.transform(X_test)

        # 4. Klasifikacia
        model = train_model(train_lsi, y_train)
        _, metrics = evaluate_model(model, test_lsi, y_test)

        results.append({
            "Topics": k,
            **metrics
        })

        print(
            f"Topics {k}: train {train_lsi.shape}, "
            f"test {test_lsi.shape}, "
            f"F1={metrics['F1']:.4f}"
        )

    # 5. Porovnanie vysledkov
    print("\nLSI comparison:")
    print(pd.DataFrame(results).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

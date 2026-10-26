
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from .text_data import build_vocabulary, create_frequency_matrix
from .weighting import calculate_idf, calculate_tfidf


def prepare_data(messages, categories):
    # Rozdelenie dat pred vytvorenim slovnika a IDF
    train_messages, test_messages, y_train, y_test = train_test_split(
        messages,
        categories,
        test_size=0.2,
        random_state=42,
        stratify=categories
    )

    # Slovnik vytvarame iba z trenovacich dat
    vocabulary = build_vocabulary(train_messages)

    train_matrix = create_frequency_matrix(
        train_messages, vocabulary
    )

    test_matrix = create_frequency_matrix(
        test_messages, vocabulary
    )

    return train_matrix, test_matrix, y_train, y_test, vocabulary


def train_model(X_train, y_train):
    # Trenovanie logistickej regresie
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    # Predikcia testovacich zaznamov
    predictions = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions,
            pos_label="spam", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions,
            pos_label="spam", zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions,
            pos_label="spam", zero_division=0
        )
    }

    return predictions, metrics


def compare_weightings(
    train_matrix, test_matrix, y_train, y_test
):
    # Porovnanie roznych sposobov vahovania
    tf_variants = [
        "binarna",
        "absolutna",
        "logaritmicka"
    ]

    idf_variants = [
        "standardna",
        "vyhladena",
        "jednotkova"
    ]

    results = []

    for tf_type in tf_variants:
        for idf_type in idf_variants:

            # IDF pocitame iba z trenovacich dat
            idf = calculate_idf(train_matrix, idf_type)

            X_train = calculate_tfidf(
                train_matrix, tf_type, idf
            )

            X_test = calculate_tfidf(
                test_matrix, tf_type, idf
            )

            model = train_model(X_train, y_train)

            _, metrics = evaluate_model(
                model, X_test, y_test
            )

            results.append({
                "TF": tf_type,
                "IDF": idf_type,
                **metrics
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "F1", ascending=False
    )

    return results_df


def predict_message(
    message, model, vocabulary, tf_type, idf
):
    # Predikcia kategorie novej spravy
    message_matrix = create_frequency_matrix(
        [message], vocabulary
    )

    message_tfidf = calculate_tfidf(
        message_matrix, tf_type, idf
    )

    prediction = model.predict(message_tfidf)

    return prediction[0]

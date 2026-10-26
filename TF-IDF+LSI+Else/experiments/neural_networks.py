
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.text_data import load_data, tokenize


# Nastavenia
SEED = 42
MAX_LENGTH = 60
EMBEDDING_DIM = 100
EPOCHS = 5
BATCH_SIZE = 64

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def create_model(model_type, embedding_matrix):
    vocab_size = embedding_matrix.shape[0]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(MAX_LENGTH,)))

    # Predtrenovane Word2Vec embeddingy
    model.add(tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        weights=[embedding_matrix],
        mask_zero=(model_type != "CNN"),
        trainable=True
    ))

    if model_type == "CNN":
        model.add(tf.keras.layers.Conv1D(
            filters=64, kernel_size=3, activation="relu"
        ))
        model.add(tf.keras.layers.GlobalMaxPooling1D())

    elif model_type == "RNN":
        model.add(tf.keras.layers.SimpleRNN(32))

    elif model_type == "LSTM":
        model.add(tf.keras.layers.LSTM(32))

    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    # 1. Nacitanie dat
    messages, categories = load_data("data/spam.csv")

    train_messages, test_messages, y_train, y_test = train_test_split(
        messages,
        categories,
        test_size=0.2,
        random_state=42,
        stratify=categories
    )

    train_tokens = [tokenize(text) for text in train_messages]
    test_tokens = [tokenize(text) for text in test_messages]

    # 2. Slovnik a sekvencie tokenov
    train_texts = [" ".join(tokens) for tokens in train_tokens]
    test_texts = [" ".join(tokens) for tokens in test_tokens]

    vectorizer = tf.keras.layers.TextVectorization(
        standardize=None,
        split="whitespace",
        output_mode="int",
        output_sequence_length=MAX_LENGTH
    )
    vectorizer.adapt(tf.constant(train_texts))

    X_train = vectorizer(tf.constant(train_texts)).numpy()
    X_test = vectorizer(tf.constant(test_texts)).numpy()

    y_train = np.array([1 if x == "spam" else 0 for x in y_train])
    y_test = np.array([1 if x == "spam" else 0 for x in y_test])

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # 3. Word2Vec Skip-gram z predchadzajucej ulohy
    w2v = Word2Vec.load("models/word2vec_1.model")

    vocabulary = vectorizer.get_vocabulary()
    embedding_matrix = np.zeros(
        (len(vocabulary), EMBEDDING_DIM), dtype=np.float32
    )

    for i, word in enumerate(vocabulary):
        if word in w2v.wv:
            embedding_matrix[i] = w2v.wv[word]

    print("Vocabulary size:", len(vocabulary))
    print("Embedding matrix:", embedding_matrix.shape)

    # 4. Trenovanie troch modelov
    results = []

    for model_type in ["CNN", "RNN", "LSTM"]:
        print(f"\nTraining {model_type}")

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)

        model = create_model(model_type, embedding_matrix)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            validation_split=0.15,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=1
        )

        # 5. Vyhodnotenie na testovacich datach
        probabilities = model.predict(X_test, verbose=0).ravel()
        predictions = (probabilities >= 0.5).astype(int)

        metrics = {
            "Model": model_type,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_test, predictions, zero_division=0
            )
        }

        results.append(metrics)
        print(f"{model_type} F1: {metrics['F1']:.4f}")

    # 6. Porovnanie
    results_df = pd.DataFrame(results)

    print("\nNeural networks comparison:")
    print(results_df.round(4).to_string(index=False))

    results_df.to_csv("results/neural_results.csv", index=False)
    print("\nResults saved to results/neural_results.csv")


if __name__ == "__main__":
    main()

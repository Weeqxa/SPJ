
import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split

from src.text_data import load_data, tokenize
from src.classification import train_model, evaluate_model


def document_vector(tokens, model):
    # Priemer embeddingov slov v dokumente
    vectors = [
        model.wv[word]
        for word in tokens
        if word in model.wv
    ]

    if not vectors:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


def main():
    # 1. Nacitanie datasetu
    messages, categories = load_data("data/spam.csv")

    # Rovnaky train/test split ako pri TF-IDF a LSI
    train_messages, test_messages, y_train, y_test = train_test_split(
        messages,
        categories,
        test_size=0.2,
        random_state=42,
        stratify=categories
    )

    # 2. Tokenizacia
    train_tokens = [tokenize(text) for text in train_messages]
    test_tokens = [tokenize(text) for text in test_messages]

    print("Train documents:", len(train_tokens))
    print("Test documents:", len(test_tokens))

    results = []

    # 3. CBOW a Skip-gram
    for name, sg in [("CBOW", 0), ("Skip-gram", 1)]:
        print(f"\nTraining Word2Vec: {name}")

        model = Word2Vec(
            sentences=train_tokens,
            vector_size=100,
            window=5,
            min_count=1,
            workers=1,
            sg=sg,
            epochs=20,
            seed=42
        )

        print("Vocabulary size:", len(model.wv))
        print("Embedding dimension:", model.vector_size)

        # 4. Reprezentacia dokumentov
        X_train = np.array([
            document_vector(tokens, model)
            for tokens in train_tokens
        ])

        X_test = np.array([
            document_vector(tokens, model)
            for tokens in test_tokens
        ])

        print("Train shape:", X_train.shape)
        print("Test shape:", X_test.shape)

        # 5. Klasifikacia
        classifier = train_model(X_train, y_train)
        _, metrics = evaluate_model(classifier, X_test, y_test)

        results.append({
            "Model": name,
            **metrics
        })

        # Ulozenie modelu
        model.save(f"models/word2vec_{sg}.model")

    # 6. Porovnanie
    print("\nWord2Vec comparison:")
    print(pd.DataFrame(results).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

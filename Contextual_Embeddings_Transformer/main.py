
from pathlib import Path

import pandas as pd

from experiments.contextual_embeddings import (
    load_bert_model,
    get_contextual_embedding
)

from experiments.similarity import (
    create_similarity_matrix
)

from experiments.visualization import (
    plot_similarity_matrix
)


# Target word for the experiment.
TARGET_WORD = "bank"

# Sentences with different meanings of the target word.
SENTENCES = {
    "S1 - Financial institution": {
        "text": "I deposited money in the bank.",
        "meaning": "Financial institution"
    },
    "S2 - Financial institution": {
        "text": "She works at the bank.",
        "meaning": "Financial institution"
    },
    "S3 - River bank": {
        "text": "We sat on the bank of the river.",
        "meaning": "River bank"
    },
    "S4 - River bank": {
        "text": "The fisherman walked along the river bank.",
        "meaning": "River bank"
    }
}


def main():
    """
    Run the contextual embeddings experiment.
    """

    print("=" * 60)
    print("CONTEXTUAL EMBEDDINGS EXPERIMENT")
    print("=" * 60)

    # Create results directory.
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained BERT.
    tokenizer, model = load_bert_model()

    # Extract contextual embeddings.
    embeddings = {}

    print("\nExtracting contextual embeddings...\n")

    for label, data in SENTENCES.items():
        sentence = data["text"]

        embedding = get_contextual_embedding(
            sentence,
            TARGET_WORD,
            tokenizer,
            model
        )

        embeddings[label] = embedding

        print(f"{label}")
        print(f"Sentence: {sentence}")
        print(f"Embedding shape: {embedding.shape}")
        print("-" * 60)

    # Create cosine similarity matrix.
    similarity_matrix = create_similarity_matrix(embeddings)

    print("\nCOSINE SIMILARITY MATRIX")
    print(similarity_matrix.round(4))

    # Save similarity matrix.
    csv_path = results_dir / "similarity_matrix.csv"

    similarity_matrix.to_csv(csv_path)

    print(f"\nSimilarity matrix saved to: {csv_path}")

    # Create heatmap.
    heatmap_path = results_dir / "similarity_heatmap.png"

    plot_similarity_matrix(
        similarity_matrix,
        str(heatmap_path)
    )

    # Compare average similarities between meanings.
    labels = list(SENTENCES.keys())

    financial_indices = [
        i for i, data in enumerate(SENTENCES.values())
        if data["meaning"] == "Financial institution"
    ]

    river_indices = [
        i for i, data in enumerate(SENTENCES.values())
        if data["meaning"] == "River bank"
    ]

    values = similarity_matrix.values

    financial_similarity = values[
        financial_indices[0], financial_indices[1]
    ]

    river_similarity = values[
        river_indices[0], river_indices[1]
    ]

    cross_similarities = [
        values[i, j]
        for i in financial_indices
        for j in river_indices
    ]

    average_cross_similarity = sum(cross_similarities) / len(
        cross_similarities
    )

    print("\nEXPERIMENT RESULTS")
    print("-" * 60)

    print(
        f"Financial institution similarity: "
        f"{financial_similarity:.4f}"
    )

    print(
        f"River bank similarity: "
        f"{river_similarity:.4f}"
    )

    print(
        f"Average similarity between different meanings: "
        f"{average_cross_similarity:.4f}"
    )

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()

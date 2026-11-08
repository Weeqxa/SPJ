
import numpy as np
import pandas as pd


def cosine_similarity(vector_a: np.ndarray,
                      vector_b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vector_a: First embedding vector.
        vector_b: Second embedding vector.

    Returns:
        Cosine similarity value.
    """

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cannot compare zero vectors.")

    similarity = np.dot(vector_a, vector_b) / (norm_a * norm_b)

    return float(similarity)


def create_similarity_matrix(
    embeddings: dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Create a cosine similarity matrix for multiple embeddings.

    Args:
        embeddings: Dictionary containing sentence labels
                    and their embedding vectors.

    Returns:
        Pandas DataFrame with pairwise similarities.
    """

    if not embeddings:
        raise ValueError("Embeddings dictionary cannot be empty.")

    labels = list(embeddings.keys())
    vectors = np.stack(list(embeddings.values()))

    if not np.all(np.isfinite(vectors)):
        raise ValueError("Embeddings contain invalid values.")

    norms = np.linalg.norm(vectors, axis=1)

    if np.any(norms == 0):
        raise ValueError("Embeddings cannot contain zero vectors.")

    # Normalize vectors.
    normalized_vectors = vectors / norms[:, np.newaxis]

    # Calculate pairwise cosine similarities.
    similarity_matrix = normalized_vectors @ normalized_vectors.T

    return pd.DataFrame(
        similarity_matrix,
        index=labels,
        columns=labels
    )

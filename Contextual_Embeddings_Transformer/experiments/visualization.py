
import matplotlib.pyplot as plt
import pandas as pd


def plot_similarity_matrix(
    similarity_matrix: pd.DataFrame,
    output_path: str = "results/similarity_heatmap.png"
):
    """
    Visualize cosine similarity matrix using a heatmap.

    Args:
        similarity_matrix: DataFrame with pairwise similarities.
        output_path: Path where the heatmap will be saved.
    """

    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(
        similarity_matrix.values,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    # Add labels.
    ax.set_xticks(range(len(similarity_matrix.columns)))
    ax.set_yticks(range(len(similarity_matrix.index)))

    ax.set_xticklabels(
        similarity_matrix.columns,
        rotation=30,
        ha="right"
    )

    ax.set_yticklabels(similarity_matrix.index)

    # Add similarity values inside cells.
    for i in range(len(similarity_matrix.index)):
        for j in range(len(similarity_matrix.columns)):
            value = similarity_matrix.iloc[i, j]

            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="black"
            )

    ax.set_title("Contextual Embeddings - Cosine Similarity")
    ax.set_xlabel("Sentences")
    ax.set_ylabel("Sentences")

    fig.colorbar(
        image,
        ax=ax,
        label="Cosine Similarity"
    )

    fig.tight_layout()

    # Save visualization.
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

    print(f"Heatmap saved to: {output_path}")

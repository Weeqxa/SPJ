
import json
import numpy as np

from tqdm import tqdm

from config import (
    PROCESSED_DATA_DIR,
    EMBEDDING_DIMENSION,
)


# =====================================================
# PATHS
# =====================================================

CHUNKS_FILE = PROCESSED_DATA_DIR / "chunks.jsonl"
EMBEDDINGS_FILE = PROCESSED_DATA_DIR / "embeddings.npy"

DOCUMENT_EMBEDDINGS_FILE = (
    PROCESSED_DATA_DIR / "document_embeddings.npy"
)

DOCUMENT_METADATA_FILE = (
    PROCESSED_DATA_DIR / "document_metadata.jsonl"
)


# =====================================================
# DOCUMENT EMBEDDING GENERATION
# =====================================================

def create_document_embeddings():

    print("=" * 60)
    print("DOCUMENT EMBEDDINGS GENERATION")
    print("=" * 60)

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(CHUNKS_FILE)

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(EMBEDDINGS_FILE)

    # Load embeddings using memory mapping
    embeddings = np.load(
        EMBEDDINGS_FILE,
        mmap_mode="r",
    )

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    if embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError("Incorrect embedding dimension.")

    print(f"Chunk embeddings: {embeddings.shape}")

    total_chunks = embeddings.shape[0]

    # Count documents from the chunks file
    print("\nCounting documents...")

    document_ids = set()

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        for line in tqdm(file, desc="Reading document IDs"):
            if line.strip():
                chunk = json.loads(line)
                document_ids.add(chunk["document_id"])

    total_documents = len(document_ids)

    print(f"Unique documents: {total_documents:,}")

    # Create output memory-mapped array
    document_embeddings = np.lib.format.open_memmap(
        DOCUMENT_EMBEDDINGS_FILE,
        mode="w+",
        dtype=np.float32,
        shape=(total_documents, EMBEDDING_DIMENSION),
    )

    metadata_file = open(
        DOCUMENT_METADATA_FILE,
        "w",
        encoding="utf-8",
    )

    current_document_id = None
    current_source = None
    current_sum = np.zeros(
        EMBEDDING_DIMENSION,
        dtype=np.float64,
    )

    current_chunk_count = 0
    document_index = 0
    processed_chunks = 0

    closed_documents = set()

    def save_current_document():

        nonlocal document_index

        if current_chunk_count == 0:
            return

        # Mean of normalized chunk vectors
        mean_vector = (
            current_sum / current_chunk_count
        )

        # L2 normalization
        norm = np.linalg.norm(mean_vector)

        if norm == 0 or not np.isfinite(norm):
            raise ValueError(
                f"Invalid vector: {current_document_id}"
            )

        mean_vector = (
            mean_vector / norm
        )

        document_embeddings[document_index] = (
            mean_vector.astype(np.float32)
        )

        metadata = {
            "document_index": document_index,
            "document_id": current_document_id,
            "source": current_source,
            "chunk_count": current_chunk_count,
        }

        metadata_file.write(
            json.dumps(
                metadata,
                ensure_ascii=False,
            ) + "\n"
        )

        document_index += 1

    # Process chunks in their original order
    print("\nAggregating chunk embeddings...")

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        for line in tqdm(
            file,
            total=total_chunks,
            desc="Processing chunks",
        ):

            if not line.strip():
                continue

            chunk = json.loads(line)

            document_id = chunk["document_id"]

            if document_id != current_document_id:

                # Save previous document
                save_current_document()

                if current_document_id is not None:
                    closed_documents.add(
                        current_document_id
                    )

                # Check document ordering
                if document_id in closed_documents:
                    raise ValueError(
                        "Document chunks are not grouped "
                        f"by document_id: {document_id}"
                    )

                current_document_id = document_id
                current_source = chunk.get("source")

                current_sum = np.zeros(
                    EMBEDDING_DIMENSION,
                    dtype=np.float64,
                )

                current_chunk_count = 0

            # Add corresponding chunk embedding
            vector = embeddings[processed_chunks]

            current_sum += vector.astype(
                np.float64,
                copy=False,
            )

            current_chunk_count += 1
            processed_chunks += 1

    # Save last document
    save_current_document()

    metadata_file.close()

    document_embeddings.flush()

    del document_embeddings
    del embeddings

    # Final validation
    print("\nValidating document embeddings...")

    result = np.load(
        DOCUMENT_EMBEDDINGS_FILE,
        mmap_mode="r",
    )

    print(f"Shape: {result.shape}")
    print(f"Documents: {document_index:,}")
    print(f"Chunks processed: {processed_chunks:,}")

    if processed_chunks != total_chunks:
        raise ValueError("Not all chunks were processed!")

    if document_index != total_documents:
        raise ValueError("Document count mismatch!")

    if not np.isfinite(result).all():
        raise ValueError("Invalid values in embeddings!")

    sample = result[:min(100, len(result))]
    norms = np.linalg.norm(sample, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Document vectors are not normalized!")

    print("\n" + "=" * 60)
    print("DOCUMENT EMBEDDINGS COMPLETED!")
    print("=" * 60)

    print(f"Documents: {document_index:,}")
    print(f"Dimensions: {EMBEDDING_DIMENSION}")
    print(f"Output: {DOCUMENT_EMBEDDINGS_FILE}")
    print(f"Metadata: {DOCUMENT_METADATA_FILE}")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    create_document_embeddings()

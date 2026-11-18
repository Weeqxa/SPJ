
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDING_BATCH_SIZE,
    HF_CACHE_DIR,
    PROCESSED_DATA_DIR,
)



# =====================================================
# PATHS
# =====================================================

CHUNKS_FILE = PROCESSED_DATA_DIR / "chunks.jsonl"

EMBEDDINGS_FILE = PROCESSED_DATA_DIR / "embeddings.npy"

INFO_FILE = PROCESSED_DATA_DIR / "embedding_info.json"



# =====================================================
# MODEL LOADING
# =====================================================


def load_embedding_model():
    """
    Завантажує embedding-модель на NVIDIA GPU,
    якщо CUDA доступна.
    """

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nUsing device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        cache_folder=str(HF_CACHE_DIR / "models"),
        device=device,
    )

    print("Embedding model loaded successfully!")
    print(f"Device: {model.device}")
    print(f"Embedding dimension: {model.get_embedding_dimension()}")
    print(f"Maximum sequence length: {model.max_seq_length}")

    return model



# =====================================================
# COUNT CHUNKS
# =====================================================

def count_chunks(input_path):
    """
    Підраховує кількість chunks у JSONL-файлі.
    """

    count = 0

    with open(input_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                count += 1

    return count


# =====================================================
# LOAD CHUNKS BATCH
# =====================================================

def load_chunks_batch(file, batch_size):
    """
    Зчитує одну порцію chunks із JSONL.
    """

    batch = []

    for line in file:
        if not line.strip():
            continue

        batch.append(json.loads(line))

        if len(batch) >= batch_size:
            break

    return batch


# =====================================================
# CREATE EMBEDDINGS
# =====================================================

def create_embeddings(
    model,
    input_path,
    output_path,
    batch_size=EMBEDDING_BATCH_SIZE,
):
    """
    Створює embeddings для всіх chunks.

    Використовує batch processing та NumPy memmap,
    щоб не зберігати всі вектори в оперативній пам'яті.

    Для multilingual E5:
    - passage: для документів;
    - query: для пошукових запитів.
    """

    print("\nStarting embedding generation...")

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {input_path}"
        )

    total_chunks = count_chunks(input_path)

    if total_chunks == 0:
        raise ValueError("No chunks found!")

    print(f"Total chunks: {total_chunks:,}")
    print(f"Batch size: {batch_size}")

    # Створюємо файл для векторів
    embeddings = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_chunks, EMBEDDING_DIMENSION),
    )

    processed = 0

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        total_batches = (
            total_chunks + batch_size - 1
        ) // batch_size

        with tqdm(
            total=total_batches,
            desc="Creating embeddings",
        ) as progress:

            while True:

                batch = load_chunks_batch(
                    file,
                    batch_size
                )

                if not batch:
                    break

                # Додаємо passage: для E5
                texts = [
                    "passage: " + item["text"]
                    for item in batch
                ]

                # Перевіряємо кількість токенів до генерації embeddings.
                tokenized = model.tokenizer(
                    texts,
                    add_special_tokens=True,
                    truncation=False,
                )

                token_lengths = [
                    len(input_ids)
                    for input_ids in tokenized["input_ids"]
                ]

                max_tokens = model.max_seq_length

                if any(length > max_tokens for length in token_lengths):
                    raise ValueError(
                        f"Chunk exceeds embedding model limit "
                        f"of {max_tokens} tokens. "
                        f"Maximum found: {max(token_lengths)}"
                    )

                vectors = model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )

                vectors = np.asarray(
                    vectors,
                    dtype=np.float32
                )

                if vectors.shape != (
                    len(batch),
                    EMBEDDING_DIMENSION
                ):
                    raise ValueError(
                        f"Unexpected embedding shape: {vectors.shape}"
                    )

                embeddings[
                    processed:processed + len(batch)
                ] = vectors

                processed += len(batch)

                progress.update(1)

    embeddings.flush()

    del embeddings

    print("\nEmbedding generation completed!")

    print(f"Processed chunks: {processed:,}")
    print(f"Embedding shape: ({processed}, {EMBEDDING_DIMENSION})")
    print(f"Saved to: {output_path}")

    return processed


# =====================================================
# VALIDATE EMBEDDINGS
# =====================================================

def validate_embeddings(embeddings_path):
    """
    Перевіряє створені embeddings:
    - форму масиву;
    - тип даних;
    - відсутність NaN та Inf;
    - нормалізацію векторів.
    """

    print("\nValidating embeddings...")

    embeddings = np.load(
        embeddings_path,
        mmap_mode="r"
    )

    print(f"Shape: {embeddings.shape}")
    print(f"Data type: {embeddings.dtype}")

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D!")

    if embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError("Incorrect embedding dimension!")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or Inf!")

    # Перевіряємо кілька векторів
    sample = embeddings[:min(100, len(embeddings))]

    norms = np.linalg.norm(sample, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError(
            "Embeddings are not normalized!"
        )

    print("Embeddings validation successful!")

    return True


# =====================================================
# SAVE EMBEDDING INFORMATION
# =====================================================

def save_embedding_info(
    total_chunks,
    embeddings_path,
):
    """
    Зберігає інформацію про embeddings.
    """

    info = {
        "model": EMBEDDING_MODEL,
        "dimension": EMBEDDING_DIMENSION,
        "total_chunks": total_chunks,
        "batch_size": EMBEDDING_BATCH_SIZE,
        "normalized": True,
        "document_prefix": "passage:",
        "query_prefix": "query:",
        "input_file": str(CHUNKS_FILE),
        "output_file": str(embeddings_path),
    }

    with open(
        INFO_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            info,
            file,
            ensure_ascii=False,
            indent=4
        )

    print(f"Embedding information saved to: {INFO_FILE}")


# =====================================================
# MAIN
# =====================================================

def main():

    print("=" * 55)
    print("RAG EMBEDDINGS")
    print("=" * 55)

    # 1. Завантажуємо модель
    model = load_embedding_model()

    # 2. Створюємо embeddings
    total_chunks = create_embeddings(
        model=model,
        input_path=CHUNKS_FILE,
        output_path=EMBEDDINGS_FILE,
    )

    # 3. Перевіряємо результат
    validate_embeddings(
        EMBEDDINGS_FILE
    )

    # 4. Зберігаємо інформацію
    save_embedding_info(
        total_chunks=total_chunks,
        embeddings_path=EMBEDDINGS_FILE,
    )

    print("\n" + "=" * 55)
    print("EMBEDDINGS PIPELINE COMPLETED!")
    print("=" * 55)


if __name__ == "__main__":
    main()

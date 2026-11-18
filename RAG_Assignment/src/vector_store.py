
import json
import heapq
import numpy as np
import pandas as pd

from pathlib import Path
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    HF_CACHE_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    TOP_K,
)



# =====================================================
# PATHS
# =====================================================

CHUNKS_FILE = PROCESSED_DATA_DIR / "chunks.jsonl"

EMBEDDINGS_FILE = PROCESSED_DATA_DIR / "embeddings.npy"

SEARCH_RESULTS_FILE = RESULTS_DIR / "retrieval_results.csv"

# Кількість векторів за одну операцію пошуку
SEARCH_BATCH_SIZE = 8192



# =====================================================
# LOAD CHUNKS
# =====================================================

def load_chunks(input_path):
    """
    Завантажує chunks у пам'ять.

    Використовується лише для невеликих тестових наборів.
    Для повної Wikipedia використовуємо потоковий пошук.
    """

    print("\nLoading chunks...")

    chunks = []

    with open(input_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"Loaded chunks: {len(chunks):,}")

    return chunks


# =====================================================
# LOAD EMBEDDINGS
# =====================================================

def load_embeddings(input_path):
    """
    Відкриває embeddings через memory mapping.

    Вектори не завантажуються повністю в RAM.
    """

    print("\nLoading embeddings with memory mapping...")

    embeddings = np.load(
        input_path,
        mmap_mode="r"
    )

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Data type: {embeddings.dtype}")

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array!")

    if embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError("Incorrect embedding dimension!")

    return embeddings


# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

def load_embedding_model():
    """
    Завантажує embedding-модель.
    """

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        cache_folder=str(HF_CACHE_DIR / "models"),
    )

    print("Embedding model loaded!")

    return model


# =====================================================
# CREATE QUERY EMBEDDING
# =====================================================

def create_query_embedding(model, query):
    """
    Створює embedding питання.

    Для E5 використовується префікс query:.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty!")

    query_text = "query: " + query.strip()

    embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    if embedding.shape != (1, EMBEDDING_DIMENSION):
        raise ValueError("Incorrect query embedding shape!")

    return embedding[0]


# =====================================================
# SEARCH IN MEMORY
# =====================================================

def search_similar_chunks(
    query,
    model,
    chunks,
    embeddings,
    top_k=TOP_K,
):
    """
    Пошук для невеликих наборів даних.

    Використовується для початкових тестів.
    """

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks does not match embeddings!"
        )

    if not chunks:
        raise ValueError("No chunks available!")

    if top_k < 1:
        raise ValueError("top_k must be at least 1!")

    query_embedding = create_query_embedding(
        model,
        query
    )

    similarities = np.dot(
        embeddings,
        query_embedding
    )

    actual_top_k = min(top_k, len(chunks))

    top_indices = np.argpartition(
        similarities,
        -actual_top_k
    )[-actual_top_k:]

    top_indices = top_indices[
        np.argsort(similarities[top_indices])[::-1]
    ]

    results = []

    for rank, index in enumerate(top_indices, start=1):

        chunk = chunks[int(index)]

        results.append({
            "rank": rank,
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "source": chunk.get("source"),
            "chunk_index": chunk["chunk_index"],
            "similarity_score": float(similarities[index]),
            "text": chunk["text"],
        })

    return results


# =====================================================
# STREAMING SEARCH
# =====================================================

def search_similar_chunks_streaming(
    query,
    model,
    chunks_path,
    embeddings_path,
    top_k=TOP_K,
    batch_size=SEARCH_BATCH_SIZE,
):
    """
    Пошук chunks для великої колекції.

    Особливості:
    - embeddings читаються через memory mapping;
    - текстові chunks читаються поступово;
    - в оперативній пам'яті зберігаються лише
      поточний batch та top_k результатів.

    Порядок записів у JSONL і embeddings повинен
    відповідати один одному.
    """

    print("\nStarting streaming vector search...")

    chunks_path = Path(chunks_path)
    embeddings_path = Path(embeddings_path)

    if not chunks_path.exists():
        raise FileNotFoundError(chunks_path)

    if not embeddings_path.exists():
        raise FileNotFoundError(embeddings_path)

    if top_k < 1:
        raise ValueError("top_k must be at least 1!")

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1!")

    # 1. Відкриваємо embeddings без повного завантаження
    embeddings = load_embeddings(embeddings_path)

    total_vectors = embeddings.shape[0]

    # 2. Створюємо embedding запиту
    query_embedding = create_query_embedding(
        model,
        query
    )

    # 3. Heap зберігатиме тільки найкращі результати
    best_results = []

    processed = 0

    print(f"Total vectors: {total_vectors:,}")
    print(f"Search batch size: {batch_size:,}")

    with open(
        chunks_path,
        "r",
        encoding="utf-8"
    ) as file:

        while processed < total_vectors:

            # Зчитуємо поточну порцію текстових chunks
            batch_chunks = []

            for _ in range(
                min(batch_size, total_vectors - processed)
            ):

                line = file.readline()

                if not line:
                    break

                if line.strip():
                    batch_chunks.append(json.loads(line))

            if not batch_chunks:
                raise ValueError(
                    "Chunks file ended before all embeddings were processed!"
                )

            # Перевірка відповідності кількості
            if len(batch_chunks) != min(
                batch_size,
                total_vectors - processed
            ):
                raise ValueError(
                    "Number of chunks does not match embeddings!"
                )

            # 4. Отримуємо відповідну частину embeddings
            batch_vectors = embeddings[
                processed:processed + len(batch_chunks)
            ]

            # 5. Обчислюємо cosine similarity
            scores = np.dot(
                batch_vectors,
                query_embedding
            )

            # 6. Відбираємо найкращі результати поточного batch
            candidate_count = min(
                top_k,
                len(batch_chunks)
            )

            candidate_indices = np.argpartition(
                scores,
                -candidate_count
            )[-candidate_count:]

            # 7. Додаємо кандидатів до загального heap
            for local_index in candidate_indices:

                score = float(scores[local_index])

                chunk = batch_chunks[int(local_index)]

                result = {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "source": chunk.get("source"),
                    "chunk_index": chunk["chunk_index"],
                    "similarity_score": score,
                    "text": chunk["text"],
                }

                heap_item = (
                    score,
                    processed + int(local_index),
                    result,
                )

                if len(best_results) < top_k:
                    heapq.heappush(
                        best_results,
                        heap_item
                    )

                elif score > best_results[0][0]:
                    heapq.heapreplace(
                        best_results,
                        heap_item
                    )

            processed += len(batch_chunks)

            print(
                f"\rProcessed: {processed:,}/{total_vectors:,}",
                end="",
                flush=True,
            )

    print("\nStreaming search completed!")

    # 8. Сортуємо результати
    best_results.sort(
        key=lambda item: item[0],
        reverse=True
    )

    results = []

    for rank, item in enumerate(best_results, start=1):

        result = item[2].copy()

        result["rank"] = rank

        results.append(result)

    return results


# =====================================================
# DISPLAY RESULTS
# =====================================================

def display_search_results(results):
    """
    Виводить знайдені chunks.
    """

    print("\n" + "=" * 70)
    print("SEARCH RESULTS")
    print("=" * 70)

    for result in results:

        print(f"\nRank: {result['rank']}")
        print(f"Similarity: {result['similarity_score']:.4f}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Document ID: {result['document_id']}")
        print(f"Source: {result['source']}")

        print("\nText:")
        print(result["text"])

        print("-" * 70)


# =====================================================
# SAVE RESULTS
# =====================================================

def save_search_results(
    query,
    results,
    output_path=SEARCH_RESULTS_FILE,
):
    """
    Зберігає результати пошуку у CSV.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    for result in results:

        row = result.copy()
        row["query"] = query

        rows.append(row)

    dataframe = pd.DataFrame(rows)

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nResults saved to: {output_path}")


# =====================================================
# MAIN
# =====================================================

def main():

    print("=" * 70)
    print("RAG VECTOR SEARCH")
    print("=" * 70)

    # 1. Завантажуємо embedding-модель
    model = load_embedding_model()

    # 2. Тестове питання
    query = "Ktoré mesto je hlavné mesto Slovenska?"

    print("\nQuestion:")
    print(query)

    # 3. Потоковий пошук
    results = search_similar_chunks_streaming(
        query=query,
        model=model,
        chunks_path=CHUNKS_FILE,
        embeddings_path=EMBEDDINGS_FILE,
        top_k=TOP_K,
    )

    # 4. Відображення
    display_search_results(results)

    # 5. Збереження
    save_search_results(
        query=query,
        results=results,
    )

    print("\n" + "=" * 70)
    print("VECTOR SEARCH COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()

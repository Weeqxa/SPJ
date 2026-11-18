
import json
import numpy as np
import pandas as pd

from config import (
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    EMBEDDING_DIMENSION,
    TOP_K,
    RESULTS_RETRIEVAL,
)

from data_loader import load_wiki_dataset
from vector_store import (
    load_embedding_model,
    create_query_embedding,
)

from llm import (
    create_ollama_client,
    check_model_availability,
    generate_rag_answer,
)

# =====================================================
# PATHS
# =====================================================

DOCUMENT_EMBEDDINGS_FILE = (
    PROCESSED_DATA_DIR / "document_embeddings.npy"
)

DOCUMENT_METADATA_FILE = (
    PROCESSED_DATA_DIR / "document_metadata.jsonl"
)


# =====================================================
# LOAD DOCUMENT EMBEDDINGS
# =====================================================

def load_document_embeddings():

    print("\nLoading document embeddings...")

    if not DOCUMENT_EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(DOCUMENT_EMBEDDINGS_FILE)

    embeddings = np.load(
        DOCUMENT_EMBEDDINGS_FILE,
        mmap_mode="r",
    )

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D!")

    if embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError("Incorrect embedding dimension!")

    print(f"Document embeddings: {embeddings.shape}")

    return embeddings


# =====================================================
# LOAD DOCUMENT METADATA
# =====================================================

def load_document_metadata():

    print("\nLoading document metadata...")

    metadata = []

    with open(
        DOCUMENT_METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if line.strip():
                metadata.append(json.loads(line))

    print(f"Documents loaded: {len(metadata):,}")

    return metadata


# =====================================================
# DOCUMENT RETRIEVAL
# =====================================================

def search_similar_documents(
    query,
    model,
    embeddings,
    metadata,
    top_k=TOP_K,
):

    print("\nSearching for relevant documents...")

    if len(embeddings) != len(metadata):
        raise ValueError(
            "Document embeddings and metadata do not match!"
        )

    if top_k < 1:
        raise ValueError("top_k must be at least 1!")

    if not query.strip():
        raise ValueError("Query cannot be empty!")

    # 1. Create query embedding
    query_embedding = create_query_embedding(
        model,
        query,
    )

    # 2. Calculate cosine similarity
    # Both vectors are L2-normalized.
    similarities = np.dot(
        embeddings,
        query_embedding,
    )

    # 3. Select top-k documents
    actual_top_k = min(
        top_k,
        len(metadata),
    )

    indices = np.argpartition(
        similarities,
        -actual_top_k,
    )[-actual_top_k:]

    indices = indices[
        np.argsort(similarities[indices])[::-1]
    ]

    # 4. Prepare results
    results = []

    for rank, index in enumerate(indices, start=1):

        document = metadata[int(index)]

        results.append({
            "rank": rank,
            "document_index": int(index),
            "document_id": document["document_id"],
            "source": document.get("source"),
            "chunk_count": document["chunk_count"],
            "similarity_score": float(similarities[index]),
        })

    return results


# =====================================================
# LOAD ORIGINAL DOCUMENT TEXTS
# =====================================================

def attach_document_texts(results, dataset):

    print("\nLoading original document texts...")

    required_ids = {
        str(result["document_id"])
        for result in results
    }

    found_documents = {}

    for document in dataset:

        document_id = str(document.get("id"))

        if document_id in required_ids:

            found_documents[document_id] = document.get(
                "text",
                "",
            )

        if len(found_documents) == len(required_ids):
            break

    for result in results:

        document_id = str(result["document_id"])

        result["text"] = found_documents.get(
            document_id,
            "",
        )

        if not result["text"]:
            print(
                f"WARNING: Text not found for {document_id}"
            )

    return results


# =====================================================
# DISPLAY RESULTS
# =====================================================

def display_results(results):

    print("\n" + "=" * 70)
    print("DOCUMENT RETRIEVAL RESULTS")
    print("=" * 70)

    for result in results:

        print(f"\nRank: {result['rank']}")

        print(f"Document ID: {result['document_id']}")

        print(f"Similarity: {result['similarity_score']:.4f}")

        print(f"Source: {result['source']}")

        print(f"Chunks in document: {result['chunk_count']}")

        print("\nDocument text:")

        print(result["text"][:2000])

        print("\n" + "-" * 70)



# =====================================================
# DOCUMENT RETRIEVAL ANSWER GENERATION
# =====================================================

def generate_document_answer(
    query,
    results,
    ollama_client,
):
    """
    Generates an answer using retrieved full documents.
    """

    print("\nPreparing document context...")

    context_parts = []

    remaining_chars = 12000

    for result in results:

        text = result.get("text", "")

        if not text:
            continue

        text = text[:remaining_chars]

        if not text:
            break

        context_parts.append(
            f"""
Document ID: {result['document_id']}
Similarity: {result['similarity_score']:.4f}

Content:
{text}
"""
        )

        remaining_chars -= len(text)

        if remaining_chars <= 0:
            break

    if not context_parts:
        raise RuntimeError(
            "No document context available!"
        )

    context = "\n".join(context_parts)

    print("\nGenerating document retrieval answer...")

    answer = generate_rag_answer(
        question=query,
        context=context,
        client=ollama_client,
    )

    print("\nDOCUMENT RETRIEVAL ANSWER:")
    print(answer)

    return answer






# =====================================================
# SAVE RESULTS
# =====================================================

def save_results(query, results):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for result in results:

        row = result.copy()
        row["query"] = query

        rows.append(row)

    dataframe = pd.DataFrame(rows)

    dataframe.to_csv(
        RESULTS_RETRIEVAL,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nResults saved to: {RESULTS_RETRIEVAL}")


# =====================================================
# MAIN
# =====================================================

def main():

    print("=" * 70)
    print("DOCUMENT RETRIEVAL SYSTEM")
    print("=" * 70)

    # 1. Load embedding model
    model = load_embedding_model()

    # 2. Load document embeddings
    embeddings = load_document_embeddings()

    # 3. Load document metadata
    metadata = load_document_metadata()

    # 4. Load original Wikipedia dataset
    dataset = load_wiki_dataset()

    # 5. Test question
    query = "Ktoré mesto je hlavné mesto Slovenska?"

    print("\nQuestion:")
    print(query)

    # 6. Search similar documents
    results = search_similar_documents(
        query=query,
        model=model,
        embeddings=embeddings,
        metadata=metadata,
        top_k=TOP_K,
    )

    # 7. Attach original document texts
    results = attach_document_texts(
        results,
        dataset,
    )

    # 8. Display results
    display_results(results)

    # 9. Connect to Ollama
    ollama_client = create_ollama_client()

    check_model_availability(ollama_client)

    # 10. Generate answer
    answer = generate_document_answer(
        query=query,
        results=results,
        ollama_client=ollama_client,
    )

    # 11. Add answer to results
    for result in results:
        result["answer"] = answer

    # 12. Save results
    save_results(
        query,
        results,
    )

    print("\n" + "=" * 70)
    print("DOCUMENT RETRIEVAL COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()

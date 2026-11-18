
import json
from datetime import datetime

from config import (
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    TOP_K,
)

from vector_store import (
    load_embedding_model,
    search_similar_chunks_streaming,
)

from llm import (
    create_ollama_client,
    check_model_availability,
    generate_rag_answer,
)


# =====================================================
# CONFIGURATION
# =====================================================

CHUNKS_FILE = PROCESSED_DATA_DIR / "chunks.jsonl"

EMBEDDINGS_FILE = PROCESSED_DATA_DIR / "embeddings.npy"

RAG_ANSWERS_FILE = RESULTS_DIR / "rag_answers.json"

LLM_MODEL = "qwen2.5:3b"


# =====================================================
# CONTEXT PREPARATION
# =====================================================

def prepare_context(retrieved_chunks):
    """
    Формує контекст для LLM із знайдених chunks.
    """

    context_parts = []

    for chunk in retrieved_chunks:

        context_part = f"""
[Source {chunk['rank']}]
Document ID: {chunk['document_id']}
Chunk ID: {chunk['chunk_id']}
Similarity score: {chunk['similarity_score']:.4f}

Text:
{chunk['text']}
"""

        context_parts.append(context_part)

    return "\n".join(context_parts)


# =====================================================
# RETRIEVAL
# =====================================================

def retrieve_context(
    question,
    embedding_model,
    top_k=TOP_K,
):
    """
    Знаходить релевантні chunks у повній колекції.

    Використовує streaming search, тому не завантажує
    всі chunks та embeddings одночасно в RAM.
    """

    print("\nSearching full Wikipedia collection...")

    retrieved_chunks = search_similar_chunks_streaming(
        query=question,
        model=embedding_model,
        chunks_path=CHUNKS_FILE,
        embeddings_path=EMBEDDINGS_FILE,
        top_k=top_k,
    )

    print(f"\nRetrieved {len(retrieved_chunks)} chunks.")

    return retrieved_chunks


# =====================================================
# RAG PIPELINE
# =====================================================

def run_rag(
    question,
    embedding_model,
    ollama_client,
    top_k=TOP_K,
):
    """
    Повний RAG pipeline:

    1. Отримує питання.
    2. Шукає релевантні chunks.
    3. Формує контекст.
    4. Передає контекст і питання в LLM.
    5. Отримує відповідь.
    """

    print("\n" + "=" * 65)
    print("RAG PIPELINE")
    print("=" * 65)

    print("\nQuestion:")
    print(question)

    # 1. Retrieval
    retrieved_chunks = retrieve_context(
        question=question,
        embedding_model=embedding_model,
        top_k=top_k,
    )

    if not retrieved_chunks:
        raise RuntimeError(
            "No relevant chunks were retrieved!"
        )

    # 2. Context preparation
    context = prepare_context(
        retrieved_chunks
    )

    print("\nContext prepared successfully!")

    # 3. Generate answer
    print("\nGenerating RAG answer...")

    answer = generate_rag_answer(
        question=question,
        context=context,
        client=ollama_client,
    )

    print("\nRAG ANSWER:")
    print(answer)

    return {
        "question": question,
        "answer": answer,
        "context": context,
        "retrieved_chunks": retrieved_chunks,
        "model": LLM_MODEL,
        "top_k": top_k,
    }


# =====================================================
# DISPLAY SOURCES
# =====================================================

def display_sources(retrieved_chunks):
    """
    Виводить джерела, використані системою RAG.
    """

    print("\n" + "=" * 65)
    print("RETRIEVED SOURCES")
    print("=" * 65)

    for chunk in retrieved_chunks:

        print(f"\nRank: {chunk['rank']}")

        print(f"Document ID: {chunk['document_id']}")

        print(f"Chunk ID: {chunk['chunk_id']}")

        print(
            f"Similarity: {chunk['similarity_score']:.4f}"
        )

        print("\nText:")
        print(chunk["text"])

        print("-" * 65)


# =====================================================
# SAVE RAG RESULT
# =====================================================

def save_rag_result(result):
    """
    Зберігає результат RAG у JSON.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_results = []

    if RAG_ANSWERS_FILE.exists():

        with open(
            RAG_ANSWERS_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            existing_results = json.load(file)

    result_to_save = {
        "timestamp": datetime.now().isoformat(),
        "question": result["question"],
        "answer": result["answer"],
        "model": result["model"],
        "top_k": result["top_k"],
        "retrieved_chunks": result["retrieved_chunks"],
    }

    existing_results.append(result_to_save)

    with open(
        RAG_ANSWERS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            existing_results,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print(
        f"\nRAG result saved to: {RAG_ANSWERS_FILE}"
    )


# =====================================================
# MAIN
# =====================================================

def main():

    print("=" * 65)
    print("FULL WIKIPEDIA RAG SYSTEM")
    print("=" * 65)

    # 1. Load embedding model
    embedding_model = load_embedding_model()

    # 2. Connect to Ollama
    ollama_client = create_ollama_client()

    check_model_availability(
        ollama_client
    )

    # 3. Test question
    question = (
        "Ktoré mesto je hlavné mesto Slovenska?"
    )

    # 4. Run RAG
    result = run_rag(
        question=question,
        embedding_model=embedding_model,
        ollama_client=ollama_client,
        top_k=TOP_K,
    )

    # 5. Display sources
    display_sources(
        result["retrieved_chunks"]
    )

    # 6. Save result
    save_rag_result(result)

    print("\n" + "=" * 65)
    print("FULL WIKIPEDIA RAG COMPLETED!")
    print("=" * 65)


if __name__ == "__main__":
    main()

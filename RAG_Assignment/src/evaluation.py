
import json
import re
import string
import pandas as pd

from tqdm import tqdm

from config import (
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    EVALUATION_LIMIT,
    RANDOM_SEED,
    TOP_K,
)

from data_loader import (
    load_qa_dataset,
    prepare_evaluation_questions,
    load_wiki_dataset,
)

from vector_store import (
    load_embedding_model,
    search_similar_chunks_streaming,
)

from retrieval import (
    load_document_embeddings,
    load_document_metadata,
    search_similar_documents,
    attach_document_texts,
    generate_document_answer,
)

from llm import (
    create_ollama_client,
    check_model_availability,
    generate_standalone_answer,
)

from rag import run_rag


# =====================================================
# PATHS
# =====================================================

QUESTIONS_FILE = PROCESSED_DATA_DIR / "questions.jsonl"

EVALUATION_RESULTS_FILE = RESULTS_DIR / "evaluation_results.csv"

EVALUATION_SUMMARY_FILE = RESULTS_DIR / "evaluation_summary.json"

CHUNKS_FILE = PROCESSED_DATA_DIR / "chunks.jsonl"
EMBEDDINGS_FILE = PROCESSED_DATA_DIR / "embeddings.npy"


# =====================================================
# LOAD QUESTIONS
# =====================================================

def load_evaluation_questions():

    print("\nLoading evaluation questions...")

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(QUESTIONS_FILE)

    questions = []

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                questions.append(json.loads(line))

    if not questions:
        raise ValueError("No evaluation questions found!")

    questions = questions[:EVALUATION_LIMIT]

    print(f"Questions loaded: {len(questions)}")

    return questions


# =====================================================
# ANSWER NORMALIZATION
# =====================================================

def normalize_answer(answer):

    if not answer:
        return ""

    answer = answer.lower()

    answer = "".join(
        char for char in answer
        if char not in string.punctuation
    )

    answer = re.sub(r"\b(a|an|the)\b", " ", answer)

    answer = " ".join(answer.split())

    return answer


# =====================================================
# EXACT MATCH
# =====================================================

def calculate_exact_match(prediction, references):

    prediction = normalize_answer(prediction)

    if not references:
        return 0.0

    return float(
        any(
            prediction == normalize_answer(reference)
            for reference in references
        )
    )


# =====================================================
# TOKEN F1
# =====================================================

def calculate_f1(prediction, references):

    prediction_tokens = normalize_answer(
        prediction
    ).split()

    if not prediction_tokens or not references:
        return 0.0

    best_f1 = 0.0

    for reference in references:

        reference_tokens = normalize_answer(
            reference
        ).split()

        if not reference_tokens:
            continue

        common = {}

        for token in prediction_tokens:
            common[token] = common.get(token, 0) + 1

        overlap = 0

        for token in reference_tokens:

            if common.get(token, 0) > 0:
                overlap += 1
                common[token] -= 1

        if overlap == 0:
            continue

        precision = overlap / len(prediction_tokens)

        recall = overlap / len(reference_tokens)

        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

        best_f1 = max(best_f1, f1)

    return best_f1


# =====================================================
# EXTRACT REFERENCES
# =====================================================

def extract_references(answers):

    if isinstance(answers, dict):

        texts = answers.get("text", [])

        if isinstance(texts, str):
            return [texts] if texts else []

        return [
            text for text in texts
            if isinstance(text, str) and text.strip()
        ]

    if isinstance(answers, list):

        references = []

        for item in answers:

            if isinstance(item, str):
                references.append(item)

            elif isinstance(item, dict):
                references.extend(
                    extract_references(item)
                )

        return references

    if isinstance(answers, str):
        return [answers] if answers.strip() else []

    return []


# =====================================================
# EVALUATE ONE ANSWER
# =====================================================

def evaluate_answer(prediction, references):

    return {
        "exact_match": calculate_exact_match(
            prediction,
            references,
        ),
        "f1": calculate_f1(
            prediction,
            references,
        ),
    }


# =====================================================
# CHECK RAG GROUNDING
# =====================================================

def check_rag_grounding(answer, retrieved_chunks):

    normalized_answer = normalize_answer(answer)

    normalized_context = normalize_answer(
        " ".join(
            chunk["text"]
            for chunk in retrieved_chunks
        )
    )

    answer_tokens = set(normalized_answer.split())

    context_tokens = set(normalized_context.split())

    if not answer_tokens:
        return 0.0

    supported_tokens = answer_tokens.intersection(
        context_tokens
    )

    return len(supported_tokens) / len(answer_tokens)


# =====================================================
# EVALUATION
# =====================================================

def run_evaluation():

    print("=" * 70)
    print("RAG SYSTEM EVALUATION")
    print("=" * 70)

    # 1. Load questions
    questions = load_evaluation_questions()

    # 2. Load embedding model
    embedding_model = load_embedding_model()

    # 3. Load document retrieval data
    document_embeddings = load_document_embeddings()

    document_metadata = load_document_metadata()

    # 4. Load Ollama
    ollama_client = create_ollama_client()

    check_model_availability(ollama_client)

    evaluation_results = []

    # 5. Process questions
    for index, item in enumerate(
        tqdm(questions, desc="Evaluating"),
        start=1,
    ):

        question = item["question"]

        references = extract_references(
            item.get("answers")
        )

        if not references:
            print(f"\nSkipping question {index}: no references.")
            continue

        print("\n" + "=" * 70)
        print(f"QUESTION {index}/{len(questions)}")
        print(question)

        # ---------------------------------------------
        # A. STANDALONE LLM
        # ---------------------------------------------

        standalone_answer = generate_standalone_answer(
            question=question,
            client=ollama_client,
        )

        standalone_metrics = evaluate_answer(
            standalone_answer,
            references,
        )

        # ---------------------------------------------
        # B. DOCUMENT RETRIEVAL
        # ---------------------------------------------

        document_results = search_similar_documents(
            query=question,
            model=embedding_model,
            embeddings=document_embeddings,
            metadata=document_metadata,
            top_k=TOP_K,
        )

        document_results = attach_document_texts(
            document_results,
            wiki_dataset,
        )

        document_answer = generate_document_answer(
            query=question,
            results=document_results,
            ollama_client=ollama_client,
        )

        document_metrics = evaluate_answer(
            document_answer,
            references,
        )

        # ---------------------------------------------
        # C. RAG
        # ---------------------------------------------

        rag_result = run_rag(
            question=question,
            embedding_model=embedding_model,
            ollama_client=ollama_client,
            top_k=TOP_K,
        )

        rag_answer = rag_result["answer"]

        rag_metrics = evaluate_answer(
            rag_answer,
            references,
        )

        grounding_score = check_rag_grounding(
            rag_answer,
            rag_result["retrieved_chunks"],
        )

        # ---------------------------------------------
        # SAVE QUESTION RESULT
        # ---------------------------------------------

        result = {
            "question_id": item.get("id"),
            "question": question,
            "reference_answers": json.dumps(
                references,
                ensure_ascii=False,
            ),

            "standalone_answer": standalone_answer,
            "standalone_em": standalone_metrics["exact_match"],
            "standalone_f1": standalone_metrics["f1"],

            "document_answer": document_answer,
            "document_em": document_metrics["exact_match"],
            "document_f1": document_metrics["f1"],

            "rag_answer": rag_answer,
            "rag_em": rag_metrics["exact_match"],
            "rag_f1": rag_metrics["f1"],

            "rag_grounding": grounding_score,

            "rag_sources": json.dumps(
                [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "document_id": chunk["document_id"],
                        "similarity_score": chunk["similarity_score"],
                    }
                    for chunk in rag_result["retrieved_chunks"]
                ],
                ensure_ascii=False,
            ),
        }

        evaluation_results.append(result)

        print("\nMETRICS:")
        print(f"Standalone F1: {result['standalone_f1']:.4f}")
        print(f"Document F1:   {result['document_f1']:.4f}")
        print(f"RAG F1:        {result['rag_f1']:.4f}")
        print(f"RAG Grounding: {result['rag_grounding']:.4f}")

    return evaluation_results


# =====================================================
# SAVE EVALUATION
# =====================================================

def save_evaluation(results):

    if not results:
        raise ValueError("No evaluation results to save!")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(results)

    dataframe.to_csv(
        EVALUATION_RESULTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    metrics = [
        "standalone_em",
        "standalone_f1",
        "document_em",
        "document_f1",
        "rag_em",
        "rag_f1",
        "rag_grounding",
    ]

    summary = {
        "total_questions": len(dataframe),
        "model": "qwen2.5:3b",
        "top_k": TOP_K,
        "metrics": {
            metric: float(dataframe[metric].mean())
            for metric in metrics
        },
    }

    with open(
        EVALUATION_SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    for metric, value in summary["metrics"].items():
        print(f"{metric}: {value:.4f}")

    print(f"\nCSV saved: {EVALUATION_RESULTS_FILE}")
    print(f"Summary saved: {EVALUATION_SUMMARY_FILE}")


# =====================================================
# MAIN
# =====================================================

def main():

    global wiki_dataset

    wiki_dataset = load_wiki_dataset()

    results = run_evaluation()

    save_evaluation(results)

    print("\nEVALUATION COMPLETED!")


if __name__ == "__main__":
    main()


import json
from pathlib import Path

from config import (
    WIKI_DATASET_ID,
    QA_DATASET_ID,
    HF_CACHE_DIR,
    PROCESSED_DATA_DIR,
    EVALUATION_LIMIT,
    RANDOM_SEED,
)

from datasets import load_dataset


# =====================================================
# DATASET LOADING
# =====================================================

def load_wiki_dataset():
    """
    Завантажує Slovak Wikipedia dataset.

    Дані зберігаються у кеші Hugging Face на диску D.
    Повертає Dataset із записами Wikipedia.
    """

    print("\nLoading Slovak Wikipedia dataset...")

    dataset = load_dataset(
        WIKI_DATASET_ID,
        split="train",
        cache_dir=str(HF_CACHE_DIR / "datasets"),
    )

    print("Wikipedia dataset loaded successfully!")
    print(f"Number of documents: {len(dataset):,}")
    print(f"Columns: {dataset.column_names}")

    return dataset



def load_qa_dataset():
    """
    Завантажує SK-QuAD dataset.
    Використовує правильну конфігурацію skquad.
    """

    print("\nLoading SK-QuAD dataset...")

    dataset = load_dataset(
        QA_DATASET_ID,
        "skquad",
        cache_dir=str(HF_CACHE_DIR / "datasets"),
    )

    print("SK-QuAD dataset loaded successfully!")

    for split_name, split_data in dataset.items():
        print(
            f"Split '{split_name}': "
            f"{len(split_data):,} records"
        )

    return dataset



# =====================================================
# EVALUATION SPLIT SELECTION
# =====================================================

def select_evaluation_split(dataset):
    """
    Вибирає окремий split для оцінювання.

    Пріоритет:
    1. test
    2. validation
    3. dev
    4. train (якщо інших немає)
    """

    available_splits = list(dataset.keys())

    preferred_splits = [
        "test",
        "validation",
        "dev",
    ]

    for split_name in preferred_splits:
        if split_name in available_splits:
            print(
                f"\nSelected evaluation split: {split_name}"
            )
            return dataset[split_name]

    if "train" in available_splits:
        print(
            "\nWARNING: No dedicated evaluation split found."
        )
        print("Using train split for evaluation.")

        return dataset["train"]

    raise ValueError(
        f"No suitable evaluation split found. "
        f"Available splits: {available_splits}"
    )


# =====================================================
# EVALUATION DATA PREPARATION
# =====================================================

def prepare_evaluation_questions(qa_dataset):
    """
    Створює фіксовану вибірку питань.

    Вибірка використовується для порівняння:
    - RAG
    - Document Retrieval
    - Standalone LLM

    Однакові питання використовуються
    для всіх трьох підходів.
    """

    print("\nPreparing evaluation questions...")

    selected_split = select_evaluation_split(qa_dataset)

    sample_size = min(
        EVALUATION_LIMIT,
        len(selected_split)
    )

    evaluation_data = (
        selected_split
        .shuffle(seed=RANDOM_SEED)
        .select(range(sample_size))
    )

    print(
        f"Selected {len(evaluation_data)} "
        f"questions for evaluation."
    )

    return evaluation_data


# =====================================================
# DATA SAVING
# =====================================================

def save_evaluation_questions(dataset):
    """
    Зберігає питання та ground-truth відповіді
    у JSONL форматі.

    Кожен рядок файлу є окремим JSON об'єктом.
    """

    output_path = (
        PROCESSED_DATA_DIR / "questions.jsonl"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nSaving evaluation questions...")

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        for item in dataset:
            record = {
                "id": item.get("id"),
                "title": item.get("title"),
                "context": item.get("context"),
                "question": item.get("question"),
                "answers": item.get("answers"),
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    print(f"Questions saved to: {output_path}")

    return output_path


# =====================================================
# DATASET INFORMATION
# =====================================================

def save_dataset_info(wiki_dataset, qa_dataset):
    """
    Зберігає інформацію про завантажені датасети.
    """

    info = {
        "wiki": {
            "name": WIKI_DATASET_ID,
            "documents": len(wiki_dataset),
            "columns": wiki_dataset.column_names,
        },
        "qa": {
            "name": QA_DATASET_ID,
            "splits": {
                split_name: len(split_data)
                for split_name, split_data
                in qa_dataset.items()
            },
        },
    }

    output_path = (
        PROCESSED_DATA_DIR / "dataset_info.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            info,
            file,
            ensure_ascii=False,
            indent=4
        )

    print(f"Dataset information saved to: {output_path}")


# =====================================================
# MAIN
# =====================================================

def main():

    print("=" * 55)
    print("RAG DATA LOADER")
    print("=" * 55)

    # 1. Завантаження Wikipedia
    wiki_dataset = load_wiki_dataset()

    # 2. Завантаження SK-QuAD
    qa_dataset = load_qa_dataset()

    # 3. Підготовка питань для оцінювання
    evaluation_questions = prepare_evaluation_questions(
        qa_dataset
    )

    # 4. Збереження питань
    save_evaluation_questions(
        evaluation_questions
    )

    # 5. Збереження інформації про датасети
    save_dataset_info(
        wiki_dataset,
        qa_dataset
    )

    print("\n" + "=" * 55)
    print("DATA LOADING COMPLETED SUCCESSFULLY!")
    print("=" * 55)

    print(f"Wiki documents: {len(wiki_dataset):,}")
    print(f"Evaluation questions: {len(evaluation_questions):,}")

    print("\nExample question:")
    print(evaluation_questions[0]["question"])

    print("\nExample answers:")
    print(evaluation_questions[0]["answers"])


if __name__ == "__main__":
    main()


import json
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_EMBEDDING_TOKENS,
    EMBEDDING_MODEL,
    HF_CACHE_DIR,
    PROCESSED_DATA_DIR,
)

from data_loader import load_wiki_dataset


# =====================================================
# CONFIGURATION
# =====================================================

MAX_DOCUMENTS = None

OUTPUT_PATH = PROCESSED_DATA_DIR / "chunks.jsonl"

TOKENIZER_CACHE = HF_CACHE_DIR / "models"

PASSAGE_PREFIX = "passage:"


# =====================================================
# LOAD TOKENIZER
# =====================================================

def load_tokenizer():
    """
    Завантажує токенізатор тієї самої моделі,
    яка буде створювати embeddings.
    """

    print("\nLoading embedding tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL,
        cache_dir=str(TOKENIZER_CACHE),
    )

    print("Tokenizer loaded successfully!")

    return tokenizer


# =====================================================
# CHECK CHUNK LENGTH
# =====================================================

def count_embedding_tokens(tokenizer, text):
    """
    Підраховує токени разом із passage:
    та спеціальними токенами моделі.
    """

    encoded = tokenizer(
        PASSAGE_PREFIX + " " + text,
        add_special_tokens=True,
        truncation=False,
    )

    return len(encoded["input_ids"])


# =====================================================
# DOCUMENT CHUNKING
# =====================================================

def split_document_into_chunks(
    text,
    document_id,
    tokenizer,
    source=None,
):
    """
    Розбиває документ на token-based chunks.

    CHUNK_SIZE та CHUNK_OVERLAP вимірюються
    у токенах токенізатора E5.

    Кожен готовий chunk перевіряється разом
    із passage: та спеціальними токенами.
    """

    if not isinstance(text, str) or not text.strip():
        return []

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if not token_ids:
        return []

    chunks = []

    step = CHUNK_SIZE - CHUNK_OVERLAP

    if step <= 0:
        raise ValueError(
            "CHUNK_SIZE must be greater than CHUNK_OVERLAP."
        )

    start = 0
    chunk_index = 0

    while start < len(token_ids):

        end = min(
            start + CHUNK_SIZE,
            len(token_ids),
        )

        # Якщо після decode chunk перевищує ліміт,
        # поступово зменшуємо його довжину.
        while end > start:

            chunk_text = tokenizer.decode(
                token_ids[start:end],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            actual_token_count = count_embedding_tokens(
                tokenizer,
                chunk_text,
            )

            if actual_token_count <= MAX_EMBEDDING_TOKENS:
                break

            end = max(
                start + 1,
                end - 8,
            )

        if end <= start:
            raise ValueError(
                f"Cannot create valid chunk for document {document_id}"
            )

        if actual_token_count > MAX_EMBEDDING_TOKENS:
            raise ValueError(
                f"Single token sequence exceeds model limit "
                f"in document {document_id}"
            )

        chunk = {
            "chunk_id": f"{document_id}_{chunk_index:05d}",
            "document_id": str(document_id),
            "source": source,
            "chunk_index": chunk_index,
            "token_start": start,
            "token_end": end,
            "token_count": actual_token_count,
            "text": chunk_text,
        }

        chunks.append(chunk)

        chunk_index += 1

        if end == len(token_ids):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# =====================================================
# PROCESS WIKIPEDIA
# =====================================================

def process_wiki_dataset(
    dataset,
    tokenizer,
    output_path,
    max_documents=None,
):
    """
    Обробляє Wikipedia документ за документом.

    Результати записуються у JSONL поступово.
    """

    total_documents = len(dataset)

    if max_documents is not None:
        total_documents = min(
            max_documents,
            total_documents,
        )

    print("\nStarting token-based chunking...")

    print(f"Documents: {total_documents:,}")
    print(f"Chunk size: {CHUNK_SIZE} tokens")
    print(f"Chunk overlap: {CHUNK_OVERLAP} tokens")
    print(f"Maximum embedding length: {MAX_EMBEDDING_TOKENS}")

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_chunks = 0
    skipped_documents = 0

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as output_file:

        for index in tqdm(
            range(total_documents),
            total=total_documents,
            desc="Processing Wikipedia",
        ):

            document = dataset[index]

            document_id = document.get("id", index)

            source = document.get("source")

            text = document.get("text")

            chunks = split_document_into_chunks(
                text=text,
                document_id=document_id,
                source=source,
                tokenizer=tokenizer,
            )

            if not chunks:
                skipped_documents += 1
                continue

            for chunk in chunks:

                output_file.write(
                    json.dumps(
                        chunk,
                        ensure_ascii=False,
                    ) + "\n"
                )

                total_chunks += 1

    print("\nChunking completed!")

    print(f"Documents processed: {total_documents:,}")
    print(f"Chunks created: {total_chunks:,}")
    print(f"Skipped documents: {skipped_documents:,}")

    print(f"Output: {output_path}")

    return {
        "documents_processed": total_documents,
        "chunks_created": total_chunks,
        "skipped_documents": skipped_documents,
        "output_path": str(output_path),
    }


# =====================================================
# VALIDATE CHUNKS
# =====================================================

def validate_chunks(output_path, tokenizer):
    """
    Перевіряє кожен chunk за реальним
    токенізатором embedding-моделі.
    """

    print("\nValidating chunks...")

    required_fields = [
        "chunk_id",
        "document_id",
        "source",
        "chunk_index",
        "token_start",
        "token_end",
        "token_count",
        "text",
    ]

    total_checked = 0
    seen_ids = set()

    with open(
        output_path,
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            tqdm(file, desc="Validating chunks"),
            start=1,
        ):

            if not line.strip():
                continue

            chunk = json.loads(line)

            for field in required_fields:
                if field not in chunk:
                    raise ValueError(
                        f"Missing field {field} at line {line_number}"
                    )

            if chunk["chunk_id"] in seen_ids:
                raise ValueError(
                    f"Duplicate chunk ID: {chunk['chunk_id']}"
                )

            seen_ids.add(chunk["chunk_id"])

            actual_tokens = count_embedding_tokens(
                tokenizer,
                chunk["text"],
            )

            if actual_tokens != chunk["token_count"]:
                raise ValueError(
                    f"Token count mismatch at line {line_number}"
                )

            if actual_tokens > MAX_EMBEDDING_TOKENS:
                raise ValueError(
                    f"Chunk exceeds model limit at line {line_number}"
                )

            if chunk["token_end"] <= chunk["token_start"]:
                raise ValueError(
                    f"Invalid token positions at line {line_number}"
                )

            if not chunk["text"].strip():
                raise ValueError(
                    f"Empty chunk at line {line_number}"
                )

            total_checked += 1

    print("\nValidation successful!")
    print(f"Validated chunks: {total_checked:,}")

    return total_checked


# =====================================================
# MAIN
# =====================================================

def main():

    print("=" * 60)
    print("TOKEN-BASED WIKIPEDIA CHUNKING")
    print("=" * 60)

    # 1. Завантаження Wikipedia
    wiki_dataset = load_wiki_dataset()

    # 2. Завантаження токенізатора
    tokenizer = load_tokenizer()

    # 3. Створення chunks
    result = process_wiki_dataset(
        dataset=wiki_dataset,
        tokenizer=tokenizer,
        output_path=OUTPUT_PATH,
        max_documents=MAX_DOCUMENTS,
    )

    # 4. Перевірка
    validated_chunks = validate_chunks(
        OUTPUT_PATH,
        tokenizer,
    )

    # 5. Підсумок
    print("\n" + "=" * 60)
    print("TOKEN-BASED CHUNKING COMPLETED!")
    print("=" * 60)

    print(f"Documents: {result['documents_processed']:,}")
    print(f"Chunks: {result['chunks_created']:,}")
    print(f"Validated: {validated_chunks:,}")
    print(f"Skipped: {result['skipped_documents']:,}")

    print(f"Output: {result['output_path']}")


if __name__ == "__main__":
    main()

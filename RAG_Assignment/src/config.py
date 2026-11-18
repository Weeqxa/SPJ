
import os
from pathlib import Path

# =====================================================
# PROJECT PATHS
# =====================================================

# Коренева папка проєкту
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Основні папки
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"

# Hugging Face cache (disk D)
HF_CACHE_DIR = Path(r"D:\TUKE\SPJ\RAG_cache\huggingface")

# Створення необхідних папок
for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    VECTOR_DB_DIR,
    HF_CACHE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# =====================================================
# HUGGING FACE SETTINGS
# =====================================================

# Встановлюємо кеш до імпорту бібліотек Hugging Face
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault(
    "HF_DATASETS_CACHE",
    str(HF_CACHE_DIR / "datasets")
)


# =====================================================
# DATASETS
# =====================================================

WIKI_DATASET_ID = "PiotrSty/skwiki-slovak"
QA_DATASET_ID = "TUKE-DeutscheTelekom/skquad"

# Атрибути датасетів
WIKI_TEXT_COLUMN = "text"
QA_QUESTION_COLUMN = "question"
QA_ANSWERS_COLUMN = "answers"


# =====================================================
# CHUNKING SETTINGS
# =====================================================

# Розмір chunk у токенах
CHUNK_SIZE = 440

# Перекриття у токенах
CHUNK_OVERLAP = 88

# Максимальна довжина embedding-моделі
MAX_EMBEDDING_TOKENS = 512





# =====================================================
# EMBEDDING SETTINGS
# =====================================================

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# Розмір вектора embedding
EMBEDDING_DIMENSION = 384

# Кількість текстів для одного batch
EMBEDDING_BATCH_SIZE = 8


# =====================================================
# RETRIEVAL SETTINGS
# =====================================================

# Кількість найрелевантніших chunks
TOP_K = 5

# Метрика подібності
SIMILARITY_METRIC = "cosine"


# =====================================================
# LLM SETTINGS
# =====================================================

LLM_MODEL = "qwen2.5:3b"

# Ollama API
OLLAMA_HOST = "http://localhost:11434"

# Параметри генерації
LLM_TEMPERATURE = 0.1
LLM_NUM_CTX = 4096


# =====================================================
# EVALUATION SETTINGS
# =====================================================

# Кількість питань для початкового тестування
EVALUATION_LIMIT = 100

# Фіксований seed для відтворюваності вибірки
RANDOM_SEED = 42


# =====================================================
# OUTPUT SETTINGS
# =====================================================

RESULTS_RETRIEVAL = RESULTS_DIR / "retrieval_results.csv"
RESULTS_EVALUATION = RESULTS_DIR / "evaluation_results.csv"
RESULTS_ANSWERS = RESULTS_DIR / "answers.json"


if __name__ == "__main__":
    print("RAG PROJECT CONFIGURATION")
    print("=" * 45)

    print(f"Project root:       {PROJECT_ROOT}")
    print(f"Data directory:     {DATA_DIR}")
    print(f"HF cache:           {HF_CACHE_DIR}")
    print(f"Vector DB:          {VECTOR_DB_DIR}")

    print("-" * 45)

    print(f"Wiki dataset:       {WIKI_DATASET_ID}")
    print(f"QA dataset:         {QA_DATASET_ID}")

    print("-" * 45)

    print(f"Chunk size:         {CHUNK_SIZE} words")
    print(f"Chunk overlap:      {CHUNK_OVERLAP} words")
    print(f"Top K:              {TOP_K}")

    print("-" * 45)

    print(f"Embedding model:    {EMBEDDING_MODEL}")
    print(f"Embedding dimension:{EMBEDDING_DIMENSION}")
    print(f"LLM model:          {LLM_MODEL}")

    print("-" * 45)
    print("Configuration loaded successfully!")

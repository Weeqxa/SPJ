
import argparse
import subprocess
import sys
from pathlib import Path


# =====================================================
# PROJECT CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

PYTHON = sys.executable


# =====================================================
# AVAILABLE PIPELINES
# =====================================================

PIPELINES = {
    "data": "data_loader.py",
    "chunking": "chunking.py",
    "embeddings": "embeddings.py",
    "document-embeddings": "document_embeddings.py",
    "rag": "rag.py",
    "retrieval": "retrieval.py",
    "evaluation": "evaluation.py",
}


# =====================================================
# RUN PIPELINE
# =====================================================

def run_pipeline(stage):

    if stage not in PIPELINES:
        raise ValueError(f"Unknown pipeline: {stage}")

    script = SRC_DIR / PIPELINES[stage]

    if not script.exists():
        raise FileNotFoundError(
            f"Script not found: {script}"
        )

    print("\n" + "=" * 65)
    print(f"RUNNING: {stage.upper()}")
    print("=" * 65)

    subprocess.run(
        [PYTHON, str(script)],
        cwd=str(PROJECT_ROOT),
        check=True,
    )

    print(f"\n{stage.upper()} COMPLETED SUCCESSFULLY!")


# =====================================================
# DISPLAY AVAILABLE PIPELINES
# =====================================================

def display_pipelines():

    print("=" * 65)
    print("RAG ASSIGNMENT - MAIN CONTROLLER")
    print("=" * 65)

    print("\nAvailable pipelines:\n")

    for name, script in PIPELINES.items():
        print(f"  {name:<22} -> {script}")

    print("\nUsage examples:")
    print("  python main.py --stage data")
    print("  python main.py --stage chunking")
    print("  python main.py --stage embeddings")
    print("  python main.py --stage rag")
    print("  python main.py --stage retrieval")
    print("  python main.py --stage evaluation")

    print("\n" + "=" * 65)


# =====================================================
# MAIN
# =====================================================

def main():

    parser = argparse.ArgumentParser(
        description="Slovak Wikipedia RAG Assignment"
    )

    parser.add_argument(
        "--stage",
        choices=list(PIPELINES.keys()),
        help="Pipeline to execute",
    )

    args = parser.parse_args()

    if args.stage is None:
        display_pipelines()
        return

    try:
        run_pipeline(args.stage)

    except subprocess.CalledProcessError as error:
        print(
            f"\nERROR: Pipeline failed with exit code "
            f"{error.returncode}"
        )
        sys.exit(error.returncode)

    except Exception as error:
        print(f"\nERROR: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()

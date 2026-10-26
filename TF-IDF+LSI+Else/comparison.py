
import pandas as pd
from pathlib import Path

# Cesty k suborom
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Vysledky predchadzajucich experimentov
results = [
    ["TF-IDF", 0.9865, 1.0000, 0.8993, 0.9470],
    ["LSI 20", 0.9812, 0.9776, 0.8792, 0.9258],
    ["LSI 50", 0.9830, 0.9710, 0.8993, 0.9338],
    ["LSI 100", 0.9883, 0.9857, 0.9262, 0.9550],
    ["Word2Vec CBOW", 0.9740, 0.9225, 0.8792, 0.9003],
    ["Word2Vec Skip-gram", 0.9830, 0.9851, 0.8859, 0.9329],
]

columns = ["Model", "Accuracy", "Precision", "Recall", "F1"]
df = pd.DataFrame(results, columns=columns)

# Vysledky neuronovych sieti
neural_df = pd.read_csv(RESULTS_DIR / "neural_results.csv")

# Spojenie vysledkov
df = pd.concat([df, neural_df], ignore_index=True)
df = df.sort_values("F1", ascending=False)

print("\nPorovnanie vsetkych modelov:")
print(df.round(4).to_string(index=False))

best = df.iloc[0]
print("\nNajlepsi model:", best["Model"])
print("Accuracy:", round(best["Accuracy"], 4))
print("F1:", round(best["F1"], 4))

# Ulozenie tabulky
df.to_csv(RESULTS_DIR / "comparison_results.csv", index=False)
print("\nVysledky ulozene do results/comparison_results.csv")

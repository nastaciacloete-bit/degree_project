from pathlib import Path
import json
import pandas as pd


def main():
    rows = []

    vader_path = Path("results/vader_baseline_results.csv")
    if vader_path.exists():
        vader_df = pd.read_csv(vader_path)
        rows.extend(vader_df.to_dict(orient="records"))

    classical_path = Path("results/model_comparison.csv")
    if classical_path.exists():
        classical_df = pd.read_csv(classical_path)
        rows.extend(classical_df.to_dict(orient="records"))

    transformer_path = Path("results/transformer_training_metadata.json")
    if transformer_path.exists():
        with open(transformer_path, "r", encoding="utf-8") as f:
            tmeta = json.load(f)

        rows.append({
            "model": "distilbert",
            "accuracy": tmeta.get("accuracy"),
            "precision": tmeta.get("precision"),
            "recall": tmeta.get("recall"),
            "f1": tmeta.get("f1"),
            "cv_f1_mean": None,
            "cv_f1_std": None,
        })

    if not rows:
        raise ValueError("No results found to combine.")

    df = pd.DataFrame(rows)
    df.to_csv("results/final_model_comparison.csv", index=False)

    print(df.to_string(index=False))
    print("\nSaved to results/final_model_comparison.csv")


if __name__ == "__main__":
    main()
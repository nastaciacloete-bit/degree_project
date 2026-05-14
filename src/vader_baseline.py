from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from csv_dataset import load_csv_dataset, clean_text


def main():
    parser = argparse.ArgumentParser(description="Evaluate VADER baseline on a labelled dataset.")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--text-col", default=None, help="Name or index of text column")
    parser.add_argument("--label-col", default=None, help="Name or index of label column")
    parser.add_argument("--encoding", default=None, help="CSV encoding")
    parser.add_argument("--no-header", action="store_true", help="CSV has no header")
    args = parser.parse_args()

    header = None if args.no_header else "infer"

    df, chosen_text, chosen_label, mapping = load_csv_dataset(
        args.csv,
        text_col=args.text_col,
        label_col=args.label_col,
        encoding=args.encoding,
        header=header,
    )

    if "label" not in df.columns:
        raise ValueError("Dataset must include labels for evaluation.")

    analyzer = SentimentIntensityAnalyzer()

    def predict_vader(text: str) -> int:
        score = analyzer.polarity_scores(clean_text(text))["compound"]
        return 1 if score >= 0.05 else 0

    df["pred"] = df["text_clean"].apply(predict_vader)

    accuracy = accuracy_score(df["label"], df["pred"])
    precision = precision_score(df["label"], df["pred"], zero_division=0)
    recall = recall_score(df["label"], df["pred"], zero_division=0)
    f1 = f1_score(df["label"], df["pred"], zero_division=0)

    results = pd.DataFrame([{
        "model": "vader_baseline",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }])

    Path("results").mkdir(exist_ok=True)
    results.to_csv("results/vader_baseline_results.csv", index=False)

    print(results.to_string(index=False))
    print("\nSaved to results/vader_baseline_results.csv")


if __name__ == "__main__":
    main()
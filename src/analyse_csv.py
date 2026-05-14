from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from csv_dataset import clean_text


def main():
    parser = argparse.ArgumentParser(
        description="Analyse any CSV and add sentiment predictions."
    )
    parser.add_argument("--csv", required=True, help="Path to input CSV")
    parser.add_argument("--text-col", required=True, help="Name of the text column")
    parser.add_argument(
        "--output",
        default="output_with_sentiment.csv",
        help="Path to output CSV",
    )
    parser.add_argument(
        "--model-path",
        default="models/model.pkl",
        help="Path to saved model",
    )
    parser.add_argument(
        "--vectorizer-path",
        default="models/vectorizer.pkl",
        help="Path to saved vectorizer",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="Optional CSV encoding, e.g. latin-1",
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)
    vectorizer_path = Path(args.vectorizer_path)
    csv_path = Path(args.csv)
    output_path = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.resolve()}")
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path.resolve()}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    print("Loading model...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    print("Loading CSV...")
    if args.encoding:
        df = pd.read_csv(csv_path, encoding=args.encoding)
    else:
        df = pd.read_csv(csv_path)

    if args.text_col not in df.columns:
        raise ValueError(
            f"text column '{args.text_col}' not found. Available columns: {list(df.columns)}"
        )

    print("Cleaning text...")
    df["text_clean"] = df[args.text_col].astype(str).map(clean_text)

    print("Vectorizing text...")
    X = vectorizer.transform(df["text_clean"])

    print("Predicting sentiment...")
    predictions = model.predict(X)

    df["sentiment_label"] = predictions
    df["sentiment"] = df["sentiment_label"].map({0: "negative", 1: "positive"})

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
        df["positive_confidence"] = probs
        df["negative_confidence"] = 1 - probs

    print("Saving results...")
    df.to_csv(output_path, index=False)

    total = len(df)
    if total == 0:
        raise ValueError("Input CSV contains no rows.")

    positive = int((df["sentiment_label"] == 1).sum())
    negative = int((df["sentiment_label"] == 0).sum())

    positive_pct = (positive / total) * 100
    negative_pct = (negative / total) * 100

    print(f"Positive: {positive:,} ({positive_pct:.2f}%)")
    print(f"Negative: {negative:,} ({negative_pct:.2f}%)")

    plt.figure(figsize=(6, 4))
    plt.bar(["Negative", "Positive"], [negative, positive])
    plt.title("Sentiment Distribution")
    plt.ylabel("Number of Rows")
    plt.tight_layout()

    Path("results").mkdir(exist_ok=True)

    chart_name = f"{output_path.stem}_sentiment_distribution.png"
    chart_path = Path("results") / chart_name
    plt.savefig(chart_path)
    plt.close()

    print(f"Chart saved to: {chart_path}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
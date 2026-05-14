from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from csv_dataset import load_csv_dataset


def main():
    parser = argparse.ArgumentParser(description="Compare multiple sentiment models.")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--text-col", default=None, help="Name or index of text column")
    parser.add_argument("--label-col", default=None, help="Name or index of label column")
    parser.add_argument("--encoding", default=None, help="CSV encoding")
    parser.add_argument("--no-header", action="store_true", help="CSV has no header")
    parser.add_argument("--max-features", type=int, default=200000)

    args = parser.parse_args()
    header = None if args.no_header else "infer"

    print("Loading dataset...")
    df, chosen_text, chosen_label, mapping = load_csv_dataset(
        args.csv,
        text_col=args.text_col,
        label_col=args.label_col,
        encoding=args.encoding,
        header=header,
    )

    if "label" not in df.columns:
        raise ValueError("Dataset must include labels for model comparison.")

    X = df["text_clean"].values
    y = df["label"].values

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Preparing model pipelines...")

    models = {
        "logreg": Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=args.max_features,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),

        "svm": Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=args.max_features,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            )),
            ("clf", LinearSVC(class_weight="balanced")),
        ]),

        "nb": Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=args.max_features,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            )),
            ("clf", MultinomialNB()),
        ]),
    }

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, average="binary", zero_division=0)
        recall = recall_score(y_test, pred, average="binary", zero_division=0)
        f1 = f1_score(y_test, pred, average="binary", zero_division=0)

        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")

        results.append({
            "model": name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
        })

    results_df = pd.DataFrame(results).sort_values(by="cv_f1_mean", ascending=False)

    Path("results").mkdir(exist_ok=True)
    results_df.to_csv("results/model_comparison.csv", index=False)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print("\nSaved to results/model_comparison.csv")


if __name__ == "__main__":
    main()
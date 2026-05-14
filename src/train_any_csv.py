from __future__ import annotations
import argparse
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from csv_dataset import load_csv_dataset


def build_model(model_name: str):
    if model_name == "logreg":
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    if model_name == "svm":
        return LinearSVC(class_weight="balanced")
    if model_name == "nb":
        return MultinomialNB()
    raise ValueError(f"Unsupported model: {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Train TF-IDF + ML model on any CSV dataset."
    )
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--text-col", default=None, help="Name or index of the text column")
    parser.add_argument("--label-col", default=None, help="Name or index of the label column")
    parser.add_argument("--encoding", default=None, help="CSV encoding, e.g. latin-1")
    parser.add_argument("--max-features", type=int, default=200000)
    parser.add_argument("--no-header", action="store_true", help="CSV has no header row")
    parser.add_argument(
        "--model",
        default="logreg",
        choices=["logreg", "svm", "nb"],
        help="Model to train: logreg, svm or nb",
    )

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

    print(f"Loaded: {len(df):,} rows from {Path(args.csv).name}")
    print(f"Text column: {chosen_text}")

    if chosen_label is None or "label" not in df.columns:
        print("No label column detected/provided. Supervised training requires labels.")
        print("Provide --label-col to train a classifier.")
        return

    print(f"Label column: {chosen_label}")
    if mapping is not None:
        print(f"Label mapping: {mapping}")

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

    print("Building TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"Training model: {args.model}")
    model = build_model(args.model)
    model.fit(X_train_vec, y_train)

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    joblib.dump(model, "models/model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("Model saved to models/")

    print("Making predictions...")
    pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average="binary", zero_division=0)
    recall = recall_score(y_test, pred, average="binary", zero_division=0)
    f1 = f1_score(y_test, pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, digits=4)

    roc_auc = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test_vec)[:, 1]
        roc_auc = roc_auc_score(y_test, probs)

    print("\nTEST RESULTS")
    print("Model:", args.model)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    if roc_auc is not None:
        print("ROC-AUC:", roc_auc)
    print("\nConfusion matrix:\n", cm)
    print("\nClassification report:\n", report)

    print("Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=args.max_features,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )),
        ("classifier", build_model(args.model)),
    ])

    cv_scores = cross_val_score(cv_pipeline, X, y, cv=cv, scoring="f1")
    cv_f1_mean = float(cv_scores.mean())
    cv_f1_std = float(cv_scores.std())

    print(f"Cross-validation F1: {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")

    metadata = {
        "dataset": str(args.csv),
        "text_column": str(chosen_text),
        "label_column": str(chosen_label),
        "label_mapping": mapping,
        "model": args.model,
        "max_features": args.max_features,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "cv_f1_mean": cv_f1_mean,
        "cv_f1_std": cv_f1_std,
        "confusion_matrix": cm.tolist(),
    }

    with open("results/training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    with open("results/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    error_df = pd.DataFrame({
        "text": X_test,
        "actual": y_test,
        "predicted": pred
    })
    error_df = error_df[error_df["actual"] != error_df["predicted"]]
    error_df.to_csv("results/misclassified_examples.csv", index=False)

    print("Training metadata saved to results/training_metadata.json")
    print("Classification report saved to results/classification_report.txt")
    print("Misclassified examples saved to results/misclassified_examples.csv")


if __name__ == "__main__":
    main()
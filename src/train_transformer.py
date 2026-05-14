from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from csv_dataset import load_csv_dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Train a transformer sentiment classifier on a CSV dataset.")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--text-col", default=None, help="Name or index of the text column")
    parser.add_argument("--label-col", default=None, help="Name or index of the label column")
    parser.add_argument("--encoding", default=None, help="CSV encoding")
    parser.add_argument("--no-header", action="store_true", help="CSV has no header row")
    parser.add_argument("--model-name", default="distilbert-base-uncased", help="Hugging Face model name")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of rows to sample from dataset (to reduce training time)")
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
        raise ValueError("Dataset must include labels for supervised transformer training.")
    
    if args.sample_size is not None and len(df) > args.sample_size:
        df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled dataset down to {len(df):,} rows")

    print(f"Loaded {len(df):,} rows")
    print(f"Text column: {chosen_text}")
    print(f"Label column: {chosen_label}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text_clean"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    train_df = pd.DataFrame({"text": X_train.values, "label": y_train.values})
    test_df = pd.DataFrame({"text": X_test.values, "label": y_test.values})

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset = train_dataset.remove_columns(["text"])
    test_dataset = test_dataset.remove_columns(["text"])

    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    Path("models/transformer_model").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    training_args = TrainingArguments(
        output_dir="models/transformer_model/checkpoints",
        eval_strategy="epoch",
        save_strategy="no",
        logging_dir="models/transformer_model/logs",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("Training transformer...")
    trainer.train()

    print("Evaluating...")
    eval_results = trainer.evaluate()

    preds_output = trainer.predict(test_dataset)
    preds = np.argmax(preds_output.predictions, axis=1)
    labels = preds_output.label_ids

    report = classification_report(labels, preds, digits=4)
    cm = confusion_matrix(labels, preds)

    print("\nTEST RESULTS")
    print(eval_results)
    print("\nConfusion matrix:\n", cm)
    print("\nClassification report:\n", report)

    print("Saving model and tokenizer...")
    trainer.save_model("models/transformer_model")
    tokenizer.save_pretrained("models/transformer_model")

    metadata = {
        "dataset": str(args.csv),
        "text_column": str(chosen_text),
        "label_column": str(chosen_label),
        "label_mapping": mapping,
        "model": "distilbert",
        "hf_model_name": args.model_name,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accuracy": float(eval_results.get("eval_accuracy", 0.0)),
        "precision": float(eval_results.get("eval_precision", 0.0)),
        "recall": float(eval_results.get("eval_recall", 0.0)),
        "f1": float(eval_results.get("eval_f1", 0.0)),
        "confusion_matrix": cm.tolist(),
    }

    with open("results/transformer_training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    with open("results/transformer_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    error_df = pd.DataFrame({
        "text": X_test.values,
        "actual": labels,
        "predicted": preds,
    })
    error_df = error_df[error_df["actual"] != error_df["predicted"]]
    error_df.to_csv("results/transformer_misclassified_examples.csv", index=False)

    print("Saved transformer model to models/transformer_model")
    print("Saved metadata to results/transformer_training_metadata.json")


if __name__ == "__main__":
    main()
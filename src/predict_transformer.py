from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from csv_dataset import clean_text


def main():
    model_path = Path("models/transformer_model")

    if not model_path.exists():
        raise FileNotFoundError(f"Transformer model not found: {model_path.resolve()}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.eval()

    while True:
        text = input("Enter text (or 'quit'): ").strip()
        if text.lower() == "quit":
            break
        if not text:
            print("Please enter some text.")
            continue

        cleaned = clean_text(text)

        inputs = tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            pred = torch.argmax(probs).item()

        sentiment = "Positive" if pred == 1 else "Negative"
        confidence = probs[pred].item()

        print(f"{sentiment} sentiment (confidence: {confidence:.2%})")


if __name__ == "__main__":
    main()
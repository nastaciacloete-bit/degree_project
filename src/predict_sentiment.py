from pathlib import Path

import joblib

from csv_dataset import clean_text


def main():
    model_path = Path("models/model.pkl")
    vectorizer_path = Path("models/vectorizer.pkl")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.resolve()}")
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path.resolve()}")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    while True:
        text = input("Enter text (or 'quit'): ").strip()
        if text.lower() == "quit":
            break
        if not text:
            print("Please enter some text.")
            continue

        cleaned = clean_text(text)
        X = vectorizer.transform([cleaned])
        pred = model.predict(X)[0]

        if pred == 1:
            print("Positive sentiment")
        else:
            print("Negative sentiment")


if __name__ == "__main__":
    main()
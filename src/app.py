from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.csv_dataset import clean_text

st.set_page_config(page_title="📊 Social Media Sentiment Analysis", layout="wide")

st.title("Social Media Sentiment Analysis")

st.sidebar.header("About")
st.sidebar.write(
    """
This application performs sentiment analysis on social media text.

Models available:
- Logistic Regression / SVM / Naive Bayes (saved classical ML model)
- DistilBERT transformer

Features:
- TF-IDF vectorisation
- Machine learning classification
- Transformer-based prediction
- CSV upload and analysis
"""
)

st.markdown(
    """
This tool analyses social media text using machine learning and transformer models.

Upload a CSV file or enter text to predict sentiment.
"""
)

st.sidebar.header("Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose model type",
    ["Classical ML", "Transformer (DistilBERT)"]
)

classical_model = None
classical_vectorizer = None
transformer_tokenizer = None
transformer_model = None

if model_choice == "Classical ML":
    model_path = Path("models/model.pkl")
    vectorizer_path = Path("models/vectorizer.pkl")

    if not model_path.exists() or not vectorizer_path.exists():
        st.error("Classical ML model or vectorizer not found. Train the classical model first.")
        st.stop()

    classical_model = joblib.load(model_path)
    classical_vectorizer = joblib.load(vectorizer_path)

    metadata_path = Path("results/training_metadata.json")
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        st.sidebar.subheader("Current Classical Model")
        st.sidebar.write(f"Model: {metadata.get('model', 'Unknown')}")
        st.sidebar.write(f"Accuracy: {metadata.get('accuracy', 0):.4f}")
        st.sidebar.write(f"F1: {metadata.get('f1', 0):.4f}")

else:
    transformer_path = Path("models/transformer_model")

    if not transformer_path.exists():
        st.error("Transformer model not found. Train the transformer first.")
        st.stop()

    transformer_tokenizer = AutoTokenizer.from_pretrained(transformer_path)
    transformer_model = AutoModelForSequenceClassification.from_pretrained(transformer_path)
    transformer_model.eval()

    metadata_path = Path("results/transformer_training_metadata.json")
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        st.sidebar.subheader("Current Transformer Model")
        st.sidebar.write(f"Model: {metadata.get('model', 'Unknown')}")
        st.sidebar.write(f"Accuracy: {metadata.get('accuracy', 0):.4f}")
        st.sidebar.write(f"F1: {metadata.get('f1', 0):.4f}")


def predict_single_text(text: str):
    cleaned = clean_text(text)

    if model_choice == "Classical ML":
        X = classical_vectorizer.transform([cleaned])
        pred = classical_model.predict(X)[0]

        confidence = None
        if hasattr(classical_model, "predict_proba"):
            prob = classical_model.predict_proba(X)[0][1]
            confidence = prob if pred == 1 else (1 - prob)

        return pred, confidence

    inputs = transformer_tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64,
    )

    with torch.no_grad():
        outputs = transformer_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred = torch.argmax(probs).item()
        confidence = probs[pred].item()

    return pred, confidence


st.header("Single Text Prediction")

user_text = st.text_area("Enter text to analyse")

if st.button("Predict Sentiment"):
    if user_text.strip():
        pred, confidence = predict_single_text(user_text)
        sentiment = "Positive" if pred == 1 else "Negative"

        confidence_text = ""
        if confidence is not None:
            confidence_text = f" (confidence: {confidence:.2%})"

        if sentiment == "Positive":
            st.success(f"Sentiment: {sentiment}{confidence_text} 😊")
        else:
            st.error(f"Sentiment: {sentiment}{confidence_text} 😞")
    else:
        st.warning("Please enter some text.")

st.divider()

st.header("CSV Analysis")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
text_column = st.text_input("Enter the name of the text column")

if uploaded_file is not None and text_column:
    df = pd.read_csv(uploaded_file)

    if text_column not in df.columns:
        st.error(f"Column '{text_column}' not found in CSV.")
    else:
        df["text_clean"] = df[text_column].astype(str).map(clean_text)

        if model_choice == "Classical ML":
            X = classical_vectorizer.transform(df["text_clean"])
            df["sentiment_label"] = classical_model.predict(X)

            if hasattr(classical_model, "predict_proba"):
                probs = classical_model.predict_proba(X)[:, 1]
                df["positive_confidence"] = probs
                df["negative_confidence"] = 1 - probs

        else:
            predictions = []
            confidences = []

            for text in df["text_clean"].tolist():
                inputs = transformer_tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=64,
                )

                with torch.no_grad():
                    outputs = transformer_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)[0]
                    pred = torch.argmax(probs).item()

                predictions.append(pred)
                confidences.append(probs[pred].item())

            df["sentiment_label"] = predictions
            df["prediction_confidence"] = confidences

        df["sentiment"] = df["sentiment_label"].map({0: "negative", 1: "positive"})

        st.subheader("Preview")
        st.dataframe(df.head())

        st.subheader("Sentiment distribution")

        sentiment_counts = df["sentiment"].value_counts()

        total = len(df)
        positive = int((df["sentiment"] == "positive").sum())
        negative = int((df["sentiment"] == "negative").sum())
        positive_pct = (positive / total) * 100 if total > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Posts Analysed", total)
        col2.metric("Positive", positive)
        col3.metric("Negative", negative)

        st.metric("Positive %", f"{positive_pct:.2f}%")

        col1, col2 = st.columns(2)

        with col1:
            st.bar_chart(sentiment_counts)

        with col2:
            st.write("Pie chart")
            pie_fig, pie_ax = plt.subplots()
            sentiment_counts.plot.pie(autopct="%1.1f%%", ylabel="", ax=pie_ax)
            st.pyplot(pie_fig)
            plt.close(pie_fig)

        csv = df.to_csv(index=False).encode("utf-8")

        st.subheader("Word Clouds")

        positive_text = " ".join(df[df["sentiment"] == "positive"]["text_clean"])
        negative_text = " ".join(df[df["sentiment"] == "negative"]["text_clean"])

        col1, col2 = st.columns(2)

        with col1:
            st.write("Positive Words")
            if positive_text.strip():
                wc = WordCloud(width=800, height=400, background_color="white").generate(positive_text)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No positive words available for word cloud.")

        with col2:
            st.write("Negative Words")
            if negative_text.strip():
                wc = WordCloud(width=800, height=400, background_color="white").generate(negative_text)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No negative words available for word cloud.")

        st.download_button(
            "Download analysed CSV",
            csv,
            "analysed_results.csv",
            "text/csv",
        )
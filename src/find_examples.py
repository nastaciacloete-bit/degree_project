import pandas as pd
import joblib
from csv_dataset import clean_text 

model = joblib.load('../models/model.pkl')
vectorizer = joblib.load('../models/vectorizer.pkl')

df = pd.read_csv('../results/misclassified_examples.csv')

df = pd.read_csv('../results/misclassified_examples.csv')

df = df.dropna(subset=['text']) 
# --------------------------------------

X_vec = vectorizer.transform(df['text'])

X_vec = vectorizer.transform(df['text'])
probs = model.predict_proba(X_vec)

df['prob_positive'] = probs[:, 1]

subjective_cases = df[(df['prob_positive'] > 0.48) & (df['prob_positive'] < 0.52)]

print("--- SUBJECTIVE / BORDERLINE EXAMPLES ---")
print(subjective_cases[['text', 'prob_positive']].head(10))


negation_words = ["not", "can't", "don't", "didn't", "never"]
negation_errors = df[
    (df['text'].str.contains('|'.join(negation_words), case=False)) & 
    (df['actual'] == 1) & 
    (df['predicted'] == 0)
]

print("\n--- NEGATION FAILURES ---")
print(negation_errors[['text', 'prob_positive']].head(10))
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Create results folder
Path("results").mkdir(exist_ok=True)

# Logistic Regression confusion matrix
cm_logreg = np.array([
    [130179, 29821],
    [26667, 133333]
])

# DistilBERT confusion matrix
cm_bert = np.array([
    [393, 106],
    [94, 407]
])

labels = ["Negative", "Positive"]

# ----- Logistic Regression Heatmap -----
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_logreg,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()

plt.savefig("results/logreg_confusion_matrix.png")
plt.close()

# ----- DistilBERT Heatmap -----
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_bert,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("DistilBERT Confusion Matrix")
plt.tight_layout()

plt.savefig("results/distilbert_confusion_matrix.png")
plt.close()

print("Confusion matrices saved to results/")
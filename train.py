import pandas as pd
import numpy as np
import os
import sys
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from preprocessing import preprocess
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)
DATA_PATH = os.path.join("data", "spam.csv")
if not os.path.exists(DATA_PATH):
    print("ERROR: data/spam.csv not found. Run setup_dataset.py first.")
    sys.exit(1)
df = pd.read_csv(DATA_PATH, encoding="latin-1")
df = df[["label", "message"]].dropna()
df["label"] = df["label"].str.strip().str.lower()
df = df[df["label"].isin(["spam", "ham"])].reset_index(drop=True)
print(f"Dataset loaded: {len(df)} messages")
print(f"Spam: {(df['label']=='spam').sum()}  |  Ham: {(df['label']=='ham').sum()}")
plt.figure(figsize=(6, 4))
ax = sns.countplot(x="label", data=df, palette={"ham": "#4caf50", "spam": "#f44336"})
ax.set_title("Spam vs Ham Distribution")
ax.set_xlabel("Label")
ax.set_ylabel("Count")
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
plt.tight_layout()
plt.savefig("outputs/class_distribution.png")
plt.close()
print("\nPreprocessing text...")
df["clean"] = df["message"].apply(preprocess)
X = df["clean"]
y = (df["label"] == "spam").astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Support Vector Machine": SVC(kernel="linear", probability=True, random_state=42)
}
results = {}
print("\n" + "=" * 52)
print("TRAINING MODELS")
print("=" * 52)
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred) * 100
    rec = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    results[name] = {"model": model, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "y_pred": y_pred}
print("\n" + "=" * 52)
print("MODEL EVALUATION")
print("=" * 52)
for name, r in results.items():
    print(f"\n{name}")
    print(f"  Accuracy  : {r['accuracy']:.2f}%")
    print(f"  Precision : {r['precision']:.2f}%")
    print(f"  Recall    : {r['recall']:.2f}%")
    print(f"  F1 Score  : {r['f1']:.2f}%")
    print(f"\n  Classification Report:\n{classification_report(y_test, r['y_pred'], target_names=['Ham','Spam'])}")
    print("-" * 52)
best_name = max(results, key=lambda n: (results[n]["precision"], results[n]["f1"], results[n]["accuracy"]))
best = results[best_name]
print("\n" + "=" * 52)
print("BEST MODEL")
print("=" * 52)
print(f"\n  Best Model : {best_name}")
print(f"  Accuracy   : {best['accuracy']:.2f}%")
print(f"  Precision  : {best['precision']:.2f}%")
print(f"  Recall     : {best['recall']:.2f}%")
print(f"  F1 Score   : {best['f1']:.2f}%")
cm = confusion_matrix(y_test, best["y_pred"])
print(f"\n  Confusion Matrix ({best_name}):")
print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
print(f"\n  False Positives (Ham → Spam): {cm[0,1]}")
print(f"  False Negatives (Spam → Ham): {cm[1,0]}")
joblib.dump(best["model"], "models/best_model.pkl")
print(f"\n  Model saved: models/best_model.pkl")
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], ax=ax)
ax.set_title(f"Confusion Matrix — {best_name}")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()
model_names = list(results.keys())
metrics = ["accuracy", "precision", "recall", "f1"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
x = np.arange(len(model_names))
width = 0.2
fig, ax = plt.subplots(figsize=(10, 5))
for i, (m, ml) in enumerate(zip(metrics, metric_labels)):
    vals = [results[n][m] for n in model_names]
    ax.bar(x + i * width, vals, width, label=ml)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_names)
ax.set_ylim(0, 110)
ax.set_ylabel("Score (%)")
ax.set_title("Model Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.close()
feature_names = vectorizer.get_feature_names_out()
best_model = best["model"]
if hasattr(best_model, "feature_log_prob_"):
    log_prob = best_model.feature_log_prob_[1]
    top_idx = np.argsort(log_prob)[-20:]
    top_words = [feature_names[i] for i in top_idx]
    top_scores = log_prob[top_idx]
elif hasattr(best_model, "coef_"):
    coef = best_model.coef_[0]
    top_idx = np.argsort(coef)[-20:]
    top_words = [feature_names[i] for i in top_idx]
    top_scores = coef[top_idx]
else:
    top_words = []
    top_scores = []
if top_words:
    print("\n  Top Spam-Indicating Keywords:")
    for w, s in zip(reversed(top_words), reversed(top_scores)):
        print(f"    {w:20s}  score: {s:.4f}")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_words, top_scores, color="#f44336")
    ax.set_title("Top Spam-Indicating Keywords")
    ax.set_xlabel("Feature Score")
    plt.tight_layout()
    plt.savefig("outputs/spam_keywords.png")
    plt.close()
print("\n  Visualizations saved to outputs/")
print("\n" + "=" * 52)
print("Training complete.")
print("=" * 52)

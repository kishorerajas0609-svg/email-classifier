# Email Spam Classifier

A Machine Learning + NLP project that automatically classifies emails/messages as **SPAM** or **HAM (legitimate)**.

---

## Project Description

Spam emails flood inboxes with irrelevant or potentially malicious content. This project uses supervised Machine Learning and Natural Language Processing (NLP) to automatically distinguish spam from legitimate messages. It is trained on a labeled dataset and compares three ML algorithms to find the best classifier with **high accuracy** and a **low false-positive rate**.

A **false positive** occurs when a legitimate (Ham) email is incorrectly marked as Spam. Reducing false positives is critical so that real emails are never lost.

---

## Features

- Spam vs Ham detection
- Full NLP text preprocessing pipeline
- TF-IDF feature extraction
- Three ML algorithms compared automatically
- Best model selected by precision and F1-score
- Real-time classification via terminal
- Top spam keyword visualization (learned from data)
- Confusion matrix with false positive / false negative counts
- Model and vectorizer saved for reuse

---

## Technologies

- Python
- scikit-learn
- pandas
- NumPy
- NLTK
- Matplotlib
- Seaborn
- joblib

---

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| Naive Bayes | Fast probabilistic classifier; works well on text |
| Logistic Regression | Linear model; strong baseline for NLP |
| Support Vector Machine (SVM) | Finds optimal decision boundary; often best on text |

---

## Dataset

Uses the **UCI SMS Spam Collection Dataset** (or any compatible labeled dataset).

Format — tab-separated values (TSV):
```
ham    Hey, are you free tomorrow?
spam   Congratulations! You have won a prize. Call now!
```

Place your file as `dataset.txt` in the project root, then run `setup_dataset.py`.

---

## Project Structure

```
Email-Spam-Classifier/
├── data/
│   └── spam.csv
├── outputs/
│   ├── class_distribution.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   └── spam_keywords.png
├── models/
│   ├── best_model.pkl
│   └── tfidf_vectorizer.pkl
├── dataset.txt           ← Your dataset goes here
├── setup_dataset.py
├── preprocessing.py
├── train.py
├── predict.py
├── main.py
├── requirements.txt
└── README.md
```

---

## NLP Pipeline

```
Raw Message
    ↓
Lowercase
    ↓
Remove punctuation / special characters
    ↓
Tokenization
    ↓
Stop-word removal (NLTK)
    ↓
Stemming (PorterStemmer)
    ↓
TF-IDF Feature Extraction (1-gram + 2-gram)
    ↓
ML Model (Naive Bayes / Logistic Regression / SVM)
    ↓
SPAM / HAM
```

---

## How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Add your dataset
Place `dataset.txt` (tab-separated: label + message) in the project root.

### Step 3 — Convert dataset
```bash
python setup_dataset.py
```

### Step 4 — Run the full pipeline
```bash
python main.py
```

This will:
1. Load and clean the dataset
2. Preprocess text with NLP
3. Extract TF-IDF features
4. Train Naive Bayes, Logistic Regression, SVM
5. Evaluate and compare all three
6. Select and save the best model
7. Generate all visualizations in `outputs/`
8. Start the real-time classification terminal

### Predict only (after training)
```bash
python predict.py
```

---

## Results

*(Fill in after running `main.py`)*

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Naive Bayes | | | | |
| Logistic Regression | | | | |
| Support Vector Machine | | | | |

**Best Model:** ___

---

## Future Scope

- Phishing email detection
- Scam and fraud detection
- Advertisement / promotional email filtering
- Integration with existing email systems (Gmail API, Outlook)
- Multi-language spam detection

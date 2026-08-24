import os
import sys
import joblib
from preprocessing import preprocess
MODEL_PATH = "models/best_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    print("ERROR: Model not found. Run 'python main.py' first to train.")
    sys.exit(1)
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
def predict_message(text):
    cleaned  = preprocess(text)
    features = vectorizer.transform([cleaned])
    pred     = model.predict(features)[0]
    proba    = model.predict_proba(features)[0]
    label    = "SPAM" if pred == 1 else "HAM"
    return label, max(proba) * 100
print("\n" + "=" * 52)
print("EMAIL SPAM CLASSIFIER")
print("=" * 52)
while True:
    print("\nEnter an email/message (or 'exit' to quit):")
    try:
        text = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break
    if text.lower() in ("exit", "quit"):
        print("Goodbye!")
        break
    if not text:
        print("Please enter a message.")
        continue
    label, confidence = predict_message(text)
    print("\n" + "=" * 52)
    print("PREDICTION")
    print("=" * 52)
    print(f"  Prediction : {label}")
    print(f"  Confidence : {confidence:.2f}%")
    print("=" * 52)
    try:
        again = input("\nTest another message? (y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break
    if again != "y":
        print("Goodbye!")
        break

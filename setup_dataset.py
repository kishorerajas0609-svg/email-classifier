import pandas as pd
import os
import sys
src = "dataset.txt"
dst = os.path.join("data", "spam.csv")
os.makedirs("data", exist_ok=True)
if not os.path.exists(src):
    print(f"ERROR: '{src}' not found.")
    print("Place your dataset.txt file in the project root folder.")
    sys.exit(1)
try:
    df = pd.read_csv(src, sep="\t", header=None, names=["label", "message"], encoding="latin-1")
except Exception as e:
    print(f"ERROR reading dataset.txt: {e}")
    sys.exit(1)
df = df[["label", "message"]].dropna()
df["label"] = df["label"].str.strip().str.lower()
valid_labels = {"spam", "ham"}
found_labels = set(df["label"].unique())
if not found_labels.issubset(valid_labels):
    print("ERROR: Labels must be 'spam' or 'ham'.")
    print("Found:", found_labels)
    sys.exit(1)
df.to_csv(dst, index=False)
print(f"Dataset saved to: {dst}")
print(f"Total messages   : {len(df)}")
print(f"Spam             : {(df['label'] == 'spam').sum()}")
print(f"Ham              : {(df['label'] == 'ham').sum()}")

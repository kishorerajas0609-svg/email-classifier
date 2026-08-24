import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 1]
    return " ".join(tokens)

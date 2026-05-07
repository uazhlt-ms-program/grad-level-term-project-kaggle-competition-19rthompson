from typing import Iterator, Iterable, List, Tuple, Text, Union
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pandas as pd



df = pd.read_csv("data/train.csv")
df["LABEL"] = df["LABEL"].astype(int)

df["TEXT"] = df["TEXT"].fillna("")
df["LABEL"] = df["LABEL"].astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    df["TEXT"],
    df["LABEL"],
    test_size=0.2,
    random_state=42,
    stratify=df["LABEL"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        min_df=2,             # ignore super rare words
        max_df=0.9            # ignore super common words
    )),
    ("clf", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

preds = model.predict(X_val)

print(classification_report(y_val, preds))
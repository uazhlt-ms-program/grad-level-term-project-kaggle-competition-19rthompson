from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
import pandas as pd



train_df = pd.read_csv("data/train.csv")
train_df["TEXT"] = train_df["TEXT"].fillna("").astype(str)
train_df["LABEL"] = train_df["LABEL"].astype(int)

test_df = pd.read_csv("data/test.csv")
test_df["TEXT"] = test_df["TEXT"].fillna("").astype(str)


X = train_df["TEXT"]
y = train_df["LABEL"]

model = Pipeline([
    ("features", FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range = (1,2),
            min_df=2,
            max_df=0.9
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3,5),
            min_df=2
        ))
    ])),
    ("clf", LinearSVC())
])

model.fit(X, y)

preds = model.predict(test_df["TEXT"])

submission = pd.DataFrame({
    "id": test_df["ID"],
    "label": preds
})

submission.to_csv("data/submission.csv", index=False)
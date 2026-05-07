import sys
import ast
import argparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
import pandas as pd


def logistic(ngrams = (1,2)):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=ngrams,   # unigrams + bigrams
            min_df=2,             # ignore super rare words
            max_df=0.9            # ignore super common words
        )),
        ("clf", LogisticRegression(max_iter=1000))
    ])

def svc(ngrams = (1,2)):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=ngrams,
            min_df=2,
            max_df=0.9
        )),
        ("clf", LinearSVC())
    ])

def union(ngrams = (1,2)):
    return Pipeline([
    ("features", FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range = ngrams,
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


def classify(my_args):
    df = pd.read_csv("data/train.csv")

    df["TEXT"] = df["TEXT"].fillna("")
    df["LABEL"] = df["LABEL"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        df["TEXT"],
        df["LABEL"],
        test_size=0.2,
        random_state=42,
        stratify=df["LABEL"]
    )
    ngrams = my_args.ngram_range
    if my_args.classifier_type == "logistic":
        model = logistic(ngrams)
    elif my_args.classifier_type == "svc":
        model = svc(ngrams)
    elif my_args.classifier_type == "union":
        model = union(ngrams)

    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    print(classification_report(y_val, preds))


def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Image Classification with CNN')
    parser.add_argument('action', default='classify',
                        choices=[ "classify" ], 
                        nargs='?', help="desired action")

    parser.add_argument('--classifier-type',  '-c', default="logistic",     type=str,   help="which type of classifyer to use (default=linear)", choices = ["svc", "linear","union"])
    parser.add_argument('--ngram-range', '-n', default = "(1,2)", type = str, help = "enter a tuple ex'(1,2)' for ngram range" )
    my_args = parser.parse_args(argv[1:])
    my_args.ngram_range = ast.literal_eval(my_args.ngram_range)
    return my_args

def main(argv):
    my_args = parse_args(argv)
    classify(my_args)



if __name__ == "__main__":
    main(sys.argv)
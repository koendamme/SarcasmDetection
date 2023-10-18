import json
import contractions
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd


def preprocess(comment):
    comment = comment.lower()
    comment = contractions.fix(comment)
    result = ""
    for char in comment:
        if char not in string.punctuation:
            result += char

    return result


if __name__ == '__main__':
    df = pd.read_csv("data/train-balanced-sarcasm.csv")
    df = df.dropna()
    X_train = df.comment.to_numpy()
    y_train = df.label.to_numpy()

    data = []
    stop_words = set(stopwords.words('english'))

    d = {"data": [], "label": []}

    for comment, label in zip(X_train, y_train):
        temp = []
        comment = preprocess(comment)

        for token in word_tokenize(comment):
            if token not in stop_words:
                temp.append(token)
        d["data"].append(temp)
        d["label"].append(int(label))

    with open("data.json", "w") as f:
        json.dump(d, f)
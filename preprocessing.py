import json
import contractions
import string

import pandas
import sklearn.model_selection
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
    X_train = df.comment.to_numpy()[:5000]
    y_train = df.label.to_numpy()[:5000]

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

    d = pandas.DataFrame.from_dict(d)

    train, val = sklearn.model_selection.train_test_split(d, train_size=0.9)
    print(train.head)
    print()
    print(val.head)
    d_train = {"data": train['data'].tolist(), "label": train['label'].tolist()}
    d_val = {"data": val.data.tolist(), "label": val.label.tolist()}

    with open("train_data.json", "w") as f:
        json.dump(d_train, f)

    with open("val_data.json", "w") as f:
        json.dump(d_val, f)
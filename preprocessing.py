import json
import contractions
import string

import pandas
import sklearn.model_selection
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
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
    df = pd.read_csv("data2/train-balanced.csv", sep='\t',
                     names=['label', 'comment', 'author', 'subreddit', 'score', 'ups', 'downs', 'date', 'created_utc', 'parent_comment'])
    df = df.dropna()
    X_train = df.comment.to_numpy()[:5000]
    y_train = df.label.to_numpy()[:5000]

    data = []
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    d = {"data": [], "label": []}

    for comment, label in zip(X_train, y_train):
        temp = []
        comment = preprocess(comment)

        for token in word_tokenize(comment):
            if token not in stop_words:
                stemmed_token = ps.stem(token)
                temp.append(stemmed_token)
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

    # df = pd.read_csv("data2/test-balanced.csv", sep='\t',
    #                  names=['label', 'comment', 'author', 'subreddit', 'score', 'ups', 'downs', 'date', 'created_utc', 'parent_comment'])
    # # df = df.dropna()
    # # X_test = df['comment'].to_numpy()
    # # y_test = df['label'].to_numpy()
    #
    # sarc_df = df[df['label'] == 1][:round(df.shape[0]*.05)]
    # non_sarc_df = df[df['label'] == 0]
    # print(len(sarc_df), len(non_sarc_df))
    #
    # new_df = pd.concat([sarc_df, non_sarc_df])
    # new_df = new_df.dropna()
    #
    # X_test = new_df['comment'].to_numpy()
    # y_test = new_df['label'].to_numpy()
    #
    # data = []
    # stop_words = set(stopwords.words('english'))
    #
    # d_test = {"data": [], "label": []}
    #
    # for comment, label in zip(X_test, y_test):
    #     temp = []
    #     comment = preprocess(comment)
    #
    #     for token in word_tokenize(comment):
    #         if token not in stop_words:
    #             temp.append(token)
    #     d_test["data"].append(temp)
    #     d_test["label"].append(int(label))
    #
    # with open("unbalanced_test_data.json", "w") as f:
    #     json.dump(d_test, f)
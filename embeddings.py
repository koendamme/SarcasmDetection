from gensim.models import Word2Vec
import json


def word2vec(data_path, save_dir, embedding_size=100):
    with open(data_path, 'r') as f:
        d = json.load(f)

    w2v = Word2Vec(sentences=d["data"], min_count=1, vector_size=embedding_size)
    w2v.save(f"{save_dir}/word2vec.model")


if __name__ == '__main__':
    word2vec("data.json", "word2vec")

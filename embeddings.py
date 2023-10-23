from gensim.models import Word2Vec
import json


def word2vec(data_path, save_dir, embedding_size=100):
    with open(data_path, 'r') as f:
        d = json.load(f)

    w2v = Word2Vec(sentences=d["data"], vector_size=embedding_size)
    w2v.save(save_dir)


if __name__ == '__main__':
    word2vec("train_data.json", "word2vec/word2vec_small.model", 50)

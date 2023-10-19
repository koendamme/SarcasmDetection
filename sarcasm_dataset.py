import json
from torch.utils.data import Dataset
from gensim.models import Word2Vec
import torch
import numpy as np


class SarcasmDataset(Dataset):
    def __init__(self, data_file, embedding_model_path, pool_sequence=True):
        self.pool_sequence = pool_sequence
        with open(data_file, "r") as f:
            self.data = json.load(f)

        self.w2v = Word2Vec.load(embedding_model_path)

    def __len__(self):
        return len(self.data["data"])

    def __getitem__(self, idx):
        sentence, label = self.data["data"][idx], self.data["label"][idx]

        x = np.zeros((len(sentence), self.w2v.vector_size))

        for i, token in enumerate(sentence):
            x[i, :] = self.w2v.wv[token]

        if self.pool_sequence:
            x = np.sum(x, axis=0)

        return torch.Tensor(x), torch.Tensor([label])

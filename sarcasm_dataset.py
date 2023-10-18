import json
from torch.utils.data import Dataset
from gensim.models import Word2Vec
import torch


class SarcasmDataset(Dataset):
    def __init__(self, data_file, embedding_model_path):
        with open(data_file, "r") as f:
            self.data = json.load(f)

        self.w2v = Word2Vec.load(embedding_model_path)

    def __len__(self):
        return len(self.data["data"])

    def __getitem__(self, idx):
        sentence, label = self.data["data"][idx], self.data["label"][idx]

        pooled = torch.zeros(self.w2v.vector_size)

        for word in sentence:
            pooled += self.w2v.wv[word]

        return pooled, torch.Tensor([label])

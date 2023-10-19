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

        x = torch.zeros((len(sentence), self.w2v.vector_size))

        return x, torch.Tensor([label])

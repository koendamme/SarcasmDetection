from sarcasm_dataset import SarcasmDataset
from torch.utils.data import DataLoader
from util import custom_collate_fn
from gru import BiDirGRU
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded device: {device}")

    dataset_test = SarcasmDataset("unbalanced_test_data.json", "word2vec/word2vec_100.model", pool_sequence=False)
    print(np.sum(dataset_test.data["label"])/len(dataset_test))

    loader_test = DataLoader(dataset_test, batch_size=16, collate_fn=custom_collate_fn)

    model = BiDirGRU(embedding_size=100)
    model.load_state_dict(torch.load("models/frompc/epoch18.pt", map_location=device))
    model.eval()

    bce = nn.BCELoss()

    preds = []
    for i, (x_batch, y_batch) in enumerate(tqdm(loader_test)):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        output = model(x_batch)

        pred = torch.round(output)

        # preds[i:i+len(x_batch)] = preds
        preds.extend(pred.detach().numpy())

    print(accuracy_score(preds, dataset_test.data["label"]))
    print(recall_score(preds, dataset_test.data["label"]))
    print(precision_score(preds, dataset_test.data["label"]))
    print(f1_score(preds, dataset_test.data["label"]))

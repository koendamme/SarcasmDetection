import pandas as pd
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sarcasm_dataset import SarcasmDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from simple_nn import NeuralNetwork
from gru import GRU
nltk.download('stopwords')
nltk.download('punkt')


def custom_collate_fn(data):
    max_sequence_length = 20

    x_batch = np.zeros((len(data), max_sequence_length, data[0][0].shape[1]))
    y_batch = np.zeros((len(data), 1))

    for i in range(len(data)):
        idx = min(data[i][0].shape[0], max_sequence_length)
        x_batch[i, :idx, :] = data[i][0][:idx, :]
        y_batch[i, 0] = data[i][1]

    return torch.Tensor(x_batch), torch.Tensor(y_batch)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded device: {device}")

    dataset = SarcasmDataset("data.json", "word2vec/word2vec.model")
    loader = DataLoader(dataset, collate_fn=custom_collate_fn, batch_size=16)

    n_epochs = 10
    embedding_size = 100
    # model = NeuralNetwork(embedding_size)
    model = GRU(embedding_size)
    model = model.to(device)
    bce = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    prev_params = list(model.parameters())[0].clone()

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        step = 0

        print(f"Epoch {str(epoch + 1)}/{str(n_epochs)}. Training...")
        for x_batch, y_batch in tqdm(loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(x_batch)
            loss = bce(output, y_batch)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            step += 1

        train_loss /= step
        print(train_loss)


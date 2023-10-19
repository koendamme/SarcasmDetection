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


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded device: {device}")

    dataset = SarcasmDataset("data.json", "word2vec/word2vec.model")
    loader = DataLoader(dataset, batch_size=16)

    n_epochs = 10
    embedding_size = 100
    model = NeuralNetwork(embedding_size)
    # model = GRU(embedding_size)
    model = model.to(device)
    bce = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        step = 0

        print("Training...")
        for x_batch, y_batch in tqdm(loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            print(x_batch.shape)

            output = model(x_batch)

            output = torch.round(output)
            loss = bce(torch.round(output), y_batch)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            step += 1
        print(loss)

        train_loss /= step
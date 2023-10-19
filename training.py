import pandas as pd
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchsummary as torchsummary

from sarcasm_dataset import SarcasmDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from simple_nn import NeuralNetwork
nltk.download('stopwords')
nltk.download('punkt')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded device: {device}")

    dataset = SarcasmDataset("train_data.json", "word2vec/word2vec.model")
    dataset_val = SarcasmDataset("val_data.json", "word2vec/word2vec.model")
    loader = DataLoader(dataset, batch_size=16)
    loader_val = DataLoader(dataset_val, batch_size=16)

    val_losses = []

    n_epochs = 10
    embedding_size = 100
    model = NeuralNetwork(embedding_size)
    model = model.to(device)
    # loss = F.binary_cross_entropy
    bce = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)

    for epoch in range(n_epochs):
        model.train(True)
        train_loss = 0
        step = 0

        print("Training...")
        for x_batch, y_batch in tqdm(loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()

            output = model(x_batch)
            output = torch.round(output)
            loss = bce(torch.round(output), y_batch)

            # Backpropagation

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            step += 1

        train_loss /= step
        # print(output)
        # model.eval()
        # val_loss = 0
        # step = 0
        # print("Validating...")
        # for x_batch, y_batch in tqdm(loader_val):
        #     x_batch = x_batch.to(device)
        #     y_batch = y_batch.to(device)
        #
        #     output = model(x_batch)
        #
        #     output = torch.round(output)
        #     loss = F.binary_cross_entropy(torch.round(output), y_batch)
        #     val_loss += loss.item()
        #     step += 1
        # val_losses.append(val_loss / step)
        # print("validation loss:", val_loss/step)
        print("training loss:", train_loss)




# print(val_losses)
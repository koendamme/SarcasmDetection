import copy

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

    n_epochs = 10
    embedding_size = 100
    val_losses = np.zeros(n_epochs)
    train_losses = np.zeros(n_epochs)
    models = []

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
            loss = bce(output, y_batch)

            # Backpropagation
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            step += 1

        train_loss /= step
        train_losses[epoch] = train_loss

        model.eval()
        val_loss = 0
        step = 0
        print("Validating...")
        try:
            for x_batch, y_batch in tqdm(loader_val):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                output = model(x_batch)

                # output = torch.round(output)
                loss = F.binary_cross_entropy(output, y_batch)
                val_loss += loss.item()
                step += 1
        except KeyError:
            print("Key not present.")
            continue
        val_losses[epoch] = val_loss / step
        models.append(copy.deepcopy(model))

    opt_epoch = np.argmin(val_losses)
    opt_model = models[opt_epoch]

    print("validation loss: ", val_losses)
    print("training loss: ", train_losses)

    dataset_test = SarcasmDataset("test_data.json", "word2vec/word2vec.model")
    loader_test = DataLoader(dataset_test, batch_size=16)

    opt_model.eval()
    test_loss = 0
    step = 0
    print("Testing...")
    for x_batch, y_batch in tqdm(loader_val):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        output = model(x_batch)

        # output = torch.round(output)
        loss = F.binary_cross_entropy(output, y_batch)
        test_loss += loss.item()
        step += 1
    test_loss = test_loss / step

    print(test_loss)



# print(val_losses)
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
from torch.utils.tensorboard import SummaryWriter
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

    dataset = SarcasmDataset("train_data.json", "word2vec/word2vec.model", pool_sequence=False)
    dataset_val = SarcasmDataset("val_data.json", "word2vec/word2vec.model", pool_sequence=False)

    # Remove collate_fn when training neural network
    loader = DataLoader(dataset, batch_size=16, collate_fn=custom_collate_fn)
    loader_val = DataLoader(dataset_val, batch_size=16, collate_fn=custom_collate_fn)

    # For writing to tensorboard. run in terminal:
    # pip install tensorboard
    # tensorboard --logdir=runs
    writer = SummaryWriter()

    val_losses = []

    n_epochs = 100
    embedding_size = 100
    # model = NeuralNetwork(embedding_size)
    model = GRU(embedding_size)
    model = model.to(device)

    bce = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        step = 0

        print(f"Epoch {str(epoch + 1)}/{str(n_epochs)}. Training...")
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

        writer.add_scalar("Loss/train", train_loss/step, epoch)
        train_loss /= step

        model.eval()
        val_loss = 0
        step = 0
        print("Validating...")
        for x_batch, y_batch in tqdm(loader_val):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(x_batch)

            loss = bce(output, y_batch)
            val_loss += loss.item()
            step += 1
        val_losses.append(val_loss / step)
        writer.add_scalar("Loss/val", val_loss/step, epoch)

    writer.flush()
    writer.close()



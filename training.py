import copy
import pandas as pd
import nltk
import torch
import torch.nn as nn
import numpy as np
from sarcasm_dataset import SarcasmDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from simple_nn import NeuralNetwork
from torch.utils.tensorboard import SummaryWriter
from gru import GRU, BiDirGRU
import os
from datetime import datetime
from util import custom_collate_fn
# nltk.download('stopwords')
# nltk.download('punkt')


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded device: {device}")

    dataset = SarcasmDataset("train_data.json", "word2vec/word2vec_small.model", pool_sequence=False)
    dataset_val = SarcasmDataset("val_data.json", "word2vec/word2vec_small.model", pool_sequence=False)

    # Remove collate_fn when training neural network
    loader = DataLoader(dataset, batch_size=16, collate_fn=custom_collate_fn)
    loader_val = DataLoader(dataset_val, batch_size=16, collate_fn=custom_collate_fn)

    # For writing to tensorboard. run in terminal:
    # pip install tensorboard
    # tensorboard --logdir=runs
    writer = SummaryWriter()

    n_epochs = 20
    embedding_size = 100
    # model = NeuralNetwork(embedding_size)
    # model = GRU(embedding_size)
    model = BiDirGRU(embedding_size)
    model = model.to(device)
    models = []

    bce = nn.BCELoss()
    lr = .01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    date_str = datetime.now().strftime('%d-%b_%H-%M')
    print(date_str)

    dir_name = f"models/{model.__class__.__name__}+{bce.__class__.__name__}+{str(lr)}+{optimizer.__class__.__name__}+{date_str}"
    os.mkdir(dir_name)

    val_losses = np.zeros(n_epochs)

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

        writer.add_scalar("Loss/val", val_loss/step, epoch)
        torch.save(model.state_dict(), f"{dir_name}/epoch{epoch+1}.pt")
        val_losses[epoch] = val_loss / step
        models.append(copy.deepcopy(model))

    writer.flush()
    writer.close()

    opt_epoch = np.argmin(val_losses)
    with open(f"{dir_name}/opt_model.txt" "w") as f:
        f.write(str(opt_epoch + 1))

if __name__ == '__main__':
    train()

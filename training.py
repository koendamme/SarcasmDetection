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
from util import custom_collate_fn, EarlyStopper
from torch.optim.lr_scheduler import LinearLR
# nltk.download('stopwords')
# nltk.download('punkt')


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded device: {device}")

    dataset = SarcasmDataset("train_data.json", "word2vec/word2vec_100.model", pool_sequence=True)
    dataset_val = SarcasmDataset("val_data.json", "word2vec/word2vec_100.model", pool_sequence=True)

    # Remove collate_fn when training neural network
    loader = DataLoader(dataset, batch_size=512, num_workers=8, pin_memory=True, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=512, num_workers=4, pin_memory=True, shuffle=True)

    # For writing to tensorboard. run in terminal:
    # pip install tensorboard
    # tensorboard --logdir=runs
    writer = SummaryWriter()

    n_epochs = 100
    embedding_size = 100
    model = NeuralNetwork(embedding_size)
    # model = GRU(embedding_size)
    # model = BiDirGRU(embedding_size)
    model = model.to(device)

    bce = nn.BCELoss()
    lr = .001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=.1, total_iters=10)

    date_str = datetime.now().strftime('%d-%b_%H-%M')
    print(date_str)

    dir_name = f"models/{model.__class__.__name__}+{bce.__class__.__name__}+{str(lr)}+{optimizer.__class__.__name__}+{date_str}"
    os.mkdir(dir_name)

    val_losses = []

    early_stopper = EarlyStopper(patience=5, min_delta=0.01)

    for epoch in range(n_epochs):
        model.train()
        running_train_loss = 0
        n_corrects = 0
        step = 0

        print(f"Epoch {str(epoch + 1)}/{str(n_epochs)}. Training...")
        for x_batch, y_batch in tqdm(loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()

            output = model(x_batch)
            predictions = torch.round(output)

            n_corrects += torch.sum(predictions == y_batch)

            loss = bce(output, y_batch)

            # Backpropagation
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            step += 1
        train_loss = running_train_loss/step
        train_accuracy = n_corrects/len(dataset)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)

        model.eval()
        running_val_loss = 0
        n_corrects = 0
        step = 0
        print("Validating...")
        for x_batch, y_batch in tqdm(loader_val):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(x_batch)
            predictions = torch.round(output)

            n_corrects += torch.sum(predictions == y_batch)

            loss = bce(output, y_batch)

            running_val_loss += loss.item()
            step += 1

        val_loss = running_val_loss/step
        val_losses.append(val_loss)
        val_accuracy = n_corrects/len(dataset_val)
        scheduler.step()
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)

        torch.save(model.state_dict(), f"{dir_name}/epoch{epoch+1}.pt")

        if early_stopper.early_stop(val_loss):
            print(f"Early stopping at epoch {epoch + 1}!")
            break

    writer.flush()
    writer.close()

    opt_epoch = np.argmin(val_losses)
    with open(f"{dir_name}/opt_model.txt", "w") as f:
        f.write(str(opt_epoch + 1))


if __name__ == '__main__':
    train()

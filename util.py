import numpy as np
import torch


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def custom_collate_fn(data):
    max_sequence_length = 20

    x_batch = np.zeros((len(data), max_sequence_length, data[0][0].shape[1]))
    y_batch = np.zeros((len(data), 1))

    for i in range(len(data)):
        idx = min(data[i][0].shape[0], max_sequence_length)
        x_batch[i, :idx, :] = data[i][0][:idx, :]
        y_batch[i, 0] = data[i][1]

    return torch.Tensor(x_batch), torch.Tensor(y_batch)
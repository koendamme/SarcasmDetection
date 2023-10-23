import numpy as np
import torch


def custom_collate_fn(data):
    max_sequence_length = 20

    x_batch = np.zeros((len(data), max_sequence_length, data[0][0].shape[1]))
    y_batch = np.zeros((len(data), 1))

    for i in range(len(data)):
        idx = min(data[i][0].shape[0], max_sequence_length)
        x_batch[i, :idx, :] = data[i][0][:idx, :]
        y_batch[i, 0] = data[i][1]

    return torch.Tensor(x_batch), torch.Tensor(y_batch)
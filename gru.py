import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GRU(nn.Module):
    def __init__(self, embedding_size):
        super(GRU, self).__init__()

        self.gru = nn.GRU(input_size=embedding_size, hidden_size=100)
        self.fc = nn.Linear(100, 1)

    def forward(self, x):
        _, x = self.gru(x)
        x = self.fc(x)
        o = torch.sigmoid(x)
        return o


if __name__ == '__main__':
    sentence = ["Hello", "this", "is", "a", "test"]

    sequence = np.zeros((len(sentence), 100))

    for i, token in enumerate(sequence):
        sequence[i, :] = np.random.rand(1, 100)

    sequence = torch.Tensor(sequence)
    batch = torch.stack([sequence, sequence, sequence], dim=1)
    print(batch.shape)
    gru = GRU(100)

    o = gru(batch)
    print(o)


    # print(batch.shape)
    # gru = nn.GRU(input_size=100, hidden_size=100, batch_first=True)
    #
    # output, h_n = gru(batch)
    #
    # print(output.shape, h_n.shape)
    #
    # print(output[0, -1, :] == h_n[0, 0, :])





import torch
import torch.nn as nn
import numpy as np


class GRU(nn.Module):
    def __init__(self, embedding_size):
        super(GRU, self).__init__()

        self.gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=100,
            batch_first=True,
            bidirectional=True,
            num_layers=4)

        self.fc = nn.Linear(100, 1)

    def forward(self, x):
        _, x = self.gru(x)
        x = self.fc(x[0])
        o = torch.sigmoid(x)
        return o


if __name__ == '__main__':
    sentence = ["Hello", "this", "is", "a", "test"]

    sequence = np.random.rand(len(sentence), 100)

    padded_seq = np.concatenate((sequence, np.zeros((2, 100))), axis=0)
    # print(padded_seq.shape)

    # sequence = torch.Tensor(sequence)
    batch = np.stack([padded_seq, padded_seq, padded_seq], axis=0)
    batch = torch.Tensor(batch)
    print(batch.shape)
    gru = nn.GRU(
            input_size=100,
            hidden_size=100,
            batch_first=True,
            bidirectional=False,
            num_layers=1)

    o, h = gru(torch.Tensor(batch))
    print(o.shape, h.shape)

    # print(o2)


    # print(batch.shape)
    # gru = nn.GRU(input_size=100, hidden_size=100, batch_first=True)
    #
    # output, h_n = gru(batch)
    #
    # print(output.shape, h_n.shape)
    #
    # print(output[0, -1, :] == h_n[0, 0, :])





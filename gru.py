import torch
import torch.nn as nn
import numpy as np


class GRU(nn.Module):
    def __init__(self, embedding_size):
        super(GRU, self).__init__()

        self.gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=300,
            batch_first=True,
            bidirectional=False,
            num_layers=1)

        self.fc = nn.Linear(300, 1)

    def forward(self, x):
        _, x = self.gru(x)
        x = self.fc(x[0])
        o = torch.sigmoid(x)
        return o


class BiDirGRU(nn.Module):
    def __init__(self, embedding_size):
        super(BiDirGRU, self).__init__()

        self.gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=100,
            batch_first=True,
            bidirectional=True,
            dropout=.1,
            num_layers=1
        )

        self.fc1 = nn.Linear(200, 500)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(500, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, x):
        _, x = self.gru(x)
        x = torch.flatten(torch.permute(x, (1, 0, 2)), start_dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
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
            hidden_size=300,
            batch_first=True,
            bidirectional=True,
            num_layers=1)

    o, h = gru(torch.Tensor(batch))

    flattened = torch.flatten(torch.permute(h, (1, 0, 2)), start_dim=1)
    print(o.shape, h.shape, flattened.shape)

    # print(o2)


    # print(batch.shape)
    # gru = nn.GRU(input_size=100, hidden_size=100, batch_first=True)
    #
    # output, h_n = gru(batch)
    #
    # print(output.shape, h_n.shape)
    #
    # print(output[0, -1, :] == h_n[0, 0, :])





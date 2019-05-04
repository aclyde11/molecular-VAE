import numpy as np
from sklearn import model_selection
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F



class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class MolecularVAE(nn.Module):
    def __init__(self, max_len, vocab_size, h_size = 501, latent_size=292):
        super(MolecularVAE, self).__init__()

        self.max_len = max_len
        self.vocab_size = vocab_size
        print("VOCAB SIZE " , vocab_size)

        self.linear = TimeDistributed(nn.Linear(h_size, vocab_size))

        self.conv1d1 = nn.Conv1d(vocab_size, 9, kernel_size=9)
        self.conv1d2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv1d3 = nn.Conv1d(9, 10, kernel_size=9)
        # self.conv1d4 = nn.Conv1d(10, 10, kernel_size=9)
        # self.conv1d5 = nn.Conv1d(10, 10, kernel_size=9)

        self.fc0 = nn.Linear(680, 500)
        self.fc11 = nn.Linear(500, latent_size)
        self.fc12 = nn.Linear(500, latent_size)

        self.fc2 = nn.Linear(latent_size, latent_size)
        self.gru = nn.GRU(latent_size, h_size, 3, batch_first=True)
        self.relu = nn.ReLU()
        self.selu = nn.SELU()

    def encode(self, x):
        bs = x.shape
        h = self.relu(self.conv1d1(x.permute(0,2,1)))
        h = self.relu(self.conv1d2(h))
        h = self.relu(self.conv1d3(h))
        # h = self.relu(self.conv1d4(h))
        # h = self.relu(self.conv1d5(h))

        h = h.view(bs[0], -1)
        h = self.selu(self.fc0(h))
        return self.fc11(h), self.fc12(h)

    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = 1e-2 * torch.randn_like(std)
            w = eps.mul(std).add_(mu)
            return w
        else:
            return mu

    def decode(self, z):
        z = self.selu(self.fc2(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, self.max_len, 1)
        out, _ = self.gru(z)
        out = self.linear(out)
        return out

    def forward(self, x):
        x = self.embedding(x)
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

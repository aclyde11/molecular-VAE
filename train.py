import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.functional as F
from data_loader import MoleLoader
from models import MolecularVAE
import pandas as pd
import pickle


def loss_function(recon_x, x, mu, logvar):
    BCE = torch.cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD



df = pd.read_csv("/vol/ml/aclyde/ZINC/zinc_cleaned.smi", header=None)
msk = np.random.rand(len(df)) < 0.8
df_train = df[~msk]
df_test = df[~msk]

max_len = 255
vocab = pickle.load(open("outvocab.pkl", 'wb'))

train_dataset = MoleLoader(df_train, vocab, max_len)
test_dataset  = MoleLoader(df_test, vocab, max_len)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=250, shuffle=True, num_workers = 64, pin_memory = True)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=250, shuffle=True, num_workers =  64, pin_memory = True)
torch.manual_seed(42)

epochs = 100

model = MolecularVAE().cuda()
model = nn.DataParallel(model)
optimizer = optim.Adam(model.parameters())

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)

        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'{epoch} / {batch_idx}\t{loss:.4f}')
    print('train', train_loss / len(train_loader.dataset))
    return train_loss / len(train_loader.dataset)

def test(epoch):
    model.eval()
    test_loss = 0
    for batch_idx, data in enumerate(test_loader):
        data = data.cuda()
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).item()
    print('test', test_loss / len(test_loader))
    return test_loss

for epoch in range(1, epochs + 1):
    train_loss = train(epoch)
    test(epoch)

import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.functional as F
from data_loader import MoleLoader
from models import MolecularVAE
import pandas as pd
import pickle
from tqdm import tqdm

def onehot_initialization_v2(a):
    ncols = len(vocab)
    out = np.zeros( (a.size,ncols), dtype=np.uint8)
    out[np.arange(a.size),a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out

def loss_function(recon_x, x, mu, logvar):
    recon_x = recon_x.contiguous().view(-1)
    x = x.contiguous().view(-1)
    bce = nn.BCELoss(size_average=True)
    xent_loss = max_len * bce(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return xent_loss + KLD



df = pd.read_csv("/vol/ml/aclyde/ZINC/zinc_cleaned.smi", header=None)
df = df.iloc[0:1000000,:]
print(df.head())
print(df.shape)
max_len = 0


vocab = set()
for i in tqdm(df.itertuples(index=False)):
    for c in i[0]:
        vocab.add(c)
    max_len = max(max_len, len(i[0]))
vocab.add(' ')


vocab = {c : i for i, c in enumerate(list(vocab))}
charset = {i : c for i, c in enumerate(list(vocab))}
print(vocab)
msk = np.random.rand(len(df)) < 0.8
df_train = df[~msk]
df_test = df[~msk]

max_len += 2
lossf = nn.CrossEntropyLoss()
train_dataset = MoleLoader(df_train, vocab, max_len)
test_dataset  = MoleLoader(df_test, vocab, max_len)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers = 32, pin_memory = True)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers =  32, pin_memory = True)
torch.manual_seed(42)

epochs = 1000

model = MolecularVAE(i=max_len, c=len(vocab)).cuda()
#model = nn.DataParallel(model)
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=15, verbose=True, cooldown=10)
log_interval = 100


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (_, ohe) in enumerate(train_loader):
        ohe = ohe.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(ohe)

        loss = loss_function(recon_batch, ohe, mu, logvar)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)

        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'train: {epoch} / {batch_idx}\t{loss:.4f}')
    print('train', train_loss / len(train_loader.dataset))
    return train_loss / len(train_loader.dataset)

def test(epoch):
    model.eval()
    test_loss = 0
    for batch_idx, (data, ohe) in enumerate(test_loader):
        data = data.cuda()
        ohe = ohe.cuda()

        recon_batch, mu, logvar = model(ohe)
        test_loss += loss_function(recon_batch, ohe, mu, logvar).item()

        if batch_idx % log_interval == 0:
            _, preds = torch.max(recon_batch, dim=2)
            preds = preds.cpu().numpy()
            targets_copy = data.cpu().numpy()
            for i in range(4):
                sample = preds[i, ...]
                target = targets_copy[i, ...]
                print("ORIG: {}\nNEW : {}".format(
                    "".join([charset[chars] for chars in target]),
                    "".join([charset[chars] for chars in sample])
                ))

    print('test', test_loss / len(test_loader))
    return test_loss

for epoch in range(1, epochs + 1):
    train_loss = train(epoch)
    val_loss = test(epoch)
    scheduler.step(val_loss)
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    torch.save( { 'model_state_dict' : model.state_dict(),
                  'optimizer_state_dict' : optimizer.state_dict(),
                  'epoch' : epoch,
                  'charset' : charset,
                  'max_len' : max_len,
                  'lr'      : lr
    }, "save.pt")

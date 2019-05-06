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
from rdkit import Chem

from comet_ml import Experiment

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
    kl_loss = -0.5 * torch.mean(1. + mu - logvar ** 2. -
                                torch.exp(mu))
    return xent_loss + kl_loss



df = pd.read_csv("/vol/ml/aclyde/ZINC/zinc_cleaned.smi", header=None)
df = df.iloc[0:1000000,:]
max_len = 0


vocab = set()
bads = []
for i in tqdm(df.itertuples(index=True)):
    try:
        i = str(Chem.MolToSmiles(Chem.MolFromSmiles(i[1]), True))
        for c in i:
            vocab.add(c)
        max_len = max(max_len, len(i))
    except:
        bads.append(i[0])
vocab.add(' ')

df = df.drop(bads, axis=0)
print(df.head())
print(df.shape)

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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers = 32, pin_memory = True)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=True, num_workers =  32, pin_memory = True)
torch.manual_seed(42)

epochs = 3000

model = MolecularVAE(i=max_len, c=len(vocab)).cuda()
#model = nn.DataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
log_interval = 100

experirment = Experiment(project_name='pytorch', auto_metric_logging=False)

def train(epoch):
    with experirment.train():
        model.train()
        train_loss = 0
        for batch_idx, (data, ohe) in enumerate(train_loader):
            data = data.cuda()
            ohe = ohe.cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)

            loss = loss_function(recon_batch, ohe, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
            experirment.log_metric('loss', loss.item())
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(f'train: {epoch} / {batch_idx}\t{loss:.4f}')
        print('train', train_loss / len(train_loader.dataset))
        return train_loss / len(train_loader.dataset)

def test(epoch):
    with experirment.test():
        model.eval()
        test_loss = 0
        for batch_idx, (data, ohe) in enumerate(test_loader):
            data = data.cuda()
            ohe = ohe.cuda()

            recon_batch, mu, logvar = model(data)
            loss =  loss_function(recon_batch, ohe, mu, logvar)
            test_loss += loss.item()
            experirment.log_metric('loss', loss.item())
            num_right = 0
            _, preds = torch.max(recon_batch, dim=2)

            for i in range(recon_batch.shape[0]):
                num_right += int(torch.eq(preds[i,...], data[i,...]))

            experirment.log_metric('accuracy', float(num_right)/float(recon_batch.shape[0]))

            if batch_idx % log_interval == 0:
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
    lr = 0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    torch.save( { 'model_state_dict' : model.state_dict(),
                  'optimizer_state_dict' : optimizer.state_dict(),
                  'epoch' : epoch,
                  'charset' : charset,
                  'max_len' : max_len,
                  'lr'      : lr
    }, "save.pt")

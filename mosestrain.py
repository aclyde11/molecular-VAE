from comet_ml import Experiment


import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch import nn, optim
from tqdm import tqdm

from data_loader import MoleLoader
from models import MolecularVAE
import argparse
from config import get_parser
from vocab import OneHotVocab
from mosesfile import VAE

epochs=100
df = pd.read_csv("/vol/ml/aclyde/ZINC/zinc_cleaned_cannon.smi", header=None)
max_len = 128
smiles = []
for i in df.itertuples(index=False):
    if len(i < max_len):
        smiles.append(i)

vocab = OneHotVocab.from_data(smiles)
experiment = Experiment(project_name='pytorch')


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                           num_workers=8 * 6,
                                           pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8 * 6,
                                          pin_memory=True)

model = VAE(vocab, get_parser())
optimizer = optim.Adam((p for p in model.vae.parameters() if p.requires_grad), lr=0.0003)


def train():
    return

def test():
    return

def sample():
    return


for epoch in range(1, epochs + 1):

    experirment.log_current_epoch(epoch)
    train_loss = train(epoch)
    val_loss = test(epoch)
    print(val_loss)
    scheduler.step(val_loss)
    lr = 0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    experirment.log_metric('lr', lr)
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'charset': charset,
                'max_len': max_len,
                'lr': lr,
                'latent_size': args.latent_size
                }, "save_" + str(args.batch_size) + "_" + str(args.optimizer) + "_" + str(args.latent_size) + ".pt")


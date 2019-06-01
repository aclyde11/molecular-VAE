
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
from torch.nn.utils import clip_grad_norm_
import pickle
import math
import mosesvae
import mosesvocab
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.preprocessing import MinMaxScaler
import random
import os
import argparse



run_bindings = True

class KLAnnealer:
    def __init__(self, n_epoch):
        self.i_start = 0
        self.w_start = 0
        self.w_max = 1
        self.n_epoch = n_epoch

        self.inc = (self.w_max - self.w_start) / (self.n_epoch - self.i_start)

    def __call__(self, i):
        k = (i - self.i_start) if i >= self.i_start else 0
        return self.w_start + k * self.inc


class CosineAnnealingLRWithRestart(_LRScheduler):
    def __init__(self, optimizer):
        self.n_period = 10
        self.n_mult = 1
        self.lr_end = 1e-4

        self.current_epoch = 0
        self.t_end = self.n_period

        # Also calls first epoch
        super().__init__(optimizer, -1)

    def get_lr(self):
        return [self.lr_end + (base_lr - self.lr_end) *
                (1 + math.cos(math.pi * self.current_epoch / self.t_end)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        if self.current_epoch == self.t_end:
            self.current_epoch = 0
            self.t_end = self.n_mult * self.t_end

def string2tensor(vocab, string):
    ids = vocab.string2ids(string, add_bos=True, add_eos=True)
    tensor = torch.tensor(
        ids, dtype=torch.long
    )

    return tensor


def tensor2string(vocab, tensor):
    ids = tensor.tolist()
    string = vocab.ids2string(ids, rem_bos=True, rem_eos=True)

    return string

def get_collate_fn_binding():
    def collate(data):
        strs, bindings = list(zip(*data))

        lens = np.array(list(map(lambda x : len(x), strs)))
        ordering = np.argsort(lens)[::-1]
        sorted_strs = [strs[i] for i in ordering]
        bindings = [bindings[i] for i in ordering]
        data.sort(key=len, reverse=True)
        tensors = [string2tensor(vocab, string)
                   for string in sorted_strs]

        for i in bindings:
            if i is None:
                print("NONE~!")
                exit()

        return tensors, torch.from_numpy(np.array(bindings)).float()

    return collate

def get_collate_fn():
    def collate(data):
        data.sort(key=len, reverse=True)
        tensors = [string2tensor(vocab, string)
                   for string in data]

        return tensors

    return collate



class BindingDataSet(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df):
        self.df = df


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        biinding = self.df.iloc[idx,0]
        smile = self.df.iloc[idx, 1]
        if biinding is None:
            print("PANIC NONE")
            exit()
        return smile, biinding


df = pd.read_csv("/workspace/zinc_subset_docking_scores.smi", header=None)
bindings = pd.read_table("/workspace/hybrid_score.txt", skiprows=1, header=None)
bindings.iloc[:, 0] = list(map(lambda x : int(x.split('_')[1]), list(bindings.iloc[:, 0])))
mmss = MinMaxScaler()
bindings.iloc[:, 1] = mmss.fit_transform( -1.0 * np.array((bindings.iloc[:, 1].astype(np.float32))).reshape(-1, 1))
bindings = bindings.set_index(bindings.columns[0])
bindings = bindings[[1]].join(df, how='left', lsuffix='hybrid')

print(bindings.head())

#df = df.sample(5000000, replace=False)
max_len = 0
print(df.head())
print(df.shape)
df = df.iloc[:,0].astype(str).tolist()

# vocab = mosesvocab.OneHotVocab.from_data(bindings.iloc[:,1].astype(str).tolist())
with open("vocab.pkl", 'rb') as f:
    vocab = pickle.load(f)
bdata = BindingDataSet(bindings)

n_epochs = 50

model = mosesvae.VAE(vocab)
pt = torch.load("trained_save.pt", map_location='cpu')
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in pt.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.eval()

def _train_epoch_binding(model, epoch, tqdm_data, kl_weight, optimizer=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    kl_loss_values = mosesvocab.CircularBuffer(10)
    recon_loss_values = mosesvocab.CircularBuffer(10)
    loss_values =mosesvocab.CircularBuffer(10)
    binding_loss_values = mosesvocab.CircularBuffer(10)
    for i, (input_batch, binding) in enumerate(tqdm_data):
        input_batch = tuple(data.cuda() for data in input_batch)
        binding = binding.cuda().view(-1, 1)
        # Forwardd
        kl_loss, recon_loss, binding_loss = model(input_batch, binding)


        kl_loss = torch.sum(kl_loss, 0)
        recon_loss = torch.sum(recon_loss, 0)
        binding_loss = torch.sum(binding_loss, 0)

        loss_weight = 0
        if epoch < 5:
            loss_weight = 0
        else:
            loss_weight = kl_weight

        loss = kl_weight * kl_loss + recon_loss + kl_weight * binding_loss


        # Backward
        if optimizer is not None:
            optimizer.zero_grad()

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()
            clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                            50)
            optimizer.step()

        # Log
        kl_loss_values.add(kl_loss.item())
        recon_loss_values.add(recon_loss.item())
        loss_values.add(loss.item())
        binding_loss_values.add(binding_loss.item())
        lr = (optimizer.param_groups[0]['lr']
              if optimizer is not None
              else None)

        # Update tqdm
        kl_loss_value = kl_loss_values.mean()
        recon_loss_value = recon_loss_values.mean()
        loss_value = loss_values.mean()
        binding_loss_value = binding_loss_values.mean()
        postfix = [f'loss={loss_value:.5f}',
                   f'(kl={kl_loss_value:.5f}',
                   f'recon={recon_loss_value:.5f})',
                   f'klw={kl_weight:.5f} lr={lr:.5f}'
                   f'bloss={binding_loss_value:.5f}']
        tqdm_data.set_postfix_str(' '.join(postfix))

    postfix = {
        'epoch': epoch,
        'kl_weight': kl_weight,
        'lr': lr,
        'kl_loss': kl_loss_value,
        'recon_loss': recon_loss_value,
        'loss': loss_value,
        'mode': 'Eval' if optimizer is None else 'Train'}

    return postfix

# Epoch start
totals = []

for epoch in range(100):
    # Epoch start


    res, binding = model.module.sample(1000)
    binding = mmss.inverse_transform(binding.reshape(-1, 1))
    binding = binding.reshape(-1)
    pd.DataFrame([res, binding]).to_csv("out_tests.csv")
    df = pd.DataFrame([res, binding])
    print(df)
    for i in range(20):
        print(res[i], binding[i])
        print("Binding stats: ", np.mean(binding), np.std(binding))


    # Epoch end
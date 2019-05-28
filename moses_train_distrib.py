from comet_ml import Experiment

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

import math
import mosesvae
import mosesvocab
from torch.optim.lr_scheduler import _LRScheduler

import random
from apex import amp, optimizers
import os
import argparse

parser = argparse.ArgumentParser()
# FOR DISTRIBUTED:  Parse for the local_rank argument, which will be supplied
# automatically by torch.distributed.launch.
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()
args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

train_sampler = None
if args.distributed:
    # FOR DISTRIBUTED:  Set the device according to local_rank.
    torch.cuda.set_device(args.local_rank)

    # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
    # environment variables, and requires that you use init_method=`env://`.
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')




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
        self.lr_end = 3 * 1e-4

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

def get_collate_fn():
    def collate(data):
        data.sort(key=len, reverse=True)
        tensors = [string2tensor(vocab, string)
                   for string in data]

        return tensors

    return collate



df = pd.read_csv("/vol/ml/aclyde/ZINC/zinc_cleaned.smi", nrows=2000000, header=None)
max_len = 0
print(df.head())
print(df.shape)
df = df.iloc[:,0].astype(str).tolist()

vocab = mosesvocab.OneHotVocab.from_data(df)
train_sampler = torch.utils.data.distributed.DistributedSampler(df)

train_loader = torch.utils.data.DataLoader(df, batch_size=128,
                          shuffle=False,
                          num_workers=8, collate_fn=get_collate_fn(),
                          worker_init_fn=mosesvocab.set_torch_seed_to_all_gens,
                                           pin_memory=True, sampler=train_sampler)

n_epochs = 50

model = mosesvae.VAE(vocab).cuda()
optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad),
                               lr=3*1e-4)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)


kl_annealer = KLAnnealer(n_epochs)
lr_annealer = CosineAnnealingLRWithRestart(optimizer)

model.zero_grad()


def _train_epoch(model, epoch, tqdm_data, kl_weight, optimizer=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    kl_loss_values = mosesvocab.CircularBuffer(1000)
    recon_loss_values = mosesvocab.CircularBuffer(1000)
    loss_values =mosesvocab.CircularBuffer(1000)
    for i, input_batch in enumerate(tqdm_data):
        input_batch = tuple(data.cuda() for data in input_batch)

        # Forwardd
        kl_loss, recon_loss = model(input_batch)
        kl_loss = torch.sum(kl_loss, 0)
        recon_loss = torch.sum(recon_loss, 0)

        loss = kl_weight * kl_loss + recon_loss


        # Backward
        if optimizer is not None:
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                            50)
            optimizer.step()

        # Log
        kl_loss_values.add(kl_loss.item())
        recon_loss_values.add(recon_loss.item())
        loss_values.add(loss.item())
        lr = (optimizer.param_groups[0]['lr']
              if optimizer is not None
              else None)

        # Update tqdm
        kl_loss_value = kl_loss_values.mean()
        recon_loss_value = recon_loss_values.mean()
        loss_value = loss_values.mean()
        postfix = [f'loss={loss_value:.5f}',
                   f'(kl={kl_loss_value:.5f}',
                   f'recon={recon_loss_value:.5f})',
                   f'klw={kl_weight:.5f} lr={lr:.5f}']
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
for epoch in range(n_epochs):
    # Epoch start
    kl_weight = kl_annealer(epoch)

    tqdm_data = tqdm(train_loader,
                     desc='Training (epoch #{})'.format(epoch))
    postfix = _train_epoch(model, epoch,
                                tqdm_data, kl_weight, optimizer)
    if args.local_rank == 0:
        torch.save(model.state_dict(), "trained_save.pt")

    # Epoch end
    lr_annealer.step()
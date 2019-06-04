
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
from rdkit.Chem import Crippen
from torch.nn.utils import clip_grad_norm_
import pickle
import math
import mosesvae
import mosesvocab
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.preprocessing import MinMaxScaler
import random
import os
import selfies
import re
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
# FOR DISTRIBUTED:  Parse for the local_rank argument, which will be supplied
# automatically by torch.distributed.launch.
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
# args.distributed = False
# if 'WORLD_SIZE' in os.environ:
#     args.distributed = int(os.environ['WORLD_SIZE']) > 1
#
# train_sampler = None
# if args.distributed:
#     # FOR DISTRIBUTED:  Set the device according to local_rank.
#
#
#     # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
#     # environment variables, and requires that you use init_method=`env://`.
#     torch.distributed.init_process_group(backend='nccl',
#                                          init_method='env://')


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
        smile = self.df.iloc[idx, 0]
        logp = self.df.iloc[idx, 1]
        return smile, logp



df = pd.read_csv("../dataset_v1.csv")

#df = df.sample(2000000, replace=False, random_state=42)
max_len = 0
selfs = []
counter = 51
sym_table = {}
logp = []
cannon_smiles = []
tqdm_range = tqdm(range(df.shape[0]))
for i in tqdm_range:
    try:
        original = str(df.iloc[i,0])

        m = Chem.MolFromSmiles(original)
        cannmon = Chem.MolToSmiles(m)
        ls = Crippen.MolLogP(m)
        selfie = selfies.encoder(cannmon)
        selfien = []
        re.findall("\[(.*?)\]", selfie)
        for sym in re.findall("\[(.*?)\]", selfie):
            if sym in sym_table:
                selfien.append(sym_table[sym])
            else:
                sym_table[sym] = chr(counter)
                counter += 1
                selfien.append(sym_table[sym])
        selfs.append(selfien)
        cannon_smiles.append(cannmon)
        logp.append(ls)

        postfix = [f'len=%s' % (len(sym_table))]
        tqdm_range.set_postfix_str(' '.join(postfix))
    except KeyboardInterrupt:
        exit()
    except:
        print("ERROR...")

df = pd.DataFrame(pd.Series(selfs))
df['logp'] = logp
df['cannon'] = cannon_smiles
df.to_csv("selfies.csv")
print(df.head())
print(df.shape)

charset = {k: v for v, k in sym_table.items()}
vocab = mosesvocab.OneHotVocab(sym_table.values())
bdata = BindingDataSet(df)
# train_sampler = torch.utils.data.distributed.DistributedSampler(bdata)
train_loader = torch.utils.data.DataLoader(bdata, batch_size=512,
                          shuffle=True,
                          num_workers=32, collate_fn=get_collate_fn_binding(),
                          worker_init_fn=mosesvocab.set_torch_seed_to_all_gens,
                                           pin_memory=True,)

n_epochs = 50

model = mosesvae.VAE(vocab).cuda()
binding_optimizer = None

optimizer = optim.Adam(model.parameters() ,
                               lr=3*1e-4 )
# model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
# model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)


kl_annealer = KLAnnealer(50)
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
        kl_loss, recon_loss, _ = model(input_batch)
        kl_loss = torch.sum(kl_loss, 0)
        recon_loss = torch.sum(recon_loss, 0)

        loss = kl_weight * kl_loss + recon_loss


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
        kl_loss, recon_loss, binding_loss, _ = model(input_batch, binding)


        kl_loss = torch.sum(kl_loss, 0)
        recon_loss = torch.sum(recon_loss, 0)
        binding_loss = torch.sum(binding_loss, 0)

        loss = min(1.0, kl_weight) * kl_loss + recon_loss + binding_loss


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
for epoch in range(100):
    # Epoch start
    kl_weight = kl_annealer(epoch)

    tqdm_data = tqdm(train_loader,
                     desc='Training (epoch #{})'.format(epoch))
    postfix = _train_epoch_binding(model, epoch,
                                tqdm_data, kl_weight, optimizer)
    if args.local_rank == 0:
        torch.save(model.state_dict(), "trained_save.pt")
        with open('vocab.pkl', 'wb') as f:
            pickle.dump(vocab, f)

        res, binding, _ = model.sample(1024)
        binding = binding.reshape(-1)
        pd.DataFrame([res, binding]).to_csv("out_tests.csv")
        for i in range(20):
            print(selfies.decoder("".join(['[' + charset[sym] + ']' for sym in res[i]])), binding[i])
        print("Binding stats: ", np.mean(binding), np.std(binding), np.max(binding), np.min(binding))

    # Epoch end
    lr_annealer.step()
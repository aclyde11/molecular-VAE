
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
        return len(self.df)

    def __getitem__(self, idx):
        smile = self.df[idx]
        return smile, 0

class SmilesLoaderSelfies(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        selfie = self.df.iloc[idx, 0]
        return selfie, 0

# df = pd.read_csv("../dataset_v1.csv")
# # df = df.sample(, replace=False, random_state=42)
# max_len = 0
# selfs = []
# counter = 51
# sym_table = {}
# cannon_smiles = []
# tqdm_range = tqdm(range(df.shape[0]))
# for i in tqdm_range:
#     try:
#         original = str(df.iloc[i,0])
#         if len(original) > 150:
#             continue
#         m = Chem.MolFromSmiles(original)
#         cannmon = Chem.MolToSmiles(m)
#         # selfie = cannmon
#         selfie = selfies.encoder(cannmon)
#         selfien = []
#         for sym in re.findall("\[(.*?)\]", selfie):
#         # for sym in selfie:
#             if sym in sym_table:
#                 selfien.append(sym_table[sym])
#             else:
#                 sym_table[sym] = chr(counter)
#                 counter += 1
#                 selfien.append(sym_table[sym])
#         selfs.append(selfien)
#         cannon_smiles.append(cannmon)
#
#         postfix = [f'len=%s' % (len(sym_table))]
#         tqdm_range.set_postfix_str(' '.join(postfix))
#     except KeyboardInterrupt:
#         exit()
#     except:
#         print("ERROR...")
#
# charset = {k: v for v, k in sym_table.items()}
# vocab = mosesvocab.OneHotVocab(sym_table.values())

with open("sym_table.pkl", 'rb') as f:
    sym_table = pickle.load(f)
with open("charset.pkl", 'rb') as f:
    charset = pickle.load(f)
with open("vocab.pkl", 'rb') as f:
    vocab = pickle.load(f)
with open("selfs.pkl", 'rb') as f:
    selfs = pickle.load(f)
with open("cannon_smiles.pkl", 'rb') as f:
    cannon_smiles = pickle.load(f)

# with open("sym_table.pkl", 'wb') as f:
#     pickle.dump(sym_table, f)
# with open("charset.pkl", 'wb') as f:
#     pickle.dump(charset, f)
# with open("vocab.pkl", 'wb') as f:
#     pickle.dump(vocab, f)
# with open("selfs.pkl", 'wb') as f:
#     pickle.dump(selfs, f)
# with open("cannon_smiles.pkl", 'wb') as f:
#     pickle.dump(cannon_smiles, f)


#
# df = pd.DataFrame(pd.Series(selfs))
# df['cannon'] = cannon_smiles
# print(df.head())
# print(df.shape)
# print(df.iloc[0,0][0])

bdata = BindingDataSet(selfs)
# train_sampler = torch.utils.data.distributed.DistributedSampler(bdata)
train_loader = torch.utils.data.DataLoader(bdata, batch_size=128,
                          shuffle=True,
                          num_workers=32, collate_fn=get_collate_fn_binding(),
                          worker_init_fn=mosesvocab.set_torch_seed_to_all_gens,
                                           pin_memory=True,)
train_loader_agg = torch.utils.data.DataLoader(bdata, batch_size=128,
                          shuffle=False,
                          sampler=torch.utils.data.RandomSampler(bdata, replacement=True, num_samples=50000),
                          num_workers=32, collate_fn=get_collate_fn_binding(),
                          worker_init_fn=mosesvocab.set_torch_seed_to_all_gens,
                                           pin_memory=True,)

def get_train_loader_agg():
    return torch.utils.data.DataLoader(bdata, batch_size=128,
                          shuffle=False,
                          sampler=torch.utils.data.RandomSampler(bdata, replacement=True, num_samples=50000),
                          num_workers=32, collate_fn=get_collate_fn_binding(),
                          worker_init_fn=mosesvocab.set_torch_seed_to_all_gens,
                                           pin_memory=True,)
n_epochs = 100

model = mosesvae.VAE(vocab).cuda()
binding_optimizer = None

# optimizer = optim.Adam(model.parameters() ,
#                                lr=3*1e-3 )
encoder_optimizer = optim.Adam(model.encoder.parameters(), lr=5e-4)
decoder_optimizer = optim.Adam(model.decoder.parameters(), lr=1e-4)
# model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
# model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)


kl_annealer = KLAnnealer(5)
lr_annealer_d = CosineAnnealingLRWithRestart(encoder_optimizer)
lr_annealer_e = CosineAnnealingLRWithRestart(decoder_optimizer)

model.zero_grad()



def _train_epoch_binding(model, epoch, tqdm_data, kl_weight, encoder_optim, decoder_optim):
    model.train()

    kl_loss_values = mosesvocab.CircularBuffer(10)
    recon_loss_values = mosesvocab.CircularBuffer(10)
    loss_values =mosesvocab.CircularBuffer(10)
    binding_loss_values = mosesvocab.CircularBuffer(10)
    for i, (input_batch, binding) in enumerate(tqdm_data):

        if epoch < 5:
            if i % 150 == 0:
                train_loader_agg_tqdm = tqdm(get_train_loader_agg(),
                     desc='Training encoder (epoch #{})'.format(epoch))
                for (input_batch, binding) in train_loader_agg_tqdm:
                    encoder_optimizer.zero_grad()
                    decoder_optimizer.zero_grad()
                    input_batch = tuple(data.cuda() for data in input_batch)
                    binding = binding.cuda().view(-1, 1)
                    # Forwardd
                    kl_loss, recon_loss, _, logvar, x, y = model(input_batch, binding)
                    kl_loss = torch.sum(kl_loss, 0)
                    recon_loss = torch.sum(recon_loss, 0)
                    _, predict = torch.max(F.softmax(y, dim=-1), -1)

                    correct = float((x == predict).sum().cpu().detach().item()) / float(x.shape[0] * x.shape[1])

                    loss = min(kl_weight* 0.5 + 1e-5, 1) * kl_loss + recon_loss
                    # loss = kl_loss + recon_loss
                    loss.backward()
                    clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                                    50)
                    encoder_optimizer.step()
                    loss_value = loss.item()
                    kl_loss_value = kl_loss.item()
                    recon_loss_value = recon_loss.item()

                    postfix = [f'loss={loss_value:.5f}',
                               f'(kl={kl_loss_value:.5f}',
                               f'recon={recon_loss_value:.5f})',
                               f'correct={correct:.5f}']
                    train_loader_agg_tqdm.set_postfix_str(' '.join(postfix))


        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        input_batch = tuple(data.cuda() for data in input_batch)
        binding = binding.cuda().view(-1, 1)
        # Forwardd
        kl_loss, recon_loss, _, logvar, x, y = model(input_batch, binding)
        _, predict = torch.max(F.softmax(y, dim=-1), -1)

        correct = float((x == predict).sum().cpu().detach().item()) / float(x.shape[0] * x.shape[1])

        kl_loss = torch.sum(kl_loss, 0)
        recon_loss = torch.sum(recon_loss, 0)

        loss = min(kl_weight* 5e-2 + 1e-5,1) * kl_loss + recon_loss
        # loss = kl_loss + recon_loss

        loss.backward()
        clip_grad_norm_((p for p in model.parameters() if p.requires_grad),
                        50)

        encoder_optimizer.step()
        decoder_optimizer.step()

        # Log
        kl_loss_values.add(kl_loss.item())
        recon_loss_values.add(recon_loss.item())
        loss_values.add(loss.item())
        lr = (encoder_optim.param_groups[0]['lr']
              if encoder_optim is not None
              else None)

        # Update tqdm
        kl_loss_value = kl_loss_values.mean()
        recon_loss_value = recon_loss_values.mean()
        loss_value = loss_values.mean()
        postfix = [f'loss={loss_value:.5f}',
                   f'(kl={kl_loss_value:.5f}',
                   f'recon={recon_loss_value:.5f})',
                   f'klw={kl_weight:.5f} lr={lr:.5f}'
                   f'correct={correct:.5f}']
        tqdm_data.set_postfix_str(' '.join(postfix))

    postfix = {
        'epoch': epoch,
        'kl_weight': kl_weight,
        'lr': lr,
        'kl_loss': kl_loss_value,
        'recon_loss': recon_loss_value,
        'loss': loss_value}

    return postfix


# Epoch start


print("STARTING THING I WANT.....")
# df = pd.read_csv("../combined_smiles.csv", header=None)
# df = pd.read_csv("../dataset_v1.csv")
# df = df.sample(50000, replace=False, random_state=42)
# seflie = []
# smile = []
# for i, row in df.iterrows():
#     try:
#         m = Chem.MolFromSmiles(row[0])
#         cannmon = Chem.MolToSmiles(m)
#         ls = Crippen.MolLogP(m)
#         selfie_ = selfies.encoder(cannmon)
#         seflie.append(selfie_)
#         smile.append(row[0])
#         print(selfie_)
#     except:
#         print("ERROR....")
#
#
# xs = pd.DataFrame(seflie)
# xs['ins'] = smile
# xs = xs.set_index("ins")
# print(xs)
# bdata = SmilesLoaderSelfies(xs)
# train_loader = torch.utils.data.DataLoader(bdata, batch_size=1,
#                           shuffle=False,
#                           num_workers=32, collate_fn=get_collate_fn_binding(),
#                           worker_init_fn=mosesvocab.set_torch_seed_to_all_gens,
#                                            pin_memory=True,)
# model.eval()

# hasher = {}
# zs = []
# smiles = []
# count = 0
# for i in tqdm(range(100)):
#     res, _, z = model.sample(2096)
#     z = z.detach().cpu().numpy()
#
#     smis = []
#     for i in range(2096):
#         count += 1
#         try:
#             s = selfies.decoder("".join(['[' + charset[sym] + ']' for sym in res[i]]))
#             m = Chem.MolFromSmiles(s)
#             s = Chem.MolToSmiles(m)
#             if s is not None:
#                 if s in hasher:
#                     hasher[s] += 1
#                 else:
#                     hasher[s] = 1
#                 smiles.append(s)
#                 zs.append(z[i,...])
#         except:
#             print("ERROR!!!")
#
#     # dfx = pd.DataFrame([res, binding])
#     print("LEN ", len(hasher), "TOAL VALID: ", float(len(hasher))/float(count))
# np.savez("zs.npz", np.stack(zs, axis=0))
# np.savez("ts.npz", np.array(smiles))

# hasher = {}
# zs = []
# smiles = []
# count = 0
#
# z_ = torch.randn(1, 128)
# z = z_.repeat([121, 1])
# x_ax = 23
# y_ax = 56
# step = 1e-6
# for i in range(121):
#     x = i / 11 - 5
#     y = i % 11 - 5
#     z[i,x_ax] = x * step + z[i,x_ax]
#     z[i,12] = -1 * x * step + z[i,12]
#
#     z[i,y_ax] = x * step + z[i,y_ax]
#     z[i,42] = -1 * x * step + z[i,42]
#
#
# for i in tqdm(range(1)):
#     res, _, z = model.sample(121, z=z)
#     z = z.detach().cpu().numpy()
#
#     smis = []
#     for i in range(2096):
#         count += 1
#         try:
#             s = selfies.decoder("".join(['[' + charset[sym] + ']' for sym in res[i]]))
#             m = Chem.MolFromSmiles(s)
#             s = Chem.MolToSmiles(m)
#             if s is not None:
#                 if s in hasher:
#                     hasher[s] += 1
#                 else:
#                     hasher[s] = 1
#                 smiles.append(s)
#                 zs.append(z[i,...])
#         except:
#             print("ERROR!!!")
#
#     # dfx = pd.DataFrame([res, binding])
#     print("LEN ", len(hasher), "TOAL VALID: ", float(len(hasher))/float(count))
# np.savez("zs.npz", np.stack(zs, axis=0))
# np.savez("ts.npz", np.array(smiles))

# for i in tqdm(range(100)):
#     res, _, z = model.sample(2096)
#     z = z.detach().cpu().numpy()
#
#     smis = []
#     for i in range(2096):
#         count += 1
#         try:
#             s = selfies.decoder("".join(['[' + charset[sym] + ']' for sym in res[i]]))
#             m = Chem.MolFromSmiles(s)
#             s = Chem.MolToSmiles(m)
#             if s is not None:
#                 if s in hasher:
#                     hasher[s] += 1
#                 else:
#                     hasher[s] = 1
#                 smiles.append(s)
#                 zs.append(z[i,...])
#         except:
#             print("ERROR!!!")
#
#     # dfx = pd.DataFrame([res, binding])
#     print("LEN ", len(hasher), "TOAL VALID: ", float(len(hasher))/float(count))
# np.savez("zs.npz", np.stack(zs, axis=0))
# np.savez("ts.npz", np.array(smiles))

# for i, (x, b) in enumerate(train_loader):
#     input_batch = tuple(data.cuda() for data in x)
#     b = b.cuda().float()
#     _,_,_,z = model(input_batch, b)
#     vecs.append(z.detach().cpu().numpy())
# # xs.to_csv("smiles_computed.csv")
# np.savez("z_vae_moses.npz", np.concatenate(vecs, axis=0))

for epoch in range(100):


    kl_weight = kl_annealer(epoch)


    tqdm_data = tqdm(train_loader,
                     desc='Training (epoch #{})'.format(epoch))
    postfix = _train_epoch_binding(model, epoch,
                                tqdm_data, kl_weight, encoder_optim=encoder_optimizer, decoder_optim=decoder_optimizer)
    torch.save(model.state_dict(), "trained_save_small.pt")
    # with open('vocab.pkl', 'wb') as f:
    #     pickle.dump(vocab, f)

    res, _ = model.sample(1024)
    pd.DataFrame([res]).to_csv("out_tests.csv")
    try:
        for i in range(50):
            print(selfies.decoder("".join(['[' + charset[sym] + ']' for sym in res[i]])))
            # print("".join([ charset[sym] for sym in res[i]]))
    except Exception as e:
        print("error...")
        print("Not sure why nothing printed..")
        print(str(e))

    # Epoch end
    lr_annealer_e.step()
    lr_annealer_d.step()
import pickle


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
from multiprocessing import Process, Pipe

def gen_proc(comm, iters=10000, i=0, batch_size=4096):
    with open("sym_table.pkl", 'rb') as f:
        sym_table = pickle.load(f)
    with open("charset.pkl", 'rb') as f:
        charset = pickle.load(f)
    with open("vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    model = mosesvae.VAE(vocab).cuda()
    model.load_state_dict(torch.load("trained_save.pt"))
    model = model.cuda(i)

    for _ in tqdm(range(iters)):
        count = 0

        res, _, _ = model.sample(batch_size)

        smis = []
        for i in range(batch_size):
            count += 1
            try:
                s = selfies.decoder("".join(['[' + charset[sym] + ']' for sym in res[i]]))
                smis.append(s)
            except:
                print("ERROR!!!")
        comm.send((smis, count))
    comm.close()

if __name__ == '__main__':
    parent_conn, child_conn = Pipe()
    p = Process(target=gen_proc, args=(child_conn,))
    p.start()
    print(parent_conn.recv())  # prints "[42, None, 'hello']"
    p.kill()



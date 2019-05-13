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
import sys
from rdkit import Chem
checkpoint = torch.load(sys.argv[1], map_location='cpu')

model = MolecularVAE(i=checkpoint['max_len'], c=len(checkpoint['charset']), o=360)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.cuda()
charset = checkpoint['charset']

model.eval()
b_size = 2000
max_times = 1e9
counter = 0
total =0
with open(sys.argv[2], 'w') as f:
    with torch.no_grad():
        while not (total > max_times or counter > 10000):
            sampler = torch.rand(size=(b_size, 360)).cuda()
            recon_batch = model.decoder(sampler)
            _, preds = torch.max(recon_batch, dim=2)
            preds = preds.cpu().numpy()
            for i in tqdm(range(b_size)):
                sample = preds[i, ...]
                out = "".join([charset[chars] for chars in sample]).rstrip()
                m = Chem.SmilesToMol(out)
                total += 1
                if m is not None:
                    counter += 1
                    print("counter = ", counter)
                    f.write(out)
                    f.write("\n")


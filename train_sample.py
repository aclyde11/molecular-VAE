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

checkpoint = torch.load(sys.argv[1], map_location='cpu')

model = MolecularVAE(i=checkpoint['max_len'], c=len(checkpoint['charset']), o=360).cuda()
model.load_state_dict(checkpoint['model_state_dict'])
model = model.cuda()
charset = checkpoint['charset']

model.eval()
b_size = 1000

with open(sys.argv[2], 'w') as f:
    with torch.no_grad():
        for batch_idx in range(100):
            sampler = torch.rand(size=(100, 360))
            recon_batch = model.decoder(sampler)
            _, preds = torch.max(recon_batch, dim=2)
            preds = preds.cpu().numpy()
            for i in range(b_size):
                sample = preds[i, ...]
                out = "".join([charset[chars] for chars in sample]).rstrip()
                f.write(out)
                f.write("\n")


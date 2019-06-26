import torch
import torch.utils.data

from rdkit import Chem

import pickle
import mosesvae

import selfies
import argparse
import time
from tqdm import tqdm
from multiprocessing import Process, Pipe, Queue, Manager, Value
import numpy as np
def interpolate_points(x,y, sampling):
    from sklearn.linear_model import LinearRegression
    ln = LinearRegression()
    data = np.stack((x,y))
    data_train = np.array([0, 1]).reshape(-1, 1)
    ln.fit(data_train, data)

    return ln.predict(sampling.reshape(-1, 1)).astype(np.float32)


dir='smiles_kinase'
i=0
try:
    with open(dir + "/charset.pkl", 'rb') as f:
        charset = pickle.load(f)
    with open(dir + "/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    model = mosesvae.VAE(vocab)
    model.load_state_dict(torch.load(dir + "/trained_save_small.pt", map_location='cpu'))
    model = model.cuda(i)

    for _ in range(1):
        count = 0
        data_latent  = torch.randn(100, 128,
                    device='cuda')
        pt_1 = data_latent[0, ...].cpu().numpy()
        pt_2 = data_latent[1 + 1, ...].cpu().numpy()
        sample_vec = interpolate_points(pt_1, pt_2,
                                        np.linspace(0, 1, num=100, endpoint=True))
        sample_vec = torch.from_numpy(sample_vec).cuda()
        res, _ = model.sample(100, z=sample_vec)

        smis = []
        for i in range(100):
            count += 1
            try:
                s = "".join([charset[sym]  for sym in res[i]])
                smis.append(s)
            except:
                None
                # print("ERROR!!!")
                # print('res', res[i])
                # print("charset", charset)
        for i in smis:
            print(i)
except KeyboardInterrupt:
    print("exiting")
    exit()
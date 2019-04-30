import torch
from torchvision import datasets, transforms
import numpy as np


class MoleLoader(torch.utils.data.Dataset):
    def __init__(self, df, vocab, max_len=70, num=None):
        super(MoleLoader, self).__init__()

        self.df = df
        self.vocab = vocab
        self.max_len = max_len




    def __len__(self):
        return self.df.shape[0]



    def __getitem__(self, item):
        smile = str(self.df.iloc[item, 0]).ljust(self.max_len, ' ')
        embedding = np.array([self.vocab[x] for x in smile])
        embedding = torch.LongTensor(embedding)

        return embedding

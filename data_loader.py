import torch
from torchvision import datasets, transforms
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class MoleLoader(torch.utils.data.Dataset):
    def __init__(self, df, vocab, max_len=70, num=None):
        super(MoleLoader, self).__init__()

        self.df = df
        self.vocab = vocab
        self.max_len = max_len
        self.one_hot_encoder =  OneHotEncoder(categories=list(range(len(vocab))) ,handle_unknown='error')



    def __len__(self):
        return self.df.shape[0]

    def one_hot_encode(self, item):
        s = self.one_hot_encoder.fit(item)
        # stacks = []
        # for i in item:
        #     print(i)
        #     tmp = self.one_hot_encoder.fit(i)
        #     print(tmp.shape)
        #     stacks.append(tmp)
        # s = np.stack(stacks)
        print(s.shape)
        return s


    def __getitem__(self, item):
        smile = str(self.df.iloc[item, 0]).ljust(self.max_len, ' ')
        embedding = np.array([self.vocab[x] for x in smile])
        embedding = self.one_hot_encode(embedding)
        return embedding

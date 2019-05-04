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
        self.one_hot_encoder =  OneHotEncoder(categories=[list(range(len(vocab)))], handle_unknown='error', sparse=False)



    def __len__(self):
        return self.df.shape[0]

    def one_hot_encode(self, item):
        items = item.reshape(-1, 1)
        s = self.one_hot_encoder.fit_transform(items)
        return s


    def __getitem__(self, item):
        smile = str(self.df.iloc[item, 0]).ljust(self.max_len, ' ')
        embedding = np.array([self.vocab[x] for x in smile])
        embedding_ohe = self.one_hot_encode(embedding)
        return torch.LongTensor(embedding), torch.FloatTensor(embedding_ohe)

import numpy as np
from featurizer import OneHotFeaturizer
from tqdm import tqdm

from optparse import OptionParser
import pickle

def convert_to_embed(smi):
    return np.array([vocab[x] for x in smi])



if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path")
    parser.add_option("-n", "--split", dest="nsplits", default=10)
    parser.add_option("-j", "--jobs", dest="njobs", default=8)
    parser.add_option("-o", "--out", dest="out", default="out")
    parser.add_option("-v", "--vocab", dest='vocab', default=None)
    parser.add_option("-m", "--max_len", dest='max_len', default=None)
    opts, args = parser.parse_args()
    opts.njobs = int(opts.njobs)

    vocab = set()
    max_len = 0
    smiles = []


    if opts.vocab is None:
        with open(opts.train_path) as f:
            for smi in tqdm(f):
                smi = smi.rstrip()
                smiles.append(smi)
                max_len = max(len(smi), max_len)
                for i in smi:
                    vocab.add(i)

        vocab.add(' ')
        vocab = list(vocab)
        vocab = { c : v for v, c in enumerate(vocab)}


        with open(opts.out + "vocab.pkl", 'wb') as f:
            pickle.dump(vocab, f)
    else:
        with open(opts.train_path) as f:
            for smi in tqdm(f):
                smi = smi.rstrip()
                smiles.append(smi)
        with open(opts.vocab, 'rb') as f:
            vocab = pickle.load(f)
            max_len = opts.max_len
    print("max length ", max_len)
    print("done with vocab")
    #ohf = OneHotFeaturizer(vocab=vocab, padlength=max_len + 5)
    #oh_smiles = ohf.featurize(smiles)

    sd = []
    for smi in tqdm(smiles):
        sd.append(convert_to_embed(smi))
    oh_smiles = np.array(sd)
    print(oh_smiles.shape)


    np.save("out", arr=oh_smiles)

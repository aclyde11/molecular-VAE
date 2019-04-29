import numpy as np
from featurizer import OneHotFeaturizer
from tqdm import tqdm

from optparse import OptionParser


def convert_to_embed(smi):
    return np.array([vocab[x] for x in smi])



if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path")
    parser.add_option("-n", "--split", dest="nsplits", default=10)
    parser.add_option("-j", "--jobs", dest="njobs", default=8)
    opts, args = parser.parse_args()
    opts.njobs = int(opts.njobs)

    vocab = set()
    max_len = 0
    smiles = []
    with open(opts.train_path) as f:
        for smi in tqdm(f):
            smi = smi.rstrip()
            smiles.append(smi)
            max_len = max(len(smi), max_len)
            for i in smi:
                vocab.add(i)

    vocab.add(' ')
    vocab = list(vocab)



    #ohf = OneHotFeaturizer(vocab=vocab, padlength=max_len + 5)
    #oh_smiles = ohf.featurize(smiles)

    oh_smiles = np.array([convert_to_embed(smi) for smi in smiles])
    print(oh_smiles.shape)


    np.savez_compressed('output.npz', arr=oh_smiles)

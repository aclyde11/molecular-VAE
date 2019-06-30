import pandas as pd
import numpy as  np
from tqdm import tqdm
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols

from oe_analysis import FastRocker


class JohnSim():
    def __init__(self):
        self.df  = pd.read_csv("/workspace/john_smiles_kinasei.smi", sep=' ', header=None)
        self.df = self.df.set_index(self.df.columns[1])
        self.john_fps = [FingerprintMols.FingerprintMol(Chem.MolFromSmiles(x[1][0])) for x in tqdm(self.df.iterrows())]


    def simalarity_to_john(self, s1):
        fp_ms = FingerprintMols.FingerprintMol(Chem.MolFromSmiles(s1))
        maxes = 0
        for fp in self.john_fps:
            maxes = max(maxes, DataStructs.FingerprintSimilarity(fp_ms, fp))
        return maxes

johnsim = JohnSim()
fast_rocs = FastRocker("../data.oeb")
df = pd.read_csv("finetuning/out_samples.smi", header=None)

with open("log.txt", 'w', buffering=1) as f:
    f.write("smiles,color,pdb_match,sim\n")
    for _, smi in df.iterrows():
        smi = smi[0]
        x = fast_rocs.get_color(smi)
        y = johnsim.simalarity_to_john(smi)
        if x is not None:
            f.write(smi + ",")
            f.write(str(x[0]) + ",")
            f.write(str(x[1]) + ",")
            f.write(str(y) + "\n")
import pandas as pd
import numpy as  np
from tqdm import tqdm
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols

from oe_analysis import FastRocker


class JohnSim():
    def __init__(self):
        self.df  = pd.read_csv("/workspace/pdb.smi", sep=' ', header=None)
        self.fingerprints = []
        for i, row in tqdm(self.df.iterrows()):
            smile = row[0]
            name = row[1]
            try:
                fp = FingerprintMols.FingerprintMol(Chem.MolFromSmiles(smile))
                self.fingerprints.append((name, fp))
            except:
                continue



    def simalarity_to_john(self, s1):
        fp_ms = FingerprintMols.FingerprintMol(Chem.MolFromSmiles(s1))
        maxes = 0
        argmax = 0
        for name,fp in self.fingerprints:
            tmp = DataStructs.FingerprintSimilarity(fp_ms, fp)
            if tmp > maxes:
                maxes = tmp
                argmax = name
        return maxes, argmax

johnsim = JohnSim()
fast_rocs = FastRocker("../data.oeb")
df = pd.read_csv("/workspace/python3jtnn/fast_molvae/sampled.smi", header=None)

with open("/workspace/python3jtnn/fast_molvae/log.txt", 'w', buffering=1) as f:
    f.write("name,smiles,color,pdb_match,sim,ligand\n")
    for _, smi in df.iterrows():
        print(smi)
        smi_name = smi[1]
        smi = smi[0]
        x = fast_rocs.get_color(smi)
        if x is not None:
            y, name = johnsim.simalarity_to_john(smi)
            f.write(smi_name + ",")
            f.write(smi + ",")
            f.write(str(x[0]) + ",")
            f.write(str(x[1]) + ",")
            f.write(str(y) + ",")
            f.write(str(name) + "\n")
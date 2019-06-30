import pandas as pd
import numpy as  np


from oe_analysis import FastRocker

fast_rocs = FastRocker("../data.oeb")
df = pd.read_csv("finetuning/out_samples.smi", header=None)

with open("log.txt", 'w', buffering=1) as f:
    f.write("smiles,color,pdb_match,sim\n")
    for _, smi in df.iterrows():
        smi = smi[0]
        x = fast_rocs.get_color(smi)
        if x is not None:
            f.write(smi + ",")
            f.write(str(x[0]) + ",")
            f.write(str(x[1]) + ",")
            f.write("0\n")
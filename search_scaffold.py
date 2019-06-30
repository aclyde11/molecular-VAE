import pandas as pd
import numpy as  np


from oe_analysis import FastRocker

fast_rocs = FastRocker("../data.oeb")
df = pd.read_csv("finetuning/out_samples.smi", header=None)
for _, smi in df.iterrows():
    smi = smi[0]
    print(fast_rocs.get_color(smi))

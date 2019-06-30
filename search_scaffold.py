import torch
import torch.utils.data

from rdkit import Chem
import oe_analysis
import pickle
import mosesvae

import selfies
import argparse
import subprocess
from oe_analysis import FastRocker
import time
from tqdm import tqdm
from multiprocessing import Process, Pipe, Queue, Manager, Value

def gen_proc(iters, i, batch_size, dir, selfies):
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    print("Generator on", i)

    fastrocs = FastRocker("../data.oeb")
    try:
        with open(dir + "/charset.pkl", 'rb') as f:
            charset = pickle.load(f)
        with open(dir + "/vocab.pkl", 'rb') as f:
            vocab = pickle.load(f)
        model = mosesvae.VAE(vocab)
        model.load_state_dict(torch.load(dir + "/trained_save_small.pt", map_location='cpu'))
        model = model.cuda(i)

        for _ in range(iters):
            count = 0

            res, _ = model.sample(batch_size)

            smis = []
            for i in range(batch_size):
                count += 1
                try:
                    if selfies:
                        s = "".join(['[' + charset[sym] + ']' for sym in res[i]])
                        s = selfies.decoder(s)
                    else:
                        s = "".join([charset[sym]  for sym in res[i]])
                    smis.append(s)
                except:
                    None
                    # print("ERROR!!!")
                    # print('res', res[i])
                    # print("charset", charset
            for smi in smis:
                print(smi, "calling fast rocs~!")
                score, idx = fastrocs.get_color(smi)
                print(score, idx)

    except KeyboardInterrupt:
        print("exiting")
        exit()


def hasher(q, hasher, valid, total, i, s, stop, pause, new_unique):
    from rdkit import rdBase

    while not stop.value:
        if not q.empty():
            smis, count = q.get(block=True)
            total.value += count
            for smi in smis:
                try:
                    if s:
                        smi = selfies.decoder(smi)
                    m = Chem.MolFromSmiles(smi)
                    s = Chem.MolToSmiles(m)
                    if s is not None:
                        valid.value += 1
                        if s in hasher:
                            hasher[s] += 1
                        else:
                            hasher[s] = 1
                            new_unique.append(s)
                except KeyboardInterrupt:
                    print("Bye")
                    exit()
                except:
                    None
        while pause.value:
            time.sleep(60)

def reporter(q, d, valid, total, dir, stop, pause, new_unique):
    print("Starting up reporter.")
    start_time = time.time()
    iter = 0
    mol_counter = 0
    mols = {}
    with open("log_small.csv", 'w', buffering=1) as f:
        f.write("time,unique,valid,total\n")
        try:
            while not stop:
                try:
                    iter += 1
                    time.sleep(5)
                    curr_time = time.time()
                    u = len(d)
                    v = valid.value
                    t = total.value
                    f.write("{0},{1},{2},{3}\n".format(curr_time, u, v, t))
                    print("Reporting! ")
                    print("Queue length: ",     q.qsize())
                    print("Unique: ", u, float(u)/ t)
                    print("Valid: ", v, float(v)  / t)
                    print("Sampled: ", t)
                    print("Samples per second: ", float(t) / float(curr_time - start_time) )
                    print("Unique per second: ", float(u) / float(curr_time - start_time) )
                    if iter % 10 == 0:
                        with open(dir + "/out_samples.smi", 'w') as outf:
                            for i in d.keys():
                                outf.write(i + "\n")
                    if len(new_unique >= 100):
                        pause.value = True
                        with open("unique_out" + ".smi") as f:
                            for i in range(len(new_unique)):
                                mols[mol_counter] = i
                                f.write(new_unique[i] + " " + "m_" + str(mol_counter) + "\n" )
                        new_unique.clear()

                    print("Pausing. Then exiting")
                    time.sleep(10)
                    stop.value = True
                    exit()
                        # send off to rocs and omega.

                    subprocess.check_call(['./omega2 -in unique_out.smi -out unique_out.oeb.gz'])
                    hits = oe_analysis.get_color("data.oeb.gz", "unique_out"  + ".oeb.gz")
                    ##find best hit and output it.

                except ZeroDivisionError:
                    print("eh zero error.")

        except KeyboardInterrupt:
            print("Exiting")
            exit()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="")
    parser.add_argument('-s', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--hashers', type=int, default=1)
    args = parser.parse_args()
    manager = Manager()
    gen_proc(10000,42,4096 * 2, args.in_dir, args.s)




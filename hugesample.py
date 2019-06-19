import torch
import torch.utils.data

from rdkit import Chem

import pickle
import mosesvae

import selfies
import time
from tqdm import tqdm
from multiprocessing import Process, Pipe, Queue, Manager, Value

def gen_proc(comm, iters=10000, i=0, batch_size=4096):
    try:
        with open("sym_table.pkl", 'rb') as f:
            sym_table = pickle.load(f)
        with open("charset.pkl", 'rb') as f:
            charset = pickle.load(f)
        with open("vocab.pkl", 'rb') as f:
            vocab = pickle.load(f)
        model = mosesvae.VAE(vocab).cuda()
        model.load_state_dict(torch.load("trained_save.pt"))
        model = model.cuda(i)

        for _ in tqdm(range(iters)):
            count = 0

            res, _, _ = model.sample(batch_size)

            smis = []
            for i in range(batch_size):
                count += 1
                try:
                    s = selfies.decoder("".join(['[' + charset[sym] + ']' for sym in res[i]]))
                    smis.append(s)
                except:
                    print("ERROR!!!")
            comm.put((smis, count))
    except KeyboardInterrupt:
        print("exiting")
        exit()


def hasher(q, hasher, valid, total, i):
    from rdkit import rdBase
    rdBase.DisableLog('rdApp.error')
    print("Hasher Thread on", i)

    while True:
        if not q.empty():
            smis, count = q.get(block=True)
            total.value += count
            for smi in smis:
                try:
                    m = Chem.MolFromSmiles(smi)
                    s = Chem.MolToSmiles(m)
                    if s is not None:
                        valid.value += 1
                        if s in hasher:
                            hasher[s] += 1
                        else:
                            hasher[s] = 1
                except KeyboardInterrupt:
                    print("Bye")
                    exit()
                except:
                    print("error...")

def reporter(q, d, valid):
    try:
        while True:
            time.sleep(2)
            print("Reporting! ")
            print("Queue length: ",     q.qsize())
            print("Valid: ", len(d))
            print("Valid: ", valid.value)
    except KeyboardInterrupt:
        print("Exiting")
        exit()


if __name__ == '__main__':
    manager = Manager()
    valid = Value('i', 0)
    total = Value('i', 0)
    d = manager.dict()
    q = Queue()
    p = Process(target=gen_proc, args=(q,10000,0,4096)) ##workers
    h = Process(target=hasher, args=(q, d, valid, total, 0)) ## hasher
    r = Process(target=reporter, args=(q, d, valid))

    p.start()
    h.start()
    r.start()
    try:
        p.join()
        h.join()
        r.join()
    except KeyboardInterrupt:
        p.kill()
        h.kill()
        r.kill()



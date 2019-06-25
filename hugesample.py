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
    print("Generator on", i)
    try:
        with open("kinase/charset.pkl", 'rb') as f:
            charset = pickle.load(f)
        with open("kinase/vocab.pkl", 'rb') as f:
            vocab = pickle.load(f)
        model = mosesvae.VAE(vocab)
        model.load_state_dict(torch.load("kinase/trained_save_small.pt", map_location='cpu'))
        model = model.cuda(i)

        for _ in range(iters):
            count = 0

            res, _ = model.sample(batch_size)

            smis = []
            for i in range(batch_size):
                count += 1
                try:
                    s = "".join(['[' + charset[sym] + ']' for sym in res[i]])
                    smis.append(s)
                except:
                    print("ERROR!!!")
                    print('res', res[i])
                    print("charset", charset)
            comm.put((smis, count))
            if comm.qsize() > 100:
                time.sleep(20)
    except KeyboardInterrupt:
        print("exiting")
        exit()


def hasher(q, hasher, valid, total, i):
    from rdkit import rdBase
    rdBase.DisableLog('rdApp.error')
    print("Hasher Thread on", i)
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    while True:
        if not q.empty():
            smis, count = q.get(block=True)
            total.value += count
            for smi in smis:
                try:
                    # smi = selfies.decoder(smi)
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
                    None

def reporter(q, d, valid, total):
    print("Starting up reporter.")
    start_time = time.time()
    with open("log_small.csv", 'w', buffering=1) as f:
        f.write("time,unique,valid,total\n")
        try:
            while True:
                try:
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
                except ZeroDivisionError:
                    print("eh zero error.")

        except KeyboardInterrupt:
            print("Exiting")
            exit()



if __name__ == '__main__':
    manager = Manager()
    valid = Value('i', 0)
    total = Value('i', 0)
    d = manager.dict()
    q = Queue()
    ps = []
    for i in range(1):
        ps.append(Process(target=gen_proc, args=(q,10000,i,4096 * 2))) ##workers
    hs = []
    for i in range(1 * 1):
        hs.append(Process(target=hasher, args=(q, d, valid, total, i))) ## hasher

    r = Process(target=reporter, args=(q, d, valid, total))

    for  p in ps:
        p.start()
    for h in hs:
        h.start()
    r.start()
    try:
        for p in ps:
            p.join()
        for h in hs:
            h.join()
        r.join()
    except:
        for p in ps:
            p.kill()
        for h in hs:
            h.kill()
        r.kill()



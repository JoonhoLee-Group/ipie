import numpy as np
import cupy as cp
import time
from mpi4py import MPI
import sys
import os 
os.environ["I_MPI_PMI_LIBRARY"] = '/cm/shared/apps/slurm/20.02.6/lib64/libpmi2.so'

divide = 2
nao = 1000 // divide
nocc = 200  // divide
naux = 4000  // divide
nwalkers = int(sys.argv[1])

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

""" MPI CPU """
chol = np.random.rand(naux, nao*nao)
x = np.random.rand(nwalkers, naux)
t0 = time.time()
VHS = np.dot(x, chol)
t1 = time.time()
print("MPI Rank {} CPU Time: {}".format(rank, t1 - t0))

""" MPI GPU """
chol = np.random.rand(naux, nao*nao)
x = np.random.rand(nwalkers, naux)
with cp.cuda.Device(rank):
    warmup = cp.dot(cp.array(np.random.rand(2,2)),cp.array(np.random.rand(2,2)))
    x = cp.array(x)
    chol = cp.array(chol)
    t0 = time.time()
    VHS = cp.dot(x, chol)
    t1 = time.time()
    print(VHS.dtype)
print("MPI Rank {} - GPU Time: {} on GPU {}".format(rank, t1 - t0, VHS.device))

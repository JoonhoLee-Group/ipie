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
rchol = np.random.rand(naux,nocc*nao)
walkers_batch_Ghalf = np.random.rand(2, nwalkers, nao * nocc)
t0 = time.time()
vfb = rchol.dot(walkers_batch_Ghalf[0].T) + rchol.dot(walkers_batch_Ghalf[1].T)
t1 = time.time()
print("MPI Rank {} CPU Time: {}".format(rank, t1 - t0))




""" MPI GPU """
rchol = np.random.rand(naux,nocc*nao)
walkers_batch_Ghalf = np.random.rand(2, nwalkers, nao * nocc)
with cp.cuda.Device(rank):
    warmup = cp.dot(cp.array(np.random.rand(2,2)),cp.array(np.random.rand(2,2)))
    rchol = cp.asarray(rchol)
    walkers_batch_Ghalf = cp.asarray(walkers_batch_Ghalf)
    t0 = time.time()
    vfb = rchol.dot(walkers_batch_Ghalf[0].T) + rchol.dot(walkers_batch_Ghalf[1].T)
    t1 = time.time()
print("MPI Rank {} - CPU/GPU Time: {} on GPU {}".format(rank, t1 - t0, vfb.device))

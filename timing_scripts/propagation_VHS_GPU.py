import numpy as np
import cupy as cp
import time
from mpi4py import MPI
import sys
import os
#os.environ["I_MPI_PMI_LIBRARY"] = '/cm/shared/apps/slurm/20.02.6/lib64/libpmi2.so'
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["MKL_NUM_THREADS"] = '1'

divide = 1
nao = 439 // divide
nocca = 94 // divide
noccb = 92 // divide
naux = 3468 // divide
nwalkers = 50
nsteps = 125

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

""" MPI CPU """
chol = np.random.rand(naux, nao*nao)
x = np.random.rand(nwalkers, naux)
#t0 = time.time()
#for i in range(nsteps):
#    VHS = np.dot(x, chol)
#t1 = time.time()
#print("MPI Rank {} CPU Time: {}".format(rank, t1 - t0))

""" MPI GPU """
chol = np.random.rand(naux, nao*nao)
x = np.random.rand(nwalkers, naux)
with cp.cuda.Device(rank):
    warmup = cp.dot(cp.array(np.random.rand(2,2)),cp.array(np.random.rand(2,2)))
    x = cp.array(x, dtype = cp.float64)
    chol = cp.array(chol, dtype = cp.float64)
    t0 = time.time()
    for i in range(nsteps):
        VHS = cp.dot(x, chol)
    t1 = time.time()
    print(VHS.dtype)
print("MPI Rank {} - GPU Time: {} on GPU {}".format(rank, t1 - t0, VHS.device))
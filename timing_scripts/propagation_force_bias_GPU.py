import os
import sys
import time

import cupy as cp
import numpy as np
from mpi4py import MPI

# os.environ["I_MPI_PMI_LIBRARY"] = '/cm/shared/apps/slurm/20.02.6/lib64/libpmi2.so'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
rchol_a = np.random.rand(naux, nocca * nao)
rchol_b = np.random.rand(naux, noccb * nao)
walkers_batch_Ghalfa = np.random.rand(nwalkers, nao * nocca) + 1.0j * np.random.rand(
    nwalkers, nao * nocca
)
walkers_batch_Ghalfb = np.random.rand(nwalkers, nao * noccb) + 1.0j * np.random.rand(
    nwalkers, nao * noccb
)
# t0 = time.time()
# for i in range(nsteps):
#    vfb = rchol_a.dot(walkers_batch_Ghalfa.T) + rchol_b.dot(walkers_batch_Ghalfb.T)
# t1 = time.time()
# print("MPI Rank {} CPU Time: {}".format(rank, t1 - t0))

""" MPI GPU """
with cp.cuda.Device(rank):
    warmup = cp.dot(cp.array(np.random.rand(2, 2)), cp.array(np.random.rand(2, 2)))
    rchol_a = cp.asarray(rchol_a, dtype=cp.float64)
    rchol_b = cp.asarray(rchol_b, dtype=cp.float64)
    walkers_batch_Ghalfa = cp.asarray(walkers_batch_Ghalfa)
    walkers_batch_Ghalfb = cp.asarray(walkers_batch_Ghalfb)
    vfb = cp.zeros((nwalkers, naux), dtype=cp.complex64)
    tmp = cp.zeros((naux, nwalkers), dtype=cp.complex64)
    t0 = time.time()
    for i in range(nsteps):
        tmp[:, :].real = rchol_a.dot(walkers_batch_Ghalfa.real.T) + rchol_b.dot(
            walkers_batch_Ghalfb.real.T
        )
        tmp[:, :].imag = rchol_a.dot(walkers_batch_Ghalfa.imag.T) + rchol_b.dot(
            walkers_batch_Ghalfb.imag.T
        )
        vfb = tmp.T.copy()
    t1 = time.time()
print("MPI Rank {} - CPU/GPU Time: {} on GPU {}".format(rank, t1 - t0, vfb.device))

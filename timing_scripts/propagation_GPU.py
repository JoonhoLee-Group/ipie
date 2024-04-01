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

chol = np.random.rand(naux, nao * nao)
x = np.random.rand(nwalkers, naux)
VHS = x.dot(chol)
VHS = VHS.reshape(nwalkers, nao, nao)
walkers_batch_phi0 = np.random.rand(nwalkers, nao, nocca + noccb) + 1.0j * np.random.rand(
    nwalkers, nao, nocca + noccb
)

# """ MPI CPU """
# walkers_batch_phi = walkers_batch_phi0.copy()
# t0 = time.time()
# for t in range(nsteps):
#    for iw in range(nwalkers):
#        for i in range (6):
#               walkers_batch_phi[iw] = VHS[iw].dot(walkers_batch_phi[iw])
# t1 = time.time()
# print("MPI Rank {} CPU Time: {}".format(rank, (t1 - t0)))

""" MPI GPU using for loop"""
with cp.cuda.Device(rank):
    mat = cp.array(np.random.rand(2, 2))
    warmup = cp.dot(cp.array(np.random.rand(2, 2)), cp.array(np.random.rand(2, 2)))
    walkers_batch_phi = cp.asarray(walkers_batch_phi0.copy(), dtype=cp.complex64)
    t0 = time.time()
    VHS = cp.asarray(VHS, dtype=cp.float32)
    for t in range(nsteps):
        for iw in range(nwalkers):
            for i in range(6):
                walkers_batch_phi[iw] = VHS[iw].dot(walkers_batch_phi[iw].real) + 1.0j * VHS[
                    iw
                ].dot(walkers_batch_phi[iw].imag)
    t1 = time.time()
print("MPI Rank {} GPU for loop Time: {} on GPU {}".format(rank, (t1 - t0), VHS.device))


""" MPI GPU using einsum"""
with cp.cuda.Device(rank):
    mat = cp.array(np.random.rand(2, 2))
    warmup = cp.einsum("ab,bc->ac", mat, mat, optimize=True)
    walkers_batch_phi = cp.asarray(walkers_batch_phi0.copy(), dtype=cp.complex64)
    t0 = time.time()
    VHS = cp.asarray(VHS, dtype=cp.float32).reshape(nwalkers, nao, nao)
    for t in range(nsteps):
        for i in range(6):
            walkers_batch_phi = cp.einsum(
                "wmn,wni->wmi", VHS, walkers_batch_phi.real, optimize=True
            ) + 1.0j * cp.einsum("wmn,wni->wmi", VHS, walkers_batch_phi.imag, optimize=True)
    t1 = time.time()
print("MPI Rank {} GPU einsum Time: {} on GPU {}".format(rank, (t1 - t0), VHS.device))

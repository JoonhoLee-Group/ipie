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

chol = np.random.rand(naux,nao*nao)
x = np.random.rand(nwalkers, naux)
VHS = x.dot(chol)
VHS = VHS.reshape(nwalkers, nao, nao)
walkers_batch_phi0 = np.random.rand(nwalkers, nao, nocc)

""" MPI CPU """
walkers_batch_phi = walkers_batch_phi0.copy()
t0 = time.time()
for iw in range(nwalkers):
    for i in range (6):
           walkers_batch_phi[iw] = VHS[iw].dot(walkers_batch_phi[iw])
t1 = time.time()
print("MPI Rank {} CPU Time: {}".format(rank, (t1 - t0)))

""" MPI GPU using for loop"""
with cp.cuda.Device(rank):
    mat = cp.array(np.random.rand(2,2))
    warmup = cp.dot(cp.array(np.random.rand(2,2)),cp.array(np.random.rand(2,2)))
    walkers_batch_phi = cp.asarray(walkers_batch_phi0.copy())
    t0 = time.time()
    VHS = cp.asarray(VHS)    
    for iw in range(nwalkers):
        for i in range (6):
               walkers_batch_phi[iw] = VHS[iw].dot(walkers_batch_phi[iw])
    t1 = time.time()
print("MPI Rank {} GPU for loop Time: {} on GPU {}".format(rank, (t1 - t0), VHS.device))


""" MPI GPU using einsum"""
with cp.cuda.Device(rank):
    mat = cp.array(np.random.rand(2,2))
    warmup = cp.einsum("ab,bc->ac", mat, mat, optimize=True)
    walkers_batch_phi = cp.asarray(walkers_batch_phi0.copy())
    t0 = time.time()
    VHS = cp.asarray(VHS).reshape(nwalkers, nao, nao)
    for i in range (6):
        walkers_batch_phi = cp.einsum("wmn,wni->wmi", VHS, walkers_batch_phi, optimize=True)
    t1 = time.time()
print("MPI Rank {} GPU einsum Time: {} on GPU {}".format(rank, (t1 - t0), VHS.device))


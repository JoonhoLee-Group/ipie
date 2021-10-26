import numpy as np
import cupy as cp
import time
from mpi4py import MPI
import sys
import os
os.environ["I_MPI_PMI_LIBRARY"] = '/cm/shared/apps/slurm/20.02.6/lib64/libpmi2.so'

divide = 5
nao = 1000 // divide
nocc = 200  // divide
naux = 4000  // divide
nwalkers = int(sys.argv[1])

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


# """ MPI CPU """
# rchol = np.random.rand(naux,nocc*nao)
# walkers_batch_Ghalf = np.random.rand(2, nwalkers, nao * nocc)
# t0 = time.time()
# vfb = rchol.dot(walkers_batch_Ghalf[0].T) + rchol.dot(walkers_batch_Ghalf[1].T)
# t1 = time.time()
# print("MPI Rank {} CPU Time: {}".format(rank, t1 - t0))


""" MPI GPU """
rchol = np.random.rand(naux,nocc*nao)
walkers_batch_Ghalf = np.random.rand(2, nwalkers, nao * nocc)
with cp.cuda.Device(rank % 2):
    warmup = cp.dot(cp.array(np.random.rand(2,2)),cp.array(np.random.rand(2,2)))
    rchol_gpu = cp.asarray(rchol)
    walkers_batch_Ghalf_gpu = cp.asarray(walkers_batch_Ghalf)
    recvbuf = cp.empty_like(walkers_batch_Ghalf_gpu)
    t0 = time.time()
    vfb = rchol_gpu.dot(walkers_batch_Ghalf_gpu[0].T) + rchol_gpu.dot(walkers_batch_Ghalf_gpu[1].T)
    t1 = time.time()
print("MPI Rank {} - CPU/GPU Time: {} on GPU {}".format(rank, t1 - t0, vfb.device))

cp.cuda.get_current_stream().synchronize()
if rank == 0:
    comm.Send(walkers_batch_Ghalf_gpu, dest=1, tag=13)
elif rank == 1:
    comm.Recv(recvbuf, source=0, tag=13)
   
with cp.cuda.Device(rank % 2):
    print("send   : ", walkers_batch_Ghalf_gpu.device, walkers_batch_Ghalf_gpu.sum())
    print("receive: ", recvbuf.device, recvbuf.sum())


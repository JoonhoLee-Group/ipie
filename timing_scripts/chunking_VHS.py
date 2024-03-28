import sys
import time

import numpy
from mpi4py import MPI

from ipie.utils.mpi import MPIHandler, get_shared_array, get_shared_comm

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

handler = MPIHandler(comm, options={"nmembers": 3}, verbose=(rank == 0))

nwalkers = 50
n = 10
nchol = n * 4
shape = (nchol, n * n)

if handler.srank == 0:
    chol = numpy.random.random(shape)
else:
    chol = None
chol = comm.bcast(chol)
chol_chunk = handler.scatter_group(chol)
chol_chunk = chol_chunk.T.copy()

xshifted = numpy.random.random((nchol, nwalkers))

chol_idxs = [i for i in range(nchol)]
chol_idxs_chunk = handler.scatter_group(chol_idxs)

senders = numpy.array([i for i in range(handler.scomm.size)])
receivers = numpy.array([i for i in range(1, handler.scomm.size)] + [0])

VHS_send = chol_chunk.dot(xshifted[chol_idxs_chunk, :])
VHS_recv = numpy.zeros_like(VHS_send)

xshifted_send = xshifted.copy()
xshifted_recv = numpy.zeros_like(xshifted)

ssize = handler.scomm.size
srank = handler.scomm.rank

for icycle in range(handler.ssize - 1):
    for isend, sender in enumerate(senders):
        if srank == isend:
            handler.scomm.Send(xshifted_send, dest=receivers[isend], tag=1)
            handler.scomm.Send(VHS_send, dest=receivers[isend], tag=2)
        elif srank == receivers[isend]:
            sender = numpy.where(receivers == srank)[0]
            handler.scomm.Recv(xshifted_recv, source=sender, tag=1)
            handler.scomm.Recv(VHS_recv, source=sender, tag=2)
    handler.scomm.barrier()
    # prepare sending
    VHS_send = VHS_recv + chol_chunk.dot(xshifted_recv[chol_idxs_chunk, :])
    xshifted_send = xshifted_recv.copy()

for isend, sender in enumerate(senders):
    if handler.scomm.rank == sender:  # sending 1 xshifted to 0 xshifted_buf
        handler.scomm.Send(VHS_send, dest=receivers[isend], tag=1)
    elif srank == receivers[isend]:
        sender = numpy.where(receivers == srank)[0]
        handler.scomm.Recv(VHS_recv, source=sender, tag=1)

VHS = VHS_recv.copy()

# if (handler.srank == 0):
VHS_ref = chol.T.dot(xshifted)
assert numpy.allclose(VHS, VHS_ref)

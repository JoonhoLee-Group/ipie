import sys
import time
import numpy
from mpi4py import MPI
from ipie.utils.mpi import get_shared_comm, get_shared_array, MPIHandler

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

handler = MPIHandler(comm,options={"nmembers":3}, verbose=(rank==0))

ssize = handler.scomm.size
srank = handler.scomm.rank

nwalkers = 40
nbsf = 100
nchol = nbsf
shape = (nchol,nbsf*nbsf)

na = 18
nb = 17

rchola = numpy.random.random((nchol,na*nbsf))
rcholb = numpy.random.random((nchol,nb*nbsf))

rchola = comm.bcast(rchola)
rcholb = comm.bcast(rcholb)

rchola_chunk = handler.scatter_group(rchola)
rcholb_chunk = handler.scatter_group(rcholb)

# distinct GF for each processor
Ghalfa = numpy.random.random((nwalkers, na*nbsf)) + 1.j * numpy.random.random((nwalkers, na*nbsf))
Ghalfb = numpy.random.random((nwalkers, nb*nbsf)) + 1.j * numpy.random.random((nwalkers, nb*nbsf))

chol_idxs = [i for i in range(nchol)]
chol_idxs_chunk = handler.scatter_group(chol_idxs)

Ghalfa_send = Ghalfa.copy()
Ghalfb_send = Ghalfb.copy()

Ghalfa_recv = numpy.zeros_like(Ghalfa)
Ghalfb_recv = numpy.zeros_like(Ghalfb)

vbias_batch_real_recv = numpy.zeros((nchol,nwalkers))
vbias_batch_imag_recv = numpy.zeros((nchol,nwalkers))

vbias_batch_real_send = numpy.zeros((nchol,nwalkers))
vbias_batch_imag_send = numpy.zeros((nchol,nwalkers))

vbias_batch_real_send[chol_idxs_chunk,:] = rchola_chunk.dot(Ghalfa.T.real) + rcholb_chunk.dot(Ghalfb.T.real)
vbias_batch_imag_send[chol_idxs_chunk,:] = rchola_chunk.dot(Ghalfa.T.imag) + rcholb_chunk.dot(Ghalfb.T.imag)

senders = handler.senders
receivers = handler.receivers

for icycle in range(handler.ssize-1):
    for isend, sender in enumerate(senders):
        if srank == isend:
            handler.scomm.Send(Ghalfa_send,dest=receivers[isend], tag=1)
            handler.scomm.Send(Ghalfb_send,dest=receivers[isend], tag=2)
            handler.scomm.Send(vbias_batch_real_send,dest=receivers[isend], tag=3)
            handler.scomm.Send(vbias_batch_imag_send,dest=receivers[isend], tag=4)
        elif srank == receivers[isend]:
            sender = numpy.where(receivers == srank)[0]
            handler.scomm.Recv(Ghalfa_recv,source=sender, tag=1)
            handler.scomm.Recv(Ghalfb_recv,source=sender, tag=2)
            handler.scomm.Recv(vbias_batch_real_recv,source=sender, tag=3)
            handler.scomm.Recv(vbias_batch_imag_recv,source=sender, tag=4)
    handler.scomm.barrier()

    # prepare sending
    vbias_batch_real_send = vbias_batch_real_recv.copy()
    vbias_batch_imag_send = vbias_batch_imag_recv.copy()
    vbias_batch_real_send[chol_idxs_chunk,:] = rchola_chunk.dot(Ghalfa_recv.T.real) + rcholb_chunk.dot(Ghalfb_recv.T.real)
    vbias_batch_imag_send[chol_idxs_chunk,:] = rchola_chunk.dot(Ghalfa_recv.T.imag) + rcholb_chunk.dot(Ghalfb_recv.T.imag)
    Ghalfa_send = Ghalfa_recv.copy()
    Ghalfb_send = Ghalfb_recv.copy()

if (len(senders)>1):
    for isend, sender in enumerate(senders):
        if (handler.scomm.rank == sender): # sending 1 xshifted to 0 xshifted_buf
            handler.scomm.Send(vbias_batch_real_send,dest=receivers[isend], tag=1)
            handler.scomm.Send(vbias_batch_imag_send,dest=receivers[isend], tag=2)
        elif srank == receivers[isend]:
            sender = numpy.where(receivers == srank)[0]
            handler.scomm.Recv(vbias_batch_real_recv,source=sender, tag=1)
            handler.scomm.Recv(vbias_batch_imag_recv,source=sender, tag=2)

# vbias_batch_real = rchola.dot(Ghalfa.T.real) + rcholb.dot(Ghalfb.T.real)
# vbias_batch_imag = rchola.dot(Ghalfa.T.imag) + rcholb.dot(Ghalfb.T.imag)
vbias_batch = vbias_batch_real_recv + 1.j * vbias_batch_imag_recv

# vbias_batch = numpy.empty((walker_batch.nwalkers, hamiltonian.nchol), dtype=Ghalfa.dtype)
# vbias_batch.real = vbias_batch_real.T.copy()
# vbias_batch.imag = vbias_batch_imag.T.copy()

vbias_batch_ref = rchola.dot(Ghalfa.T) + rcholb.dot(Ghalfb.T)
# print(vbias_batch-vbias_batch_ref)
# if (srank == 1):
#     print("rchola = {}".format(rchola))
#     print("rchola_chunk = {}".format(rchola_chunk))
#     vbias_batch_ref = rchola.dot(Ghalfa.T)# + rcholb.dot(Ghalfb.T)
#     print("vbias_batch_ref.real in {} = {}".format(srank, vbias_batch_ref.real))
#     vbias_batch_real_send = numpy.zeros((nchol,nwalkers))
#     tmp = rchola_chunk.dot(Ghalfa.T.real)# + rcholb_chunk.dot(Ghalfb.T.real)
#     print("vbias_batch_real_send.real in {} = {}".format(srank, tmp))

assert(numpy.allclose(vbias_batch, vbias_batch_ref))









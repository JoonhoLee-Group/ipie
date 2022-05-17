import sys
import time
import numpy
from mpi4py import MPI
from ipie.utils.mpi import get_shared_comm, get_shared_array, MPIHandler
from ipie.estimators.local_energy_sd import exx_kernel_batch_real_rchol, ecoul_kernel_batch_real_rchol_uhf

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

handler = MPIHandler(comm,options={"nmembers":3}, verbose=(rank==0))

ssize = handler.scomm.size
srank = handler.scomm.rank

nwalkers = 40
nbasis = 100
nchol = nbasis
shape = (nchol,nbasis*nbasis)

nalpha = 18
nbeta = 17

rchola = numpy.random.random((nchol,nalpha*nbasis))
rcholb = numpy.random.random((nchol,nbeta*nbasis))

rchola = comm.bcast(rchola)
rcholb = comm.bcast(rcholb)

rchola_chunk = handler.scatter_group(rchola)
rcholb_chunk = handler.scatter_group(rcholb)

# distinct GF for each processor
Ghalfa = numpy.random.random((nwalkers, nalpha*nbasis)) + 1.j * numpy.random.random((nwalkers, nalpha*nbasis))
Ghalfb = numpy.random.random((nwalkers, nbeta*nbasis)) + 1.j * numpy.random.random((nwalkers, nbeta*nbasis))

chol_idxs = [i for i in range(nchol)]
chol_idxs_chunk = handler.scatter_group(chol_idxs)

Ghalfa_send = Ghalfa.copy()
Ghalfb_send = Ghalfb.copy()

Ghalfa_recv = numpy.zeros_like(Ghalfa)
Ghalfb_recv = numpy.zeros_like(Ghalfb)

senders = handler.senders
receivers = handler.receivers

Ghalfa = Ghalfa.reshape(nwalkers, nalpha*nbasis)
Ghalfb = Ghalfb.reshape(nwalkers, nbeta*nbasis)
ecoul_send = ecoul_kernel_batch_real_rchol_uhf(rchola_chunk, rcholb_chunk, Ghalfa, Ghalfb)
Ghalfa = Ghalfa.reshape(nwalkers, nalpha, nbasis)
Ghalfb = Ghalfb.reshape(nwalkers, nbeta, nbasis)
exx_send = exx_kernel_batch_real_rchol(rchola_chunk, Ghalfa)
exx_send += exx_kernel_batch_real_rchol(rcholb_chunk, Ghalfb)

exx_recv = exx_send.copy()
ecoul_recv = ecoul_send.copy()

for icycle in range(handler.ssize-1):
    for isend, sender in enumerate(senders):
        if srank == isend:
            handler.scomm.Send(Ghalfa_send,dest=receivers[isend], tag=1)
            handler.scomm.Send(Ghalfb_send,dest=receivers[isend], tag=2)
            handler.scomm.Send(ecoul_send,dest=receivers[isend], tag=3)
            handler.scomm.Send(exx_send,dest=receivers[isend], tag=4)
        elif srank == receivers[isend]:
            sender = numpy.where(receivers == srank)[0]
            handler.scomm.Recv(Ghalfa_recv,source=sender, tag=1)
            handler.scomm.Recv(Ghalfb_recv,source=sender, tag=2)
            handler.scomm.Recv(ecoul_recv,source=sender, tag=3)
            handler.scomm.Recv(exx_recv,source=sender, tag=4)
    handler.scomm.barrier()

    # prepare sending
    ecoul_send = ecoul_recv.copy()
    Ghalfa_recv = Ghalfa_recv.reshape(nwalkers, nalpha*nbasis)
    Ghalfb_recv = Ghalfb_recv.reshape(nwalkers, nbeta*nbasis)
    ecoul_send += ecoul_kernel_batch_real_rchol_uhf(rchola_chunk, rcholb_chunk, Ghalfa_recv, Ghalfb_recv)
    Ghalfa_recv = Ghalfa_recv.reshape(nwalkers, nalpha, nbasis)
    Ghalfb_recv = Ghalfb_recv.reshape(nwalkers, nbeta, nbasis)
    exx_send = exx_recv.copy()
    exx_send += exx_kernel_batch_real_rchol(rchola_chunk, Ghalfa_recv)
    exx_send += exx_kernel_batch_real_rchol(rcholb_chunk, Ghalfb_recv)
    Ghalfa_send = Ghalfa_recv.copy()
    Ghalfb_send = Ghalfb_recv.copy()

if (len(senders)>1):
    for isend, sender in enumerate(senders):
        if (handler.scomm.rank == sender): # sending 1 xshifted to 0 xshifted_buf
            handler.scomm.Send(ecoul_send,dest=receivers[isend], tag=1)
            handler.scomm.Send(exx_send,dest=receivers[isend], tag=2)
        elif srank == receivers[isend]:
            sender = numpy.where(receivers == srank)[0]
            handler.scomm.Recv(ecoul_recv,source=sender, tag=1)
            handler.scomm.Recv(exx_recv,source=sender, tag=2)

Ghalfa = Ghalfa.reshape(nwalkers, nalpha*nbasis)
Ghalfb = Ghalfb.reshape(nwalkers, nbeta*nbasis)
ecoul_ref = ecoul_kernel_batch_real_rchol_uhf(rchola, rcholb, Ghalfa, Ghalfb)
Ghalfa = Ghalfa.reshape(nwalkers, nalpha, nbasis)
Ghalfb = Ghalfb.reshape(nwalkers, nbeta, nbasis)
exx_ref = exx_kernel_batch_real_rchol(rchola, Ghalfa) + exx_kernel_batch_real_rchol(rcholb, Ghalfb)
assert(numpy.allclose(ecoul_ref,ecoul_recv))
assert(numpy.allclose(exx_ref,exx_recv))


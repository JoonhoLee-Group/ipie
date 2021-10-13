import os
os.environ['MKL_NUM_THREADS'] = '1'
import cupy
import numpy
import timeit
import time
from mpi4py import MPI
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["MKL_NUM_THREADS"] = '1'

def local_energy_generic_cholesky_opt(Ghalfa, Ghalfb, rchola, rcholb):
    # Element wise multiplication.
    nalpha = Ghalfa.shape[0]
    nbeta = Ghalfb.shape[0]
    nbasis = Ghalfa.shape[-1]

    Xa = rchola.dot(Ghalfa.real.ravel()) + 1.j * rchola.dot(Ghalfa.imag.ravel())
    Xb = rcholb.dot(Ghalfb.real.ravel()) + 1.j * rcholb.dot(Ghalfb.imag.ravel())

    ecoul = cupy.dot(Xa,Xa)
    ecoul += cupy.dot(Xb,Xb)
    ecoul += 2*cupy.dot(Xa,Xb)

    GhalfaT = Ghalfa.T.copy()
    GhalfbT = Ghalfb.T.copy() # nbasis x nocc

    Ta = cupy.zeros((nalpha,nalpha), dtype=numpy.complex128)
    Tb = cupy.zeros((nbeta,nbeta), dtype=numpy.complex128)

    exx  = 0.j  # we will iterate over cholesky index to update Ex energy for alpha and beta
    for x in range(naux):  # write a cython function that calls blas for this.
        rmi_a = rchola[x].reshape((nalpha,nbasis))
        rmi_b = rcholb[x].reshape((nbeta,nbasis))
        Ta[:,:].real = rmi_a.dot(GhalfaT.real)
        Ta[:,:].imag = rmi_a.dot(GhalfaT.imag)  # this is a (nalpha, nalpha)
        Tb[:,:].real = rmi_b.dot(GhalfbT.real)
        Tb[:,:].imag = rmi_b.dot(GhalfbT.imag) # this is (nbeta, nbeta)
        exx += cupy.trace(Ta.dot(Ta)) + numpy.trace(Tb.dot(Tb))

    e2b = 0.5 * (ecoul - exx)

    return e2b

def local_energy_generic_batch(Ghalfa_batch, Ghalfb_batch, rchola, rcholb):
    # Element wise multiplication.
    nwalkers = Ghalfa_batch.shape[0]
    nalpha = Ghalfa_batch.shape[1]
    nbeta = Ghalfb_batch.shape[1]
    nbasis = Ghalfa_batch.shape[-1]

    Ghalfa_batch = Ghalfa_batch.reshape(nwalkers, nalpha*nbasis)
    Ghalfb_batch = Ghalfb_batch.reshape(nwalkers, nbeta*nbasis)

    Xa = rchola.dot(Ghalfa_batch.real.T) + 1.j * rchola.dot(Ghalfa_batch.imag.T) # naux x nwalkers
    Xb = rcholb.dot(Ghalfb_batch.real.T) + 1.j * rcholb.dot(Ghalfb_batch.imag.T) # naux x nwalkers

    ecoul = cupy.einsum("xw,xw->w", Xa, Xa, optimize=True)
    ecoul += cupy.einsum("xw,xw->w", Xb, Xb, optimize=True)
    ecoul += 2. * cupy.einsum("xw,xw->w", Xa, Xb, optimize=True)

    Ghalfa_batch = Ghalfa_batch.reshape(nwalkers, nalpha, nbasis)
    Ghalfb_batch = Ghalfb_batch.reshape(nwalkers, nbeta, nbasis)

    GhalfaT_batch = Ghalfa_batch.transpose(0,2,1).copy() # nw x nbasis x nocc
    GhalfbT_batch = Ghalfb_batch.transpose(0,2,1).copy() # nw x nbasis x nocc

    Ta = cupy.zeros((nwalkers, nalpha,nalpha), dtype=numpy.complex128)
    Tb = cupy.zeros((nwalkers, nbeta,nbeta), dtype=numpy.complex128)

    exx  = cupy.zeros(nwalkers, dtype=numpy.complex128)  # we will iterate over cholesky index to update Ex energy for alpha and beta
    for x in range(naux):  # write a cython function that calls blas for this.
        rmi_a = rchola[x].reshape((nalpha,nbasis))
        rmi_b = rcholb[x].reshape((nbeta,nbasis))
        Ta[:,:,:].real = rmi_a.dot(GhalfaT_batch.real).transpose(1,0,2)
        Ta[:,:,:].imag = rmi_a.dot(GhalfaT_batch.imag).transpose(1,0,2)
        Tb[:,:,:].real = rmi_b.dot(GhalfbT_batch.real).transpose(1,0,2)
        Tb[:,:,:].imag = rmi_b.dot(GhalfbT_batch.imag).transpose(1,0,2)

        exx += cupy.einsum("wij,wji->w",Ta,Ta,optimize=True) + numpy.einsum("wij,wji->w",Tb,Tb,optimize=True)

    e2b = 0.5 * (ecoul - exx)

    return e2b

divide = 5
nao = 439 // divide
nocca = 94 // divide
noccb = 92 // divide
naux = 3468 // divide
nwalkers = 50
nblocks = 5

Ghalfa_batch = numpy.random.randn(nwalkers * nao * nocca).reshape(nwalkers, nocca,nao) + 1.j * numpy.random.randn(nwalkers * nao * nocca).reshape(nwalkers, nocca, nao)
Ghalfb_batch = numpy.random.randn(nwalkers * nao * noccb).reshape(nwalkers, noccb,nao)+ 1.j * numpy.random.randn(nwalkers * nao * noccb).reshape(nwalkers, noccb, nao)
rchola = numpy.random.randn(naux * nao * nocca).reshape(naux, nocca*nao)
rcholb = numpy.random.randn(naux * nao * noccb).reshape(naux, noccb*nao)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
with cupy.cuda.Device(rank):
    mat = cupy.array(numpy.random.rand(2,2))
    warmup = cupy.einsum("ab,bc->ac", mat, mat, optimize=True)

    Ghalfa_batch = cupy.asarray(Ghalfa_batch, dtype=cupy.complex128)
    Ghalfb_batch = cupy.asarray(Ghalfb_batch, dtype=cupy.complex128)
    rchola = cupy.asarray(rchola, dtype=cupy.float64)
    rcholb = cupy.asarray(rcholb, dtype=cupy.float64)

    start = time.time()
    energies = cupy.zeros(nwalkers, dtype=cupy.complex128)
    for i in range (nblocks):
        for iw in range (nwalkers):
            energies[iw] = local_energy_generic_cholesky_opt(Ghalfa_batch[iw], Ghalfb_batch[iw], rchola, rcholb)
    print("Current algorithm = {}".format(time.time()-start))

    start = time.time()
    for i in range (nblocks):
        energies_batch = local_energy_generic_batch(Ghalfa_batch, Ghalfb_batch, rchola, rcholb)
    print("batch algorithm = {}".format(time.time()-start))

    assert(cupy.allclose(energies,energies_batch))





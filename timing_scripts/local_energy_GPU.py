import os

os.environ["MKL_NUM_THREADS"] = "1"
import math
import time
import timeit

import cupy
import numpy
from mpi4py import MPI
from numba import cuda

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def local_energy_generic_cholesky_opt(Ghalfa, Ghalfb, rchola, rcholb):
    # Element wise multiplication.
    nalpha = Ghalfa.shape[0]
    nbeta = Ghalfb.shape[0]
    nbasis = Ghalfa.shape[-1]

    Xa = rchola.dot(Ghalfa.real.ravel()) + 1.0j * rchola.dot(Ghalfa.imag.ravel())
    Xb = rcholb.dot(Ghalfb.real.ravel()) + 1.0j * rcholb.dot(Ghalfb.imag.ravel())

    ecoul = cupy.dot(Xa, Xa)
    ecoul += cupy.dot(Xb, Xb)
    ecoul += 2 * cupy.dot(Xa, Xb)

    GhalfaT = Ghalfa.T.copy()
    GhalfbT = Ghalfb.T.copy()  # nbasis x nocc

    Ta = cupy.zeros((nalpha, nalpha), dtype=numpy.complex128)
    Tb = cupy.zeros((nbeta, nbeta), dtype=numpy.complex128)

    exx = 0.0j  # we will iterate over cholesky index to update Ex energy for alpha and beta
    for x in range(naux):  # write a cython function that calls blas for this.
        rmi_a = rchola[x].reshape((nalpha, nbasis))
        rmi_b = rcholb[x].reshape((nbeta, nbasis))
        Ta[:, :].real = rmi_a.dot(GhalfaT.real)
        Ta[:, :].imag = rmi_a.dot(GhalfaT.imag)  # this is a (nalpha, nalpha)
        Tb[:, :].real = rmi_b.dot(GhalfbT.real)
        Tb[:, :].imag = rmi_b.dot(GhalfbT.imag)  # this is (nbeta, nbeta)
        exx += cupy.trace(Ta.dot(Ta)) + numpy.trace(Tb.dot(Tb))

    e2b = 0.5 * (ecoul - exx)

    return e2b


def local_energy_generic_batch_old(Ghalfa_batch, Ghalfb_batch, rchola, rcholb):
    # Element wise multiplication.
    nwalkers = Ghalfa_batch.shape[0]
    nalpha = Ghalfa_batch.shape[1]
    nbeta = Ghalfb_batch.shape[1]
    nbasis = Ghalfa_batch.shape[-1]

    Ghalfa_batch = Ghalfa_batch.reshape(nwalkers, nalpha * nbasis)
    Ghalfb_batch = Ghalfb_batch.reshape(nwalkers, nbeta * nbasis)

    Xa = rchola.dot(Ghalfa_batch.real.T) + 1.0j * rchola.dot(Ghalfa_batch.imag.T)  # naux x nwalkers
    Xb = rcholb.dot(Ghalfb_batch.real.T) + 1.0j * rcholb.dot(Ghalfb_batch.imag.T)  # naux x nwalkers

    ecoul = cupy.einsum("xw,xw->w", Xa, Xa, optimize=True)
    ecoul += cupy.einsum("xw,xw->w", Xb, Xb, optimize=True)
    ecoul += 2.0 * cupy.einsum("xw,xw->w", Xa, Xb, optimize=True)

    Ghalfa_batch = Ghalfa_batch.reshape(nwalkers, nalpha, nbasis)
    Ghalfb_batch = Ghalfb_batch.reshape(nwalkers, nbeta, nbasis)

    GhalfaT_batch = Ghalfa_batch.transpose(0, 2, 1).copy()  # nw x nbasis x nocc
    GhalfbT_batch = Ghalfb_batch.transpose(0, 2, 1).copy()  # nw x nbasis x nocc

    GhalfaT_batch_real = GhalfaT_batch.real.copy()
    GhalfaT_batch_imag = GhalfaT_batch.imag.copy()
    GhalfbT_batch_real = GhalfbT_batch.real.copy()
    GhalfbT_batch_imag = GhalfbT_batch.imag.copy()

    Ta = cupy.zeros((nwalkers, nalpha, nalpha), dtype=numpy.complex128)
    Tb = cupy.zeros((nwalkers, nbeta, nbeta), dtype=numpy.complex128)

    exx = cupy.zeros(
        nwalkers, dtype=numpy.complex128
    )  # we will iterate over cholesky index to update Ex energy for alpha and beta
    for x in range(naux):  # write a cython function that calls blas for this.
        rmi_a = rchola[x].reshape((nalpha, nbasis))
        rmi_b = rcholb[x].reshape((nbeta, nbasis))
        Ta[:, :, :].real = rmi_a.dot(GhalfaT_batch_real).transpose(1, 0, 2)
        Ta[:, :, :].imag = rmi_a.dot(GhalfaT_batch_imag).transpose(1, 0, 2)
        Tb[:, :, :].real = rmi_b.dot(GhalfbT_batch_real).transpose(1, 0, 2)
        Tb[:, :, :].imag = rmi_b.dot(GhalfbT_batch_imag).transpose(1, 0, 2)

        exx += cupy.einsum("wij,wji->w", Ta, Ta, optimize=True) + cupy.einsum(
            "wij,wji->w", Tb, Tb, optimize=True
        )

    e2b = 0.5 * (ecoul - exx)

    return e2b


def local_energy_generic_batch(Ghalfa_batch, Ghalfb_batch, rchola, rcholb):
    # Element wise multiplication.
    nwalkers = Ghalfa_batch.shape[0]
    nalpha = Ghalfa_batch.shape[1]
    nbeta = Ghalfb_batch.shape[1]
    nbasis = Ghalfa_batch.shape[-1]

    Ghalfa_batch = Ghalfa_batch.reshape(nwalkers, nalpha * nbasis)
    Ghalfb_batch = Ghalfb_batch.reshape(nwalkers, nbeta * nbasis)

    Xa = rchola.dot(Ghalfa_batch.real.T) + 1.0j * rchola.dot(Ghalfa_batch.imag.T)  # naux x nwalkers
    Xb = rcholb.dot(Ghalfb_batch.real.T) + 1.0j * rcholb.dot(Ghalfb_batch.imag.T)  # naux x nwalkers

    ecoul = cupy.einsum("xw,xw->w", Xa, Xa, optimize=True)
    ecoul += cupy.einsum("xw,xw->w", Xb, Xb, optimize=True)
    ecoul += 2.0 * cupy.einsum("xw,xw->w", Xa, Xb, optimize=True)

    Ghalfa_batch = Ghalfa_batch.reshape(nwalkers, nalpha, nbasis)
    Ghalfb_batch = Ghalfb_batch.reshape(nwalkers, nbeta, nbasis)

    # Ghalfa_batch_real = Ghalfa_batch.real.copy()
    # Ghalfa_batch_imag = Ghalfa_batch.imag.copy()
    # Ghalfb_batch_real = Ghalfb_batch.real.copy()
    # Ghalfb_batch_imag = Ghalfb_batch.imag.copy()

    Ta = cupy.zeros((nwalkers, nalpha, nalpha), dtype=numpy.complex128)
    Tb = cupy.zeros((nwalkers, nbeta, nbeta), dtype=numpy.complex128)

    exx = cupy.zeros(
        nwalkers, dtype=numpy.complex128
    )  # we will iterate over cholesky index to update Ex energy for alpha and beta
    for x in range(naux):  # write a cython function that calls blas for this.
        rmi_a = rchola[x].reshape((nalpha, nbasis))
        rmi_b = rcholb[x].reshape((nbeta, nbasis))
        Ta = Ghalfa_batch @ rmi_a.T
        Tb = Ghalfb_batch @ rmi_b.T

        exx += cupy.einsum("wij,wji->w", Ta, Ta, optimize=True) + cupy.einsum(
            "wij,wji->w", Tb, Tb, optimize=True
        )

    e2b = 0.5 * (ecoul - exx)

    return e2b


@cuda.jit
def exx_numba(exx_chol, Ta, Tb, Ghalfa_batch, Ghalfb_batch, rchola, rcholb):
    nwalkers = Ghalfa_batch.shape[0]
    nalpha = Ghalfa_batch.shape[1]
    nbeta = Ghalfb_batch.shape[1]
    nbasis = Ghalfa_batch.shape[-1]
    nchol = rchola.shape[0]
    pos = cuda.grid(1)
    pos1 = pos // nchol
    pos2 = pos - pos1 * nchol

    if pos1 < nwalkers and pos2 < nchol:
        rmi_a = rchola[pos2]
        rmi_b = rcholb[pos2]
        Ta = rmi_a.dot(Ghalfa_batch[pos1])
        Tb = rmi_b.dot(Ghalfb_batch[pos1])
        # for i in range(nalpha):
        #    for j in range(nalpha):
        #        exx_chol[pos1,pos2] = Ta[i,j]*Ta[j,i]
        # for i in range(nalpha):
        #    for j in range(nalpha):
        #        exx_chol[pos1,pos2] = Tb[i,j]*Tb[j,i]


def local_energy_generic_numba(Ghalfa_batch, Ghalfb_batch, rchola, rcholb):
    # Element wise multiplication.
    nwalkers = Ghalfa_batch.shape[0]
    nalpha = Ghalfa_batch.shape[1]
    nbeta = Ghalfb_batch.shape[1]
    nbasis = Ghalfa_batch.shape[-1]
    nchol = rchola.shape[0]

    Ghalfa_batch = Ghalfa_batch.reshape(nwalkers, nalpha * nbasis)
    Ghalfb_batch = Ghalfb_batch.reshape(nwalkers, nbeta * nbasis)

    Xa = rchola.dot(Ghalfa_batch.real.T) + 1.0j * rchola.dot(Ghalfa_batch.imag.T)  # naux x nwalkers
    Xb = rcholb.dot(Ghalfb_batch.real.T) + 1.0j * rcholb.dot(Ghalfb_batch.imag.T)  # naux x nwalkers

    ecoul = cupy.einsum("xw,xw->w", Xa, Xa, optimize=True)
    ecoul += cupy.einsum("xw,xw->w", Xb, Xb, optimize=True)
    ecoul += 2.0 * cupy.einsum("xw,xw->w", Xa, Xb, optimize=True)

    Ghalfa_batch = Ghalfa_batch.reshape(nwalkers, nalpha, nbasis)
    Ghalfb_batch = Ghalfb_batch.reshape(nwalkers, nbeta, nbasis)
    Ta = cupy.zeros((nwalkers, nalpha, nalpha), dtype=numpy.float64)
    Tb = cupy.zeros((nwalkers, nbeta, nbeta), dtype=numpy.float64)

    Ghalfa_batch_real = Ghalfa_batch.real.copy()
    Ghalfa_batch_imag = Ghalfa_batch.imag.copy()
    Ghalfb_batch_real = Ghalfb_batch.real.copy()
    Ghalfb_batch_imag = Ghalfb_batch.imag.copy()

    rchola = rchola.reshape(nchol, nalpha, nbasis)
    rcholb = rcholb.reshape(nchol, nbeta, nbasis)

    exx_chol = cupy.zeros(
        (nwalkers, nchol), dtype=numpy.float64
    )  # we will iterate over cholesky index to update Ex energy for alpha and beta
    threadsperblock = 512
    blockspergrid = math.ceil(nchol * nwalkers / threadsperblock)
    exx_numba[blockspergrid, threadsperblock](
        exx_chol, Ta, Tb, Ghalfa_batch_real, Ghalfb_batch_real, rchola, rcholb
    )
    exx = cupy.einsum("wx->w", exx_chol)
    # nchol = rchola.shape[0]
    # rchola = rchola.reshape(nchol, nalpha, nbasis)
    # rcholb = rcholb.reshape(nchol, nbeta, nbasis)

    # Txij = cupy.einsum("xim,wjm->wxji", rchola, Ghalfa_batch)
    # exx = cupy.einsum("wxji,wxij->w",Txij,Txij)
    # Txij = cupy.einsum("xim,wjm->wxji", rcholb, Ghalfb_batch)
    # exx += cupy.einsum("wxji,wxij->w",Txij,Txij)

    # exx = cupy.einsum("xim,xjn,win,wjm->w",rchola, rchola, Ghalfa_batch, Ghalfa_batch, optimize=True)\
    #    + cupy.einsum("xim,xjn,win,wjm->w",rcholb, rcholb, Ghalfb_batch, Ghalfb_batch, optimize=True)

    e2b = 0.5 * (ecoul - exx)

    return e2b


divide = 5
nao = 439 // divide
nocca = 94 // divide
noccb = 92 // divide
naux = 3468 // divide
nwalkers = 50
nblocks = 5

Ghalfa_batch = numpy.random.randn(nwalkers * nao * nocca).reshape(
    nwalkers, nocca, nao
) + 1.0j * numpy.random.randn(nwalkers * nao * nocca).reshape(nwalkers, nocca, nao)
Ghalfb_batch = numpy.random.randn(nwalkers * nao * noccb).reshape(
    nwalkers, noccb, nao
) + 1.0j * numpy.random.randn(nwalkers * nao * noccb).reshape(nwalkers, noccb, nao)
rchola = numpy.random.randn(naux * nao * nocca).reshape(naux, nocca * nao)
rcholb = numpy.random.randn(naux * nao * noccb).reshape(naux, noccb * nao)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
nchol = naux
nalpha = nocca
nbeta = noccb
nbasis = nao
# Ghalfa_batch = Ghalfa_batch.reshape(nwalkers, nalpha, nbasis)
# Ghalfb_batch = Ghalfb_batch.reshape(nwalkers, nbeta, nbasis)
# rchola = rchola.reshape(nchol, nalpha, nbasis)
# rcholb = rcholb.reshape(nchol, nbeta, nbasis)
# path_info = numpy.einsum_path("xim,xjn,win,wjm->w",rcholb, rcholb, Ghalfb_batch, Ghalfb_batch, optimize='greedy')
# print(path_info[0])
# print(path_info[1])
# path_info = numpy.einsum_path("xim,wjm->wxij",rcholb, Ghalfb_batch, optimize='greedy')
# print(path_info[0])
# print(path_info[1])
# Txij = numpy.zeros((nwalkers,nchol,noccb,noccb))
# path_info = numpy.einsum_path("wxij,wxij->w",Txij,Txij, optimize='greedy')
# print(path_info[0])
# print(path_info[1])
# Txij = numpy.zeros((nwalkers,nchol,noccb,noccb))
# print(Txij.size)
# exit()
with cupy.cuda.Device(rank):
    mat = cupy.array(numpy.random.rand(2, 2))
    warmup = cupy.einsum("ab,bc->ac", mat, mat, optimize=True)

    Ghalfa_batch = cupy.asarray(Ghalfa_batch, dtype=cupy.complex128)
    Ghalfb_batch = cupy.asarray(Ghalfb_batch, dtype=cupy.complex128)
    rchola = cupy.asarray(rchola, dtype=cupy.float64)
    rcholb = cupy.asarray(rcholb, dtype=cupy.float64)

    # energies_batch = local_energy_generic_batch_old(Ghalfa_batch, Ghalfb_batch, rchola, rcholb)
    # start = time.time()
    # for i in range (nblocks):
    #    energies_batch = local_energy_generic_batch_old(Ghalfa_batch, Ghalfb_batch, rchola, rcholb)
    # print("Current algorithm = {}".format(time.time()-start))

    # energies_batch2 = local_energy_generic_batch(Ghalfa_batch, Ghalfb_batch, rchola, rcholb)
    # start = time.time()
    # for i in range (nblocks):
    #    energies_batch2 = local_energy_generic_batch(Ghalfa_batch, Ghalfb_batch, rchola, rcholb)
    # print("new batch algorithm = {}".format(time.time()-start))

    energies_batch3 = local_energy_generic_numba(Ghalfa_batch, Ghalfb_batch, rchola, rcholb)
    start = time.time()
    for i in range(nblocks):
        energies_batch3 = local_energy_generic_numba(Ghalfa_batch, Ghalfb_batch, rchola, rcholb)
    print("new einsum algorithm = {}".format(time.time() - start))

    assert cupy.allclose(energies_batch, energies_batch2)
    assert cupy.allclose(energies_batch3, energies_batch2)

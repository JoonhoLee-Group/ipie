import os

os.environ["MKL_NUM_THREADS"] = "1"
import time
import timeit

import numpy


def local_energy_generic_cholesky_opt(Ghalfa, Ghalfb, rchola, rcholb):
    # Element wise multiplication.
    nalpha = Ghalfa.shape[0]
    nbeta = Ghalfb.shape[0]
    nbasis = Ghalfa.shape[-1]

    Xa = rchola.dot(Ghalfa.real.ravel()) + 1.0j * rchola.dot(Ghalfa.imag.ravel())
    Xb = rcholb.dot(Ghalfb.real.ravel()) + 1.0j * rcholb.dot(Ghalfb.imag.ravel())

    ecoul = numpy.dot(Xa, Xa)
    ecoul += numpy.dot(Xb, Xb)
    ecoul += 2 * numpy.dot(Xa, Xb)

    GhalfaT = Ghalfa.T.copy()
    GhalfbT = Ghalfb.T.copy()  # nbasis x nocc

    Ta = numpy.zeros((nalpha, nalpha), dtype=numpy.complex128)
    Tb = numpy.zeros((nbeta, nbeta), dtype=numpy.complex128)

    exx = 0.0j  # we will iterate over cholesky index to update Ex energy for alpha and beta
    for x in range(naux):  # write a cython function that calls blas for this.
        rmi_a = rchola[x].reshape((nalpha, nbasis))
        rmi_b = rcholb[x].reshape((nbeta, nbasis))
        Ta[:, :].real = rmi_a.dot(GhalfaT.real)
        Ta[:, :].imag = rmi_a.dot(GhalfaT.imag)  # this is a (nalpha, nalpha)
        Tb[:, :].real = rmi_b.dot(GhalfbT.real)
        Tb[:, :].imag = rmi_b.dot(GhalfbT.imag)  # this is (nbeta, nbeta)
        exx += numpy.trace(Ta.dot(Ta)) + numpy.trace(Tb.dot(Tb))

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

    ecoul = numpy.einsum("xw,xw->w", Xa, Xa, optimize=True)
    ecoul += numpy.einsum("xw,xw->w", Xb, Xb, optimize=True)
    ecoul += 2.0 * numpy.einsum("xw,xw->w", Xa, Xb, optimize=True)

    Ghalfa_batch = Ghalfa_batch.reshape(nwalkers, nalpha, nbasis)
    Ghalfb_batch = Ghalfb_batch.reshape(nwalkers, nbeta, nbasis)

    GhalfaT_batch = Ghalfa_batch.transpose(0, 2, 1).copy()  # nw x nbasis x nocc
    GhalfbT_batch = Ghalfb_batch.transpose(0, 2, 1).copy()  # nw x nbasis x nocc

    Ta = numpy.zeros((nwalkers, nalpha, nalpha), dtype=numpy.complex128)
    Tb = numpy.zeros((nwalkers, nbeta, nbeta), dtype=numpy.complex128)

    exx = numpy.zeros(
        nwalkers, dtype=numpy.complex128
    )  # we will iterate over cholesky index to update Ex energy for alpha and beta
    for x in range(naux):  # write a cython function that calls blas for this.
        rmi_a = rchola[x].reshape((nalpha, nbasis))
        rmi_b = rcholb[x].reshape((nbeta, nbasis))
        Ta[:, :, :].real = rmi_a.dot(GhalfaT_batch.real).transpose(1, 0, 2)
        Ta[:, :, :].imag = rmi_a.dot(GhalfaT_batch.imag).transpose(1, 0, 2)
        Tb[:, :, :].real = rmi_b.dot(GhalfbT_batch.real).transpose(1, 0, 2)
        Tb[:, :, :].imag = rmi_b.dot(GhalfbT_batch.imag).transpose(1, 0, 2)

        exx += numpy.einsum("wij,wji->w", Ta, Ta, optimize=True) + numpy.einsum(
            "wij,wji->w", Tb, Tb, optimize=True
        )

    e2b = 0.5 * (ecoul - exx)

    return e2b


nwalkers = 20

nao = 200
nocc = 50
naux = nao * 4
Ghalfa_batch = numpy.random.randn(nwalkers * nao * nocc).reshape(
    nwalkers, nocc, nao
) + 1.0j * numpy.random.randn(nwalkers * nao * nocc).reshape(nwalkers, nocc, nao)
Ghalfb_batch = numpy.random.randn(nwalkers * nao * nocc).reshape(
    nwalkers, nocc, nao
) + 1.0j * numpy.random.randn(nwalkers * nao * nocc).reshape(nwalkers, nocc, nao)
rchola = numpy.random.randn(naux * nao * nocc).reshape(naux, nocc * nao)
rcholb = numpy.random.randn(naux * nao * nocc).reshape(naux, nocc * nao)
energies = []

start = time.time()
for iw in range(nwalkers):
    energies += [
        local_energy_generic_cholesky_opt(Ghalfa_batch[iw], Ghalfb_batch[iw], rchola, rcholb)
    ]
energies = numpy.complex128(energies)
print("Current algorithm = {}".format(time.time() - start))

start = time.time()
energies_batch = local_energy_generic_batch(Ghalfa_batch, Ghalfb_batch, rchola, rcholb)
print("batch algorithm = {}".format(time.time() - start))

assert numpy.allclose(energies, energies_batch)

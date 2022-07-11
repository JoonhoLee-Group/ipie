import os

import numpy

# numpy.show_config()
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import time

import numba
from numba import jit
from pyscf.lib.numpy_helper import zdot


def local_energy_generic_cholesky_opt_rhf(Ghalfa, rchola):
    # Element wise multiplication.
    nalpha = Ghalfa.shape[0]
    nbasis = Ghalfa.shape[1]

    GhalfaT = Ghalfa.T.copy()
    
    Ta = numpy.zeros((nalpha,nalpha), dtype=numpy.complex128)

    exx  = 0.j  # we will iterate over cholesky index to update Ex energy for alpha and beta
    for rmi_a in rchola:
        rmi_a = rmi_a.reshape((nalpha,nbasis))
        Ta[:,:].real = rmi_a.dot(GhalfaT.real) 
        Ta[:,:].imag = rmi_a.dot(GhalfaT.imag)  # this is a (nalpha, nalpha)
        exx += numpy.einsum("ij,ji->", Ta, Ta) * 2.0

    e2b = -0.5 * exx

    return e2b

@jit(nopython=True,fastmath=True)
def local_energy_numba(rchola, Ghalfa_batch):
    naux = rchola.shape[0]
    nwalkers = Ghalfa_batch.shape[0]
    nocc = Ghalfa_batch.shape[1]

    T = numpy.zeros((nocc,nocc), dtype=numpy.complex128)
    exx = numpy.zeros((nwalkers), dtype=numpy.complex128)
    for iw in range(nwalkers):
        Greal = Ghalfa_batch[iw].real.copy()
        Gimag = Ghalfa_batch[iw].imag.copy()
        for jx in range(naux):
            T = rchola[jx].dot(Greal.T) + 1.j * rchola[jx].dot(Gimag.T)
            exx[iw] += numpy.dot(T.ravel(), T.T.ravel())
    return exx

for nmult in [1,2,3,4,5,6]:
    nao = 108 * nmult
    nocc = 15 * nmult
    nwalkers = 50
    naux = 693 * nmult
    
    rchola = numpy.random.randn(naux * nao * nocc).reshape(naux, nocc,nao)
    T = numpy.zeros((nocc,nocc), dtype=numpy.complex128)
    exx = numpy.zeros((nwalkers), dtype=numpy.complex128)
    Ghalfa_batch = numpy.random.randn(nwalkers * nao * nocc).reshape(nwalkers, nocc, nao) + 1.j * numpy.random.randn(nwalkers * nao * nocc).reshape(nwalkers, nocc, nao)
    
    start = time.time()
    
    # for iw in range(nwalkers):
    # 	exx[iw] = local_energy_generic_cholesky_opt_rhf(Ghalfa_batch[iw], rchola)
    # print("{}: new algorithm (RHF) = {}".format(nmult, time.time()-start))
    
    if nmult == 1:
	    start = time.time()
	    exx = local_energy_numba(rchola, Ghalfa_batch)
	    # print("Numba (w. compilation) = {}".format(time.time()-start))
    
    start = time.time()
    exx = local_energy_numba(rchola, Ghalfa_batch)
    print("{}: Numba = {}".format(nmult,time.time()-start))




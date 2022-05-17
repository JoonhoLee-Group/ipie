import os
from line_profiler import LineProfiler

os.environ['MKL_NUM_THREADS'] = '1'
import numpy
import timeit
import time
from ipie.estimators.opt_local_energy import local_energy_generic_cholesky_exx, local_energy_generic_cholesky_exx_batch, local_energy_generic_cholesky_exx_rhf_batch

def local_energy_generic_cholesky_opt(Ghalfa, Ghalfb, rchola, rcholb):
    # Element wise multiplication.
    nalpha = Ghalfa.shape[0]
    nbeta = Ghalfb.shape[0]
    nbasis = Ghalfa.shape[-1]

    GhalfaT = Ghalfa.T.copy()
    GhalfbT = Ghalfb.T.copy() # nbasis x nocc
    
    Ta = numpy.zeros((nalpha,nalpha), dtype=numpy.complex128)
    Tb = numpy.zeros((nbeta,nbeta), dtype=numpy.complex128)

    exx  = 0.j  # we will iterate over cholesky index to update Ex energy for alpha and beta
    for x in range(naux):  # write a cython function that calls blas for this.
        rmi_a = rchola[x].reshape((nalpha,nbasis))
        rmi_b = rcholb[x].reshape((nbeta,nbasis))
        Ta[:,:].real = rmi_a.dot(GhalfaT.real) 
        Ta[:,:].imag = rmi_a.dot(GhalfaT.imag)  # this is a (nalpha, nalpha)
        Tb[:,:].real = rmi_b.dot(GhalfbT.real) 
        Tb[:,:].imag = rmi_b.dot(GhalfbT.imag) # this is (nbeta, nbeta)
        exx += numpy.trace(Ta.dot(Ta)) + numpy.trace(Tb.dot(Tb))

    # e2b = 0.5 * (ecoul - exx)
    e2b = -0.5 * exx

    return e2b

def local_energy_generic_cholesky_opt_cubic_rhf(Ghalfa, Ghalfb, rchola, rcholb):
    nalpha = Ghalfa.shape[0]
    nbeta = Ghalfb.shape[0]
    nbasis = Ghalfa.shape[-1]

    GhalfaT = Ghalfa.T.copy()
    # GhalfbT = Ghalfb.T.copy() # nbasis x nocc
    
    Ta = numpy.zeros((naux, nalpha,nalpha), dtype=numpy.complex128)
    # Tb = numpy.zeros((naux, nbeta,nbeta), dtype=numpy.complex128)

    rchola = rchola.reshape((naux,nalpha,nbasis))
    # rcholb = rcholb.reshape((naux,nbeta,nbasis))

    Ta.real = numpy.einsum("xim,mj->xij",rchola,GhalfaT.real, optimize=True)
    Ta.imag = numpy.einsum("xim,mj->xij",rchola,GhalfaT.imag, optimize=True)
    # Tb.real = numpy.einsum("xim,mj->xij",rcholb,GhalfbT.real, optimize=True)
    # Tb.imag = numpy.einsum("xim,mj->xij",rcholb,GhalfbT.imag, optimize=True)
    
    # exxa = numpy.tensordot(Ta, Ta, axes=((0,1,2),(0,2,1)))
    # exxb = numpy.tensordot(Tb, Tb, axes=((0,1,2),(0,2,1)))
    exxa = numpy.einsum("xij,xji->",Ta, Ta, optimize=True)
    # exxb = numpy.einsum("xij,xji->",Tb, Tb, optimize=True)

    rchola = rchola.reshape((naux,nalpha*nbasis))
    # rcholb = rcholb.reshape((naux,nbeta*nbasis))

    # return -0.5 *(exxa+exxb)
    return -0.5 *(exxa)

def local_energy_generic_cholesky_opt_new(Ghalfa, Ghalfb, rchola, rcholb):
    # Element wise multiplication.
    nalpha = Ghalfa.shape[0]
    nbeta = Ghalfb.shape[0]
    nbasis = Ghalfa.shape[-1]

    GhalfaT = Ghalfa.T.copy()
    GhalfbT = Ghalfb.T.copy() # nbasis x nocc
    
    Ta = numpy.zeros((nalpha,nalpha), dtype=numpy.complex128)
    Tb = numpy.zeros((nbeta,nbeta), dtype=numpy.complex128)

    exx  = 0.j  # we will iterate over cholesky index to update Ex energy for alpha and beta
    for x in range(naux):  # write a cython function that calls blas for this.
        rmi_a = rchola[x].reshape((nalpha,nbasis))
        rmi_b = rcholb[x].reshape((nbeta,nbasis))
        Ta[:,:].real = rmi_a.dot(GhalfaT.real) 
        Ta[:,:].imag = rmi_a.dot(GhalfaT.imag)  # this is a (nalpha, nalpha)
        Tb[:,:].real = rmi_b.dot(GhalfbT.real) 
        Tb[:,:].imag = rmi_b.dot(GhalfbT.imag) # this is (nbeta, nbeta)
        exx += numpy.einsum("ij,ji->", Ta, Ta) + numpy.einsum("ij,ji->", Tb, Tb)

    # e2b = 0.5 * (ecoul - exx)
    e2b = -0.5 * exx

    return e2b

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

def local_energy_generic_cholesky_opt_rhf_batch(Ghalfa_batch, rchola):
    # Element wise multiplication.
    nwalkers = Ghalfa_batch.shape[0]
    nalpha = Ghalfa_batch.shape[1]
    nbasis = Ghalfa_batch.shape[-1]
    nchol = rchola.shape[0]

    Ghalfa_batch = Ghalfa_batch.reshape(nwalkers, nalpha, nbasis)
    GhalfaT_batch = Ghalfa_batch.transpose(0,2,1).copy() # nw x nbasis x nocc

    Ta = numpy.zeros((nalpha,nalpha), dtype=numpy.complex128)
    exx = numpy.zeros((nwalkers), dtype=numpy.complex128)

    for x in range(nchol):
        rmi_a = rchola[x].reshape((nalpha,nbasis))
        for iw in range(nwalkers):
            Ta[:,:].real = rmi_a.dot(GhalfaT_batch[iw].real)
            Ta[:,:].imag = rmi_a.dot(GhalfaT_batch[iw].imag)
            exx[iw] += 2.*numpy.einsum("ij,ji->",Ta,Ta)

    e2b = -0.5 * exx

    return e2b

def local_energy_generic_batch(Ghalfa_batch, Ghalfb_batch, rchola, rcholb):
    # Element wise multiplication.
    nwalkers = Ghalfa_batch.shape[0]
    nalpha = Ghalfa_batch.shape[1]
    nbeta = Ghalfb_batch.shape[1]
    nbasis = Ghalfa_batch.shape[-1]

    Ghalfa_batch = Ghalfa_batch.reshape(nwalkers, nalpha, nbasis)
    Ghalfb_batch = Ghalfb_batch.reshape(nwalkers, nbeta, nbasis)

    GhalfaT_batch = Ghalfa_batch.transpose(0,2,1).copy() # nw x nbasis x nocc
    GhalfbT_batch = Ghalfb_batch.transpose(0,2,1).copy() # nw x nbasis x nocc
    
    Ta = numpy.zeros((nwalkers, nalpha,nalpha), dtype=numpy.complex128)
    Tb = numpy.zeros((nwalkers, nbeta,nbeta), dtype=numpy.complex128)

    exx  = numpy.zeros(nwalkers, dtype=numpy.complex128)  # we will iterate over cholesky index to update Ex energy for alpha and beta
    for x in range(naux):  # write a cython function that calls blas for this.
        rmi_a = rchola[x].reshape((nalpha,nbasis))
        rmi_b = rcholb[x].reshape((nbeta,nbasis))
        Ta[:,:,:].real = rmi_a.dot(GhalfaT_batch.real).transpose(1,0,2)
        Ta[:,:,:].imag = rmi_a.dot(GhalfaT_batch.imag).transpose(1,0,2)
        Tb[:,:,:].real = rmi_b.dot(GhalfbT_batch.real).transpose(1,0,2)
        Tb[:,:,:].imag = rmi_b.dot(GhalfbT_batch.imag).transpose(1,0,2)

        exx += numpy.einsum("wij,wji->w",Ta,Ta,optimize=True) + numpy.einsum("wij,wji->w",Tb,Tb,optimize=True) 

    # e2b = 0.5 * (ecoul - exx)
    e2b = -0.5 * exx

    return e2b

nwalkers = 50

nmult = 1
nao = 108 * nmult
nocc = 15 * nmult
naux = 693 * nmult
Ghalfa_batch = numpy.random.randn(nwalkers * nao * nocc).reshape(nwalkers, nocc,nao) + 1.j * numpy.random.randn(nwalkers * nao * nocc).reshape(nwalkers, nocc, nao)
Ghalfb_batch = numpy.random.randn(nwalkers * nao * nocc).reshape(nwalkers, nocc,nao)+ 1.j * numpy.random.randn(nwalkers * nao * nocc).reshape(nwalkers, nocc, nao)
rchola = numpy.random.randn(naux * nao * nocc).reshape(naux, nocc*nao)
rcholb = numpy.random.randn(naux * nao * nocc).reshape(naux, nocc*nao)
energies = []
energies2 = []
energies3 = []
energies4 = []
energies5 = []
energies6 = []

# start = time.time()
# for iw in range (nwalkers):
#     energies += [local_energy_generic_cholesky_opt(Ghalfa_batch[iw], Ghalfb_batch[iw], rchola, rcholb)]
# energies = numpy.complex128(energies)
# print("Current algorithm = {}".format(time.time()-start))

# start = time.time()
# for iw in range (nwalkers):
#     energies2 += [local_energy_generic_cholesky_opt_new(Ghalfa_batch[iw], Ghalfb_batch[iw], rchola, rcholb)]
# energies2 = numpy.complex128(energies2)
# print("new algorithm = {}".format(time.time()-start))

start = time.time()
for iw in range (nwalkers):
    energies3 += [local_energy_generic_cholesky_opt_rhf(Ghalfa_batch[iw], rchola)]
energies3 = numpy.complex128(energies3)
print("new algorithm (RHF) = {}".format(time.time()-start))

start = time.time()
for iw in range (nwalkers):
    energies6 += [local_energy_generic_cholesky_opt_cubic_rhf(Ghalfa_batch[iw], Ghalfb_batch[iw], rchola, rcholb)]
energies6 = numpy.complex128(energies6)
print("cubic algorithm (RHF) = {}".format(time.time()-start))

# start = time.time()
# energies4 = local_energy_generic_cholesky_opt_rhf_batch(Ghalfa_batch, Ghalfb_batch, rchola, rcholb)
# energies4 = numpy.complex128(energies4)
# print("new algorithm batch (RHF) = {}".format(time.time()-start))

start = time.time()
rchola = rchola.reshape(naux,nocc,nao)
GhalfaT_batch = Ghalfa_batch.transpose(0,2,1).copy()

energies5 = local_energy_generic_cholesky_exx_rhf_batch(GhalfaT_batch, rchola)
energies5 = numpy.complex128(energies5)
print("new algorithm batch (RHF, Cython) = {}".format(time.time()-start))

# assert(numpy.allclose(energies,energies2))
# assert(numpy.allclose(energies3,energies4))
# assert(numpy.allclose(energies4,energies5))
# # assert(numpy.allclose(energies,energies3))
# # assert(numpy.allclose(energies,energies4))
import cProfile
pr = cProfile.Profile()
pr.enable()

start = time.time()
for iw in range (nwalkers):
    energies3 += [local_energy_generic_cholesky_opt_rhf(Ghalfa_batch[iw], rchola)]
# lp = LineProfiler()
# lp_wrapper = lp(local_energy_generic_cholesky_opt_rhf_batch)
# lp_wrapper(Ghalfa_batch, rchola)
# lp.print_stats()
print("local_energy_generic_cholesky_opt_rhf profiled = {}".format(time.time()-start))

pr.disable()
pr.print_stats(sort='tottime')

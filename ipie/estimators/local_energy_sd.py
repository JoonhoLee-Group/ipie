import time
import numpy
from numba import jit
from math import ceil

from ipie.estimators.local_energy import local_energy_G
from ipie.utils.misc import is_cupy

@jit(nopython=True,fastmath=True)
def exx_kernel_batch_real_rchol(rchola, Ghalfa_batch):
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot

    naux = rchola.shape[0]
    nwalkers = Ghalfa_batch.shape[0]
    nocc = Ghalfa_batch.shape[1]
    nbsf = Ghalfa_batch.shape[2]

    T = zeros((nocc,nocc), dtype=numpy.complex128)
    exx = zeros((nwalkers), dtype=numpy.complex128)
    rchol = rchola.reshape((naux,nocc,nbsf))
    for iw in range(nwalkers):
        Greal = Ghalfa_batch[iw].T.real.copy()
        Gimag = Ghalfa_batch[iw].T.imag.copy()
        for jx in range(naux):
            T = rchol[jx].dot(Greal) + 1.j * rchol[jx].dot(Gimag)
            exx[iw] += dot(T.ravel(), T.T.ravel())
    exx *= 0.5
    return exx

@jit(nopython=True,fastmath=True)
def exx_kernel_batch_complex_rchol(rchola, Ghalfa_batch):
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot

    naux = rchola.shape[0]
    nwalkers = Ghalfa_batch.shape[0]
    nocc = Ghalfa_batch.shape[1]
    nbsf = Ghalfa_batch.shape[2]

    T = zeros((nocc,nocc), dtype=numpy.complex128)
    exx = zeros((nwalkers), dtype=numpy.complex128)
    for iw in range(nwalkers):
        Ghalfa = Ghalfa_batch[iw]
        for jx in range(naux):
            rcholx = rchola[jx].reshape(nocc,nbsf)
            T = rcholx.dot(Ghalfa.T)
            exx[iw] += dot(T.ravel(), T.T.ravel())
    exx *= 0.5
    return exx

@jit(nopython=True,fastmath=True)
def ecoul_kernel_batch_real_rchol_rhf(rchola, Ghalfa_batch):
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot
    nwalkers = Ghalfa_batch.shape[0]
    Ghalfa_batch_real = Ghalfa_batch.real.copy()
    Ghalfa_batch_imag = Ghalfa_batch.imag.copy()
    X = rchola.dot(Ghalfa_batch_real.T) + 1.j * rchola.dot(Ghalfa_batch_imag.T) # naux x nwalkers
    ecoul = zeros (nwalkers, dtype = numpy.complex128)
    X = X.T.copy()
    for iw in range(nwalkers):
        ecoul[iw] += 2. * dot(X[iw],X[iw])

    return ecoul

@jit(nopython=True,fastmath=True)
def ecoul_kernel_batch_real_rchol_uhf(rchola, rcholb, Ghalfa_batch, Ghalfb_batch):
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot
    nwalkers = Ghalfa_batch.shape[0]
    Ghalfa_batch_real = Ghalfa_batch.real.copy()
    Ghalfa_batch_imag = Ghalfa_batch.imag.copy()
    Ghalfb_batch_real = Ghalfb_batch.real.copy()
    Ghalfb_batch_imag = Ghalfb_batch.imag.copy()
    X = rchola.dot(Ghalfa_batch_real.T) + 1.j * rchola.dot(Ghalfa_batch_imag.T) # naux x nwalkers
    X += rcholb.dot(Ghalfb_batch_real.T) + 1.j * rcholb.dot(Ghalfb_batch_imag.T) # naux x nwalkers
    ecoul = zeros(nwalkers, dtype = numpy.complex128)
    X = X.T.copy()
    for iw in range(nwalkers):
        ecoul[iw] += dot(X[iw], X[iw])
    ecoul *= 0.5
    return ecoul

@jit(nopython=True,fastmath=True)
def ecoul_kernel_batch_complex_rchol_rhf(rchola, Ghalfa_batch):
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot

    X = rchola.dot(Ghalfa_batch.T)
    ecoul = zeros (nwalkers, dtype = numpy.complex128)
    X = X.T.copy()
    for iw in range(nwalkers):
        ecoul[iw] += 2. * dot(X[iw],X[iw])
    return ecoul

@jit(nopython=True,fastmath=True)
def ecoul_kernel_batch_complex_rchol_uhf(rchola, rcholb, Ghalfa_batch, Ghalfb_batch):
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot

    X = rchola.dot(Ghalfa_batch.T)
    X += rcholb.dot(Ghalfb_batch.T)
    ecoul = zeros (nwalkers, dtype = numpy.complex128)
    X = X.T.copy()
    for iw in range(nwalkers):
        ecoul[iw] += dot(X[iw],X[iw])
    ecoul *= 0.5
    return ecoul

def local_energy_single_det_batch(system, hamiltonian, walker_batch, trial):
    if is_cupy(trial.psi): # if even one array is a cupy array we should assume the rest is done with cupy
        import cupy
        assert(cupy.is_available())
        array = cupy.array
    else:
        array = numpy.array

    energy = []
    nwalkers = walker_batch.nwalkers
    for idx in range(nwalkers):
        G = [walker_batch.Ga[idx],walker_batch.Gb[idx]]
        Ghalf = [walker_batch.Ghalfa[idx],walker_batch.Ghalfb[idx]]
        energy += [list(local_energy_G(system, hamiltonian, trial, G, Ghalf))]

    energy = array(energy, dtype=numpy.complex128)
    return energy

def local_energy_single_det_batch_einsum(system, hamiltonian, walker_batch, trial):

    if is_cupy(trial.psi): # if even one array is a cupy array we should assume the rest is done with cupy
        import cupy
        assert(cupy.is_available())
        einsum = cupy.einsum
        zeros = cupy.zeros
        isrealobj = cupy.isrealobj
    else:
        einsum = numpy.einsum
        zeros = numpy.zeros
        isrealobj = numpy.isrealobj

    nwalkers = walker_batch.Ghalfa.shape[0]
    nalpha = walker_batch.Ghalfa.shape[1]
    nbeta = walker_batch.Ghalfb.shape[1]
    nbasis = walker_batch.Ghalfa.shape[-1]
    nchol = hamiltonian.nchol

    walker_batch.Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha*nbasis)
    walker_batch.Ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nbeta*nbasis)

    e1b = walker_batch.Ghalfa.dot(trial._rH1a.ravel()) + walker_batch.Ghalfb.dot(trial._rH1b.ravel()) + hamiltonian.ecore

    if (isrealobj(trial._rchola)):
        Xa = trial._rchola.dot(walker_batch.Ghalfa.real.T) + 1.j * trial._rchola.dot(walker_batch.Ghalfa.imag.T) # naux x nwalkers
        Xb = trial._rcholb.dot(walker_batch.Ghalfb.real.T) + 1.j * trial._rcholb.dot(walker_batch.Ghalfb.imag.T) # naux x nwalkers
    else:
        Xa = trial._rchola.dot(walker_batch.Ghalfa.T)
        Xb = trial._rcholb.dot(walker_batch.Ghalfb.T)

    ecoul = einsum("xw,xw->w", Xa, Xa, optimize=True)
    ecoul += einsum("xw,xw->w", Xb, Xb, optimize=True)
    ecoul += 2. * einsum("xw,xw->w", Xa, Xb, optimize=True)

    walker_batch.Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha, nbasis)
    walker_batch.Ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nbeta, nbasis)

    Ta = zeros((nwalkers, nalpha,nalpha), dtype=numpy.complex128)
    Tb = zeros((nwalkers, nbeta,nbeta), dtype=numpy.complex128)

    GhalfaT_batch = walker_batch.Ghalfa.transpose(0,2,1).copy() # nw x nbasis x nocc
    GhalfbT_batch = walker_batch.Ghalfb.transpose(0,2,1).copy() # nw x nbasis x nocc

    exx  = zeros(nwalkers, dtype=numpy.complex128)  # we will iterate over cholesky index to update Ex energy for alpha and beta
    # breakpoint()
    for x in range(nchol):  # write a cython function that calls blas for this.
        rmi_a = trial._rchola[x].reshape((nalpha,nbasis))
        rmi_b = trial._rcholb[x].reshape((nbeta,nbasis))
        if (isrealobj(trial._rchola)): # this is actually fasater
            Ta[:,:,:].real = rmi_a.dot(GhalfaT_batch.real).transpose(1,0,2)
            Ta[:,:,:].imag = rmi_a.dot(GhalfaT_batch.imag).transpose(1,0,2)
            Tb[:,:,:].real = rmi_b.dot(GhalfbT_batch.real).transpose(1,0,2)
            Tb[:,:,:].imag = rmi_b.dot(GhalfbT_batch.imag).transpose(1,0,2)
        else:
            Ta = rmi_a.dot(GhalfaT_batch).transpose(1,0,2)
            Tb = rmi_b.dot(GhalfbT_batch).transpose(1,0,2)
        # this James Spencer's change is actually slower
        # Ta = walker_batch.Ghalfa @ rmi_a.T
        # Tb = walker_batch.Ghalfb @ rmi_b.T

        exx += einsum("wij,wji->w",Ta,Ta,optimize=True) + einsum("wij,wji->w",Tb,Tb,optimize=True)

    e2b = 0.5 * (ecoul - exx)

    energy = zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:,0] = e1b+e2b
    energy[:,1] = e1b
    energy[:,2] = e2b

    return energy

def local_energy_single_det_rhf_batch(system, hamiltonian, walker_batch, trial):

    if is_cupy(trial.psi): # if even one array is a cupy array we should assume the rest is done with cupy
        import cupy
        assert(cupy.is_available())
        einsum = cupy.einsum
        zeros = cupy.zeros
        isrealobj = cupy.isrealobj
    else:
        einsum = numpy.einsum
        zeros = numpy.zeros
        isrealobj = numpy.isrealobj

    nwalkers = walker_batch.Ghalfa.shape[0]
    nalpha = walker_batch.Ghalfa.shape[1]
    nbasis = hamiltonian.nbasis
    nchol = hamiltonian.nchol

    walker_batch.Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha*nbasis)

    e1b = 2.0 * walker_batch.Ghalfa.dot(trial._rH1a.ravel()) + hamiltonian.ecore

    if (isrealobj(trial._rchola)):
        ecoul = ecoul_kernel_batch_real_rchol_rhf(trial._rchola, walker_batch.Ghalfa)
    else:
        ecoul = ecoul_kernel_batch_complex_rchol_rhf(trial._rchola, walker_batch.Ghalfa)

    walker_batch.Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha, nbasis)

    if (isrealobj(trial._rchola)):
        exx = 2. * exx_kernel_batch_real_rchol (trial._rchola, walker_batch.Ghalfa)
    else:
        exx = 2. * exx_kernel_batch_complex_rchol (trial._rchola, walker_batch.Ghalfa)

    e2b = ecoul - exx

    energy = zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:,0] = e1b+e2b
    energy[:,1] = e1b
    energy[:,2] = e2b

    return energy

def two_body_energy_uhf(trial, walker_batch):
    if is_cupy(trial.psi):
        isrealobj = cupy.isrealobj
    else:
        isrealobj = numpy.isrealobj
    nwalkers = walker_batch.Ghalfa.shape[0]
    nalpha = walker_batch.Ghalfa.shape[1]
    nbeta = walker_batch.Ghalfb.shape[1]
    nbasis = walker_batch.Ghalfa.shape[2]
    Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha*nbasis)
    Ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nbeta*nbasis)
    if isrealobj(trial._rchola):
        ecoul = ecoul_kernel_batch_real_rchol_uhf(trial._rchola, trial._rcholb, Ghalfa, Ghalfb)
        exx = exx_kernel_batch_real_rchol(trial._rchola, walker_batch.Ghalfa) + exx_kernel_batch_real_rchol(trial._rcholb, walker_batch.Ghalfb)
    else:
        ecoul = ecoul_kernel_batch_complex_rchol_uhf(trial._rchola, trial._rcholb, Ghalfa, Ghalfb)
        exx = exx_kernel_batch_complex_rchol(trial._rchola, walker_batch.Ghalfa) + exx_kernel_batch_complex_rchol(trial._rcholb, walker_batch.Ghalfb)
    return ecoul - exx

def local_energy_single_det_uhf_batch(system, hamiltonian, walker_batch, trial):

    if is_cupy(trial.psi): # if even one array is a cupy array we should assume the rest is done with cupy
        import cupy
        assert(cupy.is_available())
        einsum = cupy.einsum
        zeros = cupy.zeros
        isrealobj = cupy.isrealobj
    else:
        einsum = numpy.einsum
        zeros = numpy.zeros
        isrealobj = numpy.isrealobj

    nwalkers = walker_batch.Ghalfa.shape[0]
    nalpha = walker_batch.Ghalfa.shape[1]
    nbeta = walker_batch.Ghalfb.shape[1]
    nbasis = hamiltonian.nbasis

    Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha*nbasis)
    Ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nbeta*nbasis)

    e1b = walker_batch.Ghalfa.dot(trial._rH1a.ravel())
    e1b += walker_batch.Ghalfb.dot(trial._rH1b.ravel())
    e1b += hamiltonian.ecore

    if (isrealobj(trial._rchola)):
        ecoul = ecoul_kernel_batch_real_rchol_uhf(trial._rchola, trial._rcholb, walker_batch.Ghalfa, walker_batch.Ghalfb)
    else:
        ecoul = ecoul_kernel_batch_complex_rchol_uhf(trial._rchola, trial._rcholb, walker_batch.Ghalfa, walker_batch.Ghalfb)

    if (isrealobj(trial._rchola)):
        exx = exx_kernel_batch_real_rchol (trial._rchola, walker_batch.Ghalfa) + exx_kernel_batch_real_rchol (trial._rcholb, walker_batch.Ghalfb)
    else:
        exx = exx_kernel_batch_complex_rchol (trial._rchola, walker_batch.Ghalfa) + exx_kernel_batch_complex_rchol (trial._rcholb, walker_batch.Ghalfb)

    e2b = ecoul - exx

    energy = zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:,0] = e1b+e2b
    energy[:,1] = e1b
    energy[:,2] = e2b

    return energy

def local_energy_single_det_batch_gpu_old(system, hamiltonian, walker_batch, trial):

    if is_cupy(trial.psi): # if even one array is a cupy array we should assume the rest is done with cupy
        import cupy
        assert(cupy.is_available())
        einsum = cupy.einsum
        zeros = cupy.zeros
        isrealobj = cupy.isrealobj
    else:
        einsum = numpy.einsum
        zeros = numpy.zeros
        isrealobj = numpy.isrealobj

    nwalkers = walker_batch.Ghalfa.shape[0]
    nalpha = walker_batch.Ghalfa.shape[1]
    nbeta = walker_batch.Ghalfb.shape[1]
    nbasis = walker_batch.Ghalfa.shape[-1]
    nchol = hamiltonian.nchol

    walker_batch.Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha*nbasis)
    walker_batch.Ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nbeta*nbasis)

    e1b = walker_batch.Ghalfa.dot(trial._rH1a.ravel()) + walker_batch.Ghalfb.dot(trial._rH1b.ravel()) + hamiltonian.ecore

    if (isrealobj(trial._rchola)):
        Xa = trial._rchola.dot(walker_batch.Ghalfa.real.T) + 1.j * trial._rchola.dot(walker_batch.Ghalfa.imag.T) # naux x nwalkers
        Xb = trial._rcholb.dot(walker_batch.Ghalfb.real.T) + 1.j * trial._rcholb.dot(walker_batch.Ghalfb.imag.T) # naux x nwalkers
    else:
        Xa = trial._rchola.dot(walker_batch.Ghalfa.T)
        Xb = trial._rcholb.dot(walker_batch.Ghalfb.T)

    ecoul = einsum("xw,xw->w", Xa, Xa, optimize=True)
    ecoul += einsum("xw,xw->w", Xb, Xb, optimize=True)
    ecoul += 2. * einsum("xw,xw->w", Xa, Xb, optimize=True)

    walker_batch.Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha, nbasis)
    walker_batch.Ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nbeta, nbasis)

    trial._rchola = trial._rchola.reshape(nchol, nalpha, nbasis)
    trial._rcholb = trial._rcholb.reshape(nchol, nbeta, nbasis)

    Txij = einsum("xim,wjm->wxji", trial._rchola, walker_batch.Ghalfa)
    exx  = einsum("wxji,wxij->w",Txij,Txij)
    Txij = einsum("xim,wjm->wxji", trial._rcholb, walker_batch.Ghalfb)
    exx += einsum("wxji,wxij->w",Txij,Txij)

    # exx = einsum("xim,xjn,win,wjm->w",trial._rchola, trial._rchola, walker_batch.Ghalfa, walker_batch.Ghalfa, optimize=True)\
    #     + einsum("xim,xjn,win,wjm->w",trial._rcholb, trial._rcholb, walker_batch.Ghalfb, walker_batch.Ghalfb, optimize=True)

    trial._rchola = trial._rchola.reshape(nchol, nalpha*nbasis)
    trial._rcholb = trial._rcholb.reshape(nchol, nbeta*nbasis)

    e2b = 0.5 * (ecoul - exx)

    energy = zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:,0] = e1b+e2b
    energy[:,1] = e1b
    energy[:,2] = e2b

    if is_cupy(trial.psi): # if even one array is a cupy array we should assume the rest is done with cupy
        import cupy
        cupy.cuda.stream.get_current_stream().synchronize()
    return energy

def local_energy_single_det_batch_gpu(
        system,
        hamiltonian,
        walker_batch,
        trial,
        max_mem=2
        ):

    if is_cupy(trial.psi): # if even one array is a cupy array we should assume the rest is done with cupy
        import cupy
        assert(cupy.is_available())
        einsum = cupy.einsum
        zeros = cupy.zeros
        isrealobj = cupy.isrealobj
        from ipie.estimators.kernels.gpu import exchange as kernels
        dot = cupy.dot
        complex128 = cupy.complex128
    else:
        einsum = numpy.einsum
        zeros = numpy.zeros
        isrealobj = numpy.isrealobj
        dot = numpy.dot
        complex128 = numpy.complex128

    nwalkers = walker_batch.Ghalfa.shape[0]
    nalpha = walker_batch.Ghalfa.shape[1]
    nbeta = walker_batch.Ghalfb.shape[1]
    nbasis = walker_batch.Ghalfa.shape[-1]
    nchol = hamiltonian.nchol

    Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha*nbasis)
    Ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nbeta*nbasis)

    e1b = Ghalfa.dot(trial._rH1a.ravel()) + Ghalfb.dot(trial._rH1b.ravel()) + hamiltonian.ecore

    if (isrealobj(trial._rchola)):
        Xa = trial._rchola.dot(Ghalfa.real.T) + 1.j * trial._rchola.dot(Ghalfa.imag.T) # naux x nwalkers
        Xb = trial._rcholb.dot(Ghalfb.real.T) + 1.j * trial._rcholb.dot(Ghalfb.imag.T) # naux x nwalkers
    else:
        Xa = trial._rchola.dot(Ghalfa.T)
        Xb = trial._rcholb.dot(Ghalfb.T)

    ecoul = einsum("xw,xw->w", Xa, Xa, optimize=True)
    ecoul += einsum("xw,xw->w", Xb, Xb, optimize=True)
    ecoul += 2. * einsum("xw,xw->w", Xa, Xb, optimize=True)

    max_nocc = max(nalpha, nbeta)
    mem_needed = 16 * nwalkers * max_nocc * max_nocc * nchol / (1024.0**3.0)
    num_chunks = max(1, ceil(mem_needed / max_mem))
    chunk_size = ceil(nchol / num_chunks)

    # Buffer for large intermediate tensor
    buff = zeros(shape=(nwalkers*chunk_size*max_nocc*max_nocc), dtype=complex128)
    nchol_chunk = chunk_size
    nchol_left = nchol
    exx = zeros(nwalkers, dtype=complex128)
    Ghalfa = walker_batch.Ghalfa.reshape((nwalkers*nalpha, nbasis))
    Ghalfb = walker_batch.Ghalfb.reshape((nwalkers*nbeta, nbasis))
    for i in range(num_chunks):
        nchol_chunk = min(nchol_chunk, nchol_left)
        chol_sls = slice(i*chunk_size, i*chunk_size + nchol_chunk)
        size = nwalkers * nchol_chunk * nalpha * nalpha
        # alpha-alpha
        Txij = buff[:size].reshape((nchol_chunk*nalpha, nwalkers*nalpha))
        rchol = trial._rchola[chol_sls].reshape((nchol_chunk*nalpha, nbasis))
        dot(rchol, Ghalfa.T, out=Txij)
        Txij = Txij.reshape((nchol_chunk, nalpha, nwalkers, nalpha))
        kernels.exchange_reduction(Txij, exx)
        # beta-beta
        size = nwalkers * nchol_chunk * nbeta * nbeta
        Txij = buff[:size].reshape((nchol_chunk*nbeta, nwalkers*nbeta))
        rchol = trial._rcholb[chol_sls].reshape((nchol_chunk*nbeta, nbasis))
        dot(rchol, Ghalfb.T, out=Txij)
        Txij = Txij.reshape((nchol_chunk, nbeta, nwalkers, nbeta))
        kernels.exchange_reduction(Txij, exx)
        nchol_left -= chunk_size

    e2b = 0.5 * (ecoul - exx)

    energy = zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:,0] = e1b + e2b
    energy[:,1] = e1b
    energy[:,2] = e2b

    if is_cupy(trial.psi): # if even one array is a cupy array we should assume the rest is done with cupy
        import cupy
        cupy.cuda.stream.get_current_stream().synchronize()
    return energy

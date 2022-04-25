import time
import numpy
from ipie.estimators.local_energy import local_energy_G
from ipie.utils.misc import is_cupy
from numba import jit

@jit(nopython=True,fastmath=True)
def exx_kernel_batch(rchola, Ghalfa_batch):
    # if is_cupy(rchola): # if even one array is a cupy array we should assume the rest is done with cupy
    #     import cupy
    #     assert(cupy.is_available())
    #     zeros = cupy.zeros
    #     dot = cupy.dot
    # else:
    zeros = numpy.zeros
    dot = numpy.dot

    naux = rchola.shape[0]
    nwalkers = Ghalfa_batch.shape[0]
    nocc = Ghalfa_batch.shape[1]
    nbsf = Ghalfa_batch.shape[2]

    T = zeros((nocc,nocc), dtype=numpy.complex128)
    exx = zeros((nwalkers), dtype=numpy.complex128)
    for iw in range(nwalkers):
        Greal = Ghalfa_batch[iw].real.copy()
        Gimag = Ghalfa_batch[iw].imag.copy()
        for jx in range(naux):
            rcholx = rchola[jx].reshape(nocc,nbsf)
            T = rcholx.dot(Greal.T) + 1.j * rcholx.dot(Gimag.T)
            exx[iw] += dot(T.ravel(), T.T.ravel())
    exx *= 0.5
    return exx

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

    exx  = zeros(nwalkers, dtype=numpy.complex128)  # we will iterate over cholesky index to update Ex energy for alpha and beta
    # breakpoint()
    for x in range(nchol):  # write a cython function that calls blas for this.
        rmi_a = trial._rchola[x].reshape((nalpha,nbasis))
        rmi_b = trial._rcholb[x].reshape((nbeta,nbasis))
        # if (isrealobj(trial._rchola)):
            # Ta += rmi_a.dot(GhalfaT_batch.real).transpose(1,0,2) + 1j * rmi_a.dot(GhalfaT_batch.imag).transpose(1,0,2)
            # Tb += rmi_b.dot(GhalfbT_batch.real).transpose(1,0,2) + 1j * rmi_b.dot(GhalfbT_batch.imag).transpose(1,0,2)
        # else:
            # Ta += rmi_a.dot(GhalfaT_batch).transpose(1,0,2)
            # Tb += rmi_b.dot(GhalfbT_batch).transpose(1,0,2)
        Ta = walker_batch.Ghalfa @ rmi_a.T
        Tb = walker_batch.Ghalfb @ rmi_b.T

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
        Xa = trial._rchola.dot(walker_batch.Ghalfa.real.T) + 1.j * trial._rchola.dot(walker_batch.Ghalfa.imag.T) # naux x nwalkers
    else:
        Xa = trial._rchola.dot(walker_batch.Ghalfa.T)

    ecoul = 2. * einsum("xw,xw->w", Xa, Xa, optimize=True)

    walker_batch.Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha, nbasis)
    exx = 2. * exx_kernel_batch (trial._rchola, walker_batch.Ghalfa)

    e2b = ecoul - exx

    energy = zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:,0] = e1b+e2b
    energy[:,1] = e1b
    energy[:,2] = e2b

    return energy

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
    nchol = hamiltonian.nchol

    walker_batch.Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha*nbasis)
    walker_batch.Ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nbeta*nbasis)

    e1b = walker_batch.Ghalfa.dot(trial._rH1a.ravel())
    e1b += walker_batch.Ghalfb.dot(trial._rH1b.ravel())
    e1b += hamiltonian.ecore

    if (isrealobj(trial._rchola)):
        Xa = trial._rchola.dot(walker_batch.Ghalfa.real.T) + 1.j * trial._rchola.dot(walker_batch.Ghalfa.imag.T) # naux x nwalkers
        Xb = trial._rcholb.dot(walker_batch.Ghalfb.real.T) + 1.j * trial._rcholb.dot(walker_batch.Ghalfb.imag.T) # naux x nwalkers
    else:
        Xa = trial._rchola.dot(walker_batch.Ghalfa.T)
        Xb = trial._rchola.dot(walker_batch.Ghalfb.T)
    
    ecoul = einsum("xw,xw->w", Xa, Xa, optimize=True) 
    ecoul += einsum("xw,xw->w", Xb, Xb, optimize=True)
    ecoul += 2.*einsum("xw,xw->w", Xa, Xb, optimize=True)
    ecoul *= 0.5

    walker_batch.Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha, nbasis)
    walker_batch.Ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nbeta, nbasis)
    exx = exx_kernel_batch (trial._rchola, walker_batch.Ghalfa) + exx_kernel_batch (trial._rcholb, walker_batch.Ghalfb)

    e2b = ecoul - exx

    energy = zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:,0] = e1b+e2b
    energy[:,1] = e1b
    energy[:,2] = e2b

    return energy

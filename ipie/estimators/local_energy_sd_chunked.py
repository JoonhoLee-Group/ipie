import time
import numpy
from ipie.utils.misc import is_cupy
from numba import jit

def local_energy_single_det_uhf_batch_chunked(system, hamiltonian, walker_batch, trial):

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
        X = trial._rchola.dot(walker_batch.Ghalfa.real.T) + 1.j * trial._rchola.dot(walker_batch.Ghalfa.imag.T) # naux x nwalkers
        X += trial._rcholb.dot(walker_batch.Ghalfb.real.T) + 1.j * trial._rcholb.dot(walker_batch.Ghalfb.imag.T) # naux x nwalkers
    else:
        X = trial._rchola.dot(walker_batch.Ghalfa.T)
        X += trial._rchola.dot(walker_batch.Ghalfb.T)
    
    ecoul = 0.5 * einsum("xw,xw->w", X, X, optimize=True) 

    walker_batch.Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha, nbasis)
    walker_batch.Ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nbeta, nbasis)
    exx = exx_kernel_batch (trial._rchola, walker_batch.Ghalfa) + exx_kernel_batch (trial._rcholb, walker_batch.Ghalfb)

    e2b = ecoul - exx

    energy = zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:,0] = e1b+e2b
    energy[:,1] = e1b
    energy[:,2] = e2b

    return energy

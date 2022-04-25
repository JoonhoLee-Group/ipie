import time
import numpy
from ipie.legacy.estimators.local_energy import local_energy_generic_cholesky
from ipie.estimators.local_energy import local_energy_G
from ipie.estimators.local_energy_wicks import local_energy_multi_det_trial_wicks_batch,\
                                               local_energy_multi_det_trial_wicks_batch_opt
from ipie.utils.misc import is_cupy

# TODO: should pass hamiltonian here and make it work for all possible types
# this is a generic local_energy handler. So many possible combinations of local energy strategies...
def local_energy_batch(system, hamiltonian, walker_batch, trial):

    if (walker_batch.name == "SingleDetWalkerBatch"):
        if (is_cupy(walker_batch.phia)):
            return local_energy_single_det_batch_einsum(system, hamiltonian, walker_batch, trial)
        elif (walker_batch.rhf):
            return local_energy_single_det_rhf_batch(system, hamiltonian, walker_batch, trial)
        else:
            return local_energy_single_det_batch(system, hamiltonian, walker_batch, trial)
    elif (walker_batch.name == "MultiDetTrialWalkerBatch" and trial.wicks == False):
        return local_energy_multi_det_trial_batch(system, hamiltonian, walker_batch, trial)
    elif trial.name == "MultiSlater" and trial.ndets > 1 and trial.wicks == True and not trial.optimized:
        # return local_energy_multi_det_trial_batch(system, hamiltonian, walker_batch, trial)
        return local_energy_multi_det_trial_wicks_batch(system, hamiltonian, walker_batch, trial)
    elif trial.name == "MultiSlater" and trial.ndets > 1 and trial.wicks == True and trial.optimized == True:
        # return local_energy_multi_det_trial_batch(system, hamiltonian, walker_batch, trial)
        return local_energy_multi_det_trial_wicks_batch_opt(system, hamiltonian, walker_batch, trial)

# Naive way to evaluate local energy
def local_energy_multi_det_trial_batch(system, hamiltonian, walker_batch, trial):
    energy = []
    ndets = trial.ndets
    nwalkers = walker_batch.nwalkers
    # ndets x nwalkers
    for iwalker, (w, Ga, Gb, Ghalfa, Ghalfb) in enumerate(zip(walker_batch.det_weights,
                        walker_batch.Gia, walker_batch.Gib,
                        walker_batch.Gihalfa, walker_batch.Gihalfb)):
        denom = 0.0 + 0.0j
        numer0 = 0.0 + 0.0j
        numer1 = 0.0 + 0.0j
        numer2 = 0.0 + 0.0j
        for idet in range(ndets):
            # construct "local" green's functions for each component of A
            G = [Ga[idet], Gb[idet]]
            Ghalf = [Ghalfa[idet], Ghalfb[idet]]
            # return (e1b+e2b+ham.ecore, e1b+ham.ecore, e2b)
            e = list(local_energy_G(system, hamiltonian, trial, G, Ghalf=None))
            numer0 += w[idet] * e[0]
            numer1 += w[idet] * e[1]
            numer2 += w[idet] * e[2]
            denom += w[idet]
        # return (e1b+e2b+ham.ecore, e1b+ham.ecore, e2b)
        energy += [list([numer0/denom, numer1/denom, numer2/denom])]

    energy = numpy.array(energy, dtype=numpy.complex128)
    return energy

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

    # Ga = walker_batch.Ga.reshape((nwalkers, nbasis*nbasis))
    # Gb = walker_batch.Gb.reshape((nwalkers, nbasis*nbasis))
    # e1b = Ga.dot(hamiltonian.H1[0].ravel()) + Gb.dot(hamiltonian.H1[1].ravel()) + hamiltonian.ecore

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

    # GhalfaT_batch = walker_batch.Ghalfa.transpose(0,2,1).copy() # nw x nbasis x nocc
    # GhalfbT_batch = walker_batch.Ghalfb.transpose(0,2,1).copy() # nw x nbasis x nocc

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
    GhalfaT_batch = walker_batch.Ghalfa.transpose(0,2,1).copy() # nw x nbasis x nocc

    Ta = zeros((nalpha,nalpha), dtype=numpy.complex128)
    exx = zeros((nwalkers), dtype=numpy.complex128)

    for x in range(nchol):
        rmi_a = trial._rchola[x].reshape((nalpha,nbasis))
        for iw in range(nwalkers):
            if (isrealobj(trial._rchola)):
                Ta[:,:].real = rmi_a.dot(GhalfaT_batch[iw].real)
                Ta[:,:].imag = rmi_a.dot(GhalfaT_batch[iw].imag)
            else:
                Ta[:,:] = rmi_a.dot(GhalfaT_batch[iw])
            exx[iw] += einsum("ij,ji->",Ta,Ta)

    e2b = ecoul - exx

    energy = zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:,0] = e1b+e2b
    energy[:,1] = e1b
    energy[:,2] = e2b

    return energy

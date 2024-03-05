import numpy as np

from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.trial_wavefunction.eph_trial_base import EphTrialWavefunctionBase
from ipie.addons.eph.walkers.eph_walkers import EphWalkers
from ipie.systems.generic import Generic
from ipie.utils.backend import arraylib as xp 


def local_energy_holstein(
    system: Generic, 
    hamiltonian: HolsteinModel, 
    walkers: EphWalkers, 
    trial: EphTrialWavefunctionBase
):
    """Computes the local energy for the Holstein model."""
#    energy = xp.zeros((walkers.nwalkers, 5), dtype=xp.complex128)

#    gf = trial.calc_greens_function(walkers)
    
#    energy[:, 1] = np.sum(hamiltonian.T[0] * gf[0], axis=(1,2))
#    if system.ndown > 0:
#        energy[:, 1] += np.sum(hamiltonian.T[1] * gf[1], axis=(1,2))

#    energy[:, 2] = np.sum(np.diagonal(gf[0], axis1=1, axis2=2) * walkers.x, axis=1)
#    if system.ndown > 0:
#        energy[:, 2] +=  np.sum(np.diagonal(gf[1], axis1=1, axis2=2) * walkers.x, axis=1)
#    energy[:, 2] *= hamiltonian.const

#    energy[:, 3] = 0.5 * hamiltonian.m * hamiltonian.w0**2 * np.sum(walkers.x**2, axis=1)
#    energy[:, 3] -= 0.5 * hamiltonian.nsites * hamiltonian.w0
#    energy[:, 4] = -0.5 * trial.calc_phonon_laplacian_locenergy(walkers) / hamiltonian.m
    
#    energy[:, 0] = np.sum(energy[:,1:], axis=1)
   ## 
    energy = xp.zeros((walkers.nwalkers, 5), dtype=xp.complex128)

    #get greens_function from estimators
    gf = trial.calc_greens_function(walkers)
    
    #TODO make this nicer
    for n in range(walkers.nwalkers):
        energy[n, 1] = np.einsum('ij->', hamiltonian.T[0] * gf[0][n] + hamiltonian.T[1] * gf[1][n])

    #TODO this performs too many summations
    energy[:, 2] = hamiltonian.const * np.einsum('nii,ni->n', gf[0] + gf[1], walkers.x)

    energy[:, 3] = 0.5 * hamiltonian.m * hamiltonian.w0**2 * np.einsum('ni->n', walkers.x**2)
    energy[:, 3] -= 0.5 * hamiltonian.nsites * hamiltonian.w0
    energy[:, 4] = -0.5 * trial.calc_phonon_laplacian_locenergy(walkers) / hamiltonian.m
    energy[:, 0] = np.sum(energy[:,1:], axis=1)
    
    return energy




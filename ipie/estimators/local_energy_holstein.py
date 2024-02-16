import numpy as np

from ipie.hamiltonians.elph.holstein import HolsteinModel
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.holstein.eph_trial_base import EphTrialWavefunctionBase
from ipie.utils.backend import arraylib as xp 
from ipie.walkers.eph_walkers import EphWalkers

verbose=False

#TODO the whole thing, call stuff correctly
def local_energy_holstein(
    system: Generic, 
    hamiltonian: HolsteinModel, 
    walkers: EphWalkers, 
    trial: EphTrialWavefunctionBase
):
    """"""
    energy = xp.zeros((walkers.nwalkers, 5), dtype=xp.complex128)

    #get greens_function from estimators
    greensfct = trial.calc_greens_function(walkers)
    if verbose:
        print('gf walker0:      ', greensfct[0])
    #TODO make this nicer
    for n in range(walkers.nwalkers):
        gf = greensfct[n]
        energy[n, 1] = np.einsum('ij->', hamiltonian.T[0] * gf) #NOTE 1e hack
    energy[:, 2] = hamiltonian.const * np.einsum('nii,ni->n', greensfct, walkers.x)
    if verbose:
        print('const, phonons, elph:  ', hamiltonian.const, walkers.x[0,0], energy[0, 2])

    energy[:, 3] = 0.5 * hamiltonian.m * hamiltonian.w0**2 * np.einsum('ni->n', walkers.x**2)
    energy[:, 3] -= 0.5 * hamiltonian.nsites * hamiltonian.w0
    if verbose:
        print('e_ph pot:    ', energy[0, 3])
    
    energy[:, 4] = -0.5 * trial.calc_phonon_laplacian_locenergy(walkers) / hamiltonian.m
    if verbose:
        print('e_ph kin:    ',  energy[0, 4])

    energy[:, 0] = np.sum(energy[:,1:], axis=1)
    if verbose:
        print('energy:  ', energy[0,0])
    return energy




import numpy
from pyqumc.estimators.local_energy import local_energy_G

# TODO: should pass hamiltonian here and make it work for all possible types
# this is a generic local_energy handler. So many possible combinations of local energy strategies...
def local_energy_batch(system, hamiltonian, walker_batch, trial):
    energy = []
# def local_energy_G(system, hamiltonian, trial, G, Ghalf=None, X=None, Lap=None):
    nwalkers = walker_batch.nwalkers
    for iw in range(nwalkers):
        energy += [local_energy_G(system, hamiltonian, trial, walker_batch.G[iw], walker_batch.Ghalf[iw])]
    return energy
    

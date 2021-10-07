import numpy
from pyqumc.estimators.local_energy import local_energy_G

# TODO: should pass hamiltonian here and make it work for all possible types
# this is a generic local_energy handler. So many possible combinations of local energy strategies...
def local_energy_batch(system, hamiltonian, walker_batch, trial):
    energy = []
# def local_energy_G(system, hamiltonian, trial, G, Ghalf=None, X=None, Lap=None):
    nwalkers = walker_batch.nwalkers
    for iw in range(nwalkers):
        G = numpy.array([walker_batch.G[0,iw],walker_batch.G[1,iw]], dtype=walker_batch.G.dtype)
        Ghalf = numpy.array([walker_batch.Ghalf[0,iw],walker_batch.Ghalf[1,iw]], dtype=walker_batch.G.dtype)
        energy += [local_energy_G(system, hamiltonian, trial, G, Ghalf)]
    return energy
    

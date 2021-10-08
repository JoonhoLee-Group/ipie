import numpy
from pyqumc.estimators.local_energy import local_energy_G

# TODO: should pass hamiltonian here and make it work for all possible types
# this is a generic local_energy handler. So many possible combinations of local energy strategies...
def local_energy_batch(system, hamiltonian, walker_batch, trial, iw = None):
    energy = []
    if (iw == None):
        nwalkers = walker_batch.nwalkers
        for idx in range(nwalkers):
            G = [walker_batch.Ga[idx],walker_batch.Gb[idx]]
            Ghalf = [walker_batch.Ghalfa[idx],walker_batch.Ghalfb[idx]]
            energy += [list(local_energy_G(system, hamiltonian, trial, G, Ghalf))]

        energy = numpy.array(energy, dtype=numpy.complex128)
        return energy
    else:
        G = [walker_batch.Ga[iw],walker_batch.Gb[iw]]
        Ghalf = [walker_batch.Ghalfa[iw],walker_batch.Ghalfb[iw]]
        energy += [list(local_energy_G(system, hamiltonian, trial, G, Ghalf))]
        energy = numpy.array(energy, dtype=numpy.complex128)
        return energy
    

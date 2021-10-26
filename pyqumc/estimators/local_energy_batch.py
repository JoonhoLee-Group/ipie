import numpy
from pyqumc.estimators.local_energy import local_energy_G

# TODO: should pass hamiltonian here and make it work for all possible types
# this is a generic local_energy handler. So many possible combinations of local energy strategies...
def local_energy_batch(system, hamiltonian, walker_batch, trial, iw = None):

    if (walker_batch.name == "SingleDetWalkerBatch"):
        return local_energy_single_det_batch(system, hamiltonian, walker_batch, trial, iw = iw)
    elif (walker_batch.name == "MultiDetTrialWalkerBatch"):
        return local_energy_multi_det_trial_batch(system, hamiltonian, walker_batch, trial, iw = iw)


def local_energy_multi_det_trial_batch(system, hamiltonian, walker_batch, trial, iw = None):
    energy = []
    ndets = trial.ndets
    if (iw == None):
        nwalkers = walker_batch.nwalkers
        # ndets x nwalkers
        for iwalker, (w, Ga, Gb, Ghalfa, Ghalfb) in enumerate(zip(walker_batch.det_weights, 
                            walker_batch.Gia, walker_batch.Gib, 
                            walker_batch.Gihalfa, walker_batch.Gihalfb)):
            numpy.set_printoptions(precision=5, suppress=True)
            # print(Ga)
            denom = 0.0 + 0.0j
            numer0 = 0.0 + 0.0j
            numer1 = 0.0 + 0.0j
            numer2 = 0.0 + 0.0j
            for idet in range(ndets):
                # construct "local" green's functions for each component of A
                G = [Ga[idet], Gb[idet]]
                Ghalf = [Ghalfa[idet], Ghalfb[idet]]
                e = list(local_energy_G(system, hamiltonian, trial, G, Ghalf=None))
                numer0 += w[idet] * e[0]
                numer1 += w[idet] * e[1]
                numer2 += w[idet] * e[2]
                denom += w[idet]
            energy += [list([numer0/denom, numer1/denom, numer2/denom])]

    else:
        denom = 0.0 + 0.0j
        numer0 = 0.0 + 0.0j
        numer1 = 0.0 + 0.0j
        numer2 = 0.0 + 0.0j
        # ndets x nwalkers
        w = walker_batch.det_weights[iw]
        Ga = walker_batch.Gia[iw]
        Gb = walker_batch.Gib[iw]
        Ghalfa = walker_batch.Gihalfa[iw]
        Ghalfb = walker_batch.Gihalfb[iw]
        for idet in range(ndets):
            # construct "local" green's functions for each component of A
            G = [Ga[idet], Gb[idet]]
            Ghalf = [Ghalfa[idet], Ghalfb[idet]]
            e = list(local_energy_G(system, hamiltonian, trial, G, Ghalf=None))
            numer0 += w[idet] * e[0]
            numer1 += w[idet] * e[1]
            numer2 += w[idet] * e[2]
            denom += w[idet]
        energy += [list([numer0/denom, numer1/denom, numer2/denom])]

    energy = numpy.array(energy, dtype=numpy.complex128)
    return energy

def local_energy_single_det_batch(system, hamiltonian, walker_batch, trial, iw = None):
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
    

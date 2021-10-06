import numpy
import pytest
from pyqumc.hamiltonians.hubbard import Hubbard
from pyqumc.systems.generic import Generic
from pyqumc.propagation.hubbard import Hirsch
from pyqumc.trial_wavefunction.multi_slater import MultiSlater
from pyqumc.walkers.single_det import SingleDetWalker
from pyqumc.utils.misc import dotdict
from pyqumc.estimators.greens_function import gab
from pyqumc.estimators.local_energy import local_energy

@pytest.mark.unit
def test_overlap():
    options = {'nx': 4, 'ny': 4, 'nup': 8, 'ndown': 8, 'U': 4}
    system = Generic((8,8), verbose=False)
    ham = Hubbard(options, verbose=False)
    eigs, eigv = numpy.linalg.eigh(ham.H1[0])
    coeffs = numpy.array([1.0+0j])
    wfn = numpy.zeros((1,ham.nbasis,system.ne))
    wfn[0,:,:system.nup] = eigv[:,:system.nup].copy()
    wfn[0,:,system.nup:] = eigv[:,:system.ndown].copy()
    trial = MultiSlater(system, ham, (coeffs, wfn))
    trial.psi = trial.psi[0]

    nwalkers = 10
    walkers = [SingleDetWalker(system, ham, trial) for i in range(nwalkers)]

    nup = system.nup
    for iw, walker in enumerate(walkers):
        ovlp = numpy.dot(trial.psi[:,:nup].conj().T, walkers[iw].phi[:,:nup])
        id_exp = numpy.dot(walkers[iw].inv_ovlp[0], ovlp)
        numpy.testing.assert_allclose(id_exp, numpy.eye(nup), atol=1e-12)

if __name__=="__main__":
    test_overlap()
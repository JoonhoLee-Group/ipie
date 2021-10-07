import pytest
import numpy
from pyqumc.utils.misc import dotdict
from pyqumc.trial_wavefunction.multi_slater import MultiSlater
from pyqumc.systems.generic import Generic
from pyqumc.propagation.continuous import Continuous
from pyqumc.hamiltonians.generic import Generic as HamGeneric
from pyqumc.walkers.single_det_batch import SingleDetWalkerBatch
from pyqumc.walkers.single_det import SingleDetWalker
from pyqumc.utils.testing import (
        generate_hamiltonian,
        get_random_nomsd,
        get_random_phmsd
        )

@pytest.mark.unit
def test_hybrid():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5,5)
    nwalkers = 10
    nsteps = 10
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=1, init=True)
    trial = MultiSlater(system, ham, wfn, init=init)
    trial.half_rotate(system, ham)

    numpy.random.seed(7)

    trial.psi = trial.psi[0]
    trial.calculate_energy(system, ham)
    options = {'hybrid': True}
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = Continuous(system, ham, trial, qmc, options=options)

    walkers = [SingleDetWalker(system, ham, trial) for iw in range(nwalkers)]
    for i in range (nsteps):
        for walker in walkers:
            prop.propagate_walker(walker, system, ham, trial, trial.energy)
            detR = walker.reortho(trial) # reorthogonalizing to stablize

    numpy.random.seed(7)
    qmc = dotdict({'dt': 0.005, 'nstblz': 5, 'batched': True, 'nwalkers': 10})
    prop = Continuous(system, ham, trial, qmc, options=options)
    walker_batch = SingleDetWalkerBatch(system, ham, trial, nwalkers)
    for i in range (nsteps):
        prop.propagate_walker_batch(walker_batch, system, ham, trial, trial.energy)
        walker_batch.reortho()

    for iw in range(nwalkers):
        assert numpy.allclose(walker_batch.phi[iw], walkers[iw].phi)

if __name__ == '__main__':
    test_hybrid()

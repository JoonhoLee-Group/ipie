import numpy
import os
import pytest
from pie.systems.generic import Generic
from pie.hamiltonians.generic import Generic as HamGeneric
from pie.trial_wavefunction.multi_slater import MultiSlater
from pie.propagation.generic import GenericContinuous
from pie.propagation.continuous import Continuous
from pie.utils.misc import dotdict
from pie.utils.testing import (
        generate_hamiltonian,
        get_random_nomsd,
        get_random_phmsd
        )
from pie.walkers.multi_det import MultiDetWalker
from pie.walkers.multi_det_batch import MultiDetTrialWalkerBatch
from pie.estimators.greens_function import gab_spin, gab

from pie.estimators.greens_function_batch import greens_function_multi_det_wicks, greens_function_multi_det
from pie.estimators.local_energy_batch import local_energy_multi_det_trial_batch, local_energy_multi_det_trial_wicks_batch

from pie.estimators.local_energy import local_energy_multi_det


@pytest.mark.unit
def test_phmsd_local_energy():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5,5)
    nwalkers = 10
    nsteps = 100
    ndets = 5
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=5, init=True)
    trial = MultiSlater(system, ham, wfn, init=init, options = {'wicks':True})
    trial.calculate_energy(system, ham)
    options = {'hybrid': True}
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = Continuous(system, ham, trial, qmc, options=options)

    numpy.random.seed(7)
    qmc = dotdict({'dt': 0.005, 'nstblz': 5, 'batched': True, 'nwalkers': 10})
    prop = Continuous(system, ham, trial, qmc, options=options)
    walker_batch = MultiDetTrialWalkerBatch(system, ham, trial, nwalkers)
    for i in range (nsteps):
        prop.propagate_walker_batch(walker_batch, system, ham, trial, trial.energy)
        walker_batch.reortho()

    greens_function_multi_det_wicks(walker_batch, trial) # compute green's function using Wick's theorem
    e_wicks = local_energy_multi_det_trial_wicks_batch(system, ham, walker_batch, trial)
    greens_function_multi_det(walker_batch, trial)
    e_dumb = local_energy_multi_det_trial_batch(system, ham, walker_batch, trial)

    assert numpy.allclose(e_dumb, e_wicks)


if __name__ == '__main__':
    test_phmsd_local_energy()

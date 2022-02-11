import pytest
import numpy
from pyqumc.utils.misc import dotdict
from pyqumc.trial_wavefunction.multi_slater import MultiSlater
from pyqumc.systems.generic import Generic
from pyqumc.propagation.continuous import Continuous
from pyqumc.propagation.force_bias import construct_force_bias_batch
from pyqumc.hamiltonians.generic import Generic as HamGeneric
from pyqumc.walkers.single_det_batch import SingleDetWalkerBatch
from pyqumc.walkers.multi_det_batch import MultiDetTrialWalkerBatch
from pyqumc.walkers.single_det import SingleDetWalker
from pyqumc.walkers.multi_det import MultiDetWalker
from pyqumc.utils.testing import (
        generate_hamiltonian,
        get_random_nomsd,
        get_random_phmsd
        )
from pyqumc.estimators.local_energy import local_energy_generic_cholesky_opt
from pyqumc.estimators.local_energy_batch import local_energy_single_det_batch, local_energy_single_det_batch_einsum

try:
    import cupy
    no_gpu = not cupy.is_available()
except:
    no_gpu = True

@pytest.mark.unit
@pytest.mark.skipif(no_gpu, reason="gpu not found.")
def test_local_energy_single_det_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5,5)
    nwalkers = 10
    nsteps = 25
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=1, init=True)
    trial = MultiSlater(system, ham, wfn, init=init)
    trial.half_rotate(system, ham)
    trial.psi = trial.psi[0]
    trial.psia = trial.psia[0]
    trial.psib = trial.psib[0]
    trial.calculate_energy(system, ham)

    numpy.random.seed(7)

    options = {'hybrid': True}
    qmc = dotdict({'dt': 0.005, 'nstblz': 5, 'batched': True, 'nwalkers': nwalkers})
    prop = Continuous(system, ham, trial, qmc, options=options)
    walker_batch = SingleDetWalkerBatch(system, ham, trial, nwalkers)

    if not no_gpu:
        prop.cast_to_cupy()
        ham.cast_to_cupy()
        trial.cast_to_cupy()
        walker_batch.cast_to_cupy()

    for i in range (nsteps):
        prop.propagate_walker_batch(walker_batch, system, ham, trial, trial.energy)
        walker_batch.reortho()

    energies = local_energy_single_det_batch(system, ham, walker_batch, trial)
    energies_einsum = local_energy_single_det_batch_einsum(system, ham, walker_batch, trial)

    assert numpy.allclose(energies, energies_einsum)
    for iw in range(nwalkers):
        energy = local_energy_single_det_batch(system, ham, walker_batch, trial, iw = iw)
        assert numpy.allclose(energy, energies[iw])
        assert numpy.allclose(energy, energies_einsum[iw])

if __name__ == '__main__':
    test_local_energy_single_det_batch()

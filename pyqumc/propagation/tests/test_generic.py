import numpy
import os
import pytest
from pyqumc.systems.generic import Generic
from pyqumc.hamiltonians.generic import Generic as HamGeneric
from pyqumc.trial_wavefunction.multi_slater import MultiSlater
from pyqumc.propagation.generic import GenericContinuous
from pyqumc.propagation.continuous import Continuous
from pyqumc.utils.misc import dotdict
from pyqumc.utils.testing import (
        generate_hamiltonian,
        get_random_nomsd,
        get_random_phmsd
        )
from pyqumc.walkers.multi_det import MultiDetWalker

@pytest.mark.unit
def test_nomsd():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5,5)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    wfn = get_random_nomsd(system.nup, system.ndown, ham.nbasis, ndet=3)
    trial = MultiSlater(system, ham, wfn)
    walker = MultiDetWalker(system, ham, trial)
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = GenericContinuous(system, ham, trial, qmc)
    fb = prop.construct_force_bias(ham, walker, trial)
    prop.construct_VHS(ham, fb)

@pytest.mark.unit
def test_phmsd():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5,5)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=3, init=True)
    trial = MultiSlater(system, ham, wfn, init=init)
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = GenericContinuous(system, ham, trial, qmc)
    walker = MultiDetWalker(system, ham, trial)
    fb = prop.construct_force_bias(ham, walker, trial)
    vhs = prop.construct_VHS(ham, fb)

@pytest.mark.unit
def test_local_energy():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5,5)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=3, init=True)
    trial = MultiSlater(system, ham, wfn, init=init)
    trial.calculate_energy(system, ham)
    options = {'hybrid': False}
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = Continuous(system, ham, trial, qmc, options=options)
    walker = MultiDetWalker(system, ham, trial)
    for i in range(0,10):
        prop.propagate_walker(walker, system, ham, trial, trial.energy)
    assert walker.weight == pytest.approx(0.68797524675701)

@pytest.mark.unit
def test_hybrid():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5,5)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=3, init=True)
    trial = MultiSlater(system, ham, wfn, init=init)
    trial.calculate_energy(system, ham)
    options = {'hybrid': True}
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = Continuous(system, ham, trial, qmc, options=options)
    walker = MultiDetWalker(system, ham, trial)
    for i in range(0,10):
        prop.propagate_walker(walker, system, ham, trial, trial.energy)

    assert walker.weight == pytest.approx(0.7430443466368197)

import numpy
import os
import pytest
from pauxy.systems.generic import Generic
from pauxy.trial_wavefunction.multi_slater import MultiSlater
from pauxy.propagation.generic import GenericContinuous
from pauxy.propagation.continuous import Continuous
from pauxy.utils.misc import dotdict
from pauxy.utils.testing import (
        generate_hamiltonian,
        get_random_nomsd,
        get_random_phmsd
        )
from pauxy.walkers.multi_det import MultiDetWalker

@pytest.mark.unit
def test_nomsd():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5,5)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec, h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    wfn = get_random_nomsd(system, ndet=3)
    trial = MultiSlater(system, wfn)
    walker = MultiDetWalker(system, trial)
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = GenericContinuous(system, trial, qmc)
    fb = prop.construct_force_bias(system, walker, trial)
    prop.construct_VHS(system, fb)

@pytest.mark.unit
def test_phmsd():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5,5)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec, h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system, ndet=3, init=True)
    trial = MultiSlater(system, wfn, init=init)
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = GenericContinuous(system, trial, qmc)
    walker = MultiDetWalker(system, trial)
    fb = prop.construct_force_bias(system, walker, trial)
    vhs = prop.construct_VHS(system, fb)

@pytest.mark.unit
def test_local_energy():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5,5)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec, h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system, ndet=3, init=True)
    trial = MultiSlater(system, wfn, init=init)
    trial.calculate_energy(system)
    options = {'hybrid': False}
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = Continuous(system, trial, qmc, options=options)
    walker = MultiDetWalker(system, trial)
    for i in range(0,10):
        prop.propagate_walker(walker, system, trial, trial.energy)
    assert walker.weight == pytest.approx(0.68797524675701)

@pytest.mark.unit
def test_hybrid():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5,5)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec, h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system, ndet=3, init=True)
    trial = MultiSlater(system, wfn, init=init)
    trial.calculate_energy(system)
    options = {'hybrid': True}
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = Continuous(system, trial, qmc, options=options)
    walker = MultiDetWalker(system, trial)
    for i in range(0,10):
        prop.propagate_walker(walker, system, trial, trial.energy)

    assert walker.weight == pytest.approx(0.7430443466368197)

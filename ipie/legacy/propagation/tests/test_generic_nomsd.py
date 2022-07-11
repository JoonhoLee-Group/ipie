import os

import numpy
import pytest

from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.legacy.estimators.greens_function import gab, gab_spin
from ipie.legacy.propagation.continuous import Continuous
from ipie.legacy.propagation.generic import GenericContinuous
from ipie.legacy.trial_wavefunction.multi_slater import MultiSlater
from ipie.legacy.walkers.multi_det import MultiDetWalker
from ipie.systems.generic import Generic
from ipie.utils.misc import dotdict
from ipie.utils.testing import (generate_hamiltonian, get_random_nomsd,
                                get_random_phmsd)


@pytest.mark.unit
def test_nomsd():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5, 5)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
    )
    wfn = get_random_nomsd(system.nup, system.ndown, ham.nbasis, ndet=3)
    trial = MultiSlater(system, ham, wfn)
    walker = MultiDetWalker(system, ham, trial)
    qmc = dotdict({"dt": 0.005, "nstblz": 5})
    prop = GenericContinuous(system, ham, trial, qmc)
    fb = prop.construct_force_bias(ham, walker, trial)
    prop.construct_VHS(ham, fb)


@pytest.mark.unit
def test_hybrid():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5, 5)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
    )
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(
        system.nup, system.ndown, ham.nbasis, ndet=3, init=True
    )
    trial = MultiSlater(system, ham, wfn, init=init)
    trial.calculate_energy(system, ham)
    options = {"hybrid": True}
    qmc = dotdict({"dt": 0.005, "nstblz": 5})
    prop = Continuous(system, ham, trial, qmc, options=options)
    walker = MultiDetWalker(system, ham, trial)
    for i in range(0, 10):
        prop.propagate_walker(walker, system, ham, trial, trial.energy)
    assert walker.weight == pytest.approx(
        0.7540668958301742
    )  # new value after dixing contract_one_body bug
    # assert walker.weight == pytest.approx(0.8296101502964913) # new value after fixing the force bias and contract_one_body bug


if __name__ == "__main__":
    test_nomsd()
    test_hybrid()

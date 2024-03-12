import pytest
import numpy

from ipie.addons.eph.utils.testing import get_random_sys_holstein
from ipie.addons.eph.trial_wavefunction.toyozawa import ToyozawaTrial
from ipie.addons.eph.trial_wavefunction.variational.toyozawa_variational import (
    variational_trial_toyozawa,
)
from ipie.addons.eph.trial_wavefunction.coherent_state import CoherentStateTrial
from ipie.addons.eph.trial_wavefunction.variational.coherent_state_variational import (
    variational_trial,
)


@pytest.mark.unit
def test_variational_energy_toyozawa():
    seed = 7
    numpy.random.seed(seed)
    nelec = (2, 2)
    nbasis = 4
    pbc = True
    sys, ham = get_random_sys_holstein(nelec, nbasis, pbc)
    ie = numpy.random.random((nbasis, nelec[0] + nelec[1]))
    ip = numpy.random.random(nbasis)
    etrial, p, e = variational_trial_toyozawa(ip, ie, ham, sys, verbose=-1)

    wfn = numpy.column_stack([p, e])
    trial = ToyozawaTrial(wavefunction=wfn, w0=ham.w0, num_elec=nelec, num_basis=nbasis)
    trial.set_etrial(ham)
    assert etrial == pytest.approx(trial.energy)


@pytest.mark.unit
def test_variational_energy_coherent_state():
    seed = 7
    numpy.random.seed(seed)
    nelec = (2, 2)
    nbasis = 4
    pbc = True
    sys, ham = get_random_sys_holstein(nelec, nbasis, pbc)
    ie = numpy.random.random((nbasis, nelec[0] + nelec[1]))
    ip = numpy.random.random(nbasis)
    etrial, p, e = variational_trial(ip, ie, ham, sys)

    wfn = numpy.column_stack([p, e])
    trial = CoherentStateTrial(wavefunction=wfn, w0=ham.w0, num_elec=nelec, num_basis=nbasis)
    trial.set_etrial(ham)
    assert etrial == pytest.approx(trial.energy)


if __name__ == "__main__":
    test_variational_energy_toyozawa()
    test_variational_energy_coherent_state()

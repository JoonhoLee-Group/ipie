import tempfile

import pytest

from ipie.addons.eph.estimators.energy import EnergyEstimator
from ipie.addons.eph.utils.testing import gen_random_test_instances


@pytest.mark.unit
def test_energy_estimator():
    pbc = True
    nbasis = 4
    nelec = (2, 2)
    trial_type = "toyozawa"
    nwalkers = 500
    sys, ham, walkers, trial = gen_random_test_instances(nelec, nbasis, nwalkers, trial_type)
    estim = EnergyEstimator(sys, ham, trial)
    estim.compute_estimator(sys, walkers, ham, trial)
    assert len(estim.names) == 6
    assert estim["ENumer"].real == pytest.approx(-3136.7469620055163)
    assert estim["ETotal"] == pytest.approx(0.0)
    tmp = estim.data.copy()
    estim.post_reduce_hook(tmp)
    assert tmp[estim.get_index("ETotal")].real == pytest.approx(-6.273493924011032)
    assert estim.print_to_stdout
    assert estim.ascii_filename == None
    assert estim.shape == (6,)


if __name__ == "__main__":
    test_energy_estimator()

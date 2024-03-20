import pytest
import numpy

from ipie.addons.eph.utils.testing import build_random_trial, get_random_wavefunction
from ipie.addons.eph.walkers.eph_walkers import EPhWalkers


@pytest.mark.unit
def test_trial_methods():
    seed = 7
    numpy.random.seed(seed)
    nelec = (2, 2)
    nbasis = 4
    trial_type = "coherent_state"
    pbc = True
    w0 = numpy.random.rand()
    nwalkers = 100

    trial = build_random_trial(nelec, nbasis, w0, trial_type)

    wfn = get_random_wavefunction(nelec, nbasis)
    walkers = EPhWalkers(wfn, nelec[0], nelec[1], nbasis, nwalkers)
    walkers.build(trial)

    assert trial.calc_overlap(walkers)[0] == pytest.approx(0.0008183683599516786)
    assert trial.calc_phonon_overlap(walkers)[0] == pytest.approx(0.8042312422253469)
    assert trial.calc_phonon_gradient(walkers)[0, 0] == pytest.approx(-0.170210708173531)
    assert trial.calc_phonon_laplacian(walkers)[0] == pytest.approx(-3.5642631271035823)
    assert trial.calc_electronic_overlap(walkers)[0] == pytest.approx(0.0010175784239458462)
    assert trial.calc_greens_function(walkers)[0][0, 0, 0] == pytest.approx(2.759966097679107)


if __name__ == "__main__":
    test_trial_methods()

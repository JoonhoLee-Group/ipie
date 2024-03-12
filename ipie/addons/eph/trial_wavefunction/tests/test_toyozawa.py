import pytest
import numpy

from ipie.addons.eph.utils.testing import build_random_trial, get_random_wavefunction
from ipie.addons.eph.walkers.eph_walkers import EPhWalkers


@pytest.mark.unit
def test_trial_methods():
    seed = 9
    numpy.random.seed(seed)
    nelec = (2, 2)
    nbasis = 4
    trial_type = "toyozawa"
    pbc = True
    w0 = numpy.random.rand()
    nwalkers = 100

    trial = build_random_trial(nelec, nbasis, w0, trial_type)

    wfn = get_random_wavefunction(nelec, nbasis)
    walkers = EPhWalkers(wfn, nelec[0], nelec[1], nbasis, nwalkers)
    walkers.build(trial)

    assert trial.calc_overlap(walkers)[0] == pytest.approx(0.05496697699720597)
    assert trial.calc_phonon_overlap(walkers)[0] == pytest.approx(3.3244959521904325)
    assert trial.calc_phonon_gradient(walkers)[0, 0] == pytest.approx(-0.028056870680617366)
    assert trial.calc_phonon_laplacian(walkers, walkers.ovlp_perm)[0] == pytest.approx(
        -3.672010193624128
    )
    assert trial.calc_electronic_overlap(walkers)[0] == pytest.approx(0.06480736261172242)
    assert trial.calc_greens_function(walkers)[0][0, 0, 0] == pytest.approx(0.3341677951334292)


if __name__ == "__main__":
    test_trial_methods()

import numpy
import pytest
from ipie.addons.eph.walkers.eph_walkers import EPhWalkers
from ipie.addons.eph.utils.testing import get_random_wavefunction, build_random_trial


@pytest.mark.unit
def test_ephwalkers():
    nwalkers = 100
    nelec = (3, 3)
    nbasis = 6
    wfn = get_random_wavefunction(nelec, nbasis)
    walkers = EPhWalkers(wfn, nelec[0], nelec[1], nbasis, nwalkers)
    assert numpy.allclose(walkers.phonon_disp, wfn[:, 0])
    assert numpy.allclose(walkers.phia, wfn[:, 1 : 1 + nelec[0]].reshape(nbasis, nelec[0]))
    assert numpy.allclose(walkers.phib, wfn[:, 1 + nelec[0] :].reshape(nbasis, nelec[1]))


@pytest.mark.unit
def test_overlap_init():
    nwalkers = 100
    nelec = (3, 3)
    nbasis = 6
    seed = 7
    numpy.random.seed(7)
    w0 = numpy.random.rand()
    trial_type = "coherent_state"
    trial = build_random_trial(nelec, nbasis, w0, trial_type)
    wfn = get_random_wavefunction(nelec, nbasis)
    walkers = EPhWalkers(wfn, nelec[0], nelec[1], nbasis, nwalkers)
    walkers.build(trial)
    assert len(walkers.buff_names) == 11
    assert walkers.ovlp[0].real == pytest.approx(0.0007324519852172784)
    assert walkers.ph_ovlp[0].real == pytest.approx(0.7141097126634587)
    assert walkers.el_ovlp[0].real == pytest.approx(0.0010256855105434813)


if __name__ == "__main__":
    test_ephwalkers()
    test_overlap_init()

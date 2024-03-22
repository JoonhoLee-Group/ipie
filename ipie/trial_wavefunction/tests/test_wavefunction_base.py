import pytest

from ipie.utils.testing import get_random_phmsd_opt
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase


# @pytest.mark.unit # disabled because abstract class can't be instantiated directly
def test_wavefunction_base():
    num_basis = 10
    nalpha, nbeta = (5, 5)
    wavefunction, _ = get_random_phmsd_opt(nalpha, nbeta, num_basis, ndet=100)
    trial = TrialWavefunctionBase(
        wavefunction,
        (nalpha, nbeta),
        num_basis,
    )
    assert trial.nelec == (nalpha, nbeta)
    assert trial.nbasis == num_basis


if __name__ == '__main__':
    test_wavefunction_base()

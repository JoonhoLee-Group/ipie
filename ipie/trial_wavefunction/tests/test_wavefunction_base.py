import pytest

from ipie.utils.testing import get_random_phmsd_opt
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase

@pytest.mark.unit
def test_wavefunction_base():
    num_basis = 10
    nalpha, nbeta = (5, 5)
    wavefunction, _ = get_random_phmsd_opt(nalpha, nbeta, num_basis, ndet=100)
    trial = TrialWavefunctionBase(
            wavefunction,
            (nalpha, nbeta),
            num_basis,
            )
    assert trial.num_elec == (nalpha, nbeta)
    assert trial.num_basis == num_basis

import numpy as np
import pytest

from ipie.estimators.energy import EnergyEstimator
from ipie.estimators.handler import (
        EstimatorHandler,
        EstimatorHelper
        )
from ipie.utils.testing import gen_random_test_instances

@pytest.mark.unit
def test_energy_estimator():
    nmo = 10
    nocc = 8
    naux = 30
    nwalker = 10
    system, ham, walker_batch, trial = gen_random_test_instances(nmo, nocc, naux, nwalker)
    estim = EnergyEstimator(system=system, ham=ham, trial=trial)
    estim.compute_estimator(system, walker_batch, ham, trial)
    assert len(estim.names) == 7
    data = estim.data
    assert data[0] == pytest.approx(10.0)
    assert data[2].real == pytest.approx(-967.611820215707)
    assert data[estim.get_index('ETotal')] == pytest.approx(0.0)
    estim.post_reduce_hook(data)
    assert data[estim.get_index('ETotal')] == pytest.approx(-96.7611820215707+303.8425744212367j)
    assert estim.print_to_stdout
    assert estim.ascii_filename == None
    assert estim.shape == (7,)
    header = estim.header_to_text
    data_to_text = estim.data_to_text(data)
    print("")
    print(header)
    print(data_to_text)

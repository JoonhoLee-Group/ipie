import h5py
import json
import numpy as np
import os
import pytest

from ipie.estimators.utils import H5EstimatorHelper
from ipie.analysis.blocking import (
        analyse_estimates,
        reblock_minimal
        )


@pytest.mark.unit
def test_analyse_estimates():
    np.random.seed(7)
    nestim = 11
    nblock = 500
    data = np.random.random((nblock, nestim)) + 1j*np.random.random((nblock, nestim))
    header = [
        "Iteration",
        "WeightFactor",
        "Weight",
        "ENumer",
        "EDenom",
        "ETotal",
        "E1Body",
        "E2Body",
        "EHybrid",
        "Overlap",
        "Time",
    ]
    # fake some data
    metadata = {
            'system': {
                'nbasis': 1,
                'name': 'Generic',
                'nup': 1,
                'ndown': 1,
                'integral_file': 'none'
                },
            'qmc': {
                'nsteps': 1,
                'dt': 0.05
                },
            'propagators': {
                'free_projection': False
                },
            'trial': {
                'energy': -1.0
                }
            }
    with h5py.File('test.h5', 'w') as fh5:
        fh5["basic/headers"] = np.array(header).astype("S")
        fh5["metadata"] = json.dumps(metadata)
    helper = H5EstimatorHelper('test.h5', 'basic')
    for i in range(nblock):
        helper.push(data[i], "energies")
        helper.increment()

    data = analyse_estimates(['test.h5'], start_time=0)
    assert np.allclose(
            data['ETotal'].values,
            [4.8279208658716e-01]
            )
    assert np.allclose(
            data['ETotal_ac'].values,
            [4.832190780587988e-01]
            )
    assert np.allclose(
            data['ETotal_error'].values[0],
            [1.1124618641309e-02]
            )
    assert np.allclose(
            data['ETotal_error_ac'].values,
            [1.301950758507343e-02]
            )

@pytest.mark.unit
def test_analyse_estimates_textfile():
    np.random.seed(7)
    nestim = 11
    nblock = 500
    data = np.random.random((nblock, nestim)) + 1j*np.random.random((nblock, nestim))
    header = [
        "Iteration",
        "WeightFactor",
        "Weight",
        "ENumer",
        "EDenom",
        "ETotal",
        "E1Body",
        "E2Body",
        "EHybrid",
        "Overlap",
        "Time",
    ]
    # fake some data
    with open('test.dat', 'w') as f:
        f.write(' '.join(h for h in header) + '\n')
        for i in range(nblock):
            f.write(' '.join("{:13.10e}".format(val.real) for val in data[i]) + '\n')
        f.write("# End Time\n")

    data = reblock_minimal(['test.dat'], start_block=1)

    assert np.allclose(
            data['ETotal_ac'].values,
            [4.832190780587988e-01]
            )
    assert np.allclose(
            data['ETotal_error_ac'].values,
            [1.301950758507343e-02]
            )

def teardown_module():
    cwd = os.getcwd()
    files = ["test.h5", "test.dat", "analysed_test.h5"]
    for f in files:
        try:
            os.remove(cwd + "/" + f)
        except OSError:
            pass

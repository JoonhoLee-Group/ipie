import tempfile

import numpy
import pytest

from ipie.analysis.extraction import extract_data
from ipie.config import MPI

try:
    from ipie.legacy.qmc.thermal_afqmc import ThermalAFQMC

    _no_cython = False
except ModuleNotFoundError:
    _no_cython = True


@pytest.mark.skipif(_no_cython, reason="Need to build cython modules.")
@pytest.mark.driver
def test_ueg():
    with tempfile.NamedTemporaryFile() as tmpf:
        options = {
            "get_sha1": False,
            "qmc": {
                "timestep": 0.05,
                "rng_seed": 8,
                "nblocks": 1,
                "nwalkers": 10,
                "beta": 0.5,
                "pop_control_freq": 1,
                "stabilise_freq": 10,
            },
            "model": {
                "name": "UEG",
                "rs": 1.0,
                "ecut": 4,
                "nup": 1,
                "ndown": 1,
            },
            "hamiltonian": {"name": "UEG", "mu": 0.245},
            "trial": {"name": "one_body"},
            "estimates": {"filename": tmpf.name},
            "walkers": {"low_rank": True, "pop_control": "comb", "low_rank_thresh": 1e-6},
        }
        comm = MPI.COMM_WORLD
        afqmc = ThermalAFQMC(comm=comm, options=options, verbose=0)
        afqmc.run(comm=comm)
        afqmc.finalise(verbose=0)
        data = extract_data(afqmc.estimators.filename, "basic", "energies")
        numpy.testing.assert_almost_equal(
            numpy.real(data.WeightFactor.values), numpy.array([10.0, 9.8826616])
        )
        numpy.testing.assert_almost_equal(
            numpy.real(data.Nav.values), numpy.array([1.99999991, 2.5848349])
        )
        numpy.testing.assert_almost_equal(
            numpy.real(data.ETotal.values), numpy.array([5.97385568, 8.1896957])
        )


if __name__ == "__main__":
    test_ueg()

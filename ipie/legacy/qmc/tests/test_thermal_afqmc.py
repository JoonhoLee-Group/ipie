import os

import numpy
import pytest
from mpi4py import MPI

from ipie.analysis.extraction import extract_data
from ipie.legacy.qmc.thermal_afqmc import ThermalAFQMC
from ipie.legacy.systems.ueg import UEG
from ipie.legacy.trial_wavefunction.hartree_fock import HartreeFock
from ipie.qmc.calc import setup_calculation
from ipie.systems.generic import Generic
from ipie.utils.testing import generate_hamiltonian


@pytest.mark.driver
def test_ueg():
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
        "estimates": {"filename": "estimates.test_thermal_ueg.h5"},
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


def teardown_module(self):
    cwd = os.getcwd()
    files = ["estimates.test_thermal_ueg.h5"]
    for f in files:
        try:
            os.remove(cwd + "/" + f)
        except OSError:
            pass


if __name__ == "__main__":
    test_ueg()

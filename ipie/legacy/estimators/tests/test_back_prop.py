import os

import numpy
import pytest

from ipie.legacy.estimators.back_propagation import BackPropagation
from ipie.legacy.hamiltonians.ueg import UEG as HamUEG
from ipie.legacy.propagation.continuous import Continuous
from ipie.legacy.systems.ueg import UEG
from ipie.legacy.trial_wavefunction.hartree_fock import HartreeFock
from ipie.legacy.walkers.handler import Walkers
from ipie.utils.misc import dotdict


@pytest.mark.unit
def test_back_prop():
    sys = UEG({"rs": 2, "nup": 7, "ndown": 7, "ecut": 1.0})
    ham = HamUEG(sys, {"rs": 2, "nup": 7, "ndown": 7, "ecut": 1.0})
    bp_opt = {"tau_bp": 1.0, "nsplit": 4}
    qmc = dotdict({"dt": 0.05, "nstblz": 10, "nwalkers": 1})
    trial = HartreeFock(sys, ham, {})
    numpy.random.seed(8)
    prop = Continuous(sys, ham, trial, qmc)
    est = BackPropagation(
        bp_opt, True, "estimates.0.h5", qmc, sys, trial, numpy.complex128, prop.BT_BP
    )
    walkers = Walkers(
        sys, ham, trial, qmc, walker_opts={}, nbp=est.nmax, nprop_tot=est.nmax
    )
    wlk = walkers.walkers[0]
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    for i in range(0, 2 * est.nmax):
        prop.propagate_walker(wlk, sys, ham, trial, 0)
        if i % 10 == 0:
            walkers.orthogonalise(trial, False)
        est.update_uhf(qmc, sys, ham, trial, walkers, 100)
        est.print_step(comm, comm.size, i, 10)


def teardown_module(self):
    cwd = os.getcwd()
    files = ["estimates.0.h5"]
    for f in files:
        try:
            os.remove(cwd + "/" + f)
        except OSError:
            pass


if __name__ == "__main__":
    test_back_prop()

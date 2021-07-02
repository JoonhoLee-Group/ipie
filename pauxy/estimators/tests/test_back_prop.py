import pytest
import numpy
import os
from pauxy.systems.ueg import UEG
from pauxy.utils.misc import dotdict
from pauxy.trial_wavefunction.hartree_fock import HartreeFock
from pauxy.estimators.back_propagation import BackPropagation
from pauxy.propagation.continuous import Continuous
from pauxy.walkers.handler import Walkers

@pytest.mark.unit
def test_back_prop():
    sys = UEG({'rs': 2, 'nup': 7, 'ndown': 7, 'ecut': 1.0})
    bp_opt = {'tau_bp': 1.0, 'nsplit': 4}
    qmc = dotdict({'dt': 0.05, 'nstblz': 10, 'nwalkers': 1})
    trial = HartreeFock(sys, {})
    numpy.random.seed(8)
    prop = Continuous(sys, trial, qmc)
    est = BackPropagation(bp_opt, True, 'estimates.0.h5', qmc, sys, trial,
                          numpy.complex128, prop.BT_BP)
    walkers = Walkers(sys, trial, qmc, walker_opts={}, nbp=est.nmax, nprop_tot=est.nmax)
    wlk = walkers.walkers[0]
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    for i in range(0, 2*est.nmax):
        prop.propagate_walker(wlk, sys, trial, 0)
        if i % 10 == 0:
            walkers.orthogonalise(trial, False)
        est.update_uhf(sys, qmc, trial, walkers, 100)
        est.print_step(comm, comm.size, i, 10)

def teardown_module(self):
    cwd = os.getcwd()
    files = ['estimates.0.h5']
    for f in files:
        try:
            os.remove(cwd+'/'+f)
        except OSError:
            pass

if __name__ == '__main__':
    test_back_prop()

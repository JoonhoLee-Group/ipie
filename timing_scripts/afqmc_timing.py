import mpi4py
import numpy

from ipie.hamiltonians.utils import get_hamiltonian
from ipie.legacy.estimators.local_energy import local_energy
from ipie.legacy.qmc.afqmc import AFQMC
from ipie.legacy.walkers.single_det import SingleDetWalker
from ipie.propagation.utils import get_propagator_driver
from ipie.qmc.options import QMCOpts
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.utils import get_trial_wavefunction
from ipie.utils.io import get_input_value
from ipie.utils.mpi import get_shared_comm

mpi4py.rc.recv_mprobe = False
import cProfile

from mpi4py import MPI

nelec = (94, 92)
options = {
    "system": {
        "nup": 94,
        "ndown": 92,
    },
    "hamiltonian": {"name": "Generic", "integrals": "afqmc.h5"},
    "qmc": {"dt": 0.005, "nwalkers": 1, "blocks": 1},
    "trial": {"filename": "afqmc.h5"},
    "estimators": {},
}
numpy.random.seed(7)
sys = Generic(nelec=nelec)
comm = MPI.COMM_WORLD
verbose = True
shared_comm = get_shared_comm(comm, verbose=verbose)

qmc_opts = get_input_value(options, "qmc", default={}, verbose=verbose)
ham_opts = get_input_value(options, "hamiltonian", default={}, verbose=verbose)
twf_opts = get_input_value(options, "trial", default={}, verbose=verbose)
prop_opts = get_input_value(options, "propoagator", default={}, verbose=verbose)
qmc = QMCOpts(qmc_opts, sys, verbose=True)
ham = get_hamiltonian(sys, ham_opts, verbose=True, comm=shared_comm)

trial = get_trial_wavefunction(
    sys, ham, options=twf_opts, comm=comm, scomm=shared_comm, verbose=verbose
)
trial.psi = trial.psi[0]  # Super hacky thing to do; this needs to be fixed...
trial.calculate_energy(sys, ham)  # this is to get the energy shift

prop = get_propagator_driver(sys, ham, trial, qmc, options=prop_opts, verbose=verbose)

walker = SingleDetWalker(sys, ham, trial)

nsteps = 2

pr = cProfile.Profile()
pr.enable()

for i in range(nsteps):
    prop.propagate_walker(walker, sys, ham, trial, trial.energy)
    detR = walker.reortho(trial)  # reorthogonalizing to stablize

pr.disable()
pr.print_stats(sort="cumtime")

walker.greens_function(trial)  # Green's function gets updated
etot = local_energy(sys, ham, walker, trial)[0]

print("a sample of local_energy = {}".format(etot))
# a sample of local_energy = (-2244.6424862764557+0.00047855044660360946j)

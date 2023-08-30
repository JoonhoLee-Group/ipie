"""Helper Routines for setting up a calculation"""
# todo : handle more gracefully.

try:
    import mpi4py

    mpi4py.rc.recv_mprobe = False
    from ipie.config import MPI

    # import dill
    # MPI.pickle.__init__(dill.dumps, dill.loads)
    parallel = True
except ImportError:
    parallel = False
from ipie.thermal.qmc.thermal_afqmc import ThermalAFQMC
from ipie.utils.io import get_input_value


def get_driver(options, comm):
    verbosity = options.get("verbosity", 1)
    qmc_opts = get_input_value(options, "qmc", default={}, alias=["qmc_options"])
    beta = get_input_value(qmc_opts, "beta", default=None)
    batched = get_input_value(qmc_opts, "batched", default=True)  # ,
    # verbose=verbosity)
    if beta is not None:
        afqmc = ThermalAFQMC(comm, options=options, parallel=comm.size > 1, verbose=verbosity)
    else:
        raise NotImplementedError

    return afqmc




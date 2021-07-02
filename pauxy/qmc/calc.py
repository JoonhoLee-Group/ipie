"""Helper Routines for setting up a calculation"""
# todo : handle more gracefully.
import time
import numpy
import json
import h5py
import sys
try:
    import mpi4py
    mpi4py.rc.recv_mprobe = False
    from mpi4py import MPI
    # import dill
    # MPI.pickle.__init__(dill.dumps, dill.loads)
    parallel = True
except ImportError:
    parallel = False
from pauxy.qmc.afqmc import AFQMC
from pauxy.qmc.thermal_afqmc import ThermalAFQMC
from pauxy.estimators.handler import Estimators
from pauxy.utils.io import  to_json, get_input_value
from pauxy.utils.misc import serialise
from pauxy.walkers.handler import Walkers
from pauxy.qmc.comm import FakeComm


def init_communicator():
    if parallel:
        comm = MPI.COMM_WORLD
    else:
        comm = FakeComm()
    return comm

def setup_calculation(input_options):
    comm = init_communicator()
    if isinstance(input_options, str):
        options = read_input(input_options, comm, verbose=True)
    else:
        options = input_options
    afqmc = get_driver(options, comm)
    return (afqmc, comm)

def get_driver(options, comm):
    qmc_opts = get_input_value(options, 'qmc', default={},
                               alias=['qmc_options'])
    beta = get_input_value(qmc_opts, 'beta', default=None)
    verbosity = options.get('verbosity', 1)
    if beta is not None:
        afqmc = ThermalAFQMC(comm, options=options,
                             parallel=comm.size>1,
                             verbose=verbosity)
    else:
        afqmc = AFQMC(comm, options=options,
                      parallel=comm.size>1,
                      verbose=verbosity)
    return afqmc

def read_input(input_file, comm, verbose=False):
    """Helper function to parse input file and setup parallel calculation.

    Parameters
    ----------
    input_file : string
        Input filename.
    verbose : bool
        If true print out set up information.

    Returns
    -------
    options : dict
        Python dict of input options.
    comm : MPI communicator
        Communicator object. If mpi4py is not installed then we return a fake
        communicator.
    """
    if comm.rank == 0:
        if verbose:
            print('# Initialising PAUXY simulation from %s'%input_file)
        with open(input_file) as inp:
            options = json.load(inp)
        inp.close()
    else:
        options = None
    options = comm.bcast(options, root=0)

    return options


def set_rng_seed(seed, comm):
    if seed is None:
        # only set "random" part of seed on parent processor so we can reproduce
        # results in when running in parallel.
        if comm.rank == 0:
            seed = numpy.array([numpy.random.randint(0, 1e8)], dtype='i4')
            # Can't directly json serialise numpy arrays
            qmc_opts['rng_seed'] = seed[0].item()
        else:
            seed = numpy.empty(1, dtype='i4')
        comm.Bcast(seed, root=0)
        seed = seed[0]
    seed = seed + comm.rank
    numpy.random.seed(seed)
    return seed


def setup_parallel(options, comm=None, verbose=False):
    """Wrapper routine for initialising simulation

    Parameters
    ----------
    options : dict
        Input options.
    comm : MPI communicator
        MPI communicator object.
    verbose : bool
        If true print out set up information.

    Returns
    -------
    afqmc : :class:`pauxy.afqmc.CPMC`
        CPMC driver.
    """
    if comm.rank == 0:
        afqmc = get_driver(options, comm)
        system = afqmc.system
        print("# Setup base driver.")
    else:
        afqmc = None
        system = None
    system = comm.bcast(system)
    afqmc = comm.bcast(afqmc, root=0)
    afqmc.init_time = time.time()
    if afqmc.trial.error:
        print("# Error in constructing trial wavefunction. Exiting")
        sys.exit()
    afqmc.rank = comm.Get_rank()
    afqmc.nprocs = comm.Get_size()
    afqmc.root = afqmc.rank == 0
    # We can't serialise '_io.BufferWriter' object, so just delay initialisation
    # of estimators object to after MPI communication.
    # Simpler to just ensure a fixed number of walkers per core.
    afqmc.qmc.nwalkers = int(afqmc.qmc.nwalkers/afqmc.nprocs)
    afqmc.qmc.ntot_walkers = afqmc.qmc.nwalkers * afqmc.nprocs
    if afqmc.qmc.nwalkers == 0:
        # This should occur on all processors so we don't need to worry about
        # race conditions / mpi4py hanging.
        if afqmc.root and afqmc.verbosity > 1:
            print("# WARNING: Not enough walkers for selected core count."
                  "There must be at least one walker per core set in the "
                  "input file. Setting one walker per core.")
        afqmc.qmc.nwalkers = 1

    estimator_opts = options.get('estimates', {})
    walker_opts = options.get('walkers', {'weight': 1})
    estimator_opts['stack_size'] = walker_opts.get('stack_size', 1)
    afqmc.estimators = (
        Estimators(estimator_opts,
                   afqmc.root,
                   afqmc.qmc,
                   afqmc.system,
                   afqmc.trial,
                   afqmc.propagators.BT_BP,
                   verbose=(comm.rank==0 and verbose))
    )
    afqmc.psi = Walkers(walker_opts, afqmc.system,
                        afqmc.trial,
                        afqmc.qmc,
                        verbose=(comm.rank==0 and verbose),
                        comm=comm)
    afqmc.psi.add_field_config(afqmc.estimators.nprop_tot,
                               afqmc.estimators.nbp,
                               afqmc.system,
                               numpy.complex128)
    if comm.rank == 0:
        json_string = to_json(afqmc)
        afqmc.estimators.json_string = json_string
        afqmc.estimators.dump_metadata()
        afqmc.estimators.estimators['mixed'].print_key()
        afqmc.estimators.estimators['mixed'].print_header()

    return afqmc

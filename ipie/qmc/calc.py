# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Fionn Malone <fmalone@google.com>
#          Joonho Lee
#

"""Helper Routines for setting up a calculation"""
# todo : handle more gracefully.
import json
import sys
import time

import h5py
import numpy

try:
    import mpi4py

    mpi4py.rc.recv_mprobe = False
    from mpi4py import MPI

    # import dill
    # MPI.pickle.__init__(dill.dumps, dill.loads)
    parallel = True
except ImportError:
    parallel = False

from ipie.estimators.handler import EstimatorHandler
from ipie.qmc.afqmc import AFQMC
from ipie.qmc.comm import FakeComm
from ipie.utils.io import get_input_value, to_json
from ipie.utils.misc import serialise

from ipie.systems.utils import get_system
from ipie.hamiltonians.utils import get_hamiltonian
from ipie.trial_wavefunction.utils import get_trial_wavefunction
from ipie.walkers.uhf_walkers import UHFWalkersTrial, get_initial_walker
from ipie.utils.mpi import MPIHandler

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


def get_driver(options: dict, comm: MPI.COMM_WORLD) -> AFQMC:
    verbosity = options.get("verbosity", 1)
    qmc_opts = get_input_value(options, "qmc", default={}, alias=["qmc_options"])
    sys_opts = get_input_value(
        options,
        "system",
        default={},
        alias=["model"],
        verbose=verbosity > 1,
    )
    ham_opts = get_input_value(
        options, "hamiltonian", default={}, verbose=verbosity > 1
    )
    # backward compatibility with previous code (to be removed)
    for item in sys_opts.items():
        if item[0].lower() == "name" and "name" in ham_opts.keys():
            continue
        ham_opts[item[0]] = item[1]

    twf_opt = get_input_value(
        options,
        "trial",
        default={},
        alias=["trial_wavefunction"],
        verbose=verbosity > 1,
    )

    wlk_opts = get_input_value(
        options,
        "walkers",
        default={},
        alias=["walker", "walker_opts"],
        verbose=verbosity > 1,
    )
    wlk_opts["pop_control"] = wlk_opts.get("pop_control", "pair_branch")
    wlk_opts["population_control"] = wlk_opts["pop_control"]

    beta = get_input_value(qmc_opts, "beta", default=None)
    if comm.rank != 0:
        verbosity = 0
    batched = get_input_value(qmc_opts, "batched", default=True, verbose=verbosity)

    if beta is not None or batched == False:
        raise ValueError("Trying to use legacy features which aren't supported.")
    else:
        from ipie.qmc.options import QMCOpts
        qmc = QMCOpts(qmc_opts, verbose=0)
        if qmc.nwalkers is None:
            assert qmc.nwalkers_per_task is not None
            qmc.nwalkers = qmc.nwalkers_per_task * comm.size
        if qmc.nwalkers_per_task is None:
            assert qmc.nwalkers is not None
            qmc.nwalkers_per_task = int(qmc.nwalkers / comm.size)

        mpi_handler = MPIHandler(comm, qmc_opts, verbose=verbosity)
        system = get_system(sys_opts, verbose=verbosity, comm=comm)
        # Have to deal with shared comm in the future. I think we will remove this...
        hamiltonian = get_hamiltonian(
            system, ham_opts, verbose=verbosity, comm=comm
        )
        trial = get_trial_wavefunction(
            system,
            hamiltonian,
            options=twf_opt,
            comm=comm,
            scomm=comm,
            verbose=verbosity,
        )
        ndets, initial_walker = get_initial_walker(trial)
        walkers = UHFWalkersTrial[type(trial)](initial_walker, system.nup, system.ndown, hamiltonian.nbasis,
                                         qmc.nwalkers_per_task, qmc.nwalkers, qmc.nsteps,
                                         ndets=ndets,
                                         mpi_handler = mpi_handler, pop_control_method="pair_branch")
        walkers.build(trial) # any intermediates that require information from trial
        afqmc = AFQMC(
            comm,  options=options, 
            system=system,
            hamiltonian=hamiltonian,
            trial=trial,
            walkers=walkers,
            seed=qmc.rng_seed, 
            nwalkers=qmc.nwalkers,
            nwalkers_per_task=qmc.nwalkers_per_task,
            num_steps_per_block=qmc.nsteps, 
            num_blocks=qmc.nblocks, 
            timestep=qmc.dt,
            stabilise_freq=qmc.nstblz,
            pop_control_freq=qmc.npop_control,
            verbose=verbosity
        )

    return afqmc


def build_afqmc_driver(
    comm: mpi4py.MPI.Intracomm,
    nelec: tuple,
    wavefunction_file: str = "wavefunction.h5",
    hamiltonian_file: str = "hamiltonian.h5",
    num_walkers_per_task: int = 10,
    estimator_filename: str = "estimates.0.h5",
    verbosity: int = 0,
):
    if comm.rank != 0:
        verbosity = 0
    options = {
        "system": {
            "nup": nelec[0],
            "ndown": nelec[1],
        },
        "qmc": {"nwalkers_per_task": num_walkers_per_task},
        "hamiltonian": {"integrals": hamiltonian_file},
        "trial": {"filename": wavefunction_file},
        "estimators": {"overwrite": True, "filename": estimator_filename},
    }
    return get_driver(options,comm)


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
            print("# Initialising pie simulation from %s" % input_file)
        try:
            with open(input_file) as inp:
                options = json.load(inp)
        except FileNotFoundError:
            options = None
    else:
        options = None
    options = comm.bcast(options, root=0)
    if options == None:
        raise FileNotFoundError

    return options


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
    afqmc : :class:`pie.afqmc.CPMC`
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
    afqmc.qmc.nwalkers = int(afqmc.qmc.nwalkers / afqmc.nprocs)
    afqmc.qmc.ntot_walkers = afqmc.qmc.nwalkers * afqmc.nprocs
    if afqmc.qmc.nwalkers == 0:
        # This should occur on all processors so we don't need to worry about
        # race conditions / mpi4py hanging.
        if afqmc.root and afqmc.verbosity > 1:
            print(
                "# WARNING: Not enough walkers for selected core count."
                "There must be at least one walker per core set in the "
                "input file. Setting one walker per core."
            )
        afqmc.qmc.nwalkers = 1

    estimator_opts = options.get("estimates", {})
    walker_opts = options.get("walkers", {"weight": 1})
    afqmc.estimators = EstimatorHandler(
        comm,
        afqmc.system,
        afqmc.hamiltonian,
        afqmc.trial,
        options=estimator_opts,
        verbose=(comm.rank == 0 and verbose),
    )
    afqmc.psi = Walkers(
        walker_opts,
        afqmc.system,
        afqmc.trial,
        afqmc.qmc,
        verbose=(comm.rank == 0 and verbose),
        comm=comm,
    )
    afqmc.psi.add_field_config(
        afqmc.estimators.nprop_tot, afqmc.estimators.nbp, afqmc.system, numpy.complex128
    )
    if comm.rank == 0:
        json_string = to_json(afqmc)
        afqmc.estimators.json_string = json_string
        afqmc.estimators.dump_metadata()

    return afqmc

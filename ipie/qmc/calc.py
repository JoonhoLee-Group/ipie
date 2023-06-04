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

try:
    # TODO: WTF is this?
    import mpi4py

    mpi4py.rc.recv_mprobe = False
    from mpi4py import MPI

    parallel = True
except ImportError:
    parallel = False

from ipie.hamiltonians.utils import get_hamiltonian
from ipie.qmc.afqmc import AFQMC
from ipie.qmc.comm import FakeComm
from ipie.systems.utils import get_system
from ipie.trial_wavefunction.utils import get_trial_wavefunction
from ipie.utils.io import get_input_value
from ipie.utils.mpi import MPIHandler
from ipie.walkers.walkers_dispatch import get_initial_walker, UHFWalkersTrial


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
    ham_opts = get_input_value(options, "hamiltonian", default={}, verbose=verbosity > 1)
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
        mpi_handler = MPIHandler(comm, nmembers=qmc_opts.get("nmembers", 1), verbose=verbosity)
        system = get_system(
            sys_opts, verbose=verbosity, comm=comm
        )  # Have to deal with shared comm in the future. I think we will remove this...
        hamiltonian = get_hamiltonian(system, ham_opts, verbose=verbosity, comm=comm)
        wfn_file = get_input_value(twf_opt, "filename", default="", alias=["wfn_file"])
        trial = get_trial_wavefunction(
            system,
            hamiltonian,
            wfn_file,
            comm=comm,
            scomm=comm,
            verbose=verbosity,
            ndets=get_input_value(twf_opt, "ndets", default=1, alias=["num_dets"]),
            ndets_props=get_input_value(
                twf_opt, "ndets_props", default=1, alias=["num_dets_props"]
            ),
            ndet_chunks=get_input_value(
                twf_opt, "ndet_chunks", default=1, alias=["num_det_chunks"]
            ),
        )
        _, initial_walker = get_initial_walker(trial)
        walkers = UHFWalkersTrial(
            trial,
            initial_walker,
            system.nup,
            system.ndown,
            hamiltonian.nbasis,
            qmc.nwalkers,
            mpi_handler=mpi_handler,
        )
        walkers.build(trial)  # any intermediates that require information from trial
        est_opts = get_input_value(
            options,
            "estimators",
            default={},
            alias=["estimates"],
            verbose=verbosity > 1,
        )
        afqmc = AFQMC(
            comm,
            system=system,
            hamiltonian=hamiltonian,
            trial=trial,
            walkers=walkers,
            seed=qmc.rng_seed,
            nwalkers=qmc.nwalkers,
            num_steps_per_block=qmc.nsteps,
            num_blocks=qmc.nblocks,
            timestep=qmc.dt,
            stabilise_freq=qmc.nstblz,
            pop_control_freq=qmc.npop_control,
            verbose=verbosity,
            filename=est_opts.get("filename", "estimates.h5"),
        )

        afqmc.pcontrol.method = wlk_opts["population_control"]

    return afqmc


def build_afqmc_driver(
    comm: mpi4py.MPI.Intracomm,
    nelec: tuple,
    wavefunction_file: str = "wavefunction.h5",
    hamiltonian_file: str = "hamiltonian.h5",
    num_walkers_per_task: int = 10,
    estimator_filename: str = "estimates.0.h5",
    seed: int = None,
    verbosity: int = 0,
):
    if comm.rank != 0:
        verbosity = 0
    options = {
        "system": {
            "nup": nelec[0],
            "ndown": nelec[1],
        },
        "qmc": {"nwalkers": num_walkers_per_task, "rng_seed": seed},
        "hamiltonian": {"integrals": hamiltonian_file},
        "trial": {"filename": wavefunction_file},
        "estimators": {"overwrite": True, "filename": estimator_filename},
    }
    return get_driver(options, comm)


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
            print(f"# Initialising pie simulation from {input_file}")
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

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
import json

from ipie.config import MPI
from ipie.utils.mpi import MPIHandler
from ipie.utils.io import get_input_value

from ipie.systems.utils import get_system
from ipie.hamiltonians.utils import get_hamiltonian
from ipie.qmc.calc import init_communicator, setup_calculation, read_input

from ipie.addons.thermal.trial.utils import get_trial_density_matrix
from ipie.addons.thermal.walkers.uhf_walkers import UHFThermalWalkers
from ipie.addons.thermal.propagation.propagator import Propagator
from ipie.addons.thermal.qmc.options import ThermalQMCOpts, ThermalQMCParams
from ipie.addons.thermal.qmc.thermal_afqmc import ThermalAFQMC


def get_driver(options: dict, comm: MPI.COMM_WORLD) -> ThermalAFQMC:
    verbosity = options.get("verbosity", 1)
    qmc_opts = get_input_value(options, "qmc", default={}, alias=["qmc_options"])

    sys_opts = get_input_value(
        options, "system", default={}, alias=["model"], verbose=verbosity > 1
    )
    ham_opts = get_input_value(options, "hamiltonian", default={}, verbose=verbosity > 1)
    # backward compatibility with previous code (to be removed)
    for item in sys_opts.items():
        if item[0].lower() == "name" and "name" in ham_opts.keys():
            continue
        ham_opts[item[0]] = item[1]

    tdm_opts = get_input_value(
        options, "trial", default={}, alias=["trial_density_matrix"], verbose=verbosity > 1
    )

    wlk_opts = get_input_value(
        options, "walkers", default={}, alias=["walker", "walker_opts"], verbose=verbosity > 1
    )

    if comm.rank != 0:
        verbosity = 0
    lowrank = get_input_value(wlk_opts, "lowrank", default=False, alias=["low_rank"], verbose=verbosity)
    batched = get_input_value(qmc_opts, "batched", default=False, verbose=verbosity)
    debug = get_input_value(qmc_opts, "debug", default=False, verbose=verbosity)

    if (lowrank == True) or (batched == True):
        raise ValueError("Option not supported in thermal code.")
    else:
        qmc = ThermalQMCOpts(qmc_opts, verbose=0)
        mpi_handler = MPIHandler(nmembers=qmc_opts.get("nmembers", 1), verbose=verbosity)
        system = get_system(
            sys_opts, verbose=verbosity, comm=comm
        )  # Have to deal with shared comm in the future. I think we will remove this...
        ham_file = get_input_value(ham_opts, "integrals", None, verbose=verbosity)
        if ham_file is None:
            raise ValueError("Hamiltonian filename not specified.")
        pack_chol = get_input_value(
            ham_opts, "symmetry", True, alias=["pack_chol", "pack_cholesky"], verbose=verbosity
        )
        hamiltonian = get_hamiltonian(
            ham_file, mpi_handler.scomm, pack_chol=pack_chol, verbose=verbosity
        )
        num_elec = (system.nup, system.ndown)
        trial = get_trial_density_matrix(
            hamiltonian,
            num_elec,
            qmc.beta,
            qmc.dt,
            options=tdm_opts,
            comm=comm,
            verbose=verbosity,
        )
        stack_size = get_input_value(wlk_opts, 'stack_size', default=10, verbose=verbosity)
        lowrank_thresh = get_input_value(wlk_opts, 'lowrank_thresh', default=1e-6, alias=["low_rank_thresh"], verbose=verbosity)
        walkers = UHFThermalWalkers(
                    trial, hamiltonian.nbasis, qmc.nwalkers, stack_size=stack_size, 
                    lowrank=lowrank, lowrank_thresh=lowrank_thresh, verbose=verbosity)

        if (comm.rank == 0) and (qmc.nsteps > 1):
            print("Only num_steps_per_block = 1 allowed in thermal code! Resetting to value of 1.")

        # pylint: disable = no-value-for-parameter
        params = ThermalQMCParams(
            mu=qmc.mu,
            beta=qmc.beta,
            num_walkers=qmc.nwalkers,
            total_num_walkers=qmc.nwalkers * comm.size,
            num_blocks=qmc.nblocks,
            timestep=qmc.dt,
            num_stblz=qmc.nstblz,
            pop_control_freq=qmc.npop_control,
            pop_control_method=qmc.pop_control_method,
            rng_seed=qmc.rng_seed,
        )
        propagator = Propagator[type(hamiltonian)](params.timestep, params.mu)
        propagator.build(hamiltonian, trial, walkers, mpi_handler)
        afqmc = ThermalAFQMC(
            system,
            hamiltonian,
            trial,
            walkers,
            propagator,
            mpi_handler,
            params,
            debug=debug,
            verbose=verbosity,
        )

    return afqmc


def build_thermal_afqmc_driver(
    comm,
    nelec: tuple,
    hamiltonian_file: str = "hamiltonian.h5",
    seed: int = None,
    options: dict = None,
    verbosity: int = 0,
):
    if comm.rank != 0:
        verbosity = 0
    
    sys_opts = {"nup": nelec[0], "ndown": nelec[1]}
    ham_opts = {"integrals": hamiltonian_file}
    qmc_opts = {"rng_seed": seed}

    options["system"] = sys_opts
    options["hamiltonian"] = ham_opts
    options["qmc"].update(qmc_opts)
    options["verbosity"] = verbosity

    return get_driver(options, comm)

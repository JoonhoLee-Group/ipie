
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
#          Joonho Lee <linusjoonho@gmail.com>
#

"""Driver to perform AFQMC calculation"""
import copy
import json
import sys
import time
import uuid
import warnings
from math import exp

import h5py
import numpy

from ipie.config import config

from ipie.estimators.handler import EstimatorHandler
from ipie.estimators.local_energy_batch import local_energy_batch
from ipie.hamiltonians.utils import get_hamiltonian
from ipie.propagation.utils import get_propagator_driver
from ipie.qmc.options import QMCOpts
from ipie.qmc.utils import set_rng_seed
from ipie.systems.utils import get_system
from ipie.trial_wavefunction.utils import get_trial_wavefunction
from ipie.utils.io import get_input_value, serialise, to_json
from ipie.utils.misc import (get_git_info, print_env_info,
                             is_cupy)
from ipie.utils.mpi import MPIHandler
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import get_host_memory, synchronize
from ipie.walkers.walker_batch_handler import WalkerBatchHandler


class AFQMCBatch(object):
    """AFQMCBatch driver.

    Zero temperature AFQMCBatch using open ended random walk.

    This object contains all the instances of the classes which parse input
    options.

    Parameters
    ----------
    model : dict
        Input parameters for model system.
    qmc_opts : dict
        Input options relating to qmc parameters.
    estimates : dict
        Input options relating to what estimator to calculate.
    trial : dict
        Input options relating to trial wavefunction.
    propagator : dict
        Input options relating to propagator.
    parallel : bool
        If true we are running in parallel.
    verbose : bool
        If true we print out additional setup information.

    Attributes
    ----------
    uuid : string
        Simulation state uuid.
    sha1 : string
        Git hash.
    seed : int
        RNG seed. This is set during initialisation in calc.
    root : bool
        If true we are on the root / master processor.
    nprocs : int
        Number of processors.
    rank : int
        Processor id.
    cplx : bool
        If true then most numpy arrays are complex valued.
    system : system object.
        Container for model input options.
    qmc : :class:`pie.state.QMCOpts` object.
        Container for qmc input options.
    trial : :class:`pie.trial_wavefunction.X' object
        Trial wavefunction class.
    propagators : :class:`pie.propagation.Projectors` object
        Container for system specific propagation routines.
    estimators : :class:`pie.estimators.Estimators` object
        Estimator handler.
    psi : :class:`pie.walkers.Walkers` object
        Walker handler. Stores the AFQMC wavefunction.
    """

    def __init__(
        self,
        comm,
        options=None,
        system=None,
        hamiltonian=None,
        trial=None,
        parallel=False,
        verbose=False,
    ):
        if verbose is not None:
            self.verbosity = verbose
            if comm.rank != 0:
                self.verbosity = 0
            verbose = verbose > 0 and comm.rank == 0
        # 1. Environment attributes
        if comm.rank == 0:
            self.uuid = str(uuid.uuid1())
            get_sha1 = options.get("get_sha1", True)
            if get_sha1:
                try:
                    self.sha1, self.branch, self.local_mods = get_git_info()
                except:
                    self.sha1 = "None"
                    self.branch = "None"
                    self.local_mods = []
            else:
                self.sha1 = "None"
                self.branch = "None"
                self.local_mods = []
            if verbose:
                self.sys_info = print_env_info(
                    self.sha1, self.branch, self.local_mods, self.uuid, comm.size
                )
        # Hack - this is modified later if running in parallel on
        # initialisation.
        self.root = comm.rank == 0
        self.rank = comm.rank
        self._init_time = time.time()
        self.run_time = time.asctime()

        qmc_opt = get_input_value(
            options,
            "qmc",
            default={},
            alias=["qmc_options"],
            verbose=self.verbosity > 1,
        )

        self.mpi_handler = MPIHandler(comm, qmc_opt, verbose=verbose)
        self.shared_comm = self.mpi_handler.shared_comm
        # 2. Calculation objects.
        if system is not None:
            self.system = system
        else:
            sys_opts = get_input_value(
                options,
                "system",
                default={},
                alias=["model"],
                verbose=self.verbosity > 1,
            )
            self.system = get_system(sys_opts, verbose=verbose, comm=self.shared_comm)
        if hamiltonian is not None:
            self.hamiltonian = hamiltonian
        else:
            ham_opts = get_input_value(
                options, "hamiltonian", default={}, verbose=self.verbosity > 1
            )
            # backward compatibility with previous code (to be removed)
            for item in sys_opts.items():
                if item[0].lower() == "name" and "name" in ham_opts.keys():
                    continue
                ham_opts[item[0]] = item[1]
            self.hamiltonian = get_hamiltonian(
                self.system, ham_opts, verbose=verbose, comm=self.shared_comm
            )

        self.qmc = QMCOpts(qmc_opt, verbose=verbose)
        if config.get_option('use_gpu'):
            ngpus = xp.cuda.runtime.getDeviceCount()
            props = xp.cuda.runtime.getDeviceProperties(0)
            xp.cuda.runtime.setDevice(self.shared_comm.rank)
            if comm.rank == 0:
                if ngpus > comm.size:
                    print(
                        "# There are unused GPUs ({} MPI tasks but {} GPUs). "
                        " Check if this is really what you wanted.".format(
                            comm.size, ngpus
                        )
                    )


        if self.qmc.nwalkers is None:
            assert self.qmc.nwalkers_per_task is not None
            self.qmc.nwalkers = self.qmc.nwalkers_per_task * comm.size
        if self.qmc.nwalkers_per_task is None:
            assert self.qmc.nwalkers is not None
            self.qmc.nwalkers_per_task = int(self.qmc.nwalkers / comm.size)
        # Reset number of walkers so they are evenly distributed across
        # cores/ranks.
        # Number of walkers per core/rank.
        self.qmc.nwalkers = int(
            self.qmc.nwalkers / comm.size
        )  # This should be gone in the future
        assert self.qmc.nwalkers == self.qmc.nwalkers_per_task
        # Total number of walkers.
        if self.qmc.nwalkers == 0:
            if comm.rank == 0:
                print("# WARNING: Not enough walkers for selected core count.")
                print(
                    "# There must be at least one walker per core set in the "
                    "input file."
                )
                print("# Setting one walker per core.")
            self.qmc.nwalkers = 1
        self.qmc.ntot_walkers = self.qmc.nwalkers * comm.size

        self.qmc.rng_seed = set_rng_seed(self.qmc.rng_seed, comm)

        self.cplx = self.determine_dtype(options.get("propagator", {}), self.system)

        twf_opt = get_input_value(
            options,
            "trial",
            default={},
            alias=["trial_wavefunction"],
            verbose=self.verbosity > 1,
        )
        if trial is not None:
            self.trial = trial
        else:
            self.trial = get_trial_wavefunction(
                self.system,
                self.hamiltonian,
                options=twf_opt,
                comm=comm,
                scomm=self.shared_comm,
                verbose=verbose,
            )
        mem = get_host_memory()
        if comm.rank == 0:
            if self.trial.compute_trial_energy:
                self.trial.calculate_energy(self.system, self.hamiltonian)
                print("# Trial wfn energy is {}".format(self.trial.energy))
            else:
                print("# WARNING: skipping trial energy calculation is requested.")

        if self.trial.compute_trial_energy:
            self.trial.e1b = comm.bcast(self.trial.e1b, root=0)
            self.trial.e2b = comm.bcast(self.trial.e2b, root=0)

        comm.barrier()
        prop_opt = options.get("propagator", {})
        if comm.rank == 0:
            print("# Getting propagator driver")
        self.propagators = get_propagator_driver(
            self.system,
            self.hamiltonian,
            self.trial,
            self.qmc,
            options=prop_opt,
            verbose=verbose,
        )
        self.tsetup = time.time() - self._init_time
        wlk_opts = get_input_value(
            options,
            "walkers",
            default={},
            alias=["walker", "walker_opts"],
            verbose=self.verbosity > 1,
        )
        if comm.rank == 0:
            print("# Getting WalkerBatchHandler")
        self.psi = WalkerBatchHandler(
            self.system,
            self.hamiltonian,
            self.trial,
            self.qmc,
            walker_opts=wlk_opts,
            mpi_handler=self.mpi_handler,
            verbose=verbose,
        )
        est_opts = get_input_value(
            options,
            "estimators",
            default={},
            alias=["estimates", "estimator"],
            verbose=self.verbosity > 1,
        )
        est_opts["stack_size"] = wlk_opts.get("stack_size", 1)
        self.estimators = EstimatorHandler(
                comm,
                self.system,
                self.hamiltonian,
                self.trial,
                walker_state=self.psi.accumulator_factors,
                options=est_opts,
                verbose=(comm.rank == 0 and verbose))

        if self.mpi_handler.nmembers > 1:
            if comm.rank == 0:
                print("# Chunking hamiltonian.")
            self.hamiltonian.chunk(self.mpi_handler)
            if comm.rank == 0:
                print("# Chunking trial.")
            self.trial.chunk(self.mpi_handler)

        if config.get_option('use_gpu'):
            self.propagators.cast_to_cupy(verbose and comm.rank == 0)
            self.hamiltonian.cast_to_cupy(verbose and comm.rank == 0)
            self.trial.cast_to_cupy(verbose and comm.rank == 0)
            self.psi.walkers_batch.cast_to_cupy(verbose and comm.rank == 0)
        else:
            if comm.rank == 0:
                try:
                    import cupy
                    _have_cupy = True
                except:
                    _have_cupy = False
                print("# NOTE: cupy available but qmc.gpu == False.")
                print(
                    "#       If this is unintended set gpu option in qmc" "  section."
                )

        if comm.rank == 0:
            mem_avail = get_host_memory()
            print("# Available memory on the node is {:4.3f} GB".format(mem_avail))
            json.encoder.FLOAT_REPR = lambda o: format(o, ".6f")
            json_string = to_json(self)
            self.estimators.json_string = json_string

    def run(self, psi=None, comm=None, verbose=True):
        """Perform AFQMC simulation on state object using open-ended random walk.

        Parameters
        ----------
        psi : :class:`pie.walker.Walkers` object
            Initial wavefunction / distribution of walkers.
        comm : MPI communicator
        """
        tzero_setup = time.time()
        if psi is not None:
            self.psi = psi
        self.setup_timers()
        eshift = 0.0

        self.psi.orthogonalise(self.trial, self.propagators.free_projection)

        total_steps = self.qmc.nsteps * self.qmc.nblocks
        # Delay initialization incase user defined estimators added after
        # construction.
        self.estimators.initialize(comm)
        # Calculate estimates for initial distribution of walkers.
        self.estimators.compute_estimators(
            comm,
            self.system,
            self.hamiltonian,
            self.trial,
            self.psi.walkers_batch,
        )
        self.psi.update_accumulators()
        self.estimators.print_block(comm, 0, self.psi.accumulator_factors)
        self.psi.zero_accumulators()

        synchronize()
        self.tsetup += time.time() - tzero_setup

        for step in range(1, total_steps + 1):
            synchronize()
            start_step = time.time()
            if step % self.qmc.nstblz == 0:
                start = time.time()
                self.psi.orthogonalise(self.trial, self.propagators.free_projection)
                synchronize()
                self.tortho += time.time() - start
            start = time.time()

            self.propagators.propagate_walker_batch(
                self.psi.walkers_batch,
                self.system,
                self.hamiltonian,
                self.trial,
                eshift,
            )

            self.tprop_fbias = self.propagators.tfbias
            self.tprop_ovlp = self.propagators.tovlp
            self.tprop_update = self.propagators.tupdate
            self.tprop_gf = self.propagators.tgf
            self.tprop_vhs = self.propagators.tvhs
            self.tprop_gemm = self.propagators.tgemm

            start_clip = time.time()
            if step > 1:
                # wbound = min(100.0, self.psi.walkers_batch.total_weight * 0.10) # bounds are supposed to be the smaller of 100 and 0.1 * tot weight but not clear how useful this is
                wbound = self.psi.walkers_batch.total_weight * 0.10
                xp.clip(
                    self.psi.walkers_batch.weight,
                    a_min=-wbound,
                    a_max=wbound,
                    out=self.psi.walkers_batch.weight,
                )  # in-place clipping
    
            synchronize()
            self.tprop_clip += time.time() - start_clip

            start_barrier = time.time()
            if step % self.qmc.npop_control == 0:
                comm.Barrier()
            self.tprop_barrier += time.time() - start_barrier

            self.tprop += time.time() - start
            if step % self.qmc.npop_control == 0:
                start = time.time()
                self.psi.pop_control(comm)
                synchronize()
                self.tpopc += time.time() - start
                self.tpopc_send = self.psi.send_time
                self.tpopc_recv = self.psi.recv_time
                self.tpopc_comm = self.psi.communication_time
                self.tpopc_non_comm = self.psi.non_communication_time

            # accumulate weight, hybrid energy etc. across block
            start = time.time()
            self.psi.update_accumulators()
            synchronize()
            self.testim += time.time() - start # we dump this time into estimator
            # calculate estimators
            start = time.time()
            if step % self.qmc.nsteps == 0:
                self.estimators.compute_estimators(
                    comm,
                    self.system,
                    self.hamiltonian,
                    self.trial,
                    self.psi.walkers_batch,
                )
                self.estimators.print_block(
                        comm, step//self.qmc.nsteps,
                        self.psi.accumulator_factors
                        )
                self.psi.zero_accumulators()
            synchronize()
            self.testim += time.time() - start
            if self.psi.write_restart and step % self.psi.write_freq == 0:
                self.psi.write_walkers_batch(comm)
            if step < self.qmc.neqlb:
                eshift = self.psi.accumulator_factors.eshift
            else:
                eshift += self.psi.accumulator_factors.eshift - eshift
            synchronize()
            self.tstep += time.time() - start_step

    def finalise(self, verbose=False):
        """Tidy up.

        Parameters
        ----------
        verbose : bool
            If true print out some information to stdout.
        """
        nsteps = max(self.qmc.nsteps, 1)
        nblocks = max(self.qmc.nblocks, 1)
        nstblz = max(nsteps // self.qmc.nstblz, 1)
        npcon = max(nsteps // self.qmc.npop_control, 1)
        if self.root:
            if verbose:
                print("# End Time: {:s}".format(time.asctime()))
                print(
                    "# Running time : {:.6f} seconds".format(
                        (time.time() - self._init_time)
                    )
                )
                print(
                    "# Timing breakdown (per call, total calls per block, total blocks):"
                )
                print("# - Setup: {:.6f} s".format(self.tsetup))
                print(
                    "# - Block: {:.6f} s / block for {} total blocks".format(
                        self.tstep / (nblocks), nblocks
                    )
                )
                print(
                    "# - Propagation: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tprop / (nblocks * nsteps), nsteps, nblocks
                    )
                )
                print(
                    "#     -       Force bias: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tprop_fbias / (nblocks * nsteps), nsteps, nblocks
                    )
                )
                print(
                    "#     -              VHS: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tprop_vhs / (nblocks * nsteps), nsteps, nblocks
                    )
                )
                print(
                    "#     - Green's Function: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tprop_gf / (nblocks * nsteps), nsteps, nblocks
                    )
                )
                print(
                    "#     -          Overlap: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tprop_ovlp / (nblocks * nsteps), nsteps, nblocks
                    )
                )
                print(
                    "#     -   Weights Update: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        (self.tprop_update + self.tprop_clip) / (nblocks * nsteps),
                        nsteps,
                        nblocks,
                    )
                )
                print(
                    "#     -  GEMM operations: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tprop_gemm / (nblocks * nsteps), nsteps, nblocks
                    )
                )
                print(
                    "#     -          Barrier: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tprop_barrier / (nblocks * nsteps), nsteps, nblocks
                    )
                )
                print(
                    "# - Estimators: {:.6f} s / call for {} call(s)".format(
                        self.testim / nblocks, nblocks
                    )
                )
                print(
                    "# - Orthogonalisation: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tortho / (nstblz * nblocks), nstblz, nblocks
                    )
                )
                print(
                    "# - Population control: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tpopc / (npcon * nblocks), npcon, nblocks
                    )
                )
                print(
                    "#       -     Commnication: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tpopc_comm / (npcon * nblocks), npcon, nblocks
                    )
                )
                print(
                    "#       - Non-Commnication: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tpopc_non_comm / (npcon * nblocks), npcon, nblocks
                    )
                )

    def determine_dtype(self, propagator, system):
        """Determine dtype for trial wavefunction and walkers.

        Parameters
        ----------
        propagator : dict
            Propagator input options.
        system : object
            system object.
        """
        hs_type = propagator.get("hubbard_stratonovich", "discrete")
        continuous = "continuous" in hs_type
        twist = system.ktwist.all() is not None
        return continuous or twist

    def get_energy(self, skip=0):
        """Get mixed estimate for the energy.

        Returns
        -------
        (energy, error) : tuple
            Mixed estimate for the energy and standard error.
        """
        filename = self.estimators.h5f_name
        from ipie.analysis import blocking

        try:
            eloc = blocking.reblock_local_energy(filename, skip)
        except IndexError:
            eloc = None
        except ValueError:
            eloc = None
        return eloc

    def setup_timers(self):
        self.tortho = 0
        self.tprop = 0

        self.tprop_fbias = 0.0
        self.tprop_ovlp = 0.0
        self.tprop_update = 0.0
        self.tprop_gf = 0.0
        self.tprop_vhs = 0.0
        self.tprop_gemm = 0.0
        self.tprop_clip = 0.0
        self.tprop_barrier = 0.0

        self.testim = 0
        self.tpopc = 0
        self.tpopc_comm = 0
        self.tpopc_non_comm = 0
        self.tstep = 0

    def get_one_rdm(self, skip=0):
        """Get back-propagated estimate for the one RDM.

        Returns
        -------
        rdm : :class:`numpy.ndarray`
            Back propagated estimate for 1RMD.
        error : :class:`numpy.ndarray`
            Standard error in the RDM.
        """
        from ipie.analysis import blocking

        filename = self.estimators.h5f_name
        try:
            bp_rdm, bp_rdm_err = blocking.reblock_rdm(filename)
        except IndexError:
            bp_rdm, bp_rdm_err = None, None
        return (bp_rdm, bp_rdm_err)

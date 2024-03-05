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
import json
import time
from typing import Dict, Optional, Tuple

from ipie.addons.free_projection.estimators.handler import EstimatorHandlerFP
from ipie.addons.free_projection.propagation.free_propagation import FreePropagation
from ipie.addons.free_projection.qmc.options import QMCParamsFP
from ipie.addons.free_projection.walkers.uhf_walkers import UHFWalkersFP
from ipie.estimators.estimator_base import EstimatorBase
from ipie.hamiltonians.utils import get_hamiltonian
from ipie.qmc.afqmc import AFQMC
from ipie.trial_wavefunction.utils import get_trial_wavefunction
from ipie.utils.backend import synchronize
from ipie.utils.io import to_json
from ipie.utils.mpi import MPIHandler
from ipie.walkers.base_walkers import WalkerAccumulator
from ipie.walkers.walkers_dispatch import get_initial_walker


class FPAFQMC(AFQMC):
    """Free projection AFQMC driver."""

    def __init__(
        self,
        system,
        hamiltonian,
        trial,
        walkers,
        propagator,
        params: QMCParamsFP,
        verbose: int = 0,
    ):
        super().__init__(system, hamiltonian, trial, walkers, propagator, params, verbose=verbose)

    @staticmethod
    # TODO: wavefunction type, trial type, hamiltonian type
    def build(
        num_elec: Tuple[int, int],
        hamiltonian,
        trial_wavefunction,
        walkers=None,
        num_walkers: int = 100,
        seed: int = None,
        num_steps_per_block: int = 25,
        num_blocks: int = 100,
        timestep: float = 0.005,
        stabilize_freq=5,
        pop_control_freq=-1,
        verbose=True,
        mpi_handler=None,
        ene_0=0.0,
        num_iterations_fp=1,
    ) -> "FPAFQMC":
        """Factory method to build AFQMC driver from hamiltonian and trial wavefunction.

        Parameters
        ----------
        num_elec: tuple(int, int)
            Number of alpha and beta electrons.
        hamiltonian :
            Hamiltonian describing the system.
        trial_wavefunction :
            Trial wavefunction
        num_walkers : int
            Number of walkers per MPI process used in the simulation. The TOTAL
                number of walkers is num_walkers * number of processes.
        num_steps_per_block : int
            Number of Monte Carlo steps before estimators are evaluatied.
                Default 25.
        num_blocks : int
            Number of blocks to perform. Total number of steps = num_blocks *
                num_steps_per_block.
        timestep : float
            Imaginary timestep. Default 0.005.
        stabilize_freq : float
            Frequency at which to perform QR factorization of walkers (in units
                of steps.) Default 25.
        pop_control_freq : int
            Not performed in free projection.
        verbose : bool
            Log verbosity. Default True i.e. print information to stdout.
        ene_0 : float
            Energy guess for the desired state.
        num_iterations_fp : int
            Number of iterations of free projection.
        """

        driver = AFQMC.build(
            num_elec,
            hamiltonian,
            trial_wavefunction,
            walkers,
            num_walkers,
            seed,
            num_steps_per_block,
            num_blocks,
            timestep,
            stabilize_freq,
            pop_control_freq,
            verbose,
            mpi_handler,
        )
        if mpi_handler is None:
            mpi_handler = MPIHandler()
            comm = mpi_handler.comm
        else:
            comm = mpi_handler.comm
        fp_prop = FreePropagation(timestep, verbose=verbose, exp_nmax=10, ene_0=ene_0)
        fp_prop.build(hamiltonian, driver.trial, walkers, mpi_handler)
        if walkers is None:
            _, initial_walker = get_initial_walker(driver.trial)
            # TODO this is a factory method not a class
            walkers = UHFWalkersFP(
                initial_walker,
                driver.system.nup,
                driver.system.ndown,
                hamiltonian.nbasis,
                num_walkers,
                mpi_handler,
            )
            walkers.build(driver.trial)  # any intermediates that require information from trial
        params = QMCParamsFP(
            num_walkers=num_walkers,
            total_num_walkers=num_walkers * comm.size,
            num_blocks=num_blocks,
            num_steps_per_block=num_steps_per_block,
            timestep=timestep,
            num_stblz=stabilize_freq,
            pop_control_freq=pop_control_freq,
            rng_seed=seed,
            num_iterations_fp=num_iterations_fp,
        )
        return FPAFQMC(
            driver.system,
            driver.hamiltonian,
            driver.trial,
            driver.walkers,
            fp_prop,
            params,
            verbose=(verbose and comm.rank == 0),
        )

    @staticmethod
    # TODO: wavefunction type, trial type, hamiltonian type
    def build_from_hdf5(
        num_elec: Tuple[int, int],
        ham_file,
        wfn_file,
        num_walkers: int = 100,
        seed: int = None,
        num_steps_per_block: int = 25,
        num_blocks: int = 100,
        timestep: float = 0.005,
        stabilize_freq=5,
        pop_control_freq=5,
        num_dets_chunk=1,
        num_dets_for_trial_props=100,
        pack_cholesky=True,
        verbose=True,
    ) -> "FPAFQMC":
        """Factory method to build FPAFQMC driver from hamiltonian and trial wavefunction.

        Parameters
        ----------
        num_elec: tuple(int, int)
            Number of alpha and beta electrons.
        ham_file : str
            Path to Hamiltonian describing the system.
        wfn_file : str
            Path to Trial wavefunction
        num_walkers : int
            Number of walkers per MPI process used in the simulation. The TOTAL
                number of walkers is num_walkers * number of processes.
        num_steps_per_block : int
            Number of Monte Carlo steps before estimators are evaluatied.
                Default 25.
        num_blocks : int
            Number of blocks to perform. Total number of steps = num_blocks *
                num_steps_per_block.
        timestep : float
            Imaginary timestep. Default 0.005.
        stabilize_freq : float
            Frequency at which to perform QR factorization of walkers (in units
                of steps.) Default 25.
        pop_control_freq : int
            Frequency at which to perform population control (in units of
                steps.) Default 25.
        num_det_chunks : int
            Size of chunks of determinants to process during batching. Default=1 (no batching).
        num_dets_for_trial_props: int
            Number of determinants to use to evaluate trial wavefunction properties.
        pack_cholesky : bool
            Use symmetry to reduce memory consumption of integrals. Default True.
        verbose : bool
            Log verbosity. Default True i.e. print information to stdout.
        """
        mpi_handler = MPIHandler()
        _verbose = verbose and mpi_handler.comm.rank == 0
        ham = get_hamiltonian(
            ham_file, mpi_handler.scomm, verbose=_verbose, pack_chol=pack_cholesky
        )
        trial = get_trial_wavefunction(
            num_elec,
            ham.nbasis,
            wfn_file,
            ndet_chunks=num_dets_chunk,
            ndets_props=num_dets_for_trial_props,
            verbose=_verbose,
        )
        trial.half_rotate(ham, mpi_handler.scomm)
        return FPAFQMC.build(
            trial.nelec,
            ham,
            trial,
            num_walkers=num_walkers,
            seed=seed,
            num_steps_per_block=num_steps_per_block,
            num_blocks=num_blocks,
            timestep=timestep,
            stabilize_freq=stabilize_freq,
            pop_control_freq=pop_control_freq,
            verbose=verbose,
            mpi_handler=mpi_handler,
        )

    def setup_estimators(
        self, filename, additional_estimators: Optional[Dict[str, EstimatorBase]] = None
    ):
        self.accumulators = WalkerAccumulator(
            ["Weight", "WeightFactor", "HybridEnergy"], self.params.num_steps_per_block
        )
        comm = self.mpi_handler.comm
        self.estimators = []
        for i in range(self.params.num_blocks):
            self.estimators.append(
                EstimatorHandlerFP(
                    self.mpi_handler.comm,
                    self.system,
                    self.hamiltonian,
                    self.trial,
                    walker_state=self.accumulators,
                    verbose=(comm.rank == 0 and self.verbose),
                    filename=f"{filename}.{i}",
                    observables=("energy",),
                )
            )
        if additional_estimators is not None:
            raise NotImplementedError(
                "Additional estimators not implemented yet for free projection."
            )
        # TODO: Move this to estimator and log uuid etc in serialization
        json.encoder.FLOAT_REPR = lambda o: format(o, ".6f")
        json_string = to_json(self)
        for e in self.estimators:
            e.json_string = json_string

        for i, e in enumerate(self.estimators):
            e.initialize(comm, i == 0)

    def run(
        self,
        psi=None,
        estimator_filename="estimate.h5",
        verbose=True,
        additional_estimators: Optional[Dict[str, EstimatorBase]] = None,
    ):
        """Perform FP AFQMC simulation on state object by Gaussian sampling of short time projection.

        Parameters
        ----------
        psi : :class:`pie.walker.Walkers` object
            Initial wavefunction / distribution of walkers. Default None.
        estimator_filename : str
            File to write estimates to.
        additional_estimators : dict
            Dictionary of additional estimators to evaluate.
        """
        self.setup_timers()
        tzero_setup = time.time()
        if psi is not None:
            self.walkers = psi
        self.setup_timers()
        eshift = 0.0
        self.walkers.orthogonalise()

        self.get_env_info()
        self.copy_to_gpu()
        self.distribute_hamiltonian()
        self.setup_estimators(estimator_filename, additional_estimators=additional_estimators)

        total_steps = self.params.num_steps_per_block * self.params.num_blocks

        synchronize()
        comm = self.mpi_handler.comm
        self.tsetup += time.time() - tzero_setup

        for iter in range(self.params.num_iterations_fp):
            block_number = 0
            _, initial_walker = get_initial_walker(self.trial)
            # TODO this is a factory method not a class
            initial_walkers = UHFWalkersFP(
                initial_walker,
                self.system.nup,
                self.system.ndown,
                self.hamiltonian.nbasis,
                self.params.num_walkers,
                self.mpi_handler,
            )
            initial_walkers.build(self.trial)
            self.walkers = initial_walkers
            for step in range(1, total_steps + 1):
                synchronize()
                start_step = time.time()
                if step % self.params.num_stblz == 0:
                    start = time.time()
                    self.walkers.orthogonalise()
                    synchronize()
                    self.tortho += time.time() - start
                start = time.time()

                self.propagator.propagate_walkers(
                    self.walkers, self.hamiltonian, self.trial, eshift
                )

                self.tprop_ovlp = self.propagator.timer.tovlp
                self.tprop_update = self.propagator.timer.tupdate
                self.tprop_gf = self.propagator.timer.tgf
                self.tprop_vhs = self.propagator.timer.tvhs
                self.tprop_gemm = self.propagator.timer.tgemm

                # accumulate weight, hybrid energy etc. across block
                start = time.time()
                # self.accumulators.update(self.walkers)
                synchronize()
                self.testim += time.time() - start  # we dump this time into estimator
                # calculate estimators
                start = time.time()
                if step % self.params.num_steps_per_block == 0:
                    self.estimators[block_number].compute_estimators(
                        comm, self.system, self.hamiltonian, self.trial, self.walkers
                    )
                    self.estimators[block_number].print_block(
                        comm,
                        iter,
                        self.accumulators,
                        time_step=block_number,
                    )
                    block_number += 1
                synchronize()
                self.testim += time.time() - start

                # restart write features disabled
                # if self.walkers.write_restart and step % self.walkers.write_freq == 0:
                #     self.walkers.write_walkers_batch(comm)
                # self.accumulators.zero()
                synchronize()
                self.tstep += time.time() - start_step

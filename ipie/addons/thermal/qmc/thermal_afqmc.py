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
"""Driver to perform Thermal AFQMC calculation"""
import json
import time
from typing import Dict, Optional, Tuple

import numpy

from ipie.addons.thermal.estimators.handler import ThermalEstimatorHandler
from ipie.addons.thermal.propagation.propagator import Propagator
from ipie.addons.thermal.qmc.options import ThermalQMCParams
from ipie.addons.thermal.walkers.pop_controller import ThermalPopController
from ipie.addons.thermal.walkers.uhf_walkers import UHFThermalWalkers
from ipie.estimators.estimator_base import EstimatorBase
from ipie.qmc.afqmc import AFQMCBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize
from ipie.utils.io import to_json
from ipie.utils.mpi import MPIHandler
from ipie.walkers.base_walkers import WalkerAccumulator


class ThermalAFQMC(AFQMCBase):
    """Thermal AFQMC driver.

    Parameters
    ----------
    hamiltonian :
        Hamiltonian describing the system.
    trial :
        Trial density matrix.
    walkers :
        Walkers used for open ended random walk.
    propagator :
        Class describing how to propagate walkers.
    params :
        Parameters of simulation. See ThermalQMCParams for description.
    verbose : bool
        How much information to print.

    Attributes
    ----------
    _parallel_rng_seed : int
        Seed deduced from params.rng_seed which is generally different on each
            MPI process.
    """

    def __init__(
        self,
        hamiltonian,
        trial,
        walkers,
        propagator,
        mpi_handler,
        params: ThermalQMCParams,
        debug: bool = False,
        verbose: int = 0,
    ):
        super().__init__(
            None, hamiltonian, trial, walkers, propagator, mpi_handler, params, verbose
        )
        self.debug = debug

        if self.debug and verbose:
            print("# Using legacy `update_weights`.")

    @staticmethod
    def build(
        nelec: Tuple[int, int],
        mu: float,
        beta: float,
        hamiltonian,
        trial,
        nwalkers: int = 100,
        stack_size: int = 10,
        seed: Optional[int] = None,
        nblocks: int = 100,
        timestep: float = 0.005,
        stabilize_freq: int = 5,
        pop_control_freq: int = 5,
        pop_control_method: str = "pair_branch",
        lowrank: bool = False,
        lowrank_thresh: float = 1e-6,
        debug: bool = False,
        verbose: int = 0,
        mpi_handler=None,
    ) -> "ThermalAFQMC":
        """Factory method to build thermal AFQMC driver from hamiltonian and trial density matrix.

        Parameters
        ----------
        num_elec : tuple(int, int)
            Number of alpha and beta electrons.
        mu : float
            Chemical potential.
        beta : float
            Inverse temperature.
        hamiltonian :
            Hamiltonian describing the system.
        trial :
            Trial density matrix.
        nwalkers : int
            Number of walkers per MPI process used in the simulation. The TOTAL
                number of walkers is nwalkers * number of processes.
        nblocks : int
            Number of blocks to perform.
        timestep : float
            Imaginary timestep. Default 0.005.
        stabilize_freq : float
            Frequency at which to perform QR factorization of walkers (in units
                of steps.) Default 25.
        pop_control_freq : int
            Frequency at which to perform population control (in units of
                steps.) Default 25.
        lowrank : bool
            Low-rank algorithm for thermal propagation. Doesn't work for now!
        lowrank_thresh : bool
            Threshold for low-rank algorithm.
        verbose : bool
            Log verbosity. Default True i.e. print information to stdout.
        """
        if mpi_handler is None:
            mpi_handler = MPIHandler()
            comm = mpi_handler.comm

        else:
            comm = mpi_handler.comm

        # pylint: disable = no-value-for-parameter
        params = ThermalQMCParams(
            mu=mu,
            beta=beta,
            num_walkers=nwalkers,
            total_num_walkers=nwalkers * comm.size,
            num_blocks=nblocks,
            timestep=timestep,
            num_stblz=stabilize_freq,
            pop_control_freq=pop_control_freq,
            pop_control_method=pop_control_method,
            rng_seed=seed,
        )

        walkers = UHFThermalWalkers(
            trial,
            hamiltonian.nbasis,
            nwalkers,
            stack_size=stack_size,
            lowrank=lowrank,
            lowrank_thresh=lowrank_thresh,
            mpi_handler=mpi_handler,
            verbose=verbose,
        )
        propagator = Propagator[type(hamiltonian)](timestep, mu, lowrank=lowrank, verbose=verbose)
        propagator.build(
            hamiltonian, trial=trial, walkers=walkers, mpi_handler=mpi_handler, verbose=verbose
        )
        return ThermalAFQMC(
            hamiltonian,
            trial,
            walkers,
            propagator,
            mpi_handler,
            params,
            debug=debug,
            verbose=verbose,
        )

    def run(
        self,
        walkers=None,
        estimator_filename=None,
        verbose: bool = True,
        additional_estimators: Optional[Dict[str, EstimatorBase]] = None,
        print_time_slice: bool = False,
    ):
        """Perform Thermal AFQMC simulation on state object using open-ended random walk.

        Parameters
        ----------
        state : :class:`pie.state.State` object
            Model and qmc parameters.

        walkers: :class:`pie.walker.Walkers` object
            Initial wavefunction / distribution of walkers.

        estimator_filename : str
            File to write estimates to.

        additional_estimators : dict
            Dictionary of additional estimators to evaluate.
        """
        # Setup.
        self.setup_timers()
        ft_setup = time.time()
        eshift = 0.0

        if walkers is not None:
            self.walkers = walkers

        self.pcontrol = ThermalPopController(
            self.params.num_walkers,
            self.params.num_steps_per_block,
            self.mpi_handler,
            self.params.pop_control_method,
            verbose=self.verbose,
        )

        self.get_env_info()
        self.setup_estimators(estimator_filename, additional_estimators=additional_estimators)

        synchronize()
        comm = self.mpi_handler.comm
        self.tsetup += time.time() - ft_setup

        # Propagate.
        total_steps = self.params.num_steps_per_block * self.params.num_blocks
        # TODO: This magic value of 2 is pretty much never controlled on input.
        # Moreover I'm not convinced having a two stage shift update actually
        # matters at all.
        neqlb_steps = 2.0 / self.params.timestep
        nslices = numpy.rint(self.params.beta / self.params.timestep).astype(int)

        for step in range(1, total_steps + 1):
            synchronize()
            start_step = time.time()

            for t in range(nslices):
                if self.verbosity >= 2 and comm.rank == 0:
                    print(" # Timeslice %d of %d." % (t, nslices))

                start = time.time()
                self.propagator.propagate_walkers(
                    self.walkers, self.hamiltonian, self.trial, eshift, debug=self.debug
                )

                self.tprop_fbias = self.propagator.timer.tfbias
                self.tprop_update = self.propagator.timer.tupdate
                self.tprop_vhs = self.propagator.timer.tvhs
                self.tprop_gemm = self.propagator.timer.tgemm

                start_clip = time.time()
                if t > 0:
                    wbound = self.pcontrol.total_weight * 0.10
                    xp.clip(
                        self.walkers.weight, a_min=-wbound, a_max=wbound, out=self.walkers.weight
                    )  # In-place clipping.

                synchronize()
                self.tprop_clip += time.time() - start_clip

                start_barrier = time.time()
                if t % self.params.pop_control_freq == 0:
                    comm.Barrier()

                self.tprop_barrier += time.time() - start_barrier
                self.tprop += time.time() - start

                if (t > 0) and (t % self.params.pop_control_freq == 0):
                    start = time.time()
                    self.pcontrol.pop_control(self.walkers, comm)
                    synchronize()
                    self.tpopc += time.time() - start
                    self.tpopc_send = self.pcontrol.timer.send_time
                    self.tpopc_recv = self.pcontrol.timer.recv_time
                    self.tpopc_comm = self.pcontrol.timer.communication_time
                    self.tpopc_non_comm = self.pcontrol.timer.non_communication_time

                # Print estimators at each time slice.
                if print_time_slice:
                    self.estimators.compute_estimators(
                        hamiltonian=self.hamiltonian, trial=self.trial, walker_batch=self.walkers
                    )
                    self.estimators.print_time_slice(comm, t, self.accumulators)

            # Accumulate weight, hybrid energy etc. across block.
            start = time.time()
            self.accumulators.update(self.walkers)
            self.testim += time.time() - start

            # Calculate estimators.
            start = time.time()
            if step % self.params.num_steps_per_block == 0:
                self.estimators.compute_estimators(
                    hamiltonian=self.hamiltonian, trial=self.trial, walker_batch=self.walkers
                )

                self.estimators.print_block(
                    comm, step // self.params.num_steps_per_block, self.accumulators
                )
                self.accumulators.zero()

            synchronize()
            self.testim += time.time() - start

            if step < neqlb_steps:
                eshift = self.accumulators.eshift

            else:
                eshift += self.accumulators.eshift - eshift

            self.walkers.reset(self.trial)  # Reset stack, weights, phase.

            synchronize()
            self.tstep += time.time() - start_step

    def setup_estimators(
        self, filename, additional_estimators: Optional[Dict[str, EstimatorBase]] = None
    ):
        self.accumulators = WalkerAccumulator(
            ["Weight", "WeightFactor", "HybridEnergy"], self.params.num_steps_per_block
        )
        comm = self.mpi_handler.comm
        self.estimators = ThermalEstimatorHandler(
            self.mpi_handler.comm,
            self.hamiltonian,
            self.trial,
            walker_state=self.accumulators,
            verbose=(comm.rank == 0 and self.verbose),
            filename=filename,
        )

        if additional_estimators is not None:
            for k, v in additional_estimators.items():
                self.estimators[k] = v

        # TODO: Move this to estimator and log uuid etc in serialization
        json.encoder.FLOAT_REPR = lambda o: format(o, ".6f")
        json_string = to_json(self)
        self.estimators.json_string = json_string
        self.estimators.initialize(comm)

        # Calculate estimates for initial distribution of walkers.
        self.estimators.compute_estimators(
            hamiltonian=self.hamiltonian, trial=self.trial, walker_batch=self.walkers
        )
        self.accumulators.update(self.walkers)
        self.estimators.print_block(comm, 0, self.accumulators)
        self.accumulators.zero()

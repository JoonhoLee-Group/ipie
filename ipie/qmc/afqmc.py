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
import uuid
from typing import Union

from ipie.config import config
from ipie.estimators.handler import EstimatorHandler
from ipie.propagation.propagator import Propagator
from ipie.qmc.options import QMCOpts
from ipie.qmc.utils import set_rng_seed
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import get_host_memory, synchronize
from ipie.utils.io import to_json
from ipie.utils.misc import get_git_info, print_env_info
from ipie.utils.mpi import MPIHandler
from ipie.walkers.base_walkers import WalkerAccumulator
from ipie.walkers.pop_controller import PopController


class AFQMC(object):
    """AFQMC driver.

    Zero temperature AFQMC using open ended random walk.

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
        system=None,
        hamiltonian=None,
        trial=None,
        walkers=None,
        propagator=None,
        verbose: int = 0,
        seed: int = None,
        nwalkers: int = 100,
        num_steps_per_block: int = 25,
        num_blocks: int = 100,
        timestep: float = 0.005,
        stabilise_freq=5,
        pop_control_freq=5,
        filename: Union[str, None] = None,
    ):
        if verbose is not None:
            self.verbosity = verbose
            if comm.rank != 0:
                self.verbosity = 0
            verbose = verbose > 0 and comm.rank == 0
        # 1. Environment attributes
        if comm.rank == 0:
            self.uuid = str(uuid.uuid1())
            try:
                self.sha1, self.branch, self.local_mods = get_git_info()
            except:
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

        qmc_opt = {
            "nwalkers": nwalkers,
            "timestep": timestep,
            "nsteps": num_steps_per_block,
            "num_blocks": num_blocks,
            "pop_control_freq": pop_control_freq,
            "stabilise_freq": stabilise_freq,
        }

        self.mpi_handler = MPIHandler(comm, nmembers=1, verbose=verbose)
        self.shared_comm = self.mpi_handler.shared_comm
        # 2. Calculation objects.
        self.system = system
        self.hamiltonian = hamiltonian
        self.trial = trial

        self.qmc = QMCOpts(qmc_opt, verbose=verbose)
        if config.get_option("use_gpu"):
            ngpus = xp.cuda.runtime.getDeviceCount()
            _ = xp.cuda.runtime.getDeviceProperties(0)
            xp.cuda.runtime.setDevice(self.shared_comm.rank)
            if comm.rank == 0:
                if ngpus > comm.size:
                    print(
                        "# There are unused GPUs ({} MPI tasks but {} GPUs). "
                        " Check if this is really what you wanted.".format(comm.size, ngpus)
                    )

        # Total number of walkers.
        if self.qmc.nwalkers == 0:
            if comm.rank == 0:
                print("# WARNING: Not enough walkers for selected core count.")
                print("# There must be at least one walker per core set in the " "input file.")
                print("# Setting one walker per core.")
            self.qmc.nwalkers = 1
        self.qmc.ntot_walkers = self.qmc.nwalkers * comm.size

        if seed is None:
            self.qmc.rng_seed = set_rng_seed(self.qmc.rng_seed, comm)
        else:
            self.qmc.rng_seed = set_rng_seed(seed, comm)

        if comm.rank == 0:
            if self.trial.compute_trial_energy:
                self.trial.calculate_energy(self.system, self.hamiltonian)
                print(f"# Trial wfn energy is {self.trial.energy}")
            else:
                print("# WARNING: skipping trial energy calculation is requested.")

        if self.trial.compute_trial_energy:
            self.trial.e1b = comm.bcast(self.trial.e1b, root=0)
            self.trial.e2b = comm.bcast(self.trial.e2b, root=0)

        comm.barrier()

        # set walkers
        self.walkers = walkers

        if propagator is None:
            self.propagator = Propagator[type(self.hamiltonian)](self.qmc.dt)
            self.propagator.build(
                self.hamiltonian, self.trial, self.walkers, self.mpi_handler, verbose
            )
        else:
            self.propagator = propagator

        self.tsetup = time.time() - self._init_time

        # Using only default population control
        self.pcontrol = PopController(
            self.qmc.nwalkers, num_steps_per_block, self.mpi_handler, verbose=verbose
        )
        self.accumulators = WalkerAccumulator(
            ["Weight", "WeightFactor", "HybridEnergy"], self.qmc.nsteps
        )  # lagacy purposes??

        self.estimators = EstimatorHandler(
            comm,
            self.system,
            self.hamiltonian,
            self.trial,
            walker_state=self.accumulators,
            verbose=(comm.rank == 0 and verbose),
            filename=filename,
        )

        if self.mpi_handler.nmembers > 1:
            if comm.rank == 0:
                print("# Chunking hamiltonian.")
            self.hamiltonian.chunk(self.mpi_handler)
            if comm.rank == 0:
                print("# Chunking trial.")
            self.trial.chunk(self.mpi_handler)

        if config.get_option("use_gpu"):
            self.propagator.cast_to_cupy(verbose and comm.rank == 0)
            self.hamiltonian.cast_to_cupy(verbose and comm.rank == 0)
            self.trial.cast_to_cupy(verbose and comm.rank == 0)
            self.walkers.cast_to_cupy(verbose and comm.rank == 0)

        if comm.rank == 0:
            mem_avail = get_host_memory()
            print(f"# Available memory on the node is {mem_avail:4.3f} GB")
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
            self.walkers = psi
        self.setup_timers()
        eshift = 0.0

        self.walkers.orthogonalise()

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
            self.walkers,
        )
        self.accumulators.update(self.walkers)
        self.estimators.print_block(comm, 0, self.accumulators)
        self.accumulators.zero()

        synchronize()
        self.tsetup += time.time() - tzero_setup

        for step in range(1, total_steps + 1):
            synchronize()
            start_step = time.time()
            if step % self.qmc.nstblz == 0:
                start = time.time()
                self.walkers.orthogonalise()
                synchronize()
                self.tortho += time.time() - start
            start = time.time()

            self.propagator.propagate_walkers(
                self.walkers,
                self.hamiltonian,
                self.trial,
                eshift,
            )

            self.tprop_fbias = self.propagator.timer.tfbias
            self.tprop_ovlp = self.propagator.timer.tovlp
            self.tprop_update = self.propagator.timer.tupdate
            self.tprop_gf = self.propagator.timer.tgf
            self.tprop_vhs = self.propagator.timer.tvhs
            self.tprop_gemm = self.propagator.timer.tgemm

            start_clip = time.time()
            if step > 1:
                wbound = self.pcontrol.total_weight * 0.10
                xp.clip(
                    self.walkers.weight,
                    a_min=-wbound,
                    a_max=wbound,
                    out=self.walkers.weight,
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
                self.pcontrol.pop_control(self.walkers, comm)
                synchronize()
                self.tpopc += time.time() - start
                self.tpopc_send = self.pcontrol.timer.send_time
                self.tpopc_recv = self.pcontrol.timer.recv_time
                self.tpopc_comm = self.pcontrol.timer.communication_time
                self.tpopc_non_comm = self.pcontrol.timer.non_communication_time

            # accumulate weight, hybrid energy etc. across block
            start = time.time()
            self.accumulators.update(self.walkers)
            synchronize()
            self.testim += time.time() - start  # we dump this time into estimator
            # calculate estimators
            start = time.time()
            if step % self.qmc.nsteps == 0:
                self.estimators.compute_estimators(
                    comm,
                    self.system,
                    self.hamiltonian,
                    self.trial,
                    self.walkers,
                )
                self.estimators.print_block(comm, step // self.qmc.nsteps, self.accumulators)
                self.accumulators.zero()
            synchronize()
            self.testim += time.time() - start

            # restart write features disabled
            # if self.walkers.write_restart and step % self.walkers.write_freq == 0:
            #     self.walkers.write_walkers_batch(comm)

            if step < self.qmc.neqlb:
                eshift = self.accumulators.eshift
            else:
                eshift += self.accumulators.eshift - eshift
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
                print(f"# End Time: {time.asctime():s}")
                print(f"# Running time : {time.time() - self._init_time:.6f} seconds")
                print("# Timing breakdown (per call, total calls per block, total blocks):")
                print(f"# - Setup: {self.tsetup:.6f} s")
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

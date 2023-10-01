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
from typing import Dict, Optional, Tuple

from ipie.config import config
from ipie.estimators.estimator_base import EstimatorBase
from ipie.estimators.handler import EstimatorHandler
from ipie.hamiltonians.utils import get_hamiltonian
from ipie.propagation.propagator import Propagator
from ipie.qmc.options import QMCParams
from ipie.qmc.utils import set_rng_seed
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.utils import get_trial_wavefunction
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import get_host_memory, synchronize
from ipie.utils.io import to_json
from ipie.utils.misc import get_git_info, print_env_info
from ipie.utils.mpi import MPIHandler
from ipie.walkers.base_walkers import WalkerAccumulator
from ipie.walkers.pop_controller import PopController
from ipie.walkers.walkers_dispatch import get_initial_walker, UHFWalkersTrial


class AFQMC(object):
    """AFQMC driver.

    Parameters
    ----------
    system :
        System class. TODO Remove this?
    hamiltonian :
        Hamiltonian describing the system.
    trial :
        Trial wavefunction
    walkers :
        Walkers used for open ended random walk.
    propagator :
        Class describing how to propagate walkers.
    params :
        Parameters of simulation. See QMCParams for description.
    verbose : bool
        How much information to print.

    Attributes
    ----------
    _parallel_rng_seed : int
        Seed deduced from params.rng_seed which is generally different on each
            MPI process.
    """

    def __init__(
        self, system, hamiltonian, trial, walkers, propagator, params: QMCParams, verbose: int = 0
    ):
        self.system = system
        self.hamiltonian = hamiltonian
        self.trial = trial
        self.walkers = walkers
        self.propagator = propagator
        self.mpi_handler = MPIHandler()
        self.shared_comm = self.mpi_handler.shared_comm
        self.verbose = verbose
        self.verbosity = int(verbose)
        self.params = params
        self._init_time = time.time()
        self._parallel_rng_seed = set_rng_seed(params.rng_seed, self.mpi_handler.comm)

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
        pop_control_freq=5,
        verbose=True,
        mpi_handler=None,
    ) -> "AFQMC":
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
            Frequency at which to perform population control (in units of
                steps.) Default 25.
        verbose : bool
            Log verbosity. Default True i.e. print information to stdout.
        """
        if mpi_handler is None:
            mpi_handler = MPIHandler()
            comm = mpi_handler.comm
        else:
            comm = mpi_handler.comm
        params = QMCParams(
            num_walkers=num_walkers,
            total_num_walkers=num_walkers * comm.size,
            num_blocks=num_blocks,
            num_steps_per_block=num_steps_per_block,
            timestep=timestep,
            num_stblz=stabilize_freq,
            pop_control_freq=pop_control_freq,
            rng_seed=seed,
        )
        # 2. Calculation objects.
        system = Generic(num_elec)
        # TODO: do logic and logging in the function.
        if trial_wavefunction.compute_trial_energy:
            trial_wavefunction.calculate_energy(system, hamiltonian)
            trial_wavefunction.e1b = comm.bcast(trial_wavefunction.e1b, root=0)
            trial_wavefunction.e2b = comm.bcast(trial_wavefunction.e2b, root=0)
        comm.barrier()
        if walkers is None:
            _, initial_walker = get_initial_walker(trial_wavefunction)
            # TODO this is a factory method not a class
            walkers = UHFWalkersTrial(
                trial_wavefunction,
                initial_walker,
                system.nup,
                system.ndown,
                hamiltonian.nbasis,
                num_walkers,
                mpi_handler,
            )
            walkers.build(
                trial_wavefunction
            )  # any intermediates that require information from trial
        # TODO: this is a factory not a class
        propagator = Propagator[type(hamiltonian)](params.timestep)
        propagator.build(hamiltonian, trial_wavefunction, walkers, mpi_handler)
        return AFQMC(
            system,
            hamiltonian,
            trial_wavefunction,
            walkers,
            propagator,
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
        pack_cholesky=True,
        verbose=True,
    ) -> "AFQMC":
        """Factory method to build AFQMC driver from hamiltonian and trial wavefunction.

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
            num_elec, ham.nbasis, wfn_file, ndet_chunks=num_dets_chunk, verbose=_verbose
        )
        trial.half_rotate(ham, mpi_handler.scomm)
        return AFQMC.build(
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

    def distribute_hamiltonian(self):
        if self.mpi_handler.nmembers > 1:
            if self.mpi_handler.comm.rank == 0:
                print("# Chunking hamiltonian.")
            self.hamiltonian.chunk(self.mpi_handler)
            if self.mpi_handler.comm.rank == 0:
                print("# Chunking trial.")
            self.trial.chunk(self.mpi_handler)

    def copy_to_gpu(self):
        comm = self.mpi_handler.comm
        if config.get_option("use_gpu"):
            ngpus = xp.cuda.runtime.getDeviceCount()
            _ = xp.cuda.runtime.getDeviceProperties(0)
            xp.cuda.runtime.setDevice(self.shared_comm.rank)
            if comm.rank == 0:
                if ngpus > comm.size:
                    print(
                        f"# There are unused GPUs ({comm.size} MPI tasks but {ngpus} GPUs). "
                        " Check if this is really what you wanted."
                    )
            self.propagator.cast_to_cupy(self.verbose and comm.rank == 0)
            self.hamiltonian.cast_to_cupy(self.verbose and comm.rank == 0)
            self.trial.cast_to_cupy(self.verbose and comm.rank == 0)
            self.walkers.cast_to_cupy(self.verbose and comm.rank == 0)

    def get_env_info(self):
        # TODO: Move this somewhere else.
        this_uuid = str(uuid.uuid1())
        try:
            sha1, branch, local_mods = get_git_info()
        except:
            sha1 = "None"
            branch = "None"
            local_mods = []
        if self.verbose:
            self.sys_info = print_env_info(
                sha1, branch, local_mods, this_uuid, self.mpi_handler.size
            )
            mem_avail = get_host_memory()
            print(f"# MPI communicator : {type(self.mpi_handler.comm)}")
            print(f"# Available memory on the node is {mem_avail:4.3f} GB")

    def setup_estimators(
        self, filename, additional_estimators: Optional[Dict[str, EstimatorBase]] = None
    ):
        self.accumulators = WalkerAccumulator(
            ["Weight", "WeightFactor", "HybridEnergy"], self.params.num_steps_per_block
        )
        comm = self.mpi_handler.comm
        self.estimators = EstimatorHandler(
            self.mpi_handler.comm,
            self.system,
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
            comm, self.system, self.hamiltonian, self.trial, self.walkers
        )
        self.accumulators.update(self.walkers)
        self.estimators.print_block(comm, 0, self.accumulators)
        self.accumulators.zero()

    def setup_timers(self):
        # TODO: Better timer
        self.tsetup = 0
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

    def run(
        self,
        psi=None,
        estimator_filename=None,
        verbose=True,
        additional_estimators: Optional[Dict[str, EstimatorBase]] = None,
    ):
        """Perform AFQMC simulation on state object using open-ended random walk.

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

        self.pcontrol = PopController(
            self.params.num_walkers,
            self.params.num_steps_per_block,
            self.mpi_handler,
            verbose=self.verbose,
        )

        self.get_env_info()
        self.copy_to_gpu()
        self.distribute_hamiltonian()
        self.setup_estimators(estimator_filename, additional_estimators=additional_estimators)

        # TODO: This magic value of 2 is pretty much never controlled on input.
        # Moreover I'm not convinced having a two stage shift update actually
        # matters at all.
        num_eqlb_steps = 2.0 / self.params.timestep

        total_steps = self.params.num_steps_per_block * self.params.num_blocks

        synchronize()
        comm = self.mpi_handler.comm
        self.tsetup += time.time() - tzero_setup

        for step in range(1, total_steps + 1):
            synchronize()
            start_step = time.time()
            if step % self.params.num_stblz == 0:
                start = time.time()
                self.walkers.orthogonalise()
                synchronize()
                self.tortho += time.time() - start
            start = time.time()

            self.propagator.propagate_walkers(self.walkers, self.hamiltonian, self.trial, eshift)

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
                    self.walkers.weight, a_min=-wbound, a_max=wbound, out=self.walkers.weight
                )  # in-place clipping

            synchronize()
            self.tprop_clip += time.time() - start_clip

            start_barrier = time.time()
            if step % self.params.pop_control_freq == 0:
                comm.Barrier()
            self.tprop_barrier += time.time() - start_barrier

            self.tprop += time.time() - start
            if step % self.params.pop_control_freq == 0:
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
            if step % self.params.num_steps_per_block == 0:
                self.estimators.compute_estimators(
                    comm, self.system, self.hamiltonian, self.trial, self.walkers
                )
                self.estimators.print_block(
                    comm, step // self.params.num_steps_per_block, self.accumulators
                )
                self.accumulators.zero()
            synchronize()
            self.testim += time.time() - start

            # restart write features disabled
            # if self.walkers.write_restart and step % self.walkers.write_freq == 0:
            #     self.walkers.write_walkers_batch(comm)

            if step < num_eqlb_steps:
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
        nsteps = max(self.params.num_steps_per_block, 1)
        nblocks = max(self.params.num_blocks, 1)
        nstblz = max(nsteps // self.params.num_stblz, 1)
        npcon = max(nsteps // self.params.pop_control_freq, 1)
        if self.mpi_handler.rank == 0:
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
                        (self.tprop_update + self.tprop_clip) / (nblocks * nsteps), nsteps, nblocks
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

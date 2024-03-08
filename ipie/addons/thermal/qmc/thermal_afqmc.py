"""Driver to perform Thermal AFQMC calculation"""
import numpy
import time
import json
import uuid
from typing import Dict, Optional, Tuple

from ipie.addons.thermal.walkers.uhf_walkers import UHFThermalWalkers
from ipie.addons.thermal.propagation.propagator import Propagator
from ipie.addons.thermal.estimators.handler import ThermalEstimatorHandler
from ipie.addons.thermal.qmc.options import ThermalQMCParams

from ipie.utils.io import to_json
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import get_host_memory, synchronize
from ipie.utils.misc import get_git_info, print_env_info
from ipie.utils.mpi import MPIHandler
from ipie.systems.generic import Generic
from ipie.estimators.estimator_base import EstimatorBase
from ipie.walkers.pop_controller import PopController
from ipie.walkers.base_walkers import WalkerAccumulator
from ipie.qmc.afqmc import AFQMC
from ipie.qmc.utils import set_rng_seed


class ThermalAFQMC(AFQMC):
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
    def __init__(self, 
                 system, # For compatibility with 0T AFQMC code.
                 hamiltonian, 
                 trial, 
                 walkers, 
                 propagator, 
                 params: ThermalQMCParams, 
                 debug: bool = False,
                 verbose: bool = False):
        super().__init__(system, hamiltonian, trial, walkers, propagator, params, verbose)
        self.debug = debug
    
    @staticmethod
    def build(
            nelec: Tuple[int, int],
            mu: float,
            beta: float,
            hamiltonian,
            trial,
            nwalkers: int = 100,
            seed: int = None,
            nblocks: int = 100,
            timestep: float = 0.005,
            stabilize_freq: int = 5,
            pop_control_freq: int = 5,
            pop_control_method: str = 'pair_branch',
            lowrank: bool = False,
            debug: bool = False,
            verbose: bool = True,
            mpi_handler=None,) -> "Thermal AFQMC":
        """Factory method to build thermal AFQMC driver from hamiltonian and trial density matrix.

        Parameters
        ----------
        nelec : tuple(int, int)
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
        nsteps_per_block : int
            Number of Monte Carlo steps before estimators are evaluated.
                Default 25.
        nblocks : int
            Number of blocks to perform. Total number of steps = nblocks *
                nsteps_per_block.
        timestep : float
            Imaginary timestep. Default 0.005.
        stabilize_freq : float
            Frequency at which to perform QR factorization of walkers (in units
                of steps.) Default 25.
        pop_control_freq : int
            Frequency at which to perform population control (in units of
                steps.) Default 25.
        lowrank : bool
            Low-rank algorithm for thermal propagation.
        verbose : bool
            Log verbosity. Default True i.e. print information to stdout.
        """
        if mpi_handler is None:
            mpi_handler = MPIHandler()
            comm = mpi_handler.comm

        else:
            comm = mpi_handler.comm

        params = ThermalQMCParams(
                    beta=beta,
                    num_walkers=nwalkers,
                    total_num_walkers=nwalkers * comm.size,
                    num_blocks=nblocks,
                    timestep=timestep,
                    num_stblz=stabilize_freq,
                    pop_control_freq=pop_control_freq,
                    pop_control_method=pop_control_method,
                    rng_seed=seed)

        system = Generic(nelec) 
        walkers = UHFThermalWalkers(trial, hamiltonian.nbasis, nwalkers, 
                                    lowrank=lowrank, mpi_handler=mpi_handler, 
                                    verbose=verbose)
        propagator = Propagator[type(hamiltonian)](
                        timestep, mu, lowrank=lowrank, verbose=verbose)
        propagator.build(hamiltonian, trial=trial, walkers=walkers, 
                         mpi_handler=mpi_handler, verbose=verbose)
        return ThermalAFQMC(
                system,
                hamiltonian,
                trial,
                walkers,
                propagator,
                params,
                debug=debug,
                verbose=(verbose and comm.rank == 0))


    def run(self, 
            walkers = None, 
            verbose: bool = True,
            estimator_filename = None,
            additional_estimators: Optional[Dict[str, EstimatorBase]] = None):
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
        eshift = 0.

        if walkers is not None:
            self.walkers = walkers

        self.pcontrol = PopController(
                            self.params.num_walkers,
                            self.params.num_steps_per_block,
                            self.mpi_handler,
                            self.params.pop_control_method,
                            verbose=self.verbose)
        
        self.get_env_info()
        self.setup_estimators(estimator_filename, additional_estimators=additional_estimators)
        
        synchronize()
        comm = self.mpi_handler.comm
        self.tsetup += time.time() - ft_setup

        # Propagate.
        nwalkers = self.walkers.nwalkers
        total_steps = self.params.num_steps_per_block * self.params.num_blocks
        # TODO: This magic value of 2 is pretty much never controlled on input.
        # Moreover I'm not convinced having a two stage shift update actually
        # matters at all.
        neqlb_steps = 2.0 / self.params.timestep
        nslices = numpy.rint(self.params.beta / self.params.timestep).astype(int)

        for step in range(1, total_steps + 1):
            synchronize()
            start_path = time.time()

            for t in range(nslices):
                if self.verbosity >= 2 and comm.rank == 0:
                    print(" # Timeslice %d of %d." % (t, nslices))

                start = time.time()
                self.propagator.propagate_walkers(
                        self.walkers, self.hamiltonian, self.trial, eshift, debug=self.debug)

                self.tprop_fbias = self.propagator.timer.tfbias
                self.tprop_update = self.propagator.timer.tupdate
                self.tprop_vhs = self.propagator.timer.tvhs
                self.tprop_gemm = self.propagator.timer.tgemm
                
                start_clip = time.time()
                if t > 0:
                    wbound = self.pcontrol.total_weight * 0.10
                    xp.clip(self.walkers.weight, a_min=-wbound, a_max=wbound,
                            out=self.walkers.weight)  # In-place clipping.

                synchronize()
                self.tprop_clip += time.time() - start_clip

                start_barrier = time.time()
                if t % self.params.pop_control_freq == 0:
                    comm.Barrier()

                self.tprop_barrier += time.time() - start_barrier
                self.tprop += time.time() - start
                
                #print(f'self.walkers.weight = {self.walkers.weight}')
                if (t > 0) and (t % self.params.pop_control_freq == 0):
                    start = time.time()
                    self.pcontrol.pop_control(self.walkers, comm)
                    synchronize()
                    self.tpopc += time.time() - start
                    self.tpopc_send = self.pcontrol.timer.send_time
                    self.tpopc_recv = self.pcontrol.timer.recv_time
                    self.tpopc_comm = self.pcontrol.timer.communication_time
                    self.tpopc_non_comm = self.pcontrol.timer.non_communication_time
            
            # Accumulate weight, hybrid energy etc. across block.
            start = time.time()
            self.accumulators.update(self.walkers)
            self.testim += time.time() - start

            # Calculate estimators.
            start = time.time()
            if step % self.params.num_steps_per_block == 0:
                self.estimators.compute_estimators(self.hamiltonian,
                                                   self.trial, self.walkers)

                self.estimators.print_block(
                    comm, step // self.params.num_steps_per_block, self.accumulators)
                self.accumulators.zero()

            synchronize()
            self.testim += time.time() - start

            if step < neqlb_steps:
                eshift = self.accumulators.eshift

            else:
                eshift += self.accumulators.eshift - eshift

            self.walkers.reset(self.trial) # Reset stack, weights, phase.

            synchronize()
            self.tpath += time.time() - start_path


    def finalise(self, verbose=False):
        """Tidy up.

        Parameters
        ----------
        verbose : bool
            If true print out some information to stdout.
        """
        nsteps_per_block = max(self.params.num_steps_per_block, 1)
        nblocks = max(self.params.num_blocks, 1)
        nstblz = max(nsteps_per_block // self.params.num_stblz, 1)
        npcon = max(nsteps_per_block // self.params.pop_control_freq, 1)

        if self.mpi_handler.rank == 0:
            if verbose:
                print(f"# End Time: {time.asctime():s}")
                print(f"# Running time : {time.time() - self._init_time:.6f} seconds")
                print("# Timing breakdown (per call, total calls per block, total blocks):")
                print(f"# - Setup: {self.tsetup:.6f} s")
                print(
                    "# - Path update: {:.6f} s / block for {} total blocks".format(
                        self.tpath / (nblocks), nblocks
                    )
                )
                print(
                    "# - Propagation: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tprop / (nblocks * nsteps_per_block), nsteps_per_block, nblocks
                    )
                )
                print(
                    "#     -       Force bias: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tprop_fbias / (nblocks * nsteps_per_block), nsteps_per_block, nblocks
                    )
                )
                print(
                    "#     -              VHS: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tprop_vhs / (nblocks * nsteps_per_block), nsteps_per_block, nblocks
                    )
                )
                print(
                    "#     - Green's Function: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tprop_gf / (nblocks * nsteps_per_block), nsteps_per_block, nblocks
                    )
                )
                print(
                    "#     -          Overlap: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tprop_ovlp / (nblocks * nsteps_per_block), nsteps_per_block, nblocks
                    )
                )
                print(
                    "#     -   Weights Update: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        (self.tprop_update + self.tprop_clip) / (nblocks * nsteps_per_block),
                        nsteps_per_block,
                        nblocks,
                    )
                )
                print(
                    "#     -  GEMM operations: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tprop_gemm / (nblocks * nsteps_per_block), nsteps_per_block, nblocks
                    )
                )
                print(
                    "#     -          Barrier: {:.6f} s / call for {} call(s) in each of {} blocks".format(
                        self.tprop_barrier / (nblocks * nsteps_per_block), nsteps_per_block, nblocks
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


    def get_env_info(self):
        this_uuid = str(uuid.uuid1())

        try:
            sha1, branch, local_mods = get_git_info()

        except:
            sha1 = "None"
            branch = "None"
            local_mods = []

        if self.verbose:
            self.sys_info = print_env_info(
                sha1,
                branch,
                local_mods,
                this_uuid,
                self.mpi_handler.size,
            )

            mem_avail = get_host_memory()
            print(f"# Available memory on the node is {mem_avail:4.3f} GB")
    

    def setup_estimators(
        self,
        filename,
        additional_estimators: Optional[Dict[str, EstimatorBase]] = None):
        self.accumulators = WalkerAccumulator(
            ["Weight", "WeightFactor", "HybridEnergy"], self.params.num_steps_per_block)
        comm = self.mpi_handler.comm
        self.estimators = ThermalEstimatorHandler(
            self.mpi_handler.comm,
            self.hamiltonian,
            self.trial,
            walker_state=self.accumulators,
            verbose=(comm.rank == 0 and self.verbose),
            filename=filename)

        if additional_estimators is not None:
            for k, v in additional_estimators.items():
                self.estimators[k] = v

        # TODO: Move this to estimator and log uuid etc in serialization
        json.encoder.FLOAT_REPR = lambda o: format(o, ".6f")
        json_string = to_json(self)
        self.estimators.json_string = json_string
        self.estimators.initialize(comm)

        # Calculate estimates for initial distribution of walkers.
        self.estimators.compute_estimators(self.hamiltonian,
                                           self.trial, self.walkers)
        self.accumulators.update(self.walkers)
        self.estimators.print_block(comm, 0, self.accumulators)
        self.accumulators.zero()


    def setup_timers(self):
        self.tsetup = 0
        self.tpath = 0
        
        self.tprop = 0
        self.tprop_barrier = 0
        self.tprop_fbias = 0
        self.tprop_update = 0
        self.tprop_vhs = 0
        self.tprop_clip = 0

        self.testim = 0
        self.tpopc = 0

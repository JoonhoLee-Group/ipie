"""Driver to perform AFQMC calculation"""
import time
import uuid
from typing import Dict, Optional, Tuple

from ipie.thermal.propagation.propagator import Propagator
from ipie.thermal.walkers import UHFThermalWalkers
from ipie.thermal.qmc.options import ThermalQMCParams
from ipie.qmc.afqmc import AFQMC
from ipie.qmc.utils import set_rng_seed
from ipie.utils.misc import get_git_info, print_env_info
from ipie.systems.generic import Generic
from ipie.utils.mpi import MPIHandler


## This is now only applicable to the Generic case!
## See test_generic.py for example.

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
        Parameters of simulation. See QMCParams for description.
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
                 params: QMCParams, 
                 verbose: int = 0):
        super().__init__(system, hamiltonian, trial, walkers, propagator, params, verbose)

    def build(
            nelec: Tuple[],
            mu: float,
            beta: float,
            hamiltonian,
            trial,
            num_walkers: int = 100,
            seed: int = None,
            num_steps_per_block: int = 25,
            num_blocks: int = 100,
            timestep: float = 0.005,
            stabilize_freq: int = 5,
            pop_control_freq: int = 5,
            lowrank: bool = False,
            verbose: bool = True) -> "Thermal AFQMC":
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
        lowrank : bool
            Low-rank algorithm for thermal propagation.
        verbose : bool
            Log verbosity. Default True i.e. print information to stdout.
        """
        mpi_handler = MPIHandler()
        comm = mpi_handler.comm
        params = QMCParams(
                    num_walkers=num_walkers,
                    total_num_walkers=num_walkers * comm.size,
                    num_blocks=num_blocks,
                    num_steps_per_block=num_steps_per_block,
                    timestep=timestep,
                    beta=beta,
                    num_stblz=stabilize_freq,
                    pop_control_freq=pop_control_freq,
                    rng_seed=seed)
        
        walkers = UHFThermalWalkers(trial, hamiltonian.nbasis, num_walkers, 
                                    lowrank=lowrank, verbose=verbose)
        propagator = Propagator[type(hamiltonian)](
                        timestep, beta, lowrank=lowrank, verbose=verbose)
        propagator.build(hamiltonian, trial=trial, walkers=walkers, verbose=verbose)
        return ThermalAFQMC(
                system,
                hamiltonian,
                trial,
                walkers,
                propagator,
                params,
                verbose=(verbose and comm.rank == 0))


    def run(self, walkers=None, comm=None, verbose=None):
        """Perform Thermal AFQMC simulation on state object using open-ended random walk.

        Parameters
        ----------
        state : :class:`pie.state.State` object
            Model and qmc parameters.
        walk: :class:`pie.walker.Walkers` object
            Initial wavefunction / distribution of walkers.
        comm : MPI communicator
        """
        self.setup_timers()
        ft_setup time.time()

        if walkers is not None:
            self.walkers = walkers

        # Calculate estimates for initial distribution of walkers.
        self.estimators.estimators["mixed"].update(
            self.qmc,
            self.hamiltonian,
            self.trial,
            self.walkers,
            0,
            self.propagators.free_projection,
        )
        # Print out zeroth step for convenience.
        self.estimators.estimators["mixed"].print_step(comm, comm.size, 0, 1)
        
        num_walkers = self.walkers.nwalkers
        total_steps = self.params.num_steps_per_block * self.params.num_blocks
        num_slices = self.params.num_slices

        for step in range(1, total_steps + 1):
            start_path = time.time()

            for ts in range(num_slices):
                if self.verbosity >= 2 and comm.rank == 0:
                    print(" # Timeslice %d of %d." % (ts, num_slices))

                start = time.time()
                self.propagator.propagate_walkers(
                        self.walkers, self.hamiltonian, self.trial, eshift)
                
                start_clip = time.time()
                if step > 1:
                    wbound = self.pcontrol.total_weight * 0.10
                    xp.clip(self.walkers.weight, a_min=-wbound, a_max=wbound,
                            out=self.walkers.weight)  # in-place clipping

                self.tprop_clip += time.time() - start_clip

                start_barrier = time.time()
                if step % self.params.pop_control_freq == 0:
                    comm.Barrier()
                self.tprop_barrier += time.time() - start_barrier

                if step % self.params.pop_control_freq == 0:
                start = time.time()
                self.pcontrol.pop_control(self.walkers, comm)
                synchronize()
                self.tpopc += time.time() - start
                self.tpopc_send = self.pcontrol.timer.send_time
                self.tpopc_recv = self.pcontrol.timer.recv_time
                self.tpopc_comm = self.pcontrol.timer.communication_time
                self.tpopc_non_comm = self.pcontrol.timer.non_communication_time

                self.tprop += time.time() - start

                start = time.time()
                if ts % self.qmc.npop_control == 0 and ts != 0:
                    self.walkers.pop_control(comm)
                self.tpopc += time.time() - start
            self.tpath += time.time() - start_path
            start = time.time()
            self.estimators.update(
                self.qmc,
                self.hamiltonian,
                self.trial,
                self.walkers,
                step,
                self.propagators.free_projection,
            )
            self.testim += time.time() - start
            self.estimators.print_step(
                comm,
                comm.size,
                step,
                free_projection=self.propagators.free_projection,
            )
            self.walkers.reset(self.trial)

    def finalise(self, verbose):
        """Tidy up.

        Parameters
        ----------
        verbose : bool
            If true print out some information to stdout.
        """
        pass

    def determine_dtype(self, propagator, system):
        """Determine dtype for trial wavefunction and walkers.

        Parameters
        ----------
        propagator : dict
            Propagator input options.
        system : object
            System object.
        """
        hs_type = propagator.get("hubbard_stratonovich", "discrete")
        continuous = "continuous" in hs_type
        twist = system.ktwist.all() is not None
        return continuous or twist

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

    def setup_timers(self):
        self.tpath = 0
        self.tprop = 0
        self.tprop_clip = 0
        self.testim = 0
        self.tpopc = 0

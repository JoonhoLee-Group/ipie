"""Driver to perform AFQMC calculation"""
import time
import uuid
from ipie.thermal.propagation.utils import get_propagator
from ipie.thermal.walkers.handler import Walkers
from ipie.qmc.options import QMCOpts
from ipie.qmc.utils import set_rng_seed
from ipie.utils.misc import get_git_info, print_env_info
from ipie.qmc.afqmc import AFQMC
from ipie.qmc.options import QMCParams
from ipie.systems.generic import Generic
from ipie.utils.mpi import MPIHandler


## This is now only applicable to the Generic case!
## See test_generic.py for example.

class ThermalAFQMC(AFQMC):
    def __init__(
            self,
            num_elec,
            mu,
            beta,
            hamiltonian,
            trial,
            num_walkers: int = 100,
            seed: int = None,
            num_steps_per_block: int = 25,
            num_blocks: int = 100,
            timestep: float = 0.005,
            stabilize_freq=5,
            pop_control_freq=5,
            verbose=True,
    ):

        mpi_handler = MPIHandler()
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

        system = Generic(num_elec, verbose)
        system.mu = mu

        qmc = QMCOpts() # should be removed later after walker is cleaned up
        qmc.nwalkers = num_walkers
        qmc.ntot_walkers = num_walkers * comm.size
        qmc.beta = beta
        qmc.total_steps = num_blocks
        qmc.nsteps = 1
        qmc.ntime_slices = int(round(beta / timestep))
        qmc.rng_seed = set_rng_seed(seed, comm)
        qmc.dt = timestep

        if comm.rank == 0:
            self.uuid = str(uuid.uuid1())
            get_sha1 = True
            if get_sha1:
                self.sha1, self.branch, self.local_mods = get_git_info()
            else:
                self.sha1 = "None"
            if verbose:
                self.sys_info = print_env_info(
                    self.sha1, self.branch, self.local_mods, self.uuid, comm.size
                )

        self.qmc = qmc
        self.qmc.nstblz = 10
        self.qmc.npop_control = 1

        wlk_opts = {} # should be removed later after walker is cleaned up
        walkers = Walkers(
            system,
            hamiltonian,
            trial,
            qmc,
            walker_opts=wlk_opts,
            verbose=verbose)

        prop_opts = {} # should be removed later after walker is cleaned up

        propagator = get_propagator(
            prop_opts,
            qmc,
            system,
            hamiltonian,
            trial,
            verbose=verbose
        )
        self.propagators = propagator
        self.root = True

        super().__init__(system, hamiltonian, trial, walkers, propagator, params, verbose)

    def run(self, walkers=None, comm=None, verbose=None):
        """Perform AFQMC simulation on state object using open-ended random walk.

        Parameters
        ----------
        state : :class:`pie.state.State` object
            Model and qmc parameters.
        walk: :class:`pie.walker.Walkers` object
            Initial wavefunction / distribution of walkers.
        comm : MPI communicator
        """
        if walkers is not None:
            self.walkers = walkers
        self.setup_timers()
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

        for step in range(1, self.qmc.total_steps + 1):
            start_path = time.time()
            for ts in range(0, self.qmc.ntime_slices):
                if self.verbosity >= 2 and comm.rank == 0:
                    print(" # Timeslice %d of %d." % (ts, self.qmc.ntime_slices))
                start = time.time()
                for w in self.walkers.walkers:
                    self.propagators.propagate_walker(self.hamiltonian, w, ts, 0)
                    if (abs(w.weight) > w.total_weight * 0.10) and ts > 0:
                        w.weight = w.total_weight * 0.10
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

    def setup_timers(self):
        self.tpath = 0
        self.tprop = 0
        self.testim = 0
        self.tpopc = 0

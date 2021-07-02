"""Driver to perform AFQMC calculation"""
import sys
import json
import time
import numpy
import warnings
import uuid
from math import exp
import copy
import h5py
from pauxy.estimators.handler import Estimators
from pauxy.propagation.utils import get_propagator_driver
from pauxy.qmc.options import QMCOpts
from pauxy.qmc.utils import set_rng_seed
from pauxy.systems.utils import get_system
from pauxy.trial_wavefunction.utils import get_trial_wavefunction
from pauxy.utils.misc import (
        get_git_revision_hash,
        get_sys_info,
        get_node_mem
        )
from pauxy.utils.io import  to_json, serialise, get_input_value
from pauxy.utils.mpi import get_shared_comm
from pauxy.walkers.handler import Walkers


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
    cplx : bool
        If true then most numpy arrays are complex valued.
    system : system object.
        Container for model input options.
    qmc : :class:`pauxy.state.QMCOpts` object.
        Container for qmc input options.
    trial : :class:`pauxy.trial_wavefunction.X' object
        Trial wavefunction class.
    propagators : :class:`pauxy.propagation.Projectors` object
        Container for system specific propagation routines.
    estimators : :class:`pauxy.estimators.Estimators` object
        Estimator handler.
    psi : :class:`pauxy.walkers.Walkers` object
        Walker handler. Stores the AFQMC wavefunction.
    """

    def __init__(self, comm, options=None, system=None, trial=None,
                 parallel=False, verbose=False):
        if verbose is not None:
            self.verbosity = verbose
            if comm.rank != 0:
                self.verbosity = 0
            verbose = verbose > 0 and comm.rank == 0
        # 1. Environment attributes
        if comm.rank == 0:
            self.uuid = str(uuid.uuid1())
            get_sha1 = options.get('get_sha1', True)
            if get_sha1:
                try:
                    self.sha1, self.branch = get_git_revision_hash()
                except:
                    self.sha1 = 'None'
                    self.branch = 'None'
            else:
                self.sha1 = 'None'
                self.branch = 'None'
            if verbose:
                self.sys_info = get_sys_info(self.sha1, self.branch, self.uuid, comm.size)
        # Hack - this is modified later if running in parallel on
        # initialisation.
        self.root = comm.rank == 0
        self.rank = comm.rank
        self._init_time = time.time()
        self.run_time = time.asctime()
        self.shared_comm = get_shared_comm(comm, verbose=verbose)
        # 2. Calculation objects.
        if system is not None:
            self.system = system
        else:
            sys_opts = get_input_value(options, 'model',
                                       default={},
                                       alias=['system'],
                                       verbose=self.verbosity>1)
            self.system = get_system(sys_opts, verbose=verbose,
                                     comm=self.shared_comm)

        qmc_opt = get_input_value(options, 'qmc', default={},
                                  alias=['qmc_options'],
                                  verbose=self.verbosity>1)
        self.qmc = QMCOpts(qmc_opt, self.system,
                           verbose=self.verbosity>1)
        self.qmc.rng_seed = set_rng_seed(self.qmc.rng_seed, comm)
        self.cplx = self.determine_dtype(options.get('propagator', {}),
                                         self.system)
        twf_opt = get_input_value(options, 'trial', default={},
                                  alias=['trial_wavefunction'],
                                  verbose=self.verbosity>1)
        if trial is not None:
            self.trial = trial
        else:
            self.trial = (
                get_trial_wavefunction(self.system, options=twf_opt,
                                       comm=comm,
                                       scomm=self.shared_comm,
                                       verbose=verbose)
            )
            # if self.system.name == 'Generic':
                # self.trial.half_rotate(self.system, self.shared_comm)
        mem = get_node_mem()
        if comm.rank == 0:
            self.trial.calculate_energy(self.system)
        comm.barrier()
        prop_opt = options.get('propagator', {})
        self.propagators = get_propagator_driver(self.system, self.trial,
                                                 self.qmc, options=prop_opt,
                                                 verbose=verbose)
        self.tsetup = time.time() - self._init_time
        wlk_opts = get_input_value(options, 'walkers', default={},
                                   alias=['walker', 'walker_opts'],
                                   verbose=self.verbosity>1)
        est_opts = get_input_value(options, 'estimators', default={},
                                   alias=['estimates','estimator'],
                                   verbose=self.verbosity>1)
        est_opts['stack_size'] = wlk_opts.get('stack_size', 1)
        self.estimators = (
            Estimators(est_opts, self.root, self.qmc, self.system,
                       self.trial, self.propagators.BT_BP, verbose)
        )
        # Reset number of walkers so they are evenly distributed across
        # cores/ranks.
        # Number of walkers per core/rank.
        self.qmc.nwalkers = int(self.qmc.nwalkers/comm.size)
        # Total number of walkers.
        if self.qmc.nwalkers == 0:
            if comm.rank == 0:
                print("# WARNING: Not enough walkers for selected core count.")
                print("# There must be at least one walker per core set in the "
                      "input file.")
                print("# Setting one walker per core.")
            self.qmc.nwalkers = 1
        self.qmc.ntot_walkers = self.qmc.nwalkers * comm.size
        self.psi = Walkers(self.system, self.trial,
                           self.qmc, walker_opts=wlk_opts,
                           verbose=verbose,
                           nprop_tot=self.estimators.nprop_tot,
                           nbp=self.estimators.nbp,
                           comm=comm)
        if comm.rank == 0:
            mem_avail = get_node_mem()
            factor = float(self.system.ne) / self.system.nbasis
            mem = factor*self.trial._mem_required*self.shared_comm.size
            print("# Approx required for energy evaluation: {:.4f} GB.".format(mem))
            if mem > 0.5*mem_avail:
                print("# Warning: Memory requirements of calculation are high")
                print("# Consider using fewer walkers per node.")
                print("# Memory available: {:.6f}".format(mem_avail))
            json.encoder.FLOAT_REPR = lambda o: format(o, '.6f')
            json_string = to_json(self)
            self.estimators.json_string = json_string
            self.estimators.dump_metadata()
            if verbose:
                self.estimators.estimators['mixed'].print_key()
                self.estimators.estimators['mixed'].print_header()

    def run(self, psi=None, comm=None, verbose=True):
        """Perform AFQMC simulation on state object using open-ended random walk.

        Parameters
        ----------
        psi : :class:`pauxy.walker.Walkers` object
            Initial wavefunction / distribution of walkers.
        comm : MPI communicator
        """
        if psi is not None:
            self.psi = psi
        self.setup_timers()
        w0 = self.psi.walkers[0]
        eshift = 0
        (etot, e1b, e2b) = w0.local_energy(self.system, rchol=self.trial._rchol, eri=self.trial._eri, UVT=self.trial._UVT)
        # Calculate estimates for initial distribution of walkers.
        self.estimators.estimators['mixed'].update(self.system, self.qmc,
                                                   self.trial, self.psi, 0,
                                                   self.propagators.free_projection)
        # Print out zeroth step for convenience.
        if verbose:
            self.estimators.estimators['mixed'].print_step(comm, comm.size, 0, 1)

        for step in range(1, self.qmc.total_steps + 1):
            start_step = time.time()
            if step % self.qmc.nstblz == 0:
                start = time.time()
                self.psi.orthogonalise(self.trial,
                                       self.propagators.free_projection)
                self.tortho += time.time() - start
            start = time.time()
            for w in self.psi.walkers:
                if abs(w.weight) > 1e-8:
                    self.propagators.propagate_walker(w, self.system,
                                                      self.trial, eshift)
                if (abs(w.weight) > w.total_weight * 0.10) and step > 1:
                    w.weight = w.total_weight * 0.10
            self.tprop += time.time() - start
            if step % self.qmc.npop_control == 0:
                start = time.time()
                self.psi.pop_control(comm)
                self.tpopc += time.time() - start
            # calculate estimators
            start = time.time()
            self.estimators.update(self.system, self.qmc,
                                   self.trial, self.psi, step,
                                   self.propagators.free_projection)
            self.testim += time.time() - start
            self.estimators.print_step(comm, comm.size, step)
            if self.psi.write_restart and step % self.psi.write_freq == 0:
                self.psi.write_walkers(comm)
            if step < self.qmc.neqlb:
                eshift = self.estimators.estimators['mixed'].get_shift(self.propagators.hybrid)
            else:
                eshift += (self.estimators.estimators['mixed'].get_shift()-eshift)
            self.tstep += time.time() - start_step

    def finalise(self, verbose=False):
        """Tidy up.

        Parameters
        ----------
        verbose : bool
            If true print out some information to stdout.
        """
        if self.root:
            if verbose:
                print("# End Time: {:s}".format(time.asctime()))
                print("# Running time : {:.6f} seconds"
                      .format((time.time() - self._init_time)))
                print("# Timing breakdown (per processor, per block/step):")
                print("# - Setup: {:.6f} s".format(self.tsetup))
                nsteps = max(self.qmc.nsteps, 1)
                nstblz = max(nsteps // self.qmc.nstblz, 1)
                npcon = max(nsteps // self.qmc.npop_control, 1)
                print("# - Step: {:.6f} s".format((self.tstep/nsteps)))
                print("# - Orthogonalisation: {:.6f} s".format(self.tortho/nstblz))
                print("# - Propagation: {:.6f} s".format(self.tprop/nsteps))
                print("# - Estimators: {:.6f} s".format(self.testim/nsteps))
                print("# - Population control: {:.6f} s".format(self.tpopc/npcon))


    def determine_dtype(self, propagator, system):
        """Determine dtype for trial wavefunction and walkers.

        Parameters
        ----------
        propagator : dict
            Propagator input options.
        system : object
            System object.
        """
        hs_type = propagator.get('hubbard_stratonovich', 'discrete')
        continuous = 'continuous' in hs_type
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
        from pauxy.analysis import blocking
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
        self.testim = 0
        self.tpopc = 0
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
        from pauxy.analysis import blocking
        filename = self.estimators.h5f_name
        try:
            bp_rdm, bp_rdm_err = blocking.reblock_rdm(filename)
        except IndexError:
            bp_rdm, bp_rdm_err = None, None
        return (bp_rdm, bp_rdm_err)

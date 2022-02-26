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
from pie.estimators.handler import Estimators
from pie.estimators.local_energy_batch import local_energy_batch
from pie.propagation.utils import get_propagator_driver
from pie.qmc.options import QMCOpts
from pie.qmc.utils import set_rng_seed
from pie.systems.utils import get_system
from pie.hamiltonians.utils import get_hamiltonian
from pie.trial_wavefunction.utils import get_trial_wavefunction
from pie.utils.misc import (
        get_git_revision_hash,
        get_sys_info,
        get_node_mem
        )
from pie.utils.io import  to_json, serialise, get_input_value
from pie.utils.mpi import get_shared_comm
from pie.walkers.walker_batch_handler import WalkerBatchHandler
from pie.utils.misc import is_cupy


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

    def __init__(self, comm, options=None, system=None, hamiltonian=None, trial=None,
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
        
        qmc_opt = get_input_value(options, 'qmc', default={},
                                  alias=['qmc_options'],
                                  verbose=self.verbosity>1)
        gpu = get_input_value(qmc_opt, 'gpu', default=False, verbose=self.verbosity>1)

        self.shared_comm = comm

        # 2. Calculation objects.
        if system is not None:
            self.system = system
        else:
            sys_opts = get_input_value(options, 'system',
                                       default={},
                                       alias=['model'],
                                       verbose=self.verbosity>1)
            self.system = get_system(sys_opts, verbose=verbose,
                                     comm=self.shared_comm)
        if hamiltonian is not None:
            self.hamiltonian = hamiltonian
        else:
            ham_opts = get_input_value(options, 'hamiltonian',
                                       default={},
                                       verbose=self.verbosity>1)
            # backward compatibility with previous code (to be removed)
            for item in sys_opts.items():
                if item[0].lower() == 'name' and 'name' in ham_opts.keys():
                    continue
                ham_opts[item[0]] = item[1]
            self.hamiltonian = get_hamiltonian (self.system, ham_opts, verbose = verbose, comm=self.shared_comm)

        self.qmc = QMCOpts(qmc_opt, self.system,
                           verbose=self.verbosity>1)
        if (self.qmc.nwalkers == None):
            assert(self.qmc.nwalkers_per_task is not None)
            self.qmc.nwalkers = self.qmc.nwalkers_per_task * comm.size
        if (self.qmc.nwalkers_per_task == None):
            assert(self.qmc.nwalkers is not None)
            self.qmc.nwalkers_per_task = int(self.qmc.nwalkers/comm.size)
        # Reset number of walkers so they are evenly distributed across
        # cores/ranks.
        # Number of walkers per core/rank.
        self.qmc.nwalkers = int(self.qmc.nwalkers/comm.size) # This should be gone in the future
        assert(self.qmc.nwalkers == self.qmc.nwalkers_per_task)
        # Total number of walkers.
        if self.qmc.nwalkers == 0:
            if comm.rank == 0:
                print("# WARNING: Not enough walkers for selected core count.")
                print("# There must be at least one walker per core set in the "
                      "input file.")
                print("# Setting one walker per core.")
            self.qmc.nwalkers = 1
        self.qmc.ntot_walkers = self.qmc.nwalkers * comm.size
            
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
                get_trial_wavefunction(self.system, self.hamiltonian, options=twf_opt,
                                       comm=comm,
                                       scomm=self.shared_comm,
                                       verbose=verbose)
            )
        mem = get_node_mem()
        if comm.rank == 0:
            self.trial.calculate_energy(self.system, self.hamiltonian)
            print("# Trial wfn energy is {}".format(self.trial.energy))
        comm.barrier()
        prop_opt = options.get('propagator', {})
        if comm.rank == 0:
            print("# Getting propagator driver")
        self.propagators = get_propagator_driver(self.system, self.hamiltonian, self.trial,
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
            Estimators(est_opts, self.root, self.qmc, self.system, self.hamiltonian,
                       self.trial, self.propagators.BT_BP, verbose)
        )
        if comm.rank == 0:
            print("# Getting WalkerBatchHandler")
        self.psi = WalkerBatchHandler(self.system, self.hamiltonian, self.trial,
                           self.qmc, walker_opts=wlk_opts,
                           verbose=verbose,
                           nprop_tot=self.estimators.nprop_tot,
                           nbp=self.estimators.nbp,
                           comm=comm)

        if (self.qmc.gpu):
            try:
                import cupy
                assert(cupy.is_available())
            except:
                if comm.rank == 0:
                    print("# cupy is unavailble but GPU calculation is requested")
                exit()
            ngpus = cupy.cuda.runtime.getDeviceCount()
            props = cupy.cuda.runtime.getDeviceProperties(0)
            name = props['name'].decode()
            device = cupy.cuda.Device(0) 
            if comm.rank == 0:
                print("# {} GPUs are available".format(ngpus))
                print("# Device name: {}".format(name))
                print("# Compute capability: {}".format(device.compute_capability))
                if (ngpus > comm.size):
                    print("# There are unused GPUs ({} MPI tasks but {} GPUs). Check if this is really what you wanted.".format(comm.size,ngpus))

            if (ngpus < comm.size):
                if comm.rank == 0:
                    print("# Not enough GPUs availalbe. {} MPI tasks requested but {} GPUs available.".format(comm.size, ngpus))
                exit()
            
            if comm.rank == 0:
                print("# Casting numpy arrays to cupy arrays")

            if comm.rank == 0:
                print("# Casting arrays in propagators")
            self.propagators.cast_to_cupy(verbose)
            if comm.rank == 0:
                print("# Casting arrays in hamiltonian")
            self.hamiltonian.cast_to_cupy(verbose)
            if comm.rank == 0:
                print("# Casting arrays in trial")
            self.trial.cast_to_cupy(verbose)
            if comm.rank == 0:
                print("# Casting arrays in walkers_batch")
            self.psi.walkers_batch.cast_to_cupy(verbose)

        if comm.rank == 0:
            mem_avail = get_node_mem()
            print("# Available memory on the node is {} MB".format(mem_avail))
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
        psi : :class:`pie.walker.Walkers` object
            Initial wavefunction / distribution of walkers.
        comm : MPI communicator
        """

        if is_cupy(self.psi.walkers_batch.phia): # if even one array is a cupy array we should assume the rest is done with cupy
            import cupy
            assert(cupy.is_available())
            zeros = cupy.zeros
            abs = cupy.abs
            ndarray = cupy.ndarray
            array = cupy.asnumpy
            sum = cupy.sum
        else:
            zeros = numpy.zeros
            ndarray = numpy.ndarray
            array = numpy.array
            abs = numpy.abs
            sum = numpy.sum

        if psi is not None:
            self.psi = psi
        self.setup_timers()
        eshift = 0.0

        # Calculate estimates for initial distribution of walkers.
        self.estimators.estimators['mixed'].update_batch(self.qmc, self.system, self.hamiltonian,
                                                   self.trial, self.psi.walkers_batch, 0,
                                                   self.propagators.free_projection)

        # Print out zeroth step for convenience.
        if verbose:
            self.estimators.estimators['mixed'].print_step(comm, comm.size, 0, 1)

        for step in range(1, self.qmc.total_steps + 1):
            start_step = time.time()
            if step % self.qmc.nstblz == 0:
                start = time.time()
                self.psi.orthogonalise(self.trial, self.propagators.free_projection)
                self.tortho += time.time() - start
            start = time.time()

            self.propagators.propagate_walker_batch(self.psi.walkers_batch, self.system, self.hamiltonian, self.trial, eshift)
            
            self.tprop_fbias = self.propagators.tfbias
            self.tprop_ovlp = self.propagators.tovlp
            self.tprop_update = self.propagators.tupdate
            self.tprop_gf = self.propagators.tgf
            self.tprop_vhs = self.propagators.tvhs
            self.tprop_gemm = self.propagators.tgemm

            rescale_idx = abs(self.psi.walkers_batch.weight) > self.psi.walkers_batch.total_weight * 0.10
            if step > 1:
                nrescales = sum(rescale_idx)
                if type(nrescales) == ndarray:
                    nrescales = array(nrescales)
                    nrescales = int(nrescales[()])
                if (nrescales > 0):
                    new_weights = zeros(nrescales, dtype = numpy.float64)
                    new_weights.fill(self.psi.walkers_batch.total_weight * 0.10)
                    self.psi.walkers_batch.weight[rescale_idx] = new_weights

            if step % self.qmc.npop_control == 0:
                comm.Barrier()
            self.tprop += time.time() - start
            if step % self.qmc.npop_control == 0:
                start = time.time()
                self.psi.pop_control(comm)
                self.tpopc += time.time() - start
                self.tpopc_send = self.psi.send_time
                self.tpopc_recv = self.psi.recv_time
                self.tpopc_comm = self.psi.communication_time
                self.tpopc_non_comm = self.psi.non_communication_time

            # calculate estimators
            start = time.time()
            self.estimators.update_batch(self.qmc, self.system, self.hamiltonian,
                                   self.trial, self.psi.walkers_batch, step,
                                   self.propagators.free_projection)
            self.estimators.print_step(comm, comm.size, step)
            self.testim += time.time() - start
            if self.psi.write_restart and step % self.psi.write_freq == 0:
                self.psi.write_walkers_batch(comm)
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
        nsteps = max(self.qmc.nsteps, 1)
        nblocks = max(self.qmc.nblocks, 1)
        nstblz = max(nsteps // self.qmc.nstblz, 1)
        npcon = max(nsteps // self.qmc.npop_control, 1)
        if self.root:
            if verbose:
                print("# End Time: {:s}".format(time.asctime()))
                print("# Running time : {:.6f} seconds"
                      .format((time.time() - self._init_time)))
                print("# Timing breakdown (per call, total calls per block, total blocks):")
                print("# - Setup: {:.6f} s".format(self.tsetup))
                print("# - Step: {:.6f} s / call for {} calls in each of {} blocks".format(self.tstep/(nblocks*nsteps), nsteps, nblocks))
                print("# - Propagation: {:.6f} s / call for {} call(s) in each of {} blocks".format(self.tprop/(nblocks*nsteps), nsteps, nblocks))
                print("#     -       Force bias: {:.6f} s / call for {} call(s) in each of {} blocks".format(self.tprop_fbias/(nblocks*nsteps), nsteps, nblocks))
                print("#     -              VHS: {:.6f} s / call for {} call(s) in each of {} blocks".format(self.tprop_vhs/(nblocks*nsteps), nsteps, nblocks))
                print("#     - Green's Function: {:.6f} s / call for {} call(s) in each of {} blocks".format(self.tprop_gf/(nblocks*nsteps), nsteps, nblocks))
                print("#     -          Overlap: {:.6f} s / call for {} call(s) in each of {} blocks".format(self.tprop_ovlp/(nblocks*nsteps), nsteps, nblocks))
                print("#     -   Weights Update: {:.6f} s / call for {} call(s) in each of {} blocks".format(self.tprop_update/(nblocks*nsteps), nsteps, nblocks))
                print("#     -  GEMM operations: {:.6f} s / call for {} call(s) in each of {} blocks".format(self.tprop_gemm/(nblocks*nsteps), nsteps, nblocks))
                print("# - Estimators: {:.6f} s / call for {} call(s)".format(self.testim/nblocks, nblocks))
                print("# - Orthogonalisation: {:.6f} s / call for {} call(s) in each of {} blocks".format(self.tortho/(nstblz*nblocks), nstblz, nblocks))
                print("# - Population control: {:.6f} s / call for {} call(s) in each of {} blocks".format(self.tpopc/(npcon*nblocks), npcon, nblocks))
                print("#       -     Commnication: {:.6f} s / call for {} call(s) in each of {} blocks".format(self.tpopc_comm/(npcon*nblocks), npcon, nblocks))
                print("#       - Non-Commnication: {:.6f} s / call for {} call(s) in each of {} blocks".format(self.tpopc_non_comm/(npcon*nblocks), npcon, nblocks))


    def determine_dtype(self, propagator, system):
        """Determine dtype for trial wavefunction and walkers.

        Parameters
        ----------
        propagator : dict
            Propagator input options.
        system : object
            system object.
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
        from pie.analysis import blocking
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
        from pie.analysis import blocking
        filename = self.estimators.h5f_name
        try:
            bp_rdm, bp_rdm_err = blocking.reblock_rdm(filename)
        except IndexError:
            bp_rdm, bp_rdm_err = None, None
        return (bp_rdm, bp_rdm_err)

"""Routines and classes for estimation of observables."""

from __future__ import print_function

import copy
import os
import time
import warnings

import h5py
import numpy
import scipy.linalg

from ipie.estimators.arbitrary import ArbitraryEstimators
from ipie.estimators.mixed import Mixed
from ipie.utils.io import get_input_value


class Estimators(object):
    """Container for qmc estimates of observables.

    Parameters
    ----------
    estimates : dict
        input options detailing which estimators to calculate. By default only
        mixed estimates will be calculated.
    root : bool
        True if on root/master processor.
    qmc : :class:`ipie.state.QMCOpts` object.
        Container for qmc input options.
    system : :class:`ipie.hubbard.Hubbard` / system object in general.
        Container for model input options.
    trial : :class:`ipie.trial_wavefunction.X' object
        Trial wavefunction class.
    BT2 : :class:`numpy.ndarray`
        One body propagator.
    verbose : bool
        If true we print out additional setup information.

    Attributes
    ----------
    estimators : dict
        Dictionary of estimator objects.
    """

    def __init__(
        self, estimates, comm, qmc, system, hamiltonian, trial, BT2, verbose=False
    ):
        if verbose:
            print("# Setting up estimator object.")
        if comm.rank == 0:
            self.index = estimates.get("index", 0)
            self.filename = estimates.get("filename", None)
            self.basename = estimates.get("basename", "estimates")
            if self.filename is None:
                overwrite = estimates.get("overwrite", True)
                self.filename = self.basename + ".%s.h5" % self.index
                while os.path.isfile(self.filename) and not overwrite:
                    self.index = int(self.filename.split(".")[1])
                    self.index = self.index + 1
                    self.filename = self.basename + ".%s.h5" % self.index
            with h5py.File(self.filename, "w") as fh5:
                pass
            if verbose:
                print("# Writing estimator data to {}.".format(self.filename))
        else:
            self.filename = None
        # Sub-members:
        # 1. Back-propagation
        mixed = estimates.get("mixed", {})
        self.estimators = {}
        dtype = complex
        self.estimators["mixed"] = Mixed(
            mixed, system, hamiltonian, comm.rank==0, self.filename, qmc, trial, dtype
        )
        arbitrary = get_input_value(
                estimates,
                "arbitrary",
                {},
                alias=['user', 'miscellaneous'],
                verbose=verbose)
        if arbitrary != {}:
            self.estimators["arbitrary"] = ArbitraryEstimators(
                arbitrary, system, hamiltonian, comm, self.filename, qmc, trial, dtype
            )
        self.nprop_tot = None
        self.nbp = None

        if verbose:
            print("# Finished settting up estimator object.")

    def reset(self, root):
        if root:
            self.increment_file_number()
            self.dump_metadata()
            for k, e in self.estimators.items():
                e.setup_output(self.filename)

    def dump_metadata(self):
        with h5py.File(self.filename, "a") as fh5:
            fh5["metadata"] = self.json_string

    def increment_file_number(self):
        self.index = self.index + 1
        self.filename = self.basename + ".%s.h5" % self.index

    def print_step(self, comm, nprocs, step, nsteps=None, free_projection=False):
        """Print QMC estimates.

        Parameters
        ----------
        comm :
            MPI communicator.
        nprocs : int
            Number of processors.
        step : int
            Current iteration number.
        nmeasure : int
            Number of steps between measurements.
        """
        for k, e in self.estimators.items():
            e.print_step(
                comm, nprocs, step, nsteps=nsteps, free_projection=free_projection
            )

    def update_batch(
        self, qmc, system, hamiltonian, trial, psi, step, free_projection=False
    ):
        """Update estimators with bached psi

        Parameters
        ----------
        system : system object in general.
            Container for model input options.
        qmc : :class:`ipie.state.QMCOpts` object.
            Container for qmc input options.
        trial : :class:`ipie.trial_wavefunction.X' object
            Trial wavefunction class.
        # TODO FDM: Mark for removal
        psi : :class:`ipie.legacy.walkers.WalkersBatch` object
            CPMC wavefunction.
        step : int
            Current simulation step
        free_projection : bool
            True if doing free projection.
        """
        for k, e in self.estimators.items():
            e.update_batch(qmc, system, hamiltonian, trial, psi, step, free_projection)

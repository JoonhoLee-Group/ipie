"""Routines and classes for estimation of observables."""

from __future__ import print_function

import copy
import os
import time
import warnings

import h5py
import numpy
import scipy.linalg

from ipie.estimators.energy import EnergyEstimator
from ipie.utils.io import get_input_value

# Some supported (non-custom) estimators
_predefined_estimators = {
        'energy': EnergyEstimator,
        }


class EstimatorHandler(object):
    """Container for qmc options of observables.

    Parameters
    ----------
    comm : MPI.COMM_WORLD
        MPI Communicator
    qmc : :class:`ipie.state.QMCOpts` object.
        Container for qmc input options.
    system : :class:`ipie.hubbard.Hubbard` / system object in general.
        Container for model input options.
    trial : :class:`ipie.trial_wavefunction.X' object
        Trial wavefunction class.
    verbose : bool
        If true we print out additional setup information.
    options: dict
        input options detailing which estimators to calculate. By default only
        mixed options will be calculated.

    Attributes
    ----------
    estimators : dict
        Dictionary of estimator objects.
    """

    def __init__(
        self,
        comm,
        qmc,
        system
        hamiltonian,
        trial,
        options={},
        verbose=False
    ):
        if verbose:
            print("# Setting up estimator object.")
        if comm.rank == 0:
            self.index = options.get("index", 0)
            self.filename = options.get("filename", None)
            self.basename = options.get("basename", "options")
            if self.filename is None:
                overwrite = options.get("overwrite", True)
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
        observables = options.get("observables", {"energy": {}})
        self.estimators = {}
        for obs, obs_dict in observables.items():
            try:
                self.estimators[obs] = (
                        predefined_estimators[obs](
                            comm=comm,
                            qmc=qmc,
                            system=system,
                            ham=hamiltonian,
                            trial=trial,
                            options=obs_dict
                            )
                        )
            except KeyError:
                raise RuntimeError(f"unknown observable: {obs}")
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
        """Print QMC options.

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

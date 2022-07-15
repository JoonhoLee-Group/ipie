import h5py
import numpy as np

from mpi4py import MPI

from ipie.estimators.utils import H5EstimatorHelper
from ipie.observables.overlap import compute_walker_overlaps
from ipie.utils.io import get_input_value
from ipie.utils.misc import is_cupy


class ArbitraryEstimators(object):

    def  __init__(
            self,
            options,
            system,
            hamiltonian,
            comm,
            filename,
            qmc,
            trial,
            dtype
            ):

        self.observables = {
                'walker_overlaps': {
                    'shape': (qmc.ntot_walkers,),
                    'function': compute_walker_overlaps
                    }
            }
        self.buffer_size = get_input_value(options, 'buffer_size', default=1000)
        self.nsteps = qmc.nsteps
        self.to_compute = {}
        obs_keys = self.observables.keys()
        total_size = 0
        sizes = []
        for k, v in options.items():
            if k in obs_keys:
                estim_size = np.product(self.observables[k]['shape'])
                self.observables[k]['size'] = estim_size
                total_size += estim_size
                self.to_compute[k] = self.observables[k]
                if comm.rank == 0:
                    with h5py.File(filename, "a") as fh5:
                        fh5[f'arbitrary/{k}_shape'] = self.observables[k]['shape']
        total_size = max(1, total_size)
        if qmc.gpu:
            import cupy
            self.local_estimates = cp.zeros(total_size, dtype=cp.complex128)
        else:
            self.local_estimates = np.zeros(total_size, dtype=np.complex128)
        self.global_estimates = np.zeros(total_size, dtype=np.complex128)

        if comm.rank == 0:
            self.output = H5EstimatorHelper(filename, "arbitrary",
                    chunk_size=self.buffer_size, shape=(total_size,)
                    )
        self._slice = slice(comm.rank*qmc.nwalkers, (comm.rank+1)*qmc.nwalkers)

    def update_batch(
        self, qmc, system, hamiltonian, trial, walker_batch, step, free_projection=False
    ):
        """Update mixed estimates for walkers.

        Parameters
        ----------
        qmc : :class:`ipie.state.QMCOpts` object.
            Container for qmc input options.
        system : system object.
            Container for model input options.
        hamiltonian : hamiltonian object.
            Container for hamiltonian input options.
        trial : :class:`ipie.trial_wavefunction.X' object
            Trial wavefunction class.
        psi : :class:`ipie.legacy.walkers.Walkers` object
            CPMC wavefunction.
        step : int
            Current simulation step
        free_projection : bool
            True if doing free projection.
        """
        if is_cupy(
            walker_batch.weight
        ):  # if even one array is a cupy array we should assume the rest is done with cupy
            import cupy

            assert cupy.is_available()
            array = cupy.array
            zeros = cupy.zeros
            sum = cupy.sum
            abs = cupy.abs
        else:
            array = np.array
            zeros = np.zeros
            sum = np.sum
            abs = np.abs

        for k, obs in self.to_compute.items():
            self.local_estimates[self._slice] = (
                    obs['function'](walker_batch, trial).ravel()
                    )

    def print_step(self, comm, nprocs, step, nsteps=None, free_projection=False):
        """Print mixed estimates to file.

        This reduces estimates arrays over processors. On return estimates
        arrays are zerod.

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
        if is_cupy(
            self.local_estimates
        ):  # if even one array is a cupy array we should assume the rest is done with cupy
            import cupy

            assert cupy.is_available()
            array = cupy.asnumpy
        else:
            array = np.array

        if step % self.nsteps != 0:
            return
        if nsteps is None:
            nsteps = self.nsteps
        es = array(self.local_estimates)
        comm.Reduce(es, self.global_estimates, op=MPI.SUM)
        gs = self.global_estimates
        if comm.rank == 0:
            self.output.push_to_chunk(gs, "data")
            self.output.increment()
        self.zero()

    def zero(self):
        """Zero (in the appropriate sense) various estimator arrays."""
        self.local_estimates[:] = 0
        self.global_estimates[:] = 0

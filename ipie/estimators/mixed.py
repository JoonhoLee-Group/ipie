import h5py
import numpy

try:
    from mpi4py import MPI

    mpi_sum = MPI.SUM
except ImportError:
    mpi_sum = None
import scipy.linalg
import time

from ipie.estimators.utils import H5EstimatorHelper
from ipie.estimators.local_energy_batch import local_energy_batch as local_energy
from ipie.estimators.greens_function_batch import greens_function

from ipie.utils.io import format_fixed_width_strings, format_fixed_width_floats
from ipie.utils.misc import dotdict, is_cupy


class Mixed(object):
    """Class for computing mixed estimates.

    Parameters
    ----------
    mixed : dict
        Input options for mixed estimates.
    root : bool
        True if on root/master processor.
    qmc : :class:`ipie.state.QMCOpts` object.
        Container for qmc input options.
    trial : :class:`ipie.trial_wavefunction.X' object
        Trial wavefunction class.
    dtype : complex or float
        Output type.

    Attributes
    ----------
    nmeasure : int
        Max number of measurements.
    nreg : int
        Number of regular estimates (exluding iteration).
    G : :class:`numpy.ndarray`
        One-particle RDM.
    estimates : :class:`numpy.ndarray`
        Store for mixed estimates per processor.
    global_estimates : :class:`numpy.ndarray`
        Store for mixed estimates accross all processors.
    names : :class:`ipie.estimators.EstimEnum`
        Enum for locating estimates in estimates array.
    header : int
        Output header.
    key : dict
        Explanation of output.
    output : :class:`ipie.estimators.H5EstimatorHelper`
        Class for outputting data to HDF5 group.
    output : :class:`ipie.estimators.H5EstimatorHelper`
        Class for outputting rdm data to HDF5 group.
    """

    def __init__(
        self, mixed_opts, system, hamiltonian, root, filename, qmc, trial, dtype
    ):
        self.eval_energy = mixed_opts.get("evaluate_energy", True)
        self.energy_eval_freq = mixed_opts.get("energy_eval_freq", None)
        if self.energy_eval_freq is None:
            self.energy_eval_freq = qmc.nsteps
        self.verbose = mixed_opts.get("verbose", True)

        # number of steps per block
        self.nsteps = qmc.nsteps
        self.header = [
            "Iteration",
            "WeightFactor",
            "Weight",
            "ENumer",
            "EDenom",
            "ETotal",
            "E1Body",
            "E2Body",
            "EHybrid",
            "Overlap",
        ]

        self.header.append("Time")
        self.nreg = len(self.header[1:])
        self.dtype = dtype
        self.G = numpy.zeros((2, hamiltonian.nbasis, hamiltonian.nbasis), dtype)

        dms_size = 0

        self.eshift = numpy.array([0, 0])

        if qmc.gpu:
            import cupy

            self.estimates = cupy.zeros(self.nreg + dms_size, dtype=dtype)
            self.names = get_estimator_enum(False)
            self.estimates[self.names.time] = time.time()
        else:
            self.estimates = numpy.zeros(self.nreg + dms_size, dtype=dtype)
            self.names = get_estimator_enum(False)
            self.estimates[self.names.time] = time.time()

        self.global_estimates = numpy.zeros(self.nreg + dms_size, dtype=dtype)
        self.key = {
            "Iteration": "Simulation iteration. iteration*dt = tau.",
            "WeightFactor": "Rescaling Factor from population control.",
            "Weight": "Total walker weight.",
            "E_num": "Numerator for projected energy estimator.",
            "E_denom": "Denominator for projected energy estimator.",
            "ETotal": "Projected energy estimator.",
            "E1Body": "Mixed one-body energy estimator.",
            "E2Body": "Mixed two-body energy estimator.",
            "EHybrid": "Hybrid energy.",
            "Overlap": "Walker average overlap.",
            "Nav": "Average number of electrons.",
            "Time": "Time per processor to complete one iteration.",
        }
        if root:
            self.setup_output(filename)

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
            array = numpy.array
            zeros = numpy.zeros
            sum = numpy.sum
            abs = numpy.abs

        # When using importance sampling we only need to know the current
        # walkers weight as well as the local energy, the walker's overlap
        # with the trial wavefunction is not needed.
        if step % self.energy_eval_freq == 0:
            greens_function(walker_batch, trial)
            if self.eval_energy:
                energy = local_energy(system, hamiltonian, walker_batch, trial)
            else:
                energy = zeros(walker_batch.nwalkers, 3, dtype=numpy.complex128)
            self.estimates[self.names.enumer] += sum(
                walker_batch.weight * energy[:, 0].real
            )
            self.estimates[self.names.e1b : self.names.e2b + 1] += array(
                [
                    sum(walker_batch.weight * energy[:, 1].real),
                    sum(walker_batch.weight * energy[:, 2].real),
                ]
            )
            self.estimates[self.names.edenom] += sum(walker_batch.weight)

        self.estimates[self.names.uweight] += sum(walker_batch.unscaled_weight)
        self.estimates[self.names.weight] += sum(walker_batch.weight)
        self.estimates[self.names.ovlp] += sum(
            walker_batch.weight * abs(walker_batch.ovlp)
        )
        self.estimates[self.names.ehyb] += sum(
            walker_batch.weight * walker_batch.hybrid_energy
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
            self.estimates
        ):  # if even one array is a cupy array we should assume the rest is done with cupy
            import cupy

            assert cupy.is_available()
            array = cupy.asnumpy
        else:
            array = numpy.array

        if step % self.nsteps != 0:
            return
        if nsteps is None:
            nsteps = self.nsteps
        es = array(self.estimates)
        ns = self.names
        es[ns.time] = (time.time() - es[ns.time]) / nprocs
        es[ns.uweight : ns.weight + 1] /= nsteps
        es[ns.ehyb : ns.time + 1] /= nsteps
        comm.Reduce(es, self.global_estimates, op=mpi_sum)
        gs = self.global_estimates
        if comm.rank == 0:
            gs[ns.eproj] = gs[ns.enumer]
            gs[ns.eproj : ns.e2b + 1] = gs[ns.eproj : ns.e2b + 1] / gs[ns.edenom]
            gs[ns.ehyb] /= gs[ns.weight]
            gs[ns.ovlp] /= gs[ns.weight]
            eshift = numpy.array([gs[ns.ehyb], gs[ns.eproj]])
        else:
            eshift = numpy.array([0, 0])
        eshift = comm.bcast(eshift, root=0)
        self.eshift = eshift
        if comm.rank == 0:
            if self.verbose:
                print(format_fixed_width_floats([step] + list(gs[: ns.time + 1].real)))
            self.output.push([step] + list(gs[: ns.time + 1]), "energies")
            self.output.increment()
        self.zero()

    def print_key(self, eol="", encode=False):
        """Print out information about what the estimates are.

        Parameters
        ----------
        eol : string, optional
            String to append to output, e.g., Default : ''.
        encode : bool
            In True encode output to be utf-8.
        """
        header = (
            eol
            + "# Explanation of output column headers:\n"
            + "# -------------------------------------"
            + eol
        )
        if encode:
            header = header.encode("utf-8")
        print(header)
        for (k, v) in self.key.items():
            s = "# %s : %s" % (k, v) + eol
            if encode:
                s = s.encode("utf-8")
            print(s)

    def print_header(self, eol="", encode=False):
        r"""Print out header for estimators

        Parameters
        ----------
        eol : string, optional
            String to append to output, Default : ''.
        encode : bool
            In True encode output to be utf-8.

        Returns
        -------
        None
        """
        s = format_fixed_width_strings(self.header) + eol
        if encode:
            s = s.encode("utf-8")
        print(s)

    def get_shift(self, hybrid=True):
        """get hybrid shift.

        parameters
        ----------
        hybrid : bool
            true if using hybrid propgation
        returns
        -------
        eshift : float
            walker averaged hybrid energy.
        """
        if hybrid:
            return self.eshift[0].real
        else:
            return self.eshift[1].real

    def zero(self):
        """Zero (in the appropriate sense) various estimator arrays."""
        self.estimates[:] = 0
        self.global_estimates[:] = 0
        self.estimates[self.names.time] = time.time()

    def setup_output(self, filename):
        with h5py.File(filename, "a") as fh5:
            fh5["basic/headers"] = numpy.array(self.header).astype("S")
        self.output = H5EstimatorHelper(filename, "basic")


def get_estimator_enum(thermal=False):
    keys = [
        "uweight",
        "weight",
        "enumer",
        "edenom",
        "eproj",
        "e1b",
        "e2b",
        "ehyb",
        "ovlp",
    ]
    if thermal:
        keys.append("nav")
    keys.append("time")
    enum = {}
    for v, k in enumerate(keys):
        enum[k] = v
    return dotdict(enum)

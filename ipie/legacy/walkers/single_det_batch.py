import numpy
import scipy.linalg

from ipie.estimators.greens_function_batch import greens_function
from ipie.legacy.trial_wavefunction.free_electron import FreeElectron
from ipie.legacy.trial_wavefunction.harmonic_oscillator import \
    HarmonicOscillator
from ipie.legacy.walkers.stack import FieldConfig
from ipie.propagation.overlap import calc_overlap_single_det, get_calc_overlap
from ipie.utils.linalg import sherman_morrison
from ipie.utils.misc import get_numeric_names
from ipie.walkers.walker_batch import WalkerBatch


class SingleDetWalkerBatch(WalkerBatch):
    """UHF style walker.

    Parameters
    ----------
    system : object
        System object.
    hamiltonian : object
        Hamiltonian object.
    trial : object
        Trial wavefunction object.
    nwalkers : int
        The number of walkers in this batch
    walker_opts : dict
        Input options
    index : int
        Element of trial wavefunction to initalise walker to.
    nprop_tot : int
        Number of back propagation steps (including imaginary time correlation
                functions.)
    nbp : int
        Number of back propagation steps.
    """

    def __init__(
        self,
        system,
        hamiltonian,
        trial,
        nwalkers,
        walker_opts={},
        index=0,
        nprop_tot=None,
        nbp=None,
    ):
        WalkerBatch.__init__(
            self,
            system,
            hamiltonian,
            trial,
            nwalkers,
            walker_opts=walker_opts,
            index=index,
            nprop_tot=nprop_tot,
            nbp=nbp,
        )
        self.name = "SingleDetWalkerBatch"

        calc_overlap = get_calc_overlap(trial)
        self.ot = calc_overlap(self, trial)
        self.ovlp = self.ot
        self.le_oratio = 1.0

        self.Ga = numpy.zeros(
            shape=(nwalkers, hamiltonian.nbasis, hamiltonian.nbasis),
            dtype=numpy.complex128,
        )
        if self.rhf:
            self.Gb = None
        else:
            self.Gb = numpy.zeros(
                shape=(nwalkers, hamiltonian.nbasis, hamiltonian.nbasis),
                dtype=numpy.complex128,
            )

        self.Ghalfa = numpy.zeros(
            shape=(nwalkers, system.nup, hamiltonian.nbasis), dtype=numpy.complex128
        )
        if self.rhf:
            self.Ghalfb = None
        else:
            self.Ghalfb = numpy.zeros(
                shape=(nwalkers, system.ndown, hamiltonian.nbasis),
                dtype=numpy.complex128,
            )

        greens_function(self, trial)

    # This function casts relevant member variables into cupy arrays
    def cast_to_cupy(self, verbose=False):
        WalkerBatch.cast_to_cupy(self, verbose)

        size = self.Ga.size + self.Gb.size + self.Ghalfa.size + self.Ghalfb.size
        size += self.ot.size
        size += self.ovlp.size
        if verbose:
            expected_bytes = size * 16.0
            print(
                "# SingleDetWalkerBatch: expected to allocate {:4.3f} GB".format(
                    expected_bytes / 1024**3
                )
            )

        import cupy

        self.ot = cupy.asarray(self.ot)
        self.ovlp = cupy.asarray(self.ovlp)
        self.Ga = cupy.asarray(self.Ga)
        self.Gb = cupy.asarray(self.Gb)
        self.Ghalfa = cupy.asarray(self.Ghalfa)
        self.Ghalfb = cupy.asarray(self.Ghalfb)

        free_bytes, total_bytes = cupy.cuda.Device().mem_info
        used_bytes = total_bytes - free_bytes
        if verbose:
            print(
                "# SingleDetWalkerBatch: using {:4.3f} GB out of {:4.3f} GB memory on GPU".format(
                    used_bytes / 1024**3, total_bytes / 1024**3
                )
            )

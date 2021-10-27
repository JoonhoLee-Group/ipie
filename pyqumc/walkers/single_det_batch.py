import numpy
import scipy.linalg
from pyqumc.trial_wavefunction.free_electron import FreeElectron
from pyqumc.utils.linalg import sherman_morrison
from pyqumc.walkers.stack import FieldConfig
from pyqumc.walkers.walker_batch import WalkerBatch
from pyqumc.utils.misc import get_numeric_names
from pyqumc.trial_wavefunction.harmonic_oscillator import HarmonicOscillator
from pyqumc.estimators.greens_function import greens_function_single_det
from pyqumc.propagation.overlap import calc_overlap_single_det

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

    def __init__(self, system, hamiltonian, trial, nwalkers, walker_opts={}, index=0, nprop_tot=None, nbp=None):
        WalkerBatch.__init__(self, system, hamiltonian, trial, nwalkers, 
                        walker_opts=walker_opts, index=index,
                        nprop_tot=nprop_tot, nbp=nbp)
        self.name = "SingleDetWalkerBatch"

        self.ot = calc_overlap_single_det(self, trial)
        self.ovlp = self.ot
        self.le_oratio = 1.0

        self.Ga = numpy.zeros(shape=(nwalkers, hamiltonian.nbasis, hamiltonian.nbasis),
                             dtype=numpy.complex128)
        self.Gb = numpy.zeros(shape=(nwalkers, hamiltonian.nbasis, hamiltonian.nbasis),
                             dtype=numpy.complex128)

        self.Ghalfa = numpy.zeros(shape=(nwalkers, system.nup, hamiltonian.nbasis),
                                 dtype=numpy.complex128)
        self.Ghalfb = numpy.zeros(shape=(nwalkers, system.ndown, hamiltonian.nbasis),
                                 dtype=numpy.complex128)
        
        greens_function_single_det(self, trial)
    
    # This function casts relevant member variables into cupy arrays
    def cast_to_gpu (self):
        WalkerBatch.cast_to_gpu(self)
        import cupy
        self.ot = cupy.array(ot)
        self.ovlp = cupy.array(ovlp)
        self.Ga = cupy.array(self.Ga)
        self.Gb = cupy.array(self.Gb)
        self.Ghalfa = cupy.array(self.Ghalfa)
        self.Ghalfb = cupy.array(self.Ghalfb)

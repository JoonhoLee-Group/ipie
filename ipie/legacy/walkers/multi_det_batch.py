import copy

import numpy
import scipy.linalg

from ipie.estimators.greens_function_batch import greens_function_multi_det
from ipie.legacy.estimators.local_energy import local_energy_multi_det
from ipie.propagation.overlap import get_calc_overlap
from ipie.utils.misc import get_numeric_names
from ipie.walkers.walker_batch import WalkerBatch


class MultiDetTrialWalkerBatch(WalkerBatch):
    """Single-det walker for multi-det trials.

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
        det_weights="zeros",
        verbose=False,
        nprop_tot=None,
        nbp=None,
    ):
        if verbose:
            print("# Setting up ipie.legacy.walkers.MultiDetTrialWalkerBatch object.")

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
        self.name = "MultiDetTrialWalkerBatch"
        self.ndets = trial.psi.shape[0]

        # TODO: RENAME to something less like weight
        # This stores an array of overlap matrices with the various elements of
        # the trial wavefunction.
        self.det_weights = numpy.zeros(
            (self.nwalkers, self.ndets), dtype=numpy.complex128
        )
        self.det_ovlpas = numpy.zeros(
            (self.nwalkers, self.ndets), dtype=numpy.complex128
        )
        self.det_ovlpbs = numpy.zeros(
            (self.nwalkers, self.ndets), dtype=numpy.complex128
        )

        # Compute initial overlap. Avoids issues with singular matrices for
        # PHMSD.
        calc_overlap = get_calc_overlap(trial)

        self.ot = calc_overlap(self, trial)
        # TODO: fix name.
        self.ovlp = self.ot
        self.le_oratio = 1.0
        if verbose:
            print(
                "# Initial overlap of walker with trial wavefunction: {:13.8e}".format(
                    self.ot.real
                )
            )

        # Green's functions for various elements of the trial wavefunction.
        if trial.ortho_expansion:
            self.G0a = numpy.zeros(
                shape=(self.nwalkers, hamiltonian.nbasis, hamiltonian.nbasis),
                dtype=numpy.complex128,
            )  # reference 1-GF
            self.G0b = numpy.zeros(
                shape=(self.nwalkers, hamiltonian.nbasis, hamiltonian.nbasis),
                dtype=numpy.complex128,
            )  # reference 1-GF
            # self.Ghalf0a = numpy.zeros(shape=(self.nwalkers, system.nup,hamiltonian.nbasis), dtype=numpy.complex128) # reference 1-GF
            # self.Ghalf0b = numpy.zeros(shape=(self.nwalkers, system.ndown,hamiltonian.nbasis), dtype=numpy.complex128) # reference 1-GF
            self.Q0a = numpy.zeros(
                shape=(self.nwalkers, hamiltonian.nbasis, hamiltonian.nbasis),
                dtype=numpy.complex128,
            )  # reference 1-GF
            self.Q0b = numpy.zeros(
                shape=(self.nwalkers, hamiltonian.nbasis, hamiltonian.nbasis),
                dtype=numpy.complex128,
            )  # reference 1-GF
            self.CIa = numpy.zeros(
                shape=(self.nwalkers, hamiltonian.nbasis, system.nup),
                dtype=numpy.complex128,
            )
            self.CIb = numpy.zeros(
                shape=(self.nwalkers, hamiltonian.nbasis, system.ndown),
                dtype=numpy.complex128,
            )
        # else:
        self.Gia = numpy.zeros(
            shape=(self.nwalkers, self.ndets, hamiltonian.nbasis, hamiltonian.nbasis),
            dtype=numpy.complex128,
        )
        self.Gib = numpy.zeros(
            shape=(self.nwalkers, self.ndets, hamiltonian.nbasis, hamiltonian.nbasis),
            dtype=numpy.complex128,
        )
        self.Gihalfa = numpy.zeros(
            shape=(self.nwalkers, self.ndets, system.nup, hamiltonian.nbasis),
            dtype=numpy.complex128,
        )
        self.Gihalfb = numpy.zeros(
            shape=(self.nwalkers, self.ndets, system.ndown, hamiltonian.nbasis),
            dtype=numpy.complex128,
        )

        # Actual green's function contracted over determinant index in Gi above.
        # i.e., <psi_T|c_i^d c_j|phi>
        self.Ga = numpy.zeros(
            shape=(nwalkers, hamiltonian.nbasis, hamiltonian.nbasis),
            dtype=numpy.complex128,
        )
        self.Gb = numpy.zeros(
            shape=(nwalkers, hamiltonian.nbasis, hamiltonian.nbasis),
            dtype=numpy.complex128,
        )
        self.Ghalfa = numpy.zeros(
            shape=(nwalkers, system.nup, hamiltonian.nbasis), dtype=numpy.complex128
        )
        self.Ghalfb = numpy.zeros(
            shape=(nwalkers, system.ndown, hamiltonian.nbasis), dtype=numpy.complex128
        )

        # Contains overlaps of the current walker with the trial wavefunction.
        greens_function_multi_det(self, trial)

    def contract_one_body(self, ints, trial):
        numer = 0.0
        denom = 0.0
        for iw in range(self.nwalkers):
            for i, Gia, Gib in enumerate(zip(self.Gia[iw], self.Gib[iw])):
                ofac = trial.coeffs[i].conj() * self.ovlpas[i] * self.ovlpbs[i]
                numer += ofac * numpy.dot((Gia + Gib).ravel(), ints.ravel())
                denom += ofac
        return numer / denom

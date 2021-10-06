import numpy
import scipy.linalg
from pyqumc.trial_wavefunction.free_electron import FreeElectron
from pyqumc.utils.linalg import sherman_morrison
from pyqumc.walkers.stack import FieldConfig
from pyqumc.walkers.walker_batch import WalkerBatch
from pyqumc.utils.misc import get_numeric_names
from pyqumc.trial_wavefunction.harmonic_oscillator import HarmonicOscillator

class SingleDetWalkerBatch(WalkerBatch):
    """UHF style walker.

    Parameters
    ----------
    system : object
        System object.
    trial : object
        Trial wavefunction object.
    options : dict
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
        self.inv_ovlp = [[0.0, 0.0] for iw in range(self.nwalkers)]

        self.inverse_overlap(trial)
        self.ot = self.calc_overlap(trial)
        self.le_oratio = 1.0
        self.ovlp = self.ot

        self.G = numpy.zeros(shape=(nwalkers, 2, hamiltonian.nbasis, hamiltonian.nbasis),
                             dtype=trial.psi.dtype)

        self.Ghalf = numpy.zeros(shape=(nwalkers, 2, system.nup, hamiltonian.nbasis),
                                 dtype=trial.psi.dtype)
        self.greens_function(trial)
        self.buff_names, self.buff_size = get_numeric_names(self.__dict__)

    def inverse_overlap(self, trial):
        """Compute inverse overlap matrix from scratch.

        Parameters
        ----------
        trial : :class:`numpy.ndarray`
            Trial wavefunction.
        """
        nup = self.nup
        ndown = self.ndown

        for iw in range(self.nwalkers):
            self.inv_ovlp[iw][0] = (
                scipy.linalg.inv((trial.psi[:,:nup].conj()).T.dot(self.phi[iw][:,:nup]))
            )
            self.inv_ovlp[iw][1] = numpy.zeros(self.inv_ovlp[iw][0].shape)
            if (ndown>0):
                self.inv_ovlp[iw][1] = (
                    scipy.linalg.inv((trial.psi[:,nup:].conj()).T.dot(self.phi[iw][:,nup:]))
                )

    def update_inverse_overlap(self, trial, vtup, vtdown, i):
        """Update inverse overlap matrix given a single row update of walker.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.
        vtup : :class:`numpy.ndarray`
            Update vector for spin up sector.
        vtdown : :class:`numpy.ndarray`
            Update vector for spin down sector.
        i : int
            Basis index.
        """
        nup = self.nup
        ndown = self.ndown

        for iw in range(self.nwalkers):
            self.inv_ovlp[iw][0] = (
                sherman_morrison(self.inv_ovlp[iw][0], trial.psi[i,:nup].conj(), vtup[iw])
            )
            self.inv_ovlp[iw][1] = (
                sherman_morrison(self.inv_ovlp[iw][1], trial.psi[i,nup:].conj(), vtdown[iw])
            )
#   Hubbard model specific function
    # def calc_otrial(self, trial):
    #     """Caculate overlap with trial wavefunction.

    #     Parameters
    #     ----------
    #     trial : object
    #         Trial wavefunction object.

    #     Returns
    #     -------
    #     ot : float / complex
    #         Overlap.
    #     """

    #     ot = []
    #     for iw in range(self.nwalkers):
    #         sign_a, logdet_a = numpy.linalg.slogdet(self.inv_ovlp[iw][0])
    #         nbeta = self.ndown
    #         sign_b, logdet_b = 1.0, 0.0
    #         if nbeta > 0:
    #             sign_b, logdet_b = numpy.linalg.slogdet(self.inv_ovlp[iw][1])
    #         det = sign_a*sign_b*numpy.exp(logdet_a+logdet_b-self.log_shift)
    #         ot += [1.0/det]

    #     return ot

    def calc_overlap(self, trial):
        """Caculate overlap with trial wavefunction.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.

        Returns
        -------
        ot : float / complex
            Overlap.
        """
        na = self.ndown
        nb = self.ndown
        ot = []
        for iw in range(self.nwalkers):
            Oalpha = numpy.dot(trial.psi[:,:na].conj().T, self.phi[iw][:,:na])
            sign_a, logdet_a = numpy.linalg.slogdet(Oalpha)
            logdet_b, sign_b = 0.0, 1.0
            if nb > 0:
                Obeta = numpy.dot(trial.psi[:,na:].conj().T, self.phi[iw][:,na:])
                sign_b, logdet_b = numpy.linalg.slogdet(Obeta)

            ot += [sign_a*sign_b*numpy.exp(logdet_a+logdet_b-self.log_shift[iw])]

        return ot

    def reortho(self, trial):
        """reorthogonalise walker.

        parameters
        ----------
        trial : object
            trial wavefunction object. for interface consistency.
        """
        nup = self.nup
        ndown = self.ndown

        detR = []
        for iw in range(self.nwalkers):
            (self.phi[iw][:,:nup], Rup) = scipy.linalg.qr(self.phi[iw][:,:nup],
                                                      mode='economic')
            Rdown = numpy.zeros(Rup.shape)
            if ndown > 0:
                (self.phi[iw][:,nup:], Rdn) = scipy.linalg.qr(self.phi[iw][:,nup:],
                                                            mode='economic')
            # TODO: FDM This isn't really necessary, the absolute value of the
            # weight is used for population control so this shouldn't matter.
            # I think this is a legacy thing.
            # Wanted detR factors to remain positive, dump the sign in orbitals.
            Rup_diag = numpy.diag(Rup)
            signs_up = numpy.sign(Rup_diag)
            if ndown > 0:
                Rdn_diag = numpy.diag(Rdn)
                signs_dn = numpy.sign(Rdn_diag)
            self.phi[iw][:,:nup] = numpy.dot(self.phi[iw][:,:nup], numpy.diag(signs_up))
            if ndown > 0:
                self.phi[iw][:,nup:] = numpy.dot(self.phi[iw][:,nup:], numpy.diag(signs_dn))
            # include overlap factor
            # det(R) = \prod_ii R_ii
            # det(R) = exp(log(det(R))) = exp((sum_i log R_ii) - C)
            # C factor included to avoid over/underflow
            log_det = numpy.sum(numpy.log(numpy.abs(Rup_diag)))
            if ndown > 0:
                log_det += numpy.sum(numpy.log(numpy.abs(Rdn_diag)))
            detR += [numpy.exp(log_det-self.detR_shift)]
            self.log_detR[iw] += numpy.log(detR)
            self.detR[iw] = detR
            self.ot[iw] = self.ot[iw] / detR
            self.ovlp[iw] = self.ot[iw]
        return detR

    def greens_function(self, trial):
        """Compute walker's green's function.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.
        Returns
        -------
        det : float64 / complex128
            Determinant of overlap matrix.
        """
        nup = self.nup
        ndown = self.ndown

        det = []

        for iw in range(self.nwalkers):
            ovlp = numpy.dot(self.phi[iw][:,:nup].T, trial.psi[:,:nup].conj())
            self.Ghalf[iw][0] = numpy.dot(scipy.linalg.inv(ovlp), self.phi[iw][:,:nup].T)
            self.G[iw][0] = numpy.dot(trial.psi[:,:nup].conj(), self.Ghalf[iw][0])
            sign_a, log_ovlp_a = numpy.linalg.slogdet(ovlp)
            sign_b, log_ovlp_b = 1.0, 0.0
            if ndown > 0:
                ovlp = numpy.dot(self.phi[iw][:,nup:].T, trial.psi[:,nup:].conj())
                sign_b, log_ovlp_b = numpy.linalg.slogdet(ovlp)
                self.Ghalf[iw][1] = numpy.dot(scipy.linalg.inv(ovlp), self.phi[iw][:,nup:].T)
                self.G[iw][1] = numpy.dot(trial.psi[:,nup:].conj(), self.Ghalf[iw][1])
            det += [sign_a*sign_b*numpy.exp(log_ovlp_a+log_ovlp_b-self.log_shift[iw])]

        return det

    def rotated_greens_function(self):
        """Compute "rotated" walker's green's function.

        Green's function without trial wavefunction multiplication.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.
        """
        nup = self.nup
        ndown = self.ndown
        for iw in range(self.nwalkers):
            self.Ghalf[iw][0] = self.phi[iw][:,:nup].dot(self.inv_ovlp[iw][0])
            self.Ghalf[iw][1] = numpy.zeros(self.Ghalf[iw][0].shape)
            if (ndown>0):
                self.Ghalf[iw][1] = self.phi[iw][:,nup:].dot(self.inv_ovlp[iw][1])


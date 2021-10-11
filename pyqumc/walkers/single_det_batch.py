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
        self.inv_ovlpa = numpy.zeros((self.nwalkers, system.nup, system.nup), dtype=numpy.complex128)
        self.inv_ovlpb = numpy.zeros((self.nwalkers, system.ndown, system.ndown), dtype=numpy.complex128)

        self.inverse_overlap(trial)
        self.ot = self.calc_overlap(trial)
        self.le_oratio = 1.0
        self.ovlp = self.ot

        self.Ga = numpy.zeros(shape=(nwalkers, hamiltonian.nbasis, hamiltonian.nbasis),
                             dtype=numpy.complex128)
        self.Gb = numpy.zeros(shape=(nwalkers, hamiltonian.nbasis, hamiltonian.nbasis),
                             dtype=numpy.complex128)

        self.Ghalfa = numpy.zeros(shape=(nwalkers, system.nup, hamiltonian.nbasis),
                                 dtype=numpy.complex128)
        self.Ghalfb = numpy.zeros(shape=(nwalkers, system.ndown, hamiltonian.nbasis),
                                 dtype=numpy.complex128)
        self.greens_function(trial)
        # self.buff_names, self.buff_size = get_numeric_names(self.__dict__)
        # self.buff_size /= float(self.nwalkers)

        # Grab objects that are walker specific
        # WARNING!! One has to add names to the list here if new objects are added
        # self.buff_names = ["weight", "unscaled_weight", "phase", "alive", "phi", 
        #                    "ot", "ovlp", "eloc", "ot_bp", "weight_bp", "phi_old",
        #                    "hybrid_energy", "weights", "inv_ovlpa", "inv_ovlpb", "Ga", "Gb", "Ghalfa", "Ghalfb"]
        self.buff_names = ["weight", "unscaled_weight", "phase", "phi", "hybrid_energy"]
        self.buff_size = round(self.set_buff_size_single_walker()/float(self.nwalkers))


    def set_buff_size_single_walker(self):
        names = []
        size = 0
        for k, v in self.__dict__.items():
            # try:
            #     print(k, v.size)
            # except AttributeError:
            #     print("failed", k, v)
            if (not (k in self.buff_names)):
                continue
            if isinstance(v, (numpy.ndarray)):
                names.append(k)
                size += v.size
            elif isinstance(v, (int, float, complex)):
                names.append(k)
                size += 1
            elif isinstance(v, list):
                names.append(k)
                for l in v:
                    if isinstance(l, (numpy.ndarray)):
                        size += l.size
                    elif isinstance(l, (int, float, complex)):
                        size += 1
        return size


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
            self.inv_ovlpa[iw] = (
                scipy.linalg.inv((trial.psi[:,:nup].conj()).T.dot(self.phi[iw][:,:nup]))
            )
            if (ndown>0):
                self.inv_ovlpb[iw] = (
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
            self.inv_ovlpa[iw] = (
                sherman_morrison(self.inv_ovlp[iw][0], trial.psi[i,:nup].conj(), vtup[iw])
            )
            self.inv_ovlpb[iw] = (
                sherman_morrison(self.inv_ovlp[iw][1], trial.psi[i,nup:].conj(), vtdown[iw])
            )

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
        ot = numpy.zeros(self.nwalkers, dtype=numpy.complex128)
        for iw in range(self.nwalkers):
            Oalpha = numpy.dot(trial.psi[:,:na].conj().T, self.phi[iw][:,:na])
            sign_a, logdet_a = numpy.linalg.slogdet(Oalpha)
            logdet_b, sign_b = 0.0, 1.0
            if nb > 0:
                Obeta = numpy.dot(trial.psi[:,na:].conj().T, self.phi[iw][:,na:])
                sign_b, logdet_b = numpy.linalg.slogdet(Obeta)

            ot[iw] = sign_a*sign_b*numpy.exp(logdet_a+logdet_b-self.log_shift[iw])


        return ot

    def reortho(self):
        """reorthogonalise walkers.

        parameters
        ----------
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
            detR += [numpy.exp(log_det-self.detR_shift[iw])]
            self.log_detR[iw] += numpy.log(detR[iw])
            self.detR[iw] = detR[iw]
            # print(self.ot[iw], detR[iw])
            self.ot[iw] = self.ot[iw] / detR[iw]
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
            ovlp_inv = scipy.linalg.inv(ovlp)
            self.Ghalfa[iw] = numpy.dot(ovlp_inv, self.phi[iw][:,:nup].T)
            self.Ga[iw] = numpy.dot(trial.psi[:,:nup].conj(), self.Ghalfa[iw])
            sign_a, log_ovlp_a = numpy.linalg.slogdet(ovlp)
            sign_b, log_ovlp_b = 1.0, 0.0
            if ndown > 0:
                ovlp = numpy.dot(self.phi[iw][:,nup:].T, trial.psi[:,nup:].conj())
                sign_b, log_ovlp_b = numpy.linalg.slogdet(ovlp)
                self.Ghalfb[iw] = numpy.dot(scipy.linalg.inv(ovlp), self.phi[iw][:,nup:].T)
                self.Gb[iw] = numpy.dot(trial.psi[:,nup:].conj(), self.Ghalfb[iw])
            det += [sign_a*sign_b*numpy.exp(log_ovlp_a+log_ovlp_b-self.log_shift[iw])]
        det = numpy.array(det, dtype=numpy.complex128)

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
            self.Ghalfa[iw] = self.phi[iw][:,:nup].dot(self.inv_ovlp[iw][0])
            self.Ghalfb[iw] = numpy.zeros(self.Ghalfa[iw].shape)
            if (ndown>0):
                self.Ghalfb[iw] = self.phi[iw][:,nup:].dot(self.inv_ovlp[iw][1])


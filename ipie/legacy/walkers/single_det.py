import numpy
import scipy.linalg

from ipie.legacy.trial_wavefunction.free_electron import FreeElectron
from ipie.legacy.trial_wavefunction.harmonic_oscillator import \
    HarmonicOscillator
from ipie.legacy.walkers.stack import FieldConfig
from ipie.legacy.walkers.walker import Walker
from ipie.utils.linalg import sherman_morrison
from ipie.utils.misc import get_numeric_names

# from ipie.estimators.mixed import local_energy, local_energy_hh




class SingleDetWalker(Walker):
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

    def __init__(
        self,
        system,
        hamiltonian,
        trial,
        walker_opts={},
        index=0,
        nprop_tot=None,
        nbp=None,
    ):
        Walker.__init__(
            self,
            system,
            hamiltonian,
            trial,
            walker_opts=walker_opts,
            index=index,
            nprop_tot=nprop_tot,
            nbp=nbp,
        )
        self.name = "SingleDetWalker"
        self.inv_ovlp = [0.0, 0.0]

        self.phi_boson = None

        if system.name == "HubbardHolstein":
            shift = trial.shift.copy()
            self.X = numpy.real(shift).copy()

            tmptrial = HarmonicOscillator(m=system.m, w=system.w0, order=0, shift=shift)

            sqtau = numpy.sqrt(0.005)
            nstep = 250
            # simple VMC
            for istep in range(nstep):
                chi = numpy.random.randn(system.nbasis)  # Random move
                # propose a move
                posnew = self.X + sqtau * chi
                # calculate Metropolis-Rosenbluth-Teller acceptance probability
                wfold = tmptrial.value(self.X)
                wfnew = tmptrial.value(posnew)
                pacc = wfnew * wfnew / (wfold * wfold)
                # get indices of accepted moves
                u = numpy.random.random(1)
                if u < pacc:
                    self.X = posnew.copy()
            self.Lap = tmptrial.laplacian(self.X)
            self.phi_boson = tmptrial.value(self.X)

        self.inverse_overlap(trial)
        self.ot = self.calc_overlap(trial)
        self.le_oratio = 1.0
        self.ovlp = self.ot

        self.G = numpy.zeros(
            shape=(2, hamiltonian.nbasis, hamiltonian.nbasis), dtype=numpy.complex128
        )

        self.Ghalf = [
            numpy.zeros(shape=(system.nup, hamiltonian.nbasis), dtype=numpy.complex128),
            numpy.zeros(
                shape=(system.ndown, hamiltonian.nbasis), dtype=numpy.complex128
            ),
        ]
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

        self.inv_ovlp[0] = scipy.linalg.inv(
            (trial.psi[:, :nup].conj()).T.dot(self.phi[:, :nup])
        )

        self.inv_ovlp[1] = numpy.zeros(self.inv_ovlp[0].shape)
        if ndown > 0:
            self.inv_ovlp[1] = scipy.linalg.inv(
                (trial.psi[:, nup:].conj()).T.dot(self.phi[:, nup:])
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

        self.inv_ovlp[0] = sherman_morrison(
            self.inv_ovlp[0], trial.psi[i, :nup].conj(), vtup
        )
        self.inv_ovlp[1] = sherman_morrison(
            self.inv_ovlp[1], trial.psi[i, nup:].conj(), vtdown
        )

    def calc_otrial(self, trial):
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
        sign_a, logdet_a = numpy.linalg.slogdet(self.inv_ovlp[0])
        nbeta = self.ndown
        sign_b, logdet_b = 1.0, 0.0
        if nbeta > 0:
            sign_b, logdet_b = numpy.linalg.slogdet(self.inv_ovlp[1])
        det = sign_a * sign_b * numpy.exp(logdet_a + logdet_b - self.log_shift)

        ot = 1.0 / det

        if self.phi_boson is not None:
            boson_trial = HarmonicOscillator(
                m=trial.m, w=trial.w0, order=0, shift=trial.shift
            )
            self.phi_boson = boson_trial.value(self.X)
            ot *= self.phi_boson

        return ot

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
        na = self.nup
        Oalpha = numpy.dot(trial.psi[:, :na].conj().T, self.phi[:, :na])
        sign_a, logdet_a = numpy.linalg.slogdet(Oalpha)
        logdet_b, sign_b = 0.0, 1.0
        if self.ndown > 0:
            Obeta = numpy.dot(trial.psi[:, na:].conj().T, self.phi[:, na:])
            sign_b, logdet_b = numpy.linalg.slogdet(Obeta)

        ot = sign_a * sign_b * numpy.exp(logdet_a + logdet_b - self.log_shift)

        if self.phi_boson is not None:
            boson_trial = HarmonicOscillator(
                m=trial.m, w=trial.w0, order=0, shift=trial.shift
            )
            self.phi_boson = boson_trial.value(self.X)
            ot *= self.phi_boson

        return ot

    def update_overlap(self, probs, xi, coeffs):
        """Update overlap.

        Parameters
        ----------
        probs : :class:`numpy.ndarray`
            Probabilities for chosing particular field configuration.
        xi : int
            Chosen field configuration.
        coeffs : :class:`numpy.ndarray`
            Trial wavefunction coefficients. For interface consistency.
        """
        self.ot = 2 * self.ot * probs[xi]

    def reortho(self, trial):
        """reorthogonalise walker.

        parameters
        ----------
        trial : object
            trial wavefunction object. for interface consistency.
        """
        nup = self.nup
        ndown = self.ndown
        (self.phi[:, :nup], Rup) = scipy.linalg.qr(self.phi[:, :nup], mode="economic")
        Rdown = numpy.zeros(Rup.shape)
        if ndown > 0:
            (self.phi[:, nup:], Rdn) = scipy.linalg.qr(
                self.phi[:, nup:], mode="economic"
            )
        # TODO: FDM This isn't really necessary, the absolute value of the
        # weight is used for population control so this shouldn't matter.
        # I think this is a legacy thing.
        # Wanted detR factors to remain positive, dump the sign in orbitals.
        Rup_diag = numpy.diag(Rup)
        signs_up = numpy.sign(Rup_diag)
        if ndown > 0:
            Rdn_diag = numpy.diag(Rdn)
            signs_dn = numpy.sign(Rdn_diag)
        self.phi[:, :nup] = numpy.dot(self.phi[:, :nup], numpy.diag(signs_up))
        if ndown > 0:
            self.phi[:, nup:] = numpy.dot(self.phi[:, nup:], numpy.diag(signs_dn))
        # include overlap factor
        # det(R) = \prod_ii R_ii
        # det(R) = exp(log(det(R))) = exp((sum_i log R_ii) - C)
        # C factor included to avoid over/underflow
        log_det = numpy.sum(numpy.log(numpy.abs(Rup_diag)))
        if ndown > 0:
            log_det += numpy.sum(numpy.log(numpy.abs(Rdn_diag)))
        detR = numpy.exp(log_det - self.detR_shift)
        self.log_detR += numpy.log(detR)
        self.detR = detR
        self.ot = self.ot / detR
        self.ovlp = self.ot
        return detR

    def reortho_excite(self, trial):
        """reorthogonalise walker.

        parameters
        ----------
        trial : object
            trial wavefunction object. for interface consistency.
        """
        nup = self.nup
        ndown = self.ndown
        # print (self.phi[:,:self.nup])
        buff = numpy.copy(self.trial_buff).astype(trial.psi.dtype)
        buff[:, : self.ia[0]] = self.phi[:, : self.ia[0]]
        buff[:, self.ia[0] + 1 : self.nup] = self.phi[:, self.ia[0] : self.nup - 1]
        buff[:, -1] = self.phi[:, self.nup - 1]
        (buff, Rup) = scipy.linalg.qr(buff, mode="economic")
        Rdown = numpy.zeros(Rup.shape)
        if ndown > 0:
            (self.phi[:, nup:], Rdown) = scipy.linalg.qr(
                self.phi[:, nup:], mode="economic"
            )
        signs_up = numpy.diag(numpy.sign(numpy.diag(Rup)))
        if ndown > 0:
            signs_down = numpy.diag(numpy.sign(numpy.diag(Rdown)))
        buff = buff.dot(signs_up)
        self.phi[:, : self.ia[0]] = numpy.copy(buff[:, : self.ia[0]])
        self.phi[:, self.ia[0] : self.nup - 1] = numpy.copy(
            buff[:, self.ia[0] + 1 : self.nup]
        )
        self.phi[:, self.nup - 1] = numpy.copy(buff[:, -1])
        if ndown > 0:
            self.phi[:, nup:] = self.phi[:, nup:].dot(signs_down)
        drup = scipy.linalg.det(signs_up.dot(Rup))
        drdn = 1.0
        if ndown > 0:
            drdn = scipy.linalg.det(signs_down.dot(Rdown))
        detR = drup * drdn
        # This only affects free projection
        self.ot = self.ot / detR
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

        ovlp = numpy.dot(self.phi[:, :nup].T, trial.psi[:, :nup].conj())
        self.Ghalf[0] = numpy.dot(scipy.linalg.inv(ovlp), self.phi[:, :nup].T)
        self.G[0] = numpy.dot(trial.psi[:, :nup].conj(), self.Ghalf[0])
        sign_a, log_ovlp_a = numpy.linalg.slogdet(ovlp)
        sign_b, log_ovlp_b = 1.0, 0.0
        if ndown > 0:
            ovlp = numpy.dot(self.phi[:, nup:].T, trial.psi[:, nup:].conj())
            sign_b, log_ovlp_b = numpy.linalg.slogdet(ovlp)
            self.Ghalf[1] = numpy.dot(scipy.linalg.inv(ovlp), self.phi[:, nup:].T)
            self.G[1] = numpy.dot(trial.psi[:, nup:].conj(), self.Ghalf[1])
        det = sign_a * sign_b * numpy.exp(log_ovlp_a + log_ovlp_b - self.log_shift)
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
        self.Ghalf[0] = self.phi[:, :nup].dot(self.inv_ovlp[0])
        self.Ghalf[1] = numpy.zeros(self.Ghalf[0].shape)
        if ndown > 0:
            self.Ghalf[1] = self.phi[:, nup:].dot(self.inv_ovlp[1])

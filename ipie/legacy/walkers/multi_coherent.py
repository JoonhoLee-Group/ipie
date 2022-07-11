import copy
import random

import numpy
import scipy.linalg

from ipie.legacy.estimators.local_energy import local_energy_multi_det_hh
from ipie.legacy.trial_wavefunction.harmonic_oscillator import (
    HarmonicOscillator, HarmonicOscillatorMomentum)
from ipie.legacy.walkers.stack import FieldConfig, PropagatorStack
from ipie.utils.linalg import sherman_morrison
from ipie.utils.misc import get_numeric_names


class MultiCoherentWalker(object):
    """Multi-Vibronic style walker.

    Parameters
    ----------
    weight : int
        Walker weight.
    system : object
        System object.
    trial : object
        Trial wavefunction object.
    index : int
        Element of trial wavefunction to initalise walker to.
    weights : string
        Initialise weights to zeros or ones.
    wfn0 : string
        Initial wavefunction.
    """

    def __init__(
        self,
        system,
        trial,
        walker_opts,
        index=0,
        nprop_tot=None,
        nbp=None,
        weights="zeros",
        verbose=False,
    ):
        if verbose:
            print("# Setting up MultiCoherentWalker object.")
        self.weight = walker_opts.get("weight", 1.0)
        self.unscaled_weight = self.weight
        self.alive = 1
        self.phase = 1 + 0j
        self.nup = system.nup
        self.E_L = 0.0
        self.phi = copy.deepcopy(trial.init)
        self.nperms = trial.nperms

        dtype = numpy.complex128
        # This stores an array of overlap matrices with the various elements of
        # the trial wavefunction.
        self.inv_ovlp = [
            numpy.zeros(shape=(self.nperms, system.nup, system.nup), dtype=dtype),
            numpy.zeros(shape=(self.nperms, system.ndown, system.ndown), dtype=dtype),
        ]

        # TODO: RENAME to something less like weight
        if weights == "zeros":
            self.weights = numpy.zeros(self.nperms, dtype=dtype)
        else:
            self.weights = numpy.ones(self.nperms, dtype=dtype)

        self.phi_boson = numpy.ones(self.nperms, dtype=dtype)
        self.ots = numpy.zeros(self.nperms, dtype=dtype)

        if system.name == "HubbardHolstein":
            if len(trial.psi.shape) == 3:
                idx = random.choice(numpy.arange(trial.nperms))
                shift = trial.shift[idx, :].copy()
            else:
                perm = random.choice(trial.perms)
                shift = trial.shift[perm].copy()
            if len(trial.psi.shape) == 3:
                shift = trial.shift[0, :].copy()
            else:
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
            self.Lapi = numpy.zeros(shape=(self.nperms, system.nbasis), dtype=dtype)

            if len(trial.psi.shape) == 3:
                for i in range(trial.nperms):
                    shift = trial.shift[i, :].copy()
                    boson_trial = HarmonicOscillator(
                        m=system.m, w=system.w0, order=0, shift=shift
                    )
                    self.Lapi[i] = boson_trial.laplacian(self.X)
            else:
                shift0 = trial.shift.copy()
                for i, perm in enumerate(trial.perms):
                    shift = shift0[perm].copy()
                    boson_trial = HarmonicOscillator(
                        m=system.m, w=system.w0, order=0, shift=shift
                    )
                    self.Lapi[i] = boson_trial.laplacian(self.X)

        # Compute initial overlap. Avoids issues with singular matrices for
        # PHMSD.
        self.ot = self.overlap_direct(trial)
        self.le_oratio = 1.0
        # Hubbard specific functionality
        self.R = numpy.zeros(shape=(trial.nperms, 2), dtype=self.phi.dtype)
        # TODO: fix name.
        self.ovlp = self.ot
        self.hybrid_energy = 0.0
        if verbose:
            print(
                "# Initial overlap of walker with trial wavefunction: {:13.8e}".format(
                    self.ot.real
                )
            )
        # Green's functions for various elements of the trial wavefunction.
        self.Gi = numpy.zeros(
            shape=(self.nperms, 2, system.nbasis, system.nbasis), dtype=dtype
        )
        # Actual green's function contracted over determinant index in Gi above.
        # i.e., <psi_T|c_i^d c_j|phi>
        self.G = numpy.zeros(shape=(2, system.nbasis, system.nbasis), dtype=dtype)
        # Contains overlaps of the current walker with the trial wavefunction.
        self.greens_function(trial)
        self.nb = system.nbasis
        self.nup = system.nup
        self.ndown = system.ndown
        # Historic wavefunction for back propagation.
        self.phi_old = copy.deepcopy(self.phi)
        # Historic wavefunction for ITCF.
        self.phi_init = copy.deepcopy(self.phi)
        # Historic wavefunction for ITCF.
        # self.phi_bp = copy.deepcopy(trial.psi)

        if nbp is not None:
            self.field_configs = FieldConfig(
                system.nfields, nprop_tot, nbp, numpy.complex128
            )
        else:
            self.field_configs = None

        self.stack = None

        self.buff_names, self.buff_size = get_numeric_names(self.__dict__)

    def overlap_direct(self, trial):
        nup = self.nup

        if len(trial.psi.shape) == 3:
            for i in range(trial.nperms):
                det = trial.psi[i, :, :].copy()
                shift = trial.shift[i, :].copy()

                trial.boson_trial.update_shift(shift)
                self.phi_boson[i] = trial.boson_trial.value(self.X)

                Oup = numpy.dot(det[:, :nup].conj().T, self.phi[:, :nup])
                Odn = numpy.dot(det[:, nup:].conj().T, self.phi[:, nup:])

                self.ots[i] = scipy.linalg.det(Oup) * scipy.linalg.det(Odn)
                if abs(self.ots[i]) > 1e-16:
                    self.inv_ovlp[0][i] = scipy.linalg.inv(Oup)
                    self.inv_ovlp[1][i] = scipy.linalg.inv(Odn)
                self.weights[i] = (
                    trial.coeffs[i].conj() * self.ots[i] * self.phi_boson[i]
                )
        else:
            shift0 = trial.shift.copy()
            psi0 = trial.psi.copy()

            for i, perm in enumerate(trial.perms):

                det = psi0[perm, :].copy()
                shift = shift0[perm].copy()

                trial.boson_trial.update_shift(shift)
                self.phi_boson[i] = trial.boson_trial.value(self.X)

                Oup = numpy.dot(det[:, :nup].conj().T, self.phi[:, :nup])
                Odn = numpy.dot(det[:, nup:].conj().T, self.phi[:, nup:])

                self.ots[i] = scipy.linalg.det(Oup) * scipy.linalg.det(Odn)
                if abs(self.ots[i]) > 1e-16:
                    self.inv_ovlp[0][i] = scipy.linalg.inv(Oup)
                    self.inv_ovlp[1][i] = scipy.linalg.inv(Odn)
                self.weights[i] = (
                    trial.coeffs[i].conj() * self.ots[i] * self.phi_boson[i]
                )

            trial.boson_trial.update_shift(shift0)
            trial.psi = psi0.copy()

        return sum(self.weights)

    def inverse_overlap(self, trial):
        """Compute inverse overlap matrix from scratch.

        Parameters
        ----------
        trial : :class:`numpy.ndarray`
            Trial wavefunction.
        """
        nup = self.nup

        if len(trial.psi.shape) == 3:
            for indx in range(trial.nperms):
                t = trial.psi[indx, :, :].copy()
                shift = trial.shift[indx, :].copy()
                trial.boson_trial.update_shift(shift)
                self.phi_boson[indx] = trial.boson_trial.value(self.X)

                Oup = numpy.dot(t[:, :nup].conj().T, self.phi[:, :nup])
                self.inv_ovlp[0][indx, :, :] = scipy.linalg.inv(Oup)
                Odn = numpy.dot(t[:, nup:].conj().T, self.phi[:, nup:])
                self.inv_ovlp[1][indx, :, :] = scipy.linalg.inv(Odn)

        else:
            psi0 = trial.psi.copy()
            shift0 = trial.shift.copy()

            for indx, perm in enumerate(trial.perms):
                t = psi0[perm, :].copy()
                shift = shift0[perm].copy()
                trial.boson_trial.update_shift(shift)
                self.phi_boson[indx] = trial.boson_trial.value(self.X)

                Oup = numpy.dot(t[:, :nup].conj().T, self.phi[:, :nup])
                self.inv_ovlp[0][indx, :, :] = scipy.linalg.inv(Oup)
                Odn = numpy.dot(t[:, nup:].conj().T, self.phi[:, nup:])
                self.inv_ovlp[1][indx, :, :] = scipy.linalg.inv(Odn)

            trial.psi = psi0.copy()
            trial.boson_trial.update_shift(shift0)

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

        if len(trial.psi.shape) == 3:
            for ix in range(trial.nperms):
                psi = trial.psi[ix, :, :].copy()
                if nup > 0:
                    self.inv_ovlp[0][ix] = sherman_morrison(
                        self.inv_ovlp[0][ix], psi[i, :nup].conj(), vtup
                    )
                if ndown > 0:
                    self.inv_ovlp[1][ix] = sherman_morrison(
                        self.inv_ovlp[1][ix], psi[i, nup:].conj(), vtdown
                    )
        else:
            for ix, perm in enumerate(trial.perms):
                psi = trial.psi[perm, :].copy()
                if nup > 0:
                    self.inv_ovlp[0][ix] = sherman_morrison(
                        self.inv_ovlp[0][ix], psi[i, :nup].conj(), vtup
                    )
                if ndown > 0:
                    self.inv_ovlp[1][ix] = sherman_morrison(
                        self.inv_ovlp[1][ix], psi[i, nup:].conj(), vtdown
                    )

    def calc_otrial(self, trial):
        """Caculate overlap with trial wavefunction.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.

        Returns
        -------
        ovlp : float / complex
            Overlap.
        """
        if len(trial.psi.shape) == 3:
            for ix in range(trial.nperms):
                det_O_up = 1.0 / scipy.linalg.det(self.inv_ovlp[0][ix])
                det_O_dn = 1.0 / scipy.linalg.det(self.inv_ovlp[1][ix])
                self.ots[ix] = det_O_up * det_O_dn

                shift = trial.shift[ix, :].copy()
                trial.boson_trial.update_shift(shift)

                self.phi_boson[ix] = trial.boson_trial.value(self.X)
                self.weights[ix] = (
                    trial.coeffs[ix].conj() * self.ots[ix] * self.phi_boson[ix]
                )
        else:
            shift0 = trial.shift.copy()
            for ix, perm in enumerate(trial.perms):
                det_O_up = 1.0 / scipy.linalg.det(self.inv_ovlp[0][ix])
                det_O_dn = 1.0 / scipy.linalg.det(self.inv_ovlp[1][ix])
                self.ots[ix] = det_O_up * det_O_dn

                shift = shift0[perm].copy()
                trial.boson_trial.update_shift(shift)

                self.phi_boson[ix] = trial.boson_trial.value(self.X)
                self.weights[ix] = (
                    trial.coeffs[ix].conj() * self.ots[ix] * self.phi_boson[ix]
                )
            trial.boson_trial.update_shift(shift0)
        return sum(self.weights)

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
            (self.phi[:, nup:], Rdown) = scipy.linalg.qr(
                self.phi[:, nup:], mode="economic"
            )
        signs_up = numpy.diag(numpy.sign(numpy.diag(Rup)))
        if ndown > 0:
            signs_down = numpy.diag(numpy.sign(numpy.diag(Rdown)))
        self.phi[:, :nup] = self.phi[:, :nup].dot(signs_up)
        if ndown > 0:
            self.phi[:, nup:] = self.phi[:, nup:].dot(signs_down)
        drup = scipy.linalg.det(signs_up.dot(Rup))
        drdn = 1.0
        if ndown > 0:
            drdn = scipy.linalg.det(signs_down.dot(Rdown))
        detR = drup * drdn
        self.ot = self.ot / detR
        return detR

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
        # Update each component's overlap and the total overlap.
        # The trial wavefunctions coeficients should be included in ots?
        self.ots = self.R[:, xi] * self.ots
        self.weights = coeffs * self.ots * self.phi_boson
        self.ot = 2.0 * self.ot * probs[xi]

    def greens_function(self, trial):
        """Compute walker's green's function.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.
        """
        nup = self.nup

        if len(trial.psi.shape) == 3:

            for ix in range(trial.nperms):
                t = trial.psi[ix, :, :].copy()

                # construct "local" green's functions for each component of psi_T
                self.Gi[ix, 0, :, :] = (
                    self.phi[:, :nup].dot(self.inv_ovlp[0][ix]).dot(t[:, :nup].conj().T)
                ).T
                self.Gi[ix, 1, :, :] = (
                    self.phi[:, nup:].dot(self.inv_ovlp[1][ix]).dot(t[:, nup:].conj().T)
                ).T
            denom = sum(self.weights)
            self.G = numpy.einsum("i,isjk->sjk", self.weights, self.Gi) / denom

        else:

            psi0 = trial.psi.copy()

            for ix, perm in enumerate(trial.perms):
                t = psi0[perm, :].copy()

                # construct "local" green's functions for each component of psi_T
                self.Gi[ix, 0, :, :] = (
                    self.phi[:, :nup].dot(self.inv_ovlp[0][ix]).dot(t[:, :nup].conj().T)
                ).T
                self.Gi[ix, 1, :, :] = (
                    self.phi[:, nup:].dot(self.inv_ovlp[1][ix]).dot(t[:, nup:].conj().T)
                ).T
            trial.psi = psi0.copy()
            denom = sum(self.weights)
            self.G = numpy.einsum("i,isjk->sjk", self.weights, self.Gi) / denom

    def local_energy(self, system, two_rdm=None, rchol=None, eri=None, UVT=None):
        """Compute walkers local energy

        Parameters
        ----------
        system : object
            System object.

        Returns
        -------
        (E, T, V) : tuple
            Mixed estimates for walker's energy components.
        """
        assert system.name == "HubbardHolstein"

        return local_energy_multi_det_hh(
            system, self.Gi, self.weights, self.X, self.Lapi, two_rdm=two_rdm
        )

    def contract_one_body(self, ints, trial):
        numer = 0.0
        denom = 0.0
        for i, Gi in enumerate(self.Gi):
            ofac = trial.coeffs[i].conj() * self.ots[i]
            numer += ofac * numpy.dot((Gi[0] + Gi[1]).ravel(), ints.ravel())
            denom += ofac
        return numer / denom

    def get_buffer(self):
        """Get walker buffer for MPI communication

        Returns
        -------
        buff : dict
            Relevant walker information for population control.
        """
        s = 0
        buff = numpy.zeros(self.buff_size, dtype=numpy.complex128)
        for d in self.buff_names:
            data = self.__dict__[d]
            if isinstance(data, (numpy.ndarray)):
                buff[s : s + data.size] = data.ravel()
                s += data.size
            elif isinstance(data, list):
                for l in data:
                    if isinstance(l, (numpy.ndarray)):
                        buff[s : s + l.size] = l.ravel()
                        s += l.size
                    elif isinstance(l, (int, float, complex)):
                        buff[s : s + 1] = l
                        s += 1
            else:
                buff[s : s + 1] = data
                s += 1
        if self.field_configs is not None:
            stack_buff = self.field_configs.get_buffer()
            return numpy.concatenate((buff, stack_buff))
        elif self.stack is not None:
            stack_buff = self.stack.get_buffer()
            return numpy.concatenate((buff, stack_buff))
        else:
            return buff

    def set_buffer(self, buff):
        """Set walker buffer following MPI communication

        Parameters
        -------
        buff : dict
            Relevant walker information for population control.
        """
        s = 0
        for d in self.buff_names:
            data = self.__dict__[d]
            if isinstance(data, numpy.ndarray):
                self.__dict__[d] = buff[s : s + data.size].reshape(data.shape).copy()
                s += data.size
            elif isinstance(data, list):
                for ix, l in enumerate(data):
                    if isinstance(l, (numpy.ndarray)):
                        self.__dict__[d][ix] = (
                            buff[s : s + l.size].reshape(l.shape).copy()
                        )
                        s += l.size
                    elif isinstance(l, (int, float, complex)):
                        self.__dict__[d][ix] = buff[s]
                        s += 1
            else:
                if isinstance(self.__dict__[d], int):
                    self.__dict__[d] = int(buff[s].real)
                elif isinstance(self.__dict__[d], float):
                    self.__dict__[d] = buff[s].real
                else:
                    self.__dict__[d] = buff[s]
                s += 1
        if self.field_configs is not None:
            self.field_configs.set_buffer(buff[self.buff_size :])
        if self.stack is not None:
            self.stack.set_buffer(buff[self.buff_size :])

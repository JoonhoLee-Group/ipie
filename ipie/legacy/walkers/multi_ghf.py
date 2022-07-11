import copy

import numpy

from ipie.legacy.estimators.hubbard import local_energy_hubbard_ghf
from ipie.legacy.trial_wavefunction.free_electron import FreeElectron
from ipie.utils.io import read_fortran_complex_numbers


class MultiGHFWalker(object):
    """Multi-GHF style walker.

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
        self, walker_opts, system, trial, index=0, weights="zeros", wfn0="init"
    ):
        self.weight = walker_opts.get("weight", 1)
        self.alive = 1
        # Initialise to a particular free electron slater determinant rather
        # than GHF. Can actually initialise to GHF by passing single GHF with
        # initial_wavefunction. The distinction is really for back propagation
        # when we may want to use the full expansion.
        self.nup = system.nup
        if wfn0 == "init":
            # Initialise walker with single determinant.
            if trial.initial_wavefunction != "free_electron":
                orbs = read_fortran_complex_numbers(trial.read_init)
                self.phi = orbs.reshape((2 * system.nbasis, system.ne), order="F")
            else:
                self.phi = numpy.zeros(
                    shape=(2 * system.nbasis, system.ne), dtype=trial.psi.dtype
                )
                tmp = FreeElectron(system, trial.psi.dtype == complex, {})
                self.phi[: system.nbasis, : system.nup] = tmp.psi[:, : system.nup]
                self.phi[system.nbasis :, system.nup :] = tmp.psi[:, system.nup :]
        else:
            self.phi = copy.deepcopy(trial.psi)
        # This stores an array of overlap matrices with the various elements of
        # the trial wavefunction.
        self.inv_ovlp = numpy.zeros(
            shape=(trial.ndets, system.ne, system.ne), dtype=self.phi.dtype
        )
        if weights == "zeros":
            self.weights = numpy.zeros(trial.ndets, dtype=trial.psi.dtype)
        else:
            self.weights = numpy.ones(trial.ndets, dtype=trial.psi.dtype)
        if wfn0 != "GHF":
            self.inverse_overlap(trial.psi)
        # Green's functions for various elements of the trial wavefunction.
        self.Gi = numpy.zeros(
            shape=(trial.ndets, 2 * system.nbasis, 2 * system.nbasis),
            dtype=self.phi.dtype,
        )
        # Should be nfields per basis * ndets.
        # Todo: update this for the continuous HS trasnform case.
        self.R = numpy.zeros(shape=(trial.ndets, 2), dtype=self.phi.dtype)
        # Actual green's function contracted over determinant index in Gi above.
        # i.e., <psi_T|c_i^d c_j|phi>
        self.G = numpy.zeros(
            shape=(2 * system.nbasis, 2 * system.nbasis), dtype=self.phi.dtype
        )
        self.ots = numpy.zeros(trial.ndets, dtype=self.phi.dtype)
        self.le_oratio = 1.0
        # Contains overlaps of the current walker with the trial wavefunction.
        if wfn0 != "GHF":
            self.ot = self.calc_otrial(trial)
            self.greens_function(trial)
            self.E_L = ipie.estimators.local_energy_ghf(
                system, self.Gi, self.weights, sum(self.weights)
            )[0].real
        self.nb = system.nbasis
        # Historic wavefunction for back propagation.
        self.phi_old = copy.deepcopy(self.phi)
        # Historic wavefunction for ITCF.
        self.phi_init = copy.deepcopy(self.phi)
        # Historic wavefunction for ITCF.
        self.phi_bp = copy.deepcopy(trial.psi)

    def inverse_overlap(self, trial):
        """Compute inverse overlap matrix from scratch.

        Parameters
        ----------
        trial : :class:`numpy.ndarray`
            Trial wavefunction.
        """
        nup = self.nup
        for (indx, t) in enumerate(trial):
            self.inv_ovlp[indx, :, :] = scipy.linalg.inv((t.conj()).T.dot(self.phi))

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
        # The trial wavefunctions coefficients should be complex conjugated
        # on initialisation!
        for (ix, inv) in enumerate(self.inv_ovlp):
            self.ots[ix] = 1.0 / scipy.linalg.det(inv)
            self.weights[ix] = trial.coeffs[ix] * self.ots[ix]
        return sum(self.weights)

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
        self.weights = coeffs * self.ots
        self.ot = 2.0 * self.ot * probs[xi]

    def reortho(self, trial):
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
        nup = self.nup
        # We assume that our walker is still block diagonal in the spin basis.
        (self.phi[: self.nb, :nup], Rup) = scipy.linalg.qr(
            self.phi[: self.nb, :nup], mode="economic"
        )
        (self.phi[self.nb :, nup:], Rdown) = scipy.linalg.qr(
            self.phi[self.nb :, nup:], mode="economic"
        )
        # Enforce a positive diagonal for the overlap.
        signs_up = numpy.diag(numpy.sign(numpy.diag(Rup)))
        signs_down = numpy.diag(numpy.sign(numpy.diag(Rdown)))
        self.phi[: self.nb, :nup] = self.phi[: self.nb, :nup].dot(signs_up)
        self.phi[self.nb :, nup:] = self.phi[self.nb :, nup:].dot(signs_down)
        # Todo: R is upper triangular.
        drup = scipy.linalg.det(signs_up.dot(Rup))
        drdn = scipy.linalg.det(signs_down.dot(Rdown))
        detR = drup * drdn
        self.inverse_overlap(trial.psi)
        self.ot = self.calc_otrial(trial)

    def greens_function(self, trial):
        """Compute walker's green's function.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.
        """
        nup = self.nup
        for (ix, t) in enumerate(trial.psi):
            # construct "local" green's functions for each component of psi_T
            self.Gi[ix, :, :] = (self.phi.dot(self.inv_ovlp[ix]).dot(t.conj().T)).T
        denom = sum(self.weights)
        self.G = numpy.einsum("i,ijk->jk", self.weights, self.Gi) / denom

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
        for (indx, t) in enumerate(trial.psi):
            self.inv_ovlp[indx, :, :] = scipy.linalg.inv((t.conj()).T.dot(self.phi))

    def local_energy(self, system, two_rdm=None):
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
        return ipie.estimators.local_energy_ghf(system, self.Gi, self.weights, self.ot)

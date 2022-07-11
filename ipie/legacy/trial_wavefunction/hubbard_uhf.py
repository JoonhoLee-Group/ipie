import copy
import time

import numpy

from ipie.legacy.estimators.greens_function import gab
from ipie.legacy.estimators.local_energy import local_energy
from ipie.legacy.hamiltonians.hubbard import decode_basis
from ipie.utils.io import get_input_value
from ipie.utils.linalg import diagonalise_sorted


class HubbardUHF(object):
    r"""HubbardUHF trial wavefunction.

    Search for HubbardUHF trial wavefunction by self consistenly solving the mean field
    Hamiltonian:

        .. math::
            H^{\sigma} = \sum_{\langle ij\rangle} \left(
                    c^{\dagger}_{i\sigma}c_{j\sigma} + h.c.\right) +
                    U_{\mathrm{eff}} \sum_i \hat{n}_{i\sigma}\langle\hat{n}_{i\bar{\sigma}}\rangle -
                    \frac{1}{2} U_{\mathrm{eff}} \sum_i \langle\hat{n}_{i\sigma}\rangle
                    \langle\hat{n}_{i\bar{\sigma}}\rangle.

    See [Xu11]_ for more details.

    .. Warning::
        This is for the Hubbard model only

    .. todo:: We should generalise in the future perhaps.

    Parameters
    ----------
    system : :class:`pie.systems.hubbard.Hubbard` object
        System parameters.
    cplx : bool
        True if the trial wavefunction etc is complex.
    trial : dict
        Trial wavefunction input options.

    Attributes
    ----------
    psi : :class:`numpy.ndarray`
        Trial wavefunction.
    eigs : :class:`numpy.array`
        One-electron eigenvalues.
    emin : float
        Ground state mean field total energy of trial wavefunction.
    """

    def __init__(self, system, hamiltonian, trial={}, verbose=0):
        assert "Hubbard" in hamiltonian.name
        if verbose:
            print("# Constructing HubbardUHF trial wavefunction")
        self.verbose = verbose
        init_time = time.time()
        self.name = "HubbardUHF"
        self.type = "HubbardUHF"
        self.initial_wavefunction = trial.get("initial_wavefunction", "trial")
        self.trial_type = complex
        # Unpack input options.
        self.ninitial = get_input_value(trial, "ninitial", default=10, verbose=verbose)
        self.nconv = get_input_value(trial, "nconv", default=5000, verbose=verbose)
        self.ueff = get_input_value(trial, "ueff", default=0.4, verbose=verbose)
        self.deps = get_input_value(trial, "deps", default=1e-8, verbose=verbose)
        self.alpha = get_input_value(trial, "alpha", default=0.5, verbose=verbose)
        # For interface compatability
        self.Ghalf = None
        self.coeffs = 1.0
        self.type = "HubbardUHF"
        self.ndets = 1
        self.initial_guess = trial.get("initial", "random")
        if self.initial_guess == "random":
            if self.verbose:
                print("# Solving HubbardUHF equations.")
            (self.psi, self.eigs, self.emin, self.error, self.nav) = self.find_uhf_wfn(
                system,
                hamiltonian,
                self.ueff,
                self.ninitial,
                self.nconv,
                self.alpha,
                self.deps,
                verbose,
            )
            if self.error:
                warnings.warn("Error in constructing trial wavefunction. Exiting")
                sys.exit()
        elif self.initial_guess == "checkerboard":
            if self.verbose:
                print("# Using checkerboard breakup.")
            self.psi, unused = self.checkerboard(
                hamiltonian.nbasis, system.nup, system.ndown
            )
        Gup = gab(self.psi[:, : system.nup], self.psi[:, : system.nup]).T
        if system.ndown > 0:
            Gdown = gab(self.psi[:, system.nup :], self.psi[:, system.nup :]).T
        else:
            Gdown = numpy.zeros_like(Gup)
        self.le_oratio = 1.0
        self.G = numpy.array([Gup, Gdown])
        self.etrial = local_energy(system, hamiltonian, self, self)[0].real
        self.bp_wfn = trial.get("bp_wfn", None)
        self.initialisation_time = time.time() - init_time
        self.init = self.psi
        self._mem_required = 0.0
        self._rchol = None

    def find_uhf_wfn(
        self, system, hamiltonian, ueff, ninit, nit_max, alpha, deps=1e-8, verbose=0
    ):
        emin = 0
        # JOONHO superhacky way. it should be fixed.
        uold = hamiltonian.U
        hamiltonian.U = ueff
        minima = []  # Local minima
        nup = system.nup
        # Search over different random starting points.
        for attempt in range(0, ninit):
            # Set up initial (random) guess for the density.
            (self.trial, eold) = self.initialise(
                hamiltonian.nbasis, system.nup, system.ndown
            )
            niup = self.density(self.trial[:, :nup])
            nidown = self.density(self.trial[:, nup:])
            niup_old = self.density(self.trial[:, :nup])
            nidown_old = self.density(self.trial[:, nup:])
            for it in range(0, nit_max):
                (niup, nidown, e_up, e_down) = self.diagonalise_mean_field(
                    system, hamiltonian, ueff, niup, nidown
                )
                # Construct Green's function to compute the energy.
                Gup = gab(self.trial[:, :nup], self.trial[:, :nup]).T
                if system.ndown > 0:
                    Gdown = gab(self.trial[:, nup:], self.trial[:, nup:]).T
                else:
                    Gdown = numpy.zeros((hamiltonian.nbasis, hamiltonian.nbasis))

                self.G = numpy.array([Gup, Gdown], dtype=Gup.dtype)
                enew = local_energy(system, hamiltonian, self, self)[0].real

                if verbose > 1:
                    print("# %d %f %f" % (it, enew, eold))
                sc = self.self_consistant(
                    enew, eold, niup, niup_old, nidown, nidown_old, it, deps, verbose
                )
                if sc:
                    # Global minimum search.
                    if attempt == 0:
                        minima.append(enew)
                        psi_accept = copy.deepcopy(self.trial)
                        e_accept = numpy.append(e_up, e_down)
                    elif all(numpy.array(minima) - enew > deps):
                        minima.append(enew)
                        psi_accept = copy.deepcopy(self.trial)
                        e_accept = numpy.append(e_up, e_down)
                    break
                else:
                    mixup = self.mix_density(niup, niup_old, alpha)
                    mixdown = self.mix_density(nidown, nidown_old, alpha)
                    niup_old = niup
                    nidown_old = nidown
                    niup = mixup
                    nidown = mixdown
                    eold = enew
            if verbose > 1:
                print(
                    "# SCF cycle: {:3d}. After {:4d} steps the minimum HubbardUHF"
                    " energy found is: {: 8f}".format(attempt, it, eold)
                )

        hamiltonian.U = uold
        if verbose:
            print("# Minimum energy found: {: 8f}".format(min(minima)))
            nocca = system.nup
            noccb = system.ndown
            MS = numpy.abs(nocca - noccb) / 2.0
            S2exact = MS * (MS + 1.0)
            Sij = psi_accept[:, :nocca].T.dot(psi_accept[:, nocca:])
            S2 = S2exact + min(nocca, noccb) - numpy.sum(numpy.abs(Sij * Sij).ravel())
            print("# <S^2> = {: 3f}".format(S2))
        try:
            return (psi_accept, e_accept, min(minima), False, [niup, nidown])
        except UnboundLocalError:
            warnings.warn(
                "Warning: No HubbardUHF wavefunction found."
                "Delta E: %f" % (enew - emin)
            )
            return (trial, numpy.append(e_up, e_down), None, True, None)

    def initialise(self, nbasis, nup, ndown):
        (e_up, ev_up) = self.random_starting_point(nbasis)
        (e_down, ev_down) = self.random_starting_point(nbasis)

        trial = numpy.zeros(shape=(nbasis, nup + ndown), dtype=numpy.complex128)
        trial[:, :nup] = ev_up[:, :nup]
        trial[:, nup:] = ev_down[:, :ndown]
        eold = sum(e_up[:nup]) + sum(e_down[:ndown])

        return (trial, eold)

    def random_starting_point(self, nbasis):
        random = numpy.random.random((nbasis, nbasis))
        random = 0.5 * (random + random.T)
        (energies, eigv) = diagonalise_sorted(random)
        return (energies, eigv)

    def checkerboard(self, nbasis, nup, ndown):
        nalpha = 0
        nbeta = 0
        wfn = numpy.zeros(shape=(nbasis, nup + ndown), dtype=numpy.complex128)
        for i in range(nbasis):
            x, y = decode_basis(4, 4, i)
            if x % 2 == 0 and y % 2 == 0:
                wfn[i, nalpha] = 1.0
                nalpha += 1
            elif x % 2 == 0 and y % 2 == 1:
                wfn[i, nup + nbeta] = -1.0
                nbeta += 1
            elif x % 2 == 1 and y % 2 == 0:
                wfn[i, nup + nbeta] = -1.0
                nbeta += 1
            elif x % 2 == 1 and y % 2 == 1:
                wfn[i, nalpha] = 1.0
                nalpha += 1
        return wfn, 10

    def density(self, wfn):
        return numpy.diag(wfn.dot((wfn.conj()).T))

    def self_consistant(
        self, enew, eold, niup, niup_old, nidown, nidown_old, it, deps=1e-8, verbose=0
    ):
        """Check if system parameters are converged"""

        depsn = deps**0.5
        ediff = abs(enew - eold)
        nup_diff = sum(abs(niup - niup_old)) / len(niup)
        ndown_diff = sum(abs(nidown - nidown_old)) / len(nidown)
        if verbose > 1:
            print("# de: %.10e dniu: %.10e dnid: %.10e" % (ediff, nup_diff, ndown_diff))

        return (ediff < deps) and (nup_diff < depsn) and (ndown_diff < depsn)

    def mix_density(self, new, old, alpha):
        return (1 - alpha) * new + alpha * old

    def diagonalise_mean_field(self, system, hamiltonian, ueff, niup, nidown):
        # mean field Hamiltonians.
        HMFU = hamiltonian.T[0] + numpy.diag(ueff * nidown)
        HMFD = hamiltonian.T[1] + numpy.diag(ueff * niup)
        (e_up, ev_up) = diagonalise_sorted(HMFU)
        (e_down, ev_down) = diagonalise_sorted(HMFD)
        # Construct new wavefunction given new density.
        self.trial[:, : system.nup] = ev_up[:, : system.nup]
        self.trial[:, system.nup :] = ev_down[:, : system.ndown]
        # Construct corresponding site densities.
        niup = self.density(self.trial[:, : system.nup])
        nidown = self.density(self.trial[:, system.nup :])
        return (niup, nidown, e_up, e_down)

    def calculate_energy(self, system):
        if self.verbose:
            print("# Computing trial energy.")
        (self.energy, self.e1b, self.e2b) = local_energy(system, self.G)
        if self.verbose:
            print(
                "# (E, E1B, E2B): (%13.8e, %13.8e, %13.8e)"
                % (self.energy.real, self.e1b.real, self.e2b.real)
            )

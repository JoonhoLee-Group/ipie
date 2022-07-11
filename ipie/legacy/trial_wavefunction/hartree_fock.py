import time

import numpy

from ipie.legacy.estimators.greens_function import gab, gab_mod
from ipie.legacy.estimators.local_energy import local_energy
from ipie.utils.io import read_qmcpack_wfn


class HartreeFock(object):
    def __init__(self, system, hamiltonian, trial_opts, verbose=False, orbs=None):
        self.verbose = verbose
        if verbose:
            print("# Parsing Hartree--Fock trial wavefunction input options.")
        init_time = time.time()
        self.name = "hartree_fock"
        self.type = "hartree_fock"
        self.initial_wavefunction = trial_opts.get(
            "initial_wavefunction", "hartree_fock"
        )
        self.ndets = 1
        self.trial_type = numpy.complex128
        self.psi = numpy.zeros(
            shape=(hamiltonian.nbasis, system.nup + system.ndown), dtype=self.trial_type
        )
        self.excite_ia = trial_opts.get("excitation", None)
        self.wfn_file = trial_opts.get("filename", None)
        if self.wfn_file is not None:
            if verbose:
                print("# Reading trial wavefunction from %s." % self.wfn_file)
            try:
                orbs_matrix = read_qmcpack_wfn(self.wfn_file)
                if verbose:
                    print("# Finished reading wavefunction.")
                msq = hamiltonian.nbasis**2
                if len(orbs_matrix) == msq:
                    orbs_matrix = orbs_matrix.reshape(
                        (hamiltonian.nbasis, hamiltonian.nbasis)
                    )
                else:
                    orbs_alpha = orbs_matrix[:msq].reshape(
                        (hamiltonian.nbasis, hamiltonian.nbasis)
                    )
                    orbs_beta = orbs_matrix[msq:].reshape(
                        (hamiltonian.nbasis, hamiltonian.nbasis)
                    )
                    orbs_matrix = numpy.array([orbs_alpha, orbs_beta])
            except UnicodeDecodeError:
                orbs_matrix = numpy.load(self.wfn_file)
        elif orbs is not None:
            orbs_matrix = orbs
        else:
            # Assuming we're in the MO basis.
            orbs_matrix = numpy.eye(hamiltonian.nbasis)
        # Assuming energy ordered basis set.
        self.full_orbs = orbs_matrix
        occ_a = numpy.arange(system.nup)
        occ_b = numpy.arange(system.ndown)
        nb = hamiltonian.nbasis
        orbs_full = numpy.copy(orbs_matrix)
        if len(orbs_matrix.shape) == 2:
            # RHF
            self.psi[:, : system.nup] = orbs_matrix[:, occ_a]
            self.psi[:, system.nup :] = orbs_matrix[:, occ_b]
            if self.excite_ia is not None:
                # Only deal with alpha spin excitation for the moment.
                i = self.excite_ia[0]
                a = self.excite_ia[1]
                # For serialisation purposes.
                self.excite_ia = numpy.array(self.excite_ia)
                if verbose:
                    print("# Exciting orbital %i to orbital %i in trial." % (i, a))
                self.psi[:, i] = orbs_matrix[:, a]
        else:
            # UHF
            self.psi[:, : system.nup] = orbs_matrix[0][:, occ_a]
            self.psi[:, system.nup :] = orbs_matrix[1][:, occ_b]
            if self.excite_ia is not None:
                # "Promotion energy" calculation.
                # Only deal with alpha spin excitation for the moment.
                i = numpy.array(self.excite_ia[0])
                a = numpy.array(self.excite_ia[1])
                if verbose:
                    print("# Exciting orbital %i to orbital %i in trial." % (i, a))
                self.psi[:, i] = orbs_matrix[:, a]
        gup, self.gup_half = gab_mod(
            self.psi[:, : system.nup], self.psi[:, : system.nup]
        )
        gup, self.gup_half = gab_mod(
            self.psi[:, : system.nup], self.psi[:, : system.nup]
        )
        gdown = numpy.zeros(gup.shape)
        self.gdown_half = numpy.zeros(self.gup_half.shape)

        if system.ndown > 0:
            gdown, self.gdown_half = gab_mod(
                self.psi[:, system.nup :], self.psi[:, system.nup :]
            )

        self.G = numpy.array([gup, gdown], dtype=self.trial_type)
        self.Ghalf = numpy.array(
            [self.gup_half, self.gdown_half], dtype=self.trial_type
        )
        self.coeffs = 1.0
        self.bp_wfn = trial_opts.get("bp_wfn", None)
        self.error = False
        self.initialisation_time = time.time() - init_time
        self.init = self.psi
        self._mem_required = 0.0
        self._rchol = None
        self._eri = None
        self._UVT = None
        if verbose:
            print("# Finished setting up trial wavefunction.")

    def calculate_energy(self, system, hamiltonian):
        if self.verbose:
            print("# Computing trial wavefunction energy.")
        start = time.time()
        (self.energy, self.e1b, self.e2b) = local_energy(
            system, hamiltonian, self, self
        )
        if self.verbose:
            print(
                "# (E, E1B, E2B): (%13.8e, %13.8e, %13.8e)"
                % (self.energy.real, self.e1b.real, self.e2b.real)
            )
            print("# Time to evaluate local energy: {} s".format(time.time() - start))

import time

import numpy
import scipy.linalg

from ipie.legacy.estimators.greens_function import (gab, gab_mod,
                                                    gab_multi_det_full)
from ipie.legacy.estimators.local_energy import local_energy
from ipie.utils.io import read_fortran_complex_numbers
from ipie.utils.linalg import diagonalise_sorted


class MultiDeterminant(object):
    def __init__(self, system, cplx, trial, parallel=False, verbose=False):
        self.verbose = verbose
        if verbose:
            print("# Parsing multi-determinant trial wavefunction input" " options.")
        init_time = time.time()
        self.name = "multi_determinant"
        self.expansion = "multi_determinant"
        self.type = "Not GHF"
        self.eigs = numpy.array([0.0])
        if cplx:
            self.trial_type = numpy.complex128
        else:
            self.trial_type = numpy.float64
        # For debugging purposes.
        self.error = False
        self.orbital_file = trial.get("orbitals", None)
        self.coeffs_file = trial.get("coefficients", None)
        self.write = trial.get("write", False)
        if self.orbital_file is not None:
            self.ndets = trial.get("ndets", None)
            self.psi = numpy.zeros((ndets, nbasis, system.ne), dtype=self.trial_type)
            self.from_ascii(system)
        elif system.orbs is not None:
            orbs = system.orbs.copy()
            self.ndets = orbs.shape[0]
            if system.frozen_core:
                nc = system.ncore
                nfv = system.nfv
                nb = system.nbasis
                orbs_core = orbs[0, :, :nc]
                orbs = orbs[:, nc : nb - nfv, :]
                Gcore, half = gab_mod(orbs_core, orbs_core)
                self.Gcore = numpy.array([Gcore, Gcore])
            self.psi = numpy.zeros(
                shape=(self.ndets, system.nactive, system.ne), dtype=self.trial_type
            )
            self.psi[:, :, : system.nup] = orbs[:, :, nc : nc + system.nup].copy()
            self.psi[:, :, system.nup :] = orbs[
                :, :, 2 * nc + system.nup : 2 * nc + system.ne
            ].copy()
            self.coeffs = system.coeffs
            self.nup = system.nup
        else:
            print("Could not construct trial wavefunction.")
            self.error = True
        nbasis = system.nbasis
        self.GAB = numpy.zeros(
            shape=(2, self.ndets, self.ndets, system.nactive, system.nactive),
            dtype=self.trial_type,
        )
        self.weights = numpy.zeros(
            shape=(2, self.ndets, self.ndets), dtype=self.trial_type
        )
        # Store the complex conjugate of the multi-determinant trial
        # wavefunction expansion coefficients for ease later.
        Gup = gab(self.psi[0, :, : system.nup], self.psi[0, :, : system.nup])
        Gdn = gab(self.psi[0, :, system.nup :], self.psi[0, :, system.nup :])
        self.G = numpy.array([Gup, Gdn])
        self.initialisation_time = time.time() - init_time
        if self.write:
            self.to_qmcpack_ascii()
        if verbose:
            print("# Number of determinants in expansion: %d" % self.ndets)
            print("# Finished setting up trial wavefunction.")

    def from_ascii(self, system):
        if self.verbose:
            print("# Reading wavefunction from %s." % self.coeffs_file)
        self.coeffs = read_fortran_complex_numbers(self.coeffs_file)
        orbitals = read_fortran_complex_numbers(self.orbital_file)
        start = 0
        skip = system.nbasis * system.ne
        end = skip
        for i in range(self.ndets):
            self.psi[i] = orbitals[start:end].reshape((nbasis, system.ne), order="F")
            start = end
            end += skip

    def energy(self, system):
        if self.verbose:
            print("# Computing trial energy.")
        (self.energy, self.e1b, self.e2b) = local_energy(system, self.G, opt=False)
        if self.verbose:
            print(
                "# (E, E1B, E2B): (%13.8e, %13.8e, %13.8e)"
                % (self.energy.real, self.e1b.real, self.e2b.real)
            )

    def to_qmcpack_ascii(self):
        output = open("wf.dat", "w")
        namelist = "&FCI\n UHF = 0\n NCI = %d\n TYPE = occ\n/" % self.psi.shape[0]
        output.write(namelist + "\n")
        norb = self.psi.shape[1]
        for (ci, phi) in zip(self.coeffs, self.psi):
            occup, cols = numpy.where(phi[:, : self.nup] == 1)
            occdn, cols = numpy.where(phi[:, self.nup :] == 1)
            dup = " ".join(str(i + 1) for i in occup)
            ddn = " ".join(str(i + phi.shape[0] + 1) for i in occdn)
            output.write("%f " % ci.real + dup + " " + ddn + "\n")

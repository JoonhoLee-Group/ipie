import cmath
import itertools
import time

import numpy
import scipy
import scipy.sparse.linalg
from scipy.linalg import expm
from scipy.optimize import minimize

from ipie.legacy.estimators.ci import simple_fci, simple_fci_bose_fermi
from ipie.legacy.estimators.greens_function import gab_spin
from ipie.legacy.estimators.hubbard import (local_energy_hubbard,
                                            local_energy_hubbard_holstein)
from ipie.legacy.estimators.local_energy import local_energy
from ipie.legacy.hamiltonians.hubbard import Hubbard
from ipie.legacy.systems.hubbard_holstein import HubbardHolstein
from ipie.legacy.trial_wavefunction.free_electron import FreeElectron
from ipie.legacy.trial_wavefunction.harmonic_oscillator import \
    HarmonicOscillator
from ipie.utils.io import read_fortran_complex_numbers
from ipie.utils.linalg import diagonalise_sorted, reortho

try:
    from jax.config import config

    config.update("jax_enable_x64", True)
    import jax
    import jax.numpy as np
    import jax.scipy.linalg as LA
    import numpy
    from jax import grad, jit

    from ipie.legacy.trial_wavefunction.coherent_state import compute_exp, gab

except ModuleNotFoundError:
    import numpy

    from ipie.legacy.estimators.greens_function import gab

    np = numpy


def gradient(x, nbasis, nup, ndown, T, U, g, m, w0, c0, restricted, relax_gamma):
    grad = numpy.array(
        jax.grad(objective_function)(
            x, nbasis, nup, ndown, T, U, g, m, w0, c0, restricted, relax_gamma
        )
    )
    return grad


def hessian(x, nbasis, nup, ndown, T, U, g, m, w0, c0, restricted, relax_gamma):
    H = numpy.array(
        jax.hessian(objective_function)(
            x, nbasis, nup, ndown, T, U, g, m, w0, c0, restricted, relax_gamma
        )
    )
    return H


def objective_function(
    x, nbasis, nup, ndown, T, U, g, m, w0, c0, restricted, relax_gamma
):
    nbasis = int(round(nbasis))
    nup = int(round(nup))
    ndown = int(round(ndown))

    nbsf = nbasis
    nocca = nup
    noccb = ndown
    nvira = nbasis - nocca
    nvirb = nbasis - noccb

    nova = nocca * nvira
    novb = noccb * nvirb

    daia = np.array(x[:nova], dtype=np.float64)
    daib = np.array(x[nova : nova + novb], dtype=np.float64)

    daia = daia.reshape((nvira, nocca))
    daib = daib.reshape((nvirb, noccb))

    if restricted:
        daib = jax.ops.index_update(daib, jax.ops.index[:, :], daia)

    theta_a = np.zeros((nbsf, nbsf), dtype=np.float64)
    theta_b = np.zeros((nbsf, nbsf), dtype=np.float64)

    theta_a = jax.ops.index_update(theta_a, jax.ops.index[nocca:nbsf, :nocca], daia)
    theta_a = jax.ops.index_update(
        theta_a, jax.ops.index[:nocca, nocca:nbsf], -np.transpose(daia)
    )

    theta_b = jax.ops.index_update(theta_b, jax.ops.index[noccb:nbsf, :noccb], daib)
    theta_b = jax.ops.index_update(
        theta_b, jax.ops.index[:noccb, noccb:nbsf], -np.transpose(daib)
    )

    Ua = np.eye(nbsf, dtype=np.float64)
    tmp = np.eye(nbsf, dtype=np.float64)
    Ua = compute_exp(Ua, tmp, theta_a)

    C0a = np.array(c0[: nbsf * nbsf].reshape((nbsf, nbsf)), dtype=np.float64)
    Ca = C0a.dot(Ua)
    Ga = gab(Ca[:, :nocca], Ca[:, :nocca])

    if noccb > 0:
        C0b = np.array(c0[nbsf * nbsf :].reshape((nbsf, nbsf)), dtype=np.float64)
        Ub = np.eye(nbsf)
        tmp = np.eye(nbsf)
        Ub = compute_exp(Ub, tmp, theta_b)
        Cb = C0b.dot(Ub)
        Gb = gab(Cb[:, :noccb], Cb[:, :noccb])
    else:
        Gb = np.zeros_like(Ga)

    G = np.array([Ga, Gb], dtype=np.float64)

    ni = np.diag(G[0] + G[1])
    nia = np.diag(G[0])
    nib = np.diag(G[1])

    sqrttwomw = np.sqrt(m * w0 * 2.0)
    phi = np.zeros(nbsf)

    gamma = np.array(x[nova + novb :], dtype=np.float64)

    if not relax_gamma:
        gamma = g * np.sqrt(2.0 / (m * w0**3)) * np.ones(nbsf)

    Eph = w0 * np.sum(phi * phi)
    Eeph = np.sum((gamma * m * w0**2 - g * sqrttwomw) * 2.0 * phi / sqrttwomw * ni)
    Eeph += np.sum((gamma**2 * m * w0**2 / 2.0 - g * gamma * sqrttwomw) * ni)

    Eee = np.sum(
        (U * np.ones(nbsf) + gamma**2 * m * w0**2 - 2.0 * g * gamma * sqrttwomw)
        * nia
        * nib
    )

    alpha = gamma * numpy.sqrt(m * w0 / 2.0)
    const = np.exp(-(alpha**2) / 2.0)
    const_mat = np.array((nbsf, nbsf), dtype=np.float64)
    const_mat = np.einsum("i,j->ij", const, const)

    Ekin = np.sum(const_mat * T[0] * G[0] + const_mat * T[1] * G[1])
    etot = Eph + Eeph + Eee + Ekin

    return etot.real


class LangFirsov(object):
    def __init__(self, system, trial, verbose=False):
        self.verbose = verbose
        if verbose:
            print("# Parsing free electron input options.")
        init_time = time.time()
        self.name = "lang_firsov"
        self.type = "lang_firsov"

        self.trial_type = complex

        self.initial_wavefunction = trial.get("initial_wavefunction", "lang_firsov")
        if verbose:
            print("# Diagonalising one-body Hamiltonian.")

        (self.eigs_up, self.eigv_up) = diagonalise_sorted(system.T[0])
        (self.eigs_dn, self.eigv_dn) = diagonalise_sorted(system.T[1])

        self.restricted = trial.get("restricted", False)
        self.reference = trial.get("reference", None)
        self.read_in = trial.get("read_in", None)
        self.psi = numpy.zeros(
            shape=(system.nbasis, system.nup + system.ndown), dtype=self.trial_type
        )

        assert system.name == "HubbardHolstein"

        self.m = system.m
        self.w0 = system.w0

        self.nocca = system.nup
        self.noccb = system.ndown

        if self.read_in is not None:
            if verbose:
                print("# Reading trial wavefunction from %s" % (self.read_in))
            try:
                self.psi = numpy.load(self.read_in)
                self.psi = self.psi.astype(self.trial_type)
            except OSError:
                if verbose:
                    print("# Trial wavefunction is not in native numpy form.")
                    print("# Assuming Fortran GHF format.")
                orbitals = read_fortran_complex_numbers(self.read_in)
                tmp = orbitals.reshape((2 * system.nbasis, system.ne), order="F")
                ups = []
                downs = []
                # deal with potential inconsistency in ghf format...
                for (i, c) in enumerate(tmp.T):
                    if all(abs(c[: system.nbasis]) > 1e-10):
                        ups.append(i)
                    else:
                        downs.append(i)
                self.psi[:, : system.nup] = tmp[: system.nbasis, ups]
                self.psi[:, system.nup :] = tmp[system.nbasis :, downs]
        else:
            # I think this is slightly cleaner than using two separate
            # matrices.
            if self.reference is not None:
                self.psi[:, : system.nup] = self.eigv_up[:, self.reference]
                self.psi[:, system.nup :] = self.eigv_dn[:, self.reference]
            else:
                self.psi[:, : system.nup] = self.eigv_up[:, : system.nup]
                self.psi[:, system.nup :] = self.eigv_dn[:, : system.ndown]

                nocca = system.nup
                noccb = system.ndown
                nvira = system.nbasis - system.nup
                nvirb = system.nbasis - system.ndown
                self.virt = numpy.zeros((system.nbasis, nvira + nvirb))

                self.virt[:, :nvira] = self.eigv_up[:, nocca : nocca + nvira]
                self.virt[:, nvira : nvira + nvirb] = self.eigv_dn[
                    :, noccb : noccb + nvirb
                ]

        gup = gab(self.psi[:, : system.nup], self.psi[:, : system.nup]).T
        if system.ndown > 0:
            gdown = gab(self.psi[:, system.nup :], self.psi[:, system.nup :]).T
        else:
            gdown = numpy.zeros_like(gup)

        self.G = numpy.array([gup, gdown])

        self.relax_gamma = trial.get("relax_gamma", False)

        # For interface compatability
        self.coeffs = 1.0
        self.ndets = 1
        self.bp_wfn = trial.get("bp_wfn", None)
        self.error = False
        self.eigs = numpy.append(self.eigs_up, self.eigs_dn)
        self.eigs.sort()

        self.gamma = (
            system.g
            * numpy.sqrt(2.0 / (system.m * system.w0**3))
            * numpy.ones(system.nbasis)
        )
        print("# Initial gamma = {}".format(self.gamma))
        self.run_variational(system)

        print("# Variational Lang-Firsov Energy = {}".format(self.energy))

        self.initialisation_time = time.time() - init_time
        self.init = self.psi.copy()

        self.shift = numpy.zeros(system.nbasis)
        self.calculate_energy(system)

        self._rchol = None
        self._eri = None
        self._UVT = None

        print("# Lang-Firsov optimized gamma = {}".format(self.gamma))
        print("# Lang-Firsov optimized shift = {}".format(self.shift))
        print("# Lang-Firsov optimized energy = {}".format(self.energy))

        if verbose:
            print("# Updated lang_firsov.")

        if verbose:
            print("# Finished initialising Lang-Firsov trial wavefunction.")

    def run_variational(self, system):
        nbsf = system.nbasis
        nocca = system.nup
        noccb = system.ndown
        nvira = system.nbasis - nocca
        nvirb = system.nbasis - noccb
        #
        nova = nocca * nvira
        novb = noccb * nvirb
        #
        x = numpy.zeros(nova + novb)

        Ca = numpy.zeros((nbsf, nbsf))
        Ca[:, :nocca] = self.psi[:, :nocca]
        Ca[:, nocca:] = self.virt[:, :nvira]
        Cb = numpy.zeros((nbsf, nbsf))
        Cb[:, :noccb] = self.psi[:, nocca:]
        Cb[:, noccb:] = self.virt[:, nvira:]
        #
        if system.ndown > 0:
            c0 = numpy.zeros(nbsf * nbsf * 2)
            c0[: nbsf * nbsf] = Ca.ravel()
            c0[nbsf * nbsf :] = Cb.ravel()
        else:
            c0 = numpy.zeros(nbsf * nbsf)
            c0[: nbsf * nbsf] = Ca.ravel()

        if self.relax_gamma:
            xtmp = numpy.zeros(nova + novb + nbsf)
            xtmp[: nova + novb] = x
            xtmp[nova + novb : nova + novb + nbsf] = self.gamma
            x = xtmp.copy()
        #
        self.shift = numpy.zeros(nbsf)
        self.energy = 1e6

        for i in range(5):  # Try 10 times
            res = minimize(
                objective_function,
                x,
                args=(
                    float(system.nbasis),
                    float(system.nup),
                    float(system.ndown),
                    system.T,
                    system.U,
                    system.g,
                    system.m,
                    system.w0,
                    c0,
                    self.restricted,
                    self.relax_gamma,
                ),
                method="L-BFGS-B",
                jac=gradient,
                options={"disp": False},
            )
            e = res.fun

            if self.verbose:
                print("# macro iter {} energy is {}".format(i, e))
            if e < self.energy and numpy.abs(self.energy - e) > 1e-6:
                self.energy = res.fun
                xconv = res.x.copy()
            else:
                break
            x = numpy.random.randn(x.shape[0]) * 1e-1 + xconv

        daia = res.x[:nova]
        daib = res.x[nova : nova + novb]

        if self.relax_gamma:
            self.gamma = res.x[nova + novb :]

        daia = daia.reshape((nvira, nocca))
        daib = daib.reshape((nvirb, noccb))

        Ua = numpy.zeros((nbsf, nbsf))
        Ub = numpy.zeros((nbsf, nbsf))

        Ua[nocca:nbsf, :nocca] = daia.copy()
        Ua[:nocca, nocca:nbsf] = -daia.T.copy()

        Ub[noccb:nbsf, :noccb] = daib.copy()
        Ub[:noccb, noccb:nbsf] = -daib.T.copy()

        if nocca > 0:
            C0a = c0[: nbsf * nbsf].reshape((nbsf, nbsf))
            Ua = expm(Ua)
            Ca = C0a.dot(Ua)

        if noccb > 0:
            C0b = c0[nbsf * nbsf :].reshape((nbsf, nbsf))
            Ub = expm(Ub)
            Cb = C0b.dot(Ub)

        self.psi[:, :nocca] = Ca[:, :nocca]
        self.psi[:, nocca:] = Cb[:, :noccb]

        self.update_electronic_greens_function(system)

    def update_electronic_greens_function(self, system, verbose=0):
        gup = gab(self.psi[:, : system.nup], self.psi[:, : system.nup]).T
        if system.ndown == 0:
            gdown = numpy.zeros_like(gup)
        else:
            gdown = gab(self.psi[:, system.nup :], self.psi[:, system.nup :]).T
        self.G = numpy.array([gup, gdown])

    def update_wfn(self, system, V, verbose=0):
        (self.eigs_up, self.eigv_up) = diagonalise_sorted(system.T[0] + V[0])
        (self.eigs_dn, self.eigv_dn) = diagonalise_sorted(system.T[1] + V[1])

        # I think this is slightly cleaner than using two separate
        # matrices.
        if self.reference is not None:
            self.psi[:, : system.nup] = self.eigv_up[:, self.reference]
            self.psi[:, system.nup :] = self.eigv_dn[:, self.reference]
        else:
            self.psi[:, : system.nup] = self.eigv_up[:, : system.nup]
            self.psi[:, system.nup :] = self.eigv_dn[:, : system.ndown]
            nocca = system.nup
            noccb = system.ndown
            nvira = system.nbasis - system.nup
            nvirb = system.nbasis - system.ndown

            self.virt[:, :nvira] = self.eigv_up[:, nocca : nocca + nvira]
            self.virt[:, nvira : nvira + nvirb] = self.eigv_dn[:, noccb : noccb + nvirb]

        gup = gab(self.psi[:, : system.nup], self.psi[:, : system.nup]).T

        h1 = system.T[0] + V[0]

        if system.ndown == 0:
            gdown = numpy.zeros_like(gup)
        else:
            gdown = gab(self.psi[:, system.nup :], self.psi[:, system.nup :]).T
        self.eigs = numpy.append(self.eigs_up, self.eigs_dn)
        self.eigs.sort()
        self.G = numpy.array([gup, gdown])

    #   Compute D_{jj}
    def compute_Dvec(self, walker):

        phi0 = self.shift.copy()
        nbsf = walker.X.shape[0]
        D = numpy.zeros(nbsf)
        for i in range(nbsf):
            phi = phi0.copy()
            phi[i] += self.gamma

            # QHO = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=X)
            # D[i] = QHO.value(walker.X)

            QHO = HarmonicOscillator(m=self.m, w=self.w0, order=0, shift=phi0[i])
            denom = QHO.value(walker.X[i])

            QHO = HarmonicOscillator(m=self.m, w=self.w0, order=0, shift=phi[i])
            num = QHO.value(walker.X[i])

            D[i] = num / denom

        return D

    #   Compute \sum_i \partial_i D_{jj} = A_{jj}
    def compute_dDvec(self, walker):

        phi0 = self.shift.copy()
        nbsf = walker.X.shape[0]
        dD = numpy.zeros(nbsf)

        for i in range(nbsf):
            phi = phi0.copy()
            phi[i] += self.gamma
            # QHO = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=X[i])
            # dD[i] = QHO.gradient(walker.X[i]) * QHO.value(walker.X[i]) # gradient is actually grad / value

            QHO = HarmonicOscillator(m=self.m, w=self.w0, order=0, shift=phi0[i])
            denom = QHO.gradient(walker.X[i])

            QHO = HarmonicOscillator(m=self.m, w=self.w0, order=0, shift=phi[i])
            num = QHO.gradient(walker.X[i])

            dD[i] = num / denom
        return dD

    #   Compute \sum_i \partial_i^2 D_{jj} = A_{jj}
    def compute_d2Dvec(self, walker):

        phi0 = self.shift.copy()
        nbsf = walker.X.shape[0]
        d2D = numpy.zeros(nbsf)

        for i in range(nbsf):
            phi = phi0.copy()
            phi[i] += self.gamma

            # QHO = HarmonicOscillator(m = self.m, w = self.w0, order = 0, shift=phi[i])
            # d2D[i] = QHO.laplacian(walker.X[i]) * QHO.value(walker.X[i]) # gradient is actually grad / value

            QHO = HarmonicOscillator(m=self.m, w=self.w0, order=0, shift=phi0[i])
            denom = QHO.laplacian(walker.X[i])

            QHO = HarmonicOscillator(m=self.m, w=self.w0, order=0, shift=phi[i])
            num = QHO.laplacian(walker.X[i])

            d2D[i] = num / denom

        return d2D

    #   Compute  <\psi_T | \partial_i D | \psi> / <\psi_T| D | \psi>
    def gradient(self, walker):

        psi0 = self.psi.copy()

        nbsf = walker.X.shape[0]

        grad = numpy.zeros(nbsf)

        # Compute denominator
        # Dvec = self.compute_Dvec(walker)
        # self.psi = numpy.einsum("m,mi->mi",Dvec, psi0)
        # ot_denom = walker.calc_otrial(self)
        self.psi[:, : self.nocca] = numpy.einsum("m,mi->mi", Dvec, psi0a)
        self.psi[:, self.nocca :] = psi0b
        walker.inverse_overlap(self)
        ot_denom = walker.calc_otrial(self)

        self.psi[:, : self.nocca] = psi0a
        self.psi[:, self.nocca :] = numpy.einsum("m,mi->mi", Dvec, psi0b)
        walker.inverse_overlap(self)
        ot_denom += walker.calc_otrial(self)

        # Compute numerator
        dD = self.compute_dDvec(walker)

        for i in range(nbsf):
            dDvec = numpy.zeros_like(dD)
            dDvec[i] = dD[i]

            self.psi[:, : self.nocca] = numpy.einsum("m,mi->mi", dDvec, psi0a)
            self.psi[:, self.nocca :] = psi0b
            walker.inverse_overlap(self)
            ot_num = walker.calc_otrial(self)

            self.psi[:, : self.nocca] = psi0a
            self.psi[:, self.nocca :] = numpy.einsum("m,mi->mi", dDvec, psi0b)
            walker.inverse_overlap(self)
            ot_num += walker.calc_otrial(self)
            grad[i] = ot_num / ot_denom

        self.psi = psi0.copy()
        return grad

    #   Compute  <\psi_T | \partial_i^2 D | \psi> / <\psi_T| D | \psi>
    def laplacian(self, walker):

        psi0 = self.psi.copy()

        psi0a = psi0[:, : self.nocca]
        psi0b = psi0[:, self.nocca :]

        nbsf = walker.X.shape[0]
        lap = numpy.zeros(nbsf)

        # Compute denominator
        Dvec = self.compute_Dvec(walker)
        self.psi[:, : self.nocca] = numpy.einsum("m,mi->mi", Dvec, psi0a)
        self.psi[:, self.nocca :] = numpy.einsum("m,mi->mi", Dvec, psi0b)
        walker.inverse_overlap(self)
        ot_denom = walker.calc_otrial(self)
        self.psi = psi0.copy()

        # Compute numerator
        d2D = self.compute_d2Dvec(walker)

        QHO = HarmonicOscillator(m=self.m, w=self.w0, order=0, shift=self.shift)
        QHO_lap = QHO.laplacian(walker.X)

        for i in range(nbsf):
            d2Dvec = Dvec.copy()
            d2Dvec[i] = d2D[i]

            self.psi[:, : self.nocca] = numpy.einsum("m,mi->mi", d2Dvec, psi0a)
            self.psi[:, self.nocca :] = numpy.einsum("m,mi->mi", d2Dvec, psi0b)
            walker.inverse_overlap(self)
            ot_num = walker.calc_otrial(self)

            lap[i] = ot_num / ot_denom * QHO_lap[i]

        self.psi = psi0.copy()

        return lap

    def calculate_energy(self, system):
        if self.verbose:
            print("# Computing trial energy.")
        sqrttwomw = numpy.sqrt(system.m * system.w0 * 2.0)
        # alpha = self.gamma * numpy.sqrt(system.m * system.w0 / 2.0)
        phi = self.shift * numpy.sqrt(system.m * system.w0 / 2.0)

        nia = numpy.diag(self.G[0])
        if system.ndown == 0:
            nib = numpy.zeros_like(nia)
        else:
            nib = numpy.diag(self.G[1])
        ni = nia + nib

        Eph = system.w0 * numpy.sum(phi * phi)
        Eeph = numpy.sum(
            (self.gamma * system.w0 - system.g * sqrttwomw) * 2.0 * phi / sqrttwomw * ni
        )
        Eeph += numpy.sum(
            (self.gamma**2 * system.w0 / 2.0 - system.g * self.gamma * sqrttwomw) * ni
        )

        Eee = numpy.sum(
            (
                system.U * numpy.ones(system.nbasis)
                + self.gamma**2 * system.w0
                - 2.0 * system.g * self.gamma * sqrttwomw
            )
            * nia
            * nib
        )

        alpha = self.gamma * numpy.sqrt(system.m * system.w0 / 2.0)
        const = numpy.exp(-(alpha**2) / 2.0)
        const_mat = numpy.einsum("i,j->ij", const, const)

        Ekin = numpy.sum(
            const_mat * system.T[0] * self.G[0] + const_mat * system.T[1] * self.G[1]
        )

        self.energy = Eph + Eeph + Eee + Ekin
        print("# Eee, Ekin, Eph, Eeph = {}, {}, {}, {}".format(Eee, Ekin, Eph, Eeph))

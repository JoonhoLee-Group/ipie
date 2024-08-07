# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Joonho Lee
#          Fionn Malone <fionn.malone@gmail.com>
#

import plum
import numpy
from numba import jit

from ipie.hamiltonians.generic import GenericComplexChol, GenericRealChol
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize


def local_energy_generic_cholesky(system, hamiltonian, G, Ghalf=None):
    r"""Calculate local for generic two-body hamiltonian.

    This uses the cholesky decomposed two-electron integrals.

    Parameters
    ----------
    system : :class:`Generic`
        generic system information
    hamiltonian : ipie hamiltonian object.
        Hamiltonian.
    G : :class:`numpy.ndarray`
        Walker's Green's function.
    Ghalf : :class:`numpy.ndarray`
        Walker's "half-rotated" Green's function.

    Returns
    -------
    (E, T, V): tuple
        Total , one and two-body energies.
    """
    # Element wise multiplication.
    e1b = numpy.sum(hamiltonian.H1[0] * G[0]) + numpy.sum(hamiltonian.H1[1] * G[1])
    nbasis = hamiltonian.nbasis
    nchol = hamiltonian.nchol
    Ga, Gb = G[0], G[1]

    if numpy.isrealobj(hamiltonian.chol):
        Xa = hamiltonian.chol.T.dot(Ga.real.ravel()) + 1.0j * hamiltonian.chol.T.dot(
            Ga.imag.ravel()
        )
        Xb = hamiltonian.chol.T.dot(Gb.real.ravel()) + 1.0j * hamiltonian.chol.T.dot(
            Gb.imag.ravel()
        )
    else:
        Xa = hamiltonian.chol.T.dot(Ga.ravel())
        Xb = hamiltonian.chol.T.dot(Gb.ravel())

    ecoul = numpy.dot(Xa, Xa)
    ecoul += numpy.dot(Xb, Xb)
    ecoul += 2 * numpy.dot(Xa, Xb)

    T = numpy.zeros((nbasis, nbasis), dtype=numpy.complex128)

    GaT = Ga.T.copy()
    GbT = Gb.T.copy()

    exx = 0.0j  # we will iterate over cholesky index to update Ex energy for alpha and beta
    if numpy.isrealobj(hamiltonian.chol):
        for x in range(nchol):  # write a cython function that calls blas for this.
            Lmn = hamiltonian.chol[:, x].reshape((nbasis, nbasis))
            T[:, :].real = GaT.real.dot(Lmn)
            T[:, :].imag = GaT.imag.dot(Lmn)
            exx += numpy.trace(T.dot(T))
            T[:, :].real = GbT.real.dot(Lmn)
            T[:, :].imag = GbT.imag.dot(Lmn)
            exx += numpy.trace(T.dot(T))
    else:
        for x in range(nchol):  # write a cython function that calls blas for this.
            Lmn = hamiltonian.chol[:, x].reshape((nbasis, nbasis))
            T[:, :] = GaT.dot(Lmn)
            exx += numpy.trace(T.dot(T))
            T[:, :] = GbT.dot(Lmn)
            exx += numpy.trace(T.dot(T))

    e2b = 0.5 * (ecoul - exx)

    return (e1b + e2b + hamiltonian.ecore, e1b + hamiltonian.ecore, e2b)


def local_energy_cholesky_opt_dG(trial, hamiltonian, Ghalfa, Ghalfb):
    r"""Calculate local for generic two-body hamiltonian.

    This uses the density difference trick.

    Parameters
    ----------
    trial : ipie trial object
        Trial wavefunction object.
    hamiltonian : ipie hamiltonian object.
        Hamiltonian.
    Ghalfa, Ghalfb : :class:`numpy.ndarray`
        Walker's half-rotated Green's function for each spin sigma.
        Shape is (nsigma, nbasis).

    Returns
    -------
    (E, T, V): tuple
        Total, one and two-body energies.
    """
    dGhalfa = Ghalfa - trial.psia.T
    dGhalfb = Ghalfb - trial.psib.T

    de1 = xp.sum(trial._rH1a * dGhalfa) + xp.sum(trial._rH1b * dGhalfb) + hamiltonian.ecore
    dde2 = xp.sum(trial._rFa_corr * Ghalfa) + xp.sum(trial._rFb_corr * Ghalfb)

    if trial.mixed_precision:
        dGhalfa = dGhalfa.astype(numpy.complex64)
        dGhalfb = dGhalfb.astype(numpy.complex64)

    deJ, deK = half_rotated_cholesky_jk_uhf(trial, hamiltonian, [dGhalfa, dGhalfb])
    de2 = deJ + deK

    if trial.mixed_precision:
        dGhalfa = dGhalfa.astype(numpy.complex128)
        dGhalfb = dGhalfb.astype(numpy.complex128)

    e1 = de1 - hamiltonian.ecore + trial.e1b
    e2 = de2 + dde2 - trial.e2b

    etot = e1 + e2

    return (etot, e1, e2)


def local_energy_cholesky_opt(trial, hamiltonian, Ghalf):
    r"""Calculate local for generic two-body hamiltonian.

    This uses the half-rotated cholesky decomposed two-electron integrals.

    Parameters
    ----------
    trial : ipie trial object
        Trial wavefunction
    hamiltonian : ipie hamiltonian object.
        Hamiltonian.
    Ghalf : list of :class:`numpy.ndarray`
        Walker's half-rotated Green's function, stored as a list of arrays with
        shape (nsigma, nbasis) for each spin sigma.

    Returns
    -------
    (E, T, V): tuple
        Total, one and two-body energies.
    """
    e1b = half_rotated_cholesky_hcore_uhf(trial, Ghalf)
    eJ, eK = half_rotated_cholesky_jk_uhf(trial, hamiltonian, Ghalf)
    e2b = eJ - eK
    ecore = hamiltonian.ecore

    return (e1b + e2b + ecore, e1b + ecore, e2b)


def half_rotated_cholesky_hcore_uhf(trial, Ghalf):
    r"""Calculate local for generic two-body hamiltonian.

    This uses the half-rotated core hamiltonian.

    Parameters
    ----------
    trial : ipie trial object
        Trial wavefunction
    Ghalf : list of :class:`numpy.ndarray`
        Walker's half-rotated Green's function, stored as a list of arrays with
        shape (nsigma, nbasis) for each spin sigma.

    Returns
    -------
    e1b : :class:`numpy.ndarray`
        One-body energy.
    """
    Ghalfa, Ghalfb = Ghalf
    rH1a = trial._rH1a
    rH1b = trial._rH1b

    # Element wise multiplication.
    e1b = xp.sum(rH1a * Ghalfa) + xp.sum(rH1b * Ghalfb)
    return e1b


@jit(nopython=True, fastmath=True)
def ecoul_kernel_real_rchol_uhf(rchola, rcholb, Ghalfa, Ghalfb):
    """Compute Coulomb contribution for real Choleskies with UHF trial.

    Parameters
    ----------
    rchola, rcholb : :class:`numpy.ndarray`
        Half-rotated cholesky for each spin.
    Ghalfa, Ghalfb : :class:`numpy.ndarray`
        Walker's half-rotated Green's function for each spin sigma.
        Shape is (nsigma, nbasis).

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        Coulomb contribution for given green's function.
    """
    # `copy` is needed to return contiguous arrays for numba.
    rchola_real = rchola.real.copy()
    rchola_imag = rchola.imag.copy()
    rcholb_real = rcholb.real.copy()
    rcholb_imag = rcholb.imag.copy()

    Ghalfa_real = Ghalfa.real.ravel()
    Ghalfa_imag = Ghalfa.imag.ravel()
    Ghalfb_real = Ghalfb.real.ravel()
    Ghalfb_imag = Ghalfb.imag.ravel()

    Xa = rchola_real.dot(Ghalfa_real) + 1.0j * rchola_imag.dot(Ghalfa_real)
    Xa += 1.0j * rchola_real.dot(Ghalfa_imag) - rchola_imag.dot(Ghalfa_imag)
    Xb = rcholb_real.dot(Ghalfb_real) + 1.0j * rcholb_imag.dot(Ghalfb_real)
    Xb += 1.0j * rcholb_real.dot(Ghalfb_imag) - rcholb_imag.dot(Ghalfb_imag)
    X = Xa + Xb

    ecoul = 0.5 * numpy.dot(X, X)
    return ecoul


@jit(nopython=True, fastmath=True)
def ecoul_kernel_complex_rchol_uhf(rchola, rcholb, rcholbara, rcholbarb, Ghalfa, Ghalfb):
    """Compute Coulomb contribution for complex Choleskies with UHF trial.

    Parameters
    ----------
    rchola, rcholb : :class:`numpy.ndarray`
        Half-rotated cholesky for each spin.
    rcholbara, rcholbarb : :class:`numpy.ndarray`
        Complex conjugate of half-rotated cholesky for each spin.
    Ghalfa, Ghalfb : :class:`numpy.ndarray`
        Walker's half-rotated Green's function for each spin sigma.
        Shape is (nsigma, nbasis).

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        Coulomb contribution for given green's function.
    """
    rchola_real = rchola.real.copy()
    rchola_imag = rchola.imag.copy()
    rcholb_real = rcholb.real.copy()
    rcholb_imag = rcholb.imag.copy()

    rcholbara_real = rcholbara.real.copy()
    rcholbara_imag = rcholbara.imag.copy()
    rcholbarb_real = rcholbarb.real.copy()
    rcholbarb_imag = rcholbarb.imag.copy()

    Ghalfa_real = Ghalfa.real.ravel()
    Ghalfa_imag = Ghalfa.imag.ravel()
    Ghalfb_real = Ghalfb.real.ravel()
    Ghalfb_imag = Ghalfb.imag.ravel()

    X1 = rchola_real.dot(Ghalfa_real) + 1.0j * rchola_imag.dot(Ghalfa_real)
    X1 += 1.0j * rchola_real.dot(Ghalfa_imag) - rchola_imag.dot(Ghalfa_imag)
    X1 += rcholb_real.dot(Ghalfb_real) + 1.0j * rcholb_imag.dot(Ghalfb_real)
    X1 += 1.0j * rcholb_real.dot(Ghalfb_imag) - rcholb_imag.dot(Ghalfb_imag)

    X2 = rcholbara_real.dot(Ghalfa_real) + 1.0j * rcholbara_imag.dot(Ghalfa_real)
    X2 += 1.0j * rcholbara_real.dot(Ghalfa_imag) - rcholbara_imag.dot(Ghalfa_imag)
    X2 += rcholbarb_real.dot(Ghalfb_real) + 1.0j * rcholbarb_imag.dot(Ghalfb_real)
    X2 += 1.0j * rcholbarb_real.dot(Ghalfb_imag) - rcholbarb_imag.dot(Ghalfb_imag)

    ecoul = 0.5 * numpy.dot(X1, X2)
    return ecoul


@jit(nopython=True, fastmath=True)
def exx_kernel_real_rchol(rchol, Ghalf):
    """Compute exchange contribution for real Choleskies with RHF/UHF trial.

    Parameters
    ----------
    rchol : :class:`numpy.ndarray`
        Half-rotated cholesky for one spin.
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated Green's function for spin sigma.
        Shape is (nsigma, nbasis).

    Returns
    -------
    exx : :class:`numpy.ndarray`
        exchange contribution for given green's function.
    """
    nchol = rchol.shape[0]
    nocc = Ghalf.shape[0]
    nbasis = Ghalf.shape[1]

    rchol_real = rchol.real.copy()
    rchol_imag = rchol.imag.copy()

    Ghalf_realT = Ghalf.T.real.copy()
    Ghalf_imagT = Ghalf.T.imag.copy()

    exx = 0.0j

    # Fix this with gpu env
    for x in range(nchol):
        rcholx_real = rchol_real[x].reshape((nocc, nbasis))
        rcholx_imag = rchol_imag[x].reshape((nocc, nbasis))

        T = rcholx_real.dot(Ghalf_realT) + 1.0j * rcholx_imag.dot(Ghalf_realT)
        T += 1.0j * rcholx_real.dot(Ghalf_imagT) - rcholx_imag.dot(Ghalf_imagT)
        exx += numpy.dot(T.ravel(), T.T.ravel())

    exx *= 0.5
    return exx


@jit(nopython=True, fastmath=True)
def exx_kernel_complex_rchol(rchol, rcholbar, Ghalf):
    """Compute exchange contribution for complex Choleskies with RHF/UHF trial.

    Parameters
    ----------
    rchol : :class:`numpy.ndarray`
        Half-rotated cholesky for one spin.
    rcholbar : :class:`numpy.ndarray`
        Complex conjugate of half-rotated cholesky for one spin.
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated Green's function for spin sigma.
        Shape is (nsigma, nbasis).

    Returns
    -------
    exx : :class:`numpy.ndarray`
        exchange contribution for given green's function.
    """
    nchol = rchol.shape[0]
    nocc = Ghalf.shape[0]
    nbasis = Ghalf.shape[1]

    rchol_real = rchol.real.copy()
    rchol_imag = rchol.imag.copy()

    rcholbar_real = rcholbar.real.copy()
    rcholbar_imag = rcholbar.imag.copy()

    Ghalf_realT = Ghalf.T.real.copy()
    Ghalf_imagT = Ghalf.T.imag.copy()

    exx = 0.0j

    for x in range(nchol):
        rcholx_real = rchol_real[x].reshape((nocc, nbasis))
        rcholx_imag = rchol_imag[x].reshape((nocc, nbasis))
        rcholbarx_real = rcholbar_real[x].reshape((nocc, nbasis))
        rcholbarx_imag = rcholbar_imag[x].reshape((nocc, nbasis))

        T1 = rcholx_real.dot(Ghalf_realT) + 1.0j * rcholx_imag.dot(Ghalf_realT)
        T1 += 1.0j * rcholx_real.dot(Ghalf_imagT) - rcholx_imag.dot(Ghalf_imagT)
        T2 = rcholbarx_real.dot(Ghalf_realT) + 1.0j * rcholbarx_imag.dot(Ghalf_realT)
        T2 += 1.0j * rcholbarx_real.dot(Ghalf_imagT) - rcholbarx_imag.dot(Ghalf_imagT)
        exx += numpy.dot(T1.ravel(), T2.T.ravel())

    exx *= 0.5
    return exx


@plum.dispatch
def half_rotated_cholesky_jk_uhf(trial, hamiltonian: GenericRealChol, Ghalf):
    """Compute exchange and coulomb contributions via jitted kernels.

    Parameters
    ----------
    trial : ipie trial object
        Trial wavefunction
    hamiltonian : ipie hamiltonian object.
        Hamiltonian.
    Ghalf : list of :class:`numpy.ndarray`
        Walker's half-rotated Green's function, stored as a list of arrays with
        shape (nsigma, nbasis) for each spin sigma.

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        Coulomb energy.
    exx : :class:`numpy.ndarray`
        Exchange energy.
    """
    Ghalfa, Ghalfb = Ghalf
    rchola = trial._rchola
    rcholb = trial._rcholb

    # Element wise multiplication.
    ecoul = ecoul_kernel_real_rchol_uhf(rchola, rcholb, Ghalfa, Ghalfb)
    exx = exx_kernel_real_rchol(rchola, Ghalfa) + exx_kernel_real_rchol(rcholb, Ghalfb)
    synchronize()
    return ecoul, exx  # JK energy


@plum.dispatch
def half_rotated_cholesky_jk_uhf(trial, hamiltonian: GenericComplexChol, Ghalf):
    """Compute exchange and coulomb contributions via jitted kernels.

    Parameters
    ----------
    trial : ipie trial object
        Trial wavefunction
    hamiltonian : ipie hamiltonian object.
        Hamiltonian.
    Ghalf : list of :class:`numpy.ndarray`
        Walker's half-rotated Green's function, stored as a list of arrays with
        shape (nsigma, nbasis) for each spin sigma.

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        Coulomb energy.
    exx : :class:`numpy.ndarray`
        Exchange energy.
    """
    Ghalfa, Ghalfb = Ghalf
    rchola = trial._rchola
    rcholb = trial._rcholb
    rcholbara = trial._rcholbara
    rcholbarb = trial._rcholbarb

    # Element wise multiplication.
    ecoul = ecoul_kernel_complex_rchol_uhf(rchola, rcholb, rcholbara, rcholbarb, Ghalfa, Ghalfb)
    exx = exx_kernel_complex_rchol(rchola, rcholbara, Ghalfa) + exx_kernel_complex_rchol(
        rcholb, rcholbarb, Ghalfb
    )
    synchronize()
    return ecoul, exx  # JK energy


@jit(nopython=True, fastmath=True)
def ecoul_kernel_real_rchol_ghf(chol, Gaa, Gbb):
    """Compute Coulomb contribution for real Choleskies.

    Parameters
    ----------
    chol : :class:`numpy.ndarray`
        Choleskies.
    Gaa, Gbb : :class:`numpy.ndarray`
        Walker's Green's function in the diagonal spin blocks.

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        Coulomb contribution for given Green's function.
    """
    Gcharge = Gaa + Gbb
    Gcharge_real = Gcharge.real.ravel()
    Gcharge_imag = Gcharge.imag.ravel()

    # Assuming `chol` is real.
    X = Gcharge_real.dot(chol) + 1.0j * Gcharge_imag.dot(chol)
    ecoul = 0.5 * numpy.dot(X, X)
    return ecoul


@jit(nopython=True, fastmath=True)
def ecoul_kernel_complex_rchol_ghf(A, B, Gaa, Gbb):
    """Compute Coulomb contribution for complex Choleskies.

    Parameters
    ----------
    A, B : :class:`numpy.ndarray`
        Linear combination of complex Choleskies.
    Gaa, Gbb : :class:`numpy.ndarray`
        Walker's Green's function in the diagonal spin blocks.

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        Coulomb contribution for given Green's function.
    """
    Gcharge = Gaa + Gbb
    XA = Gcharge.ravel().dot(A)
    XB = Gcharge.ravel().dot(B)
    ecoul = 0.5 * (numpy.dot(XA, XA) + numpy.dot(XB, XB))
    return ecoul


@jit(nopython=True, fastmath=True)
def exx_kernel_real_rchol_ghf(chol, Gaa, Gbb, Gab, Gba):
    """Compute exchange contribution for real Choleskies.

    Parameters
    ----------
    chol : :class:`numpy.ndarray`
        Choleskies.
    Gaa, Gbb, Gab, Gba : :class:`numpy.ndarray`
        Walker's Green's function in each of the 4 spin blocks.

    Returns
    -------
    exx : :class:`numpy.ndarray`
        exchange contribution for given green's function.
    """
    nbasis = Gaa.shape[0]
    nchol = chol.shape[1]

    Lmn = chol.T.copy()  # (nchol, nbasis^2)

    Gaa_realT = Gaa.T.real.copy()
    Gaa_imagT = Gaa.T.imag.copy()
    Gbb_realT = Gbb.T.real.copy()
    Gbb_realT = Gbb.T.real.copy()
    Gbb_imagT = Gbb.T.imag.copy()
    Gab_realT = Gab.T.real.copy()
    Gab_imagT = Gab.T.imag.copy()
    Gba_realT = Gba.T.real.copy()
    Gba_imagT = Gba.T.imag.copy()

    exx = 0.0j

    # We will iterate over cholesky index to update Ex energy for alpha and beta.
    # Assume `Lmn` is real.
    for x in range(nchol):
        Lmnx = Lmn[x].reshape((nbasis, nbasis))
        T = Gaa_realT.dot(Lmnx) + 1.0j * Gaa_imagT.dot(Lmnx)
        exx += numpy.trace(T.dot(T))

        T = Gbb_realT.dot(Lmnx) + 1.0j * Gbb_imagT.dot(Lmnx)
        exx += numpy.trace(T.dot(T))

        Tab = Gab_realT.dot(Lmnx) + 1.0j * Gab_imagT.dot(Lmnx)
        Tba = Gba_realT.dot(Lmnx) + 1.0j * Gba_imagT.dot(Lmnx)
        exx += 2.0 * numpy.trace(Tab.dot(Tba))

    exx *= 0.5
    return exx


@jit(nopython=True, fastmath=True)
def exx_kernel_complex_rchol_ghf(A, B, Gaa, Gbb, Gab, Gba):
    """Compute exchange contribution for complex Choleskies.

    Parameters
    ----------
    A, B : :class:`numpy.ndarray`
        Linear combination of complex Choleskies.
    Gaa, Gbb, Gab, Gba : :class:`numpy.ndarray`
        Walker's Green's function in each of the 4 spin blocks.

    Returns
    -------
    exx : :class:`numpy.ndarray`
        exchange contribution for given green's function.
    """
    nbasis = Gaa.shape[0]
    nchol = A.shape[1]

    Amn = A.T.copy()
    Bmn = B.T.copy()

    GaaT = Gaa.T.copy()
    GbbT = Gbb.T.copy()
    GabT = Gab.T.copy()
    GbaT = Gba.T.copy()

    exx = 0.0j

    # We will iterate over cholesky index to update Ex energy for alpha and beta.
    for x in range(nchol):
        Amnx = Amn[x].reshape((nbasis, nbasis))
        Bmnx = Bmn[x].reshape((nbasis, nbasis))
        TA = GaaT.dot(Amnx)
        TB = GaaT.dot(Bmnx)
        exx += numpy.trace(TA.dot(TA)) + numpy.trace(TB.dot(TB))

        TA = GbbT.dot(Amnx)
        TB = GbbT.dot(Bmnx)
        exx += numpy.trace(TA.dot(TA)) + numpy.trace(TB.dot(TB))

        TAab = GabT.dot(Amnx)
        TBab = GabT.dot(Bmnx)
        TAba = GbaT.dot(Amnx)
        TBba = GbaT.dot(Bmnx)
        exx += 2.0 * (numpy.trace(TAab.dot(TAba)) + numpy.trace(TBab.dot(TBba)))

    exx *= 0.5
    return exx


@plum.dispatch
def cholesky_jk_ghf(hamiltonian: GenericRealChol, G):
    nbasis = G.shape[0] // 2
    Gaa = G[:nbasis, :nbasis].copy()
    Gbb = G[nbasis:, nbasis:].copy()
    Gab = G[:nbasis, nbasis:].copy()
    Gba = G[nbasis:, :nbasis].copy()

    ecoul = ecoul_kernel_real_rchol_ghf(hamiltonian.chol, Gaa, Gbb)
    exx = exx_kernel_real_rchol_ghf(hamiltonian.chol, Gaa, Gbb, Gab, Gba)
    return ecoul, exx  # JK energy


@plum.dispatch
def cholesky_jk_ghf(hamiltonian: GenericComplexChol, G):
    nbasis = G.shape[0] // 2
    Gaa = G[:nbasis, :nbasis].copy()
    Gbb = G[nbasis:, nbasis:].copy()
    Gab = G[:nbasis, nbasis:].copy()
    Gba = G[nbasis:, :nbasis].copy()

    ecoul = ecoul_kernel_complex_rchol_ghf(hamiltonian.A, hamiltonian.B, Gaa, Gbb)
    exx = exx_kernel_complex_rchol_ghf(hamiltonian.A, hamiltonian.B, Gaa, Gbb, Gab, Gba)
    return ecoul, exx  # JK energy


def core_contribution_cholesky(chol, G):
    nb = G[0].shape[-1]
    cmat = chol.reshape((-1, nb * nb))
    X = numpy.dot(cmat, G[0].ravel())
    Ja = numpy.dot(cmat.T, X).reshape(nb, nb)
    T = numpy.tensordot(chol, G[0], axes=((1), (0)))
    Ka = numpy.tensordot(T, chol, axes=((0, 2), (0, 2)))
    hca = Ja - 0.5 * Ka
    X = numpy.dot(cmat, G[1].ravel())
    Jb = numpy.dot(cmat.T, X).reshape(nb, nb)
    T = numpy.tensordot(chol, G[1], axes=((1), (0)))
    Kb = numpy.tensordot(T, chol, axes=((0, 2), (0, 2)))
    hcb = Jb - 0.5 * Kb
    return (hca, hcb)


def fock_generic(system, P):
    if system.sparse:
        mf_shift = 1j * P[0].ravel() * system.hs_pot
        mf_shift += 1j * P[1].ravel() * system.hs_pot
        VMF = 1j * system.hs_pot.dot(mf_shift).reshape(system.nbasis, system.nbasis)
    else:
        mf_shift = 1j * numpy.einsum("lpq,spq->l", system.hs_pot, P)
        VMF = 1j * numpy.einsum("lpq,l->pq", system.hs_pot, mf_shift)
    return system.h1e_mod - VMF

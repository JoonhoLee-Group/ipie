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
# Authors: Fionn Malone <fionn.malone@gmail.com>
#          Joonho Lee <linusjoonho@gmail.com>
#

import numpy
from numba import jit
from math import ceil
from typing import Union

from ipie.estimators.local_energy import local_energy_G
from ipie.estimators.kernels import exchange_reduction
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize

from ipie.systems.generic import Generic
from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.walkers.ghf_walkers import GHFWalkers
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF
from ipie.estimators.generic import cholesky_jk_ghf

import plum

# Note specialisations occur to because:
# 1. Numba does not allow for mixing types without a warning so need to split
# real and complex components apart when rchol is real. Green's function is
# complex in general.
# Optimize for case when wavefunction is RHF (factor of 2 saving)


@jit(nopython=True, fastmath=True)
def ecoul_kernel_batch_real_rchol_rhf(rchola, Ghalfa_batch):
    """Compute coulomb contribution for real rchol with RHF trial.

    Parameters
    ----------
    rchol : :class:`numpy.ndarray`
        Half-rotated cholesky.
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        coulomb contribution for all walkers.
    """
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot
    nwalkers = Ghalfa_batch.shape[0]
    Ghalfa_batch_real = Ghalfa_batch.real.copy()
    Ghalfa_batch_imag = Ghalfa_batch.imag.copy()
    X = rchola.dot(Ghalfa_batch_real.T) + 1.0j * rchola.dot(Ghalfa_batch_imag.T)  # naux x nwalkers
    ecoul = zeros(nwalkers, dtype=numpy.complex128)
    X = X.T.copy()
    for iw in range(nwalkers):
        ecoul[iw] += 2.0 * dot(X[iw], X[iw])

    return ecoul


@jit(nopython=True, fastmath=True)
def ecoul_kernel_batch_complex_rchol_rhf(rchola, Ghalfa_batch):
    """Compute coulomb contribution for complex rchol with RHF trial.

    Parameters
    ----------
    rchola, rcholb : :class:`numpy.ndarray`
        Half-rotated cholesky for each spin.
    Ghalfa_batch, Ghalfb_batch : :class:`numpy.ndarray`
        Walker's half-rotated Green's function for each spin sigma.
        Shape is (nwalkers, nsigma, nbasis).

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        coulomb contribution for all walkers.
    """
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot
    nwalkers = Ghalfa_batch.shape[0]

    X = rchola.dot(Ghalfa_batch.T)
    ecoul = zeros(nwalkers, dtype=numpy.complex128)
    X = X.T.copy()
    for iw in range(nwalkers):
        ecoul[iw] += 2.0 * dot(X[iw], X[iw].conj())
    return ecoul


def local_energy_single_det_rhf_batch(system, hamiltonian, walkers, trial):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant RHF case.

    Parameters
    ----------
    system : system object
        System being studied.
    hamiltonian : hamiltonian object
        Hamiltonian being studied.
    walkers : WalkerBatch
        Walkers object.
    trial : trial object
        Trial wavefunctioni.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """
    nwalkers = walkers.Ghalfa.shape[0]
    nalpha = walkers.Ghalfa.shape[1]
    nbasis = hamiltonian.nbasis
    hamiltonian.nchol

    walkers.Ghalfa = walkers.Ghalfa.reshape(nwalkers, nalpha * nbasis)

    e1b = 2.0 * walkers.Ghalfa.dot(trial._rH1a.ravel()) + hamiltonian.ecore

    if xp.isrealobj(trial._rchola):
        ecoul = ecoul_kernel_batch_real_rchol_rhf(trial._rchola, walkers.Ghalfa)
    else:
        ecoul = ecoul_kernel_batch_complex_rchol_rhf(trial._rchola, walkers.Ghalfa)

    walkers.Ghalfa = walkers.Ghalfa.reshape(nwalkers, nalpha, nbasis)

    if xp.isrealobj(trial._rchola):
        exx = 2.0 * exx_kernel_batch_real_rchol(trial._rchola, walkers.Ghalfa)
    else:
        exx = 2.0 * exx_kernel_batch_complex_rchol(trial._rchola, trial._rcholbar, walkers.Ghalfa)

    e2b = ecoul - exx

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b

    return energy


@jit(nopython=True, fastmath=True)
def ecoul_kernel_batch_real_rchol_uhf(rchola, rcholb, Ghalfa_batch, Ghalfb_batch):
    """Compute Coulomb contribution for real Choleskies with UHF trial.

    Parameters
    ----------
    rchola, rcholb : :class:`numpy.ndarray`
        Half-rotated cholesky for each spin.
    Ghalfa_batch, Ghalfb_batch : :class:`numpy.ndarray`
        Walker's half-rotated Green's function for each spin sigma. 
        Shape is (nwalkers, nsigma, nbasis).

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        Coulomb contribution for all walkers.
    """
    nwalkers = Ghalfa_batch.shape[0]
    nchol = rchola.shape[0]

    # `copy` is needed to return contiguous arrays for numba.
    rchola_realT = rchola.T.real.copy() 
    rchola_imagT = rchola.T.imag.copy()
    rcholb_realT = rcholb.T.real.copy()
    rcholb_imagT = rcholb.T.imag.copy()
    
    Ghalfa_batch_real = Ghalfa_batch.real.copy().reshape((nwalkers, -1)) 
    Ghalfa_batch_imag = Ghalfa_batch.imag.copy().reshape((nwalkers, -1))
    Ghalfb_batch_real = Ghalfb_batch.real.copy().reshape((nwalkers, -1))
    Ghalfb_batch_imag = Ghalfb_batch.imag.copy().reshape((nwalkers, -1))
    
    # (nwalkers, nchol)
    Xa = Ghalfa_batch_real.dot(rchola_realT) + 1.j * Ghalfa_batch_real.dot(rchola_imagT)
    Xa += 1.j * Ghalfa_batch_imag.dot(rchola_realT) - Ghalfa_batch_imag.dot(rchola_imagT)
    Xb = Ghalfb_batch_real.dot(rcholb_realT) + 1.j * Ghalfb_batch_real.dot(rcholb_imagT)
    Xb += 1.j * Ghalfb_batch_imag.dot(rcholb_realT) - Ghalfb_batch_imag.dot(rcholb_imagT)
    X = Xa + Xb
    
    ecoul = numpy.zeros(nwalkers, dtype=numpy.complex128)

    for iw in range(nwalkers):
        ecoul[iw] = numpy.dot(X[iw], X[iw])

    ecoul *= 0.5
    return ecoul


@jit(nopython=True, fastmath=True)
def ecoul_kernel_batch_complex_rchol_uhf(
    rchola, rcholb, rcholbara, rcholbarb, Ghalfa_batch, Ghalfb_batch):
    """Compute coulomb contribution for real rchol with UHF trial.

    Parameters
    ----------
    rchola, rcholb : :class:`numpy.ndarray`
        Half-rotated cholesky for each spin.
    rcholbara, rcholbarb : :class:`numpy.ndarray`
        Complex conjugate of half-rotated cholesky for each spin.
    Ghalfa_batch, Ghalfb_batch : :class:`numpy.ndarray`
        Walker's half-rotated Green's function for each spin sigma.
        Shape is (nwalkers, nsigma * nbasis).

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        Coulomb contribution for all walkers.
    """
    nwalkers = Ghalfa_batch.shape[0]

    # `copy` is needed to return contiguous arrays for numba.
    rchola_realT = rchola.T.real.copy() 
    rchola_imagT = rchola.T.imag.copy()
    rcholb_realT = rcholb.T.real.copy()
    rcholb_imagT = rcholb.T.imag.copy()
    
    rcholbara_realT = rcholbara.T.real.copy() 
    rcholbara_imagT = rcholbara.T.imag.copy()
    rcholbarb_realT = rcholbarb.T.real.copy()
    rcholbarb_imagT = rcholbarb.T.imag.copy()

    Ghalfa_batch_real = Ghalfa_batch.real.copy().reshape((nwalkers, -1)) 
    Ghalfa_batch_imag = Ghalfa_batch.imag.copy().reshape((nwalkers, -1))
    Ghalfb_batch_real = Ghalfb_batch.real.copy().reshape((nwalkers, -1))
    Ghalfb_batch_imag = Ghalfb_batch.imag.copy().reshape((nwalkers, -1))

    # (nwalkers, nchol)
    X1 = Ghalfa_batch_real.dot(rchola_realT) + 1.j * Ghalfa_batch_real.dot(rchola_imagT)
    X1 += 1.j * Ghalfa_batch_imag.dot(rchola_realT) - Ghalfa_batch_imag.dot(rchola_imagT)
    X1 += Ghalfb_batch_real.dot(rcholb_realT) + 1.j * Ghalfb_batch_real.dot(rcholb_imagT)
    X1 += 1.j * Ghalfb_batch_imag.dot(rcholb_realT) - Ghalfb_batch_imag.dot(rcholb_imagT)

    X2 = Ghalfa_batch_real.dot(rcholbara_realT) + 1.j * Ghalfa_batch_real.dot(rcholbara_imagT)
    X2 += 1.j * Ghalfa_batch_imag.dot(rcholbara_realT) - Ghalfa_batch_imag.dot(rcholbara_imagT)
    X2 += Ghalfb_batch_real.dot(rcholbarb_realT) + 1.j * Ghalfb_batch_real.dot(rcholbarb_imagT)
    X2 += 1.j * Ghalfb_batch_imag.dot(rcholbarb_realT) - Ghalfb_batch_imag.dot(rcholbarb_imagT)

    ecoul = numpy.zeros(nwalkers, dtype=numpy.complex128)

    for iw in range(nwalkers):
        ecoul[iw] += numpy.dot(X1[iw], X2[iw])

    ecoul *= 0.5
    return ecoul


@jit(nopython=True, fastmath=True)
def exx_kernel_batch_real_rchol(rchol, Ghalf_batch):
    """Compute exchange contribution for real Choleskies with RHF/UHF trial.

    Parameters
    ----------
    rchol : :class:`numpy.ndarray`
        Half-rotated cholesky for one spin.
    Ghalf_batch : :class:`numpy.ndarray`
        Walker's half-rotated Green's function for spin sigma.
        Shape is (nwalkers, nsigma, nbasis).

    Returns
    -------
    exx : :class:`numpy.ndarray`
        Exchange contribution for all walkers.
    """
    nwalkers = Ghalf_batch.shape[0]
    nchol = rchol.shape[0]
    nocc = Ghalf_batch.shape[1]
    nbasis = Ghalf_batch.shape[-1]
    
    rchol_real = rchol.real.copy()
    rchol_imag = rchol.imag.copy()

    exx = numpy.zeros(nwalkers, dtype=numpy.complex128)

    for iw in range(nwalkers):
        Ghalf_realT = Ghalf_batch[iw].T.real.copy()
        Ghalf_imagT = Ghalf_batch[iw].T.imag.copy()

        for x in range(nchol):
            rcholx_real = rchol_real[x].reshape((nocc, nbasis))
            rcholx_imag = rchol_imag[x].reshape((nocc, nbasis))

            T = rcholx_real.dot(Ghalf_realT) + 1.j * rcholx_imag.dot(Ghalf_realT)
            T += 1.j * rcholx_real.dot(Ghalf_imagT) - rcholx_imag.dot(Ghalf_imagT) 
            exx[iw] += numpy.dot(T.ravel(), T.T.ravel())

    exx *= 0.5
    return exx


@jit(nopython=True, fastmath=True)
def exx_kernel_batch_complex_rchol(rchol, rcholbar, Ghalf_batch):
    """Compute coulomb contribution for complex rchol with RHF/UHF trial.

    Parameters
    ----------
    rchol : :class:`numpy.ndarray`
        Half-rotated cholesky for one spin.
    rcholbar : :class:`numpy.ndarray`
        Complex conjugate of half-rotated cholesky for one spin.
    Ghalf_batch : :class:`numpy.ndarray`
        Walker's half-rotated Green's function for spin sigma.
        Shape is (nwalkers, nsigma, nbasis).

    Returns
    -------
    exx : :class:`numpy.ndarray`
        Exchange contribution for all walkers.
    """
    nwalkers = Ghalf_batch.shape[0]
    nchol = rchol.shape[0]
    nocc = Ghalf_batch.shape[1]
    nbasis = Ghalf_batch.shape[-1]
    
    rchol_real = rchol.real.copy()
    rchol_imag = rchol.imag.copy()

    rcholbar_real = rcholbar.real.copy()
    rcholbar_imag = rcholbar.imag.copy()
    
    exx = numpy.zeros((nwalkers), dtype=numpy.complex128)

    for iw in range(nwalkers):
        Ghalf_realT = Ghalf_batch[iw].T.real.copy()
        Ghalf_imagT = Ghalf_batch[iw].T.imag.copy()

        for x in range(nchol):
            rcholx_real = rchol_real[x].reshape((nocc, nbasis))
            rcholx_imag = rchol_imag[x].reshape((nocc, nbasis))
            rcholbarx_real = rcholbar_real[x].reshape((nocc, nbasis))
            rcholbarx_imag = rcholbar_imag[x].reshape((nocc, nbasis))

            T1 = rcholx_real.dot(Ghalf_realT) + 1.j * rcholx_imag.dot(Ghalf_realT)
            T1 += 1.j * rcholx_real.dot(Ghalf_imagT) - rcholx_imag.dot(Ghalf_imagT)
            T2 = rcholbarx_real.dot(Ghalf_realT) + 1.j * rcholbarx_imag.dot(Ghalf_realT)
            T2 += 1.j * rcholbarx_real.dot(Ghalf_imagT) - rcholbarx_imag.dot(Ghalf_imagT)
            exx[iw] += numpy.dot(T1.ravel(), T2.T.ravel())

    exx *= 0.5
    return exx


@plum.dispatch
def local_energy_single_det_uhf_batch(
    system: Generic, 
    hamiltonian: GenericRealChol, 
    walkers: UHFWalkers, 
    trial: SingleDet
):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant UHF case.

    Parameters
    ----------
    system : system object
        System being studied.
    hamiltonian : hamiltonian object
        Hamiltonian being studied.
    walkers : WalkerBatch
        Walkers object.
    trial : trial object
        Trial wavefunctioni.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """
    nwalkers = walkers.Ghalfa.shape[0]
    nbasis = hamiltonian.nbasis

    Ghalfa_batch = walkers.Ghalfa.reshape((nwalkers, -1))
    Ghalfb_batch = walkers.Ghalfb.reshape((nwalkers, -1))

    e1b = Ghalfa_batch.dot(trial._rH1a.ravel())
    e1b += Ghalfb_batch.dot(trial._rH1b.ravel())
    e1b += hamiltonian.ecore

    ecoul = ecoul_kernel_batch_real_rchol_uhf(
            trial._rchola, trial._rcholb, walkers.Ghalfa, walkers.Ghalfb)
    exx = exx_kernel_batch_real_rchol(trial._rchola, walkers.Ghalfa) + \
            exx_kernel_batch_real_rchol(trial._rcholb, walkers.Ghalfb)
    e2b = ecoul - exx

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b
    return energy


@plum.dispatch
def local_energy_single_det_uhf_batch(
    system: Generic,
    hamiltonian: GenericComplexChol,
    walkers: UHFWalkers,
    trial: SingleDet,
):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant UHF case.

    Parameters
    ----------
    system : system object
        System being studied.
    hamiltonian : hamiltonian object
        Hamiltonian being studied.
    walkers : WalkerBatch
        Walkers object.
    trial : trial object
        Trial wavefunctioni.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """
    nwalkers = walkers.Ghalfa.shape[0]
    nbasis = hamiltonian.nbasis

    Ghalfa_batch = walkers.Ghalfa.reshape((nwalkers, -1))
    Ghalfb_batch = walkers.Ghalfb.reshape((nwalkers, -1))

    e1b = Ghalfa_batch.dot(trial._rH1a.ravel())
    e1b += Ghalfb_batch.dot(trial._rH1b.ravel())
    e1b += hamiltonian.ecore

    ecoul = ecoul_kernel_batch_complex_rchol_uhf(
                trial._rchola, trial._rcholb, trial._rcholbara, trial._rcholbarb, 
                walkers.Ghalfa, walkers.Ghalfb)
    exx = exx_kernel_batch_complex_rchol(trial._rchola, trial._rcholbara, walkers.Ghalfa) + \
            exx_kernel_batch_complex_rchol(trial._rcholb, trial._rcholbarb, walkers.Ghalfb)
    e2b = ecoul - exx

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b
    return energy


def two_body_energy_uhf(trial, walkers):
    """Compute two body energy only.

    Single determinant case (UHF). For use in Wick's code.

    TODO FDM: Is this used any more (maybe delete.

    Parameters
    ----------
    trial : trial object
        Trial wavefunctioni.
    walkers : WalkerBatch
        Walkers object.

    Returns
    -------
    two_body_energy : np.ndarray
        Coulomb + exchange (no core constant contribution).
    """
    nwalkers = walkers.Ghalfa.shape[0]
    nalpha = walkers.Ghalfa.shape[1]
    nbeta = walkers.Ghalfb.shape[1]
    nbasis = walkers.Ghalfa.shape[2]
    Ghalfa = walkers.Ghalfa.reshape(nwalkers, nalpha * nbasis)
    Ghalfb = walkers.Ghalfb.reshape(nwalkers, nbeta * nbasis)
    if xp.isrealobj(trial._rchola):
        ecoul = ecoul_kernel_batch_real_rchol_uhf(trial._rchola, trial._rcholb, Ghalfa, Ghalfb)
        exx = exx_kernel_batch_real_rchol(
            trial._rchola, walkers.Ghalfa
        ) + exx_kernel_batch_real_rchol(trial._rcholb, walkers.Ghalfb)
    else:
        ecoul = ecoul_kernel_batch_complex_rchol_uhf(
            trial._rchola,
            trial._rcholb,
            trial._rcholbara,
            trial.rcholbarb,
            Ghalfa,
            Ghalfb,
        )
        exx = exx_kernel_batch_complex_rchol(
            trial._rchola,
            trial.rcholbara,
            walkers.Ghalfa,
        ) + exx_kernel_batch_complex_rchol(
            trial._rcholb,
            trial._rcholbarb,
            walkers.Ghalfb,
        )
    return ecoul - exx


@jit(nopython=True, fastmath=True)
def ecoul_kernel_batch_real_rchol_ghf(chol, Gaa_batch, Gbb_batch):
    """Compute Coulomb contribution for real chol with GHF trial.

    Parameters
    ----------
    chol : :class:`numpy.ndarray`
        Cholesky.
    Gaa_batch, Gbb_batch : :class:`numpy.ndarray`
        Walker's Green's function in the diagonal spin blocks.
        Shape is (nwalkers, nbasis, nbasis)

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        Coulomb contribution for all walkers.
    """
    nwalkers = Gaa_batch.shape[0]
    Gcharge_batch = Gaa_batch + Gbb_batch
    Gcharge_batch_real = Gcharge_batch.real.copy().reshape((nwalkers, -1))
    Gcharge_batch_imag = Gcharge_batch.imag.copy().reshape((nwalkers, -1))

    # Assuming `chol` is real.
    # (nwalkers, nchol)
    X = Gcharge_batch_real.dot(chol) + 1.j * Gcharge_batch_imag.dot(chol)
    ecoul = numpy.zeros(nwalkers, dtype=numpy.complex128)

    for iw in range(nwalkers):
        ecoul[iw] = numpy.dot(X[iw], X[iw])

    ecoul *= 0.5
    return ecoul


@jit(nopython=True, fastmath=True)
def ecoul_kernel_batch_complex_rchol_ghf(A, B, Gaa_batch, Gbb_batch):
    """Compute Coulomb contribution for complex chol with GHF trial.

    Parameters
    ----------
    A, B : :class:`numpy.ndarray`
        Linear combination of complex Choleskies.
    Gaa_batch, Gbb_batch : :class:`numpy.ndarray`
        Walker's Green's function in the diagonal spin blocks.

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        Coulomb contribution for all walkers.
    """
    nwalkers = Gaa_batch.shape[0]
    Gcharge_batch = (Gaa_batch + Gbb_batch).reshape((nwalkers, -1))

    # (nwalkers, nchol)
    XA = Gcharge_batch.dot(A)
    XB = Gcharge_batch.dot(B)
    ecoul = numpy.zeros(nwalkers, dtype=numpy.complex128)

    for iw in range(nwalkers):
        ecoul[iw] = numpy.dot(XA[iw], XA[iw]) + numpy.dot(XB[iw], XB[iw])

    ecoul *= 0.5
    return ecoul


@jit(nopython=True, fastmath=True)
def exx_kernel_batch_real_rchol_ghf(chol, Gaa_batch, Gbb_batch, Gab_batch, Gba_batch):
    """Compute exchange contribution for real chol with GHF trial.

    Parameters
    ----------
    chol : :class:`numpy.ndarray`
        Cholesky.
    Gaa_batch, Gbb_batch, Gab_batch, Gba_batch : :class:`numpy.ndarray`
        Walker's Green's function in the 4 spin blocks.
        Shape is (nwalkers, nbasis, nbasis)

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        Coulomb contribution for all walkers.
    """
    nwalkers = Gaa_batch.shape[0]
    nbasis = Gaa_batch.shape[-1]
    nchol = chol.shape[1]
    
    Lmn = chol.T.copy() # (nchol, nbasis^2)
    exx = numpy.zeros(nwalkers, dtype=numpy.complex128)
    
    # Assume `Lmn` is real.
    for iw in range(nwalkers):
        # P = G.T
        Gaa_realT = Gaa_batch[iw].T.real.copy()
        Gaa_imagT = Gaa_batch[iw].T.imag.copy()
        Gbb_realT = Gbb_batch[iw].T.real.copy()
        Gbb_imagT = Gbb_batch[iw].T.imag.copy()
        Gab_realT = Gab_batch[iw].T.real.copy()
        Gab_imagT = Gab_batch[iw].T.imag.copy()
        Gba_realT = Gba_batch[iw].T.real.copy()
        Gba_imagT = Gba_batch[iw].T.imag.copy()

        for x in range(nchol):
            Lmnx = Lmn[x].reshape((nbasis, nbasis))
            T = Gaa_realT.dot(Lmnx) + 1.j * Gaa_imagT.dot(Lmnx)
            exx[iw] += numpy.trace(T.dot(T))

            T = Gbb_realT.dot(Lmnx) + 1.j * Gbb_imagT.dot(Lmnx)
            exx[iw] += numpy.trace(T.dot(T))

            Tab = Gab_realT.dot(Lmnx) + 1.j * Gab_imagT.dot(Lmnx)
            Tba = Gba_realT.dot(Lmnx) + 1.j * Gba_imagT.dot(Lmnx)
            exx[iw] += 2. * numpy.trace(Tab.dot(Tba))
    
    exx *= 0.5
    return exx


@jit(nopython=True, fastmath=True)
def exx_kernel_batch_complex_rchol_ghf(A, B, Gaa_batch, Gbb_batch, Gab_batch, Gba_batch):
    """Compute exchange contribution for complex chol with GHF trial.

    Parameters
    ----------
    A, B : :class:`numpy.ndarray`
        Linear combination of complex Choleskies.
    Gaa_batch, Gbb_batch, Gab_batch, Gba_batch : :class:`numpy.ndarray`
        Walker's Green's function in the 4 spin blocks.
        Shape is (nwalkers, nbasis, nbasis)

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        Coulomb contribution for all walkers.
    """
    nwalkers = Gaa_batch.shape[0]
    nbasis = Gaa_batch.shape[-1]
    nchol = A.shape[1]

    Amn = A.T.copy() # (nchol, nbasis^2)
    Bmn = B.T.copy()
    exx = numpy.zeros(nwalkers, dtype=numpy.complex128)

    for iw in range(nwalkers):
        # P = G.T
        GaaT = Gaa_batch[iw].T.copy() 
        GbbT = Gbb_batch[iw].T.copy()
        GabT = Gab_batch[iw].T.copy()
        GbaT = Gba_batch[iw].T.copy()
    
        for x in range(nchol):
            Amnx = Amn[x].reshape((nbasis, nbasis))
            Bmnx = Bmn[x].reshape((nbasis, nbasis))
            TA = GaaT.dot(Amnx)
            TB = GaaT.dot(Bmnx)
            exx[iw] += numpy.trace(TA.dot(TA)) + numpy.trace(TB.dot(TB))
            
            TA = GbbT.dot(Amnx)
            TB = GbbT.dot(Bmnx)
            exx[iw] += numpy.trace(TA.dot(TA)) + numpy.trace(TB.dot(TB))
            
            TAab = GabT.dot(Amnx)
            TBab = GabT.dot(Bmnx)
            TAba = GbaT.dot(Amnx)
            TBba = GbaT.dot(Bmnx)
            exx[iw] += 2.0 * (numpy.trace(TAab.dot(TAba)) + numpy.trace(TBab.dot(TBba)))
            
    exx *= 0.5
    return exx


@plum.dispatch
def local_energy_single_det_ghf_batch(
    system: Generic, 
    hamiltonian: GenericRealChol, 
    walkers: GHFWalkers, 
    trial: SingleDetGHF
):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant GHF case.

    Parameters
    ----------
    system : system object
        System being studied.
    hamiltonian : hamiltonian object
        Hamiltonian being studied.
    walkers : WalkerBatch
        Walkers object.
    trial : trial object
        Trial wavefunctioni.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """
    nwalkers = walkers.nwalkers
    nbasis = hamiltonian.nbasis
    
    Gaa_batch = walkers.G[:, :nbasis, :nbasis].copy() 
    Gbb_batch = walkers.G[:, nbasis:, nbasis:].copy()
    Gab_batch = walkers.G[:, :nbasis, nbasis:].copy()
    Gba_batch = walkers.G[:, nbasis:, :nbasis].copy()

    e1b = Gaa_batch.reshape((nwalkers, -1)).dot(hamiltonian.H1[0].ravel()) # (nwalkers,)
    e1b += Gbb_batch.reshape((nwalkers, -1)).dot(hamiltonian.H1[1].ravel()) 
    e1b += hamiltonian.ecore

    ecoul = ecoul_kernel_batch_real_rchol_ghf(
                hamiltonian.chol, Gaa_batch, Gbb_batch)
    exx = exx_kernel_batch_real_rchol_ghf(
                hamiltonian.chol, Gaa_batch, Gbb_batch, Gab_batch, Gba_batch)
    e2b = ecoul - exx

    energy = numpy.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b
    return energy


@plum.dispatch
def local_energy_single_det_ghf_batch(
    system: Generic, 
    hamiltonian: GenericComplexChol, 
    walkers: GHFWalkers, 
    trial: SingleDetGHF
):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant GHF case.

    Parameters
    ----------
    system : system object
        System being studied.
    hamiltonian : hamiltonian object
        Hamiltonian being studied.
    walkers : WalkerBatch
        Walkers object.
    trial : trial object
        Trial wavefunctioni.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """
    nwalkers = walkers.nwalkers
    nbasis = hamiltonian.nbasis

    Gaa_batch = walkers.G[:, :nbasis, :nbasis]
    Gbb_batch = walkers.G[:, nbasis:, nbasis:]
    Gab_batch = walkers.G[:, :nbasis, nbasis:]
    Gba_batch = walkers.G[:, nbasis:, :nbasis]

    e1b = Gaa_batch.reshape((nwalkers, -1)).dot(hamiltonian.H1[0].ravel()) # (nwalkers,)
    e1b += Gbb_batch.reshape((nwalkers, -1)).dot(hamiltonian.H1[1].ravel()) 
    e1b += hamiltonian.ecore

    ecoul = ecoul_kernel_batch_complex_rchol_ghf(
                hamiltonian.A, hamiltonian.B, Gaa_batch, Gbb_batch)
    exx = exx_kernel_batch_complex_rchol_ghf(
                hamiltonian.A, hamiltonian.B, Gaa_batch, Gbb_batch, Gab_batch, Gba_batch)
    e2b = ecoul - exx

    energy = numpy.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b
    return energy


def local_energy_single_det_batch(system, hamiltonian, walkers, trial):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant case.

    Parameters
    ----------
    system : system object
        System being studied.
    hamiltonian : hamiltonian object
        Hamiltonian being studied.
    walkers : WalkerBatch
        Walkers object.
    trial : trial object
        Trial wavefunctioni.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """
    energy = []
    nwalkers = walkers.nwalkers
    for idx in range(nwalkers):
        G = [walkers.Ga[idx], walkers.Gb[idx]]
        Ghalf = [walkers.Ghalfa[idx], walkers.Ghalfb[idx]]
        energy += [list(local_energy_G(system, hamiltonian, trial, G, Ghalf))]

    energy = xp.array(energy, dtype=numpy.complex128)
    return energy


def local_energy_single_det_batch_einsum(system, hamiltonian, walkers, trial):
    """Compute local energy for walker batch (all walkers at once).

    Use einsum rather than numba kernels.

    Single determinant case.

    Parameters
    ----------
    system : system object
        System being studied.
    hamiltonian : hamiltonian object
        Hamiltonian being studied.
    walkers : WalkerBatch
        Walkers object.
    trial : trial object
        Trial wavefunctioni.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """
    nalpha = walkers.Ghalfa.shape[1]
    nbeta = walkers.Ghalfb.shape[1]
    nbasis = walkers.Ghalfa.shape[-1]
    nwalkers = walkers.Ghalfa.shape[0]
    nchol = hamiltonian.nchol

    walkers.Ghalfa = walkers.Ghalfa.reshape(nwalkers, nalpha * nbasis)
    walkers.Ghalfb = walkers.Ghalfb.reshape(nwalkers, nbeta * nbasis)

    e1b = (
        walkers.Ghalfa.dot(trial._rH1a.ravel())
        + walkers.Ghalfb.dot(trial._rH1b.ravel())
        + hamiltonian.ecore
    )

    if xp.isrealobj(trial._rchola):
        Xa = trial._rchola.dot(walkers.Ghalfa.real.T) + 1.0j * trial._rchola.dot(
            walkers.Ghalfa.imag.T
        )  # naux x nwalkers
        Xb = trial._rcholb.dot(walkers.Ghalfb.real.T) + 1.0j * trial._rcholb.dot(
            walkers.Ghalfb.imag.T
        )  # naux x nwalkers
    else:
        Xa = trial._rchola.dot(walkers.Ghalfa.T)
        Xb = trial._rcholb.dot(walkers.Ghalfb.T)

    ecoul = xp.einsum("xw,xw->w", Xa, Xa, optimize=True)
    ecoul += xp.einsum("xw,xw->w", Xb, Xb, optimize=True)
    ecoul += 2.0 * xp.einsum("xw,xw->w", Xa, Xb, optimize=True)

    walkers.Ghalfa = walkers.Ghalfa.reshape(nwalkers, nalpha, nbasis)
    walkers.Ghalfb = walkers.Ghalfb.reshape(nwalkers, nbeta, nbasis)

    Ta = xp.zeros((nwalkers, nalpha, nalpha), dtype=numpy.complex128)
    Tb = xp.zeros((nwalkers, nbeta, nbeta), dtype=numpy.complex128)

    GhalfaT_batch = walkers.Ghalfa.transpose(0, 2, 1).copy()  # nw x nbasis x nocc
    GhalfbT_batch = walkers.Ghalfb.transpose(0, 2, 1).copy()  # nw x nbasis x nocc

    exx = xp.zeros(
        nwalkers, dtype=numpy.complex128
    )  # we will iterate over cholesky index to update Ex energy for alpha and beta
    # breakpoint()
    for x in range(nchol):  # write a cython function that calls blas for this.
        rmi_a = trial._rchola[x].reshape((nalpha, nbasis))
        rmi_b = trial._rcholb[x].reshape((nbeta, nbasis))
        if xp.isrealobj(trial._rchola):  # this is actually fasater
            Ta[:, :, :].real = rmi_a.dot(GhalfaT_batch.real).transpose(1, 0, 2)
            Ta[:, :, :].imag = rmi_a.dot(GhalfaT_batch.imag).transpose(1, 0, 2)
            Tb[:, :, :].real = rmi_b.dot(GhalfbT_batch.real).transpose(1, 0, 2)
            Tb[:, :, :].imag = rmi_b.dot(GhalfbT_batch.imag).transpose(1, 0, 2)
        else:
            Ta = rmi_a.dot(GhalfaT_batch).transpose(1, 0, 2)
            Tb = rmi_b.dot(GhalfbT_batch).transpose(1, 0, 2)
        # this James Spencer's change is actually slower
        # Ta = walkers.Ghalfa @ rmi_a.T
        # Tb = walkers.Ghalfb @ rmi_b.T

        exx += xp.einsum("wij,wji->w", Ta, Ta, optimize=True) + xp.einsum(
            "wij,wji->w", Tb, Tb, optimize=True
        )

    e2b = 0.5 * (ecoul - exx)

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b

    return energy


def local_energy_single_det_batch_gpu_old(system, hamiltonian, walkers, trial):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant UHF GPU case.

    einsum code path very memory intensive will OOM without warning.

    Parameters
    ----------
    system : system object
        System being studied.
    hamiltonian : hamiltonian object
        Hamiltonian being studied.
    walkers : WalkerBatch
        Walkers object.
    trial : trial object
        Trial wavefunctioni.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """
    nwalkers = walkers.Ghalfa.shape[0]
    nalpha = walkers.Ghalfa.shape[1]
    nbeta = walkers.Ghalfb.shape[1]
    nbasis = walkers.Ghalfa.shape[-1]
    nchol = hamiltonian.nchol

    walkers.Ghalfa = walkers.Ghalfa.reshape(nwalkers, nalpha * nbasis)
    walkers.Ghalfb = walkers.Ghalfb.reshape(nwalkers, nbeta * nbasis)

    e1b = (
        walkers.Ghalfa.dot(trial._rH1a.ravel())
        + walkers.Ghalfb.dot(trial._rH1b.ravel())
        + hamiltonian.ecore
    )

    if xp.isrealobj(trial._rchola):
        Xa = trial._rchola.dot(walkers.Ghalfa.real.T) + 1.0j * trial._rchola.dot(
            walkers.Ghalfa.imag.T
        )  # naux x nwalkers
        Xb = trial._rcholb.dot(walkers.Ghalfb.real.T) + 1.0j * trial._rcholb.dot(
            walkers.Ghalfb.imag.T
        )  # naux x nwalkers
    else:
        Xa = trial._rchola.dot(walkers.Ghalfa.T)
        Xb = trial._rcholb.dot(walkers.Ghalfb.T)

    ecoul = xp.einsum("xw,xw->w", Xa, Xa, optimize=True)
    ecoul += xp.einsum("xw,xw->w", Xb, Xb, optimize=True)
    ecoul += 2.0 * xp.einsum("xw,xw->w", Xa, Xb, optimize=True)

    walkers.Ghalfa = walkers.Ghalfa.reshape(nwalkers, nalpha, nbasis)
    walkers.Ghalfb = walkers.Ghalfb.reshape(nwalkers, nbeta, nbasis)

    trial._rchola = trial._rchola.reshape(nchol, nalpha, nbasis)
    trial._rcholb = trial._rcholb.reshape(nchol, nbeta, nbasis)

    Txij = xp.einsum("xim,wjm->wxji", trial._rchola, walkers.Ghalfa)
    exx = xp.einsum("wxji,wxij->w", Txij, Txij)
    Txij = xp.einsum("xim,wjm->wxji", trial._rcholb, walkers.Ghalfb)
    exx += xp.einsum("wxji,wxij->w", Txij, Txij)

    trial._rchola = trial._rchola.reshape(nchol, nalpha * nbasis)
    trial._rcholb = trial._rcholb.reshape(nchol, nbeta * nbasis)

    e2b = 0.5 * (ecoul - exx)

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b

    synchronize()

    return energy


def local_energy_single_det_batch_gpu(system, hamiltonian, walkers, trial, max_mem=2.0):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant UHF GPU case.

    Numba kernel jitted route (qmcpack algorithm)

    Parameters
    ----------
    system : system object
        System being studied.
    hamiltonian : hamiltonian object
        Hamiltonian being studied.
    walkers : WalkerBatch
        Walkers object.
    trial : trial object
        Trial wavefunctioni.
    max_mem : float
        Maximum memory in GB for intermediates. Optional. Default 2GB.
        TODO FDM: Remove following config setup.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """

    nwalkers = walkers.Ghalfa.shape[0]
    nalpha = walkers.Ghalfa.shape[1]
    nbeta = walkers.Ghalfb.shape[1]
    nbasis = walkers.Ghalfa.shape[-1]
    nchol = hamiltonian.nchol

    Ghalfa = walkers.Ghalfa.reshape(nwalkers, nalpha * nbasis)
    Ghalfb = walkers.Ghalfb.reshape(nwalkers, nbeta * nbasis)

    e1b = Ghalfa.dot(trial._rH1a.ravel()) + Ghalfb.dot(trial._rH1b.ravel()) + hamiltonian.ecore

    if xp.isrealobj(trial._rchola):
        Xa = trial._rchola.dot(Ghalfa.real.T) + 1.0j * trial._rchola.dot(
            Ghalfa.imag.T
        )  # naux x nwalkers
        Xb = trial._rcholb.dot(Ghalfb.real.T) + 1.0j * trial._rcholb.dot(
            Ghalfb.imag.T
        )  # naux x nwalkers
    else:
        Xa = trial._rchola.dot(Ghalfa.T)
        Xb = trial._rcholb.dot(Ghalfb.T)

    ecoul = xp.einsum("xw,xw->w", Xa, Xa, optimize=True)
    ecoul += xp.einsum("xw,xw->w", Xb, Xb, optimize=True)
    ecoul += 2.0 * xp.einsum("xw,xw->w", Xa, Xb, optimize=True)

    max_nocc = max(nalpha, nbeta)
    mem_needed = 16 * nwalkers * max_nocc * max_nocc * nchol / (1024.0**3.0)
    num_chunks = max(1, ceil(mem_needed / max_mem))
    chunk_size = ceil(nchol / num_chunks)
    nchol_chunks = ceil(nchol / chunk_size)

    # Buffer for large intermediate tensor
    buff = xp.zeros(shape=(nwalkers * chunk_size * max_nocc * max_nocc), dtype=xp.complex128)
    nchol_chunk_size = chunk_size
    nchol_left = nchol
    exx = xp.zeros(nwalkers, dtype=xp.complex128)
    Ghalfa = walkers.Ghalfa.reshape((nwalkers * nalpha, nbasis))
    Ghalfb = walkers.Ghalfb.reshape((nwalkers * nbeta, nbasis))
    for i in range(nchol_chunks):
        nchol_chunk = min(nchol_chunk_size, nchol_left)
        chol_sls = slice(i * chunk_size, i * chunk_size + nchol_chunk)
        size = nwalkers * nchol_chunk * nalpha * nalpha
        # alpha-alpha
        Txij = buff[:size].reshape((nchol_chunk * nalpha, nwalkers * nalpha))
        rchol = trial._rchola[chol_sls].reshape((nchol_chunk * nalpha, nbasis))
        xp.dot(rchol, Ghalfa.T, out=Txij)
        Txij = Txij.reshape((nchol_chunk, nalpha, nwalkers, nalpha))
        exchange_reduction(Txij, exx)
        # beta-beta
        size = nwalkers * nchol_chunk * nbeta * nbeta
        Txij = buff[:size].reshape((nchol_chunk * nbeta, nwalkers * nbeta))
        rchol = trial._rcholb[chol_sls].reshape((nchol_chunk * nbeta, nbasis))
        xp.dot(rchol, Ghalfb.T, out=Txij)
        Txij = Txij.reshape((nchol_chunk, nbeta, nwalkers, nbeta))
        exchange_reduction(Txij, exx)
        nchol_left -= chunk_size

    e2b = 0.5 * (ecoul - exx)

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b

    synchronize()
    return energy

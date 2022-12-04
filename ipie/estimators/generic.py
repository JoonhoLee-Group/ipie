
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

import sys

import numpy
from numba import jit

from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize



# FDM: deprecated remove?
def local_energy_generic_opt(system, G, Ghalf=None, eri=None):
    """Compute local energy using half-rotated eri tensor.
    """

    na = system.nup
    nb = system.ndown
    M = system.nbasis
    assert eri is not None

    vipjq_aa = eri[0, : na**2 * M**2].reshape((na, M, na, M))
    vipjq_bb = eri[0, na**2 * M**2 : na**2 * M**2 + nb**2 * M**2].reshape(
        (nb, M, nb, M)
    )
    vipjq_ab = eri[0, na**2 * M**2 + nb**2 * M**2 :].reshape((na, M, nb, M))

    Ga, Gb = Ghalf[0], Ghalf[1]
    # Element wise multiplication.
    e1b = numpy.sum(system.H1[0] * G[0]) + numpy.sum(system.H1[1] * G[1])
    # Coulomb
    eJaa = 0.5 * numpy.einsum("irjs,ir,js", vipjq_aa, Ga, Ga)
    eJbb = 0.5 * numpy.einsum("irjs,ir,js", vipjq_bb, Gb, Gb)
    eJab = numpy.einsum("irjs,ir,js", vipjq_ab, Ga, Gb)

    eKaa = -0.5 * numpy.einsum("irjs,is,jr", vipjq_aa, Ga, Ga)
    eKbb = -0.5 * numpy.einsum("irjs,is,jr", vipjq_bb, Gb, Gb)

    e2b = eJaa + eJbb + eJab + eKaa + eKbb

    return (e1b + e2b + system.ecore, e1b + system.ecore, e2b)


def local_energy_generic_cholesky(system, ham, G, Ghalf=None):
    r"""Calculate local for generic two-body hamiltonian.

    This uses the cholesky decomposed two-electron integrals.

    Parameters
    ----------
    system : :class:`Generic`
        generic system information
    ham : :class:`Generic`
        ab-initio hamiltonian information
    G : :class:`numpy.ndarray`
        Walker's "green's function"
    Ghalf : :class:`numpy.ndarray`
        Walker's "half-rotated" "green's function"

    Returns
    -------
    (E, T, V): tuple
        Total , one and two-body energies.
    """
    # Element wise multiplication.
    e1b = numpy.sum(ham.H1[0] * G[0]) + numpy.sum(ham.H1[1] * G[1])
    nalpha, nbeta = system.nup, system.ndown
    nbasis = ham.nbasis
    nchol = ham.nchol
    Ga, Gb = G[0], G[1]

    if numpy.isrealobj(ham.chol_vecs):
        Xa = ham.chol_vecs.T.dot(Ga.real.ravel()) + 1.0j * ham.chol_vecs.T.dot(
            Ga.imag.ravel()
        )
        Xb = ham.chol_vecs.T.dot(Gb.real.ravel()) + 1.0j * ham.chol_vecs.T.dot(
            Gb.imag.ravel()
        )
    else:
        Xa = ham.chol_vecs.T.dot(Ga.ravel())
        Xb = ham.chol_vecs.T.dot(Gb.ravel())

    ecoul = numpy.dot(Xa, Xa)
    ecoul += numpy.dot(Xb, Xb)
    ecoul += 2 * numpy.dot(Xa, Xb)

    T = numpy.zeros((nbasis, nbasis), dtype=numpy.complex128)

    GaT = Ga.T.copy()
    GbT = Gb.T.copy()

    exx = 0.0j  # we will iterate over cholesky index to update Ex energy for alpha and beta
    if numpy.isrealobj(ham.chol_vecs):
        for x in range(nchol):  # write a cython function that calls blas for this.
            Lmn = ham.chol_vecs[:, x].reshape((nbasis, nbasis))
            T[:, :].real = GaT.real.dot(Lmn)
            T[:, :].imag = GaT.imag.dot(Lmn)
            exx += numpy.trace(T.dot(T))
            T[:, :].real = GbT.real.dot(Lmn)
            T[:, :].imag = GbT.imag.dot(Lmn)
            exx += numpy.trace(T.dot(T))
    else:
        for x in range(nchol):  # write a cython function that calls blas for this.
            Lmn = ham.chol_vecs[:, x].reshape((nbasis, nbasis))
            T[:, :] = GaT.dot(Lmn)
            exx += numpy.trace(T.dot(T))
            T[:, :] = GbT.dot(Lmn)
            exx += numpy.trace(T.dot(T))

    e2b = 0.5 * (ecoul - exx)

    return (e1b + e2b + ham.ecore, e1b + ham.ecore, e2b)


def local_energy_cholesky_opt_dG(system, ecore, Ghalfa, Ghalfb, trial):
    r"""Calculate local for generic two-body hamiltonian.

    This uses the density difference trick.

    Parameters
    ----------
    system : :class:`Generic`
        generic system information
    ecore : float
        Core energy
    Ghalfa : :class:`numpy.ndarray`
        Walker's "half-rotated" alpha "green's function"
    Ghalfa : :class:`numpy.ndarray`
        Walker's "half-rotated" beta "green's function"
    trial : ipie trial object
        Trial wavefunction object.

    Returns
    -------
    (E, T, V): tuple
        Total, one and two-body energies.
    """
    dGhalfa = Ghalfa - trial.psia.T
    dGhalfb = Ghalfb - trial.psib.T

    de1 = xp.sum(trial._rH1a * dGhalfa) + xp.sum(trial._rH1b * dGhalfb) + ecore
    dde2 = xp.sum(trial._rFa_corr * Ghalfa) + xp.sum(trial._rFb_corr * Ghalfb)

    if trial.mixed_precision:
        dGhalfa = dGhalfa.astype(numpy.complex64)
        dGhalfb = dGhalfb.astype(numpy.complex64)

    deJ, deK = half_rotated_cholesky_jk(system, dGhalfa, dGhalfb, trial)
    de2 = deJ + deK

    if trial.mixed_precision:
        dGhalfa = dGhalfa.astype(numpy.complex128)
        dGhalfb = dGhalfb.astype(numpy.complex128)

    e1 = de1 - ecore + trial.e1b
    e2 = de2 + dde2 - trial.e2b

    etot = e1 + e2

    return (etot, e1, e2)


def local_energy_cholesky_opt(system, ecore, Ghalfa, Ghalfb, trial):
    r"""Calculate local for generic two-body hamiltonian.

    This uses the half-rotated cholesky decomposed two-electron integrals.

    Parameters
    ----------
    system : :class:`Generic`
        System information for Generic.
    Ghalfa : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis
    Ghalfa : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nbeta x nbasis
    trial : ipie trial object
        Trial wavefunction

    Returns
    -------
    (E, T, V): tuple
        Total, one and two-body energies.
    """
    e1b = half_rotated_cholesky_hcore(system, Ghalfa, Ghalfb, trial)
    eJ, eK = half_rotated_cholesky_jk(system, Ghalfa, Ghalfb, trial)
    e2b = eJ + eK

    return (e1b + e2b + ecore, e1b + ecore, e2b)


def half_rotated_cholesky_hcore(system, Ghalfa, Ghalfb, trial):
    r"""Calculate local for generic two-body hamiltonian.

    This uses the half-rotated core hamiltonian.

    Parameters
    ----------
    system : :class:`Generic`
        System information for Generic.
    Ghalfa : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis
    Ghalfa : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nbeta x nbasis
    trial : ipie trial object
        Trial wavefunction

    Returns
    -------
    e1b : :class:`numpy.ndarray`
        One-body energy.
    """
    rH1a = trial._rH1a
    rH1b = trial._rH1b

    # Element wise multiplication.
    e1b = xp.sum(rH1a * Ghalfa) + xp.sum(rH1b * Ghalfb)
    return e1b


@jit(nopython=True, fastmath=True)
def exx_kernel_rchol_real(rchol, Ghalf):
    """Compute exchange contribution for real rchol.

    Parameters
    ----------
    rchol : :class:`numpy.ndarray`
        Half-rotated cholesky.
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis

    Returns
    -------
    exx : :class:`numpy.ndarray`
        exchange contribution for given green's function.
    """
    naux = rchol.shape[0]
    nocc = Ghalf.shape[0]
    nbasis = Ghalf.shape[1]

    exx = 0 + 0j
    Greal = Ghalf.real.copy()
    Gimag = Ghalf.imag.copy()
    # Fix this with gpu env
    for jx in range(naux):
        rmi = rchol[jx].reshape((nocc, nbasis))
        T = rmi.dot(Greal.T) + 1.0j * rmi.dot(Gimag.T)
        exx += numpy.dot(T.ravel(), T.T.ravel())
    return exx


@jit(nopython=True, fastmath=True)
def exx_kernel_rchol_complex(rchol, Ghalf):
    """Compute exchange contribution for complex rchol.

    Parameters
    ----------
    rchol : :class:`numpy.ndarray`
        Half-rotated cholesky.
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis

    Returns
    -------
    exx : :class:`numpy.ndarray`
        exchange contribution for given green's function.
    """
    naux = rchol.shape[0]
    nwalkers = Ghalf.shape[0]
    nocc = Ghalf.shape[1]
    nbasis = Ghalf.shape[2]

    exx = 0 + 0j
    GhalfT = Ghalf.T.copy()
    # Fix this with gpu env
    for jx in range(naux):
        rmi = rchol[jx].reshape((nocc, nbasis))
        T = rmi.dot(GhalfT)
        exx += numpy.dot(T.ravel(), T.T.ravel())
    return exx


def half_rotated_cholesky_jk(system, Ghalfa, Ghalfb, trial):
    """Compute exchange and coulomb contributions via jitted kernels.

    Parameters
    ----------
    system : :class:`Generic`
        System information for Generic.
    Ghalfa : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis
    Ghalfa : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nbeta x nbasis
    trial : ipie trial object
        Trial wavefunction

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        Coulomb energy.
    exx : :class:`numpy.ndarray`
        Exchange energy.
    """

    rchola = trial._rchola
    rcholb = trial._rcholb

    # Element wise multiplication.
    nalpha, nbeta = system.nup, system.ndown
    nbasis = Ghalfa.shape[-1]
    if rchola is not None:
        naux = rchola.shape[0]
    if xp.isrealobj(rchola) and xp.isrealobj(rcholb):
        Xa = rchola.dot(Ghalfa.real.ravel()) + 1.0j * rchola.dot(Ghalfa.imag.ravel())
        Xb = rcholb.dot(Ghalfb.real.ravel()) + 1.0j * rcholb.dot(Ghalfb.imag.ravel())
    else:
        Xa = rchola.dot(Ghalfa.ravel())
        Xb = rcholb.dot(Ghalfb.ravel())

    ecoul = xp.dot(Xa, Xa)
    ecoul += xp.dot(Xb, Xb)
    ecoul += 2 * xp.dot(Xa, Xb)
    exx = 0.0j  # we will iterate over cholesky index to update Ex energy for alpha and beta
    if xp.isrealobj(rchola) and xp.isrealobj(rcholb):
        exx = exx_kernel_rchol_real(rchola, Ghalfa) + exx_kernel_rchol_real(
            rcholb, Ghalfb
        )
    else:
        exx = exx_kernel_rchol_complex(rchola, Ghalfa) + exx_kernel_rchol_complex(
            rcholb, Ghalfb
        )

    synchronize()
    return 0.5 * ecoul, -0.5 * exx  # JK energy


# FDM: Mark for deletion
def core_contribution(system, Gcore):
    hc_a = numpy.einsum("pqrs,pq->rs", system.h2e, Gcore[0]) - 0.5 * numpy.einsum(
        "prsq,pq->rs", system.h2e, Gcore[0]
    )
    hc_b = numpy.einsum("pqrs,pq->rs", system.h2e, Gcore[1]) - 0.5 * numpy.einsum(
        "prsq,pq->rs", system.h2e, Gcore[1]
    )
    return (hc_a, hc_b)


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


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
#          Joonho Lee
#

import functools
import time

import numpy
import scipy.linalg


def minor_mask(A, i, j):
    r"""computing matrix minor, i-th row and j-th column removed"""
    # return numpy.delete(numpy.delete(A,i,axis=0), j, axis=1)

    mask = numpy.ones_like(A, dtype=bool)
    mask[i, :] = False
    mask[:, j] = False

    minor = A[mask].reshape(A.shape[0] - 1, A.shape[1] - 1)

    del mask

    return minor


def minor_mask4(A, i, j, k, l):
    r"""computing matrix minor, i-th and k-th rows and j-th and l-th column removed"""

    ncols = A.shape[1]

    if ncols - 2 <= 0:
        return 1.0

    mask = numpy.ones_like(A, dtype=bool)
    mask[i, :] = False
    mask[:, j] = False
    mask[k, :] = False
    mask[:, l] = False

    minor = A[mask].reshape(A.shape[0] - 2, A.shape[1] - 2)

    del mask

    return minor


def sherman_morrison(Ainv, u, vt):
    r"""Sherman-Morrison update of a matrix inverse:

    .. math::
        (A + u \otimes v)^{-1} = A^{-1} - \frac{A^{-1}u v^{T} A^{-1}}
                                               {1+v^{T}A^{-1} u}

    Parameters
    ----------
    Ainv : numpy.ndarray
        Matrix inverse of A to be updated.
    u : numpy.array
        column vector
    vt : numpy.array
        transpose of row vector

    Returns
    -------
    Ainv : numpy.ndarray
        Updated matrix inverse.
    """

    return Ainv - (Ainv.dot(numpy.outer(u, vt)).dot(Ainv)) / (1.0 + vt.dot(Ainv).dot(u))


def diagonalise_sorted(H):
    """Diagonalise Hermitian matrix H and return sorted eigenvalues and vectors.

    Eigenvalues are sorted as e_1 < e_2 < .... < e_N, where H is an NxN
    Hermitian matrix.

    Parameters
    ----------
    H : :class:`numpy.ndarray`
        Hamiltonian matrix to be diagonalised.

    Returns
    -------
    eigs : :class:`numpy.array`
        Sorted eigenvalues
    eigv :  :class:`numpy.array`
        Sorted eigenvectors (same sorting as eigenvalues).
    """

    (eigs, eigv) = scipy.linalg.eigh(H)
    idx = eigs.argsort()
    eigs = eigs[idx]
    eigv = eigv[:, idx]

    return (eigs, eigv)


def regularise_matrix_inverse(A, cutoff=1e-10):
    """Perform inverse of singular matrix.

    First compute SVD of input matrix then add a tuneable cutoff which washes
    out elements whose singular values are close to zero.

    Parameters
    ----------
    A : class:`numpy.array`
        Input matrix.
    cutoff : float
        Cutoff parameter.

    Returns
    -------
    B : class:`numpy.array`
        Regularised matrix inverse (pseudo-inverse).
    """
    (U, D, V) = scipy.linalg.svd(A)
    D = D / (cutoff**2.0 + D**2.0)
    return (V.conj().T).dot(numpy.diag(D)).dot(U.conj().T)


def reortho(A):
    """Reorthogonalise a MxN matrix A.

    Performs a QR decomposition of A. Note that for consistency elsewhere we
    want to preserve detR > 0 which is not guaranteed. We thus factor the signs
    of the diagonal of R into Q.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        MxN matrix.

    Returns
    -------
    Q : :class:`numpy.ndarray`
        Orthogonal matrix. A = QR.
    detR : float
        Determinant of upper triangular matrix (R) from QR decomposition.
    """
    (Q, R) = scipy.linalg.qr(A, mode="economic")
    signs = numpy.diag(numpy.sign(numpy.diag(R)))
    Q = Q.dot(signs)
    detR = scipy.linalg.det(signs.dot(R))
    return (Q, detR)


def overlap(A, B):
    S = numpy.dot(A.conj().T, B)
    return S


def modified_cholesky(M, tol=1e-6, verbose=True, cmax=20):
    """Modified cholesky decomposition of matrix.

    See, e.g. [Motta17]_

    Parameters
    ----------
    M : :class:`numpy.ndarray`
        Positive semi-definite, symmetric matrix.
    tol : float
        Accuracy desired.
    verbose : bool
        If true print out convergence progress.

    Returns
    -------
    chol_vecs : :class:`numpy.ndarray`
        Matrix of cholesky vectors.
    """
    # matrix of residuals.
    assert len(M.shape) == 2
    delta = numpy.copy(M.diagonal())
    nchol_max = int(cmax * M.shape[0] ** 0.5)
    # index of largest diagonal element of residual matrix.
    nu = numpy.argmax(numpy.abs(delta))
    delta_max = delta[nu]
    if verbose:
        print("# max number of cholesky vectors = %d" % nchol_max)
        print("# iteration %d: delta_max = %f" % (0, delta_max.real))
    # Store for current approximation to input matrix.
    Mapprox = numpy.zeros(M.shape[0], dtype=M.dtype)
    chol_vecs = numpy.zeros((nchol_max, M.shape[0]), dtype=M.dtype)
    nchol = 0
    chol_vecs[0] = numpy.copy(M[:, nu]) / delta_max**0.5
    while abs(delta_max) > tol:
        # Update cholesky vector
        start = time.time()
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol].conj()
        delta = M.diagonal() - Mapprox
        nu = numpy.argmax(numpy.abs(delta))
        delta_max = numpy.abs(delta[nu])
        nchol += 1
        Munu0 = numpy.dot(chol_vecs[:nchol, nu].conj(), chol_vecs[:nchol, :])
        chol_vecs[nchol] = (M[:, nu] - Munu0) / (delta_max) ** 0.5
        if verbose:
            step_time = time.time() - start
            info = (nchol, delta_max, step_time)
            print("# iteration %d: delta_max = %13.8e: time = %13.8e" % info)

    return numpy.array(chol_vecs[:nchol])


def exponentiate_matrix(M, order=6):
    """Taylor series approximation for matrix exponential"""
    T = numpy.copy(M)
    EXPM = numpy.identity(M.shape[0], dtype=M.dtype)
    for n in range(1, order + 1):
        EXPM += T
        T = M.dot(T) / (n + 1)
    return EXPM


def molecular_orbitals_rhf(fock, AORot):
    fock_ortho = numpy.dot(AORot.conj().T, numpy.dot(fock, AORot))
    mo_energies, mo_orbs = scipy.linalg.eigh(fock_ortho)
    return (mo_energies, mo_orbs)


def molecular_orbitals_uhf(fock, AORot):
    mo_energies = numpy.zeros((2, fock.shape[-1]))
    mo_orbs = numpy.zeros((2, fock.shape[-1], fock.shape[-1]))
    fock_ortho = numpy.dot(AORot.conj().T, numpy.dot(fock[0], AORot))
    (mo_energies[0], mo_orbs[0]) = scipy.linalg.eigh(fock_ortho)
    fock_ortho = numpy.dot(AORot.conj().T, numpy.dot(fock[1], AORot))
    (mo_energies[1], mo_orbs[1]) = scipy.linalg.eigh(fock_ortho)
    return (mo_energies, mo_orbs)


def get_orthoAO(S, LINDEP_CUTOFF=1e-14):
    sdiag, Us = numpy.linalg.eigh(S)
    X = Us[:, sdiag > LINDEP_CUTOFF] / numpy.sqrt(sdiag[sdiag > LINDEP_CUTOFF])
    return X


def get_ortho_ao_mod(S, LINDEP_CUTOFF=1e-14, verbose=False):
    sdiag, Us = numpy.linalg.eigh(S)
    if verbose:
        print("sdiag = {}".format(sdiag))
    sdiag[sdiag < LINDEP_CUTOFF] = 0.0
    keep = sdiag > LINDEP_CUTOFF
    X = Us[:, keep] / numpy.sqrt(sdiag[keep])
    Smod = Us[:, keep].dot(numpy.diag(sdiag[keep])).dot(Us[:, keep].T.conj())
    return Smod, X

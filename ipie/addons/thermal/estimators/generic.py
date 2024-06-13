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
# Authors: Fionn Malone <fmalone@google.com>
#          Joonho Lee
#

import plum
import numpy
from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol


@plum.dispatch
def local_energy_generic_cholesky(hamiltonian: GenericRealChol, P):
    r"""Calculate local for generic two-body hamiltonian.

    This uses the cholesky decomposed two-electron integrals.

    Parameters
    ----------
    hamiltonian : :class:`Generic`
        ab-initio hamiltonian information
    P : :class:`numpy.ndarray`
        Walker's density matrix.

    Returns
    -------
    (E, T, V): tuple
        Local, kinetic and potential energies.
    """
    # Element wise multiplication.
    e1b = numpy.sum(hamiltonian.H1[0] * P[0]) + numpy.sum(hamiltonian.H1[1] * P[1])
    nbasis = hamiltonian.nbasis
    nchol = hamiltonian.nchol
    Pa, Pb = P[0], P[1]

    # Ecoul.
    Xa = hamiltonian.chol.T.dot(Pa.real.ravel()) + 1.0j * hamiltonian.chol.T.dot(Pa.imag.ravel())
    Xb = hamiltonian.chol.T.dot(Pb.real.ravel()) + 1.0j * hamiltonian.chol.T.dot(Pb.imag.ravel())
    X = Xa + Xb
    ecoul = 0.5 * numpy.dot(X, X)

    # Ex.
    PaT = Pa.T.copy()
    PbT = Pb.T.copy()
    T = numpy.zeros((nbasis, nbasis), dtype=numpy.complex128)
    exx = 0.0j  # we will iterate over cholesky index to update Ex energy for alpha and beta

    for x in range(nchol):  # Write a numba function that calls BLAS for this.
        Lmn = hamiltonian.chol[:, x].reshape((nbasis, nbasis))
        T[:, :].real = PaT.real.dot(Lmn)
        T[:, :].imag = PaT.imag.dot(Lmn)
        exx += numpy.trace(T.dot(T))
        T[:, :].real = PbT.real.dot(Lmn)
        T[:, :].imag = PbT.imag.dot(Lmn)
        exx += numpy.trace(T.dot(T))

    exx *= 0.5
    e2b = ecoul - exx
    return (e1b + e2b + hamiltonian.ecore, e1b + hamiltonian.ecore, e2b)


@plum.dispatch
def local_energy_generic_cholesky(hamiltonian: GenericComplexChol, P):
    r"""Calculate local for generic two-body hamiltonian.

    This uses the cholesky decomposed two-electron integrals.

    Parameters
    ----------
    hamiltonian : :class:`Generic`
        ab-initio hamiltonian information
    P : :class:`numpy.ndarray`
        Walker's density matrix.

    Returns
    -------
    (E, T, V): tuple
        Local, kinetic and potential energies.
    """
    # Element wise multiplication.
    e1b = numpy.sum(hamiltonian.H1[0] * P[0]) + numpy.sum(hamiltonian.H1[1] * P[1])
    nbasis = hamiltonian.nbasis
    nchol = hamiltonian.nchol
    Pa, Pb = P[0], P[1]

    # Ecoul.
    XAa = hamiltonian.A.T.dot(Pa.ravel())
    XAb = hamiltonian.A.T.dot(Pb.ravel())
    XA = XAa + XAb

    XBa = hamiltonian.B.T.dot(Pa.ravel())
    XBb = hamiltonian.B.T.dot(Pb.ravel())
    XB = XBa + XBb

    ecoul = 0.5 * (numpy.dot(XA, XA) + numpy.dot(XB, XB))

    # Ex.
    PaT = Pa.T.copy()
    PbT = Pb.T.copy()
    TA = numpy.zeros((nbasis, nbasis), dtype=numpy.complex128)
    TB = numpy.zeros((nbasis, nbasis), dtype=numpy.complex128)
    exx = 0.0j  # we will iterate over cholesky index to update Ex energy for alpha and beta

    for x in range(nchol):  # write a cython function that calls blas for this.
        Amn = hamiltonian.A[:, x].reshape((nbasis, nbasis))
        Bmn = hamiltonian.B[:, x].reshape((nbasis, nbasis))
        TA[:, :] = PaT.dot(Amn)
        TB[:, :] = PaT.dot(Bmn)
        exx += numpy.trace(TA.dot(TA)) + numpy.trace(TB.dot(TB))

        TA[:, :] = PbT.dot(Amn)
        TB[:, :] = PbT.dot(Bmn)
        exx += numpy.trace(TA.dot(TA)) + numpy.trace(TB.dot(TB))

    exx *= 0.5
    e2b = ecoul - exx
    return (e1b + e2b + hamiltonian.ecore, e1b + hamiltonian.ecore, e2b)


def fock_generic(hamiltonian, P):
    nbasis = hamiltonian.nbasis
    nchol = hamiltonian.nchol
    hs_pot = hamiltonian.chol.T.reshape(nchol, nbasis, nbasis)
    mf_shift = 1j * numpy.einsum("lpq,spq->l", hs_pot, P)
    VMF = 1j * numpy.einsum("lpq,l->pq", hs_pot, mf_shift)
    return hamiltonian.h1e_mod - VMF

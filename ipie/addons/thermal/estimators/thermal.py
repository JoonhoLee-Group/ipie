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

import numpy
import scipy.linalg


def one_rdm_from_G(G):
    r"""Compute one-particle reduced density matrix from Green's function.

    .. math::
        rho_{ij} = \langle c_{i}^{\dagger} c_{j} \rangle \\
                 = 1 - G_{ji}
    Parameters
    ----------
    G : :class:`numpy.ndarray`
        Thermal Green's function.

    Returns
    -------
    P : :class:`numpy.ndarray`
        Thermal 1RDM.
    """
    I = numpy.identity(G.shape[-1])
    return numpy.array([I - G[0].T, I - G[1].T], dtype=numpy.complex128)


def one_rdm_stable(BT, num_slices):
    nbasis = BT.shape[-1]
    G = []
    for spin in [0, 1]:
        # Need to construct the product A(l) = B_l B_{l-1}..B_L...B_{l+1} in
        # stable way. Iteratively construct column pivoted QR decompositions
        # (A = QDT) starting from the rightmost (product of) propagator(s).
        (Q1, R1, P1) = scipy.linalg.qr(BT[spin], pivoting=True, check_finite=False)
        # Form D matrices
        D1 = numpy.diag(R1.diagonal())
        D1inv = numpy.diag(1.0 / R1.diagonal())
        T1 = numpy.einsum("ii,ij->ij", D1inv, R1)
        # permute them
        T1[:, P1] = T1[:, range(nbasis)]

        for i in range(0, num_slices - 1):
            C2 = numpy.dot(numpy.dot(BT[spin], Q1), D1)
            (Q1, R1, P1) = scipy.linalg.qr(C2, pivoting=True, check_finite=False)
            # Compute D matrices
            D1inv = numpy.diag(1.0 / R1.diagonal())
            D1 = numpy.diag(R1.diagonal())
            tmp = numpy.einsum("ii,ij->ij", D1inv, R1)
            tmp[:, P1] = tmp[:, range(nbasis)]
            T1 = numpy.dot(tmp, T1)
        # G^{-1} = 1+A = 1+QDT = Q (Q^{-1}T^{-1}+D) T
        # Write D = Db^{-1} Ds
        # Then G^{-1} = Q Db^{-1}(Db Q^{-1}T^{-1}+Ds) T
        Db = numpy.zeros(BT[spin].shape, BT[spin].dtype)
        Ds = numpy.zeros(BT[spin].shape, BT[spin].dtype)
        for i in range(Db.shape[0]):
            absDlcr = abs(Db[i, i])
            if absDlcr > 1.0:
                Db[i, i] = 1.0 / absDlcr
                Ds[i, i] = numpy.sign(D1[i, i])
            else:
                Db[i, i] = 1.0
                Ds[i, i] = D1[i, i]

        T1inv = scipy.linalg.inv(T1, check_finite=False)
        # C = (Db Q^{-1}T^{-1}+Ds)
        C = numpy.dot(numpy.einsum("ii,ij->ij", Db, Q1.conj().T), T1inv) + Ds
        Cinv = scipy.linalg.inv(C, check_finite=False)

        # Then G = T^{-1} C^{-1} Db Q^{-1}
        # Q is unitary.
        G.append(numpy.dot(numpy.dot(T1inv, Cinv), numpy.einsum("ii,ij->ij", Db, Q1.conj().T)))
    return one_rdm_from_G(numpy.array(G))

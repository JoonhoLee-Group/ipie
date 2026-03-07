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
# Authors: Jinghong Zhang <jinghongzhang@fas.harvard.edu>
#

import numpy
import pytest

from ipie.config import config
from ipie.hamiltonians.kpt_hamiltonian import KptComplexCholSymm
from ipie.hamiltonians.kpt_isdf_hamiltonian import KptISDF
from ipie.trial_wavefunction.single_det_kpt import KptSingleDet
from ipie.utils.backend import arraylib as xp
from ipie.utils.testing import gen_random_test_input_kpt_isdf, shaped_normal


@pytest.mark.gpu
def test_isdf_against_chol():
    if not config.get_option("use_gpu"):
        pytest.skip("Requires GPU backend. Set IPIE_USE_GPU=1 to run this test.")

    from cuquantum.bindings import cutensornet
    from ipie.propagation.phaseless_kpt import (
        PhaselessKptISDF,
        construct_VHS_batch,
        construct_VHS_symm_gpu,
    )

    kmesh = (2, 1, 1)
    nk = kmesh[0] * kmesh[1] * kmesh[2]
    nbasis = 6
    naux = 3 * nbasis
    nalpha, nbeta = (2, 2)
    nwalkers = 2
    dt = 0.005

    h1e, MPQ, cgto, wavefunction, kpts = gen_random_test_input_kpt_isdf(
        kmesh, nbasis, (nalpha, nbeta), naux, seed=31
    )

    trial = KptSingleDet(wavefunction, nk, (nalpha, nbeta), nbasis)
    nq = MPQ.shape[0]
    cholM = shaped_normal((nq, cgto.shape[1], naux), cmplx=True, seed=37)
    ham_isdf = KptISDF(
        h1e=numpy.array([h1e, h1e], dtype=numpy.complex128),
        MPQ=numpy.array(MPQ, dtype=numpy.complex128),
        cholM=numpy.array(cholM, dtype=numpy.complex128),
        cgto=numpy.array(cgto, dtype=numpy.complex128),
        kpts=kpts,
        h1e_mod=numpy.zeros((2, nk, nbasis, nbasis), dtype=numpy.complex128),
    )

    chol = numpy.zeros((naux, nk, nbasis, ham_isdf.unique_nk, nbasis), dtype=numpy.complex128)
    unique_qs = numpy.concatenate((ham_isdf.Sset, ham_isdf.Qplus))
    for iq, iq_real in enumerate(unique_qs):
        for ik in range(nk):
            ikpq = ham_isdf.ikpq_mat[iq_real, ik]
            chol[:, ik, :, iq, :] = numpy.einsum(
                "Pp,Pr,Pg->gpr",
                cgto[ik].conj(),
                cgto[ikpq],
                cholM[iq],
                optimize=True,
            )

    ham_chol = KptComplexCholSymm(
        h1e=numpy.array([h1e, h1e], dtype=numpy.complex128),
        chol=chol,
        kpts=kpts,
    )

    prop_isdf = PhaselessKptISDF(dt)

    xshifted = shaped_normal((2, nwalkers, naux, ham_isdf.unique_nk), cmplx=True, seed=41)
    xshifted = xp.asarray(xshifted)

    ham_isdf.cast_to_cupy()
    ham_chol.cast_to_cupy()

    Lx, Lconjx = prop_isdf.contract_cholM_xshifted(ham_isdf, xshifted)
    handle = cutensornet.create()
    VHS_isdf = construct_VHS_batch(
        ham_isdf.cgto,
        Lx,
        Lconjx,
        ham_isdf.ikpq_mat,
        ham_isdf.ikmq_mat,
        ham_isdf.unique_k,
        handle,
    )
    cutensornet.destroy(handle)

    VHS_chol = construct_VHS_symm_gpu(
        ham_chol.chol,
        prop_isdf.sqrt_dt,
        xshifted,
        ham_chol.nk,
        ham_chol.nbasis,
        nwalkers,
        ham_chol.ikpq_mat,
        ham_chol.Sset,
        ham_chol.Qplus,
    )

    numpy.testing.assert_allclose(xp.asnumpy(VHS_isdf), xp.asnumpy(VHS_chol), atol=1e-8)

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

import numpy as np
import pytest

from ipie.hamiltonians.kpt_hamiltonian import (
    KptComplexChol,
    KptComplexCholSymm,
    KptISDF,
    construct_h1e_mod,
    construct_h1e_mod_isdf,
    construct_h1e_mod_symm,
    construct_kmq,
    construct_kpq,
    construct_mq,
)
from ipie.utils.kpt_conv import (
    find_inverted_index_batched,
    find_Qplus,
    find_self_inverse_set,
    find_translated_index_batched,
    generate_MPmesh_3d
)
from ipie.utils.testing import shaped_normal, _small_kpts_trivial_Sset, _nontrivial_Sset_kpts, expand_chol_symm_to_full

KPTS_CASES = [
    (_small_kpts_trivial_Sset, False),
    (_nontrivial_Sset_kpts, True),
]

@pytest.mark.unit
def test_construct_k_index_maps_match_kpt_conv_reference():
    kpts = _small_kpts_trivial_Sset()
    nk = kpts.shape[0]

    ikpq = construct_kpq(kpts)
    ikmq = construct_kmq(kpts)
    imq = construct_mq(kpts)

    assert ikpq.shape == (nk, nk)
    assert ikmq.shape == (nk, nk)
    assert imq.shape == (nk,)
    assert ikpq.dtype == np.int64
    assert ikmq.dtype == np.int64
    assert imq.dtype == np.int64

    # construct_kpq/construct_kmq build rows by calling find_translated_index_batched(kpts, ±kpts[iq])
    for iq in range(nk):
        ref_kpq_row = find_translated_index_batched(kpts, kpts[iq])
        ref_kmq_row = find_translated_index_batched(kpts, -kpts[iq])
        np.testing.assert_array_equal(ikpq[iq], ref_kpq_row)
        np.testing.assert_array_equal(ikmq[iq], ref_kmq_row)

    ref_imq = find_inverted_index_batched(kpts)
    np.testing.assert_array_equal(imq, ref_imq)


@pytest.mark.unit
def test_construct_h1e_mod_matches_reference_expression_full_q():
    kpts = _small_kpts_trivial_Sset()
    nk = kpts.shape[0]
    nbasis = 3
    nchol = 4

    chol = shaped_normal((nchol, nk, nbasis, nk, nbasis), cmplx=True, seed=11)
    h1e = shaped_normal((2, nk, nbasis, nbasis), cmplx=True, seed=13)

    ikpq_mat = construct_kpq(kpts)
    imq_vec = construct_mq(kpts)

    out = np.zeros_like(h1e)
    construct_h1e_mod(chol, h1e, ikpq_mat, imq_vec, out)

    # Reference implementation mirroring construct_h1e_mod
    v0 = np.zeros((nk, nbasis, nbasis), dtype=np.complex128)
    for ik in range(nk):
        for iq in range(nk):
            v0[ik] += 0.5 * np.einsum(
                "gpr,gqr->pq",
                chol[:, ik, :, iq, :],
                chol[:, ik, :, iq, :].conj(),
                optimize=True,
            )

    ref = np.zeros_like(h1e)
    ref[0] = h1e[0] - v0
    ref[1] = h1e[1] - v0

    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("kpts_fn, expect_nontrivial_sset", KPTS_CASES)
def test_kptcomplexcholsymm_builds_h1e_mod_consistently(kpts_fn, expect_nontrivial_sset):
    kpts = kpts_fn()
    nk = kpts.shape[0]
    nbasis = 2
    nchol = 3

    Sset = find_self_inverse_set(kpts)
    Qplus = find_Qplus(kpts)
    if expect_nontrivial_sset:
        assert len(Sset) > 1

    unique_nk = len(Sset) + len(Qplus)

    chol = shaped_normal((nchol, nk, nbasis, unique_nk, nbasis), cmplx=True, seed=41)
    h1e = shaped_normal((2, nk, nbasis, nbasis), cmplx=True, seed=43)

    ham = KptComplexCholSymm(h1e=h1e, chol=chol, kpts=kpts, ecore=0.0, verbose=False)

    # Direct reference call
    ikmq_mat = construct_kmq(kpts)
    ref = np.zeros_like(h1e)
    construct_h1e_mod_symm(chol, h1e, ikmq_mat, Sset, Qplus, ref)

    np.testing.assert_allclose(np.asarray(ham.h1e_mod), ref, rtol=1e-12, atol=1e-12)
    assert ham.nk == nk
    assert ham.nbasis == nbasis
    assert ham.nchol == nchol
    assert ham.unique_nk == unique_nk


@pytest.mark.unit
@pytest.mark.parametrize("kpts_fn, expect_nontrivial_sset", KPTS_CASES)
def test_construct_h1e_mod_symm_matches_nosymm(kpts_fn, expect_nontrivial_sset):
    kpts = kpts_fn()
    nk = kpts.shape[0]
    nbasis = 3
    nchol = 8

    Sset = find_self_inverse_set(kpts)
    Qplus = find_Qplus(kpts)

    if expect_nontrivial_sset:
        assert len(Sset) > 1, f"Expected nontrivial Sset, got len(Sset)={len(Sset)}"
    else:
        assert len(Sset) >= 1

    unique_q = np.concatenate((Sset, Qplus))
    unique_nk = len(unique_q)
    assert unique_nk == len(Sset) + len(Qplus)

    ikpq_mat = construct_kpq(kpts)
    imq_vec = construct_mq(kpts)
    ikmq_mat = construct_kmq(kpts)

    chol = shaped_normal((nchol, nk, nbasis, unique_nk, nbasis), cmplx=True, seed=31)
    # Expand to full chol: (nchol, nk, nbasis, nk, nbasis)
    chol_full = expand_chol_symm_to_full(chol, kpts, Sset, Qplus, ikpq_mat, imq_vec)
    np.testing.assert_allclose(chol_full[:, :, :, unique_q, :], chol, rtol=1e-12, atol=1e-12)

    # Compare h1e_mod from symm vs from full
    h1e = shaped_normal((2, nk, nbasis, nbasis), cmplx=True, seed=33)
    out_symm = np.zeros_like(h1e)
    construct_h1e_mod_symm(chol, h1e, ikmq_mat, Sset, Qplus, out_symm)
    out_full = np.zeros_like(h1e)
    construct_h1e_mod(chol_full, h1e, ikpq_mat, imq_vec, out_full)
    np.testing.assert_allclose(out_symm, out_full, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
def test_kptcomplexcholsymm_builds_h1e_mod_consistently():
    kpts = _small_kpts_trivial_Sset()
    nk = kpts.shape[0]
    nbasis = 2
    nchol = 3

    Sset = find_self_inverse_set(kpts)
    Qplus = find_Qplus(kpts)
    unique_nk = len(Sset) + len(Qplus)

    chol = shaped_normal((nchol, nk, nbasis, unique_nk, nbasis), cmplx=True, seed=41)
    h1e = shaped_normal((2, nk, nbasis, nbasis), cmplx=True, seed=43)

    ham = KptComplexCholSymm(h1e=h1e, chol=chol, kpts=kpts, ecore=0.0, verbose=False)

    # Direct reference call
    ikmq_mat = construct_kmq(kpts)
    ref = np.zeros_like(h1e)
    construct_h1e_mod_symm(chol, h1e, ikmq_mat, Sset, Qplus, ref)

    np.testing.assert_allclose(np.asarray(ham.h1e_mod), ref, rtol=1e-12, atol=1e-12)
    assert ham.nk == nk
    assert ham.nbasis == nbasis
    assert ham.nchol == nchol
    assert ham.unique_nk == unique_nk


@pytest.mark.unit
@pytest.mark.parametrize("kpts_fn, expect_nontrivial_sset", KPTS_CASES)
def test_construct_h1e_mod_isdf_matches_reference_expression(kpts_fn, expect_nontrivial_sset):
    kpts = kpts_fn()
    nk = kpts.shape[0]
    nbasis = 2
    nisdf = 4

    Sset = find_self_inverse_set(kpts)
    Qplus = find_Qplus(kpts)
    if expect_nontrivial_sset:
        assert len(Sset) > 1

    unique_nk = len(Sset) + len(Qplus)

    MPQ = shaped_normal((unique_nk, nisdf, nisdf), cmplx=True, seed=51)
    cgto = shaped_normal((nk, nisdf, nbasis), cmplx=True, seed=53)
    h1e = shaped_normal((2, nk, nbasis, nbasis), cmplx=True, seed=55)

    ikpq_mat = construct_kpq(kpts)
    ikmq_mat = construct_kmq(kpts)

    out = np.zeros_like(h1e)
    construct_h1e_mod_isdf(MPQ, cgto, h1e, ikpq_mat, ikmq_mat, Sset, Qplus, out)

    v0 = np.zeros((nk, nbasis, nbasis), dtype=np.complex128)

    for iq in range(len(Sset)):
        iq_real = int(Sset[iq])
        MPQ_iq = MPQ[iq]
        for ik in range(nk):
            ikpq = int(ikpq_mat[iq_real, ik])
            cgto_ikpq = cgto[ikpq]
            cgto_PQ = cgto_ikpq @ cgto_ikpq.T.conj()
            cgtoM = MPQ_iq * cgto_PQ
            v0[ik] += 0.5 * (cgto[ik].conj().T @ cgtoM @ cgto[ik])

    for iq in range(len(Sset), len(Sset) + len(Qplus)):
        iq_real = int(Qplus[iq - len(Sset)])
        MPQ_iq = MPQ[iq]
        for ik in range(nk):
            ikpq = int(ikpq_mat[iq_real, ik])
            cgto_ikpq = cgto[ikpq]
            ikmq = int(ikmq_mat[iq_real, ik])
            cgto_ikmq = cgto[ikmq]

            cgto_PQ = cgto_ikpq @ cgto_ikpq.T.conj()
            cgtoM = MPQ_iq * cgto_PQ
            v0[ik] += 0.5 * (cgto[ik].conj().T @ cgtoM @ cgto[ik])

            cgto_PQ = cgto_ikmq @ cgto_ikmq.T.conj()
            cgtoM = MPQ_iq.conj() * cgto_PQ
            v0[ik] += 0.5 * (cgto[ik].conj().T @ cgtoM @ cgto[ik])

    ref = np.zeros_like(h1e)
    ref[0] = h1e[0] - v0
    ref[1] = h1e[1] - v0

    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.unit
def test_kptisdf_uses_provided_h1e_mod():
    kpts = _small_kpts_trivial_Sset()
    nk = kpts.shape[0]
    nbasis = 2
    nisdf = 3
    nchol = 5

    h1e = shaped_normal((2, nk, nbasis, nbasis), cmplx=True, seed=61)
    MPQ = shaped_normal((2, nisdf, nisdf), cmplx=True, seed=63)
    cholM = shaped_normal((2, nisdf, nchol), cmplx=True, seed=65)  # [q, P, gamma]
    cgto = shaped_normal((nk, nisdf, nbasis), cmplx=True, seed=67)

    provided = shaped_normal(h1e.shape, cmplx=True, seed=69)

    ham = KptISDF(
        h1e=h1e,
        MPQ=MPQ,
        cholM=cholM,
        cgto=cgto,
        kpts=kpts,
        ecore=0.0,
        verbose=False,
        h1e_mod=provided,
    )

    assert ham.h1e_mod is provided
    assert ham.nk == nk
    assert ham.nbasis == nbasis
    assert ham.nisdf == nisdf
    assert ham.nchol == nchol

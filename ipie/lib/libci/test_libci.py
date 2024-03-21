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
# Author: Fionn Malone <fionn.malone@gmail.com>
#

import itertools
import math
from typing import List, Tuple

import numpy as np
import pytest

from ipie.estimators.local_energy import variational_energy_ortho_det
from ipie.hamiltonians.generic import GenericRealChol
from ipie.lib.libci import libci
from ipie.systems import Generic
from ipie.utils.testing import get_random_phmsd_opt, build_random_ci_wfn


def get_perm(from_orb, to_orb, bra, ket):
    """Determine sign of permutation needed to align two determinants.

    Stolen from HANDE.
    """
    nmove = 0
    perm = 0
    for o in from_orb:
        io = np.where(ket == o)[0][0]
        perm += io - nmove
        nmove += 1
    nmove = 0
    for o in to_orb:
        io = np.where(bra == o)[0][0]
        perm += io - nmove
        nmove += 1
    return perm % 2 == 1


def map_orb(orb, nbasis):
    """Map spin orbital to spatial index."""
    if orb // nbasis == 0:
        s = 0
        ix = orb
    else:
        s = 1
        ix = orb - nbasis
    return ix, s


def build_one_rdm_ref(coeffs, occa, occb, nbasis):
    denom = np.sum(coeffs.conj() * coeffs)
    Pa = np.zeros((nbasis, nbasis), dtype=np.complex128)
    Pb = np.zeros((nbasis, nbasis), dtype=np.complex128)

    dets = [list(a) + [i + nbasis for i in c] for (a, c) in zip(occa, occb)]
    spin_occs = [np.sort(d) for d in dets]
    P = [Pa, Pb]

    def map_orb(orb, nbasis):
        """Map spin orbital to spatial index."""
        if orb // nbasis == 0:
            s = 0
            ix = orb
        else:
            s = 1
            ix = orb - nbasis
        return ix, s

    num_dets = len(coeffs)
    # print(dets)

    for idet in range(num_dets):
        det_ket = spin_occs[idet]
        # zero excitation case
        for iorb in range(len(det_ket)):
            ii, spin_ii = map_orb(det_ket[iorb], nbasis)
            # print(ii, spin_ii)
            P[spin_ii][ii, ii] += coeffs[idet].conj() * coeffs[idet]
        for jdet in range(idet + 1, num_dets):
            det_bra = spin_occs[jdet]
            # Note there was a bug in mainline ipie in this function.
            # There from_orb was det_bra - det_ket, it should be det_ket - det_bra to get the
            # orbital occupied in the ket (idet)
            from_orb = list(set(det_ket) - set(det_bra))
            to_orb = list(set(det_bra) - set(det_ket))
            nex = len(from_orb)
            # print(idet, jdet, nex, det_bra, det_ket)
            if nex > 1:
                continue
            elif nex == 1:
                perm = get_perm(from_orb, to_orb, det_bra, det_ket)
                if perm:
                    phase = -1
                else:
                    phase = 1
                ii, si = map_orb(from_orb[0], nbasis)
                aa, sa = map_orb(to_orb[0], nbasis)
                if si == sa:
                    # if si == 0:
                    # print(idet, jdet, det_ket, det_bra, from_orb, to_orb, perm)
                    P[si][aa, ii] += coeffs[jdet].conj() * coeffs[idet] * phase
                    P[si][ii, aa] += coeffs[jdet] * coeffs[idet].conj() * phase
    P[0] /= denom
    P[1] /= denom
    return P


@pytest.mark.unit
@pytest.mark.libci
def test_wavefunction():
    num_alpha = 13
    num_beta = 27
    num_det = 100
    num_spat = 67
    (coeff, occa, occb), _ = get_random_phmsd_opt(num_alpha, num_beta, num_spat, ndet=num_det)
    wfn = libci.Wavefunction(coeff, occa, occb, num_spat)
    assert np.isclose(wfn.norm(), np.dot(coeff.conj(), coeff) ** 0.5)
    assert wfn.num_dets == len(coeff)
    assert wfn.num_spatial == num_spat
    assert wfn.num_elec == num_alpha + num_beta


def _build_full_wavefunction(n_spat, n_el, num_det_for_trial=None):
    """Build the uniform superposition state within the particular spin subspace."""

    # https://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
    def next_bit_permutation(v: int) -> int:
        assert v > 0
        t = (v | (v - 1)) + 1
        w = t | ((((t & -t) // (v & -v)) >> 1) - 1)
        return w

    def get_set_bits(bitstring: int, n_spat: int) -> List[int]:
        occ = []
        for i in range(n_spat):
            if bitstring & (1 << i):
                occ.append(i)
        return occ

    def all_combs_spin(n_el: Tuple[int, int], spin: int):
        if n_el[spin] == 0:
            return [[]]
        n_det_spin = math.comb(n_spat, n_el[spin])
        bitstring = (1 << n_el[spin]) - 1
        occs = []
        for _ in range(n_det_spin):
            det = bitstring
            occs.append(get_set_bits(det, n_spat))
            bitstring = next_bit_permutation(bitstring)
        return occs

    occa = all_combs_spin(n_el, 0)
    occb = all_combs_spin(n_el, 1)
    n_det = math.comb(n_spat, n_el[0]) * math.comb(n_spat, n_el[1])

    all_dets = list(itertools.product(occa, occb))
    assert len(all_dets) == n_det
    if num_det_for_trial is not None:
        skip = n_det // num_det_for_trial
    else:
        skip = 1
    occa, occb = zip(*all_dets)
    occa = np.array(occa)[::skip]
    occb = np.array(occb)[::skip]

    coeffs = np.ones(len(occa)) / len(occa) ** 0.5
    return (coeffs, occa, occb)


@pytest.mark.parametrize(
    "num_spat, num_elec, num_det",
    ((9, 2, 2), (27, 2, 351), (79, 2, 23), (1023, 4, 100), (513, 67, 39)),
)
@pytest.mark.unit
@pytest.mark.libci
def test_one_rdm(num_spat, num_elec, num_det):
    num_alpha = np.random.randint(0, num_elec + 1)
    num_beta = num_elec - num_alpha
    (coeff, occa, occb), _ = get_random_phmsd_opt(num_alpha, num_beta, num_spat, ndet=num_det)
    import time

    wfn = libci.Wavefunction(coeff, occa, occb, num_spat)
    opdm = np.array(wfn.one_rdm()).reshape((2, num_spat, num_spat))
    opdm_ref = build_one_rdm_ref(coeff, occa, occb, num_spat)
    assert np.isclose(opdm[0].trace(), num_alpha, atol=1e-12)
    assert np.isclose(opdm[1].trace(), num_beta, atol=1e-12)
    assert np.allclose(opdm, opdm_ref)


# def _dedupe(wfn):
#     wfn_dedupe = {}
#     coeffs, occa, occb = wfn
#     for c, a, b in zip(*wfn):
#         wfn_dedupe[tuple([tuple(a), tuple(b)])] = c
#     coeffs = np.array(list(wfn_dedupe.values()))
#     occ_a = []
#     occ_b = []
#     for x in wfn_dedupe.keys():
#         occ_a.append(np.array(x[0]))
#         occ_b.append(np.array(x[1]))

#     return coeffs, occ_a, occ_b


@pytest.mark.parametrize(
    "num_spat, num_elec, num_det",
    ((23, 4, 100), (69, 4, 83), (139, 3, 37)),
)
@pytest.mark.unit
@pytest.mark.libci
def test_one_rdm_high_excit(num_spat, num_elec, num_det):
    num_alpha = np.random.randint(0, num_elec + 1)
    num_beta = num_elec - num_alpha
    (coeff, occa, occb) = build_random_ci_wfn(num_spat, (num_alpha, num_beta), num_det=num_det)
    wfn = libci.Wavefunction(coeff, occa, occb, num_spat)
    opdm = np.array(wfn.one_rdm()).reshape((2, num_spat, num_spat))
    opdm_ref = build_one_rdm_ref(coeff, occa, occb, num_spat)
    assert np.isclose(opdm[0].trace(), num_alpha, atol=1e-12)
    assert np.isclose(opdm[1].trace(), num_beta, atol=1e-12)
    assert np.allclose(opdm, opdm_ref)


def _build_test_system(num_spat, num_el):
    num_alpha, num_beta = num_el
    h1e = np.random.normal(size=(num_spat, num_spat))
    h1e = 0.5 * (h1e + h1e.T)
    assert np.allclose(h1e, h1e.conj().T)
    chol = np.random.normal(size=(num_spat**2, num_spat, num_spat))
    chol = 0.5 * (chol + np.transpose(chol, (0, 2, 1)))
    h2e = np.einsum("Xpq,Xrs->pqrs", chol, chol, optimize=True)
    e0 = np.random.normal()

    sys = Generic((num_alpha, num_beta))
    chol = chol.reshape((-1, num_spat * num_spat)).T.copy()
    ham_ipie = GenericRealChol(np.array([h1e, h1e]), chol, ecore=e0)
    return sys, ham_ipie, h2e


@pytest.mark.parametrize(
    "num_spat, num_alpha, num_beta",
    itertools.chain(*[itertools.product([n], np.arange(n), np.arange(n)) for n in range(2, 6)]),
)
@pytest.mark.unit
@pytest.mark.libci
def test_variational_energy_full_space(num_spat, num_alpha, num_beta):
    (coeff, occa, occb) = _build_full_wavefunction(num_spat, (num_alpha, num_beta))
    sys, ham_ipie, h2e = _build_test_system(num_spat, (num_alpha, num_beta))
    # for x, y, z in zip(*(coeff, occa, occb)):
    #     print(x, y, z)
    ham = libci.Hamiltonian(ham_ipie.H1[0], h2e, ham_ipie.ecore)
    wfn = libci.Wavefunction(coeff, occa, occb, num_spat)
    energy = wfn.energy(ham)
    dets = [list(a) + [i + num_spat for i in c] for (a, c) in zip(occa, occb)]
    spin_occs = [np.sort(d) for d in dets]
    energy_ref = variational_energy_ortho_det(sys, ham_ipie, spin_occs, coeff)
    opdm = np.array(wfn.one_rdm()).reshape((2, num_spat, num_spat))
    opdm_ref = build_one_rdm_ref(coeff, occa, occb, num_spat)
    msg = f"n_a = {num_alpha} n_b = {num_beta}"
    assert np.allclose(opdm, opdm_ref), msg
    assert np.isclose(
        np.einsum("pq,pq->", ham_ipie.H1[0], opdm[1] + opdm[0]) + ham_ipie.ecore, energy_ref[1]
    ), msg
    # # SC0 does not match matel from ipie function
    assert np.isclose(energy[1], energy_ref[1], atol=1e-12), msg
    assert np.isclose(energy[2], energy_ref[2], atol=1e-12), msg
    assert np.isclose(energy[0], energy_ref[0], atol=1e-12), msg


@pytest.mark.parametrize(
    "num_spat, num_elec, num_det",
    (
        (9, 4, 62),
        (14, 9, 250),
        (71, 7, 20),
    ),
)
@pytest.mark.unit
@pytest.mark.libci
def test_variational_energy(num_spat, num_elec, num_det):
    num_alpha = np.random.randint(0, num_elec + 1)
    num_beta = num_elec - num_alpha
    sys, ham_ipie, h2e = _build_test_system(num_spat, (num_alpha, num_beta))
    (coeff, occa, occb), _ = get_random_phmsd_opt(
        num_alpha, num_beta, num_spat, ndet=num_det, cmplx_coeffs=True
    )
    ham = libci.Hamiltonian(ham_ipie.H1[0], h2e, ham_ipie.ecore)
    import time

    wfn = libci.Wavefunction(coeff, occa, occb, num_spat)
    energy = wfn.energy(ham)
    dets = [list(a) + [i + num_spat for i in c] for (a, c) in zip(occa, occb)]
    spin_occs = [np.sort(d) for d in dets]
    energy_ref = variational_energy_ortho_det(sys, ham_ipie, spin_occs, coeff)
    opdm = np.array(wfn.one_rdm()).reshape((2, num_spat, num_spat))
    opdm_ref = build_one_rdm_ref(coeff, occa, occb, num_spat)
    assert np.allclose(opdm, opdm_ref)
    assert np.isclose(
        np.einsum("pq,pq->", ham_ipie.H1[0], opdm[1] + opdm[0]) + ham_ipie.ecore, energy_ref[1]
    )
    # SC0 does not match matel from ipie function
    assert np.isclose(energy[1], energy_ref[1], atol=1e-12)
    assert np.isclose(energy[2], energy_ref[2], atol=1e-12)
    assert np.isclose(energy[0], energy_ref[0], atol=1e-12)

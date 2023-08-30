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

import numpy as np
import pytest

from ipie.lib.libci import libci
from ipie.utils.testing import generate_hamiltonian, get_random_phmsd_opt


def convert_phase(occa, occb):
    """Convert phase from abab to aabb ordering.

    Parameters
    ----------
    occa : list
        list of alpha occupations.
    occb : list
        list of beta occupations.

    Returns
    -------
    phases : np.ndarray
        phase factors.
    """
    ndet = len(occa)
    phases = np.zeros(ndet)
    for i in range(ndet):
        doubles = list(set(occa[i]) & set(occb[i]))
        occa0 = np.array(occa[i])
        occb0 = np.array(occb[i])

        count = 0
        for ocb in occb0:
            passing_alpha = np.where(occa0 > ocb)[0]
            count += len(passing_alpha)

        phases[i] = (-1) ** count

    return phases


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
                    P[si][aa, ii] += coeffs[jdet].conj() * coeffs[idet] * phase
                    P[si][ii, aa] += coeffs[jdet] * coeffs[idet].conj() * phase
    P[0] /= denom
    P[1] /= denom
    return P


def map_orb(orb, nbasis):
    """Map spin orbital to spatial index."""
    if orb // nbasis == 0:
        s = 0
        ix = orb
    else:
        s = 1
        ix = orb - nbasis
    return ix, s


def slater_condon0(h1e, h2e, ecore, occs):
    e1b = 0.0
    e2b = 0.0
    e1b = ecore
    nbasis = h1e.shape[0]
    for i in range(len(occs)):
        ii, spin_ii = map_orb(occs[i], nbasis)
        # Todo: Update if H1 is ever spin dependent.
        e1b += h1e[ii, ii]
        for j in range(i + 1, len(occs)):
            jj, spin_jj = map_orb(occs[j], nbasis)
            e2b += h2e[ii, ii, jj, jj]
            if spin_ii == spin_jj:
                e2b -= h2e[ii, jj, jj, ii]
    hmatel = e1b + e2b
    return hmatel, e1b, e2b


def slater_condon1(h1e, h2e, occs, i, a):
    ii, si = i // 2, i % 2
    aa, sa = a // 2, a % 2
    e1b = h1e[ii, aa]
    e2b = 0
    nbasis = h1e.shape[0]
    for j, oj in enumerate(occs):
        # \sum_j <ij|aj> - <ij|ja>
        oj, soj = map_orb(oj, nbasis)
        if 2 * oj + soj != 2 * ii + si:
            e2b += h2e[ii, aa, oj, oj]
            if soj == si:
                e2b -= h2e[ii, oj, oj, aa]
    hmatel = e1b + e2b
    return hmatel, e1b, e2b


def slater_condon2(h2e, i, j, a, b):
    ii, si = i // 2, i % 2
    jj, sj = j // 2, j % 2
    aa, sa = a // 2, a % 2
    bb, sb = b // 2, b % 2
    hmatel = 0.0
    if si == sa:
        hmatel = h2e[ii, aa, jj, bb]
    if si == sb:
        hmatel -= h2e[ii, bb, jj, aa]
    return hmatel


@pytest.mark.parametrize(
    "num_spat, num_elec, num_det",
    ((13, 8, 9), (27, 2, 1_000), (79, 2, 23), (1023, 4, 100), (513, 67, 39)),
)
@pytest.mark.unit
def test_one_rdm(num_spat, num_elec, num_det):
    num_alpha = np.random.randint(0, num_elec + 1)
    num_beta = num_elec - num_alpha
    (coeff, occa, occb), _ = get_random_phmsd_opt(num_alpha, num_beta, num_spat, ndet=num_det)
    phases = convert_phase(occa, occb)
    opdm = libci.one_rdm(phases * coeff, occa, occb, num_spat)
    opdm_ref = build_one_rdm_ref(coeff, occa, occb, num_spat)
    assert np.allclose(opdm, opdm_ref)


@pytest.mark.parametrize("num_spat, num_elec", ((8, 4), (23, 12), (11, 1)))
@pytest.mark.unit
def test_slater_condon0(num_spat, num_elec):
    num_spat = 8
    num_elec = 4
    h1e, _, e0, h2e = generate_hamiltonian(num_spat, num_elec)
    occs = np.sort(np.random.choice(np.arange(2 * num_spat), size=num_elec, replace=False))
    occs_ipie = np.sort([o // 2 if o % 2 == 0 else o // 2 + num_spat for o in occs])
    ref = slater_condon0(h1e, h2e, e0, occs_ipie)
    test = libci.slater_condon0(h1e, h2e, e0, occs)
    assert np.allclose(ref, test)


@pytest.mark.parametrize("num_spat, num_elec", ((8, 4), (23, 12), (11, 1)))
@pytest.mark.unit
def test_slater_condon1(num_spat, num_elec):
    h1e, _, e0, h2e = generate_hamiltonian(num_spat, num_elec)
    occs = np.sort(np.random.choice(np.arange(2 * num_spat), size=num_elec, replace=False))
    i = np.random.choice(occs, size=1)[0]
    vir = np.array(list(set(np.arange(2 * num_spat)) - set(occs)))
    assert len(vir) == 2 * num_spat - len(occs)
    a = np.random.choice(vir, size=1)[0]
    occs_ipie = np.sort([o // 2 if o % 2 == 0 else o // 2 + num_spat for o in occs])
    ref = slater_condon1(h1e, h2e, occs_ipie, i, a)
    test = libci.slater_condon1(h1e, h2e, occs, i, a)
    assert np.allclose(ref, test)


@pytest.mark.parametrize("num_spat, num_elec", ((8, 4), (23, 12), (11, 3)))
@pytest.mark.unit
def test_slater_condon2(num_spat, num_elec):
    h1e, _, e0, h2e = generate_hamiltonian(num_spat, num_elec)
    occs = np.sort(np.random.choice(np.arange(2 * num_spat), size=num_elec, replace=False))
    i, j = np.random.choice(occs, size=2, replace=False)
    vir = np.array(list(set(np.arange(2 * num_spat)) - set(occs)))
    assert len(vir) == 2 * num_spat - len(occs)
    a, b = np.random.choice(vir, size=2)
    ref = slater_condon2(h2e, i, j, a, b)
    test = libci.slater_condon2(h2e, i, j, a, b)
    assert np.isclose(ref, test[0])


@pytest.mark.unit
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


@pytest.mark.unit
def test_hamiltonian():
    num_spat = 13
    h1e, _, e0, h2e = generate_hamiltonian(num_spat, 4)
    ham = libci.Hamiltonian(h1e, h2e, e0)
    assert ham.num_spatial == num_spat
    assert np.allclose(h1e.ravel(), ham.h1e)
    assert np.allclose(h2e.ravel(), ham.h2e)

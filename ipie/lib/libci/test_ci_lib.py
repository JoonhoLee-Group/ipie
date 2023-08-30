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
from ipie.utils.testing import get_random_phmsd_opt


def get_perm(from_orb, to_orb, di, dj):
    """Determine sign of permutation needed to align two determinants.

    Stolen from HANDE.
    """
    nmove = 0
    perm = 0
    for o in from_orb:
        io = np.where(dj == o)[0]
        perm += io - nmove
        nmove += 1
    nmove = 0
    for o in to_orb:
        io = np.where(di == o)[0]
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
        di = spin_occs[idet]
        # zero excitation case
        for iorb in range(len(di)):
            ii, spin_ii = map_orb(di[iorb], nbasis)
            # print(ii, spin_ii)
            P[spin_ii][ii, ii] += coeffs[idet].conj() * coeffs[idet]
        for jdet in range(idet + 1, num_dets):
            dj = spin_occs[jdet]
            from_orb = list(set(dj) - set(di))
            to_orb = list(set(di) - set(dj))
            nex = len(from_orb)
            if nex > 1:
                continue
            elif nex == 1:
                perm = get_perm(from_orb, to_orb, di, dj)
                if perm:
                    phase = -1
                else:
                    phase = 1
                ii, si = map_orb(from_orb[0], nbasis)
                aa, sa = map_orb(to_orb[0], nbasis)
                if si == sa:
                    if aa == 0 and ii == 4:
                        print(
                            "this: ",
                            idet,
                            jdet,
                            coeffs[jdet],
                            coeffs[idet],
                            P[si][ii, aa],
                            perm,
                            di,
                            dj,
                        )
                    P[si][aa, ii] += coeffs[jdet].conj() * coeffs[idet] * phase
                    P[si][ii, aa] += coeffs[jdet] * coeffs[idet].conj() * phase
    P[0] /= denom
    P[1] /= denom
    # print(denom)
    return P


@pytest.mark.unit
def test_one_rdm():
    num_spat = 8
    num_alpha = 2
    num_beta = 2
    np.random.seed(7)
    (coeff, occa, occb), _ = get_random_phmsd_opt(num_alpha, num_beta, num_spat, ndet=10)
    opdm = libci.one_rdm(coeff, occa, occb, num_spat)
    opdm_ref = build_one_rdm_ref(coeff, occa, occb, num_spat)
    # print(opdm[0].diagonal())
    # print(opdm_ref[0].diagonal())
    assert np.allclose(opdm[0].diagonal(), opdm_ref[0].diagonal())
    print(opdm[0][0, 4])
    print(opdm_ref[0][0, 4])

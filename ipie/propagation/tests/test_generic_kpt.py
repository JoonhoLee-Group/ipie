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

from ipie.estimators.greens_function_kpt_single_det import greens_function_kpt_single_det
from ipie.estimators.greens_function_single_det import greens_function_single_det
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.trial_wavefunction.single_det_kpt import KptSingleDet
from ipie.utils.mpi import MPIHandler
from ipie.utils.testing import (
    _blockdiag_k_orbitals,
    gen_random_test_input_kpt,
)
from ipie.walkers.uhf_walkers import UHFWalkers


@pytest.mark.unit
def test_kpt_gf_against_molecular_code():
    kmesh = (2, 1, 1)
    nk = kmesh[0] * kmesh[1] * kmesh[2]
    nbasis = 8
    naux = 4 * nbasis
    nalpha, nbeta = (3, 2)
    nwalkers = 1

    _, _, wavefunction, _ = gen_random_test_input_kpt(kmesh, nbasis, (nalpha, nbeta), naux, seed=9)

    trial_kpt = KptSingleDet(wavefunction, nk, (nalpha, nbeta), nbasis)
    psi0a_mol = _blockdiag_k_orbitals(trial_kpt.psi0a)
    psi0b_mol = _blockdiag_k_orbitals(trial_kpt.psi0b)
    wavefunction_mol = numpy.concatenate([psi0a_mol, psi0b_mol], axis=1)
    trial_mol = SingleDet(
        wavefunction_mol,
        (nk * nalpha, nk * nbeta),
        nk * nbasis,
    )
    trial_kpt.half_rotated = True
    trial_mol.half_rotated = True

    initial_walker = wavefunction_mol.copy()
    walkers_kpt = UHFWalkers(
        initial_walker,
        nk * nalpha,
        nk * nbeta,
        nk * nbasis,
        nwalkers,
        MPIHandler(),
    )
    walkers_mol = UHFWalkers(
        initial_walker,
        nk * nalpha,
        nk * nbeta,
        nk * nbasis,
        nwalkers,
        MPIHandler(),
    )

    det_kpt, sign_kpt, log_kpt = greens_function_kpt_single_det(walkers_kpt, trial_kpt)
    det_mol = greens_function_single_det(walkers_mol, trial_mol)

    numpy.testing.assert_allclose(det_kpt, det_mol, atol=1e-10)
    numpy.testing.assert_allclose(sign_kpt * numpy.exp(log_kpt), det_kpt, atol=1e-10)
    numpy.testing.assert_allclose(walkers_kpt.Ghalfa, walkers_mol.Ghalfa, atol=1e-10)
    numpy.testing.assert_allclose(walkers_kpt.Ghalfb, walkers_mol.Ghalfb, atol=1e-10)

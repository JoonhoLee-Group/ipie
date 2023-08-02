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
# Author: Fionn Malone <fmalone@google.com>
#

import os
import tempfile

import numpy as np
import pytest

from ipie.utils.io import read_hamiltonian, read_wavefunction, write_hamiltonian, write_wavefunction


@pytest.mark.unit
def test_read_write():
    nmo = 10
    naux = 100
    hcore = np.random.random((nmo, nmo))
    LXmn = np.random.random((naux, nmo, nmo))
    e0 = 18.0
    with tempfile.NamedTemporaryFile() as tmpfile:
        write_hamiltonian(hcore, LXmn, e0, filename=tmpfile.name)
        hcore_read, LXmn_read, e0_read = read_hamiltonian(tmpfile.name)
        assert np.allclose(hcore_read, hcore)
        assert np.allclose(LXmn_read, LXmn)
        assert e0 == pytest.approx(e0_read)


@pytest.mark.unit
def test_read_write_single_det_rhf():
    nmo = 10
    nalpha = 5
    wfn = np.random.random((nmo, nalpha))
    with tempfile.NamedTemporaryFile() as tmpfile:
        write_wavefunction(wfn, filename=tmpfile.name)
        wfn_read = read_wavefunction(tmpfile.name)
        assert np.allclose(wfn, wfn_read)


@pytest.mark.unit
def test_read_write_single_det_uhf():
    nmo = 10
    nalpha = 5
    nbeta = 3
    wfna = np.random.random((nmo, nalpha))
    wfnb = np.random.random((nmo, nbeta))
    wfn = [wfna, wfnb]
    with tempfile.NamedTemporaryFile() as tmpfile:
        write_wavefunction(wfn, filename=tmpfile.name)
        wfn_read, _ = read_wavefunction(tmpfile.name)
        assert np.allclose(wfn[0], wfn_read[0])
        assert np.allclose(wfn[1], wfn_read[1])


@pytest.mark.unit
def test_read_write_noci_wavefunction():
    ndet = 10
    nmo = 10
    nalpha = 5
    nbeta = 7
    wfna = np.random.random((ndet, nmo, nalpha))
    wfnb = np.random.random((ndet, nmo, nbeta))
    ci_coeffs = np.random.random((ndet))
    with tempfile.NamedTemporaryFile() as tmpfile:
        wfn = (ci_coeffs, [wfna, wfnb])
        write_wavefunction(wfn, filename=tmpfile.name)
        wfn_read, _ = read_wavefunction(tmpfile.name)
        assert np.allclose(wfn[0], wfn_read[0])
        assert np.allclose(wfn[1][0], wfn_read[1][0])
        assert np.allclose(wfn[1][1], wfn_read[1][1])


@pytest.mark.unit
def test_read_write_particle_hole_wavefunction():
    ndet = 10
    nmo = 10
    nalpha = 5
    nbeta = 7
    occa = np.random.randint((ndet, nalpha))
    occb = np.random.randint((ndet, nbeta))
    ci_coeffs = np.random.random((ndet))
    wfn = (ci_coeffs, occa, occb)
    with tempfile.NamedTemporaryFile() as tmpfile:
        write_wavefunction(wfn, filename=tmpfile.name)
        wfn_read, _ = read_wavefunction(tmpfile.name)
        assert np.allclose(wfn[0], wfn_read[0])
        assert np.allclose(wfn[1], wfn_read[1])
        assert np.allclose(wfn[2], wfn_read[2])

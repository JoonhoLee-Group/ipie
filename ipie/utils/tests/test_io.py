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
from ipie.utils.testing import get_random_phmsd_opt
import h5py
from ipie.utils.chunk_large_chol import split_cholesky
import tempfile


@pytest.mark.unit
def test_split_cholesky():
    naux = 105
    nbas = 10
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
        temp_hdf5_filename = temp_file.name
    mock_data = np.random.rand(naux, nbas, nbas)

    with h5py.File(temp_hdf5_filename, "w") as f:
        f.create_dataset("LXmn", data=mock_data)

    nmembers = 4
    split_cholesky(temp_hdf5_filename, nmembers, verbose=False)

    collected_data = []
    total_elements = 0
    for i in range(nmembers):
        with h5py.File(f"chol_{i}.h5", "r") as f:
            chol_data = f["chol"][()]
            collected_data.append(chol_data)
            assert chol_data.ndim == 2
            assert chol_data.shape[0] == nbas**2
            total_elements += chol_data.size
    assert total_elements == mock_data.size
    collected_data = np.hstack(collected_data)
    assert np.allclose(mock_data, collected_data.T.reshape(naux, nbas, nbas))

    for i in range(nmembers):
        os.remove(f"chol_{i}.h5")


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
    wfn, _ = get_random_phmsd_opt(nalpha, nbeta, nmo, ndet=ndet)
    with tempfile.NamedTemporaryFile() as tmpfile:
        write_wavefunction(wfn, filename=tmpfile.name)
        wfn_read, _ = read_wavefunction(tmpfile.name)
        assert np.allclose(wfn[0], wfn_read[0])
        assert np.allclose(wfn[1], wfn_read[1])
        assert np.allclose(wfn[2], wfn_read[2])

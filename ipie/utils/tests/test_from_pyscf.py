
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

import os

import h5py
import numpy as np
import pytest

try:
    from pyscf import ao2mo, gto, scf, mcscf, fci

    from ipie.utils.from_pyscf import (
            gen_ipie_input_from_pyscf_chk,
            integrals_from_scf,
            integrals_from_chkfile,
            freeze_core
            )
    no_pyscf = False
except (ImportError, OSError, ValueError):
    no_pyscf = True

from ipie.utils.io import (read_hamiltonian, read_wavefunction,
                           write_json_input_file)


@pytest.mark.unit
@pytest.mark.skipif(no_pyscf, reason="pyscf not found.")
def test_from_pyscf():
    atom = gto.M(atom="Ne 0 0 0", basis="sto-3g", verbose=0, parse_arg=False)
    mf = scf.RHF(atom)
    mf.kernel()
    h1e, chol, enuc, basis_change_mat = integrals_from_scf(mf, verbose=0, chol_cut=1e-5)
    assert chol.shape[0] == 15
    assert chol.shape[1] == 5
    assert h1e.shape[0] == 5


@pytest.mark.unit
@pytest.mark.skipif(no_pyscf, reason="pyscf not found.")
def test_from_chkfile():
    atom = gto.M(
        atom=[("H", 1.5 * i, 0, 0) for i in range(0, 10)],
        basis="sto-6g",
        verbose=0,
        parse_arg=False,
    )
    mf = scf.RHF(atom)
    mf.chkfile = "scf.chk"
    mf.kernel()
    h1e, chol, enuc, basis_change_mat = integrals_from_chkfile(
        "scf.chk", verbose=0, chol_cut=1e-5
    )
    assert h1e.shape == (10, 10)
    assert chol.shape == (19, 10, 10)
    assert basis_change_mat.shape == (10, 10)
    assert enuc == pytest.approx(6.805106937254286)


@pytest.mark.unit
@pytest.mark.skipif(no_pyscf, reason="pyscf not found.")
def test_pyscf_to_ipie():
    mol = gto.M(
        atom=[("H", 1.5 * i, 0, 0) for i in range(0, 4)],
        basis="6-31g",
        verbose=0,
        parse_arg=False,
    )
    mf = scf.RHF(mol)
    mf.chkfile = "scf.chk"
    mf.kernel()
    gen_ipie_input_from_pyscf_chk("scf.chk", hamil_file="afqmc.h5",
            verbose=False)
    wfn = read_wavefunction("wavefunction.h5")
    h1e, chol, ecore = read_hamiltonian("afqmc.h5")
    write_json_input_file("input.json", "afqmc.h5", "afqmc.h5", mol.nelec,
            estimates_filename="estimates.test_from_pyscf.h5")
    gen_ipie_input_from_pyscf_chk("scf.chk", hamil_file="afqmc.h5",
            num_frozen_core=2, verbose=False)
    wfn = read_wavefunction("wavefunction.h5")
    assert len(wfn[0][0].shape) == 2
    mc = mcscf.CASSCF(mf, 2, 2)
    mc.chkfile = 'scf.chk'
    e_tot, e_cas, fcivec, mo, mo_energy = mc.kernel()
    coeff, occa, occb = zip(
        *fci.addons.large_ci(fcivec, 2, (1, 1), tol=1e-8, return_strs=False)
    )
    with h5py.File("scf.chk", 'r+') as fh5:
        fh5['mcscf/ci_coeffs'] = coeff
        fh5['mcscf/occs_alpha'] = occa
        fh5['mcscf/occs_beta'] = occb
    gen_ipie_input_from_pyscf_chk("scf.chk", hamil_file="afqmc.h5",
            num_frozen_core=2, verbose=False, mcscf=True)
    wfn = read_wavefunction("wavefunction.h5")
    assert len(wfn[0]) == 3

@pytest.mark.unit
@pytest.mark.skipif(no_pyscf, reason="pyscf not found.")
def test_pyscf_to_ipie_rohf():
    mol = gto.Mole()
    mol.basis = 'cc-pvdz'
    mol.atom = (('C', 0,0,0),)
    mol.spin = 2
    mol.verbose = 0
    mol.build()

    mf = scf.ROHF(mol)
    mf.chkfile = 'scf.chk'
    mf.kernel()
    gen_ipie_input_from_pyscf_chk("scf.chk", hamil_file="afqmc.h5",
            verbose=False)
    wfn = read_wavefunction("wavefunction.h5")
    h1e, chol, ecore = read_hamiltonian("afqmc.h5")
    assert wfn[0][0].shape[-1] == mol.nelec[0]
    assert wfn[0][1].shape[-1] == mol.nelec[1]
    assert wfn[1][0].shape[-1] == mol.nelec[0]
    assert wfn[1][1].shape[-1] == mol.nelec[1]

@pytest.mark.unit
@pytest.mark.skipif(no_pyscf, reason="pyscf not found.")
def test_frozen_core():
    atom = gto.M(atom='Ne 0 0 0', basis='sto-3g', verbose=0)
    mf = scf.RHF(atom)
    energy = mf.kernel()
    ncore = 1
    h1e, chol, enuc, basis_change_mat = integrals_from_scf(mf, verbose=0, chol_cut=1e-5)
    h1e_eff, chol_eff, efzc = freeze_core(h1e, chol, enuc, mf.mo_coeff, ncore)
    assert h1e_eff.shape ==  (2, 4, 4)
    assert chol_eff.shape, (15, 4, 4)
    # Check from CASSCF object with same core.
    mc = mcscf.CASSCF(mf, 4, (4,4))
    h1_eff_ref, ecore = mc.get_h1eff()
    assert efzc == pytest.approx(ecore)
    assert np.allclose(h1_eff_ref, h1e_eff, atol=1e-12, rtol=1e-8)

@pytest.mark.unit
@pytest.mark.skipif(no_pyscf, reason="pyscf not found.")
def test_frozen_uhf():
    mol = gto.Mole()
    mol.basis = 'cc-pvdz'
    mol.atom = (('C', 0,0,0),)
    mol.spin = 2
    mol.verbose = 0
    mol.build()
    mf = scf.UHF(mol)
    mf.chkfile = "scf.chk"
    energy = mf.kernel()
    ncore = 1
    h1e, chol, enuc, basis_change_mat = integrals_from_scf(mf, verbose=0,
                                                           chol_cut=1e-8)
    h1e_eff, chol_eff, efzc = freeze_core(h1e, chol, enuc, mf.mo_coeff, ncore)
    assert h1e_eff.shape ==  (2, 13, 13)
    assert chol_eff.shape, (15, 13, 13)
    # Check from CASSCF object with same core.
    mc = mcscf.CASSCF(mf, 13, (3,1))
    h1_eff_ref, ecore = mc.get_h1eff()
    assert efzc == pytest.approx(ecore, 1e-3)
    gen_ipie_input_from_pyscf_chk("scf.chk", hamil_file="afqmc.h5",
            verbose=False, num_frozen_core=ncore)
    wfn = read_wavefunction("wavefunction.h5")
    assert wfn[0][0].shape[-1] == mol.nelec[0] - 1
    assert wfn[0][1].shape[-1] == mol.nelec[1] - 1
    assert wfn[1][0].shape[-1] == mol.nelec[0] - 1
    assert wfn[1][1].shape[-1] == mol.nelec[1] - 1

@pytest.mark.unit
@pytest.mark.skipif(no_pyscf, reason="pyscf not found.")
def test_frozen_rohf():
    mol = gto.Mole()
    mol.basis = 'cc-pvdz'
    mol.atom = (('C', 0,0,0),)
    mol.spin = 2
    mol.verbose = 0
    mol.build()
    mf = scf.ROHF(mol)
    mf.chkfile = "scf.chk"
    energy = mf.kernel()
    ncore = 1
    h1e, chol, enuc, basis_change_mat = integrals_from_scf(mf, verbose=0,
                                                           chol_cut=1e-8)
    h1e_eff, chol_eff, efzc = freeze_core(h1e, chol, enuc, mf.mo_coeff, ncore)
    assert h1e_eff.shape ==  (2, 13, 13)
    assert chol_eff.shape, (15, 13, 13)
    # Check from CASSCF object with same core.
    mc = mcscf.CASSCF(mf, 13, (3,1))
    h1_eff_ref, ecore = mc.get_h1eff()
    assert efzc == pytest.approx(ecore)
    assert np.allclose(h1_eff_ref, h1e_eff, atol=1e-12, rtol=1e-8)
    # Test UHF codepath if not necessarily UHF
    h1e, chol, efzc = freeze_core(h1e, chol, enuc, np.array([mf.mo_coeff, mf.mo_coeff]), ncore)
    assert efzc == pytest.approx(ecore)
    assert np.allclose(h1_eff_ref, h1e_eff, atol=1e-12, rtol=1e-8)
    gen_ipie_input_from_pyscf_chk("scf.chk", hamil_file="afqmc.h5",
            verbose=False, num_frozen_core=ncore)
    wfn = read_wavefunction("wavefunction.h5")
    assert wfn[0][0].shape[-1] == mol.nelec[0] - 1
    assert wfn[0][1].shape[-1] == mol.nelec[1] - 1
    assert wfn[1][0].shape[-1] == mol.nelec[0] - 1
    assert wfn[1][1].shape[-1] == mol.nelec[1] - 1


def teardown_module(self):
    cwd = os.getcwd()
    files = ["scf.chk", "afqmc.h5", "input.json", "wavefunction.h5", "hamiltonian.h5"]
    for f in files:
        try:
            os.remove(cwd + "/" + f)
        except OSError:
            pass

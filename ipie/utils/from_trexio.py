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
# Authors: Anthony Scemama <scemama@irsamc.ups-lse.fr>
#

"""Generate AFQMC data from a TREXIO file."""
try:
  import trexio
except:
  print("TREXIO is not installed. Try pip install trexio.")
  raise

import time
from dataclasses import dataclass
from typing import Tuple, Union

import h5py
import numpy
import scipy.linalg
from pyscf import lib, scf

from ipie.estimators.generic import core_contribution_cholesky
from ipie.legacy.estimators.generic import local_energy_generic_cholesky
from ipie.legacy.estimators.greens_function import gab
from ipie.utils.io import write_hamiltonian, write_wavefunction
from ipie.utils.misc import dotdict


def gen_ipie_input_from_trexio(
    trexio_filename: str,
    verbose: bool = True,
    num_frozen_core: int = 0,
) -> None:

    trexio_file = trexio.File(trexio_filename, 'r', trexio.TREXIO_AUTO)

    scf_data = load_from_trexio(trexio_file)
    mol = scf_data["mol"]
    hcore = scf_data["hcore"]
    mo_coeffs = trexio.read_mo_coefficient(trexio_file)
    mo_occ = trexio.read_mo_occupation(trexio_file)

    mo_num   = trexio.read_mo_num(trexio_file)
    chol_num = trexio.read_mo_2e_int_eri_cholesky_num(trexio_file)

    chol = numpy.zeros( (mo_num, mo_num, chol_num) )
    BUFFER_SIZE=1000000
    offset = 0
    eof = False
    while not eof:
      indices, values, nread, eof = trexio.read_mo_2e_int_eri_cholesky(trexio_file, offset, BUFFER_SIZE)
      offset += nread
      for l, integral in enumerate(values):
         i, j, k = indices[l]
         chol[i,j,k] = integral


#    basis_change_matrix = mo_coeffs
#    ham = generate_hamiltonian(
#        mol,
#        mo_coeffs,
#        hcore,
#        basis_change_matrix,
#        num_frozen_core=num_frozen_core,
#        verbose=verbose,
#    )
    write_hamiltonian(ham.h1e, ham.chol, ham.e0, filename=hamil_file)
    nelec = (mol["nelec"][0] - num_frozen_core, mol["nelec"][1] - num_frozen_core)
    if verbose:
        print(f"# Number of electrons in simulation: {nelec}")
    ci_coeffs = scf_data["ci_coeffs"]
    occa = scf_data["occa"]
    occb = scf_data["occb"]
    write_wavefunction((ci_coeffs, occa, occb), wfn_file, nelec)


@dataclass(frozen=True)
class Hamiltonian:
    h1e: numpy.ndarray
    chol: numpy.ndarray
    e0: float


@dataclass(frozen=True)
class Wavefunction:
    wfn: Union[Tuple, numpy.ndarray]


def generate_hamiltonian(
    mol,
    mo_coeffs: numpy.ndarray,
    hcore: numpy.ndarray,
    basis_change_matrix: numpy.ndarray,
    chol_cut: float = 1e-8,
    num_frozen_core: int = 0,
    ortho_ao: bool = False,
    verbose: bool = False,
) -> Hamiltonian:
#    h1e, chol, e0 = generate_integrals(
#        mol, hcore, basis_change_matrix, chol_cut=chol_cut, verbose=verbose
#    )
    if num_frozen_core > 0:
        assert num_frozen_core <= mol.nelec[0], f"{num_frozen_core} < {mol.nelec[0]}"
        assert num_frozen_core <= mol.nelec[1], f"{num_frozen_core} < {mol.nelec[1]}"
        h1e_eff, chol_act, e0_eff = freeze_core(
            h1e, chol, e0, mo_coeffs, num_frozen_core, verbose=verbose
        )
        return Hamiltonian(h1e_eff[0], chol_act, e0_eff)
    else:
        return Hamiltonian(h1e, chol, e0)




def load_from_trexio(trexio_file):
    nup = trexio.read_electron_up_num(trexio_file)
    ndn = trexio.read_electron_dn_num(trexio_file)
    mol = { "nelec" : (nup,ndn)}
    try:
      mo_occ = trexio.read_mo_occupation(trexio_file)
    except trexio.Error:
      norb = trexio.read_mo_num(trexio_file)
      mo_occ = numpy.zeros(norb)
      mo_occ[:nup] += 1.0
      mo_occ[:ndn] += 1.0
    mo_coeff = trexio.read_mo_coefficient(trexio_file)
    hcore = trexio.read_mo_1e_int_core_hamiltonian(trexio_file)
    ndet = trexio.read_determinant_num(trexio_file)
    ci_coeffs = trexio.read_determinant_coefficient(trexio_file, 0, ndet)[0]
    determinants = trexio.read_determinant_list(trexio_file, 0, ndet)[0]
    nint = trexio.get_int64_num(trexio_file)
    occa = []
    occb = []
    for d in determinants:
       occa.append( trexio.to_orbital_list(nint, d[:nint]) )
       occb.append( trexio.to_orbital_list(nint, d[nint:]) )
    occa = numpy.array(occa)
    occb = numpy.array(occb)
    scf_data = {
        "mo_occ": mo_occ,
        "hcore": hcore,
        "mo_coeff": mo_coeff,
    }
    scf_data["ci_coeffs"] = ci_coeffs
    scf_data["occa"] = occa
    scf_data["occb"] = occb
    scf_data["mol"] = mol
    return scf_data


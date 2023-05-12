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
    # pylint: disable=import-error
    import trexio
except:
    print("TREXIO is not installed. Try pip install trexio.")
    raise ImportError

import numpy

from ipie.utils.from_pyscf import freeze_core


def gen_ipie_from_trexio(
    trexio_filename: str, num_frozen_core: int = 0, verbose: bool = True
) -> None:
    trexio_file = trexio.File(trexio_filename, "r", trexio.TREXIO_AUTO)

    mo_num = trexio.read_mo_num(trexio_file)
    if verbose:
        print(f"mo_num = {mo_num} # MOs")

    chol_num = trexio.read_mo_2e_int_eri_cholesky_num(trexio_file)
    if verbose:
        print(f"chol_num = {chol_num} # Cholesky vectors")

    nup = trexio.read_electron_up_num(trexio_file)
    ndn = trexio.read_electron_dn_num(trexio_file)
    if verbose:
        print(f"{nup} up- and {ndn} down-spin electrons")

    hcore = trexio.read_mo_1e_int_core_hamiltonian(trexio_file)
    if verbose:
        print(f"Read core Hamiltonian")

    ndet = trexio.read_determinant_num(trexio_file)
    if verbose:
        print(f"{ndet} determinants")

    ci_coeffs = trexio.read_determinant_coefficient(trexio_file, 0, ndet)[0]
    if verbose:
        print(f"Read CI coefficients")

    determinants = trexio.read_determinant_list(trexio_file, 0, ndet)[0]
    nint = trexio.get_int64_num(trexio_file)
    occa = []
    occb = []
    for d in determinants:
        occa.append(trexio.to_orbital_list(nint, d[:nint]))
        occb.append(trexio.to_orbital_list(nint, d[nint:]))
    occa = numpy.array(occa, dtype=object)
    occb = numpy.array(occb, dtype=object)
    if verbose:
        print(f"Read determinants")

    e0 = trexio.read_nucleus_repulsion(trexio_file)
    if verbose:
        print(f"E0 = {e0}")

    chol = numpy.zeros((mo_num, mo_num, chol_num))
    BUFFER_SIZE = 1000000
    offset = 0
    eof = False
    while not eof:
        indices, values, nread, eof = trexio.read_mo_2e_int_eri_cholesky(
            trexio_file, offset, BUFFER_SIZE
        )
        offset += nread
        for l, integral in enumerate(values):
            i, j, k = indices[l]
            chol[i, j, k] = integral

    if verbose:
        print(f"Read Cholesky vectors")

    if num_frozen_core > 0:
        assert num_frozen_core <= nup, f"{num_frozen_core} < {nup}"
        assert num_frozen_core <= ndn, f"{num_frozen_core} < {ndn}"
        mo_coeffs = numpy.eye(mo_num)
        h1e_eff, chol_act, e0_eff = freeze_core(
            hcore, chol.T, e0, mo_coeffs, num_frozen_core, verbose=verbose
        )
        chol_act = chol_act.T.copy()
        h1e_eff = h1e_eff[0].copy()

        occa_eff = numpy.zeros((len(occa), len(occa[0]) - num_frozen_core), dtype=numpy.int64)
        occb_eff = numpy.zeros((len(occb), len(occb[0]) - num_frozen_core), dtype=numpy.int64)

        for i, (oa, ob) in enumerate(zip(occa, occb)):
            occa_eff[i, :] = oa[num_frozen_core:]
            occb_eff[i, :] = ob[num_frozen_core:]

        occa_eff -= num_frozen_core
        occb_eff -= num_frozen_core

        result = {
            "nup": nup - num_frozen_core,
            "ndn": ndn - num_frozen_core,
            "chol": chol_act,
            "hcore": h1e_eff,
            "occa": occa_eff,
            "occb": occb_eff,
            "ci_coeffs": ci_coeffs,
            "e0": e0_eff,
        }
    else:
        result = {
            "nup": nup,
            "ndn": ndn,
            "chol": chol,
            "hcore": hcore,
            "occa": occa,
            "occb": occb,
            "ci_coeffs": ci_coeffs,
            "e0": e0,
        }
    return result

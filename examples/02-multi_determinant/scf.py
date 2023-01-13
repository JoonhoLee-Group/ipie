
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

import h5py
import numpy
import scipy.sparse
from pyscf import ao2mo, fci, gto, lib, mcscf, scf

nocca = 4
noccb = 2

mol = gto.M(
    atom=[("N", 0, 0, 0), ("N", (0, 0, 3.0))],
    basis="ccpvdz",
    verbose=3,
    spin=nocca - noccb,
    unit="Bohr",
)
mf = scf.RHF(mol)
mf.chkfile = "scf.chk"
ehf = mf.kernel()
M = 6
N = 6
mc = mcscf.CASSCF(mf, M, N)
mc.chkfile = "scf.chk"
e_tot, e_cas, fcivec, mo, mo_energy = mc.kernel()
coeff, occa, occb = zip(
    *fci.addons.large_ci(fcivec, M, (nocca, noccb), tol=1e-8, return_strs=False)
)
# Need to write wavefunction to checkpoint file.
with h5py.File("scf.chk", 'r+') as fh5:
    fh5['mcscf/ci_coeffs'] = coeff
    fh5['mcscf/occs_alpha'] = occa
    fh5['mcscf/occs_beta'] = occb

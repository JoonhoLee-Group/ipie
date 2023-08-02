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
import sys

import numpy as np
from pyscf import cc, gto, scf

from ipie.config import MPI

try:
    import cupy
except ImportError:
    sys.exit(0)

mol = gto.M(
    atom=[("H", 1.6 * i, 0, 0) for i in range(0, 4)],
    basis="sto-6g",
    verbose=4,
    unit="Bohr",
)
mf = scf.UHF(mol)
mf.chkfile = "scf.chk"
mf.kernel()

# Need to flag that we want to use GPU before **any** ipie modules are imported
from ipie.config import config

config.update_option("use_gpu", True)

from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk

# Checkpoint integrals and wavefunction
# Running in serial but still need MPI World
gen_ipie_input_from_pyscf_chk(mf.chkfile, verbose=0)

comm = MPI.COMM_WORLD

from ipie.qmc.calc import build_afqmc_driver

afqmc = build_afqmc_driver(comm, nelec=mol.nelec)
# Inspect the default qmc options
print(afqmc.qmc)
# Let us override the number of blocks to keep it short
afqmc.qmc.nblocks = 20
afqmc.run(comm=comm)
# We can extract the qmc data as as a pandas data frame like so

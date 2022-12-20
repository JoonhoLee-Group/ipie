
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

from pyscf import cc, gto, scf

atom = gto.M(atom="P 0 0 0", basis="6-31G", verbose=4, spin=3, unit="Bohr")

mf = scf.UHF(atom)
mf.chkfile = "scf.chk"
mf.kernel()

mycc = mf.CCSD(frozen=list(range(5))).run()
et = mycc.ccsd_t()
print("UCCSD(T) energy", mycc.e_tot + et)

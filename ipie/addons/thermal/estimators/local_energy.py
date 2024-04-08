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

from ipie.addons.thermal.estimators.generic import local_energy_generic_cholesky
from ipie.addons.thermal.estimators.thermal import one_rdm_from_G

def local_energy_P(hamiltonian, trial, P):
    """Compute local energy from a given density matrix P.

    Parameters
    ----------
    hamiltonian : hamiltonian object
        Hamiltonian being studied.
    trial : trial wavefunction object
        Trial wavefunction.
    P : np.ndarray
        Walker density matrix.

    Returns:
    -------
    local_energy : tuple / array
        Total, one-body and two-body energies.
    """
    assert len(P) == 2
    return local_energy_generic_cholesky(hamiltonian, P)

def local_energy(hamiltonian, walker, trial):
    return local_energy_P(hamiltonian, trial, one_rdm_from_G(walker.G))

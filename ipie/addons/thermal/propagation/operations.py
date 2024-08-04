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

from ipie.utils.backend import arraylib as xp


def apply_exponential(VHS, exp_nmax):
    """Apply exponential propagator of the HS transformation

    Parameters
    ----------
    phi : numpy array
        a state
    VHS : numpy array
        HS transformation potential

    Returns
    -------
    phi : numpy array
        Exp(VHS) * phi
    """
    # Temporary array for matrix exponentiation.
    phi = xp.identity(VHS.shape[-1], dtype=xp.complex128)
    Temp = xp.zeros(phi.shape, dtype=phi.dtype)
    xp.copyto(Temp, phi)

    for n in range(1, exp_nmax + 1):
        Temp = VHS.dot(Temp) / n
        phi += Temp

    return phi  # Shape (nbasis, nbasis).

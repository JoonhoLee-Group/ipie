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

from ipie.addons.thermal.trial.mean_field import MeanField
from ipie.addons.thermal.trial.one_body import OneBody


def get_trial_density_matrix(hamiltonian, nelec, beta, dt, options=None, 
                             comm=None, verbose=False):
    """Wrapper to select trial wavefunction class.

    Parameters
    ----------

    Returns
    -------
    trial : class or None
        Trial density matrix class.
    """
    if options is None:
        options = {}

    trial_type = options.get("name", "one_body")
    alt_convention = options.get("alt_convention", False)
    if comm is None or comm.rank == 0:
        if trial_type == "one_body_mod":
            trial = OneBody(
                hamiltonian,
                nelec,
                beta,
                dt,
                options=options,
                H1=hamiltonian.h1e_mod,
                verbose=verbose,
            )

        elif trial_type == "one_body":
            trial = OneBody(hamiltonian, nelec, beta, dt, options=options, 
                            alt_convention=alt_convention, verbose=verbose)

        elif trial_type == "thermal_hartree_fock":
            trial = MeanField(hamiltonian, nelec, beta, dt, options=options, 
                              alt_convention=alt_convention, verbose=verbose)

        else:
            trial = None

    else:
        trial = None

    if comm is not None:
        trial = comm.bcast(trial)

    return trial


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

import sys

import numpy

from ipie.legacy.estimators.greens_function import gab_spin
from ipie.trial_wavefunction.multi_slater import MultiSlater
from ipie.utils.io import (
        get_input_value,
        read_qmcpack_wfn_hdf,
        read_wavefunction)


def get_trial_wavefunction(
    system, hamiltonian, options={}, comm=None, scomm=None, verbose=0
):
    """Wrapper to select trial wavefunction class.

    Parameters
    ----------
    system : class
        System class.
    hamiltonian : class
        Hamiltonian class.
    options : dict
        Trial wavefunction input options.
    comm : mpi communicator
        Global MPI communicator
    scomm : mpi communicator
        Shared communicator
    verbose : bool
        Print information.

    Returns
    -------
    trial : class or None
        Trial wavfunction class.
    """
    if comm is not None and comm.rank == 0:
        if verbose:
            print("# Building trial wavefunction object.")
    wfn_file = get_input_value(
        options, "filename", default=None, alias=["wavefunction_file"], verbose=verbose
    )
    wfn_type = options.get("name", "MultiSlater")
    if wfn_type == "MultiSlater":
        psi0 = None
        if wfn_file is not None:
            if verbose:
                print("# Reading wavefunction from {}.".format(wfn_file))
            try:
                psit, psi0 = read_wavefunction(wfn_file)
                # TODO make this saner during wavefunction refactor.
                if psi0 is not None:
                    psi0 = numpy.hstack(psi0)
                if len(psit) < 3:
                    psit = numpy.hstack(psit)
                    read = (numpy.array([1.0+0j]), psit.reshape((1,)+psit.shape))
                else:
                    read = psit
            except RuntimeError:
                # Fall back to old format.
                read, psi0 = read_qmcpack_wfn_hdf(wfn_file)
            thresh = options.get("threshold", None)
            if thresh is not None:
                coeff = read[0]
                ndets = len(coeff[abs(coeff) > thresh])
                if verbose:
                    print(
                        "# Discarding determinants with weight "
                        "  below {}.".format(thresh)
                    )
            else:
                ndets = options.get("ndets", None)
                if ndets is None:
                    ndets = len(read[0])
            if verbose:
                print(
                    "# Number of determinants in trial wavefunction: {}".format(ndets)
                )
            if ndets is not None:
                wfn = []
                # Wavefunction is a tuple, immutable so have to iterate through
                for x in read:
                    wfn.append(x[:ndets])
        else:
            if verbose:
                print("# Guessing RHF trial wavefunction.")
            na = system.nup
            nb = system.ndown
            wfn = numpy.zeros(
                (1, hamiltonian.nbasis, system.nup + system.ndown),
                dtype=numpy.complex128,
            )
            coeffs = numpy.array([1.0 + 0j])
            I = numpy.identity(hamiltonian.nbasis, dtype=numpy.complex128)
            wfn[0, :, :na] = I[:, :na]
            wfn[0, :, na:] = I[:, :nb]
            wfn = (coeffs, wfn)
        trial = MultiSlater(
            system, hamiltonian, wfn, init=psi0, options=options, verbose=verbose
        )
        if system.name == "Generic":
            if trial.ndets == 1 or trial.ortho_expansion:
                trial.half_rotate(system, hamiltonian, scomm)
        rediag = get_input_value(
            options, "recompute_ci", default=False, alias=["rediag"], verbose=verbose
        )
        if rediag:
            if comm.rank == 0:
                if verbose:
                    print("# Recomputing trial wavefunction ci coeffs.")
                coeffs = trial.recompute_ci_coeffs(
                    system.nup, system.ndown, hamiltonian
                )
            else:
                coeffs = None
            coeffs = comm.bcast(coeffs, root=0)
            trial.coeffs = coeffs
    else:
        print("Unknown trial wavefunction type.")
        sys.exit()

    spin_proj = get_input_value(
        options, "spin_proj", default=None, alias=["spin_project"], verbose=verbose
    )
    init_walker = get_input_value(
        options, "init_walker", default=None, alias=["initial_walker"], verbose=verbose
    )
    if spin_proj:
        na, nb = system.nelec
        if verbose:
            print("# Performing spin projection for walker's initial wavefunction.")
        if comm.rank == 0:
            if init_walker == "free_electron":
                eigs, eigv = numpy.linalg.eigh(system.H1[0])
            else:
                rdm, rdmh = gab_spin(trial.psi[0], trial.psi[0], na, nb)
                eigs, eigv = numpy.linalg.eigh(rdm[0] + rdm[1])
                ix = numpy.argsort(eigs)[::-1]
                trial.noons = eigs[ix]
                eigv = eigv[:, ix]
        else:
            eigv = None
        eigv = comm.bcast(eigv, root=0)
        trial.init[:, :na] = eigv[:, :na].copy()
        trial.init[:, na:] = eigv[:, :nb].copy()

    return trial

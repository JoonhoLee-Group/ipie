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

from ipie.systems.generic import Generic
from ipie.trial_wavefunction.noci import NOCI
from ipie.trial_wavefunction.particle_hole import (
    ParticleHoleWicks,
    ParticleHoleWicksNonChunked,
)
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase
from ipie.utils.io import (
    determine_wavefunction_type,
    get_input_value,
    read_noci_wavefunction,
    read_particle_hole_wavefunction,
    read_qmcpack_wfn_hdf,
    read_single_det_wavefunction,
)


def get_trial_wavefunction(
    system, hamiltonian, options={}, comm=None, scomm=None, verbose=False
):
    """Wavefunction factory.

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
    assert comm is not None
    if comm.rank == 0:
        if verbose:
            print("# Building trial wavefunction object.")
    wfn_file = get_input_value(
        options,
        "filename",
        default="wavefunction.h5",
        alias=["wavefunction_file"],
        verbose=verbose,
    )
    wfn_type = determine_wavefunction_type(wfn_file)
    ndets = get_input_value(
        options,
        "ndets",
        default=-1,
        verbose=verbose,
    )
    ndets_props = get_input_value(
        options,
        "ndets_for_trial_props",
        default=min(ndets, 100),
        alias=["ndets_prop"],
        verbose=verbose,
    )
    ndet_chunks = get_input_value(
        options,
        "ndet_chunks",
        default=1,
        alias=["nchunks", "chunks"],
        verbose=verbose,
    )
    if wfn_type == "particle_hole":
        wfn, phi0 = read_particle_hole_wavefunction(wfn_file)
        if ndet_chunks == 1:
            trial = ParticleHoleWicksNonChunked(
                wfn,
                system.nelec,
                hamiltonian.nbasis,
                num_dets_for_trial=ndets,
                num_dets_for_props=ndets_props,
            )
        else:
            trial = ParticleHoleWicks(
                wfn,
                system.nelec,
                hamiltonian.nbasis,
                num_dets_for_trial=ndets,
                num_dets_for_props=ndets_props,
                num_det_chunks=ndet_chunks,
            )
    elif wfn_type == "noci":
        wfn, phi0 = read_noci_wavefunction(wfn_file)
        trial = NOCI(
            wfn,
            system.nelec,
            hamiltonian.nbasis,
        )
    elif wfn_type == "single_determinant":
        wfn, phi0 = read_single_det_wavefunction(wfn_file)
        trial = SingleDet(
            wfn,
            system.nelec,
            hamiltonian.nbasis,
        )
    elif wfn_type == "qmcpack":
        trial = setup_qmcpack_wavefunction(
            system.nelec,
            hamiltonian.nbasis,
            wfn_file,
            ndets,
            ndets_props,
            ndet_chunks,
        )
    else:
        raise RuntimeError("Unknown wavefunction type")
    trial.build()

    if verbose:
        print(
            "# Number of determinants in trial wavefunction: {}".format(trial.num_dets)
        )
    trial.half_rotate(system, hamiltonian, scomm)

    return trial


def setup_qmcpack_wavefunction(
    nelec: tuple,
    nbasis: int,
    wfn_file: str,
    ndets: int,
    ndets_props: int,
    ndet_chunks: int,
) -> TrialWavefunctionBase:
    wfn, phi0 = read_qmcpack_wfn_hdf(wfn_file)
    if len(wfn) == 3:
        wfn, phi0 = read_particle_hole_wavefunction(wfn_file)
        if ndet_chunks == 1:
            trial = ParticleHoleWicksNonChunked(
                wfn,
                nelec,
                nbasis,
                num_dets_for_trial=ndets,
                num_dets_for_props=ndets_props,
            )
        else:
            trial = ParticleHoleWicks(
                wfn,
                nelec,
                nbasis,
                num_dets_for_trial=ndets,
                num_dets_for_props=ndets_props,
                num_det_chunks=ndet_chunks,
            )
    elif len(wfn) == 2:
        if len(wfn[0]) == 1:
            trial = SingleDet(
                wfn[1][0],
                nelec,
                nbasis,
            )
        else:
            trial = NOCI(
                wfn,
                nelec,
                nbasis,
            )
    else:
        raise RuntimeError("Unknown QMCPACK wavefunction format.")
    return trial


def chunk_trial(trial):
    self.chunked = True  # Boolean to indicate that chunked cholesky is available

    if handler.scomm.rank == 0:  # Creating copy for every rank == 0
        self._rchola = self._rchola.copy()
        self._rcholb = self._rcholb.copy()

    self._rchola_chunk = handler.scatter_group(self._rchola)  # distribute over chol
    self._rcholb_chunk = handler.scatter_group(self._rcholb)  # distribute over chol

    tot_size = handler.allreduce_group(self._rchola_chunk.size)
    assert self._rchola.size == tot_size
    tot_size = handler.allreduce_group(self._rcholb_chunk.size)
    assert self._rcholb.size == tot_size

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

import numpy as np

from ipie.trial_wavefunction.noci import NOCI
from ipie.trial_wavefunction.particle_hole import ParticleHoleWicks, ParticleHoleWicksNonChunked
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase
from ipie.utils.io import (
    determine_wavefunction_type,
    read_noci_wavefunction,
    read_particle_hole_wavefunction,
    read_qmcpack_wfn_hdf,
    read_single_det_wavefunction,
)


def get_trial_wavefunction(
    system,
    hamiltonian,
    wfn_file: str,
    ndets: int = -1,
    ndets_props: int = 1,
    ndet_chunks: int = 1,
    comm=None,
    scomm=None,
    verbose=False,
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
    assert ndets_props <= ndets
    assert comm is not None
    if comm.rank == 0:
        if verbose:
            print("# Building trial wavefunction object.")
    wfn_type = determine_wavefunction_type(wfn_file)
    if wfn_type == "particle_hole":
        wfn, _ = read_particle_hole_wavefunction(wfn_file)
        if ndet_chunks == 1:
            trial = ParticleHoleWicksNonChunked(
                wfn,
                system.nelec,
                hamiltonian.nbasis,
                num_dets_for_trial=ndets,
                num_dets_for_props=ndets_props,
                verbose=verbose,
            )
        else:
            trial = ParticleHoleWicks(
                wfn,
                system.nelec,
                hamiltonian.nbasis,
                num_dets_for_trial=ndets,
                num_dets_for_props=ndets_props,
                num_det_chunks=ndet_chunks,
                verbose=verbose,
            )
    elif wfn_type == "noci":
        wfn, _ = read_noci_wavefunction(wfn_file)
        trial = NOCI(
            wfn,
            system.nelec,
            hamiltonian.nbasis,
        )
    elif wfn_type == "single_determinant":
        wfn, _ = read_single_det_wavefunction(wfn_file)
        trial = SingleDet(
            np.hstack(wfn),
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
        print(f"# Number of determinants in trial wavefunction: {trial.num_dets}")
    trial.half_rotate(hamiltonian, scomm)
    trial.calculate_energy(system, hamiltonian)
    if trial.compute_trial_energy:
        trial.e1b = comm.bcast(trial.e1b, root=0)
        trial.e2b = comm.bcast(trial.e2b, root=0)
    comm.barrier()

    return trial


def setup_qmcpack_wavefunction(
    nelec: tuple,
    nbasis: int,
    wfn_file: str,
    ndets: int,
    ndets_props: int,
    ndet_chunks: int,
) -> TrialWavefunctionBase:
    wfn, _ = read_qmcpack_wfn_hdf(wfn_file)
    if len(wfn) == 3:
        wfn, _ = read_particle_hole_wavefunction(wfn_file)
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


def chunk_trial(self, handler):
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

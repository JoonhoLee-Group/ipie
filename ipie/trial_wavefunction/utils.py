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

from typing import Tuple

import numpy as np

from ipie.trial_wavefunction.noci import NOCI
from ipie.trial_wavefunction.particle_hole import ParticleHole, ParticleHoleNonChunked
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
    num_elec: Tuple[int, int],
    nbasis: int,
    wfn_file: str,
    ndets: int = -1,
    ndets_props: int = -1,
    ndet_chunks: int = 1,
    verbose=False,
):
    """Wavefunction factory.

    Parameters
    ----------
    options : dict
        Trial wavefunction input options.
    verbose : bool
        Print information.

    Returns
    -------
    trial : class or None
        Trial wavfunction class.
    """
    assert ndets_props <= ndets
    wfn_type = determine_wavefunction_type(wfn_file)
    if wfn_type == "particle_hole":
        wfn, _ = read_particle_hole_wavefunction(wfn_file)
        if ndet_chunks == 1:
            trial = ParticleHoleNonChunked(
                wfn,
                num_elec,
                nbasis,
                num_dets_for_trial=ndets,
                num_dets_for_props=ndets_props,
                verbose=verbose,
            )
        else:
            trial = ParticleHole(
                wfn,
                num_elec,
                nbasis,
                num_dets_for_trial=ndets,
                num_dets_for_props=ndets_props,
                num_det_chunks=ndet_chunks,
                verbose=verbose,
            )
    elif wfn_type == "noci":
        wfn, _ = read_noci_wavefunction(wfn_file)
        ci, (wfna, wfnb) = wfn
        assert len(wfn) == 2
        na = wfna.shape[-1]
        nb = wfnb.shape[-1]
        _nbasis = wfna.shape[0]
        assert nbasis == _nbasis
        outwfn = np.zeros((wfna.shape[0], wfna.shape[1], na + nb), dtype=wfna.dtype)
        outwfn[:, :, :na] = wfna.copy()
        outwfn[:, :, na:] = wfnb.copy()
        trial = NOCI((ci, outwfn), (na, nb), nbasis)
    elif wfn_type == "single_determinant":
        wfn, _ = read_single_det_wavefunction(wfn_file)
        assert len(wfn) == 2
        na = wfn[0].shape[-1]
        nb = wfn[1].shape[-1]
        _nbasis = wfn[0].shape[0]
        assert nbasis == _nbasis
        trial = SingleDet(np.hstack(wfn), (na, nb), nbasis)
    elif wfn_type == "qmcpack":
        trial = setup_qmcpack_wavefunction(wfn_file, ndets, ndets_props, ndet_chunks)
    else:
        raise RuntimeError("Unknown wavefunction type")
    trial.build()

    return trial


def setup_qmcpack_wavefunction(
    wfn_file: str, ndets: int, ndets_props: int, ndet_chunks: int
) -> TrialWavefunctionBase:
    wfn, psi0, nelec = read_qmcpack_wfn_hdf(wfn_file, get_nelec=True)
    nbasis = psi0.shape[0]
    if len(wfn) == 3:
        if ndet_chunks == 1:
            trial = ParticleHoleNonChunked(
                wfn, nelec, nbasis, num_dets_for_trial=ndets, num_dets_for_props=ndets_props
            )
        else:
            trial = ParticleHole(
                wfn,
                nelec,
                nbasis,
                num_dets_for_trial=ndets,
                num_dets_for_props=ndets_props,
                num_det_chunks=ndet_chunks,
            )
    elif len(wfn) == 2:
        if len(wfn[0]) == 1:
            nbasis = wfn[1][0].shape[0]
            trial = SingleDet(wfn[1][0], nelec, nbasis)
        else:
            nbasis = wfn[1][0].shape[0]
            trial = NOCI(wfn, nelec, nbasis)
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

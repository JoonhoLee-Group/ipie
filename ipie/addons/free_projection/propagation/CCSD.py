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

from dataclasses import dataclass

from ipie.utils.backend import arraylib as xp


@dataclass
class CCSD:
    """CCSD initial state"""

    one_body_op: None
    hs_ops: None
    orbital_rotation: None
    n_orbs: int = None
    n_occ: int = None
    n_open: int = None
    n_exc: int = None

    def __init__(self, t1, t2, orbital_rotation=None) -> None:
        """Initialise the CCSD state.

        Parameters
        ----------
        t1 : xp.ndarray
            The T1 amplitudes.
        t2 : xp.ndarray
            The T2 amplitudes.
        orbital_rotation : xp.ndarray, optional
            The orbital rotation matrix from ccsd to afqmc basis, by default None (taken to be identity).
        """
        n_exc = t1.size
        n_occ, n_open = t1.shape
        n_orbs = n_occ + n_open
        self.one_body_op = t1.T
        doubles = xp.transpose(t2, (0, 2, 1, 3)).reshape((n_exc, n_exc))
        evals, evecs = xp.linalg.eigh(doubles)
        self.hs_ops = xp.einsum(
            "i,ijk->ijk",
            xp.sqrt(evals + 0.0j),
            xp.transpose(evecs.reshape((n_occ, n_open, n_exc)), (2, 1, 0)),
        )
        if orbital_rotation is not None:
            self.orbital_rotation = orbital_rotation
        else:
            self.orbital_rotation = xp.eye(n_orbs)
        self.n_orbs = n_orbs
        self.n_occ = n_occ
        self.n_open = n_open
        self.n_exc = n_exc

    def get_walkers(self, n_walkers: int):
        """Generate random walkers from CCSD using HS."""
        ops = (
            xp.transpose(
                xp.tile(self.one_body_op, n_walkers).reshape((self.n_open, n_walkers, self.n_occ)),
                (1, 0, 2),
            )
            + 0.0j
        )
        fields = xp.random.normal(0.0, 1.0, (n_walkers, self.n_exc)) + 0.0j
        ops += xp.einsum("ij,jkl->ikl", fields, self.hs_ops)
        walkers = (
            xp.transpose(
                xp.tile(xp.eye(self.n_orbs, self.n_occ), n_walkers).reshape(
                    (self.n_orbs, n_walkers, self.n_occ)
                ),
                (1, 0, 2),
            )
            + 0.0j
        )
        walkers[:, self.n_occ :, : self.n_occ] += ops
        walkers = xp.einsum("ij,wjk->wik", self.orbital_rotation, walkers)
        return walkers

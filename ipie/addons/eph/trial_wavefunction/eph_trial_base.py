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

import numpy as np
from typing import Tuple
from abc import ABCMeta, abstractmethod


class EPhTrialWavefunctionBase(metaclass=ABCMeta):
    """Base class for electron-phonon trial wave functions.

    Parameters
    ----------
    wavefunction : :class:`np.ndarray`
        Concatenation of trial determinants of up and down spin spaces and beta
        specifying the coherent state displacement.
    num_elec : :class:`Tuple`
        Tuple of numbers of up and down spins.
    num_basis : :class:`int`
        Number of sites of Holstein chain.
    verbose : :class:`bool`
        Print level
    """

    def __init__(
        self, wavefunction: np.ndarray, num_elec: Tuple[int, int], num_basis: int, verbose=False
    ):
        self.nelec = num_elec
        self.nbasis = num_basis
        self.nup, self.ndown = self.nelec
        self.verbose = verbose
        self._num_dets = 0
        self._max_num_dets = self._num_dets
        self.ortho_expansion = False
        self.optimized = True

        self.psia = wavefunction[: self.nup]
        self.psib = wavefunction[self.nup : self.nup + self.ndown]
        self.beta_shift = wavefunction[self.nup + self.ndown :]

        self.compute_trial_energy = False
        self.energy = None

    def set_etrial(self, ham) -> None:
        energy = self.calc_energy(ham)
        self.energy = energy

    @abstractmethod
    def calc_energy(self, ham) -> float: ...

    @abstractmethod
    def calc_overlap(self, walkers) -> np.ndarray: ...

    @abstractmethod
    def calc_phonon_overlap(self, walkers) -> np.ndarray: ...

    @abstractmethod
    def calc_phonon_gradient(self, walkers) -> np.ndarray: ...

    @abstractmethod
    def calc_phonon_laplacian(self, walkers) -> np.ndarray: ...

    @abstractmethod
    def calc_phonon_laplacian_importance(self, walkers) -> np.ndarray: ...

    @abstractmethod
    def calc_phonon_laplacian_locenergy(self, walkers) -> np.ndarray: ...

    @abstractmethod
    def calc_electronic_overlap(self, walkers) -> np.ndarray: ...

    @abstractmethod
    def calc_greens_function(self, walkers) -> np.ndarray: ...

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
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase
from typing import Tuple

# NOTE could inherit from TrialWavefunctionBase, 
# but would need to redefine abstract methods.. 

class EPhTrialWavefunctionBase:
    """Base class for electron-phonon trial wave functions.
    
    Parameters
    ----------
    wavefunction : 
        Concatenation of trial determinants of up and down spin spaces and beta
        specifying the coherent state displacement.
    num_elec : 
        Tuple of numbers of up and down spins.
    num_basis : 
        Number of sites of Holstein chain.
    verbose : 
        Print level
    """
    def __init__(self, wavefunction: np.ndarray, num_elec: Tuple[int, int], 
                 num_basis: int, verbose=False):
        self.nelec = num_elec
        self.nbasis = num_basis
        self.nalpha, self.nbeta = self.nelec
        self.verbose = verbose
        self._num_dets = 0
        self._max_num_dets = self._num_dets
        self.ortho_expansion = False
        self.optimized = True
        
        self.psia = wavefunction[:self.nalpha]
        self.psib = wavefunction[self.nalpha:self.nalpha+self.nbeta]
        self.beta_shift = wavefunction[self.nalpha+self.nbeta:] 
 
        self.compute_trial_energy = False
        self.energy = None

    def build(self) -> None:
        pass

    def set_etrial(self, energy: float) -> None:
        self.energy = energy



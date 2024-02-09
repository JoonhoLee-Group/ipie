import numpy as np
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase

class EphTrialWavefunctionBase():
    def __init__(self, wavefunction, num_elec, num_basis, verbose=False):
        #super().__init__(wavefunction, num_elec, num_basis, verbose=verbose)
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

    def set_etrial(self, energy) -> None:
        self.energy = energy




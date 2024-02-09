import numpy as np
import scipy.linalg

#TODO add greens_function_coherent_state in estimators
from ipie.estimators.greens_function_single_det import greens_function_single_det
from ipie.trial_wavefunction.eph_trial_base import EphTrialWavefunctionBase

#TODO greensfunctions are in estimators 
class CoherentState(EphTrialWavefunctionBase):
    """"""
    def __init__(self, wavefunction, num_elec, num_basis, verbose=False):
        super().__init__(wavefunction, num_elec, num_basis, verbose=verbose)
        self.num_elec = num_elec
        self.nalpha, self.nbeta = self.num_elec
        self.w0 = hamiltonian.w0
        self.m = hamiltonian.m
        self.nsites = hamiltonian.nsites

        self.psia = wavefunction[:self.nalpha, :]
        self.psia = self.psia[0]
        #NOTE 1e hack
        self.psib = wavefunction[self.nalpha:self.nalpha+self.nbeta, :]
        self.beta_shift = np.squeeze(wavefunction[-1, :])


    def build(self, walkers) -> None:
        walkers.ph_ovlp = np.zeros(walkers.nwalkers)
        walkers.el_ovlp = np.zeros(walkers.nwalkers)
        walkers.total_ovlp = np.zeros(self.nsites)

    def calculate_energy(self, system, hamiltonian):
        #TODO variational_energy_coherent_state in ipie.estimators.local_energy
        ...

    def calc_overlap(self, walkers) -> np.ndarray:
        #TODO this will be a concoction of phonon and electronic overlap
        

    def calc_phonon_overlap(self, walkers) -> np.ndarray:
        walker.ph_ov = np.exp(-(self.m * self.w0 / 2) * (walkers.x - self.beta_shift)**2) 
        walker.ph_ov = np.prod(ph_ov, axis=1)
        return ph_ov

    def calc_phonon_gradient(self, walkers) -> np.ndarray:
        grad = np.einsum('ni,n->ni', (walkers.x - self.beta_shift), ovlp)
        grad *= -self.m * self.w0
        return grad

    def calc_phonon_laplacian(self, walkers) -> np.ndarray:
        arg = (walkers.x - self.beta_shift) * self.m * self.w0
        arg2 = arg**2
        laplacian = np.sum(arg2, axis=1) - self.nsites * self.m * self.w0
        return laplacian

    def calc_phonon_laplacian_importance(self, walkers) -> np.ndarray:
        return self.calc_phonon_laplacian(walkers)

    def calc_phonon_laplacian_locenergy(self, walkers) -> np.ndarray: 
        return self.calc_phonon_laplacian(walkers)

    def calc_electronic_overlap(self, walkers) -> np.ndarray:
        walkers.el_ovlp[ip] = np.einsum('i,nie->n', self.psia[perm].conj(), walkers.phia)
        if self.nbeta > 0:
            pass


    def calc_greens_function(self, walkers) -> np.ndarray:


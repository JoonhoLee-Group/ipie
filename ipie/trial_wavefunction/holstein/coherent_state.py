import numpy as np
import scipy.linalg

#TODO add greens_function_coherent_state in estimators
from ipie.estimators.greens_function_single_det import greens_function_single_det
from ipie.trial_wavefunction.holstein.eph_trial_base import EphTrialWavefunctionBase
from ipie.utils.backend import arraylib as xp

#TODO greensfunctions are in estimators 
class CoherentStateTrial(EphTrialWavefunctionBase):
    """"""
    def __init__(self, wavefunction, hamiltonian, num_elec, num_basis, verbose=False):
        super().__init__(wavefunction, num_elec, num_basis, verbose=verbose)
        self.num_elec = num_elec
        self.nup, self.ndown = self.num_elec
        self.w0 = hamiltonian.w0
        self.m = hamiltonian.m
        self.nsites = hamiltonian.nsites

        self.beta_shift = np.squeeze(wavefunction[:, 0])
        self.psia = wavefunction[:, 1:self.nup+1]
        self.psib = wavefunction[:, self.nup+1:self.nup+self.ndown+1]

    def calculate_energy(self, system, hamiltonian):
        #TODO variational_energy_coherent_state in ipie.estimators.local_energy
        ...

    def calc_overlap(self, walkers) -> np.ndarray:
        _ = self.calc_phonon_overlap(walkers)
        _ = self.calc_electronic_overlap(walkers)
        walkers.total_ovlp = walkers.el_ovlp * walkers.ph_ovlp #np.einsum('n,n->n', walkers.el_ovlp, walkers.ph_ovlp)
        walkers.ovlp = walkers.total_ovlp
        return walkers.ovlp

    def calc_phonon_overlap(self, walkers) -> np.ndarray:
        ph_ovlp = np.exp(-(self.m * self.w0 / 2) * (walkers.x - self.beta_shift)**2) 
        walkers.ph_ovlp = np.prod(ph_ovlp, axis=1)
        return walkers.ph_ovlp

    def calc_phonon_gradient(self, walkers) -> np.ndarray:
        grad = walkers.x - self.beta_shift
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
        """Computes electronic overlap.
        
        Parameters
        ----------
        walkers : class
            EphWalkers class object
            
        Returns
        -------
        walker.el_ovlp : np.ndarray
            Electronic overlap
        """
        ovlp_a = xp.einsum("wmi,mj->wij", walkers.phia, self.psia.conj(), optimize=True)
        sign_a, log_ovlp_a = xp.linalg.slogdet(ovlp_a)

        if self.ndown > 0:
            ovlp_b = xp.einsum("wmi,mj->wij", walkers.phib, self.psib.conj(), optimize=True)
            sign_b, log_ovlp_b = xp.linalg.slogdet(ovlp_b)
            ot = sign_a * sign_b * xp.exp(log_ovlp_a + log_ovlp_b - walkers.log_shift)
        else:
            ot = sign_a * xp.exp(log_ovlp_a - walkers.log_shift)

        walkers.el_ovlp = ot

        return walkers.el_ovlp

    def calc_greens_function(self, walkers) -> np.ndarray:
        """Computes Greens function.
        
        Parameters
        ----------
        walkers : class
            EphWalkers class object
        
        Returns
        -------
        walkers.G : list
            Greens function for each spin space
        """
        inv_Oa = xp.linalg.inv(xp.einsum('ie,nif->nef', self.psia, walkers.phia.conj()), optimize=True)
        walkers.Ga = xp.einsum('nie,nef,jf->nji', walkers.phia, inv_Oa, self.psia.conj(), optimize=True)

        if self.ndown > 0:
            inv_Ob = xp.linalg.inv(xp.einsum('ie,nif->nef', self.psib, walkers.phib.conj()), optimize=True)
            walkers.Gb = xp.einsum('nie,nef,jf->nji', walkers.phib, inv_Ob, self.psib.conj(), optimize=True)

        return [walkers.Ga, walkers.Gb]


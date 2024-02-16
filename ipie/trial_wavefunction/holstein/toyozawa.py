import numpy as np
import scipy.linalg

#TODO add greens_function_coherent_state in estimators
#NOTE can use single_det for now
from ipie.estimators.greens_function_single_det import greens_function_single_det
from ipie.trial_wavefunction.holstein.eph_trial_base import EphTrialWavefunctionBase
from ipie.trial_wavefunction.holstein.coherent_state import CoherentStateTrial

verbose=False

#TODO greensfunctions are in estimators 

def circ_perm(lst):
    """"""
    cpy = lst[:]                 # take a copy because a list is a mutable object
    yield cpy
    for i in range(len(lst) - 1):
        cpy = cpy[1:] + [cpy[0]]
        yield cpy

class ToyozawaTrial(CoherentStateTrial):
    """"""
    def __init__(self, wavefunction, hamiltonian, num_elec, num_basis, verbose=False):
        super().__init__(wavefunction, hamiltonian, num_elec, num_basis, verbose=verbose)
        self.perms = list(circ_perm([i for i in range(self.nbasis)]))
        self.nperms = len(self.perms)

    def calculate_energy(self, system, hamiltonian):
        #TODO variational_energy_coherent_state in ipie.estimators.local_energy
        pass

    def calc_overlap(self, walkers) -> np.ndarray:
        #TODO this will be a concoction of phonon and electronic overlap
        _ = self.calc_phonon_overlap(walkers)
        _ = self.calc_electronic_overlap(walkers)
        walkers.total_ovlp = np.einsum('np,np->np', walkers.el_ovlp, walkers.ph_ovlp) 
        walkers.ovlp = np.sum(walkers.total_ovlp, axis=1)
        return walkers.ovlp


    def calc_phonon_overlap(self, walkers) -> np.ndarray:
        """"""
        ph_ovlp = np.zeros(walkers.nwalkers, dtype=np.complex128)
        for ip,perm in enumerate(self.perms):
            ph_ov = np.exp(-(self.m * self.w0 / 2) * (walkers.x - self.beta_shift[perm])**2)
            walkers.ph_ovlp[:, ip] = np.prod(ph_ov, axis=1)
        ph_ovlp = np.sum(walkers.ph_ovlp, axis=1)  
        if verbose:
            print('ph_ovlp: ', ph_ovlp[0])
        return ph_ovlp

    def calc_phonon_gradient(self, walkers) -> np.ndarray:  
        r"""No reevaluation of phonon overlap because it reuses the overlap from the previous
        evaluation of the laplacian. The gradient only surfaces in the quantum force."""
        grad = np.zeros_like(walkers.x, dtype=np.complex128)
        for ovlp, perm in zip(walkers.ph_ovlp.T, self.perms):
            grad += np.einsum('ni,n->ni', (walkers.x - self.beta_shift[perm]), ovlp)
        grad *= -self.m * self.w0
        grad = np.einsum('ni,n->ni', grad, 1/np.sum(walkers.ph_ovlp, axis=1))
        return grad
        
    def calc_phonon_laplacian(self, walkers, ovlps) -> np.ndarray:
        r""""""
        laplacian = np.zeros(walkers.nwalkers, dtype=np.complex128)
        for ovlp, perm in zip(ovlps.T, self.perms): 
            arg = (walkers.x - self.beta_shift[perm]) * self.m * self.w0
            arg2 = arg**2
            laplacian += (np.sum(arg2, axis=1) - self.nsites * self.m * self.w0) * ovlp
        laplacian /= np.sum(ovlps, axis=1)
        return laplacian

    def calc_phonon_laplacian_importance(self, walkers) -> np.ndarray:
        return self.calc_phonon_laplacian(walkers, walkers.ph_ovlp)

    def calc_phonon_laplacian_locenergy(self, walkers) -> np.ndarray: 
        return self.calc_phonon_laplacian(walkers, walkers.total_ovlp)

    def calc_electronic_overlap(self, walkers) -> np.ndarray:
        """"""
        for ip,perm in enumerate(self.perms):
            walkers.el_ovlp[:, ip] = np.einsum('i,nie->n', self.psia[perm].conj(), walkers.phia) #this is single electron
            if self.nbeta > 0:
                pass #TODO -> adjust ovlps shape
        el_ovlp = np.sum(walkers.el_ovlp, axis=1) #NOTE this was ph before??
        if verbose:
            print('el_ovlp: ', el_ovlp[0])
        return el_ovlp

    def calc_greens_function(self, walkers) -> np.ndarray:
        """"""
        greensfct = np.zeros((walkers.nwalkers, self.nsites, self.nsites), dtype=np.complex128)
        for ovlp, perm in zip(walkers.total_ovlp.T, self.perms):
            overlap_inv = 1 / np.einsum('i,nie->n', self.psia[perm].conj(), walkers.phia) #NOTE psi currently hacked
            greensfct += np.einsum('nie,n,j,n->nji', walkers.phia, overlap_inv, 
                                   self.psia[perm].conj(), ovlp) 
        greensfct = np.einsum('nij,n->nij', greensfct, 1 / np.sum(walkers.total_ovlp, axis=1))  #these sums can be replaced by walkers.ovlp calls
        return greensfct



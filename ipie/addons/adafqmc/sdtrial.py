import torch

def gf(A, B):
    """
    Compute the greens function G = A^* (B^T A^*)^{-1} B^T
    """
    ovlp = B.t() @ A.conj()
    G = A.conj() @ torch.inverse(ovlp) @ B.t()
    return G

class SDTrial:
    def __init__(self, wavefunction):
        """
        Single Slater determinant trial wave function
        Parameters
        -------
        wavefunction: torch.tensor
            wave function
        """
        self.psi = wavefunction
        self.rh1 = None
        self.rchol = None
        self.G = gf(self.psi, self.psi)

    def half_rot(self, hamiltonian):
        '''
        Get the half rotated hamiltonian rh1 = \Psi_T^\dagger h1, and the half rotated Cholesky vector \mathcal{L}_{pi}^{\gamma}.
        '''
        self.rh1 = self.psi.conj().t() @ hamiltonian.h1e
        self.rchol = torch.einsum('ij,aik->ajk', self.psi.conj(), hamiltonian.chol)

    def calc_overlap(self, walkers):
        return torch.einsum('ij,aik->ajk', self.psi.conj(), walkers.walker_states)

    def get_ghalf(self, walkers):
        '''
        Get the half rotated greens function \Theta, and the green's function G
        '''
        overlap = torch.einsum('ij,aik->ajk', self.psi.conj(), walkers.walker_states)
        overlap_inv = torch.inverse(overlap)
        Ghalf = torch.einsum('aij, ajk-> aik', walkers.walker_states, overlap_inv)
        return Ghalf

    def calc_force_bias(self, walkers):
        Ghalf = self.get_ghalf(walkers)
        vbias= 2.0 * torch.einsum('pij,aji->ap', self.rchol, Ghalf)
        return vbias




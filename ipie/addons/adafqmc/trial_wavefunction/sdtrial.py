import torch
from ipie.addons.adafqmc.estimators.estimator import get_trial_energy


def gf(A, B):
    """
    Compute the greens function G = A^* (B^T A^*)^{-1} B^T
    """
    ovlp = B.t() @ A.conj()
    G = A.conj() @ torch.inverse(ovlp) @ B.t()
    return G


class SDTrial:
    def __init__(self, wavefunction, nelec0):
        """Single Slater determinant trial wave function

        Parameters
        -------
        wavefunction: torch.tensor
            wave function
        """
        self.coeff = wavefunction
        self.psi = wavefunction[:, :nelec0]
        self.rh1 = None
        self.rchol = None
        self.G = gf(self.psi, self.psi)

    def half_rot(self, hamiltonian):
        r"""
        Get the half rotated hamiltonian rh1 = \Psi_T^\dagger h1, and the half rotated Cholesky vector \mathcal{L}_{pi}^{\gamma}.
        """
        self.rh1 = self.psi.conj().t() @ hamiltonian.h1e
        self.rchol = torch.einsum("ij,aik->ajk", self.psi.conj(), hamiltonian.chol)

    def calc_overlap(self, walker_states):
        return torch.einsum("ij,aik->ajk", self.psi.conj(), walker_states.real) + 1j * torch.einsum(
            "ij,aik->ajk", self.psi.conj(), walker_states.imag
        )

    def self_overlap(self):
        return torch.einsum("ij,ik->jk", self.psi.conj(), self.psi)

    def get_ghalf(self, walkers):
        r"""
        Get the half rotated greens function \Theta, and the green's function G
        """
        overlap = self.calc_overlap(walkers.walker_states)
        overlap_inv = torch.inverse(overlap)
        Ghalf = torch.einsum("aij, ajk-> aik", walkers.walker_states, overlap_inv)
        return Ghalf

    def get_trial_ghalf(self):
        r"""
        Get the half rotated greens function \Theta, and the green's function G
        """
        overlap = self.self_overlap()
        overlap_inv = torch.inverse(overlap)
        Ghalf = torch.einsum("ij, jk-> ik", self.psi, overlap_inv)
        return Ghalf

    def calc_force_bias(self, walkers):
        Ghalf = self.get_ghalf(walkers)
        vbias = 2.0 * (
            torch.einsum("pij,aji->ap", self.rchol, Ghalf.real)
            + 1j * torch.einsum("pij,aji->ap", self.rchol, Ghalf.imag)
        )
        return vbias

    def eval_energy(self, hamiltonian):
        ghalf = self.get_trial_ghalf()
        return get_trial_energy(self.rh1, ghalf, self.rchol, hamiltonian.enuc)

import torch
from typing import TypeAlias

tensor: TypeAlias = torch.Tensor


def densmat(mo_coeff: tensor, nelec0):
    """
    Compute the density matrix from the MO coefficients
    """
    mo_coeff_occ = mo_coeff[:, :nelec0]
    dm = 2.0 * (mo_coeff_occ @ mo_coeff_occ.t())
    return dm


def fock(dm, hcore, ltensor):
    """
    Fock matrix in AO basis
    F = H^{core} + \sum_{kl} P_{lk} (2J_{kl} - K_{kl})
    """
    J_intermediate = torch.einsum("lk, akl -> a", dm, ltensor)
    J = torch.einsum("a, akl -> kl", J_intermediate, ltensor)
    K_intermediate = torch.einsum("lk, ail -> aik", dm, ltensor)
    K = torch.einsum("aik, akj -> ij", K_intermediate, ltensor)
    VHF = J - 0.5 * K
    return hcore + VHF


def calcEnergy(C, nelec0, hcore, ltensor):
    """
    Energy function of Hartree Fock, given the molecular orbital coefficients.
    E(C) = .5 * \sum_{ij} P_{ji}(H^{core}_{ij} + F_{ij})
    """
    # two electron integrals
    P = densmat(C, nelec0)
    F = fock(P, hcore, ltensor)
    e1 = torch.einsum("ji, ij -> ", P, hcore)
    e2 = torch.einsum("ji, ij -> ", P, F)
    return 0.5 * (e1 + e2)


def generalized_eigen_torch(A, B):
    L = torch.linalg.cholesky(B)
    L_inv = torch.linalg.inv(L)
    A_prime = L_inv @ A @ L_inv.t().conj()
    eigenvalues, eigenvectors_prime = torch.linalg.eigh(A_prime)
    C = L_inv.t().conj() @ eigenvectors_prime
    return eigenvalues, C


def gauge_fix(mo_coeff):
    max_abs_indices = torch.argmax(torch.abs(mo_coeff), dim=0)
    max_abs_values = mo_coeff[max_abs_indices, torch.arange(mo_coeff.shape[1])]
    columns_to_adjust = max_abs_values < 0
    mo_coeff = torch.where(columns_to_adjust.repeat(mo_coeff.shape[0], 1), mo_coeff * -1, mo_coeff)
    return mo_coeff


def hartree_fock(mo_coeff0, nelec0, hcore, S, ltensor, threshold=1e-10):
    """
    Hartree-Fock solver in AO basis
    """
    ncyc = 0
    dm = densmat(mo_coeff0, nelec0)
    fmat = fock(dm, hcore, ltensor)
    e, mo_coeff = generalized_eigen_torch(fmat, S)

    # gauge fix for mocoeff
    mo_coeff = gauge_fix(mo_coeff)

    dmupd = densmat(mo_coeff, nelec0)
    while ncyc <= 30:
        ncyc += 1
        dm = dmupd
        fmat = fock(dm, hcore, ltensor)
        e, mo_coeff = generalized_eigen_torch(fmat, S)
        mo_coeff = gauge_fix(mo_coeff)
        dmupd = densmat(mo_coeff, nelec0)
        if torch.norm(dmupd - dm) <= threshold:
            break
    return e, mo_coeff

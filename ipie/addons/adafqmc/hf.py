import torch
from typing import TypeAlias
tensor: TypeAlias = torch.Tensor

def densmat(mo_coeff: tensor, nelec0):
    """
    Compute the density matrix from the MO coefficients
    """
    mo_coeff_occ = mo_coeff[:, :nelec0]
    dm = 2.0 * (mo_coeff_occ @ mo_coeff_occ.t().conj())
    return dm

def fock(dm, hcore, ltensor):
    """
    Fock matrix in AO basis
    F = H^{core} + \sum_{kl} P_{lk} (2J_{kl} - K_{kl})
    """
    J_intermediate = torch.einsum('lk, akl -> a', dm, ltensor)
    J = torch.einsum('a, akl -> kl', J_intermediate, ltensor)
    K_intermediate = torch.einsum('lk, ail -> aik', dm, ltensor)
    K = torch.einsum('aik, akj -> ij', K_intermediate, ltensor)
    VHF = J - .5 * K
    return hcore + VHF

def calcEnergy(C, nelec0, hcore, ltensor):
    """
    Energy function of Hartree Fock, given the molecular orbital coefficients.
    E(C) = .5 * \sum_{ij} P_{ji}(H^{core}_{ij} + F_{ij})
    """
    # two electron integrals
    P = densmat(C, nelec0)
    F = fock(P, hcore, ltensor)
    e1 = torch.einsum('ji, ij -> ', P, hcore)
    e2 = torch.einsum('ji, ij -> ', P, F)
    return .5 * (e1 + e2)

def generalized_eigen_torch(A, B):
    # Step 1: Cholesky decomposition of B
    L = torch.linalg.cholesky(B)

    # Step 2: Transform A into the standard eigenvalue problem
    L_inv = torch.linalg.inv(L)
    A_prime = L_inv @ A @ L_inv.t().conj()

    # Step 3: Solve the standard eigenvalue problem
    eigenvalues, eigenvectors_prime = torch.linalg.eigh(A_prime)

    # Step 4: Transform eigenvectors back to original basis
    C = L_inv.t().conj() @ eigenvectors_prime
    return eigenvalues, C

def hartree_fock(mo_coeff0, nelec0, hcore, S, ltensor, threshold = 1e-10):
    '''
    Hartree-Fock solver in AO basis
    '''
    ncyc = 0
    dm = densmat(mo_coeff0, nelec0)
    
    fmat = fock(dm, hcore, ltensor)
    e, mo_coeff = generalized_eigen_torch(fmat, S)
    idx = torch.argmax(torch.abs(mo_coeff.real), axis=0)
    mo_coeff = torch.where(mo_coeff[idx, torch.arange(len(e))].real < 0, -mo_coeff, mo_coeff)
    
    dmupd = densmat(mo_coeff, nelec0)
    for i in range(30):
        ncyc += 1
        dm = dmupd
        fmat = fock(dm, hcore, ltensor)
        e, mo_coeff = generalized_eigen_torch(fmat, S)
        idx = torch.argmax(torch.abs(mo_coeff.real), axis=0)
        mo_coeff = torch.where(mo_coeff[idx, torch.arange(len(e))].real < 0, -mo_coeff, mo_coeff)
        dmupd = densmat(mo_coeff, nelec0)
        # if torch.norm(dmupd - dm) <= threshold:
        #     break
    return e, mo_coeff

def hartree_fock_obs(mol, mo_coeff0, nelec0, hcore, S, coupling, obs, ltensor, threshold = 1e-6):
    '''
    Hartree-Fock solver in AO basis with observable coupled
    '''
    ncyc = 0
    dm = densmat(mo_coeff0, nelec0)
    assert torch.allclose(dm @ S @ dm, 2* dm)
    fmat = fock(dm, hcore + coupling * obs[0], ltensor)
    e, mo_coeff = generalized_eigen_torch(fmat, S)
    
    dmupd = densmat(mo_coeff, nelec0)
    while torch.norm(dmupd - dm) > threshold:
        ncyc += 1
        dm = dmupd
        fmat = fock(dm, hcore + coupling * obs[0], ltensor)
        e, mo_coeff = generalized_eigen_torch(fmat, S)
        dmupd = densmat(mo_coeff, nelec0)
    ehftot = calcEnergy(mo_coeff, nelec0, hcore + coupling * obs[0], ltensor) + mol.energy_nuc() + coupling * obs[1]
    return e, mo_coeff, ehftot
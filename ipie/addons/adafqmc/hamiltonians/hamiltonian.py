import torch

def construct_h1e_mod(chol, h1e):
    """Construct modified one-body Hamiltonian.
    Parameters
    -------
    chol : torch.tensor
        Cholesky vectors of the two-body Hamiltonian.
    h1e : torch.tensor
        One-body Hamiltonian.
    Returns
    -------
    h1e_mod : torch.tensor
        Modified one-body Hamiltonian.
    """
    # Subtract one-body bit following reordering of 2-body operators.
    # Eqn (17) of [Motta17]_
    v0 = 0.5 * torch.einsum('apr, arq->pq', chol, chol)
    # this h1e_mod did not go through mean field subtraction, to be subtracted in propagator
    h1e_mod = h1e - v0
    return h1e_mod

def pack_cholesky(idx1, idx2, chol):
    """Pack Cholesky to a upper triangular matrix.
    Parameters
    -------
    idx1 : torch.tensor
        Index of the first dimension.
    idx2 : torch.tensor
        Index of the second dimension.
    chol : torch.tensor
        Cholesky vectors.
    Returns
    -------
    cholpacked : torch.tensor
        Packed Cholesky tensor.
    """
    nchol, n, _ = chol.shape
    # Use advanced indexing to extract the upper triangular part of each matrix in the batch
    cholpacked = chol[:, idx1, idx2]
    return cholpacked

class HamObs:
    '''Class storing information of hamiltonian with observables coupled.
    Parameters
    -------
    nelec0 : int
        Number of spin up electrons.
    nao : int
        Number of atomic orbitals.
    h1e : torch.tensor
        One-body Hamiltonian.
    chol : torch.tensor
        Cholesky vectors of the two-body Hamiltonian.
    enuc : torch.tensor
        Nuclear repulsion energy.
    observable : list of torch.tensor
        Observables with 0th element being the matrix elements and 1th element being the constant term.
    '''
    def __init__(self, nelec0, nao, h1e, chol, enuc, observable=None, packedchol=None, obs_type='dipole'):
        self.nelec0 = nelec0
        self.nao = nao
        self.h1e = h1e
        self.enuc = enuc
        self.obs = observable
        self.obs_type = obs_type
        self.coupling_shape = h1e.shape if obs_type == '1rdm' else (1,)

        self.chol = chol
        self.nchol = chol.shape[0]
        idxuppertri = torch.triu_indices(nao, nao)
        self.idx1 = idxuppertri[0]
        self.idx2 = idxuppertri[1]
        self.packedchol = pack_cholesky(idxuppertri[0], idxuppertri[1], chol) if packedchol is None else packedchol

        # modify the one body hamiltonian
        self.h1e_mod = construct_h1e_mod(chol, h1e)

def rot_ham_with_orbs(hamobs, rot_mat):
    """Rotate the Hamiltonian with the given unitary matrix.
    Parameters
    -------
    hamobs : HamObs
        Hamiltonian with observables coupled.
    rot_mat : torch.tensor
        Unitary matrix.
    Returns
    -------
    hamobs : HamObs
        Rotated Hamiltonian.
    """
    h1e = rot_mat.conj().t() @ hamobs.h1e @ rot_mat
    chol = torch.einsum('qi, aij, jp -> aqp', rot_mat.conj().t(), hamobs.chol, rot_mat)
    obs = rot_mat.conj().t() @ hamobs.obs[0] @ rot_mat
    newobs = (obs, hamobs.obs[1])
    hamobsnew = HamObs(hamobs.nelec0, hamobs.nao, h1e, chol, hamobs.enuc, newobs, obs_type=hamobs.obs_type)
    return hamobsnew

def ham_with_obs(hamobs, coupling):
    """Add observable coupling to the one-body Hamiltonian.
    Parameters
    -------
    hamobs : HamObs
        Hamiltonian with observables coupled.
    coupling : torch.tensor
        Coupling strength \lambda, H_{\lambda} = H + \lambda \hat{O}.
    """
    hamobs.h1e = hamobs.h1e + coupling * hamobs.obs[0]
    hamobs.h1e_mod = hamobs.h1e_mod + coupling * hamobs.obs[0]
    return hamobs


    

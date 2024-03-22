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

class HamObs:
    '''Class storing information of hamiltonian with observables coupled.
    Parameters
    -------
    nelec : int
        Number of electrons.
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
    def __init__(self, nelec, nao, h1e, chol, enuc, observable):
        self.nelec = nelec
        self.nao = nao
        self.h1e = h1e
        self.enuc = enuc
        self.obs = observable

        self.chol = chol
        self.nchol = chol.shape[0]

        # modify the one body hamiltonian
        self.h1e_mod = construct_h1e_mod(chol, h1e)

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


    

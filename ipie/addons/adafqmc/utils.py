import torch
import numpy as np
from hamiltonian import HamObs
from pyscf import scf, lo, ao2mo
from sdtrial import SDTrial
from hf import hartree_fock
from ipie.utils.from_pyscf import generate_integrals

def cholesky_naive(eri):
    '''
    eri: 4d array, (nao, nao, nao, nao), chemist notation, i.e. (ij|kl) = <ik|jl>
    uses direct diagonalization if the eri is not positive definite, using mCD is a better choice and this is to be implemented 
    '''
    nao = eri.shape[0]
    Vij = torch.reshape(eri, (nao**2, -1))
    try:
        L = torch.linalg.cholesky(Vij)
        ltensor = torch.reshape(L, (-1, nao, nao))
    except torch.linalg.LinAlgError:
        # it happens, since the eri is not numerically positive definite (but Why?)
        e, v = torch.linalg.eigh(Vij)
        idx = e > 1e-12
        L = (v[:,idx] * torch.sqrt(e[idx]))
        L = torch.flip(L, [1])
        ltensor = torch.reshape(L.t(), (-1, nao, nao))
    eri_reconstruct = torch.einsum('apq, ars->pqrs', ltensor, ltensor)
    # check cholesky
    assert torch.norm(eri_reconstruct - eri) < 1e-8
    return ltensor

def generate_hamiltonian_from_pyscf(mol, observable=None):
    """
    Generate Hamiltonian from PySCF mol object
    Parameters
    ----------
    mol : pyscf.gto.Mole
        PySCF mol object
    observable : None or torch.tensor
        The observable coupled to the Hamiltonian
    Returns
    -------
    ham : HamObs object
        The Hamiltonian with the observable coupled
    """
    #lowdin orthogonalization
    ovlp_mat = mol.intor('int1e_ovlp')
    X = lo.orth.lowdin(ovlp_mat)
    hcore = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
    hcore_orth_np, chol_np, enuc = generate_integrals(mol, hcore, X, chol_cut=1e-6)
    hcore_orth = torch.tensor(hcore_orth_np, dtype=torch.complex128)
    chol = torch.tensor(chol_np, dtype=torch.complex128)
    nuc = torch.tensor([enuc], dtype=torch.float64)
    ham = HamObs(mol.nelec[0], mol.nao_nr(), hcore_orth, chol, nuc, observable)
    return ham

def generate_trial_wf_from_pyscf(mol, ham, observable=None):
    """
    Generate a trial wave function from PySCF mol object
    Parameters
    -------
    mol : pyscf.gto.Mole
        PySCF mol object
    ham : HamObs
        The Hamiltonian with the observable coupled
    observable : None or torch.tensor
        The observable coupled to the Hamiltonian
    Returns
    -------
    trial_wf : SDTrial
        The trial wave function
    """
    if observable is not None:
        raise NotImplementedError
    else:
        mf = scf.RHF(mol)
        mf.kernel()
        ovlp_mat = mol.intor('int1e_ovlp')
        X = lo.orth.lowdin(ovlp_mat)
        orth = torch.tensor(X, dtype=torch.complex128)
        mo_coeff = torch.tensor(mf.mo_coeff, dtype=torch.complex128)
        orth_coeff = torch.inverse(orth) @ mo_coeff
        _, trial_obs = hartree_fock(orth_coeff, mol.nelec[0], ham.h1e, torch.eye(mol.nao_nr(), dtype=torch.complex128), ham.chol)
        trial_wf = trial_obs[:, :mol.nelec[0]]
        return SDTrial(trial_wf)
    
def remove_outliers(data):
    """
    Remove outliers for a given list of data using MAD (Median Absolute Deviation)
    Parameters
    ----------
    data : numpy.ndarray
        List of data
    Returns
    -------
    numpy.ndarray
        List of data with outliers removed
    """
    data = data.real
    if data.ndim == 1:
        median = np.median(data)
        b = 1/np.percentile(data, 75)
        mad = b * np.median(np.abs(data - median))
        # Remove data points that are more than 3 MADs from the median
        return data[abs(data - median) < 3 * mad]
    else:
        median = np.median(data[0,:])
        b = 1/np.percentile(data[0,:], 75) # scale factor for MAD
        mad = b * np.median(np.abs(data[0,:] - median))
        # Remove data points that are more than 3 MADs from the median
        return data[:, abs(data[0,:] - median) < 3 * mad]
    
def read_input(path_to_file):
    """
    Read the input file for ad-afqmc
    """
    default_options = {
        'num_steps_per_block': 800,  
        'timestep': 0.01,  
        'num_walkers_per_process': 50,  
        'num_blocks': 100,  
        'pop_control_freq': 5,  
        'stabilize_freq': 5,  
    }
    try:
        with open(path_to_file, 'r') as file:
            for line in file:
                # Split each line into key and value parts
                key, value = line.strip().split(' ')
                # Convert the value to an integer or float if possible
                if value.isdigit():
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep the value as a string if it's not a number
                # Update the default settings with the value from the file
                default_options[key] = value
    except FileNotFoundError:
        print(f"No input file found at {path_to_file}. Using default settings.")
    return default_options

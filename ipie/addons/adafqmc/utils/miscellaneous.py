import torch
import numpy as np
#pylint: disable=import-error
from pyscf import mcscf, lo
from ipie.addons.adafqmc.hamiltonians.hamiltonian import HamObs
from ipie.addons.adafqmc.utils.hf import hartree_fock
from ipie.utils.from_pyscf import generate_integrals
from ipie.utils.mpi import get_shared_array, have_shared_mem
import h5py


def generate_hamiltonian_from_pyscf(mf, ortho_ao=True, num_frozen=0, observable=None, obs_type='dipole'):
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
    hcore = mf.get_hcore()
    if ortho_ao:
        assert num_frozen == 0, "Frozen core not supported with orthogonal AO"
        ovlp = mf.mol.intor_symmetric('int1e_ovlp')
        basis_change_matrix = lo.orth.lowdin(ovlp)
    else:
        basis_change_matrix = mf.mo_coeff

    hcore_orth_np, chol_np, enuc = generate_integrals(mf.mol, hcore, basis_change_matrix, chol_cut=1e-5, verbose=True)
    
    if num_frozen > 0:
        assert not ortho_ao, "Frozen core not supported with orthogonal AO"
        assert num_frozen <= mf.mol.nelec[0], f"Number of frozen core orbitals must be no more than {mf.mol.nelec[0]}"
        assert num_frozen <= mf.mol.nelec[1], f"Number of frozen core orbitals must be no more than {mf.mol.nelec[1]}"
        mc = mcscf.CASSCF(mf, mf.mol.nao - num_frozen, mf.mol.nelectron - 2 * num_frozen)
        mc.mo_coeff = mf.mo_coeff
        hcore_orth_np, enuc = mc.get_h1eff()
        chol = chol_np[:, mc.ncore : mc.ncore + mc.ncas, mc.ncore : mc.ncore + mc.ncas]
        hcore = torch.from_numpy(hcore_orth_np)
        chol = torch.from_numpy(chol)
        nuc = torch.tensor([enuc], dtype=torch.float64)
        ham = HamObs(mc.nelecas[0], mf.mol.nao - num_frozen, hcore, chol, nuc, observable, None, obs_type)
    else:
        hcore_orth = torch.tensor(hcore_orth_np, dtype=torch.float64)
        chol = torch.tensor(chol_np, dtype=torch.float64)
        nuc = torch.tensor([enuc], dtype=torch.float64)
        ham = HamObs(mf.mol.nelec[0], mf.mol.nao_nr(), hcore_orth, chol, nuc, observable, None, obs_type)
    return ham

def generate_hamiltonianobs_shared(comm, filename, obs_type='dipole', verbose=False):
    """
    Generate HamObs from stored integrals in shared memory
    Parameters
    ----------
    comm : MPI.COMM_WORLD
        MPI split communicator (shared memory)
    filename : str
        Path to the HDF5 file containing the Hamiltonian
    Returns
    -------
    ham : HamObs object
        The Hamiltonian with the observable coupled
    """
    shmem = have_shared_mem(comm)
    if verbose:
        print(f"Have shared memory: {shmem}")
    if shmem:
        if comm.Get_rank() == 0:
            with h5py.File(filename, 'r') as f:
                hcorenp = np.asarray(f['hcore'])
                cholnp = np.asarray(f['chol'])
                packcholnp = np.asarray(f['packedchol'])
                nucnp = np.asarray(f['nuc'])
                nelec0 = np.asarray(f['nelec0']).item()
                hcoreshape = hcorenp.shape
                cholshape = cholnp.shape
                packedcholshape = packcholnp.shape
                dtype = hcorenp.dtype
                assert 'observable_mat' in f
                obsmat = np.asarray(f['observable_mat'])
                obsconst = np.asarray(f['observable_const'])
        else:
            nelec0 = None
            hcoreshape = None
            cholshape = None
            packedcholshape = None
            dtype = None
            nucnp = None
            obsconst = None
        hcoreshape = comm.bcast(hcoreshape, root=0)
        cholshape = comm.bcast(cholshape, root=0)
        packedcholshape = comm.bcast(packedcholshape, root=0)
        dtype = comm.bcast(dtype, root=0)
        nucnp = comm.bcast(nucnp, root=0)
        obsconst = comm.bcast(obsconst, root=0)
        nelec0 = comm.bcast(nelec0, root=0)
        hcore_shared = get_shared_array(comm, hcoreshape, dtype)
        if comm.Get_rank() == 0:
            hcore_shared[:] = hcorenp[:]
        comm.Barrier()
        chol_shared = get_shared_array(comm, cholshape, dtype)
        if comm.Get_rank() == 0:
            chol_shared[:] = cholnp[:]
        comm.Barrier()
        packedchol_shared = get_shared_array(comm, packedcholshape, dtype)
        if comm.Get_rank() == 0:
            packedchol_shared[:] = packcholnp[:]
        comm.Barrier()
        obsmat_shared = get_shared_array(comm, hcoreshape, dtype)
        if comm.Get_rank() == 0:
            obsmat_shared[:] = obsmat[:]
        comm.Barrier()
        # Convert shared memory numpy array into pytorch tensors
        hcore = torch.from_numpy(hcore_shared)
        chol = torch.from_numpy(chol_shared)
        packedchol = torch.from_numpy(packedchol_shared)
        nuc = torch.from_numpy(nucnp)
        obs = (torch.from_numpy(obsmat_shared), torch.from_numpy(obsconst))
        hamobs = HamObs(nelec0, hcore.shape[0], hcore, chol, nuc, obs, packedchol, obs_type)
    else:
        raise RuntimeError("Shared memory not available")
    return hamobs

def get_hf_wgradient(initialguess, nelec0, ovlp_mat, hamobs, trial_type, coupling):
    '''
    mo_coeff: molecular orbital coefficients from hartree fock
    Returns
    -------
    trial_wf: trial wavefunction in ao basis
    '''
    if trial_type == 'RHF':
        _, trial_obs = hartree_fock(initialguess, nelec0, hamobs.h1e + coupling * hamobs.obs[0], ovlp_mat, hamobs.chol)
        return trial_obs
    else:
        raise NotImplementedError

#pylint: disable=arguments-differ
class TrialwithTangent(torch.autograd.Function):
    """
    A custom autograd function that stores the trial wavefunction with precomputed trial and tangent
    """
    @staticmethod
    def forward(ctx, coupling, trial, tangent):
        ctx.save_for_backward(tangent)  # Save precomputed gradient for backward pass
        return trial

    @staticmethod
    def backward(ctx, grad_output):
        tangent, = ctx.saved_tensors  # Retrieve precomputed gradient
        return tangent * grad_output, None, None

class TrialwithTangent_matcoupling(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coupling, trial, tangent):
        ctx.save_for_backward(tangent)  # Save precomputed gradient for backward pass
        return trial

    @staticmethod
    def backward(ctx, grad_output):
        tangent, = ctx.saved_tensors  # Retrieve precomputed gradient
        return torch.einsum('abij, ab -> ij', tangent, grad_output), None, None  # None for precomputed_grad's grad
    
def trial_tangent(coupling, trial, tangent):
    return TrialwithTangent.apply(coupling, trial, tangent)

def trial_tangent_matcoupling(coupling, trial, tangent):
    return TrialwithTangent_matcoupling.apply(coupling, trial, tangent)
    
def dump_hamiltonian(ham, filename):
    """
    Dump Hamiltonian to a HDF5 file
    Parameters
    ----------
    ham : HamObs object
        The Hamiltonian with the observable coupled
    filename : str
        Path to the HDF5 file to save the Hamiltonian
    """
    with h5py.File(filename, 'w') as f:
        f.create_dataset('hcore', data=ham.h1e)
        f.create_dataset('chol', data=ham.chol)
        f.create_dataset('nuc', data=ham.enuc)
        f.create_dataset('packedchol', data=ham.packedchol)
        f.create_dataset('nelec0', data=ham.nelec0)
        if ham.obs is not None:
            f.create_dataset('observable_mat', data=ham.obs[0])
            f.create_dataset('observable_const', data=ham.obs[1])
    return

def read_hamiltonian_from_h5(filename, obs_type='dipole'):
    """
    Read Hamiltonian from a HDF5 file
    Parameters
    ----------
    filename : str
        Path to the HDF5 file containing the Hamiltonian
    Returns
    -------
    ham : HamObs object
        The Hamiltonian with the observable coupled
    """
    with h5py.File(filename, 'r') as f:
        hcore = torch.from_numpy(f['hcore'][()])
        chol = torch.from_numpy(f['chol'][()])
        nuc = torch.from_numpy(f['nuc'][()])
        packedchol = torch.from_numpy(f['packedchol'][()])
        nelec0 = np.asarray(f['nelec0']).item()
        nao = hcore.shape[0]
        if 'observable_mat' in f:
            obsmat = torch.from_numpy(f['observable_mat'][()])
            obsconst = torch.from_numpy(f['observable_const'][()])
            obs = (obsmat, obsconst)
        else:
            obs = None
    ham = HamObs(nelec0, nao, hcore, chol, nuc, obs, packedchol, obs_type)
    return ham
    
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
    assert data.ndim == 1
    median = np.median(data)
    b = 1.4826
    mad = b * np.median(np.abs(data - median))
    # Remove data points that are more than 3 MADs from the median
    indices = np.where(np.abs(data - median) <= 10 * mad)[0]
    return indices
    
def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true'):
        return True
    elif val in ('n', 'no', 'f', 'false'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))
    
def read_input(path_to_file):
    """
    Read the input file for ad-afqmc
    """
    default_options = {
        'num_walkers_per_process': 50, 
        'num_steps_per_block': 50,  
        'ad_block_size': 800,
        'num_ad_blocks': 100, 
        'timestep': 0.01, 
        'stabilize_freq': 5,  
        'pop_control_freq': 5,  
        'seed': 0,
        'grad_checkpointing': False,
        'chkpt_size': 50,
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
                        try:
                            value = strtobool(value)
                        except ValueError:
                            pass  # Keep the value as a string if it's not a number or boolean
                # Update the default settings with the value from the file
                default_options[key] = value
    except FileNotFoundError:
        print(f"No input file found at {path_to_file}. Using default settings.")
    return default_options

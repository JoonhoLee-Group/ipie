import torch
import math
from ipie.addons.adafqmc.walkers.rhf_walkers import Walkers, reorthogonalize, sr
from ipie.addons.adafqmc.estimators.estimator import get_local_e

def construct_vhs(isqrtt: float, nao: int, idx1, idx2, packedchol, xshifted):
    """
    Construct the VHS operator
    Parameters
    -------
    packedchol: torch.tensor
        Packed Cholesky vectors of the 2-body Hamiltonian
    xshifted: torch.tensor
        shifted random numbers
    Returns
    -------
    vhs: torch.tensor
        VHS operator
    """
    packedvhs = isqrtt * (torch.matmul(xshifted.real, packedchol) + 1j * torch.matmul(xshifted.imag, packedchol))
    nwalkers = xshifted.shape[0]
    vhs = unpack_vhs(packedvhs, nwalkers, nao, idx1, idx2)
    return vhs


def unpack_vhs(packedvhs, nwalkers: int, nao: int, idx1, idx2):  
    """
    Unpack the packed VHS operator
    Parameters
    -------
    packedvhs: torch.tensor
        packed VHS operator
    nwalkers: int
        number of walkers
    nao: int
        number of atomic orbitals
    idx1: torch.tensor
        index 1 for upper triangular part
    idx2: torch.tensor
        index 2 for upper triangular part
    Returns
    -------
    vhs: torch.tensor
        VHS operator
    """
    vhs = torch.empty(nwalkers, nao, nao, dtype=packedvhs.dtype)  
    # Preparing the expanded indices for batch assignment
    walker_idx = torch.arange(nwalkers).view(-1, 1)  # Shape: [nwalkers, 1]
    upper_tri_idx1 = idx1.repeat(nwalkers, 1)
    upper_tri_idx2 = idx2.repeat(nwalkers, 1) 

    # Assign the upper triangular part
    vhs[walker_idx, upper_tri_idx1, upper_tri_idx2] = packedvhs
    # Assign the lower triangular part (mirror the upper part)
    vhs[walker_idx, upper_tri_idx2, upper_tri_idx1] = packedvhs
    return vhs

def unpack_vhs_loop(packedvhs, nwalkers: int, nao: int, idx1, idx2):  
    vhs = torch.empty(nwalkers, nao, nao, dtype=packedvhs.dtype)  
    nut = round(nao * (nao + 1) / 2)
    for iw in range(nwalkers):
        for i in range(nut):
            vhs[iw, idx1[i], idx2[i]] = packedvhs[iw, i]
            vhs[iw, idx2[i], idx1[i]] = packedvhs[iw, i]
    return vhs

def compute_mf_shift(chol, G):
    """
    Compute the mean field shift
    Parameters
    -------
    hamiltonian: Hamobs
    G: Green's function of the trial wavefunction
    Returns
    -------
    mf_shift: torch.tensor
        mean field shift
    """
    mf_shift = 2 * 1j * torch.einsum('pij, ij->p', chol, G)
    return mf_shift

def compute_exph1(h1e_mod, chol, mf_shift, dt: float):
    """
    Compute the exponential of the 1-body Hamiltonian
    Parameters
    -------
    h1e_mod: torch.tensor
        1-body Hamiltonian
    chol: torch.tensor
        Cholesky decomposition of the 2-body Hamiltonian
    mf_shift: torch.tensor
        mean field shift
    dt: float
        time step
    Returns
    -------
    hcore_exp: torch.tensor
        exponential of the 1-body Hamiltonian
    """
    # mean field subtraction
    h1e_mf = h1e_mod + torch.einsum("p,pij->ij", mf_shift.imag, chol)
    hcore_exp = torch.matrix_exp(- 0.5 * dt * h1e_mf)
    return hcore_exp

def apply_VHS(taylor_order: int, vhs, states):
    """
    Apply the VHS operator to the states
    Parameters
    -------
    taylor_order: int
        order of the taylor expansion in 2-body propagator
    vhs: torch.tensor
        VHS operator
    states: torch.tensor
        states to be propagated
    Returns
    -------
    states: torch.tensor
    """
    Temp = states.clone()
    for n in range(1, taylor_order+1):
        Temp = torch.einsum('zpq, zqr->zpr', vhs, Temp) / n
        states = states + Temp
    return states

class Propagator:
    def __init__(self, comm, dt, hamiltonian, trial, prop_block_size, taylor_order=6):
        """
        Propagator class
        Parameters
        -------
        comm: MPI.COMM_WORLD
            MPI communicator
        dt: float
            time step
        hamiltonian: Hamobs
        trial: TrialWaveFunction
        prop_block_size: int
            block size for propagation
        taylor_order: int
            order of the taylor expansion in 2-body propagator
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.dt = dt
        self.prop_block_size = prop_block_size
        self.fbbound = 1.0
        self.isqrtt = 1j * math.sqrt(dt)
        self.taylor_order = taylor_order
        self.mf_shift = compute_mf_shift(hamiltonian.chol, trial.G)
        self.expH1 = compute_exph1(hamiltonian.h1e_mod, hamiltonian.chol, self.mf_shift, self.dt)
        self.vbias = None
        self.energy_estimate = trial.eval_energy(hamiltonian)
        self.h0shift = hamiltonian.enuc - .5 * torch.dot(self.mf_shift.imag, self.mf_shift.imag)
   
    def apply_bound_force_bias(self, xbar, max_bound=1.0):
        """
        apply bound to the force bias
        Parameters
        -------
        xbar: torch.tensor
            force bias
        max_bound: float
            maximum bound for the force bias
        Returns
        -------
        xbar: torch.tensor
            force bias
        """
        absxbar = torch.abs(xbar)
        idx_to_rescale = absxbar > max_bound
        nonzeros = absxbar > 1e-13
        xbar_rescaled = xbar.clone()
        xbar_rescaled[nonzeros] = xbar_rescaled[nonzeros] / absxbar[nonzeros]
        xbar = torch.where(idx_to_rescale, xbar_rescaled, xbar)
        return xbar

    def propagate_walkers(self, walkers, hamiltonian, trial):
        """
        Propagate the walkers for one step
        Parameters
        -------
        walkers: Walker
        hamiltonian: Hamobs
        trial: TrialWaveFunction
        """
        # state propagation
        ovlp = trial.calc_overlap(walkers.walker_states)
        # preparation
        self.vbias = trial.calc_force_bias(walkers)
        xbar = -math.sqrt(self.dt) * (1j * self.vbias - self.mf_shift) 
        xbar = self.apply_bound_force_bias(xbar, self.fbbound)
        # 1-body propagation
        walker_states = torch.einsum('pq, zqr->zpr', self.expH1, walkers.walker_states.real) + 1j * torch.einsum('pq, zqr->zpr', self.expH1, walkers.walker_states.imag)
        # 2-body propagation
        x = torch.randn(walkers.nwalkers, hamiltonian.nchol, dtype=torch.float64)
        xshifted = x - xbar
        vhs = construct_vhs(self.isqrtt, hamiltonian.nao, hamiltonian.idx1, hamiltonian.idx2, hamiltonian.packedchol, xshifted)
        walker_states = apply_VHS(self.taylor_order, vhs, walker_states)
        # 1-body propagation again
        walker_states = torch.einsum('pq, zqr->zpr', self.expH1, walker_states.real) + 1j * torch.einsum('pq, zqr->zpr', self.expH1, walker_states.imag)
        # weight update
        ovlp_new = trial.calc_overlap(walker_states)
        overlap_ratio = (torch.det(ovlp_new) / torch.det(ovlp))**2

        expfb = torch.exp(torch.einsum('wi,wi->w', x, xbar.real) + 1j * torch.einsum('wi,wi->w', x, xbar.imag)- 0.5 * torch.einsum('wi,wi->w', xbar, xbar)) 
        expmf = torch.exp(-math.sqrt(self.dt) * torch.einsum('wi, i->w', xshifted, self.mf_shift))
        
        # constant term e^{dt * (E_0 - H_0)}
        constant_term = torch.exp(self.dt * (self.energy_estimate - self.h0shift))
        dtheta = torch.angle(overlap_ratio * expmf)
        factor = torch.abs(overlap_ratio * expfb * expmf * constant_term) * .5 * (torch.abs(torch.cos(dtheta)) + torch.cos(dtheta)) 
        walker_weights = walkers.walker_weights * factor

        # weight capping
        wbound = torch.sum(walker_weights) * 0.10
        walker_weights = torch.where(walker_weights < wbound, walker_weights, wbound)
        return Walkers(walkers.nwalkers, walker_states, walker_weights)
    
    def propagate_block(self, iblock, walkers, hamiltonian, trial, stabilize_freq, pop_control_freq):
        """
        Propagate the walkers for a block of steps
        Parameters
        -------
        walkers: Walker
        hamiltonian: Hamobs
        trial: TrialWaveFunction
        stabilize_freq: int
            Frequency to reorthogonalize the walkers
        pop_control_freq: int
            Frequency to perform population control
        """

        for i in range(self.prop_block_size):
            if (iblock * self.prop_block_size + i) % stabilize_freq == stabilize_freq - 1:
                walkers = reorthogonalize(walkers)
            walkers = self.propagate_walkers(walkers, hamiltonian, trial)
            if i == self.prop_block_size - 1:
                # evaluate local energy and store it in propagator
                Ghalf = trial.get_ghalf(walkers)
                local_e = get_local_e(trial.rh1, Ghalf, trial.rchol, hamiltonian.enuc)
                totwts = torch.sum(walkers.walker_weights)
                etot = torch.real(torch.sum(walkers.walker_weights * local_e) / totwts)
                self.energy_estimate = etot.detach().clone()
            if (iblock * self.prop_block_size + i) % pop_control_freq == pop_control_freq - 1:
                walkers = sr(walkers)
        return walkers, etot, totwts
    
    def propagate_block_chkpt(self, ichkpt, walkers, hamiltonian, trial, stabilize_freq, pop_control_freq):
        """
        Propagate the walkers for a block of steps
        Parameters
        -------
        walkers: Walker
        hamiltonian: Hamobs
        trial: TrialWaveFunction
        stabilize_freq: int
            Frequency to reorthogonalize the walkers
        pop_control_freq: int
            Frequency to perform population control
        """
        for i in range(self.prop_block_size):
            if (ichkpt * self.prop_block_size + i) % stabilize_freq == stabilize_freq - 1:
                walkers = reorthogonalize(walkers)
            walkers = self.propagate_walkers(walkers, hamiltonian, trial)
            if i == self.prop_block_size - 1:
                # evaluate local energy and store it in propagator
                Ghalf = trial.get_ghalf(walkers)
                local_e = get_local_e(trial.rh1, Ghalf, trial.rchol, hamiltonian.enuc)
                totwts = torch.sum(walkers.walker_weights)
                etot = torch.real(torch.sum(walkers.walker_weights * local_e) / totwts)
                self.energy_estimate = etot.detach().clone()
            if (ichkpt * self.prop_block_size + i) % pop_control_freq == pop_control_freq - 1:
                walkers = sr(walkers)
        return walkers, etot, totwts






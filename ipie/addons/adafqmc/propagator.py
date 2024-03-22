import torch
import numpy as np
import h5py
from pyscf.lib import logger

def compute_mf_shift(hamiltonian, trial):
    """
    Compute the mean field shift
    Parameters
    -------
    hamiltonian: Hamobs
    trial: TrialWaveFunction
    Returns
    -------
    mf_shift: torch.tensor
        mean field shift
    """
    mf_shift = 1j * torch.einsum('pij, ji->p', hamiltonian.chol, trial.G)
    return mf_shift

def compute_exph1(h1e_mod, chol, mf_shift, dt):
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
    h1e_mf = h1e_mod - 1j * torch.einsum("p,pij->ij", mf_shift, chol)
    hcore_exp = torch.matrix_exp(- 0.5 * dt * h1e_mf)
    return hcore_exp

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
        self.dt = dt
        self.prop_block_size = prop_block_size
        self.fbbound = 1.0
        self.taylor_order = taylor_order
        self.mf_shift = compute_mf_shift(hamiltonian, trial)
        self.expH1 = compute_exph1(hamiltonian.h1e_mod, hamiltonian.chol, self.mf_shift, self.dt)
        self.vbias = None

    def construct_vhs(self, hamiltonian, xshifted):
        """
        Construct the VHS operator
        Parameters
        -------
        hamiltonian: Hamobs
        xshifted: torch.tensor
            shifted random numbers
        Returns
        -------
        vhs: torch.tensor
            VHS operator
        """
        vhs = 1j * np.sqrt(self.dt) * torch.einsum('zn, npq->zpq', xshifted, hamiltonian.chol)
        return vhs

    def apply_VHS(self, vhs, states):
        """
        Apply the VHS operator to the states
        Parameters
        -------
        vhs: torch.tensor
            VHS operator
        states: torch.tensor
            states to be propagated
        Returns
        -------
        states: torch.tensor
        """
        Temp = states.clone()
        for n in range(1, self.taylor_order+1):
            Temp = torch.einsum('zpq, zqr->zpr', vhs, Temp) / n
            states = states + Temp
        return states
    
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
        ovlp = trial.calc_overlap(walkers)
        # preparation
        self.vbias = trial.calc_force_bias(walkers)
        xbar = -np.sqrt(self.dt) * (1j * self.vbias - self.mf_shift)
        xbar = self.apply_bound_force_bias(xbar, self.fbbound)
        # 1-body propagation
        walkers.walker_states = torch.einsum('pq, zqr->zpr', self.expH1, walkers.walker_states)
        # 2-body propagation
        x = torch.randn(walkers.nwalkers, hamiltonian.nchol, dtype=torch.float64).to(torch.complex128)
        xshifted = x - xbar

        vhs = self.construct_vhs(hamiltonian, xshifted)
        walkers.walker_states = self.apply_VHS(vhs, walkers.walker_states)

        # 1-body propagation again
        walkers.walker_states = torch.einsum('pq, zqr->zpr', self.expH1, walkers.walker_states)

        # weight update
        ovlp_new = trial.calc_overlap(walkers)
        overlap_ratio = (torch.det(ovlp_new) / torch.det(ovlp))**2

        expfb = torch.exp(torch.einsum('wi,wi->w', x, xbar)- 0.5 * torch.einsum('wi,wi->w', xbar, xbar)) 
        expmf = torch.exp(-np.sqrt(self.dt) * torch.einsum('wi, i->w', xshifted, self.mf_shift))
        dtheta = torch.angle(overlap_ratio * expmf)
        factor = torch.abs(overlap_ratio * expfb * expmf) * .5 * (torch.abs(torch.cos(dtheta)) + torch.cos(dtheta))
        walkers.walker_weights = walkers.walker_weights * factor

    def propagate_block(self, walkers, hamiltonian, trial, stabilize_freq, pop_control_freq):
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
            self.propagate_walkers(walkers, hamiltonian, trial)
            if i >= stabilize_freq and i % stabilize_freq == 0:
                walkers.reorthogonalize()
            if i >= pop_control_freq and i % pop_control_freq == 0:
                walkers.sr()






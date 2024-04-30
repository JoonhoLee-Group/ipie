from mpi4py import MPI
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from ipie.addons.adafqmc.estimators.estimator import get_local_e
from ipie.addons.adafqmc.hamiltonians.hamiltonian import ham_with_obs, rot_ham_with_orbs
from ipie.addons.adafqmc.trial_wavefunction.sdtrial import SDTrial
from ipie.addons.adafqmc.walkers.rhf_walkers import Walkers, initialize_walkers, reorthogonalize, sr
from ipie.addons.adafqmc.propagation.propagator import Propagator
from ipie.addons.adafqmc.utils.miscellaneous import remove_outliers
import h5py
import time
import math

class Dist_Params:
    def __init__(self, num_walkers_per_process, num_steps_per_block, ad_block_size, num_ad_blocks, timestep, stabilize_freq, pop_control_freq, pop_control_freq_eq, seed, grad_checkpointing=False, chkpt_size=50):
        """
        Parameters
        -------
        num_walkers_per_process : int
            Number of walkers per process
        num_steps_per_block : int
            Number of steps per AD block
        num_ad_blocks : int
            Number of AD blocks

        """
        self.num_walkers_per_process = num_walkers_per_process
        self.num_steps_per_block = num_steps_per_block
        self.ad_block_size = ad_block_size
        self.num_ad_blocks = num_ad_blocks
        self.timestep = timestep
        self.stabilize_freq = stabilize_freq
        self.pop_control_freq = pop_control_freq
        self.pop_control_freq_eq = pop_control_freq_eq
        self.seed = seed
        self.grad_checkpointing = grad_checkpointing
        self.chkpt_size = chkpt_size
        if self.grad_checkpointing and num_steps_per_block % chkpt_size != 0:
            raise ValueError("The number of steps per block should be divisible by the checkpoint size")

def prep_dist_adafqmc(
    comm,
    trial_tangent, # specify the function that obtains the trial wavefunction with precomputed trial and tangent
    num_walkers_per_process: int = 50,
    num_steps_per_block: int = 50,
    ad_block_size: int = 800,
    num_ad_blocks: int = 100,
    timestep: float = 0.005,
    stabilize_freq=5,
    pop_control_freq=5,
    pop_control_freq_eq=5,
    seed=114,
    grad_checkpointing=False,
    chkpt_size=50
):  
    """Prepares the distributed AFQMC object with the given parameters
    Parameters
    -------
    comm : MPI communicator
    trial_tangent : function
        function that stores the trial and the tangent of the trial wavefunction
    num_walkers_per_process : int
        Number of walkers per process
    num_steps_per_block : int
        Number of steps per AD block
    num_ad_blocks : int
        Number of AD blocks
    timestep : float
        Timestep for the propagator
    stabilize_freq : int
        Frequency of reorthogonalization
    pop_control_freq : int
        Frequency of population control
    Returns
    -------
    Dist_ADAFQMC object
    """
    params = Dist_Params(num_walkers_per_process, num_steps_per_block, ad_block_size, num_ad_blocks, timestep, stabilize_freq, pop_control_freq, pop_control_freq_eq, seed, grad_checkpointing, chkpt_size)
    return Dist_ADAFQMC(comm, trial_tangent, params)

class Dist_ADAFQMC:
    def __init__(self, comm, trial_tangent, params: Dist_Params):
        """
        Parameters
        -------
        comm : MPI communicator
        trial_tangent : function
            function that stores the trial and the tangent of the trial wavefunction
        params : Dist_Params
            parameters for the distributed AFQMC
        """
        # trial and walkers will be generated afterwards
        # specify the set_trial function with given tangent
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.set_trial = trial_tangent
        self.params = params

    def equilibrate_walkers(self, hamobs, trial_detached):
        """
        Equilibrates the walkers
        Parameters
        -------
        hamobs : Hamiltonian
            The Hamiltonian with the observable
        trial_detached : torch.tensor
            The trial wavefunction
        Returns
        -------
        walkers : Walkers
            The equilibrated walkers
        """
        if self.rank == 0:
            print(f"equilibrating the walkers")
        #generate the trial wavefunction with the given coupling
        trial= SDTrial(trial_detached, hamobs.nelec0)
        trial.half_rot(hamobs)
        
        walkers = initialize_walkers(trial, self.params.num_walkers_per_process)
        
        # we should initialize the propagator within the ad_block because it depends on the trial
        propagator = Propagator(self.comm, self.params.timestep, hamobs, trial, self.params.num_steps_per_block)
        num_eqlb_steps = int(2.0 / self.params.timestep) * 30

        for step in range(num_eqlb_steps):
            if step % self.params.stabilize_freq == self.params.stabilize_freq - 1:
                walkers = reorthogonalize(walkers)
            walkers = propagator.propagate_walkers(walkers, hamobs, trial)
            if step >= 1:
                wbound = torch.sum(walkers.walker_weights) * 0.10
                walkers.walker_weights = torch.where(
                walkers.walker_weights < wbound, walkers.walker_weights, wbound)
            if step % self.params.pop_control_freq_eq == self.params.pop_control_freq_eq - 1:
            # Collect the walkers from all the processes in rank 0
                walker_states_np = walkers.walker_states.detach().numpy().copy()
                walker_weights_np = walkers.walker_weights.detach().numpy().copy()
                # use comm.Gather to gather the weights and states
                all_walker_states = None
                all_walker_weights = None
                if self.rank == 0:
                    all_walker_states = np.empty((self.size, *walker_states_np.shape), dtype=np.complex128)
                    all_walker_weights = np.empty((self.size, *walker_weights_np.shape), dtype=np.float64)
                self.comm.Gather(walker_states_np, all_walker_states, root=0)
                self.comm.Gather(walker_weights_np, all_walker_weights, root=0)
                splitted_walker_states = None
                splitted_walker_weights = None
                if self.rank == 0:
                    all_walker_states = torch.tensor(np.concatenate(all_walker_states, axis=0), dtype=torch.complex128)
                    all_walker_weights = torch.tensor(np.concatenate(all_walker_weights, axis=0), dtype=torch.float64)
                    all_walkers = Walkers(self.params.num_walkers_per_process * self.size, all_walker_states, all_walker_weights)
                    all_walkers = sr(all_walkers)
                    splitted_walker_states = np.stack(np.array_split(all_walkers.walker_states.detach().numpy().copy(), self.size))
                    splitted_walker_weights = np.stack(np.array_split(all_walkers.walker_weights.detach().numpy().copy(), self.size))
                splitted_walker_states_np = np.empty(walker_states_np.shape, dtype=np.complex128)
                splitted_walker_weights_np = np.empty(walker_weights_np.shape, dtype=np.float64)
                # Scatter the reorthogonalized walkers back to all the processes
                self.comm.Scatter(splitted_walker_states, splitted_walker_states_np, root=0)
                self.comm.Scatter(splitted_walker_weights, splitted_walker_weights_np, root=0)
                walkers.walker_states = torch.tensor(splitted_walker_states_np, dtype=torch.complex128)
                walkers.walker_weights = torch.tensor(splitted_walker_weights_np, dtype=torch.float64)
        return walkers
    
    def ad_block(self, coupling, hamobs, initial_walkers, trial_detached, tangent):
        '''
        Perform a block propagation and get the block energy
        '''
        # rotate the hamiltonian in the relaxed orbital basis
        rot_mat = self.set_trial(coupling, trial_detached, tangent)
        hamrot = rot_ham_with_orbs(hamobs, rot_mat)

        hamlambda = ham_with_obs(hamrot, coupling)

        #generate the trial wavefunction with the given coupling
        trial_tg = SDTrial(torch.eye(hamrot.nao, dtype=torch.float64), hamobs.nelec0)
        trial_tg.half_rot(hamlambda)


        if self.params.grad_checkpointing:
            # we should initialize the propagator within the ad_block because it depends on the trial
            propagator = Propagator(self.comm, self.params.timestep, hamlambda, trial_tg, self.params.chkpt_size)
            assert self.params.ad_block_size % self.params.chkpt_size == 0, "The ad_block_size should be divisible by the number of steps per checkpoint block"
            num_chkpts = self.params.ad_block_size // self.params.chkpt_size
            block_etots = torch.empty(num_chkpts, dtype=torch.float64)
            block_totwts = torch.empty(num_chkpts, dtype=torch.float64)
            for i in range(num_chkpts):
                initial_walkers, etot, totwts = checkpoint(propagator.propagate_block_chkpt, i, initial_walkers, hamlambda, trial_tg, self.params.stabilize_freq, self.params.pop_control_freq, use_reentrant=False)
                block_etots[i] = etot
                block_totwts[i] = totwts
        
        else:
            propagator = Propagator(self.comm, self.params.timestep, hamlambda, trial_tg, self.params.num_steps_per_block)
            assert self.params.ad_block_size % self.params.num_steps_per_block == 0, "The ad_block_size should be divisible by the number of steps per block"
            num_blocks = self.params.ad_block_size // self.params.num_steps_per_block
            block_etots = torch.empty(num_blocks, dtype=torch.float64)
            block_totwts = torch.empty(num_blocks, dtype=torch.float64)
            for i in range(num_blocks):
                initial_walkers, etot, totwts = propagator.propagate_block(i, initial_walkers, hamlambda, trial_tg, self.params.stabilize_freq, self.params.pop_control_freq)
                block_etots[i] = etot
                block_totwts[i] = totwts

        # generate block estimate for the energy and total weights
        totwts_adblock = torch.sum(block_totwts)
        etot_adblock = torch.sum(block_etots * block_totwts) / totwts_adblock
        
        # generating new walkers as the output of this function
        new_walker_states = initial_walkers.walker_states.detach().clone()
        new_walker_weights = initial_walkers.walker_weights.detach().clone()

        new_walkers = Walkers(self.params.num_walkers_per_process, new_walker_states, new_walker_weights)
        return etot_adblock, totwts_adblock, new_walkers

    def ad_block_gradient(self, hamobs, initial_walkers, trial_detached, tangent):
        """
        Performs the block automatic differentiation
        Parameters
        -------
        hamobs : Hamiltonian
            Hamiltonian with the observable
        initial_walkers : Walkers
            initial walkers
        trial_detached : torch.tensor
            trial wavefunction 
        tangent : torch.tensor
            gradient of the trial wavefunction
        Returns:
        -------
        walkers_new : Walkers
            The new walkers
        e_estimate : torch.tensor
            The energy estimate
        observable : torch.tensor
            The observable
        wtsgrad : torch.tensor
            The gradient of the weights
        """
        coupling = torch.zeros(hamobs.coupling_shape, dtype=torch.float64, requires_grad=True)
        if self.rank == 0:
            stttime0 = time.time()
        etot, totwts, initial_walkers = self.ad_block(coupling, hamobs, initial_walkers, trial_detached, tangent)
        if self.rank == 0:
            endtime0 = time.time()
            print(f"start to do backward pass, forward time = {endtime0 - stttime0}")
            stttime1 = time.time()
        grad_e = torch.autograd.grad(etot, coupling, retain_graph=True)[0]
        grad_w = torch.autograd.grad(totwts, coupling)[0]
        if self.rank == 0:
            endtime1 = time.time()
            print(f"finished doing backward pass, backward time = {endtime1 - stttime1}")
        obs_nondetached = torch.real(grad_e) + torch.real(hamobs.obs[1])
        observable = obs_nondetached.detach().clone()
        blkwts = totwts.detach().clone()
        wtsgrad = torch.real(grad_w).detach().clone()
        e_estimate = etot.detach().clone()
        return initial_walkers, e_estimate, observable, blkwts, wtsgrad 
    
    def kernel(self, hamobs, trial_detached, tangent):
        """
        kernel function of the distributed AFQMC
        Parameters
        -------
        hamobs : Hamiltonian
            Hamiltonian with the observable
        trial_detached : torch.tensor
            trial wavefunction
        tangent : torch.tensor
            gradient of the trial wavefunction
        Returns
        -------
        energycollector : np.ndarray
            The energy collector
        gradcollector : np.ndarray
            The gradient collector
        """
        seed_rk = (self.params.seed + self.comm.rank) % (2**64)
        torch.manual_seed(seed_rk)

        # walkers in the unrelaxed mo_basis
        walkers_i = self.equilibrate_walkers(hamobs, trial_detached)
        coeffinv = torch.inverse(trial_detached)

        # walkers in the relaxed mo_basis
        walkers_states = torch.einsum('pq, aqi -> api', coeffinv, walkers_i.walker_states.real) + 1j * torch.einsum('pq, aqi -> api', coeffinv, walkers_i.walker_states.imag)

        walkers_i = Walkers(self.params.num_walkers_per_process, walkers_states, walkers_i.walker_weights)

        if self.rank == 0:
            print("finished equilibrating walkers")
        if self.rank == 0:
            energycollector = np.zeros(self.params.num_ad_blocks, dtype=np.float64)
            gradcollector = np.zeros((self.params.num_ad_blocks, *hamobs.coupling_shape), dtype=np.float64)
        else:
            energycollector = np.empty(self.params.num_ad_blocks, dtype=np.float64)
            gradcollector = np.empty((self.params.num_ad_blocks, *hamobs.coupling_shape), dtype=np.float64)
        for iblock in range(self.params.num_ad_blocks):
            # global sr after each block
            if iblock > 0:
                # Collect the walkers from all the processes in rank 0
                walker_states_np = walkers_i.walker_states.detach().numpy().copy()
                walker_weights_np = walkers_i.walker_weights.detach().numpy().copy()
                # use comm.Gather to gather the weights and states
                all_walker_states = None
                all_walker_weights = None
                if self.rank == 0:
                    all_walker_states = np.empty((self.size, *walker_states_np.shape), dtype=np.complex128)
                    all_walker_weights = np.empty((self.size, *walker_weights_np.shape), dtype=np.float64)
                self.comm.Gather(walker_states_np, all_walker_states, root=0)
                self.comm.Gather(walker_weights_np, all_walker_weights, root=0)
                splitted_walker_states = None
                splitted_walker_weights = None
                if self.rank == 0:
                    all_walker_states = torch.tensor(np.concatenate(all_walker_states, axis=0), dtype=torch.complex128)
                    all_walker_weights = torch.tensor(np.concatenate(all_walker_weights, axis=0), dtype=torch.float64)
                    all_walkers = Walkers(self.params.num_walkers_per_process * self.size, all_walker_states, all_walker_weights)
                    all_walkers = sr(all_walkers)
                    splitted_walker_states = np.stack(np.array_split(all_walkers.walker_states.detach().numpy().copy(), self.size))
                    splitted_walker_weights = np.stack(np.array_split(all_walkers.walker_weights.detach().numpy().copy(), self.size))
                splitted_walker_states_np = np.empty(walker_states_np.shape, dtype=np.complex128)
                splitted_walker_weights_np = np.empty(walker_weights_np.shape, dtype=np.float64)
                # Scatter the reorthogonalized walkers back to all the processes
                self.comm.Scatter(splitted_walker_states, splitted_walker_states_np, root=0)
                self.comm.Scatter(splitted_walker_weights, splitted_walker_weights_np, root=0)
                walkers_i.walker_states = torch.tensor(splitted_walker_states_np, dtype=torch.complex128)
                walkers_i.walker_weights = torch.tensor(splitted_walker_weights_np, dtype=torch.float64)
            walkers_i, etot_i, e_grad_i, blkwts_i, wtsgrad_i = self.ad_block_gradient(hamobs, walkers_i, trial_detached, tangent)
            total_weight_rk_i = blkwts_i.numpy()
            etot_i_npy = etot_i.numpy()
            e_grad_i_npy = e_grad_i.numpy()
            wtsgrad_i_npy = wtsgrad_i.numpy()
            # Average the energy and gradient from all the processes in rank 0
            if self.comm.rank == 0:
                collected_etot = np.empty(self.size, dtype=np.float64)
                collected_e_grad = np.empty((self.size, *hamobs.coupling_shape), dtype=np.float64)
                collected_wtsgrad = np.empty((self.size, *hamobs.coupling_shape), dtype=np.float64)
                collected_totwts = np.empty(self.size, dtype=np.float64)
            else:
                collected_etot = None
                collected_e_grad = None
                collected_wtsgrad = None
                collected_totwts = None
            self.comm.Gather(etot_i_npy, collected_etot, root=0)
            self.comm.Gather(e_grad_i_npy, collected_e_grad, root=0)
            self.comm.Gather(wtsgrad_i_npy, collected_wtsgrad, root=0)
            self.comm.Gather(total_weight_rk_i, collected_totwts, root=0)
            if self.rank == 0:
                print(f"iblock = {iblock}, collected_e_grad = {collected_e_grad.ravel()}")
                print(f"iblock = {iblock}, collected_totwts = {collected_totwts}")
                print(f"iblock = {iblock}, collected_wtsgrad = {collected_wtsgrad.ravel()}")
                print(f"iblock = {iblock}, collected_etot = {collected_etot.ravel()}")
            if self.rank == 0:
                if collected_e_grad.ndim == 3:
                    e_grad = np.sum(collected_totwts[:, np.newaxis, np.newaxis] * collected_e_grad, axis=0)
                    wgrad_e = np.sum(collected_etot[:, np.newaxis, np.newaxis] * collected_wtsgrad, axis=0)
                elif collected_e_grad.ndim == 2:
                    e_grad = np.sum(collected_totwts[:, np.newaxis] * collected_e_grad, axis=0)
                    wgrad_e = np.sum(collected_etot[:, np.newaxis] * collected_wtsgrad, axis=0)
                wgradtot = np.sum(collected_wtsgrad, axis=0)
                total_weight = np.sum(collected_totwts)
                etot = np.dot(collected_etot, collected_totwts)
                etot /= total_weight
                # obs = (e_grad + wgrad_e)/total_weight - etot * wgradtot/total_weight
                obs = e_grad/total_weight
                energycollector[iblock] = etot
                gradcollector[iblock] = obs
                print(f"iblock = {iblock}, etot = {etot}, obs = {obs}")
                
            
        # broadcast the energy and gradient to all the processes
        self.comm.Bcast(energycollector, root=0)
        self.comm.Bcast(gradcollector, root=0)
        return energycollector, gradcollector
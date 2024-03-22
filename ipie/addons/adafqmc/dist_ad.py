from mpi4py import MPI
import numpy as np
import torch
from ad_afqmc import get_local_e
from hamiltonian import ham_with_obs
from sdtrial import SDTrial
from walkers import Walkers, initialize_walkers
from propagator import Propagator
from utils import remove_outliers
import h5py

def batch_grad(y, x, v):
    """
    Compute the jacobian of y with respect to x 
    Parameters
    ----------
    y : torch.tensor
        The tensor to be differentiated
    x : torch.tensor
        The tensor with respect to which the differentiation is performed
    v : torch.tensor
        the basis vectors of the batched vjp
    Returns
    -------
    torch.tensor
        The jacobian of y with respect to x
    Note: this function is only designed for the case where y is a 1d tensor and x is a scalar
    """
    def grad(v):
        return torch.autograd.grad(y, x, v)
    return torch.vmap(grad)(v)[0].flatten()

class Dist_Params:
    def __init__(self, num_walkers_per_process, num_steps_per_block, num_blocks, timestep, stabilize_freq, pop_control_freq):
        """
        Parameters
        -------
        num_walkers_per_process : int
            Number of walkers per process
        num_steps_per_block : int
            Number of steps per AD block
        num_blocks : int
            Number of AD blocks

        """
        self.num_walkers_per_process = num_walkers_per_process
        self.num_steps_per_block = num_steps_per_block
        self.num_blocks = num_blocks
        self.timestep = timestep
        self.stabilize_freq = stabilize_freq
        self.pop_control_freq = pop_control_freq

def prep_dist_adafqmc(
    comm,
    trial_tangent, # specify the function that obtains the trial wavefunction with precomputed trial and tangent
    num_walkers_per_process: int = 3,
    num_steps_per_block: int = 25,
    num_blocks: int = 200,
    timestep: float = 0.005,
    stabilize_freq=5,
    pop_control_freq=5,
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
    num_blocks : int
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
    params = Dist_Params(num_walkers_per_process, num_steps_per_block, num_blocks, timestep, stabilize_freq, pop_control_freq)
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
        trial= SDTrial(trial_detached)
        trial.half_rot(hamobs)
        
        walkers = initialize_walkers(trial, self.params.num_walkers_per_process)
        
        # we should initialize the propagator within the ad_block because it depends on the trial
        propagator = Propagator(self.comm, self.params.timestep, hamobs, trial, self.params.num_steps_per_block)
        num_eqlb_steps = int(2.0 / self.params.timestep)

        for step in range(num_eqlb_steps):
            propagator.propagate_walkers(walkers, hamobs, trial)
            if step > 1:
                wbound = torch.sum(walkers.walker_weights) * 0.10
                walkers.walker_weights = torch.where(
                walkers.walker_weights < wbound, walkers.walker_weights, wbound)
            if self.params.stabilize_freq == self.params.pop_control_freq:
                if step >= self.params.stabilize_freq and step % self.params.stabilize_freq == 0:
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
                        all_walkers.reorthogonalize()
                        all_walkers.sr()
                        splitted_walker_states = np.stack(np.array_split(all_walkers.walker_states.detach().numpy().copy(), self.size))
                        splitted_walker_weights = np.stack(np.array_split(all_walkers.walker_weights.detach().numpy().copy(), self.size))
                    splitted_walker_states_np = np.empty(walker_states_np.shape, dtype=np.complex128)
                    splitted_walker_weights_np = np.empty(walker_weights_np.shape, dtype=np.float64)
                    # Scatter the reorthogonalized walkers back to all the processes
                    self.comm.Scatter(splitted_walker_states, splitted_walker_states_np, root=0)
                    self.comm.Scatter(splitted_walker_weights, splitted_walker_weights_np, root=0)
                    walkers.walker_states = torch.tensor(splitted_walker_states_np, dtype=torch.complex128)
                    walkers.walker_weights = torch.tensor(splitted_walker_weights_np, dtype=torch.float64)
            else:
                if step >= self.params.stabilize_freq and step % self.params.stabilize_freq == 0:
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
                        all_walkers.reorthogonalize()
                        splitted_walker_states = np.stack(np.array_split(all_walkers.walker_states.detach().numpy().copy(), self.size))
                        splitted_walker_weights = np.stack(np.array_split(all_walkers.walker_weights.detach().numpy().copy(), self.size))
                    splitted_walker_states_np = np.empty(walker_states_np.shape, dtype=np.complex128)
                    splitted_walker_weights_np = np.empty(walker_weights_np.shape, dtype=np.float64)
                    # Scatter the reorthogonalized walkers back to all the processes
                    self.comm.Scatter(splitted_walker_states, splitted_walker_states_np, root=0)
                    self.comm.Scatter(splitted_walker_weights, splitted_walker_weights_np, root=0)
                    walkers.walker_states = torch.tensor(splitted_walker_states_np, dtype=torch.complex128)
                    walkers.walker_weights = torch.tensor(splitted_walker_weights_np, dtype=torch.float64)
                if step >= self.params.pop_control_freq and step % self.params.pop_control_freq == 0:
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
                        all_walkers.sr()
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
    
    def ad_block(self, hamobs, initial_walkers, trial_detached, tangent):
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
        coupling = torch.tensor([0.], dtype=torch.complex128, requires_grad=True)
        hamlambda = ham_with_obs(hamobs, coupling)

        #generate the trial wavefunction with the given coupling
        trialwf_tg = self.set_trial(coupling, trial_detached, tangent)
        trial_tg = SDTrial(trialwf_tg)
        trial_tg.half_rot(hamlambda)

        # we should initialize the propagator within the ad_block because it depends on the trial
        propagator = Propagator(self.comm, self.params.timestep, hamlambda, trial_tg, self.params.num_steps_per_block)
        propagator.propagate_block(initial_walkers, hamlambda, trial_tg, self.params.stabilize_freq, self.params.pop_control_freq)

        Ghalf = trial_tg.get_ghalf(initial_walkers)
        local_energy = get_local_e(trial_tg.rh1, Ghalf, trial_tg.rchol, hamlambda.enuc)
        totwts = torch.sum(initial_walkers.walker_weights)
        etot = torch.real(torch.sum(initial_walkers.walker_weights * local_energy) / totwts)

        grad_e, grad_w = batch_grad(torch.stack([etot, totwts]), coupling, torch.eye(2))
        obs_nondetached = torch.real(grad_e) + hamobs.obs[1]
        observable = obs_nondetached.detach().clone()
        wtsgrad = torch.real(grad_w).detach().clone()
        
        # generating new walkers as the output of this function
        new_walker_states = initial_walkers.walker_states.detach().clone()
        new_walker_weights = initial_walkers.walker_weights.detach().clone()

        new_walkers = Walkers(self.params.num_walkers_per_process, new_walker_states, new_walker_weights)
        e_estimate = etot.detach().clone()

        return new_walkers, e_estimate, observable, wtsgrad
    
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
        walkers_i = self.equilibrate_walkers(hamobs, trial_detached)
        if self.rank == 0:
            print("finished equilibrating walkers")
        if self.rank == 0:
            energycollector = np.zeros(self.params.num_blocks, dtype=np.float64)
            gradcollector = np.zeros(self.params.num_blocks, dtype=np.complex128)
        else:
            energycollector = np.empty(self.params.num_blocks, dtype=np.float64)
            gradcollector = np.empty(self.params.num_blocks, dtype=np.complex128)
        for iblock in range(self.params.num_blocks):
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
                    all_walkers.reorthogonalize()
                    all_walkers.sr()
                    splitted_walker_states = np.stack(np.array_split(all_walkers.walker_states.detach().numpy().copy(), self.size))
                    splitted_walker_weights = np.stack(np.array_split(all_walkers.walker_weights.detach().numpy().copy(), self.size))
                splitted_walker_states_np = np.empty(walker_states_np.shape, dtype=np.complex128)
                splitted_walker_weights_np = np.empty(walker_weights_np.shape, dtype=np.float64)
                # Scatter the reorthogonalized walkers back to all the processes
                self.comm.Scatter(splitted_walker_states, splitted_walker_states_np, root=0)
                self.comm.Scatter(splitted_walker_weights, splitted_walker_weights_np, root=0)
                walkers_i.walker_states = torch.tensor(splitted_walker_states_np, dtype=torch.complex128)
                walkers_i.walker_weights = torch.tensor(splitted_walker_weights_np, dtype=torch.float64)
            walkers_i, etot_i, e_grad_i, wtsgrad_i = self.ad_block(hamobs, walkers_i, trial_detached, tangent)
            total_weight_rk_i = torch.sum(walkers_i.walker_weights).numpy()
            etot_i_npy = etot_i.numpy()
            e_grad_i_npy = e_grad_i.numpy()
            wtsgrad_i_npy = wtsgrad_i.numpy()
            # Average the energy and gradient from all the processes in rank 0
            collected_etot = np.empty(self.size, dtype=np.float64)
            collected_e_grad = np.empty(self.size, dtype=np.float64)
            collected_wtsgrad = np.empty(self.size, dtype=np.float64)
            collected_totwts = np.empty(self.size, dtype=np.float64)
            self.comm.Gather(etot_i_npy, collected_etot, root=0)
            self.comm.Gather(e_grad_i_npy, collected_e_grad, root=0)
            self.comm.Gather(wtsgrad_i_npy, collected_wtsgrad, root=0)
            self.comm.Gather(total_weight_rk_i, collected_totwts, root=0)
            if self.rank == 0:
                collected_data = np.stack([collected_e_grad, collected_wtsgrad, collected_etot, collected_totwts])
                # remove outliers
                collected_data = remove_outliers(collected_data)
                e_grad = np.dot(collected_data[0], collected_data[3])
                wgrad_e = np.dot(collected_data[1], collected_data[2])
                wgradtot = np.sum(collected_data[1])
                total_weight = np.sum(collected_data[3])
                etot = np.dot(collected_data[2], collected_data[3])
                etot /= total_weight
                obs = (e_grad + wgrad_e)/total_weight - etot * wgradtot/total_weight
                energycollector[iblock] = etot
                gradcollector[iblock] = obs
                print(f"iblock = {iblock}, etot = {etot}, e_grad = {obs}")
            
        # broadcast the energy and gradient to all the processes
        self.comm.Bcast(energycollector, root=0)
        self.comm.Bcast(gradcollector, root=0)
        return energycollector, gradcollector
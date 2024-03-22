import torch
from typing import TypeAlias

from hf import hartree_fock
from hamiltonian import ham_with_obs
from sdtrial import SDTrial
from walkers import Walkers, initialize_walkers
from propagator import Propagator

tensor: TypeAlias = torch.Tensor

def get_local_e(rh1, Ghalf, rchol, nuc):
    '''
    input:
        hcore, 2d array with size (nao, nao), the unmodified hcore 
        
    returns: 
        local_e , 1d array with size (nwalkers)
    '''
    ltheta = torch.einsum('pij, ajk->apik', rchol, Ghalf)
    tr_ltheta = torch.einsum('apii->ap', ltheta)
    tr_ltheta_ltheta = torch.einsum('apij, apji->ap', ltheta, ltheta)
    local_e1 = 2 * torch.einsum('ij, aji->a', rh1, Ghalf)
    local_e2 = .5 * torch.einsum('ap->a', (2 *tr_ltheta)**2 - 2 * tr_ltheta_ltheta)
    local_nuc = nuc
    local_e = local_nuc + local_e1 + local_e2
    return local_e

def set_trial(nelec0, nao, mo_coeff0, X, hamobs, coupling, trial_type):
    '''
    X: Löwdin X
    '''
    if trial_type == 'HF':
        orth_coeff = torch.inverse(X) @ mo_coeff0
        _, trial_obs = hartree_fock(orth_coeff, nelec0, hamobs.h1e + coupling * hamobs.obs, torch.eye(nao, dtype=torch.complex128), hamobs.chol)
        trial_wf = trial_obs[:, :nelec0]
        return SDTrial(trial_wf)
    else: 
        raise NotImplementedError

class TrialwithTangent(torch.autograd.Function):
    """
    A custom autograd function that stores the trial wavefunction with precomputed trial and tangent
    """
    @staticmethod
    def forward(ctx, coupling, trial, tangent):
        ctx.save_for_backward(tangent)  # Save precomputed gradient for backward pass
        # Normally, you would compute f(x) here, but we're directly returning x for simplicity
        return trial

    @staticmethod
    def backward(ctx, grad_output):
        tangent, = ctx.saved_tensors  # Retrieve precomputed gradient
        return tangent * grad_output, None, None  # None for precomputed_grad's grad

def trial_tangent(coupling, trial, tangent):
    return TrialwithTangent.apply(coupling, trial, tangent)

class Params:
    def __init__(self, num_walkers, num_steps_per_block, num_blocks, timestep, stabilize_freq, pop_control_freq):
        self.num_walkers = num_walkers
        self.num_steps_per_block = num_steps_per_block
        self.num_blocks = num_blocks
        self.timestep = timestep
        self.stabilize_freq = stabilize_freq
        self.pop_control_freq = pop_control_freq

def prep_adafqmc(
    trial_tangent, # specify the function that obtains the trial wavefunction with precomputed trial and tangent
    num_walkers: int = 600,
    num_steps_per_block: int = 25,
    num_blocks: int = 200,
    timestep: float = 0.005,
    stabilize_freq=5,
    pop_control_freq=5,
):  
    params = Params(num_walkers, num_steps_per_block, num_blocks, timestep, stabilize_freq, pop_control_freq)
    return ADAFQMC(trial_tangent, params)

class ADAFQMC:
    def __init__(self, trial_tangent, params: Params):
        # trial and walkers will be generated afterwards
        # specify the set_trial function with given tangent
        self.set_trial = trial_tangent
        self.params = params

    def equilibrate_walkers(self, hamobs, trial_detached):
        """
        Equilibrate the walkers
        Parameters
        -------
        hamobs : Hamiltonian
            The Hamiltonian with observable
        trial_detached : torch.tensor
            The trial wave function
        Returns
        -------
        walkers : Walkers
            The equilibrated walkers
        """
        #generate the trial wavefunction with the given coupling
        trial= SDTrial(trial_detached)
        trial.half_rot(hamobs)
        
        walkers = initialize_walkers(trial, self.params.num_walkers)
        
        # we should initialize the propagator within the ad_block because it depends on the trial
        propagator = Propagator(self.params.timestep, hamobs, trial, self.params.num_steps_per_block)
        num_eqlb_steps = int(2.0 / self.params.timestep)

        for step in range(num_eqlb_steps):
            propagator.propagate_walkers(walkers, hamobs, trial)
            # if step > 1:
            #     wbound = walkers.total_weight * 0.10
            #     walkers.walker_weights = torch.clamp(
            #     walkers.walker_weights, -wbound, wbound)
            if step >= self.params.stabilize_freq and step % self.params.stabilize_freq == 0:
                walkers.reorthogonalize()
            if step >= self.params.pop_control_freq and step % self.params.pop_control_freq == 0:
                walkers.sr()
        return walkers
        
    def ad_block(self, hamobs, initial_walkers, trial_detached, tangent):
        """
        Perform a block of ADAFQMC
        Parameters
        -------
        hamobs : Hamiltonian
            The Hamiltonian with observable
        initial_walkers : Walkers
            The initial walkers
        trial_detached : torch.tensor
            The trial wave function
        tangent : torch.tensor
            Gradient of the trial wave function
        Returns
        -------
        new_walkers : Walkers
            New walkers
        e_estimate : torch.tensor
            Energy estimate
        observable : torch.tensor
            Observable calculated using Hellmann-Feynman theorem
        """

        coupling = torch.tensor([0.], dtype=torch.complex128, requires_grad=True)
        hamlambda = ham_with_obs(hamobs, coupling)

        #generate the trial wavefunction with the given coupling
        trialwf_tg = self.set_trial(coupling, trial_detached, tangent)
        trial_tg = SDTrial(trialwf_tg)
        trial_tg.half_rot(hamlambda)

        # we should initialize the propagator within the ad_block because it depends on the trial
        propagator = Propagator(self.params.timestep, hamlambda, trial_tg, self.params.num_steps_per_block)

        propagator.propagate_block(initial_walkers, hamlambda, trial_tg, self.params.stabilize_freq, self.params.pop_control_freq)

        Ghalf = trial_tg.get_ghalf(initial_walkers)
        local_energy = get_local_e(trial_tg.rh1, Ghalf, trial_tg.rchol, hamlambda.enuc)
        etot = torch.real(torch.sum(initial_walkers.walker_weights * local_energy) / torch.sum(initial_walkers.walker_weights))

        grad_e = torch.autograd.grad(etot, coupling)[0]
        obs_nondetached = grad_e + hamobs.obs[1]
        observable = obs_nondetached.detach().clone()
        
        # generating new walkers as the output of this function
        new_walker_states = initial_walkers.walker_states.detach().clone()
        new_walker_weights = initial_walkers.walker_weights.detach().clone()

        new_walkers = Walkers(self.params.num_walkers, new_walker_states, new_walker_weights)
        e_estimate = etot.detach().clone()

        return new_walkers, e_estimate, observable

    def kernel(self, hamobs, trial_detached, tangent):
        """
        Perform ADAFQMC
        Parameters
        -------
        hamobs : HamObs
            The Hamiltonian with observable
        trial_detached : torch.tensor
            The trial wave function
        tangent : torch.tensor
            Gradient of the trial wave function
        Returns
        -------
        energycollector : torch.tensor
            Energy estimate
        gradcollector : torch.tensor
            Observable calculated using Hellmann-Feynman theorem
        """
        energycollector = torch.zeros(self.params.num_blocks, dtype=torch.float64)
        gradcollector = torch.zeros(self.params.num_blocks, dtype=torch.complex128)
        walkers = self.equilibrate_walkers(hamobs, trial_detached, tangent)
        for iblock in range(self.params.num_blocks):
            walkers, etot, e_grad = self.ad_block(hamobs, walkers, trial_detached, tangent)
            energycollector[iblock] = etot
            gradcollector[iblock] = e_grad
        return energycollector, gradcollector


import torch

def initialize_walkers(trial, nwalkers):
    """
    Initialize the walkers with the trial wave function
    Parameters
    -------
    trial : SDTrial
        The trial wave function
    nwalkers : int
        The number of walkers
    """
    assert trial.psi is not None
    walker_states = torch.stack([trial.psi] * nwalkers).detach()
    walker_weights = torch.tensor([1.] * nwalkers, dtype=torch.float64)
    return Walkers(nwalkers, walker_states, walker_weights)

class Walkers:
    def __init__(self, nwalkers, walker_states, walker_weights):
        """
        Walkers class
        Parameters
        -------
        nwalkers : int
            The number of walkers
        walker_states : torch.tensor
            walker states
        walker_weights : torch.tensor
            walker weights
        """

        self.nwalkers = nwalkers
        self.walker_states = walker_states
        self.walker_weights = walker_weights
        self.total_weight = torch.sum(walker_weights)

    def reorthogonalize(self):
        """
        Reorthogonalize the walkers
        """
        ortho_walkers, _ = torch.vmap(torch.linalg.qr)(self.walker_states)
        self.walker_states = ortho_walkers
    
    def sr(self):
        """
        stochastic reconfiguration method for population control
        """
        # rescale the weights
        self.walker_weights = self.walker_weights / torch.sum(self.walker_weights) * self.nwalkers
        cumulative_weights =torch.cumsum(self.walker_weights, dim=0)
        total_weight = cumulative_weights[-1]
        self.total_weight = total_weight
        average_weight = total_weight / self.nwalkers
        self.walker_weights = torch.ones(self.nwalkers, dtype=torch.float64) * average_weight
        zeta = torch.rand(1).item()
        #print(f"zeta for sr: {zeta}")
        z = total_weight * (torch.arange(self.nwalkers) + zeta) / self.nwalkers
        indices = torch.vmap(torch.searchsorted)(cumulative_weights, z)
        indices = torch.where(indices < self.nwalkers, indices, 0)
        self.walker_states = self.walker_states[indices]

import torch


def initialize_walkers(trial, nwalkers):
    """Initialize the walkers with the trial wave function

    Parameters
    -------
    trial : SDTrial
        The trial wave function
    nwalkers : int
        The number of walkers
    """
    assert trial.psi is not None
    trialdetach = trial.psi.detach().clone()
    walker_states = torch.stack([trialdetach] * nwalkers).to(torch.complex128)
    walker_weights = torch.tensor([1.0] * nwalkers, dtype=torch.float64)
    return Walkers(nwalkers, walker_states, walker_weights)


def reorthogonalize(walkers):
    """
    Reorthogonalize the walkers
    """
    orthowalkers, _ = torch.linalg.qr(walkers.walker_states)
    return Walkers(walkers.nwalkers, orthowalkers, walkers.walker_weights)


def sr(walkers):
    """
    stochastic reconfiguration method for population control
    """
    # rescale the weights
    walker_weights = walkers.walker_weights / torch.sum(walkers.walker_weights) * walkers.nwalkers
    cumulative_weights = torch.cumsum(walker_weights, dim=0)
    total_weight = cumulative_weights[-1]
    average_weight = total_weight / walkers.nwalkers
    walker_weights = torch.ones(walkers.nwalkers, dtype=torch.float64) * average_weight
    zeta = torch.rand(1).item()
    indices = torch.empty(walkers.nwalkers, dtype=torch.int64)
    for i in range(walkers.nwalkers):
        z = total_weight * (i + zeta) / walkers.nwalkers
        idx = torch.searchsorted(cumulative_weights, z)
        indices[i] = idx if idx < walkers.nwalkers else 0
    walker_states = walkers.walker_states[indices]
    return Walkers(walkers.nwalkers, walker_states, walker_weights)


class Walkers:
    def __init__(self, nwalkers, walker_states, walker_weights):
        """Walkers class
        
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

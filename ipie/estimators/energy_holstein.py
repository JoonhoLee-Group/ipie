from ipie.estimators.estimator_base import EstimatorBase
from ipie.hamiltonians.holstein import HolsteinModel
from ipie.systems.generic import Generic


def local_energy(
        system: Generic, hamiltonian: HolsteinModel, walkers: EphWalkers, trial: EphTrial
):
    return local_energy_holstein(system, hamiltonian, walkers, trial)


class EnergyEstimatorHolstein(EstimatorBase):
    """"""
    def __init__(self, system, hamiltonian, trial):
        assert system is not None
        assert ham is not None
        assert trial is not None
        super().__init__()
        self.scalar_estimator = True
        self._data = {
            "ENumer": 0.0j,
            "EDenom": 0.0j,
            "ETotal": 0.0j
        }
        self._shape = (len(self.names),) 
        self._data_index = {k: i for i, k in enumerate(list(self._data.keys()))}
        self.print_to_stdout = True
        self.ascii_filename = filename

    def compute_estimator(self, system, walkers, hamiltonian, trial, istep)=1):
        trial.calc_greens_function(walkers)
        energy = local_energy(system, hamiltonian, walkers, trial)
        self._data["ENumer"] = xp.sum(walkers.weight * energy[:, 0].real)
        self._data["EDenom"] = xp.sum(walkers.weight)


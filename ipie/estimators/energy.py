import numpy as np

from ipie.estimators.estimator_base import EstimatorBase
from ipie.estimators.local_energy_batch import local_energy_batch
from ipie.utils.io import get_input_value
from ipie.utils.misc import dotdict, is_cupy

class EnergyEstimator(EstimatorBase):

    def __init__(
            self,
            comm=None,
            qmc=None,
            system=None,
            ham=None,
            trial=None,
            verbose=False,
            options={}
            ):

        assert system is not None
        assert ham is not None
        assert trial is not None
        super().__init__()
        self._data = {
                "WeightFactor": 0.0j,
                "Weight": 0.0j,
                "ENumer": 0.0j,
                "EDenom": 0.0j,
                "ETotal": 0.0j,
                "E1Body": 0.0j,
                "E2Body": 0.0j,
                }
        self._shape = (len(self.names),)
        self._data_index = {k: i for i, k in enumerate(list(self._data.keys()))}
        self.print_to_stdout = True

    def compute_estimator(self, system, walker_batch, hamiltonian, trial_wavefunction):
        if is_cupy(
            walker_batch.weight
        ):  # if even one array is a cupy array we should assume the rest is done with cupy
            import cupy
            assert cupy.is_available()
            array = cupy.array
            zeros = cupy.zeros
            sum = cupy.sum
            abs = cupy.abs
        else:
            array = np.array
            zeros = np.zeros
            sum = np.sum
            abs = np.abs
        energy = local_energy_batch(system, hamiltonian, walker_batch, trial_wavefunction)
        self._data['WeightFactor'] = sum(walker_batch.unscaled_weight)
        self._data['Weight'] = sum(walker_batch.weight)
        self._data['ENumer'] = sum(walker_batch.weight * energy[:,0])
        self._data['EDenom'] = sum(walker_batch.weight)
        self._data['E1Body'] = sum(walker_batch.weight * energy[:,1])
        self._data['E2Body'] = sum(walker_batch.weight * energy[:,2])

    def get_index(self, name):
        return self._data_index.get(name)

    def post_reduce_hook(self, reduced_data):
        ix_proj = self.get_index("ETotal")
        ix_nume = self.get_index("ENumer")
        ix_deno = self.get_index("EDenom")
        reduced_data[ix_proj] = reduced_data[ix_nume] / reduced_data[ix_deno]

import numpy as np

from ipie.estimators.estimator_base import EstimatorBase
from ipie.estimators.greens_function_batch import greens_function
from ipie.estimators.local_energy_batch import local_energy_batch
from ipie.utils.io import get_input_value
from ipie.utils.misc import dotdict, is_cupy, to_numpy

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
        self._eshift = 0.0
        self.scalar_estimator = True
        self._data = {
                "ENumer": 0.0j,
                "EDenom": 0.0j,
                "ETotal": 0.0j,
                "E1Body": 0.0j,
                "E2Body": 0.0j,
                }
        self._shape = (len(self.names),)
        self._data_index = {k: i for i, k in enumerate(list(self._data.keys()))}
        self.print_to_stdout = True
        self.ascii_filename = get_input_value(options, 'filename', default=None)

    def compute_estimator(self, system, walker_batch, hamiltonian,
            trial_wavefunction, istep=1):
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
        greens_function(walker_batch, trial_wavefunction)
        energy = local_energy_batch(system, hamiltonian, walker_batch, trial_wavefunction)
        self._data['ENumer'] = to_numpy(sum(walker_batch.weight * energy[:,0].real))
        self._data['EDenom'] = to_numpy(sum(walker_batch.weight))
        self._data['E1Body'] = to_numpy(sum(walker_batch.weight * energy[:,1].real))
        self._data['E2Body'] = to_numpy(sum(walker_batch.weight * energy[:,2].real))
        return self.data

    def get_index(self, name):
        index = self._data_index.get(name, None)
        if index is None:
            raise RuntimeError(f"Unknown estimator {name}")
        return index

    def post_reduce_hook(self, reduced_data):
        ix_proj = self._data_index["ETotal"]
        ix_nume = self._data_index["ENumer"]
        ix_deno = self._data_index["EDenom"]
        reduced_data[ix_proj] = reduced_data[ix_nume] / reduced_data[ix_deno]
        ix_nume = self._data_index["E1Body"]
        reduced_data[ix_nume] = reduced_data[ix_nume] / reduced_data[ix_deno]
        ix_nume = self._data_index["E2Body"]
        reduced_data[ix_nume] = reduced_data[ix_nume] / reduced_data[ix_deno]

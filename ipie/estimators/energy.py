import numpy as np

from ipie.estimators.estimator_base import EstimatorBase
from ipie.estimators.greens_function_batch import greens_function
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
            nsteps=1,
            options={}
            ):

        assert system is not None
        assert ham is not None
        assert trial is not None
        super().__init__()
        self._eshift = 0.0
        self._data = {
                "WeightFactor": 0.0j,
                "Weight": 0.0j,
                "ENumer": 0.0j,
                "EDenom": 0.0j,
                "ETotal": 0.0j,
                "E1Body": 0.0j,
                "E2Body": 0.0j,
                "EHybrid": 0.0j,
                }
        self._shape = (len(self.names),)
        self._data_index = {k: i for i, k in enumerate(list(self._data.keys()))}
        self.nsteps = nsteps
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
        if istep % self.nsteps == 0:
            greens_function(walker_batch, trial_wavefunction)
            energy = local_energy_batch(system, hamiltonian, walker_batch, trial_wavefunction)
            self._data['ENumer'] = sum(walker_batch.weight * energy[:,0].real)
            self._data['EDenom'] = sum(walker_batch.weight)
            self._data['E1Body'] = sum(walker_batch.weight * energy[:,1].real)
            self._data['E2Body'] = sum(walker_batch.weight * energy[:,2].real)
        self._data['WeightFactor'] = sum(walker_batch.unscaled_weight)
        self._data['Weight'] = sum(walker_batch.weight)
        self._data['EHybrid'] = sum(walker_batch.weight * walker_batch.hybrid_energy)

        return self.data

    def get_shift(self):
        return self._eshift.real

    def get_index(self, name):
        index = self._data_index.get(name, None)
        if index is None:
            raise RuntimeError(f"Unknown estimator {name}")
        return index

    def post_reduce_hook(self, reduced_data, div_factor=None):
        if div_factor is None:
            div_factor = self.nsteps
        ix_proj = self._data_index["ETotal"]
        ix_nume = self._data_index["ENumer"]
        ix_deno = self._data_index["EDenom"]
        reduced_data[ix_proj] = reduced_data[ix_nume] / reduced_data[ix_deno]
        ix_nume = self._data_index["E1Body"]
        reduced_data[ix_nume] = reduced_data[ix_nume] / reduced_data[ix_deno]
        ix_nume = self._data_index["E2Body"]
        reduced_data[ix_nume] = reduced_data[ix_nume] / reduced_data[ix_deno]
        # Hack to preserve old feature of gathering shift.
        ix_nume = self._data_index["EHybrid"]
        reduced_data[ix_nume] = reduced_data[ix_nume] / reduced_data[self._data_index['Weight']]
        self._eshift = reduced_data[ix_nume]
        ix_nume = self._data_index["WeightFactor"]
        reduced_data[ix_nume] = reduced_data[ix_nume] / div_factor
        ix_nume = self._data_index["Weight"]
        reduced_data[ix_nume] = reduced_data[ix_nume] / div_factor

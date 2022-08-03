from abc import abstractmethod

import numpy as np

from ipie.utils.io import format_fixed_width_strings, format_fixed_width_floats

class EstimatorBase(object):

    def __init__(self):
        self._ascii_filename = None
        self._print_to_stdout = False
        # default to once per block
        self._write_frequency = 1
        self._data = {}

    @property
    def print_to_stdout(self) -> bool:
        return self._print_to_stdout

    @print_to_stdout.setter
    def print_to_stdout(self, val) -> None:
        self._print_to_stdout = val

    @property
    def ascii_filename(self) -> str:
        """Text file for output"""
        self._ascii_filename

    @property
    def write_frequency(self) -> str:
        """Group name for hdf5 file."""
        return self._write_frequency

    @property
    def shape(self) -> tuple:
        """Shape of estimator."""
        return self._shape

    @shape.setter
    def shape(self, shape) -> tuple:
        """Shape of estimator."""
        self._shape = shape

    @abstractmethod
    def compute_estimator(self, walker_batch, trial_wavefunction) -> np.ndarray:
        pass

    @property
    def names(self):
        return self._data.keys()

    @property
    def data(self):
        return np.array(list(self._data.values()))

    @property
    def header_to_text(self) -> str:
        return format_fixed_width_strings(self.names)

    def data_to_text(self, vals) -> str:
        vals
        assert len(vals) == len(self.names)
        return format_fixed_width_floats(vals.real)

    def post_reduce_hook(self) -> None:
        pass

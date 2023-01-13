
# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Fionn Malone <fmalone@google.com>
#

from abc import abstractmethod

import numpy as np

from ipie.utils.io import format_fixed_width_strings, format_fixed_width_floats
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host

class EstimatorBase(object):

    def __init__(self):
        self._ascii_filename = None
        self._print_to_stdout = False
        # default to once per block
        self._write_frequency = 1
        self._data = {}

    @property
    def scalar_estimator(self):
        return self._scalar_estimator

    @scalar_estimator.setter
    def scalar_estimator(self, val):
        assert isinstance(val, bool)
        self._scalar_estimator = val

    @property
    def print_to_stdout(self) -> bool:
        return self._print_to_stdout

    @print_to_stdout.setter
    def print_to_stdout(self, val) -> None:
        self._print_to_stdout = val

    @property
    def ascii_filename(self) -> str:
        """Text file for output"""
        return self._ascii_filename

    @ascii_filename.setter
    def ascii_filename(self, name):
        """Text file for output"""
        self._ascii_filename = name
        if self._ascii_filename is not None:
            with open(self._ascii_filename, 'w') as f:
                f.write(self.header_to_text + '\n')

    @property
    def write_frequency(self) -> str:
        """Group name for hdf5 file."""
        return self._write_frequency

    @property
    def shape(self) -> tuple:
        """Shape of estimator."""
        return self._shape

    @property
    def size(self) -> int:
        """Shape of estimator."""
        size = 0
        for k, v in self._data.items():
            if isinstance(v, np.ndarray):
                size += np.prod(v.shape)
            else:
                size += 1
        return size

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
        if self.scalar_estimator:
            vals = [to_host(i) for i in self._data.values()]
            return np.array(vals)
        else:
            return xp.concatenate(list(self._data.values()))

    @property
    def header_to_text(self) -> str:
        return format_fixed_width_strings(self.names)

    def data_to_text(self, vals) -> str:
        if self._ascii_filename is not None or self._print_to_stdout:
            assert len(vals) == len(self.names)
            return format_fixed_width_floats(vals.real)
        else:
            return ''

    def to_ascii_file(self, vals: str) -> None:
        if self.ascii_filename is not None:
            with open(self._ascii_filename, 'a') as f:
                f.write(vals + '\n')

    def __getitem__(self, name):
        data = self._data.get(name)
        if data is None:
            raise RuntimeError(f"Unknown estimator {name}")
        return self._data[name]

    def __setitem__(self, name, val):
        data = self._data.get(name)
        if data is None:
            raise RuntimeError(f"Unknown estimator {name}")
        if isinstance(self._data[name], np.ndarray):
            self._data[name][:] = val
        else:
            self._data[name] = val

    def zero(self):
        for k, v in self._data.items():
            if isinstance(v, np.ndarray):
                self._data[k] = np.zeros_like(v)
            else:
                self._data[k] = 0.0j

    def post_reduce_hook(self, data) -> None:
        pass

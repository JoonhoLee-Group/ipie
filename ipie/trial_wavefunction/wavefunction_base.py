from abc import abstractmethod

import numpy as np

from ipie.propagation.overlap import calc_overlap_single_det_batch

from ipie.utils.backend import cast_to_device
from ipie.utils.io import write_wavefunction


class TrialWavefunctionBase(object):
    """Base class for trialwavefunction types.

    Developer should decouple building wavefunction state and construction.

    Abstract methods build and half_rotate have to be defined for each method.
    """

    def __init__(self, wavefunction, num_elec, num_basis, init=None, verbose=False):
        self.nelec = num_elec
        self.nbasis = num_basis
        self.nalpha, self.nbeta = self.nelec
        self.verbose = verbose
        self._num_dets = 0
        self._max_num_dets = self._num_dets
        self.init = init
        self._half_rotated = False
        self.ortho_expansion = False
        self.optimized = True

    def cast_to_cupy(self) -> None:
        cast_to_device(self, self.verbose)

    @abstractmethod
    def build() -> None:
        pass

    @property
    def num_dets(self) -> int:
        return self._num_dets

    @num_dets.setter
    def num_dets(self, ndets: int) -> None:
        self._num_dets = ndets
        if self._num_dets > self._max_num_dets:
            raise RuntimeError(
                f"Requested more determinants than provided in "
                "wavefunction. {self._num_dets} vs {self._max_num_dets}"
            )

    @property
    def half_rotated(self) -> bool:
        return self._half_rotated

    @half_rotated.setter
    def half_rotated(self, is_half_rotated) -> None:
        self._half_rotated = is_half_rotated

    @abstractmethod
    def half_rotate() -> None:
        pass

    @abstractmethod
    def calc_overlap(self, walkers) -> np.ndarray:
        pass

    @abstractmethod
    def calc_greens_function(self, walkers) -> np.ndarray:
        pass

    @abstractmethod
    def calc_force_bias(self, walkers, hamiltonian, mpi_handler=None) -> np.ndarray:
        pass

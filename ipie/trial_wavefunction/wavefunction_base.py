from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np

from ipie.config import CommType, MPI
from ipie.utils.backend import cast_to_device

_wfn_type = Union[
    np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
]


class TrialWavefunctionBase(metaclass=ABCMeta):
    """Base class for trialwavefunction types.

    Developer should decouple building wavefunction state and construction.

    Abstract methods build and half_rotate have to be defined for each method.
    """

    def __init__(
        self,
        wavefunction: _wfn_type,
        num_elec: Tuple[int, int],
        num_basis: int,
        verbose: bool = False,
    ):
        self.nelec = num_elec
        self.nbasis = num_basis
        self.nalpha, self.nbeta = self.nelec
        self.verbose = verbose
        self._num_dets = 0
        self._max_num_dets = self._num_dets
        self._half_rotated = False
        self.ortho_expansion = False
        self.optimized = True

        self.compute_trial_energy = False

        self.e1b = None
        self.e2b = None
        self.energy = None

    def cast_to_cupy(self, verbose=False):
        cast_to_device(self, verbose=verbose)

    @abstractmethod
    def build(self) -> None: ...

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
    def half_rotate(self, hamiltonian, comm: Optional[CommType] = MPI.COMM_WORLD) -> None: ...

    @abstractmethod
    def calc_overlap(self, walkers) -> np.ndarray: ...

    @abstractmethod
    def calc_greens_function(self, walkers) -> np.ndarray: ...

    @abstractmethod
    def calc_force_bias(self, hamiltonian, walkers, mpi_handler) -> np.ndarray:
        pass

    def chunk(self, handler):
        self.chunked = True  # Boolean to indicate that chunked cholesky is available

        if handler.scomm.rank == 0:  # Creating copy for every rank == 0
            self._rchola = self._rchola.copy()
            self._rcholb = self._rcholb.copy()

        self._rchola_chunk = handler.scatter_group(self._rchola)  # distribute over chol
        self._rcholb_chunk = handler.scatter_group(self._rcholb)  # distribute over chol

        tot_size = handler.allreduce_group(self._rchola_chunk.size)
        assert self._rchola.size == tot_size
        tot_size = handler.allreduce_group(self._rcholb_chunk.size)
        assert self._rcholb.size == tot_size

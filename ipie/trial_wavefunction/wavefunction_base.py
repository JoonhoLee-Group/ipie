from abc import abstractmethod

from ipie.utils.backend import cast_to_device
from ipie.utils.io import write_wavefunction


class TrialWavefunctionBase(object):
    """Base class for trialwavefunction types.

    Developer should decouple building wavefunction state and construction.

    Abstract methods build and half_rotate have to be defined for each method.
    """
    def __init__(self, wavefunction, num_elec, num_basis, verbose=False):
        self.num_elec = num_elec
        self.num_basis = num_basis
        self.verbose = verbose

    def cast_to_cupy(self) -> None:
        cast_to_device(self, self.verbose)

    @abstractmethod
    def build() -> None:
        pass

    # @abstractmethod
    # def half_rotate() -> None:
        # pass

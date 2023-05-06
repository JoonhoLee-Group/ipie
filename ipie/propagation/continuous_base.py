from abc import abstractmethod, ABC


class PropagatorTimer(object):
    def __init__(self):
        self.tfbias = 0.0
        self.tovlp = 0.0
        self.tupdate = 0.0
        self.tgf = 0.0
        self.tvhs = 0.0
        self.tgemm = 0.0


class ContinuousBase(ABC):
    """A base class for continuous HS transform AFQMC propagators."""

    def __init__(self, time_step, verbose=False):
        # Derived Attributes
        self.dt = time_step
        self.verbose = verbose
        self.timer = PropagatorTimer()

    @abstractmethod
    def build(self, hamiltonian, trial=None, walkers=None, mpi_handler=None, verbose=False):
        pass

    @abstractmethod
    def propagate_walkers(self, walkers, hamiltonian, trial, eshift):
        pass

    @abstractmethod
    def propagate_walkers_one_body(self, walkers):
        pass

    @abstractmethod
    def propagate_walkers_two_body(self, walkers, hamiltonian, trial):
        pass

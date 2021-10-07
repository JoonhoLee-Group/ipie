import numpy
import sys
from pyqumc.walkers.stack import FieldConfig

class WalkerBatch(object):
    """WalkerBatch base class.

    Parameters
    ----------
    system : object
        System object.
    trial : object
        Trial wavefunction object.
    options : dict
        Input options
    index : int
        Element of trial wavefunction to initalise walker to.
    nprop_tot : int
        Number of back propagation steps (including imaginary time correlation
                functions.)
    nbp : int
        Number of back propagation steps.
    """

    def __init__(self, system, hamiltonian, trial, nwalkers, walker_opts={}, index=0, nprop_tot=None, nbp=None):
        self.nwalkers = nwalkers
        self.nup = system.nup
        self.ndown = system.ndown
        self.total_weight = 0.0

        self.weight = numpy.array([walker_opts.get('weight', 1.0) for iw in range(self.nwalkers)])
        self.unscaled_weight = self.weight.copy()
        self.phase = numpy.array([1. + 0.j for iw in range(self.nwalkers)])
        self.alive = numpy.array([1 for iw in range(self.nwalkers)])
        self.phi = numpy.array([trial.init.copy() for iw in range(self.nwalkers)])

        self.ot = numpy.array([1.0 for iw in range(self.nwalkers)])
        self.ovlp = numpy.array([1.0 for iw in range(self.nwalkers)])
        # in case we use local energy approximation to the propagation
        self.eloc = numpy.array([0.0 for iw in range(self.nwalkers)])
        # walkers overlap at time tau before backpropagation occurs
        self.ot_bp = numpy.array([1.0 for iw in range(self.nwalkers)])
        # walkers weight at time tau before backpropagation occurs
        self.weight_bp = self.weight
        # Historic wavefunction for back propagation.
        self.phi_old = numpy.array([self.phi.copy() for iw in range(self.nwalkers)])
        self.hybrid_energy = [0.0 for iw in range(self.nwalkers)]
        # Historic wavefunction for ITCF.
        self.weights = [numpy.array([1.0]) for iw in range(self.nwalkers)] # This is going to be for MSD trial... (should be named a lot better than this)
        self.detR = [1.0 for iw in range(self.nwalkers)]
        self.detR_shift = [0.0 for iw in range(self.nwalkers)]
        self.log_detR = [0.0 for iw in range(self.nwalkers)]
        self.log_shift = [0.0 for iw in range(self.nwalkers)]
        self.log_detR_shift = [0.0 for iw in range(self.nwalkers)]
        # Number of propagators to store for back propagation / ITCF.
        num_propg = [walker_opts.get('num_propg', 1) for iw in range(self.nwalkers)]
        if nbp is not None:
            self.field_configs = [FieldConfig(hamiltonian.nfields,
                                             nprop_tot, nbp,
                                             numpy.complex128) for iw in range(self.nwalkers)]
        else:
            self.field_configs = None
        self.stack = None

    def get_buffer(self, iw):
        """Get iw-th walker buffer for MPI communication

        iw : int
            the walker index of interest
        Returns
        -------
        buff : dict
            Relevant walker information for population control.
        """
        s = 0
        buff = numpy.zeros(self.buff_size, dtype=numpy.complex128)
        for d in self.buff_names:
            data = self.__dict__[d]
            assert(data.size % self.nwalkers == 0) # Only walker-specific data is being communicated
            if isinstance(data[iw], (numpy.ndarray)):
                buff[s:s+data[iw].size] = data[iw].ravel()
                s += data[iw].size
            elif isinstance(data[iw], list): # when data is list
                for l in data[iw]:
                    if isinstance(l, (numpy.ndarray)):
                        buff[s:s+l.size] = l.ravel()
                        s += l.size
                    elif isinstance(l, (int, float, complex)):
                        buff[s:s+1] = l
                        s += 1
            else:
                print("This should never be the case!")
                sys.exit()
                buff[s:s+1] = data
                s += 1
        if self.field_configs is not None:
            stack_buff = self.field_configs.get_buffer()
            return numpy.concatenate((buff,stack_buff))
        elif self.stack is not None:
            stack_buff = self.stack.get_buffer()
            return numpy.concatenate((buff,stack_buff))
        else:
            return buff

    def set_buffer(self, iw, buff):
        """Set walker buffer following MPI communication

        Parameters
        -------
        buff : dict
            Relevant walker information for population control.
        """
        s = 0
        for d in self.buff_names:
            data = self.__dict__[d]
            assert(data.size % self.nwalkers == 0) # Only walker-specific data is being communicated
            if isinstance(data[iw], numpy.ndarray):
                self.__dict__[d][iw] = buff[s:s+data.size].reshape(data[iw].shape).copy()
                s += data[iw].size
            elif isinstance(data[iw], list):
                for ix, l in enumerate(data[iw]):
                    if isinstance(l, (numpy.ndarray)):
                        self.__dict__[d][iw][ix] = buff[s:s+l.size].reshape(l.shape).copy()
                        s += l.size
                    elif isinstance(l, (int, float, complex)):
                        self.__dict__[d][iw][ix] = buff[s]
                        s += 1
            else:
                print("This should never be the case!")
                sys.exit()
                if isinstance(self.__dict__[d], int):
                    self.__dict__[d] = int(buff[s].real)
                elif isinstance(self.__dict__[d], float):
                    self.__dict__[d] = buff[s].real
                else:
                    self.__dict__[d] = buff[s]
                s += 1
        if self.field_configs is not None:
            self.field_configs.set_buffer(buff[self.buff_size:])
        if self.stack is not None:
            self.stack.set_buffer(buff[self.buff_size:])

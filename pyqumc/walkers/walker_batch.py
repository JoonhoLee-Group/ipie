import numpy
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

    def __init__(self, system, hamiltonian, trial, qmc, walker_opts={}, index=0, nprop_tot=None, nbp=None):
        self.nwalkers = qmc.nwalkers
        self.nup = system.nup
        self.ndown = system.ndown
        self.total_weight = 0.0

        self.weight = [walker_opts.get('weight', 1.0) for iw in range(self.nwalkers)]
        self.unscaled_weight = [self.weight for iw in range(self.nwalkers)]
        self.phase = [1 + 0j for iw in range(self.nwalkers)]
        self.alive = [1 for iw in range(self.nwalkers)]
        self.phi = [trial.init.copy() for iw in range(self.nwalkers)]

        self.ot = [1.0 for iw in range(self.nwalkers)]
        self.ovlp = [1.0 for iw in range(self.nwalkers)]
        # self.E_L = local_energy(system, self.G, self.Gmod, trail._rchol)[0].real
        self.E_L = [0.0 for iw in range(self.nwalkers)]
        self.eloc = [0.0 for iw in range(self.nwalkers)]
        # walkers overlap at time tau before backpropagation occurs
        self.ot_bp = [1.0 for iw in range(self.nwalkers)]
        # walkers weight at time tau before backpropagation occurs
        self.weight_bp = self.weight
        # Historic wavefunction for back propagation.
        self.phi_old = [self.phi.copy() for iw in range(self.nwalkers)]
        self.hybrid_energy = [0.0 for iw in range(self.nwalkers)]
        # Historic wavefunction for ITCF.
        self.phi_right = [self.phi.copy() for iw in range(self.nwalkers)]
        self.weights = [numpy.array([1.0]) for iw in range(self.nwalkers)]
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

    def get_buffer(self):
        """Get walker buffer for MPI communication

        Returns
        -------
        buff : dict
            Relevant walker information for population control.
        """
        s = 0
        buff = numpy.zeros(self.buff_size, dtype=numpy.complex128)
        for d in self.buff_names:
            data = self.__dict__[d]
            if isinstance(data, (numpy.ndarray)):
                buff[s:s+data.size] = data.ravel()
                s += data.size
            elif isinstance(data, list):
                for l in data:
                    if isinstance(l, (numpy.ndarray)):
                        buff[s:s+l.size] = l.ravel()
                        s += l.size
                    elif isinstance(l, (int, float, complex)):
                        buff[s:s+1] = l
                        s += 1
            else:
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

    def set_buffer(self, buff):
        """Set walker buffer following MPI communication

        Parameters
        -------
        buff : dict
            Relevant walker information for population control.
        """
        s = 0
        for d in self.buff_names:
            data = self.__dict__[d]
            if isinstance(data, numpy.ndarray):
                self.__dict__[d] = buff[s:s+data.size].reshape(data.shape).copy()
                s += data.size
            elif isinstance(data, list):
                for ix, l in enumerate(data):
                    if isinstance(l, (numpy.ndarray)):
                        self.__dict__[d][ix] = buff[s:s+l.size].reshape(l.shape).copy()
                        s += l.size
                    elif isinstance(l, (int, float, complex)):
                        self.__dict__[d][ix] = buff[s]
                        s += 1
            else:
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

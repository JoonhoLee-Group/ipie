import numpy
import scipy
import sys
from pyqumc.walkers.stack import FieldConfig
from pyqumc.utils.misc import is_cupy

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
        self.phi = numpy.array([trial.init.copy() for iw in range(self.nwalkers)], dtype=numpy.complex128)

        self.ot = numpy.array([1.0 for iw in range(self.nwalkers)])
        self.ovlp = numpy.array([1.0 for iw in range(self.nwalkers)])
        # in case we use local energy approximation to the propagation
        self.eloc = numpy.array([0.0 for iw in range(self.nwalkers)])
        # walkers overlap at time tau before backpropagation occurs
        self.ot_bp = numpy.array([1.0 for iw in range(self.nwalkers)])
        # walkers weight at time tau before backpropagation occurs
        self.weight_bp = self.weight
        # Historic wavefunction for back propagation.
        self.phi_old = self.phi.copy()
        self.hybrid_energy = numpy.array([0.0 for iw in range(self.nwalkers)])
        # Historic wavefunction for ITCF.
        # self.weights = [numpy.array([1.0]) for iw in range(self.nwalkers)] # This is going to be for MSD trial... (should be named a lot better than this)
        self.weights = numpy.zeros((self.nwalkers, 1), dtype=numpy.complex128)
        self.weights.fill(1.0)
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
        # Grab objects that are walker specific
        # WARNING!! One has to add names to the list here if new objects are added
        # self.buff_names = ["weight", "unscaled_weight", "phase", "alive", "phi", 
        #                    "ot", "ovlp", "eloc", "ot_bp", "weight_bp", "phi_old",
        #                    "hybrid_energy", "weights", "inv_ovlpa", "inv_ovlpb", "Ga", "Gb", "Ghalfa", "Ghalfb"]
        self.buff_names = ["weight", "unscaled_weight", "phase", "phi", "hybrid_energy", "ot", "ovlp"]
        self.buff_size = round(self.set_buff_size_single_walker()/float(self.nwalkers))

    # This function casts relevant member variables into cupy arrays
    def cast_to_cupy (self, verbose=False):
        import cupy

        size = self.weight.size + self.unscaled_weight.size + self.phase.size
        size += self.phi.size
        size += self.hybrid_energy.size
        size += self.ot.size
        size += self.ovlp.size
        if verbose:
            expected_bytes = size * 16.
            print("# WalkerBatch: expected to allocate {} GB".format(expected_bytes/1024**3))

        self.weight = cupy.asarray(self.weight)
        self.unscaled_weight = cupy.asarray(self.unscaled_weight)
        self.phase = cupy.asarray(self.phase)
        self.phi = cupy.asarray(self.phi)
        self.hybrid_energy = cupy.asarray(self.hybrid_energy)
        self.ot = cupy.asarray(self.ot)
        self.ovlp = cupy.asarray(self.ovlp)
        free_bytes, total_bytes = cupy.cuda.Device().mem_info
        used_bytes = total_bytes - free_bytes
        if verbose:
            print("# WalkerBatch: using {} GB out of {} GB memory on GPU".format(used_bytes/1024**3,total_bytes/1024**3))

    def set_buff_size_single_walker(self):
        if is_cupy(self.weight):
            import cupy
            ndarray = cupy.ndarray
            array = cupy.asnumpy
            isrealobj = cupy.isrealobj
        else:
            ndarray = numpy.ndarray
            array = numpy.array
            isrealobj = numpy.isrealobj

        names = []
        size = 0
        for k, v in self.__dict__.items():
            # try:
            #     print(k, v.size)
            # except AttributeError:
            #     print("failed", k, v)
            if (not (k in self.buff_names)):
                continue
            if isinstance(v, (ndarray)):
                names.append(k)
                size += v.size
            elif isinstance(v, (int, float, complex)):
                names.append(k)
                size += 1
            elif isinstance(v, list):
                names.append(k)
                for l in v:
                    if isinstance(l, (ndarray)):
                        size += l.size
                    elif isinstance(l, (int, float, complex)):
                        size += 1
        return size

    def get_buffer(self, iw):
        """Get iw-th walker buffer for MPI communication

        iw : int
            the walker index of interest
        Returns
        -------
        buff : dict
            Relevant walker information for population control.
        """
        if is_cupy(self.weight):
            import cupy
            ndarray = cupy.ndarray
            array = cupy.asnumpy
            isrealobj = cupy.isrealobj
        else:
            ndarray = numpy.ndarray
            array = numpy.array
            isrealobj = numpy.isrealobj

        s = 0
        buff = numpy.zeros(self.buff_size, dtype=numpy.complex128)
        for d in self.buff_names:
            data = self.__dict__[d]
            assert(data.size % self.nwalkers == 0) # Only walker-specific data is being communicated
            if isinstance(data[iw], (ndarray)):
                buff[s:s+data[iw].size] = array(data[iw].ravel())
                s += data[iw].size
            elif isinstance(data[iw], list): # when data is list
                for l in data[iw]:
                    if isinstance(l, (ndarray)):
                        buff[s:s+l.size] = array(l.ravel())
                        s += l.size
                    elif isinstance(l, (int, float, complex, numpy.float64, numpy.complex128)):
                        buff[s:s+1] = l
                        s += 1
            else:
                buff[s:s+1] = array(data[iw])
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
        if is_cupy(self.weight):
            import cupy
            ndarray = cupy.ndarray
            array = cupy.asarray
            isrealobj = cupy.isrealobj
        else:
            ndarray = numpy.ndarray
            array = numpy.asarray
            isrealobj = numpy.isrealobj

        s = 0
        for d in self.buff_names:
            data = self.__dict__[d]
            assert(data.size % self.nwalkers == 0) # Only walker-specific data is being communicated
            if isinstance(data[iw], ndarray):
                self.__dict__[d][iw] = array(buff[s:s+data[iw].size].reshape(data[iw].shape).copy())
                s += data[iw].size
            elif isinstance(data[iw], list):
                for ix, l in enumerate(data[iw]):
                    if isinstance(l, (ndarray)):
                        self.__dict__[d][iw][ix] = array(buff[s:s+l.size].reshape(l.shape).copy())
                        s += l.size
                    elif isinstance(l, (int, float, complex)):
                        self.__dict__[d][iw][ix] = buff[s]
                        s += 1
            else:
                if isinstance(self.__dict__[d][iw], (int, numpy.int64)):
                    self.__dict__[d][iw] = int(buff[s].real)
                elif isinstance(self.__dict__[d][iw], (float, numpy.float64)):
                    self.__dict__[d][iw] = buff[s].real
                else:
                    self.__dict__[d][iw] = buff[s]
                s += 1
        if self.field_configs is not None:
            self.field_configs.set_buffer(buff[self.buff_size:])
        if self.stack is not None:
            self.stack.set_buffer(buff[self.buff_size:])


    def reortho(self):
        """reorthogonalise walkers.

        parameters
        ----------
        """
        if(is_cupy(self.phi)):
            import cupy
            assert(cupy.is_available())
            array = cupy.array
            diag = cupy.diag
            zeros = cupy.zeros
            sum = cupy.sum
            dot = cupy.dot
            log = cupy.log
            sign = numpy.sign
            abs = cupy.abs
            exp = cupy.exp
            qr = cupy.linalg.qr
            qr_mode = 'reduced'
        else:
            array = numpy.array
            diag = numpy.diag
            zeros = numpy.zeros
            sum = numpy.sum
            dot = numpy.dot
            log = numpy.log
            sign = numpy.sign
            abs = numpy.abs
            exp = numpy.exp
            qr = scipy.linalg.qr
            qr_mode = 'economic'

        complex128 = numpy.complex128


        nup = self.nup
        ndown = self.ndown

        detR = []
        for iw in range(self.nwalkers):
            (self.phi[iw][:,:nup], Rup) = qr(self.phi[iw][:,:nup],mode=qr_mode)
            Rdown = zeros(Rup.shape)
            if ndown > 0:
                (self.phi[iw][:,nup:], Rdn) = qr(self.phi[iw][:,nup:], mode=qr_mode)
            # TODO: FDM This isn't really necessary, the absolute value of the
            # weight is used for population control so this shouldn't matter.
            # I think this is a legacy thing.
            # Wanted detR factors to remain positive, dump the sign in orbitals.
            Rup_diag = diag(Rup)
            signs_up = sign(Rup_diag)
            if ndown > 0:
                Rdn_diag = diag(Rdn)
                signs_dn = sign(Rdn_diag)
            self.phi[iw][:,:nup] = dot(self.phi[iw][:,:nup], diag(signs_up))
            if ndown > 0:
                self.phi[iw][:,nup:] = dot(self.phi[iw][:,nup:], diag(signs_dn))
            # include overlap factor
            # det(R) = \prod_ii R_ii
            # det(R) = exp(log(det(R))) = exp((sum_i log R_ii) - C)
            # C factor included to avoid over/underflow
            log_det = sum(log(abs(Rup_diag)))
            if ndown > 0:
                log_det += sum(log(abs(Rdn_diag)))
            detR += [exp(log_det-self.detR_shift[iw])]
            self.log_detR[iw] += log(detR[iw])
            self.detR[iw] = detR[iw]
            # print(self.ot[iw], detR[iw])
            self.ot[iw] = self.ot[iw] / detR[iw]
            self.ovlp[iw] = self.ot[iw]
        return detR

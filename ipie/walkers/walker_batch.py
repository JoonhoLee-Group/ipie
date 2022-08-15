import sys

import numpy
import scipy

from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host, synchronize, qr, qr_mode, cast_to_device


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

    def __init__(
        self,
        system,
        hamiltonian,
        trial,
        nwalkers,
        walker_opts={},
        index=0,
        nprop_tot=None,
        nbp=None,
        mpi_handler=None,
    ):
        self.nwalkers = nwalkers
        self.nup = system.nup
        self.ndown = system.ndown
        self.total_weight = 0.0
        self.mpi_handler = mpi_handler

        self.rhf = walker_opts.get("rhf", False)

        self.weight = numpy.array(
            [walker_opts.get("weight", 1.0) for iw in range(self.nwalkers)]
        )
        self.unscaled_weight = self.weight.copy()
        self.phase = numpy.array([1.0 + 0.0j for iw in range(self.nwalkers)])
        self.alive = numpy.array([1 for iw in range(self.nwalkers)])
        self.phia = numpy.array(
            [trial.init[:, : self.nup].copy() for iw in range(self.nwalkers)],
            dtype=numpy.complex128,
        )
        if not self.rhf:
            self.phib = numpy.array(
                [trial.init[:, self.nup :].copy() for iw in range(self.nwalkers)],
                dtype=numpy.complex128,
            )
        else:
            self.phib = None

        self.ovlp = numpy.array([1.0 for iw in range(self.nwalkers)])
        self.sgn_ovlp = numpy.array([1.0 for iw in range(self.nwalkers)])
        self.log_ovlp = numpy.array([0.0 for iw in range(self.nwalkers)])

        # in case we use local energy approximation to the propagation
        self.eloc = numpy.array([0.0 for iw in range(self.nwalkers)])

        self.hybrid_energy = numpy.array([0.0 for iw in range(self.nwalkers)])
        self.weights = numpy.zeros((self.nwalkers, 1), dtype=numpy.complex128)
        self.weights.fill(1.0)
        self.detR = [1.0 for iw in range(self.nwalkers)]
        self.detR_shift = [0.0 for iw in range(self.nwalkers)]
        self.log_detR = [0.0 for iw in range(self.nwalkers)]
        self.log_shift = numpy.array([0.0 for iw in range(self.nwalkers)])
        self.log_detR_shift = [0.0 for iw in range(self.nwalkers)]
        # Number of propagators to store for back propagation / ITCF.
        self.field_configs = None
        self.stack = None
        num_propg = [walker_opts.get("num_propg", 1) for iw in range(self.nwalkers)]
        # Grab objects that are walker specific
        # WARNING!! One has to add names to the list here if new objects are added
        # self.buff_names = ["weight", "unscaled_weight", "phase", "alive", "phi",
        #                    "ot", "ovlp", "eloc", "ot_bp", "weight_bp", "phi_old",
        #                    "hybrid_energy", "weights", "inv_ovlpa", "inv_ovlpb", "Ga", "Gb", "Ghalfa", "Ghalfb"]
        self.buff_names = [
            "weight",
            "unscaled_weight",
            "phase",
            "phia",
            "phib",
            "hybrid_energy",
            "ovlp",
            "sgn_ovlp",
            "log_ovlp",
        ]
        self.buff_size = round(
            self.set_buff_size_single_walker() / float(self.nwalkers)
        )

    # This function casts relevant member variables into cupy arrays
    def cast_to_cupy(self, verbose=False):
        cast_to_device(self, verbose)

    def set_buff_size_single_walker(self):
        names = []
        size = 0
        for k, v in self.__dict__.items():
            # try:
            #     print(k, v.size)
            # except AttributeError:
            #     print("failed", k, v)
            if not (k in self.buff_names):
                continue
            if isinstance(v, (xp.ndarray, numpy.ndarray)):
                names.append(k)
                size += v.size
            elif isinstance(v, (int, float, complex)):
                names.append(k)
                size += 1
            elif isinstance(v, list):
                names.append(k)
                for l in v:
                    if isinstance(l, (xp.ndarray)):
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
        s = 0
        buff = numpy.zeros(self.buff_size, dtype=numpy.complex128)
        for d in self.buff_names:
            data = self.__dict__[d]
            if data is None:
                continue
            assert (
                data.size % self.nwalkers == 0
            )  # Only walker-specific data is being communicated
            if isinstance(data[iw], (xp.ndarray)):
                buff[s : s + data[iw].size] = to_host(data[iw].ravel())
                s += data[iw].size
            elif isinstance(data[iw], list):  # when data is list
                for l in data[iw]:
                    if isinstance(l, (xp.ndarray)):
                        buff[s : s + l.size] = to_host(l.ravel())
                        s += l.size
                    elif isinstance(
                        l, (int, float, complex, numpy.float64, numpy.complex128)
                    ):
                        buff[s : s + 1] = l
                        s += 1
            else:
                buff[s : s + 1] = to_host(data[iw])
                s += 1
        if self.field_configs is not None:
            stack_buff = self.field_configs.get_buffer()
            return numpy.concatenate((buff, stack_buff))
        elif self.stack is not None:
            stack_buff = self.stack.get_buffer()
            return numpy.concatenate((buff, stack_buff))
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
            if data is None:
                continue
            assert (
                data.size % self.nwalkers == 0
            )  # Only walker-specific data is being communicated
            if isinstance(data[iw], xp.ndarray):
                self.__dict__[d][iw] = to_host(
                    buff[s : s + data[iw].size].reshape(data[iw].shape).copy()
                )
                s += data[iw].size
            elif isinstance(data[iw], list):
                for ix, l in enumerate(data[iw]):
                    if isinstance(l, (xp.ndarray)):
                        self.__dict__[d][iw][ix] = to_host(
                            buff[s : s + l.size].reshape(l.shape).copy()
                        )
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
            self.field_configs.set_buffer(buff[self.buff_size :])
        if self.stack is not None:
            self.stack.set_buffer(buff[self.buff_size :])

    def reortho(self):
        """reorthogonalise walkers.

        parameters
        ----------
        """
        complex128 = numpy.complex128
        nup = self.nup
        ndown = self.ndown
        detR = []
        for iw in range(self.nwalkers):
            (self.phia[iw], Rup) = qr(self.phia[iw], mode=qr_mode)
            Rdown = xp.zeros(Rup.shape)
            # TODO: FDM This isn't really necessary, the absolute value of the
            # weight is used for population control so this shouldn't matter.
            # I think this is a legacy thing.
            # Wanted detR factors to remain positive, dump the sign in orbitals.
            Rup_diag = xp.diag(Rup)
            signs_up = xp.sign(Rup_diag)
            self.phia[iw] = xp.dot(self.phia[iw], xp.diag(signs_up))

            # include overlap factor
            # det(R) = \prod_ii R_ii
            # det(R) = exp(log(det(R))) = exp((sum_i log R_ii) - C)
            # C factor included to avoid over/underflow
            log_det = xp.sum(xp.log(xp.abs(Rup_diag)))

            if ndown > 0 and not self.rhf:
                (self.phib[iw], Rdn) = qr(self.phib[iw], mode=qr_mode)
                Rdn_diag = xp.diag(Rdn)
                signs_dn = xp.sign(Rdn_diag)
                self.phib[iw] = xp.dot(self.phib[iw], xp.diag(signs_dn))
                log_det += sum(xp.log(abs(Rdn_diag)))
            elif ndown > 0 and self.rhf:
                log_det *= 2.0

            detR += [xp.exp(log_det - self.detR_shift[iw])]
            self.log_detR[iw] += xp.log(detR[iw])
            self.detR[iw] = detR[iw]
            self.ovlp[iw] = self.ovlp[iw] / detR[iw]

        synchronize()
        return detR


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
# Authors: Fionn Malone <fmalone@google.com>
#          Joonho Lee
#

import sys

import time

import numpy
import scipy

from ipie.config import config
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize, qr, qr_mode, cast_to_device, to_host
from ipie.utils.misc import is_cupy



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
        self.detR_shift = numpy.array([0.0 for iw in range(self.nwalkers)])
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
        import cupy

        size = (
            self.weight.size
            + self.unscaled_weight.size
            + self.phase.size
            + self.log_shift.size
        )
        size += self.phia.size
        if self.ndown > 0 and not self.rhf:
            size += self.phib.size
        size += self.hybrid_energy.size
        size += self.ovlp.size
        size += self.sgn_ovlp.size
        size += self.log_ovlp.size
        size += self.detR_shift.size
        if verbose:
            expected_bytes = size * 16.0
            print(
                "# WalkerBatch: expected to allocate {:4.3f} GB".format(
                    expected_bytes / 1024**3
                )
            )

        self.detR_shift = cupy.asarray(self.detR_shift)
        self.weight = cupy.asarray(self.weight)
        self.unscaled_weight = cupy.asarray(self.unscaled_weight)
        self.phase = cupy.asarray(self.phase)
        self.log_shift = cupy.asarray(self.log_shift)
        self.phia = cupy.asarray(self.phia)
        if self.ndown > 0 and not self.rhf:
            self.phib = cupy.asarray(self.phib)
        self.hybrid_energy = cupy.asarray(self.hybrid_energy)
        self.ovlp = cupy.asarray(self.ovlp)
        self.sgn_ovlp = cupy.asarray(self.sgn_ovlp)
        self.log_ovlp = cupy.asarray(self.log_ovlp)
        self.log_shift = cupy.asarray(self.log_shift)
        free_bytes, total_bytes = cupy.cuda.Device().mem_info
        used_bytes = total_bytes - free_bytes
        if verbose:
            print(
                "# WalkerBatch: using {:4.3f} GB out of {:4.3f} GB memory on GPU".format(
                    used_bytes / 1024**3, total_bytes / 1024**3
                )
            )

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

    # def get_buffer(self, iw):
        # """Get iw-th walker buffer for MPI communication

        # iw : int
            # the walker index of interest
        # Returns
        # -------
        # buff : dict
            # Relevant walker information for population control.
        # """
        # s = 0
        # buff = numpy.zeros(self.buff_size, dtype=numpy.complex128)
        # for d in self.buff_names:
            # data = self.__dict__[d]
            # if data is None:
                # continue
            # assert (
                # data.size % self.nwalkers == 0
            # )  # Only walker-specific data is being communicated
            # if isinstance(data[iw], (xp.ndarray)):
                # buff[s : s + data[iw].size] = to_host(data[iw].ravel())
                # s += data[iw].size
            # elif isinstance(data[iw], list):  # when data is list
                # for l in data[iw]:
                    # if isinstance(l, (xp.ndarray)):
                        # buff[s : s + l.size] = to_host(l.ravel())
                        # s += l.size
                    # elif isinstance(
                        # l, (int, float, complex, numpy.float64, numpy.complex128)
                    # ):
                        # buff[s : s + 1] = l
                        # s += 1
            # else:
                # buff[s : s + 1] = to_host(data[iw])
                # s += 1
        # if self.field_configs is not None:
            # stack_buff = self.field_configs.get_buffer()
            # return numpy.concatenate((buff, stack_buff))
        # elif self.stack is not None:
            # stack_buff = self.stack.get_buffer()
            # return numpy.concatenate((buff, stack_buff))
        # else:
            # return buff

    # def set_buffer(self, iw, buff):
        # """Set walker buffer following MPI communication

        # Parameters
        # -------
        # buff : dict
            # Relevant walker information for population control.
        # """
        # s = 0
        # for d in self.buff_names:
            # data = self.__dict__[d]
            # if data is None:
                # continue
            # assert (
                # data.size % self.nwalkers == 0
            # )  # Only walker-specific data is being communicated
            # if isinstance(data[iw], xp.ndarray):
                # # weights are purely real
                # if not xp.iscomplexobj(self.__dict__[d][iw]):
                    # if data[iw].size == 1:
                        # self.__dict__[d][iw] = (
                                # buff[s : s + data[iw].size].reshape(data[iw].shape).copy()
                        # ).real
                    # else:
                        # self.__dict__[d][iw] = xp.asarray(
                            # buff[s : s + data[iw].size].reshape(data[iw].shape).copy()
                        # ).real
                # else:
                    # if data[iw].size == 1:
                        # self.__dict__[d][iw] = (
                            # buff[s : s + data[iw].size].reshape(data[iw].shape).copy()
                        # )
                    # else:
                        # self.__dict__[d][iw] = xp.asarray(
                            # buff[s : s + data[iw].size].reshape(data[iw].shape).copy()
                        # )
                # s += data[iw].size
            # elif isinstance(data[iw], list):
                # for ix, l in enumerate(data[iw]):
                    # if isinstance(l, (xp.ndarray)):
                        # self.__dict__[d][iw][ix] = xp.asarray(
                            # buff[s : s + l.size].reshape(l.shape).copy()
                        # )
                        # s += l.size
                    # elif isinstance(l, (int, float, complex)):
                        # self.__dict__[d][iw][ix] = buff[s]
                        # s += 1
            # else:
                # if isinstance(self.__dict__[d][iw], (int, numpy.int64)):
                    # self.__dict__[d][iw] = int(buff[s].real)
                # elif isinstance(self.__dict__[d][iw], (float, numpy.float64)):
                    # self.__dict__[d][iw] = buff[s].real
                # else:
                    # self.__dict__[d][iw] = buff[s]
                # s += 1
        # if self.field_configs is not None:
            # self.field_configs.set_buffer(buff[self.buff_size :])
        # if self.stack is not None:
            # self.stack.set_buffer(buff[self.buff_size :])

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
            if data is None:
                continue
            assert (
                data.size % self.nwalkers == 0
            )  # Only walker-specific data is being communicated
            if isinstance(data[iw], (ndarray)):
                buff[s : s + data[iw].size] = array(data[iw].ravel())
                s += data[iw].size
            elif isinstance(data[iw], list):  # when data is list
                for l in data[iw]:
                    if isinstance(l, (ndarray)):
                        buff[s : s + l.size] = array(l.ravel())
                        s += l.size
                    elif isinstance(
                        l, (int, float, complex, numpy.float64, numpy.complex128)
                    ):
                        buff[s : s + 1] = l
                        s += 1
            else:
                buff[s : s + 1] = array(data[iw])
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
            if data is None:
                continue
            assert (
                data.size % self.nwalkers == 0
            )  # Only walker-specific data is being communicated
            if isinstance(data[iw], ndarray):
                self.__dict__[d][iw] = array(
                    buff[s : s + data[iw].size].reshape(data[iw].shape).copy()
                )
                s += data[iw].size
            elif isinstance(data[iw], list):
                for ix, l in enumerate(data[iw]):
                    if isinstance(l, (ndarray)):
                        self.__dict__[d][iw][ix] = array(
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
        if config.get_option('use_gpu'):
            return self.reortho_batched()
        complex128 = numpy.complex128
        nup = self.nup
        ndown = self.ndown
        detR = []
        for iw in range(self.nwalkers):
            (self.phia[iw], Rup) = qr(self.phia[iw], mode=qr_mode)
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

    def reortho_batched(self):
        """reorthogonalise walkers.

        parameters
        ----------
        """
        assert config.get_option('use_gpu')
        (self.phia, Rup) = qr(self.phia, mode=qr_mode)
        Rup_diag = xp.einsum("wii->wi",Rup)
        log_det = xp.einsum("wi->w", xp.log(abs(Rup_diag)))
        if self.ndown > 0 and not self.rhf:
            (self.phib, Rdn) = qr(self.phib, mode=qr_mode)
            Rdn_diag = xp.einsum("wii->wi",Rdn)
            log_det += xp.einsum("wi->w", xp.log(abs(Rdn_diag)))
        elif self.ndown > 0 and self.rhf:
            log_det *= 2.0

        self.detR = xp.exp(log_det - self.detR_shift)
        self.ovlp = self.ovlp / self.detR

        synchronize()

        return self.detR

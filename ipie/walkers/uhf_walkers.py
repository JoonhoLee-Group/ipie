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

import numpy

from ipie.config import config
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import cast_to_device
from ipie.utils.backend import synchronize, qr, qr_mode, cast_to_device, to_host
from ipie.walkers.base_walkers import BaseWalkers


class UHFWalkers(BaseWalkers):
    """UHF style walker.

    Parameters
    ----------
    system : object
        System object.
    nwalkers : int
        The number of walkers in this batch
    """

    def __init__(
        self,
        initial_walker,
        nup, ndown,
        num_walkers_local,
        num_walkers_global,
        num_steps,
        mpi_handler=None,
        pop_control_method="pair_branch",
        min_weight=0.1,
        max_weight=4,
        reconfiguration_frequency=50,
        verbose=False
    ):
        assert len(initial_walker.shape) == 2
        self.nbasis = initial_walker.shape[0]

        super().__init__(
            nup, ndown,
            num_walkers_local,
            num_walkers_global,
            num_steps,
            mpi_handler=mpi_handler,
            pop_control_method=pop_control_method,
            min_weight=min_weight,
            max_weight=max_weight,
            reconfiguration_frequency=reconfiguration_frequency,
            verbose=verbose
        )

        # should completely deprecate these
        self.field_configs = None

        self.phia = numpy.array(
            [initial_walker[:, : self.nup].copy() for iw in range(self.nwalkers)],
            dtype=numpy.complex128,
        )
        self.phib = numpy.array(
            [initial_walker[:, self.nup :].copy() for iw in range(self.nwalkers)],
            dtype=numpy.complex128,
        )

        self.Ga = numpy.zeros(
            shape=(self.nwalkers, self.nbasis, self.nbasis),
            dtype=numpy.complex128,
        )
        self.Gb = numpy.zeros(
            shape=(self.nwalkers, self.nbasis, self.nbasis),
            dtype=numpy.complex128,
        )

        self.Ghalfa = numpy.zeros(
            shape=(self.nwalkers, nup, self.nbasis), dtype=numpy.complex128
        )
        self.Ghalfb = numpy.zeros(
            shape=(self.nwalkers, ndown, self.nbasis),
            dtype=numpy.complex128,
        )

        self.buff_names += ["phia, phib"]

        self.buff_size = round(
            self.set_buff_size_single_walker() / float(self.nwalkers)
        )
        self.walker_buffer = numpy.zeros(self.buff_size, dtype=numpy.complex128)

        self.rhf = False # interfacing with old codes...

    # This function casts relevant member variables into cupy arrays
    def cast_to_cupy(self, verbose=False):
        cast_to_device(self, verbose)

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

            if ndown > 0:
                (self.phib[iw], Rdn) = qr(self.phib[iw], mode=qr_mode)
                Rdn_diag = xp.diag(Rdn)
                signs_dn = xp.sign(Rdn_diag)
                self.phib[iw] = xp.dot(self.phib[iw], xp.diag(signs_dn))
                log_det += sum(xp.log(abs(Rdn_diag)))

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

        if self.ndown > 0:
            (self.phib, Rdn) = qr(self.phib, mode=qr_mode)
            Rdn_diag = xp.einsum("wii->wi",Rdn)
            log_det += xp.einsum("wi->w", xp.log(abs(Rdn_diag)))

        self.detR = xp.exp(log_det - self.detR_shift)
        self.ovlp = self.ovlp / self.detR

        synchronize()

        return self.detR

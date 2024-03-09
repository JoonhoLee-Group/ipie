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
import plum

from ipie.config import config
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import cast_to_device, qr, qr_mode, synchronize
from ipie.walkers.base_walkers import BaseWalkers
from ipie.walkers.uhf_walkers import UHFWalkers


class GHFWalkers(BaseWalkers):
    """GHF style walker.

    Parameters
    ----------
    nwalkers : int
        The number of walkers in this batch
    """

    @plum.dispatch
    def __init__(self, walkers: UHFWalkers, verbose: bool = False):
        self.nup = walkers.nup
        self.ndown = walkers.ndown
        self.nbasis = walkers.nbasis
        self.mpi_handler = walkers.mpi_handler
        super().__init__(walkers.nwalkers, verbose=verbose)

        self.phi = numpy.zeros(
            (self.nwalkers, self.nbasis * 2, self.nup + self.ndown), dtype=walkers.phia.dtype
        )

        # for iw in range(self.nwalkers):
        self.phi[:, : self.nbasis, : self.nup] = walkers.phia
        self.phi[:, self.nbasis :, self.nup :] = walkers.phib

        self.G = numpy.zeros(
            (self.nwalkers, 2 * self.nbasis, 2 * self.nbasis), dtype=walkers.Ga.dtype
        )
        self.G[:, : self.nbasis, : self.nbasis] = walkers.Ga
        self.G[:, self.nbasis :, self.nbasis :] = walkers.Gb
        self.Ghalf = None

    @plum.dispatch
    def __init__(
        self,
        initial_walker: numpy.ndarray,
        nup: int,
        ndown: int,
        nbasis: int,
        nwalkers: int,
        mpi_handler,
        verbose: bool = False,
    ):
        assert len(initial_walker.shape) == 2
        self.nup = nup
        self.ndown = ndown
        self.nbasis = nbasis
        self.mpi_handler = mpi_handler

        super().__init__(nwalkers, verbose=verbose)

        # should completely deprecate these
        self.field_configs = None

        self.phi = numpy.array(
            [initial_walker.copy() for iw in range(self.nwalkers)],
            dtype=numpy.complex128,
        )

        # will be built only on request
        self.G = numpy.zeros(
            shape=(self.nwalkers, self.nbasis, self.nbasis),
            dtype=numpy.complex128,
        )

        self.buff_names += ["phi"]

        self.buff_size = round(self.set_buff_size_single_walker() / float(self.nwalkers))
        self.walker_buffer = numpy.zeros(self.buff_size, dtype=numpy.complex128)

    def build(self, trial):
        self.ovlp = trial.calc_greens_function(self)

    # This function casts relevant member variables into cupy arrays
    def cast_to_cupy(self, verbose=False):
        cast_to_device(self, verbose)

    def reortho(self):
        """reorthogonalise walkers.

        parameters
        ----------
        """
        if config.get_option("use_gpu"):
            return self.reortho_batched()
        detR = []
        for iw in range(self.nwalkers):
            (self.phi[iw], R) = qr(self.phi[iw], mode=qr_mode)
            # TODO: FDM This isn't really necessary, the absolute value of the
            # weight is used for population control so this shouldn't matter.
            # I think this is a legacy thing.
            # Wanted detR factors to remain positive, dump the sign in orbitals.
            R_diag = xp.diag(R)
            signs = xp.sign(R_diag)
            self.phi[iw] = xp.dot(self.phi[iw], xp.diag(signs))

            # include overlap factor
            # det(R) = \prod_ii R_ii
            # det(R) = exp(log(det(R))) = exp((sum_i log R_ii) - C)
            # C factor included to avoid over/underflow
            log_det = xp.sum(xp.log(xp.abs(R_diag)))

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
        assert config.get_option("use_gpu")
        (self.phi, Rup) = qr(self.phi, mode=qr_mode)
        Rup_diag = xp.einsum("wii->wi", Rup)
        log_det = xp.einsum("wi->w", xp.log(abs(Rup_diag)))
        self.detR = xp.exp(log_det - self.detR_shift)
        self.ovlp = self.ovlp / self.detR

        synchronize()

        return self.detR
